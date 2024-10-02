import time

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import wandb

from envs.pong import create_atari_env
from intervention_rl.models.agent_model import ActorCritic
from intervention_rl.models.blocker_model import CNNClassifier
from intervention_rl.utils import pong_blocker as blocker

def one_hot_encode_action(action, num_actions):
    """Convert an action into a one-hot encoded vector."""
    one_hot = torch.zeros(num_actions)
    one_hot[action] = 1.0
    return one_hot

def test(rank, cfg, shared_model, shared_blocker_model, counter, lock, use_wandb=False, experiment_name='experiment'):
    if use_wandb:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(project="modified_hirl", name=experiment_name, config=config_dict)

    torch.manual_seed(cfg.seed + rank)

    env = create_atari_env(cfg.env.name)
    if hasattr(env, 'get_wrapper_attr'):
        env.get_wrapper_attr('seed')(cfg.seed + rank)
    else:
        env.unwrapped.seed(cfg.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    blocker_model = CNNClassifier(env.action_space.n)

    model.eval()
    blocker_model.eval()

    state = env.reset()[0]
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()
    episode_length = 0
    episode_counter = 0
    approx_counter = 0
    episode_catastrophes = 0
    cumulative_catastrophes = 0
    use_blocker_model = False

    while True:
        with lock:
            approx_counter = counter.value
            
        if approx_counter >= cfg.algorithm.max_steps:
            break

        if approx_counter >= cfg.algorithm.max_steps_for_blocker_training:
            use_blocker_model = True
        else:
            use_blocker_model = False

        episode_length += 1

        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            blocker_model.load_state_dict(shared_blocker_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()

        full_obs = env.get_wrapper_attr('get_full_obs')()
        blocker_model_input = env.get_wrapper_attr('get_cropped_obs')()

        if use_blocker_model:
            one_hot_action = one_hot_encode_action(action, env.action_space.n)

            blocker_model_input_obs = torch.tensor(blocker_model_input, dtype=torch.float32).permute(0, 2, 1).unsqueeze(0)
            blocker_model_input_action = one_hot_action.unsqueeze(0)
            blocker_model_output = blocker_model(blocker_model_input_obs, blocker_model_input_action)
            block_decision = torch.argmax(blocker_model_output, dim=1).item()

            action = 2 if block_decision else action[0, 0]
        else:
            block_decision = blocker.should_block(full_obs, action[0, 0])
            action = 2 if block_decision else action[0, 0]

        if blocker.is_catastrophe(full_obs):
            episode_catastrophes += 1
            cumulative_catastrophes += 1

        state, reward, done, _, _ = env.step(action)
        done = done or episode_length >= cfg.algorithm.max_episode_length
        reward_sum += reward

        if done:
            with lock:
                approx_counter = counter.value

            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}, catastrophes {}, cumulative catastrophes {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                approx_counter, approx_counter / (time.time() - start_time),
                reward_sum, episode_length, episode_catastrophes, cumulative_catastrophes))

            if use_wandb:
                wandb.log({'counter': approx_counter, 'episode_reward': reward_sum, 'episode_length': episode_length, 'episode_catastrophes': episode_catastrophes, 'cumulative_catastrophes': cumulative_catastrophes})

            reward_sum = 0
            approx_counter = 0
            state = env.reset()[0]
            episode_length = 0
            episode_catastrophes = 0
            episode_counter += 1

            time.sleep(60)

        state = torch.from_numpy(state)
