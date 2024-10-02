import torch
import torch.nn.functional as F
import torch.optim as optim

from envs.pong import create_atari_env
from intervention_rl.models.agent_model import ActorCritic
from intervention_rl.models.blocker_model import CNNClassifier
from intervention_rl.utils import pong_blocker as blocker

def ensure_shared_grads(model, shared_model):
    """Copy gradients from model to shared_model."""
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def one_hot_encode_action(action, num_actions):
    """Convert an action into a one-hot encoded vector."""
    one_hot = torch.zeros(num_actions)
    one_hot[action] = 1.0
    return one_hot

def train(rank, cfg, shared_model, shared_blocker_model, counter, lock, optimizer=None, blocker_optimizer=None, shared_observations=None, shared_labels=None):
    torch.manual_seed(cfg.seed + rank)

    env = create_atari_env(cfg.env.name)
    if hasattr(env, 'get_wrapper_attr'):
        env.get_wrapper_attr('seed')(cfg.seed + rank)
    else:
        env.unwrapped.seed(cfg.seed + rank)

    # Create the ActorCritic model and CNN classifier
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    blocker_model = CNNClassifier(env.action_space.n)
    
    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=cfg.algorithm.lr)
    if blocker_optimizer is None:
        blocker_optimizer = optim.Adam(shared_blocker_model.parameters(), lr=cfg.algorithm.blocker_model_lr)

    model.train()

    state = env.reset()[0]
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    approx_counter = 0
    use_blocker_model = False # Initially, use should_block()

    while True:
        with lock:
            approx_counter = counter.value

        if approx_counter >= cfg.algorithm.max_steps:
            break

        if approx_counter >= cfg.algorithm.max_steps_for_blocker_training and not use_blocker_model:
            use_blocker_model = True
        else:
            use_blocker_model = False

        model.load_state_dict(shared_model.state_dict())
        blocker_model.load_state_dict(shared_blocker_model.state_dict())
        
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(cfg.algorithm.num_steps):
            episode_length += 1
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach().item()            
            robot_action = action

            full_obs = env.get_wrapper_attr('get_full_obs')()
            blocker_model_input = env.get_wrapper_attr('get_cropped_obs')()

            one_hot_action = one_hot_encode_action(robot_action, env.action_space.n) 

            blocker_model_input_obs = torch.tensor(blocker_model_input, dtype=torch.float32).permute(0, 2, 1).unsqueeze(0)
            blocker_model_input_action = one_hot_action.unsqueeze(0)
            blocker_model_output = blocker_model(blocker_model_input_obs, blocker_model_input_action)
            blocker_model_block_decision = torch.argmax(blocker_model_output, dim=1).item()

            if use_blocker_model:
                block_decision = blocker_model_block_decision
                action = 2 if block_decision else robot_action
                # Learn on Expert Human Action
                log_prob = log_prob.gather(1, torch.tensor([[action]]))
            else:
                block_decision = blocker.should_block(full_obs, robot_action)
                action = 2 if block_decision else robot_action
                # Learn on Expert Human Action
                log_prob = log_prob.gather(1, torch.tensor([[action]]))

                # Store observation and label for CNN training
                with lock:
                    shared_observations.append((blocker_model_input, one_hot_action))
                    shared_labels.append(block_decision)

                # Calculate uncertainty (entropy) of CNN output
                blocker_model_prob = F.softmax(blocker_model_output, dim=-1)
                blocker_model_prob = torch.clamp(blocker_model_prob, min=1e-6)
                blocker_model_entropy = -(blocker_model_prob * blocker_model_prob.log()).sum().item()

                # Log the probabilities for disagreement checking
                if block_decision == 1:  # Human says block
                    disagreement = blocker_model_prob[0, 0]  # Probability of model saying "do not block"
                else:  # Human says do not block
                    disagreement = blocker_model_prob[0, 1]  # Probability of model saying "block"

            # Step environment
            state, reward, done, _, _ = env.step(action)
            done = done or episode_length >= cfg.algorithm.max_episode_length
            reward = max(min(reward, 1), -1)  # Clip rewards

            if use_blocker_model:
                modified_reward = reward
            else:
                modified_reward = reward + (cfg.algorithm.alpha * blocker_model_entropy) + (cfg.algorithm.beta if disagreement else 0)

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                state = env.reset()[0]

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(modified_reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = cfg.algorithm.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            delta_t = rewards[i] + cfg.algorithm.gamma * values[i + 1] - values[i]
            gae = gae * cfg.algorithm.gamma * cfg.algorithm.gae_lambda + delta_t

            policy_loss = policy_loss - log_probs[i] * gae.detach() - cfg.algorithm.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + cfg.algorithm.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.algorithm.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()