import os
import time
from collections import deque
import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from blocker_model import CNNClassifier
import wandb
import blocker
import numpy as np
# import imageio
# from PIL import Image, ImageDraw

def one_hot_encode_action(action, num_actions):
    """Convert an action into a one-hot encoded vector."""
    one_hot = torch.zeros(num_actions)
    one_hot[action] = 1.0
    return one_hot

def test(rank, args, shared_model, shared_cnn_model, counter, lock, wandb_bool=False, experiment_name='experiment'):
    if wandb_bool:
        wandb.init(project="modified_hirl", name=experiment_name, config=args)

    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    if hasattr(env, 'get_wrapper_attr'):
        env.get_wrapper_attr('seed')(args.seed + rank)
    else:
        env.unwrapped.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    cnn_model = CNNClassifier(env.action_space.n)

    model.eval()
    cnn_model.eval()

    state = env.reset()[0]
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    # # Create frames directory if it doesn't exist
    # frames_dir = 'frames'
    # if not os.path.exists(frames_dir):
    #     os.makedirs(frames_dir)

    episode_length = 0
    episode_catastrophes = 0
    cumulative_catastrophes = 0
    episode_counter = 0
    use_cnn = False
    # frames = []

    while True:
        with lock:
            if counter.value >= args.max_steps:
                break

            if counter.value >= args.max_steps_for_cnn:
                use_cnn = True
            else:
                use_cnn = False

        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cnn_model.load_state_dict(shared_cnn_model.state_dict())
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
        cnn_input = env.get_wrapper_attr('get_cropped_obs')()

        # # Create a PIL image for drawing
        # pil_frame = Image.fromarray(full_obs)
        # draw = ImageDraw.Draw(pil_frame)

        # # Draw horizontal white lines
        # draw.line((0, 178, 160, 178), fill="white", width=1)
        # draw.line((0, 162, 160, 162), fill="white", width=1)

        if use_cnn:
            one_hot_action = one_hot_encode_action(action, env.action_space.n)

            cnn_input_obs = torch.tensor(cnn_input, dtype=torch.float32).permute(0, 2, 1).unsqueeze(0)
            cnn_input_action = one_hot_action.unsqueeze(0)  # Reshape for batch processing
            cnn_output = cnn_model(cnn_input_obs, cnn_input_action)
            block_decision = torch.argmax(cnn_output, dim=1).item()

            action = 2 if block_decision else action[0, 0]

            # if block_decision:
            #     draw.text((10, 30), "Should Block Detected", fill="yellow")

        else:
            block_decision = blocker.should_block(full_obs, action[0, 0])
            action = 2 if block_decision else action[0, 0]

            # if block_decision:
            #     draw.text((10, 30), "Should Block Detected", fill="yellow")

        if blocker.is_catastrophe(full_obs):
            episode_catastrophes += 1
            cumulative_catastrophes += 1
            # draw.text((10, 10), "Catastrophe Detected", fill="red")

        state, reward, done, _, _ = env.step(action)
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # # Draw red dot at the paddle bottom location
        # paddle_pos = blocker.paddle_bottom(full_obs)
        # if paddle_pos is not None:
        #     draw.ellipse((blocker.PADDLE_COLUMN - 2, paddle_pos - 2, blocker.PADDLE_COLUMN + 2, paddle_pos + 2), fill="red", outline="red")

        # # Convert back to numpy array
        # frames.append(np.array(pil_frame))

        if done:
            with lock:
                temp_counter = counter.value

            # # Save the frames as a video with macro_block_size=1
            # frame_filename = os.path.join(frames_dir, f'episode_{episode_counter}.mp4')
            # imageio.mimsave(frame_filename, frames, fps=1, macro_block_size=1)  # Save frames as a video

            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}, catastrophes {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                temp_counter, temp_counter / (time.time() - start_time),
                reward_sum, episode_length, episode_catastrophes))

            if wandb_bool:
                wandb.log({'counter': temp_counter, 'episode_reward': reward_sum, 'episode_length': episode_length, 'episode_catastrophes': episode_catastrophes, 'cumulative_catastrophes': cumulative_catastrophes})

            reward_sum = 0
            episode_length = 0
            state = env.reset()[0]
            episode_catastrophes = 0
            episode_counter += 1

            # # Reset frames for the next episode
            # frames = []

            time.sleep(60)

        state = torch.from_numpy(state)
