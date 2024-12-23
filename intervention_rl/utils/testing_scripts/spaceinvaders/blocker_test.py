import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
import cv2
import imageio
import random

class BlockerHeuristic:
    CATASTROPHE_THRESHOLD = 115
    BLOCKER_THRESHOLD = 110
    
    def __init__(self, clearance=None, block_clearance=None):
        self.CATASTROPHE_THRESHOLD = clearance if clearance is not None else self.CATASTROPHE_THRESHOLD
        self.BLOCKER_THRESHOLD = block_clearance if block_clearance is not None else self.BLOCKER_THRESHOLD

    def is_catastrophe(self, obs):
        return obs[28] >= self.CATASTROPHE_THRESHOLD
    
    def is_block_zone(self, obs):
        return obs[28] >= self.BLOCKER_THRESHOLD
    
    def should_block(self, obs, action):
        if obs is None:
            return False
        if self.is_catastrophe(obs):
            return True
        elif self.is_block_zone(obs) and action not in [3,5]:
            return True
        return False

def create_frame(frame):
    # Draw vertical lines on the image
    line_color = (255, 0, 0)  # Blue color in BGR format
    blocker_line = 110  # x-coordinate for blocker line
    catastrophe_line = 115  # x-coordinate for catastrophe line
    cv2.line(frame, (blocker_line, 0), (blocker_line, frame.shape[0]), line_color, 1)
    cv2.line(frame, (catastrophe_line, 0), (catastrophe_line, frame.shape[0]), line_color, 1)

    return frame

def main():
    # Create the environment
    env_name = 'ALE/SpaceInvaders-v5'
    seed = 123
    env = make_atari_env(env_name, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    blocker_heuristic = BlockerHeuristic()

    env.reset()
    total_steps = 1000
    frames = []

    for i in range(total_steps):
        action = 2

        ram_state = env.envs[0].unwrapped.ale.getRAM()
        if blocker_heuristic.should_block(ram_state, action):
            action = 3

        # Step in the environment with the chosen action
        obs, rewards, terminated, truncated = env.step([action])
                
        # Extract player position from RAM indices
        player_x = ram_state[28]

        # Render the current frame
        render_obs = env.envs[0].unwrapped.render()
        
        # Convert to BGR for OpenCV
        img = cv2.cvtColor(render_obs, cv2.COLOR_RGB2BGR)

        # Annotate the action and player position on the image
        action_text = f'Action: {action}'
        player_pos_text = f'Player X: {player_x}'
        catastrophe_text = f'Catastrophe'
        
        # Use a smaller font size (0.5) and thinner lines (1)
        cv2.putText(img, action_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, player_pos_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if blocker_heuristic.is_catastrophe(ram_state):
            cv2.putText(img, catastrophe_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Convert back to RGB for saving the frame
        img = create_frame(img)
        render_obs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(render_obs)

        # Reset if terminated or truncated
        if terminated or truncated:
            env.reset()

    # Save the frames as a GIF with 30 fps
    imageio.mimsave('output.gif', frames, fps=5)

if __name__ == '__main__':
    main()
