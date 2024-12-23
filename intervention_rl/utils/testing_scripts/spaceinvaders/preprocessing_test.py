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
        elif self.is_block_zone(obs) and action not in [3, 5]:
            return True
        return False

def preprocess_obs(obs):
    # Crop the observation (extract the play area)
    obs = obs[150:194, 20:140]  # Shape: (160, 160, 3)

    # Convert to grayscale
    obs = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140])  # Shape: (160, 160)

    # Normalize pixel values
    obs = obs / 255.0  # Pixel values in [0, 1]

    # Add channel dimension (required for PyTorch Conv2D input)
    obs = np.expand_dims(obs, axis=0)  # Shape: (1, 160, 160)

    return obs

def main():
    # Create the environment
    env_name = 'ALE/SpaceInvaders-v5'
    seed = 123
    env = make_atari_env(env_name, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    blocker_heuristic = BlockerHeuristic()

    obs = env.reset()
    total_steps = 1000
    frames = []

    for i in range(total_steps):
        action = 2

        # Extract RAM state
        ram_state = env.envs[0].unwrapped.ale.getRAM()

        # Decide whether to block
        if blocker_heuristic.should_block(ram_state, action):
            action = 3

        # Step in the environment
        obs, rewards, terminated, truncated = env.step([action])

        # Render the current frame
        render_obs = env.envs[0].render()

        # Preprocess observation for testing
        preprocessed_obs = preprocess_obs(render_obs)

        # Save the frame for visualization
        frames.append((preprocessed_obs * 255).astype(np.uint8).squeeze())

        # Reset if terminated or truncated
        if terminated or truncated:
            obs = env.reset()

    # Save the frames as a GIF with 5 fps
    imageio.mimsave('output.gif', frames, fps=5)

if __name__ == '__main__':
    main()
