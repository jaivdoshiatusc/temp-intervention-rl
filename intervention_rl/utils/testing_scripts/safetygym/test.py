import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
import cv2
import imageio
import random
import safety_gymnasium

def main():
    # Create the environment
    env_id = 'SafetyPointGoal1-v0'
    env = safety_gymnasium.make(env_id, render_mode="rgb_array", camera_id=1)

    obs, info = env.reset()
    total_steps = 1000
    frames = []

    for i in range(total_steps):
        act = env.action_space.sample()
        obs, reward, cost, terminated, truncated, info = env.step(act)
        import ipdb; ipdb.set_trace()

        # Render the current frame
        render_obs = env.render()
        
        # Convert to BGR for OpenCV
        img = cv2.cvtColor(render_obs, cv2.COLOR_RGB2BGR)
        render_obs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(render_obs)

        if terminated or truncated:
            obs, info = env.reset()

    # Save the frames as a GIF with 30 fps
    imageio.mimsave('output_1.gif', frames, fps=30)

if __name__ == '__main__':
    main()
