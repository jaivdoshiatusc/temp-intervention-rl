import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
import cv2
import imageio
import random

class BlockerHeuristic:
    TOLERANCE = 0.01
    PADDLE_ROW = 157
    CROPPED_SHAPE = (160, 144)
    PADDLE_COLOR = np.array([200, 72, 72])
    CATASTROPHE_THRESHOLD = 2
    BLOCKER_THRESHOLD = 2
    
    def __init__(self, clearance=None, block_clearance=None):
        self.CATASTROPHE_THRESHOLD = clearance if clearance is not None else self.CATASTROPHE_THRESHOLD
        self.BLOCKER_THRESHOLD = block_clearance if block_clearance is not None else self.BLOCKER_THRESHOLD

    def paddle_edges(self, observation):
        row = observation[self.PADDLE_ROW, :, :]  # Extract the paddle row
        color_differences = np.abs(row - self.PADDLE_COLOR)  # Difference from paddle color
        is_paddle = np.sum(color_differences, axis=1) < self.TOLERANCE  # Boolean array
        
        if not np.any(is_paddle):  # If no paddle is detected
            return None, None

        paddle_indices = np.where(is_paddle)[0]  # Indices of paddle pixels
        left_edge = paddle_indices[0]  # First pixel
        right_edge = paddle_indices[-1]  # Last pixel

        return left_edge, right_edge

    def is_catastrophe(self, obs):
        left_edge, right_edge = self.paddle_edges(obs)

        if left_edge is None or right_edge is None:
            return False  # Paddle not found, no catastrophe

        return right_edge > self.CROPPED_SHAPE[1] - self.CATASTROPHE_THRESHOLD  # Catastrophe if paddle's right edge exceeds the threshold
    
    def is_block_zone(self, obs):
        left_edge, right_edge = self.paddle_edges(obs)
        if left_edge is None or right_edge is None:
            return False  # Paddle not found, no catastrophe
        return right_edge > self.CROPPED_SHAPE[1] - self.CATASTROPHE_THRESHOLD - self.BLOCKER_THRESHOLD

    def should_block(self, obs, action):
        if obs is None:
            return False
        if self.is_catastrophe(obs):
            return True
        elif self.is_block_zone(obs) and action != 3:
            return True
        return False

def main():
    # Create the environment
    env_name = 'ALE/Breakout-v5'
    seed = 123
    env = make_atari_env(env_name, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    import ipdb; ipdb.set_trace()
    
    blocker_heuristic = BlockerHeuristic()
    
    env.reset()
    
    total_steps = 200
    actions = []
    frames = []

    for i in range(total_steps):
        action = 2  # Move the paddle left (action 3)

        # Fetch the current observation for BlockerHeuristic
        obs = env.envs[0].unwrapped.render()
        cropped_obs = obs[34:34 + 160, 8:152]
        
        if blocker_heuristic.should_block(cropped_obs, action):
            action = 3  # Override and block by moving up if needed
        
        # Now that the action is finalized (policy or blocked), step in the environment
        obs, rewards, terminated, truncated = env.step([action])
        original_obs = env.envs[0].unwrapped.render()

        # Convert to BGR for OpenCV
        img = cv2.cvtColor(original_obs, cv2.COLOR_RGB2BGR)        
        actions.append(action)

        # Annotate the action on the image
        action_text = f'Action: {action}'
        cv2.putText(img, action_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Check for catastrophe using the original observation
        is_catastrophe = blocker_heuristic.is_catastrophe(original_obs)
        if is_catastrophe:
            # Draw 'catastrophe' on the image
            cv2.putText(img, 'catastrophe', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Convert back to RGB for image saving
        original_obs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_obs = original_obs[34:34 + 160, :160]  # Crop to play area
        frames.append(original_obs)

    # Save the frames as a GIF with 10 fps
    imageio.mimsave('output.gif', frames, fps=10)

if __name__ == '__main__':
    main()
