import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
import cv2
import imageio

class PaddleHandler:
    TOLERANCE = 0.01
    PADDLE_COLUMN = 143
    PADDLE_COLOR = np.array([92, 186, 92])
    PLAY_AREA = [34, 34 + 160]
    DEFAULT_CLEARANCE = 16
    DEFAULT_BLOCK_CLEARANCE = 16

    def __init__(self, clearance=None, block_clearance=None):
        self.clearance = clearance if clearance is not None else self.DEFAULT_CLEARANCE
        self.block_clearance = block_clearance if block_clearance is not None else self.DEFAULT_BLOCK_CLEARANCE

    def paddle_bottom(self, observation, paddle="right"):
        column = observation[:, self.PADDLE_COLUMN, :] - self.PADDLE_COLOR
        found = (np.sum(np.abs(column), axis=1) < self.TOLERANCE).astype(int)
        r = np.argmax(np.flipud(found))
        r = (len(found) - r - 1)
        if not found[r]:
            return None
        else:
            return r

    def is_catastrophe(self, obs):
        y = self.paddle_bottom(obs)
        if y is None:
            return False
        return y > self.PLAY_AREA[1] - self.clearance

def main():
    # Create the environment
    env_name = 'PongNoFrameskip-v4'
    seed = 123
    env = make_atari_env(env_name, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    
    paddle_handler = PaddleHandler()
    
    env.reset()
    
    total_steps = 200
    actions = []
    # Create action pattern: 1 for 20 steps, 2 for 20 steps, and repeat
    for i in range(total_steps):
        if ((i // 20) % 2) == 0:
            action = 2
        else:
            action = 3
        actions.append(action)
    
    frames = []
    
    for i, action in enumerate(actions):
        obs, rewards, terminated, truncated = env.step([action])
        # Get the original Atari frame (210, 160, 3)
        original_obs = env.envs[0].unwrapped.render()

        # Check for catastrophe using the original observation
        is_catastrophe = paddle_handler.is_catastrophe(original_obs)
        if is_catastrophe:
            # Draw 'catastrophe' on the image
            img = cv2.cvtColor(original_obs, cv2.COLOR_RGB2BGR)
            cv2.putText(img, 'catastrophe', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            # Convert back to RGB
            original_obs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw 1-pixel horizontal lines at 162 and 178 pixels
        line_color = (255, 0, 0)  # Blue color in BGR format
        cv2.line(original_obs, (0, 162), (original_obs.shape[1], 162), line_color, 1)
        cv2.line(original_obs, (0, 178), (original_obs.shape[1], 178), line_color, 1)

        frames.append(original_obs)
    
    # Save the frames as a GIF with 10 fps
    imageio.mimsave('output.gif', frames, fps=2)

if __name__ == '__main__':
    main()
