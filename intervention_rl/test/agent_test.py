import numpy as np
import gymnasium as gym
import torch
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

class PongRollout:
    def __init__(self, model_path, total_steps=200):
        self.model_path = model_path
        self.total_steps = total_steps

    def load_model(self):
        # Assuming the model is a torch model
        self.model = torch.load(self.model_path)
        self.model.eval()

    def run_rollout(self):
        # Create the environment
        env_name = 'PongNoFrameskip-v4'
        seed = 123
        env = make_atari_env(env_name, n_envs=1, seed=seed)
        env = VecFrameStack(env, n_stack=4)
        env = VecTransposeImage(env)

        paddle_handler = PaddleHandler()
        
        obs = env.reset()

        frames = []
        
        for step in range(self.total_steps):
            # Convert observation to tensor and reshape for the model
            obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2)
            
            # Dummy action tensor (replace this with actual action space size)
            action_tensor = torch.zeros((1, 2), dtype=torch.float32)
            
            # Predict action using the model
            with torch.no_grad():
                action = self.model(obs_tensor, action_tensor)
                action = action.argmax(dim=1).item()
            
            # Execute the action in the environment
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
        imageio.mimsave('output_rollout.gif', frames, fps=10)
        print("Rollout completed and saved as output_rollout.gif")

if __name__ == '__main__':
    # Example usage
    model_path = 'path_to_your_model.pt'  # Provide the correct model path here
    pong_rollout = PongRollout(model_path)
    pong_rollout.load_model()
    pong_rollout.run_rollout()
