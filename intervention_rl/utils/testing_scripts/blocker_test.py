import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
import cv2
import imageio
import random


class BlockerHeuristic:
    TOLERANCE = 0.01
    PADDLE_COLUMN = 143
    PADDLE_COLOR = np.array([92, 186, 92])
    PLAY_AREA = [34, 34 + 160]
    DEFAULT_CLEARANCE = 8
    DEFAULT_BLOCK_CLEARANCE = 8

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
    
    def is_block_zone(self, obs):
        y = self.paddle_bottom(obs)
        if y is None:
            return False
        return y > self.PLAY_AREA[1] - self.clearance - self.block_clearance

    def should_block(self, obs, action):
        if obs is None:
            return False
        if self.is_catastrophe(obs):
            return True
        elif self.is_block_zone(obs) and action not in [2, 4]:
            return True
        return False

def main():
    # Create the environment
    env_name = 'PongNoFrameskip-v4'
    seed = 123
    env = make_atari_env(env_name, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    
    blocker_heuristic = BlockerHeuristic()
    
    env.reset()
    
    total_steps = 200
    actions = []
    
    # Random upward movement between 1 to 5 steps
    random_up_steps = random.randint(1, 5)
    print(f'Random upward steps: {random_up_steps}')
    upward_steps = 0
    move_down = False
    
    frames = []

    for i in range(total_steps):
        # Decide action based on the policy (either up or down)
        if not move_down:
            if upward_steps < random_up_steps:
                action = 2  # Move the paddle up (action 2)
                upward_steps += 1
            else:
                move_down = True
        else:
            action = 3  # Move the paddle down (action 3)

        # Fetch the current observation for BlockerHeuristic
        obs = env.envs[0].unwrapped.render()
        
        # Apply BlockerHeuristic to potentially override the downward action (action 3)
        if i > 50:
            if blocker_heuristic.should_block(obs, action):
                action = 2  # Override and block by moving up if needed
        
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

        # Draw 1-pixel horizontal lines at 162 and 178 pixels
        line_color = (255, 0, 0)  # Blue color in BGR format
        cv2.line(img, (0, 162), (img.shape[1], 162), line_color, 1)
        cv2.line(img, (0, 178), (img.shape[1], 178), line_color, 1)

        # Convert back to RGB for image saving
        original_obs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_obs = original_obs[34:34 + 160, :160]  # Crop to play area
        import ipdb; ipdb.set_trace()
        frames.append(original_obs)

    # Save the frames as a GIF with 10 fps
    imageio.mimsave('output.gif', frames, fps=10)

if __name__ == '__main__':
    main()
