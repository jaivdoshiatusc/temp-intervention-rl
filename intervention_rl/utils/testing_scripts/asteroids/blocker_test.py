import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
import cv2
import imageio
import random

# class BlockerHeuristic:
#     TOLERANCE = 0.01
#     PADDLE_ROW = 157
#     CROPPED_SHAPE = (160, 144)
#     PADDLE_COLOR = np.array([200, 72, 72])
#     CATASTROPHE_THRESHOLD = 2
#     BLOCKER_THRESHOLD = 2
    
#     def __init__(self, clearance=None, block_clearance=None):
#         self.CATASTROPHE_THRESHOLD = clearance if clearance is not None else self.CATASTROPHE_THRESHOLD
#         self.BLOCKER_THRESHOLD = block_clearance if block_clearance is not None else self.BLOCKER_THRESHOLD

#     def paddle_edges(self, observation):
#         row = observation[self.PADDLE_ROW, :, :]  # Extract the paddle row
#         color_differences = np.abs(row - self.PADDLE_COLOR)  # Difference from paddle color
#         is_paddle = np.sum(color_differences, axis=1) < self.TOLERANCE  # Boolean array
        
#         if not np.any(is_paddle):  # If no paddle is detected
#             return None, None

#         paddle_indices = np.where(is_paddle)[0]  # Indices of paddle pixels
#         left_edge = paddle_indices[0]  # First pixel
#         right_edge = paddle_indices[-1]  # Last pixel

#         return left_edge, right_edge

#     def is_catastrophe(self, obs):
#         left_edge, right_edge = self.paddle_edges(obs)

#         if left_edge is None or right_edge is None:
#             return False  # Paddle not found, no catastrophe

#         return right_edge > self.CROPPED_SHAPE[1] - self.CATASTROPHE_THRESHOLD  # Catastrophe if paddle's right edge exceeds the threshold
    
#     def is_block_zone(self, obs):
#         left_edge, right_edge = self.paddle_edges(obs)
#         if left_edge is None or right_edge is None:
#             return False  # Paddle not found, no catastrophe
#         return right_edge > self.CROPPED_SHAPE[1] - self.CATASTROPHE_THRESHOLD - self.BLOCKER_THRESHOLD

#     def should_block(self, obs, action):
#         if obs is None:
#             return False
#         if self.is_catastrophe(obs):
#             return True
#         elif self.is_block_zone(obs) and action != 3:
#             return True
#         return False

# def create_frame(frame):
#     # Draw lines on the image
#     line_color = (255, 0, 0)  # Blue color in BGR format
#     line_1 = 60
#     line_2 = 70 
#     cv2.line(frame, (0, line_1), (line_2, line_1), line_color, 1)
#     cv2.line(frame, (line_2, 0), (line_2, line_1), line_color, 1)

#     line_color = (0, 255, 0)  # Blue color in BGR format
#     line_3 = 70
#     line_4 = 80 
#     cv2.line(frame, (0, line_3), (line_4, line_3), line_color, 1)
#     cv2.line(frame, (line_4, 0), (line_4, line_3), line_color, 1)

#     return frame

def create_frame(frame):
    # Draw lines on the image
    line_color = (255, 0, 0)  # Blue color in BGR format
    line_1 = 60
    line_2 = 70 
    cv2.line(frame, (0, line_1), (frame.shape[1], line_1), line_color, 1)
    cv2.line(frame, (line_2, 0), (line_2, frame.shape[0]), line_color, 1)

    line_color = (0, 255, 0)  # Blue color in BGR format
    line_3 = 70
    line_4 = 80 
    cv2.line(frame, (0, line_3), (frame.shape[1], line_3), line_color, 1)
    cv2.line(frame, (line_4, 0), (line_4, frame.shape[0]), line_color, 1)

    return frame

def main():
    # Create the environment
    env_name = 'ALE/Asteroids-v5'
    seed = 123
    env = make_atari_env(env_name, n_envs=1, seed=seed, env_kwargs={'frameskip': 1})
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    env.reset()
    total_steps = 1000
    frames = []

    for i in range(total_steps):
        action = random.randint(0, 13)
        # Step in the environment with the chosen action
        obs, rewards, terminated, truncated = env.step([action])
        ram_state = env.envs[0].unwrapped.ale.getRAM()
                
        # Extract player position from RAM indices
        player_x = ram_state[73]
        player_y = ram_state[74]

        # Render the current frame
        render_obs = env.envs[0].unwrapped.render()
        
        # Convert to BGR for OpenCV
        img = cv2.cvtColor(render_obs, cv2.COLOR_RGB2BGR)

        # Annotate the action and player position on the image
        action_text = f'Action: {action}'
        player_pos_text = f'Player X: {player_x}, Y: {player_y}'
        
        # Use a smaller font size (0.5) and thinner lines (1)
        cv2.putText(img, action_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, player_pos_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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
