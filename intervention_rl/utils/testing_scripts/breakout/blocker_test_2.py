import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
import cv2

class BlockerHeuristic:
    TOLERANCE = 0.01
    PADDLE_ROW = 157
    CROPPED_SHAPE = (160, 144)
    PADDLE_COLOR = np.array([200, 72, 72])
    CATASTROPHE_THRESHOLD = 1
    BLOCKER_THRESHOLD = 4
    
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
        y = self.paddle_bottom(obs)
        if y is None:
            return False
        return y > self.CROPPED_SHAPE[1] - self.CATASTROPHE_THRESHOLD - self.BLOCKER_THRESHOLD

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
    env_name = 'ALE/Breakout-v5'
    seed = 121
    env = make_atari_env(env_name, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    
    blocker_heuristic = BlockerHeuristic()
    
    env.reset()
    
    # Fetch the first observation
    original_obs = env.envs[0].unwrapped.render()

    # Convert to BGR for OpenCV
    img = cv2.cvtColor(original_obs, cv2.COLOR_RGB2BGR)

    # # Annotate the image
    # action_text = "First Frame"
    # cv2.putText(img, action_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Crop to the play area
    # cropped_img = img[34:34 + 160, 8:152]
    # import ipdb; ipdb.set_trace()
    # print(img.shape)

    # Crop the observation (extract the play area)
    img = img[100:34 + 160, 8:152]


    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])  # Shape: (94, 144)

    # Normalize pixel values
    img = img / 255.0  # Pixel values in [0, 1]

    # Scale pixel values back to [0, 255] for saving as an image
    img = (img * 255).astype(np.uint8)

    # # Add a vertical line near the right side of the cropped play area
    # line_x_position = 150  # 10 pixels from the right edge of the cropped area
    # line_y_position = 157  # 10 pixels from the bottom edge of the cropped area
    # line_color = (0, 0, 255)  # Red in BGR format
    # line_thickness = 1
    # cv2.line(img, (line_x_position, 0), (line_x_position, img.shape[0]), line_color, line_thickness)
    # cv2.line(img, (0, line_y_position), (img.shape[1], line_y_position), line_color, line_thickness)

    # Save the first frame as an image

    cv2.imwrite('first_frame_debug.png', img)

if __name__ == '__main__':
    main()
