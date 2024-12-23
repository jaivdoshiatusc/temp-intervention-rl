import numpy as np

class BreakoutBlockerHeuristic:
    TOLERANCE = 0.01
    PADDLE_ROW = 157
    CROPPED_SHAPE = (160, 144)
    PADDLE_COLOR = np.array([200, 72, 72])
    DEFAULT_CLEARANCE = 2
    DEFAULT_BLOCK_CLEARANCE = 2

    def __init__(self, clearance=None, block_clearance=None):
        self.clearance = clearance if clearance is not None else self.DEFAULT_CLEARANCE
        self.block_clearance = block_clearance if block_clearance is not None else self.DEFAULT_BLOCK_CLEARANCE

    def paddle_edges(self, observation):
        observation = observation[34:34 + 160, 8:152]
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

        return right_edge > self.CROPPED_SHAPE[1] - self.clearance  # Catastrophe if paddle's right edge exceeds the threshold
    
    def is_block_zone(self, obs):
        left_edge, right_edge = self.paddle_edges(obs)
        if left_edge is None or right_edge is None:
            return False  # Paddle not found, no catastrophe
        return right_edge > self.CROPPED_SHAPE[1] - self.clearance - self.block_clearance

    def should_block(self, obs, action):
        if obs is None:
            return False
        if self.is_catastrophe(obs):
            return True
        elif self.is_block_zone(obs) and action != 3:
            return True
        return False
