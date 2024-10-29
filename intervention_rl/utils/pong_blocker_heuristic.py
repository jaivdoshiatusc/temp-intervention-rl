import numpy as np

class PongBlockerHeuristic:
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
