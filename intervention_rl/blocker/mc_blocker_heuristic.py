import numpy as np

class MCBlockerHeuristic:
    DEFAULT_CLEARANCE = -1.10
    FORCE = 0.001
    GRAVITY = 0.0025

    def __init__(self, clearance=None):
        self.clearance = clearance if clearance is not None else self.DEFAULT_CLEARANCE
    
    def detect_catastrophe(self, state, action):
        position, velocity = state

        new_velocity = np.clip(velocity + (action - 1) * self.FORCE - np.cos(3 * position) * self.GRAVITY, -0.07, 0.07)
        new_position = np.clip(position + new_velocity, -1.2, 0.6)
        
        if new_position >= 0.40 and new_velocity >= 0.1:
            return True
        return False

    def is_catastrophe(self, state):
        position, velocity = state
        if position >= 0.40 and velocity >= 0.01:
            return True
        return False
    
    def should_block(self, obs, action):
        if obs is None:
            return False
        return self.is_catastrophe(obs) or self.detect_catastrophe(obs, action)
