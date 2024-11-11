import numpy as np

class MCBlockerHeuristic:
    DEFAULT_CLEARANCE = -1.15

    def __init__(self, clearance=None):
        self.clearance = clearance if clearance is not None else self.DEFAULT_CLEARANCE
    
    def detect_catastrophe(self, state, action, force=0.001, gravity=0.0025):
        tuple_state = tuple(state)
        position = tuple_state[0]
        velocity = tuple_state[1]

        new_velocity = np.clip(velocity + (action - 1) * force - np.cos(3 * position) * gravity, -0.07, 0.07)
        new_position = np.clip(position + new_velocity, -1.2, 0.6)
        
        return new_position < self.DEFAULT_CLEARANCE

    def is_catastrophe(self, state):
        tuple_state = tuple(state)
        position = tuple_state[0]
        return position < self.DEFAULT_CLEARANCE
    
    def should_block(self, obs, action):
        if obs is None:
            return False
        if self.is_catastrophe(obs) or self.detect_catastrophe(obs, action):
            return True
        return False