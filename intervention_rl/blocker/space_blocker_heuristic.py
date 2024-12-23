import numpy as np

class SpaceInvadersBlockerHeuristic:
    CATASTROPHE_THRESHOLD = 115
    BLOCKER_THRESHOLD = 110

    def __init__(self, clearance=None, block_clearance=None):
        self.CATASTROPHE_THRESHOLD = clearance if clearance is not None else self.CATASTROPHE_THRESHOLD
        self.BLOCKER_THRESHOLD = block_clearance if block_clearance is not None else self.BLOCKER_THRESHOLD

    def is_catastrophe(self, obs):
        return obs[28] >= self.CATASTROPHE_THRESHOLD
    
    def is_block_zone(self, obs):
        return obs[28] >= self.BLOCKER_THRESHOLD
    
    def should_block(self, obs, action):
        if obs is None:
            return False
        if self.is_catastrophe(obs):
            return True
        elif self.is_block_zone(obs) and action not in [3,5]:
            return True
        return False
