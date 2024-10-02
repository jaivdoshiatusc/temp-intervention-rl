import numpy as np

TOLERANCE = 0.01
PADDLE_COLUMN = 143
PADDLE_COLOR = np.array([92, 186, 92])
PLAY_AREA = [34, 34 + 160]
DEFAULT_CLEARANCE = 16
DEFAULT_BLOCK_CLEARANCE = 16

def paddle_bottom(observation, paddle="right"):
    column = observation[:, PADDLE_COLUMN, :] - PADDLE_COLOR
    found = (np.sum(np.abs(column), axis=1) < TOLERANCE).astype(int)
    r = np.argmax(np.flipud(found))
    r = (len(found) - r - 1)
    if not found[r]:
        return None
    else:
        return r

def is_catastrophe(obs, clearance=DEFAULT_CLEARANCE):    
    y = paddle_bottom(obs)
    if y is None:
        return False
    return y > PLAY_AREA[1] - clearance

def should_block(obs,
                 action,
                 clearance=DEFAULT_CLEARANCE,
                 block_clearance=DEFAULT_BLOCK_CLEARANCE):
    if obs is None:
        return False
    if is_catastrophe(obs, clearance):
        return True
    elif is_catastrophe(obs, clearance + block_clearance) and (action != 2 or action != 4):
        return True
    return False