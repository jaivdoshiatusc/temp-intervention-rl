import cv2
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

def create_atari_env(env_id):
    env = gym.make(env_id)
    env = FullAndRescaledObsWrapper(env)
    env = NormalizedEnv(env)
    return env

def _process_frame42(frame):
    # Process to (1, 42, 42) for the actor-critic model
    frame = frame[34:34 + 160, :160]
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2, keepdims=True)  # Convert to grayscale
    frame = frame.astype(np.float32) * (1.0 / 255.0)  # Normalize to [0, 1]
    return np.moveaxis(frame, -1, 0)  # Shape to [1, 42, 42]

def _process_frame_CNN(frame):
    # New function to crop and resize to (3, 105, 80) for the CNN
    frame = frame[34:34 + 160, :160]  # Crop to play area
    frame = cv2.resize(frame, (80, 105))  # Resize to desired dimensions
    frame = frame.astype(np.float32) * (1.0 / 255.0)  # Normalize to [0, 1]
    
    # Ensure the frame has 3 channels
    if frame.ndim == 2:  # If it's a single-channel (grayscale) image
        print("Converting grayscale to RGB")
        frame = np.stack((frame,) * 3, axis=-1)  # Convert to RGB
    return frame  # Shape is [105, 80, 3]

class FullAndRescaledObsWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(FullAndRescaledObsWrapper, self).__init__(env)
        self.full_observation_space = env.observation_space
        self.observation_space = Box(0.0, 1.0, (1, 42, 42), dtype=np.float32)
        self.full_obs = None

    def observation(self, observation):
        self.full_obs = observation  # Keep the full observation
        return _process_frame42(observation)  # Return the downsized observation for the agent

    def get_full_obs(self):
        return self.full_obs  # Access full observation for blocker functions

    def get_cropped_obs(self):
        cropped_obs = _process_frame_CNN(self.full_obs) 
        return np.moveaxis(cropped_obs, -1, 0)  

class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)
