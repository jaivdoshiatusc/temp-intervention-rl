import numpy as np
import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import PPO  # Adjust if using a different algorithm
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
import cv2
import os
import pickle

class PaddleHandler:
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

class PongBlockerTrainer:
    @staticmethod
    def preprocess_obs(obs):
        # Crop the observation (extract the play area)
        obs = obs[34:34 + 160, :160]  # Shape: (160, 160, 3)

        # Convert to grayscale
        # Using the standard weights for RGB to grayscale conversion
        obs = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140])  # Shape: (160, 160)

        # Normalize pixel values
        obs = obs / 255.0  # Pixel values in [0, 1]

        # Add channel dimension (required for PyTorch Conv2D input)
        obs = np.expand_dims(obs, axis=0)  # Shape: (1, 160, 160)

        # Convert to float32 tensor
        obs = torch.tensor(obs, dtype=torch.float32)
        return obs

class DataCollector:
    def __init__(self, model_path, total_steps=200, data_dir='collected_data', save_file='blocker_dataset.pkl'):
        self.model_path = model_path
        self.total_steps = total_steps
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.save_file = save_file

    def load_model(self):
        # Load the pretrained agent model
        self.model = PPO.load(self.model_path)

    def run_rollout(self):
        # Create the Pong environment
        env_name = 'PongNoFrameskip-v4'
        seed = 123
        env = make_atari_env(env_name, n_envs=1, seed=seed)
        env = VecFrameStack(env, n_stack=4)
        env = VecTransposeImage(env)

        paddle_handler = PaddleHandler()
        obs = env.reset()

        observations = []
        actions = []
        labels = []

        for step in range(self.total_steps):
            # Predict the action using the agent's policy
            action, _states = self.model.predict(obs, deterministic=True)
            action = int(action)  # Ensure the action is an integer scalar

            # Get the original observation from the environment
            original_obs = env.envs[0].unwrapped.render()

            # Use the expert heuristic blocker to decide whether to block
            blocker_decision = paddle_handler.should_block(original_obs, action)

            # Modify the action if the blocker decides to block
            modified_action = action
            if blocker_decision:
                # Assuming the safe action is 0 (NOOP)
                modified_action = 2

            # Preprocess the observation using PongBlockerTrainer's method
            preprocessed_obs = PongBlockerTrainer.preprocess_obs(original_obs)

            # Convert tensor to NumPy array
            preprocessed_obs = preprocessed_obs.numpy()

            # Append data to lists
            observations.append(preprocessed_obs)
            actions.append(action)
            labels.append(int(blocker_decision))  # Convert boolean to int

            # Step the environment with the modified action
            obs, rewards, dones, infos = env.step([modified_action])

            if dones[0]:
                # Reset the environment if an episode is done
                obs = env.reset()

        # Save the collected data as a pickle file
        self.save_data(observations, actions, labels)
        print(f"Data collection completed and saved to {self.save_file}")

    def save_data(self, observations, actions, labels):
        # Convert lists to numpy arrays for efficient storage
        observations = np.array(observations)
        actions = np.array(actions)
        labels = np.array(labels)

        # Count the number of positive and negative labels
        num_positive = np.sum(labels == 1)
        num_negative = np.sum(labels == 0)
        total_labels = len(labels)

        print(f"Number of positive labels (1): {num_positive} out of {total_labels}")
        print(f"Number of negative labels (0): {num_negative} out of {total_labels}")

        # Check if undersampling is needed
        if num_positive > 0 and num_negative > num_positive:
            # Get indices of positive and negative samples
            positive_indices = np.where(labels == 1)[0]
            negative_indices = np.where(labels == 0)[0]

            # Randomly select negative samples equal to the number of positive samples
            np.random.seed(42)  # For reproducibility
            undersample_size = num_positive
            undersampled_negative_indices = np.random.choice(negative_indices, size=undersample_size, replace=False)

            # Combine positive and undersampled negative indices
            combined_indices = np.concatenate([positive_indices, undersampled_negative_indices])

            # Shuffle the combined indices
            np.random.shuffle(combined_indices)

            # Undersample the data
            observations = observations[combined_indices]
            actions = actions[combined_indices]
            labels = labels[combined_indices]

            print(f"After undersampling, dataset size: {len(labels)} (Positive: {num_positive}, Negative: {undersample_size})")
        else:
            print("No undersampling performed.")

        # Save the blocker dataset using pickle
        with open(os.path.join(self.data_dir, self.save_file), 'wb') as f:
            pickle.dump(
                {
                    'observations': observations,
                    'actions': actions,
                    'labels': labels
                },
                f
            )

if __name__ == '__main__':
    # Example usage
    model_path = '/home1/jpdoshi/intervention-rl/intervention_rl/results/2024-11-09-11-09-40/e-pong_a-ppo_et-hirl_ent-0.05_al-0.01_be-0.01_s-34/agent/ppo_hirl_model_2500000_steps.zip'  # Update with your model's path
    data_collector = DataCollector(model_path, total_steps=20000, data_dir='collected_data', save_file='blocker_dataset.pkl')
    data_collector.load_model()
    data_collector.run_rollout()

