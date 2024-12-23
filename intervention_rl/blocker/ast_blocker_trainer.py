import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset

class BlockerModel(nn.Module):
    def __init__(self, action_size):
        super(BlockerModel, self).__init__()
        # Change input channels from 3 to 1 for grayscale input
        self.conv1 = nn.Conv2d(1, 4, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=2)

        # Calculate the size of the flattened conv output
        # Test with a dummy input to determine the size dynamically
        self._initialize_weights()
        test_input = torch.zeros(1, 1, 94, 144)  # Updated for new input shape
        test_output = self.conv_layers(test_input)
        conv_output_size = test_output.view(1, -1).shape[1]

        self.fc1 = nn.Linear(conv_output_size + action_size, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc_out = nn.Linear(10, 2)

    def conv_layers(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x

    def forward(self, obs, action):
        # Process observation through convolutional layers
        x = self.conv_layers(obs)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer

        # Concatenate the flattened conv output with action input
        x = torch.cat([x, action], dim=1)

        # Process through fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc_out(x)

    def _initialize_weights(self):
        # Optionally, you can initialize weights here if required.
        pass

class BlockerDataset(Dataset):
    def __init__(self, observations, actions, labels, action_size):
        self.observations = observations
        self.actions = actions
        self.labels = labels
        self.action_size = action_size  # Number of possible actions

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        obs = self.preprocess_obs(self.observations[idx])
        action = self.one_hot_encode_action(self.actions[idx], self.action_size)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return obs, action, label

    @staticmethod
    def preprocess_obs(obs):
        # Crop the observation (extract the play area)
        obs = obs[100:34 + 160, 8:152]

        # Convert to grayscale
        # Using the standard weights for RGB to grayscale conversion
        obs = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140]) # Shape: (94, 144)

        # Normalize pixel values
        obs = obs / 255.0  # Pixel values in [0, 1]

        # Add channel dimension (required for PyTorch Conv2D input)
        obs = np.expand_dims(obs, axis=0)  # Shape: (1, 160, 160)

        # Convert to float32 tensor
        obs = torch.tensor(obs, dtype=torch.float32)
        return obs

    @staticmethod
    def one_hot_encode_action(action, action_size):
        action_tensor = torch.zeros(action_size, dtype=torch.float32)
        action_tensor[int(action)] = 1.0
        return action_tensor

class AsteroidsBlockerTrainer:
    def __init__(self, action_size=1, device='cpu', lr=1e-3):
        self.device = device
        self.action_size = action_size  # Number of possible actions
        self.model = BlockerModel(action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.observations = []
        self.actions = []
        self.labels = []

        self.positive_labels = 0
        self.negative_labels = 0

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))

    def store(self, obs, action, blocker_heuristic_decision):
        if blocker_heuristic_decision == 1:
            self.positive_labels += 1
        else:
            self.negative_labels += 1
        self.observations.append(obs)
        self.actions.append(action)
        self.labels.append(blocker_heuristic_decision)

    def train(self, epochs=4, batch_size=32):
        if len(self.labels) == 0:
            return  # No data to train on

        # Create dataset and dataloader
        dataset = BlockerDataset(self.observations, self.actions, self.labels, self.action_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()

        for epoch in range(epochs):
            for batch_obs, batch_actions, batch_labels in dataloader:
                batch_obs = batch_obs.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_labels = batch_labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_obs, batch_actions)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()

    @staticmethod
    def preprocess_obs(obs):
        # Crop the observation (extract the play area)
        obs = obs[100:34 + 160, 8:152]

        # Convert to grayscale
        # Using the standard weights for RGB to grayscale conversion
        obs = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140]) # Shape: (94, 144)

        # Normalize pixel values
        obs = obs / 255.0  # Pixel values in [0, 1]

        # Add channel dimension (required for PyTorch Conv2D input)
        obs = np.expand_dims(obs, axis=0)  # Shape: (1, 160, 160)

        # Convert to float32 tensor
        obs = torch.tensor(obs, dtype=torch.float32)
        return obs

    @staticmethod
    def one_hot_encode_action(action, action_size):
        action_tensor = torch.zeros(action_size, dtype=torch.float32)
        action_tensor[int(action)] = 1.0
        return action_tensor

    def should_block(self, obs, action, blocker_heuristic_decision=None):
        # Process the obs and action
        obs_tensor = self.preprocess_obs(obs).unsqueeze(0).to(self.device)  # Add batch dimension
        action_tensor = self.one_hot_encode_action(action, self.action_size).unsqueeze(0).to(self.device)
        # Shape [batch_size, action_size]

        # Forward pass through the model
        self.model.eval()
        with torch.no_grad():
            output = self.model(obs_tensor, action_tensor)
            # Calculate probabilities
            prob = F.softmax(output, dim=-1)
            prob = torch.clamp(prob, min=1e-6)
            # Calculate entropy
            entropy = -(prob * prob.log()).sum().item()
            # Determine blocker_model_decision
            blocker_model_decision = torch.argmax(prob, dim=-1).item()
            # Calculate disagreement probability
            if blocker_heuristic_decision is not None:
                if blocker_heuristic_decision == 1:  # Heuristic says block
                    disagreement_prob = prob[0, 0].item()  # Probability of model saying "do not block"
                else:  # Heuristic says do not block
                    disagreement_prob = prob[0, 1].item()  # Probability of model saying "block"
            else:
                disagreement_prob = 0.0  # If heuristic decision is not provided

        return blocker_model_decision, entropy, disagreement_prob
    
    def get_labels(self):
        return self.positive_labels, self.negative_labels
