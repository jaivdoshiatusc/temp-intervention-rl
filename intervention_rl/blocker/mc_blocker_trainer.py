import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

class BlockerModel(nn.Module):
    def __init__(self, action_size):
        super(BlockerModel, self).__init__()
        # Fully connected layers instead of convolutional layers
        self.fc1 = nn.Linear(2 + action_size, 64)  # Input size adjusted for Mountain Car's 2D observation and action space
        self.fc2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 2)  # Output for binary classification (block/do not block)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)  # Concatenate observation and action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc_out(x)

class BlockerDataset(Dataset):
    def __init__(self, observations, actions, labels, action_size):
        self.observations = observations
        self.actions = actions
        self.labels = labels
        self.action_size = action_size  # Number of possible actions

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        obs = torch.tensor(self.observations[idx], dtype=torch.float32)
        action = self.one_hot_encode_action(self.actions[idx], self.action_size)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return obs, action, label

    @staticmethod
    def one_hot_encode_action(action, action_size):
        action_tensor = torch.zeros(action_size, dtype=torch.float32)
        action_tensor[int(action)] = 1.0
        return action_tensor

class MCBlockerTrainer:
    def __init__(self, action_size=3, device='cpu', lr=1e-3):
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

    @staticmethod
    def one_hot_encode_action(action, action_size):
        action_tensor = torch.zeros(action_size, dtype=torch.float32)
        action_tensor[int(action)] = 1.0
        return action_tensor

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
    def one_hot_encode_action(action, action_size):
        action_tensor = torch.zeros(action_size, dtype=torch.float32)
        action_tensor[int(action)] = 1.0
        return action_tensor

    def should_block(self, obs, action, blocker_heuristic_decision=None):
        # Process the obs and action
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # Add batch dimension
        action_tensor = self.one_hot_encode_action(action, self.action_size).unsqueeze(0).to(self.device)

        # Forward pass through the model
        self.model.eval()
        with torch.no_grad():
            output = self.model(obs_tensor, action_tensor)
            prob = F.softmax(output, dim=-1)
            prob = torch.clamp(prob, min=1e-6)
            entropy = -(prob * prob.log()).sum().item()
            blocker_model_decision = torch.argmax(prob, dim=-1).item()
            
            if blocker_heuristic_decision is not None:
                if blocker_heuristic_decision == 1:
                    disagreement_prob = prob[0, 0].item()
                else:
                    disagreement_prob = prob[0, 1].item()
            else:
                disagreement_prob = 0.0

        return blocker_model_decision, entropy, disagreement_prob

    def get_labels(self):
        return self.positive_labels, self.negative_labels