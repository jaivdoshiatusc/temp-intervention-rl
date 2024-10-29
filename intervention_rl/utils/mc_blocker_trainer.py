import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

class BlockerModel(nn.Module):
    def __init__(self, action_size):
        super(BlockerModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=2)

        self.fc1 = nn.Linear(4 * 25 * 19 + action_size, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc_out = nn.Linear(10, 2)

    def forward(self, obs, action):
        x = torch.relu(self.conv1(obs))
        x = torch.relu(self.conv2(x))

        x = x.view(x.size(0), -1)

        x = torch.cat([x, action], dim=1)  

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
        obs = self.preprocess_obs(self.observations[idx])
        action = self.one_hot_encode_action(self.actions[idx], self.action_size)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return obs, action, label

    @staticmethod
    def preprocess_obs(obs):
        obs = obs / 255.0  # Normalize pixel values
        obs = obs.transpose(2, 0, 1)
        obs = torch.tensor(obs, dtype=torch.float32)
        return obs

    @staticmethod
    def one_hot_encode_action(action, action_size):
        action_tensor = torch.zeros(action_size, dtype=torch.float32)
        action_tensor[int(action)] = 1.0
        return action_tensor

class MCBlockerTrainer:
    def __init__(self, pretrained_weights_path=None, action_size=1, device='cpu', lr=1e-3):
        self.device = device
        self.action_size = action_size  # Number of possible actions
        self.model = BlockerModel(action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.observations = []
        self.actions = []
        self.labels = []

        if pretrained_weights_path is not None:
            self.model.load_state_dict(torch.load(pretrained_weights_path, map_location=self.device))
            self.model.eval()

    def store(self, obs, action, blocker_heuristic_decision):
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
        obs = obs / 255.0  # Normalize pixel values
        obs = obs.transpose(2, 0, 1)
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
