import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import os

# Import or define the BlockerModel class
class BlockerModel(nn.Module):
    def __init__(self, action_size):
        super(BlockerModel, self).__init__()
        # Change input channels from 3 to 1 for grayscale input
        self.conv1 = nn.Conv2d(1, 4, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=2)

        # Calculate the size of the flattened conv output
        # Test with a dummy input to determine the size dynamically
        self._initialize_weights()
        test_input = torch.zeros(1, 1, 160, 160)  # Batch size 1, 1 channel, 160x160 grayscale input
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

class PongBlockerEvaluator:
    def __init__(self, data_path, model_weights_path, action_size=6, device='cpu'):
        self.data_path = data_path
        self.model_weights_path = model_weights_path
        self.device = device
        self.action_size = action_size  # Number of possible actions in Pong (usually 6)
        self.model = BlockerModel(action_size).to(self.device)
        self.model.load_state_dict(torch.load(self.model_weights_path, map_location=self.device))
        self.model.eval()

    @staticmethod
    def one_hot_encode_action(action, action_size):
        action_tensor = torch.zeros(action_size, dtype=torch.float32)
        action_tensor[int(action)] = 1.0
        return action_tensor

    def load_data(self):
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        self.observations = data['observations']
        self.actions = data['actions']
        self.labels = data['labels']

    def evaluate(self):
        self.load_data()

        # Convert data to tensors
        observations = torch.tensor(self.observations, dtype=torch.float32).to(self.device)
        actions = [self.one_hot_encode_action(a, self.action_size) for a in self.actions]
        actions = torch.stack(actions).to(self.device)
        labels = torch.tensor(self.labels, dtype=torch.long).to(self.device)

        # Run the model on the data
        with torch.no_grad():
            outputs = self.model(observations, actions)
            _, predictions = torch.max(outputs, dim=1)

        # Move data to CPU for evaluation metrics
        labels_cpu = labels.cpu().numpy()
        predictions_cpu = predictions.cpu().numpy()

        # Compute evaluation metrics
        accuracy = accuracy_score(labels_cpu, predictions_cpu)
        precision = precision_score(labels_cpu, predictions_cpu, zero_division=0)
        recall = recall_score(labels_cpu, predictions_cpu, zero_division=0)
        conf_matrix = confusion_matrix(labels_cpu, predictions_cpu)
        class_report = classification_report(labels_cpu, predictions_cpu, zero_division=0)

        # Print evaluation results
        print("Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)

if __name__ == '__main__':
    # Paths to the data and model weights
    data_path = 'collected_data/blocker_dataset.pkl'  # Update with your data file path
    model_weights_path = '/home1/jpdoshi/intervention-rl/intervention_rl/results/e-pong_a-ppo_et-ours_ent-0.05_al-0.01_be-0.01_s-31/2024-11-11-18-04-32/blocker/blocker_model_20000_steps.pth'  # Update with your model weights file path
    # Ensure the files exist
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
    elif not os.path.exists(model_weights_path):
        print(f"Model weights file not found: {model_weights_path}")
    else:
        evaluator = PongBlockerEvaluator(data_path, model_weights_path)
        evaluator.evaluate()
