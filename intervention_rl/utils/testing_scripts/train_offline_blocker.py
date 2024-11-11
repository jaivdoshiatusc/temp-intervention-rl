import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Define the BlockerModel class
class BlockerModel(nn.Module):
    def __init__(self, action_size):
        super(BlockerModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=2)

        # Calculate the flattened size after convolution layers
        # Assuming input image size is (3, 160, 160) after preprocessing
        conv_output_size = 4 * 19 * 19  # Adjust if input size changes

        self.fc1 = nn.Linear(conv_output_size + action_size, 10)
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

# Define the BlockerDataset class
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
        # Crop and normalize the observation
        obs = obs[34:34 + 160, :160]
        obs = obs / 255.0  # Normalize pixel values
        obs = obs.transpose(2, 0, 1)  # Convert from HWC to CHW
        obs = torch.tensor(obs, dtype=torch.float32)
        return obs

    @staticmethod
    def one_hot_encode_action(action, action_size):
        action_tensor = torch.zeros(action_size, dtype=torch.float32)
        action_tensor[int(action)] = 1.0
        return action_tensor

def train_offline_blocker(pickle_file_path, action_size=6, device='cpu', lr=1e-3, epochs=8, batch_size=32, pre_trained_model_path=None):
    # Load the dataset from the pickle file
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
    observations = data['observations']
    actions = data['actions']
    labels = data['labels']

    # Create the dataset
    dataset = BlockerDataset(observations, actions, labels, action_size)

    # Split the dataset into training and evaluation sets (e.g., 80% train, 20% eval)
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    # Print the number of positive and negative labels in train and eval sets
    # For training set
    train_indices = train_dataset.indices
    train_labels = [dataset.labels[i] for i in train_indices]
    num_positive_train = sum(train_labels)
    num_negative_train = len(train_labels) - num_positive_train
    print(f"\nTraining set: {len(train_labels)} samples")
    print(f" - Positive labels: {num_positive_train}")
    print(f" - Negative labels: {num_negative_train}")

    # For evaluation set
    eval_indices = eval_dataset.indices
    eval_labels = [dataset.labels[i] for i in eval_indices]
    num_positive_eval = sum(eval_labels)
    num_negative_eval = len(eval_labels) - num_positive_eval
    print(f"\nEvaluation set: {len(eval_labels)} samples")
    print(f" - Positive labels: {num_positive_eval}")
    print(f" - Negative labels: {num_negative_eval}\n")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, optimizer, and loss function
    model = BlockerModel(action_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Load pre-trained model weights if provided
    if pre_trained_model_path is not None:
        model.load_state_dict(torch.load(pre_trained_model_path))
        print(f"Loaded pre-trained model from {pre_trained_model_path}")

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_obs, batch_actions, batch_labels in train_loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_obs, batch_actions)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_obs.size(0)

        epoch_loss /= len(train_dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Optionally, save the trained model
    model_save_path = os.path.join(os.path.dirname(pickle_file_path), 'trained_blocker_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Evaluate the trained model
    print("\nEvaluating the trained model:")
    evaluate_model(model, eval_loader, device)

    # If a pre-trained model is provided, evaluate it as well
    if pre_trained_model_path is not None:
        pre_trained_model = BlockerModel(action_size).to(device)
        pre_trained_model.load_state_dict(torch.load(pre_trained_model_path))
        print("\nEvaluating the pre-trained model:")
        evaluate_model(pre_trained_model, eval_loader, device)

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_obs, batch_actions, batch_labels in dataloader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_obs, batch_actions)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == '__main__':
    # Replace with the actual path to your pickle file
    pickle_file_path = '/home1/jpdoshi/intervention-rl/intervention_rl/results/e-pong_a-ppo_et-ours_gae-0.9_ent-0.01_lr-0.0025_ns-128_s-42/2024-11-04-23-55-40/blocker/blocker_model_dataset_20016_steps.pkl'   
    action_size = 6  # Number of possible actions in Pong
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Path to a pre-trained blocker model for comparison (set to None if not available)
    pre_trained_model_path = '/home1/jpdoshi/intervention-rl/intervention_rl/blocker/blocker_weights/ours_blocker_model_120000_steps.pth'
    train_offline_blocker(
        pickle_file_path=pickle_file_path,
        action_size=action_size,
        device=device,
        lr=1e-3,
        epochs=8,
        batch_size=32,
        pre_trained_model_path=pre_trained_model_path  # Set to None if you don't have one
    )
