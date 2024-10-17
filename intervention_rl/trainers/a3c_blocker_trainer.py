import wandb
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from intervention_rl.models.a3c_blocker_model import CNNClassifier

def blocker_train(cfg, shared_blocker_model, shared_observations, shared_labels, counter, lock, use_wandb, experiment_name='experiment'):
    blocker_model = CNNClassifier(6) # hard-coded action space size
    warning_flag = False
    while True:
        with lock:
            if counter.value >= cfg.algorithm.max_steps_for_blocker_training:
                break

            if len(shared_observations) < 1000:
                continue

            blocker_model.load_state_dict(shared_blocker_model.state_dict())

            indices = torch.randperm(len(shared_observations))

            if len(shared_labels) < len(shared_observations) and warning_flag == False:
                print(f"Warning: shared_labels length {len(shared_labels)} is less than shared_observations length {len(shared_observations)}")
                warning_flag = True
                continue  # Skip this loop iteration if the lengths do not match

            warning_flag = False  # Reset warning flag after handling

            # No longer limit to 20,000, train on the full dataset
            observations = [shared_observations[i] for i in indices]
            labels = [shared_labels[i] for i in indices]

        split_index = int(0.9 * len(observations))
        train_obs = observations[:split_index]
        train_labels = labels[:split_index]
        eval_obs = observations[split_index:]
        eval_labels = labels[split_index:]

        print(f"Training dataset size: {len(train_obs)}")
        print(f"Number of 1 labels in training: {train_labels.count(1)}")
        print(f"Number of 0 labels in training: {train_labels.count(0)}")
        print(f"Evaluation dataset size: {len(eval_obs)}")
        print(f"Number of 1 labels in evaluation: {eval_labels.count(1)}")
        print(f"Number of 0 labels in evaluation: {eval_labels.count(0)}")

        train_obs_np = np.array([o[0] for o in train_obs])
        train_obs_tensor = torch.tensor(train_obs_np, dtype=torch.float32)
        train_actions_tensor = torch.stack([o[1] for o in train_obs])  
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
        
        eval_obs_np = np.array([o[0] for o in eval_obs])
        eval_obs_tensor = torch.tensor(eval_obs_np, dtype=torch.float32)
        eval_actions_tensor = torch.stack([o[1] for o in eval_obs]) 
        eval_labels_tensor = torch.tensor(eval_labels, dtype=torch.long)

        # Training loop
        for epoch in range(10):  
            for i in range(0, len(train_obs_tensor), 32): 
                batch_obs = train_obs_tensor[i:i + 32]
                batch_actions = train_actions_tensor[i:i + 32]
                batch_labels = train_labels_tensor[i:i + 32]
                train_blocker(blocker_model, batch_obs, batch_actions, batch_labels)

        # Copy trained weights back to the shared CNN model
        shared_blocker_model.load_state_dict(blocker_model.state_dict())

        # Evaluate the model
        evaluate_blocker(shared_blocker_model, eval_obs_tensor, eval_actions_tensor, eval_labels_tensor, use_wandb, experiment_name)

def train_blocker(blocker_model, batch_obs, batch_actions, batch_labels):
    blocker_model.train()
    optimizer = torch.optim.Adam(blocker_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = blocker_model(batch_obs, batch_actions)  # Pass both observations and actions
    loss = F.cross_entropy(outputs, batch_labels)
    loss.backward()
    optimizer.step()

def evaluate_blocker(blocker_model, eval_obs_tensor, eval_actions_tensor, eval_labels_tensor, use_wandb, experiment_name):
    if use_wandb:
        wandb.init(project="modified_hirl", name=f"{experiment_name}_blocker")

    blocker_model.eval()
    with torch.no_grad():
        outputs = blocker_model(eval_obs_tensor, eval_actions_tensor)  # Pass both observations and actions
        predicted_labels = torch.argmax(outputs, dim=1)

    f1 = f1_score(eval_labels_tensor.numpy(), predicted_labels.numpy(), average='weighted')
    accuracy = accuracy_score(eval_labels_tensor.numpy(), predicted_labels.numpy())
    precision = precision_score(eval_labels_tensor.numpy(), predicted_labels.numpy(), average='weighted', zero_division=0)
    recall = recall_score(eval_labels_tensor.numpy(), predicted_labels.numpy(), average='weighted', zero_division=0)

    print(f"Evaluation Results: F1 Score = {f1:.4f}, Accuracy = {accuracy:.4f}, "
          f"Precision = {precision:.4f}, Recall = {recall:.4f}")

    if use_wandb:
        wandb.log({"F1 Score": f1, "Accuracy": accuracy, "Precision": precision, "Recall": recall})