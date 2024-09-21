import torch
import torch.nn.functional as F
from blocker_model import CNNClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import wandb

def cnn_train(args, shared_cnn_model, all_observations, all_labels, counter, lock, wandb_bool, experiment_name):
    cnn_model = CNNClassifier(6) # hard-coded action space size
    
    while True:
        with lock:
            if counter.value >= args.max_steps_for_cnn:
                break

            if len(all_observations) < 10000:
                continue

            cnn_model.load_state_dict(shared_cnn_model.state_dict())

            indices = torch.randperm(len(all_observations))
            if len(all_observations) > 20000:
                indices = indices[:20000] # Limit to 20,000 observations

            observations = [all_observations[i] for i in indices]
            labels = [all_labels[i] for i in indices]

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

        train_obs_tensor = torch.tensor([o[0] for o in train_obs], dtype=torch.float32)  
        train_actions_tensor = torch.stack([o[1] for o in train_obs])  
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
        
        eval_obs_tensor = torch.tensor([o[0] for o in eval_obs], dtype=torch.float32)  
        eval_actions_tensor = torch.stack([o[1] for o in eval_obs]) 
        eval_labels_tensor = torch.tensor(eval_labels, dtype=torch.long)

        # Training loop
        for epoch in range(10):  
            for i in range(0, len(train_obs_tensor), 32): 
                batch_obs = train_obs_tensor[i:i + 32]
                batch_actions = train_actions_tensor[i:i + 32]
                batch_labels = train_labels_tensor[i:i + 32]
                train_cnn(cnn_model, batch_obs, batch_actions, batch_labels)

        # Copy trained weights back to the shared CNN model
        shared_cnn_model.load_state_dict(cnn_model.state_dict())

        # Evaluate the model
        evaluate_cnn(shared_cnn_model, eval_obs_tensor, eval_actions_tensor, eval_labels_tensor, wandb_bool, experiment_name)

        all_observations[:] = []  # Clear all_observations
        all_labels[:] = []        # Clear all_labels

def train_cnn(cnn_model, batch_obs, batch_actions, batch_labels):
    cnn_model.train()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = cnn_model(batch_obs, batch_actions)  # Pass both observations and actions
    loss = F.cross_entropy(outputs, batch_labels)
    loss.backward()
    optimizer.step()

def evaluate_cnn(cnn_model, eval_obs_tensor, eval_actions_tensor, eval_labels_tensor, wandb_bool, experiment_name):
    if wandb_bool:
        wandb.init(project="modified_hirl", name=f"{experiment_name}_cnn")

    cnn_model.eval()
    with torch.no_grad():
        outputs = cnn_model(eval_obs_tensor, eval_actions_tensor)  # Pass both observations and actions
        predicted_labels = torch.argmax(outputs, dim=1)

    f1 = f1_score(eval_labels_tensor.numpy(), predicted_labels.numpy(), average='weighted')
    accuracy = accuracy_score(eval_labels_tensor.numpy(), predicted_labels.numpy())
    precision = precision_score(eval_labels_tensor.numpy(), predicted_labels.numpy(), average='weighted', zero_division=0)
    recall = recall_score(eval_labels_tensor.numpy(), predicted_labels.numpy(), average='weighted', zero_division=0)

    print(f"Evaluation Results: F1 Score = {f1:.4f}, Accuracy = {accuracy:.4f}, "
          f"Precision = {precision:.4f}, Recall = {recall:.4f}")

    if wandb_bool:
        wandb.log({"F1 Score": f1, "Accuracy": accuracy, "Precision": precision, "Recall": recall})
