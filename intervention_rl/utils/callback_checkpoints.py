import os
from stable_baselines3.common.callbacks import BaseCallback
import wandb

class CustomCheckpointCallback(BaseCallback):
    """
    Custom callback for saving a model every `save_freq` steps and uploading to wandb.
    
    :param save_freq: (int) The frequency with which to save the model.
    :param save_path: (str) The directory where the model will be saved.
    :param name_prefix: (str) The prefix for the saved model.
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model"):
        super(CustomCheckpointCallback, self).__init__()
        self.save_freq = save_freq  # Save model after this many steps (total across all envs)
        self.save_path = save_path
        self.name_prefix = name_prefix

        os.makedirs(self.save_path, exist_ok=True)

        # Initialize the next checkpoint step
        self.next_save_step = save_freq

    def _on_step(self) -> bool:
        # Check if it's time to save the model based on num_timesteps
        if self.num_timesteps >= self.next_save_step:
            # Save the model
            save_file_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
            self.model.save(save_file_path)

            # Log saving action
            print(f"Saving model to {save_file_path} at step {self.num_timesteps}")

            # Upload model to wandb as an artifact
            if wandb.run is not None:
                artifact = wandb.Artifact(f"model-{self.num_timesteps}-steps", type="model")
                artifact.add_file(save_file_path)
                wandb.log_artifact(artifact)

            # Schedule the next checkpoint
            self.next_save_step += self.save_freq

        return True  # Continue training
