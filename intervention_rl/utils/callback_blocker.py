from stable_baselines3.common.callbacks import BaseCallback
import os
import torch
import pickle

class BlockerTrainingCallback(BaseCallback):
    def __init__(
        self,
        train_freq=10000,
        epochs=4,
        save_freq=5000,
        save_path=None,
        save_blocker_dataset=False,
        name_prefix="blocker_model",
        verbose=0
    ):
        super(BlockerTrainingCallback, self).__init__(verbose)
        self.train_freq = train_freq
        self.epochs = epochs
        self.save_freq = save_freq
        self.save_path = save_path
        self.save_blocker_dataset = save_blocker_dataset
        self.name_prefix = name_prefix
        self.training_stopped = False
        self.blocker_switch_time = None

        # Initialize the next training and saving steps
        self.next_blocker_train_step = self.train_freq
        self.next_blocker_save_step = self.save_freq

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_training_start(self):
        self.blocker_switch_time = getattr(self.model, 'blocker_switch_time', None)
        if self.blocker_switch_time is None:
            self.logger.record(
                "blocker/initialization",
                "Blocker switch time not defined in model. Blocker will train indefinitely.",
                exclude=("stdout", "log")
            )
        else:
            self.logger.record(
                "blocker/initialization",
                f"Blocker will stop training after {self.blocker_switch_time} timesteps.",
                exclude=("stdout", "log")
            )

    def _on_step(self) -> bool:
        if self.blocker_switch_time is not None and self.num_timesteps > self.blocker_switch_time:
            if not self.training_stopped:
                self.logger.record(
                    "blocker/training_status",
                    f"Blocker training stopped after {self.blocker_switch_time} timesteps.",
                    exclude=("stdout", "log")
                )
                self.training_stopped = True
                if self.save_blocker_dataset:
                    self._save_blocker_dataset()
                self._clear_blocker_dataset()

            return True

        if not self.training_stopped:
            # Training the blocker model at specified intervals
            if self.num_timesteps >= self.next_blocker_train_step:
                if hasattr(self.model, 'blocker_model'):
                    num_data_points = len(self.model.blocker_model.observations)
                    self.logger.record(
                        "blocker/train_data",
                        f"Training BlockerModel with {num_data_points} data points at timestep {self.num_timesteps}"
                    )

                    # Train the blocker model
                    self.model.blocker_model.train(self.epochs)
                    self.logger.record(
                        "blocker/train_status",
                        f"Blocker model trained at timestep {self.num_timesteps}"
                    )

                    # Update the next training step
                    self.next_blocker_train_step += self.train_freq
                else:
                    self.logger.record(
                        "blocker/error",
                        "Model does not have 'blocker_model' attribute."
                    )

            # Saving the blocker model weights at specified intervals
            if self.save_path is not None and self.num_timesteps >= self.next_blocker_save_step:
                if hasattr(self.model, 'blocker_model'):
                    save_file = os.path.join(
                        self.save_path,
                        f"{self.name_prefix}_{self.num_timesteps}_steps.pth"
                    )
                    # Save the state_dict of the blocker model
                    torch.save(self.model.blocker_model.model.state_dict(), save_file)
                    self.logger.record(
                        "blocker/save_status",
                        f"Blocker model weights saved at timestep {self.num_timesteps} to {save_file}"
                    )

                    # Update the next save step
                    self.next_blocker_save_step += self.save_freq
                else:
                    self.logger.record(
                        "blocker/error",
                        "Model does not have 'blocker_model' attribute."
                    )

        return True
    
    def _clear_blocker_dataset(self):
        """Clear the dataset after training is complete to free up memory."""
        if hasattr(self.model, 'blocker_model'):
            self.model.blocker_model.observations.clear()
            self.model.blocker_model.actions.clear()
            self.model.blocker_model.labels.clear()
            del self.model.blocker_model.observations
            del self.model.blocker_model.actions
            del self.model.blocker_model.labels

            torch.cuda.empty_cache()  # Clear any GPU memory if CUDA is used
            self.logger.record("blocker/memory", "Blocker dataset cleared after training.")
        else:
            self.logger.record(
                "blocker/error", "Model does not have 'blocker_model' attribute to clear."
            )

    def _save_blocker_dataset(self):
        if hasattr(self.model, 'blocker_model'):
            save_file = os.path.join(
                self.save_path,
                f"{self.name_prefix}_dataset_{self.num_timesteps}_steps.pkl"
            )
            # Save the blocker dataset using pickle
            with open(save_file, 'wb') as f:
                pickle.dump(
                    {
                        'observations': self.model.blocker_model.observations,
                        'actions': self.model.blocker_model.actions,
                        'labels': self.model.blocker_model.labels
                    },
                    f
                )
            self.logger.record(
                "blocker/save_data",
                f"Blocker dataset saved at timestep {self.num_timesteps} to {save_file}"
            )
        else:
            self.logger.record(
                "blocker/error",
                "Model does not have 'blocker_model' attribute to save dataset."
            )
