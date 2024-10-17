import numpy as np
import wandb
import imageio
import io  
from stable_baselines3.common.callbacks import BaseCallback
from intervention_rl.utils.my_blocker_heuristic import BlockerHeuristic

class CustomEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, n_eval_episodes, verbose=1):
        super(CustomEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.verbose = verbose
        self.blocker_heuristic = BlockerHeuristic()  # Initialize BlockerHeuristic for catastrophe detection

        # Initialize the next evaluation step
        self.next_eval_step = self.eval_freq

    def _on_step(self) -> bool:
        # Check if it's time to evaluate
        if self.num_timesteps >= self.next_eval_step:
            self.evaluate()
            # Schedule the next evaluation
            self.next_eval_step += self.eval_freq
        return True

    def evaluate(self):
        all_episode_rewards = []
        all_episode_lengths = []
        all_catastrophes = []  # List to track the number of catastrophes per episode

        # List to hold frames for creating GIFs
        frames = []

        for episode in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            episode_catastrophes = 0  # Count catastrophes in the episode
            
            while not done:
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                new_obs, reward, done, info = self.eval_env.step(action)
                
                # Capture frames for GIF
                frame = self.eval_env.envs[0].unwrapped.render()
                frames.append(frame)

                # Check for catastrophes using BlockerHeuristic
                full_obs = self.eval_env.envs[0].unwrapped.render()
                if self.blocker_heuristic.is_catastrophe(full_obs):
                    episode_catastrophes += 1  # Increment catastrophe count

                episode_reward += reward
                episode_length += 1
                obs = new_obs

            # Log episode statistics
            all_episode_rewards.append(episode_reward)
            all_episode_lengths.append(episode_length)
            all_catastrophes.append(episode_catastrophes)

        # Compute mean and std of rewards and lengths
        mean_reward = np.mean(all_episode_rewards)
        std_reward = np.std(all_episode_rewards)
        mean_length = np.mean(all_episode_lengths)
        total_catastrophes = np.sum(all_catastrophes)
        mean_catastrophes = np.mean(all_catastrophes)

        # Record evaluation metrics
        self.logger.record('eval/mean_reward', mean_reward)
        self.logger.record('eval/std_reward', std_reward)
        self.logger.record('eval/mean_length', mean_length)
        self.logger.record('eval/total_catastrophes', total_catastrophes)  # Log total number of catastrophes
        self.logger.record('eval/mean_catastrophes', mean_catastrophes)  # Log mean number of catastrophes per episode

        # Create GIF in-memory and upload to WandB without saving locally
        gif_buffer = io.BytesIO()  # In-memory buffer
        imageio.mimsave(gif_buffer, frames, format='GIF', fps=30)  # Save the GIF to buffer
        gif_buffer.seek(0)  # Reset buffer position to the beginning

        # Log the GIF to WandB
        wandb.log({f"rollout/gif_{self.num_timesteps}": wandb.Video(gif_buffer, fps=30, format="gif")})

        gif_buffer.close()

        if self.verbose > 0:
            print(f"Eval num_timesteps={self.num_timesteps}, "
                  f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}, "
                  f"total_catastrophes={total_catastrophes}, "
                  f"mean_catastrophes={mean_catastrophes:.2f}")

        # Dump the logs
        self.logger.dump(self.num_timesteps)
