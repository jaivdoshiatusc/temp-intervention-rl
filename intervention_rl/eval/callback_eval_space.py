import numpy as np
import wandb
import imageio
import cv2
import io  
from stable_baselines3.common.callbacks import BaseCallback

class SpaceInvadersEvalCallback(BaseCallback):

    def __init__(self, cfg, eval_env, eval_freq, eval_seed, gif_freq, n_eval_episodes, new_action, verbose=1):
        super(SpaceInvadersEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_seed = eval_seed
        self.gif_freq = gif_freq
        self.n_eval_episodes = n_eval_episodes
        self.catastrophe_clearance = cfg.env.catastrophe_clearance
        self.blocker_clearance = cfg.env.blocker_clearance
        self.verbose = verbose

        self.new_action = new_action

        self.cum_catastrophe = 0
        self.cum_env_intervention = 0
        self.cum_exp_intervention = 0
        self.cum_disagreement = 0

        self.blocker_cum_catastrophe = 0
        self.blocker_cum_env_intervention = 0
        self.blocker_cum_exp_intervention = 0
        self.blocker_cum_disagreement = 0

        # Initialize the next evaluation step
        self.next_eval_step = self.eval_freq
        self.next_gif_step = self.gif_freq

    def _on_step(self) -> bool:
        # Check if it's time to evaluate
        if self.num_timesteps >= self.next_eval_step:
            self.evaluate()
            # Schedule the next evaluation
            self.next_eval_step += self.eval_freq
        return True
    
    def create_frame(self):
        frame = self.eval_env.envs[0].unwrapped.render()

        # Draw vertical lines on the image
        line_color = (255, 0, 0)  # Blue color in BGR format
        blocker_line = self.blocker_clearance
        catastrophe_line = self.catastrophe_clearance
        cv2.line(frame, (blocker_line, 0), (blocker_line, frame.shape[0]), line_color, 1)
        cv2.line(frame, (catastrophe_line, 0), (catastrophe_line, frame.shape[0]), line_color, 1)

        return frame
        
    def draw_catastrophe(self, frame):
        # Draw 'catastrophe' on the image
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        cv2.putText(img, 'catastrophe', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Red color, smaller font size, lower position
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert back to RGB for rendering

        return frame

    def evaluate(self):
        all_episode_rewards = []
        all_episode_lengths = []
        all_catastrophes = []  
        all_env_interventions = []
        all_exp_interventions = []
        all_disagreements = []

        create_gif = self.num_timesteps >= self.next_gif_step and wandb.run is not None
        frames = [] if create_gif else None

        # Access experiment parameters
        exp_type = self.model.exp_type
        blocker_switch_time = self.model.blocker_switch_time
        num_timesteps = self.num_timesteps

        for episode in range(self.n_eval_episodes):
            episode_env_interventions = 0
            episode_exp_interventions = 0
            episode_disagreements = 0

            episode_seed = self.eval_seed + episode
            self.eval_env.seed(episode_seed)
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            episode_catastrophes = 0  # Count catastrophes in the episode
            
            while not done:
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Convert action to scalar if necessary
                action_item = action
                if isinstance(action, np.ndarray):
                    action_item = action.item()

                if exp_type in ["ours", "hirl"]:
                    agent_obs = self.eval_env.envs[0].unwrapped.render()
                    blocker_obs = self.eval_env.envs[0].unwrapped.ale.getRAM()
                    blocker_heuristic_decision = self.model.blocker_heuristic.should_block(blocker_obs, action_item)
                    blocker_model_decision, _, _ = self.model.blocker_model.should_block(
                        agent_obs,
                        action_item,
                        blocker_heuristic_decision
                    )
                    if blocker_heuristic_decision != blocker_model_decision:
                        episode_disagreements += 1
                    if num_timesteps <= blocker_switch_time:
                        if blocker_heuristic_decision:
                            episode_env_interventions += 1
                            episode_exp_interventions += 1   
                            action_item = self.new_action                   
                    else:
                        if blocker_model_decision:
                            episode_env_interventions += 1 
                            action_item = self.new_action
                        if blocker_heuristic_decision:
                            episode_exp_interventions += 1                            

                elif exp_type in ["expert"]:
                    agent_obs = self.eval_env.envs[0].unwrapped.render()
                    blocker_obs = self.eval_env.envs[0].unwrapped.ale.getRAM()
                    blocker_heuristic_decision = self.model.blocker_heuristic.should_block(blocker_obs, action_item)
                    if blocker_heuristic_decision:
                        action_item = self.new_action
                        episode_env_interventions += 1
                        episode_exp_interventions += 1

                # Prepare the action for the environment
                action = np.array([action_item])

                new_obs, reward, done, info = self.eval_env.step(action)
                
                # Capture frames for GIF
                if create_gif:
                    frame = self.create_frame()

                # Check for catastrophes using BlockerHeuristic
                new_blocker_obs = self.eval_env.envs[0].unwrapped.ale.getRAM()

                if self.model.blocker_heuristic.is_catastrophe(new_blocker_obs):
                    episode_catastrophes += 1  # Increment catastrophe count
                    
                    if create_gif:
                        frame = self.draw_catastrophe(frame)

                if create_gif:        
                    frames.append(frame)

                if isinstance(reward, np.ndarray):
                    reward = reward[0]
                episode_reward += reward
                episode_length += 1
                obs = new_obs

            # Log episode statistics
            all_episode_rewards.append(episode_reward)
            all_episode_lengths.append(episode_length)
            all_catastrophes.append(episode_catastrophes)
            all_env_interventions.append(episode_env_interventions)
            all_exp_interventions.append(episode_exp_interventions)
            all_disagreements.append(episode_disagreements)

        # Compute cumulative metrics by summing over all episodes
        total_reward = sum(all_episode_rewards)
        total_ep_length = sum(all_episode_lengths)

        # Update cumulative counts
        self.cum_catastrophe += sum(all_catastrophes)
        self.cum_env_intervention += sum(all_env_interventions)
        self.cum_exp_intervention += sum(all_exp_interventions)
        self.cum_disagreement += sum(all_disagreements)

        if num_timesteps > blocker_switch_time:
            self.blocker_cum_catastrophe += sum(all_catastrophes)
            self.blocker_cum_env_intervention += sum(all_env_interventions)
            self.blocker_cum_exp_intervention += sum(all_exp_interventions)
            self.blocker_cum_disagreement += sum(all_disagreements)

        # Compute mean values
        mean_reward = np.mean(all_episode_rewards)
        std_reward = np.std(all_episode_rewards)
        mean_ep_length = np.mean(all_episode_lengths)
        mean_catastrophes = np.mean(all_catastrophes)
        mean_env_intervention = np.mean(all_env_interventions)
        mean_exp_intervention = np.mean(all_exp_interventions)
        mean_disagreement = np.mean(all_disagreements)

        # Record evaluation metrics (mean, cumulative)
        self.logger.record('eval/mean_reward', mean_reward)
        self.logger.record('eval/std_reward', std_reward)
        self.logger.record('eval/mean_ep_length', mean_ep_length)
        self.logger.record('eval/mean_catastrophes', mean_catastrophes)
        self.logger.record('eval/mean_env_intervention', mean_env_intervention)
        self.logger.record('eval/mean_exp_intervention', mean_exp_intervention)
        self.logger.record('eval/mean_disagreement', mean_disagreement)

        self.logger.record('eval/blocker_cum_catastrophe', self.blocker_cum_catastrophe)
        self.logger.record('eval/blocker_cum_env_intervention', self.blocker_cum_env_intervention)
        self.logger.record('eval/blocker_cum_exp_intervention', self.blocker_cum_exp_intervention)
        self.logger.record('eval/blocker_cum_disagreement', self.blocker_cum_disagreement)
        
        self.logger.record('eval/total_ep_length', total_ep_length)
        self.logger.record('eval/total_reward', total_reward)

        self.logger.record('eval/cum_catastrophe', self.cum_catastrophe)
        self.logger.record('eval/cum_env_intervention', self.cum_env_intervention)
        self.logger.record('eval/cum_exp_intervention', self.cum_exp_intervention)
        self.logger.record('eval/cum_disagreement', self.cum_disagreement)

        if create_gif:
            # Create GIF in-memory and upload to WandB without saving locally
            gif_buffer = io.BytesIO()  # In-memory buffer
            imageio.mimsave(gif_buffer, frames, format='GIF', fps=30)  # Save the GIF to buffer
            gif_buffer.seek(0)  # Reset buffer position to the beginning

            # Log the GIF to WandB
            wandb.log({f"eval_gif/gif": wandb.Video(gif_buffer, fps=30, format="gif")})

            gif_buffer.close()

            self.next_gif_step += self.gif_freq

        if self.verbose > 0:
            print(f"Eval num_timesteps={self.num_timesteps}, "
                  f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}, "
                  f"total_catastrophes={self.cum_catastrophe}, "
                  f"mean_catastrophes={mean_catastrophes:.2f}")

        # Dump the logs
        self.logger.dump(self.num_timesteps)
