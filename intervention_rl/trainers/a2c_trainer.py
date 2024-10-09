import gymnasium as gym
import torch
from stable_baselines3 import A2C
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from omegaconf import DictConfig
import os

class A2CTrainer:
    def __init__(self, cfg: DictConfig, exp_dir: str):
        # Extracting parameters from Hydra config
        self.env_name = cfg.env.name
        self.total_timesteps = cfg.algo.a2c.total_timesteps
        self.save_path = os.path.join(exp_dir)  # Save model to exp_dir
        self.n_steps = cfg.algo.a2c.n_steps
        self.device = torch.device(cfg.device)

        # Logging and saving frequencies
        self.log_freq = cfg.algo.a2c.log_freq
        self.eval_freq = cfg.algo.a2c.eval_freq
        self.save_freq = cfg.algo.a2c.save_freq
        self.oversight_timesteps = cfg.algo.a2c.oversight_timesteps
        self.blocker_train_freq = cfg.algo.a2c.blocker_train_freq

        # Create the environment and wrap it, with seeding
        def make_env(env_name, seed):
            env = gym.make(env_name)
            if seed is not None:
                env.unwrapped.seed(seed)  # Seed the environment
            return env

        self.env = DummyVecEnv([lambda: make_env(self.env_name, cfg.seed)])

        # Create A2C model using the policy and additional kwargs from the config
        self.model = A2C(
            cfg.algo.a2c.policy,  # Use policy from the config (CnnPolicy, MlpPolicy, etc.)
            self.env,
            learning_rate=cfg.algo.a2c.learning_rate,
            n_steps=cfg.algo.a2c.n_steps,
            gamma=cfg.algo.a2c.gamma,
            gae_lambda=cfg.algo.a2c.gae_lambda,
            ent_coef=cfg.algo.a2c.ent_coef,
            vf_coef=cfg.algo.a2c.vf_coef,
            max_grad_norm=cfg.algo.a2c.max_grad_norm,
            rms_prop_eps=cfg.algo.a2c.rms_prop_eps,
            use_rms_prop=cfg.algo.a2c.use_rms_prop,
            use_sde=cfg.algo.a2c.use_sde,
            sde_sample_freq=cfg.algo.a2c.sde_sample_freq,
            normalize_advantage=cfg.algo.a2c.normalize_advantage,
            verbose=cfg.algo.a2c.verbose,
            seed=cfg.seed,
            device=cfg.device
        )
        
        # Initialize the rollout buffer (which stores the experiences)
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.n_steps * self.env.num_envs,  # Adjust for number of envs
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.device,
            gamma=cfg.algo.a2c.gamma,
            gae_lambda=cfg.algo.a2c.gae_lambda,
            n_envs=self.env.num_envs
        )
        
        # Variable to track timesteps
        self.num_timesteps = 0

    def collect_experiences(self):
        # Reset the environment and collect data for n_steps
        obs = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)  # Convert to torch Tensor
        episode_reward = 0  # Track episode reward

        for step in range(self.n_steps):
            # Convert the observation to a tensor before passing it to the model
            action, values, log_probs = self.model.policy.forward(obs)
            
            # Step through the environment
            new_obs, rewards, dones, infos = self.env.step(action)
            
            # Convert the new observation as well
            new_obs = torch.tensor(new_obs, dtype=torch.float32).to(self.device)
            
            episode_reward += rewards.sum()

            # Add experience to the rollout buffer
            self.rollout_buffer.add(obs, action, rewards, dones, values, log_probs)
            
            # Move to the next observation
            obs = new_obs
            self.num_timesteps += 1

            if any(dones):  # Handle multiple environments done condition
                print(f"Episode done at step {self.num_timesteps}, Reward: {episode_reward}")
                break

    def train_policy(self):
        # Compute returns and advantages based on the stored rollouts
        self.rollout_buffer.compute_returns_and_advantage(last_values=torch.tensor([0.0]))

        # Update the policy using the data in the buffer
        self.model.train()

    def log_progress(self):
        """Logs training progress every log_freq timesteps."""
        print(f"Timestep {self.num_timesteps}/{self.total_timesteps}")

    def evaluate(self):
        """Placeholder for evaluation logic every eval_freq timesteps."""
        print(f"Evaluating at timestep {self.num_timesteps}")
        # You can add evaluation logic here using a separate evaluation environment

    def train_blocker(self):
        """Placeholder for training blocker logic."""
        print(f"Training blocker at timestep {self.num_timesteps}")
        # Add blocker training logic here

    def train(self):
        """
        This is the main training loop. It will run until the total number of timesteps is reached.
        """
        while self.num_timesteps < self.total_timesteps:
            # Collect experiences from the environment
            self.collect_experiences()
            
            # Train the policy using the collected experiences
            self.train_policy()

            # Log training progress every log_freq timesteps
            if self.num_timesteps % self.log_freq == 0:
                self.log_progress()

            # Evaluate the model every eval_freq timesteps
            if self.num_timesteps % self.eval_freq == 0:
                self.evaluate()

            # Train blocker every blocker_train_freq timesteps
            if self.num_timesteps <= self.oversight_timesteps and self.num_timesteps % self.blocker_train_freq == 0:
                self.train_blocker()

            # Save model periodically
            if self.num_timesteps % self.save_freq == 0:
                self.model.save(self.save_path)
                print(f"Model saved at timestep {self.num_timesteps}")
        
        # Final save after training is complete
        self.model.save(self.save_path)
        print(f"Final model saved after {self.num_timesteps} timesteps")

    def save_model(self):
        self.model.save(self.save_path)

    def load_model(self, path):
        self.model = A2C.load(path, env=self.env)
        print(f"Model loaded from {path}")

    def run_trained_model(self, episodes=5):
        for episode in range(episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
            print(f"Episode {episode + 1}: Reward: {episode_reward}")