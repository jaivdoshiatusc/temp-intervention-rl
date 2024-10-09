import os
import numpy as np
import torch
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf  # Import OmegaConf for config conversion
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import wandb

# Custom Observation Wrapper to permute the observation space to (channels, height, width)
class ChannelFirstWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ChannelFirstWrapper, self).__init__(env)
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.transpose(2, 0, 1),
            high=env.observation_space.high.transpose(2, 0, 1),
            shape=(obs_shape[2], obs_shape[0], obs_shape[1]),
            dtype=env.observation_space.dtype
        )

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))

def make_env(env_name, seed):
    env = gym.make(env_name)
    env = ChannelFirstWrapper(env)  # Apply the custom wrapper
    if seed is not None:
        env.seed(seed)
    return env

class A2CTrainer:
    def __init__(self, cfg: DictConfig, exp_dir: str):
        # Extract parameters from config
        self.cfg = cfg
        self.env_name = cfg.env.name
        self.total_timesteps = cfg.algo.a2c.total_timesteps
        self.save_path = os.path.join(exp_dir)  # Save model to exp_dir
        self.device = torch.device(cfg.device)
        self.log_freq = cfg.algo.a2c.log_freq
        self.eval_freq = cfg.algo.a2c.eval_freq

        sanitized_config = OmegaConf.to_container(self.cfg, resolve=True)

        wandb.init(
            project="modified_hirl",
            name="experiment_name",
            config=sanitized_config, 
            sync_tensorboard=True  
        )

        # Create the environment and wrap it, with seeding
        self.env = DummyVecEnv([lambda: make_env(self.env_name, cfg.seed)])

        # Create A2C model
        self.model = A2C(
            cfg.algo.a2c.policy,
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
            verbose=1,
            seed=cfg.seed,
            device=cfg.device,
            tensorboard_log=f"runs/{wandb.run.id}"  
        )

    def train(self):
        eval_env = DummyVecEnv([lambda: make_env(self.env_name, self.cfg.seed + 100)])

        num_eval_episodes = self.cfg.eval.num_rollouts

        eval_callback = EvalCallback(
            eval_env=eval_env,
            eval_freq=self.cfg.algo.a2c.eval_freq,
            n_eval_episodes=num_eval_episodes,
            best_model_save_path=None,  
            verbose=1  
        )

        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=eval_callback,
            log_interval=self.log_freq,
        )

        self.model.save(self.save_path)
        print(f"Final model saved after {self.total_timesteps} timesteps")

    def save_model(self):
        self.model.save(self.save_path)

    def load_model(self, path):
        self.model = A2C.load(path, env=self.env)
        print(f"Model loaded from {path}")

    def run_trained_model(self, episodes=5):
        for episode in range(episodes):
            obs = self.env.reset()
            done = [False]
            episode_reward = 0
            while not done[0]:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
            print(f"Episode {episode + 1}: Reward: {episode_reward}")
