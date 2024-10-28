import os
import torch
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
from intervention_rl.utils.my_a2c import A2C_HIRL
# from intervention_rl.utils.default_a2c import A2C_HIRL
from intervention_rl.utils.callback_eval_pong import PongEvalCallback
from stable_baselines3.common.callbacks import EvalCallback  # Imported EvalCallback
from intervention_rl.utils.callback_blocker import BlockerTrainingCallback
from intervention_rl.utils.callback_checkpoints import CustomCheckpointCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
import wandb

class A2CTrainer:
    def __init__(self, cfg: DictConfig, exp_dir: str):
        self.cfg = cfg
        self.exp_dir = exp_dir
        self.agent_save_path = os.path.join(exp_dir, "agent")
        self.blocker_save_path = os.path.join(exp_dir, "blocker")
        self.device = torch.device(cfg.device)

        sanitized_config = OmegaConf.to_container(self.cfg, resolve=True)

        if cfg.wandb.use:
            wandb.init(
                project=cfg.wandb.project,
                name=cfg.hp_name,
                config=sanitized_config,
                sync_tensorboard=True
            )

        # Create necessary directories for saving models
        os.makedirs(self.agent_save_path, exist_ok=True)
        os.makedirs(self.blocker_save_path, exist_ok=True)

        # Setup environment
        self.env = make_atari_env(cfg.env.name, n_envs=cfg.env.n_envs, seed=cfg.seed)
        self.env = VecFrameStack(self.env, n_stack=cfg.env.n_stack)
        self.env = VecTransposeImage(self.env)

        self.eval_env = make_atari_env(cfg.env.name, n_envs=1, seed=cfg.seed + 100)
        self.eval_env = VecFrameStack(self.eval_env, n_stack=cfg.env.n_stack)
        self.eval_env = VecTransposeImage(self.eval_env)

        # Initialize the model (A2C_HIRL)
        self.model = A2C_HIRL(
            policy=cfg.algo.a2c.policy,
            env=self.env,
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
            device=cfg.device,
            tensorboard_log=os.path.join(exp_dir, "tensorboard"),

            exp_type=cfg.algo.a2c.exp_type,
            pretrained_blocker=cfg.algo.a2c.pretrained_blocker,
            blocker_switch_time=cfg.algo.a2c.blocker_switch_time,
            alpha=cfg.algo.a2c.alpha,
            beta=cfg.algo.a2c.beta,

            catastrophe_clearance=cfg.env.catastrophe_clearance,
            blocker_clearance=cfg.env.blocker_clearance,
        )

    def train(self):
        # Custom evaluation callback
        eval_callback = PongEvalCallback(
            cfg = self.cfg,
            eval_env=self.eval_env,
            eval_freq=self.cfg.algo.a2c.eval_freq,
            eval_seed =self.cfg.algo.a2c.eval_seed,
            gif_freq=self.cfg.algo.a2c.gif_freq,
            n_eval_episodes=self.cfg.algo.a2c.eval_episodes,
            verbose=self.cfg.eval.verbose,
        )

        # eval_callback = EvalCallback(
        #     eval_env=self.eval_env,
        #     best_model_save_path=self.agent_save_path,
        #     log_path=self.agent_save_path,
        #     eval_freq=self.cfg.algo.a2c.eval_freq,
        #     n_eval_episodes=self.cfg.algo.a2c.eval_episodes,
        #     verbose=self.cfg.eval.verbose
        # )

        callback_list = [eval_callback]

        # # Blocker training callback with saving functionality
        # if self.cfg.algo.a2c.use_blocker and self.cfg.algo.a2c.train_blocker:
        #     blocker_callback = BlockerTrainingCallback(
        #         train_freq=self.cfg.algo.a2c.blocker_train_freq,   # Frequency of blocker training
        #         epochs=self.cfg.algo.a2c.blocker_epochs,           # Number of epochs to train the blocker
        #         save_freq=self.cfg.algo.a2c.blocker_save_freq,     # Frequency of saving blocker model weights
        #         save_path=self.blocker_save_path,                  # Directory to save blocker model weights
        #         name_prefix="blocker_model",                       # Prefix for saved blocker model files
        #         verbose=self.cfg.algo.a2c.verbose
        #     )

        #     callback_list.append(blocker_callback)

        # Checkpoint callback to save agent weights
        # checkpoint_callback = CheckpointCallback(
        #     save_freq=self.cfg.algo.a2c.save_freq,             # Save frequency from config
        #     save_path=self.agent_save_path,                    # Directory to save the models
        #     name_prefix="a2c_hirl_model"                       # Prefix for the saved model files
        # )
        # callback_list.append(checkpoint_callback)

        checkpoint_callback = CustomCheckpointCallback(
            save_freq=self.cfg.algo.a2c.save_freq,             # Save frequency from config
            save_path=self.agent_save_path,                    # Directory to save the models
            name_prefix="a2c_hirl_model",                      # Prefix for the saved model files
        )
        callback_list.append(checkpoint_callback)

        # Combine callbacks into a CallbackList
        callback_list = CallbackList(callback_list)

        # Start training
        self.model.learn(
            total_timesteps=self.cfg.algo.a2c.total_timesteps,
            callback=callback_list,
            log_interval=self.cfg.algo.a2c.log_freq,
        )