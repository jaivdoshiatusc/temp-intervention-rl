from __future__ import print_function

import os
import sys
import torch
import torch.multiprocessing as mp
import hydra
from omegaconf import DictConfig

from envs.pong import create_atari_env
from intervention_rl.utils import my_optim
from intervention_rl.models.agent_model import ActorCritic
from intervention_rl.models.blocker_model import CNNClassifier
from intervention_rl.rollout.test import test
from intervention_rl.trainers.trainer import train
from intervention_rl.trainers.blocker_trainer import blocker_train

@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    mp.set_start_method('spawn')  # Set start method to 'spawn'

    torch.manual_seed(cfg.seed)

    # Create shared policy model
    env = create_atari_env(cfg.env.name)
    shared_model = ActorCritic(env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()

    # Create shared CNN classifier model
    shared_blocker_model = CNNClassifier(env.action_space.n)
    shared_blocker_model.share_memory()

    # Create shared optimizer
    if cfg.no_shared:
        optimizer = None
    else:
        # Optimizer for the policy model
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=cfg.algorithm.lr)
        optimizer.share_memory()

        # Optimizer for the CNN classifier
        blocker_model_optimizer = my_optim.SharedAdam(shared_blocker_model.parameters(), lr=cfg.algorithm.blocker_model_lr)
        blocker_model_optimizer.share_memory()

    use_wandb = True if cfg.wandb else False

    processes = []

    # Shared counter and lock
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    # Shared lists for aggregated observations and labels
    manager = mp.Manager()
    all_observations = manager.list()
    all_labels = manager.list()

    # Start a separate process for testing
    p = mp.Process(target=test, args=(cfg.num_processes, cfg, shared_model, shared_blocker_model, counter, lock, use_wandb, cfg.experiment_name))
    p.start()
    processes.append(p)

    # Start the CNN training process
    p = mp.Process(target=blocker_train, args=(cfg, shared_blocker_model, all_observations, all_labels, counter, lock, use_wandb, cfg.experiment_name))
    p.start()
    processes.append(p)

    # Start the training processes
    for rank in range(0, cfg.num_processes):
        p = mp.Process(target=train, args=(rank, cfg, shared_model, shared_blocker_model, counter, lock, optimizer, blocker_model_optimizer, all_observations, all_labels))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
