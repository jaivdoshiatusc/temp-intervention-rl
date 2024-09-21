from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp

import my_optim
from envs import create_atari_env
from model import ActorCritic
from test import test
from train import train
from cnn_trainer import cnn_train
from blocker_model import CNNClassifier

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--cnn-lr', type=float, default=0.001, 
                    help='learning rate for CNN (default: 0.001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae-lambda', type=float, default=1.00,
                    help='lambda parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    help='environment to train on (default: PongDeterministic-v4)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--max-steps', type=int, default=3000000, 
                    help='maximum number of steps to train (default: 4,500,000)')
parser.add_argument('--max-steps-for-cnn', type=int, default=3000000, 
                    help='maximum number of steps to train (default: 1,000,000)')
parser.add_argument('--alpha', type=float, default=0.01,
                    help='scaling factor for uncertainty')
parser.add_argument('--beta', type=float, default=0.01, 
                    help='bonus for disagreement between CNN and algorithm')
parser.add_argument('--experiment_name', default='experiment', type=str,
                    help='name of the experiment')
parser.add_argument('--wandb', action="store_true",
                    help='use to log using wandb.')

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    mp.set_start_method('spawn')  # Set start method to 'spawn'

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    # Create shared policy model
    env = create_atari_env(args.env_name)
    shared_model = ActorCritic(env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()

    # Create shared CNN classifier model
    shared_cnn_model = CNNClassifier(env.action_space.n)
    shared_cnn_model.share_memory()

    # Create shared optimizer
    if args.no_shared:
        optimizer = None
    else:
        # Optimizer for the policy model
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

        # Optimizer for the CNN classifier
        cnn_optimizer = my_optim.SharedAdam(shared_cnn_model.parameters(), lr=args.cnn_lr)
        cnn_optimizer.share_memory()

    iswandb = True if args.wandb else False

    processes = []

    # Shared counter and lock
    counter = mp.Value('i', 0)
    cumulative_catastrophes = mp.Value('i', 0)
    lock = mp.Lock()

    # Shared lists for aggregated observations and labels
    manager = mp.Manager()
    all_observations = manager.list()
    all_labels = manager.list()

    # Start a separate process for testing
    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, shared_cnn_model, counter, lock, iswandb, args.experiment_name))
    p.start()
    processes.append(p)

    # Start the CNN training process
    p = mp.Process(target=cnn_train, args=(args, shared_cnn_model, all_observations, all_labels, counter, lock, iswandb, args.experiment_name))
    p.start()
    processes.append(p)

    # Start the training processes
    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, shared_cnn_model, counter, lock, optimizer, cnn_optimizer, all_observations, all_labels))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()