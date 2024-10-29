#!/bin/bash
#SBATCH --time=10:00:00            # Maximum runtime
#SBATCH --ntasks=1                # Number of tasks (processes)
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
#SBATCH --partition=gpu           # Partition (queue) to submit to
#SBATCH --gpus-per-task=p100:1    # Request 1 V100 GPU
#SBATCH --mem=30G                 # Total memory
#SBATCH --account=biyik_1165      # Account name
#SBATCH --output=/home1/jpdoshi/intervention-rl/intervention_rl/slurm_outputs/%x-%j.out  # Output file

# Load conda
source /spack/conda/miniforge3/24.3.0/etc/profile.d/conda.sh

# Activate the correct environment
conda activate intervention_rl

# Set PYTHONPATH to include the project root
export PYTHONPATH=$PYTHONPATH:/home1/jpdoshi/intervention-rl

# Run using the module system
python -m scripts.train env="pong" exp_type="expert_hirl" env.total_timesteps=8000000 seed=43