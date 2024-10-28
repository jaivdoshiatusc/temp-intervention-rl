# Project Name: Intervention-RL

## Description
Intervention-RL is a reinforcement learning (RL) framework designed for developing and evaluating interventions in complex environments. It provides a set of tools and algorithms to train RL agents and study their behavior under different intervention strategies.

## Installation
1. Clone the repository: `git clone https://github.com/username/intervention-rl.git`
2. Install the Package: `pip install -e .`
3. Install the required dependencies: `conda env create -f environment.yml`

## Usage
1. Navigate to the project directory: `cd intervention-rl`
2. Run the main script: `python -m scripts.train algo.a2c.exp_type="none"`
3. Choose between the following intervention strategies:
    - `"none"`: No blocker
    - `"expert_hirl"`: Algorithmic blocker, uses robot actions in buffer
    - `"expert_hirl"`: Algorithmic blocker, uses human actions in buffer
    - `"hirl"`: Uses trained blocker, without reward bonus
    - `"ours"`: Uses trained blocker, with reward bonus
4. Customize configuration variables:
    ```bash
    python -m scripts.train algo.a2c.exp_type="ours" seed=42 algo.a2c.learning_rate=0.001 env.catastrophe_clearance=8 env.blocker_clearance=8
    ```
