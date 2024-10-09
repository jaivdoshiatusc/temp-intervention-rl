from pathlib import Path
import os
import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from intervention_rl.trainers.a3c_trainer import A3CTrainer
from intervention_rl.trainers.a2c_trainer import A2CTrainer

# from intervention_rl.utils.log_utils import log

@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    hydra_cfg = HydraConfig.get()
    launcher = hydra_cfg.runtime.get("choices", {}).get("hydra/launcher", None)
    sweep = launcher in ["slurm"]

    if sweep:
        exp_dir = Path(hydra_cfg.sweep.dir) / hydra_cfg.sweep.subdir
    else:
        exp_dir = Path(hydra_cfg.run.dir)

    print(f"Experiment directory: {exp_dir}")
    print(f"Starting training, cwd: {Path.cwd()}")

    # Set environment variables
    if cfg.algo.name != "a3c":
        os.environ['OMP_NUM_THREADS'] = '1'

    if cfg.device == "cpu":
        os.environ['CUDA_VISIBLE_DEVICES'] = ""

    # Set torch seed
    torch.manual_seed(cfg.seed)

    # Initialize trainer based on algorithm specified in cfg
    if cfg.algo.name == "a3c":
        trainer = A3CTrainer(cfg, exp_dir)
    elif cfg.algo.name == "a2c":
        trainer = A2CTrainer(cfg, exp_dir)
    else:
        raise NotImplementedError(f"Trainer for {cfg.algo.name} not implemented")

    trainer.train()

if __name__ == '__main__':
    main()