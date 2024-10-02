from setuptools import find_packages, setup

setup(
    name="intervention_rl",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "hydra-core",
        "wandb",
        "numpy",
        "imageio",
        "ruff",
    ],
)