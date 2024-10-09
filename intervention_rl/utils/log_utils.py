"""
This file contains utility classes and functions for logging to stdout, stderr,
and to tensorboard.
"""

import sys

from loguru import logger
from tqdm import tqdm

logger_format = (
    "<level>{level: <2}</level> <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
    "- <level>{message}</level>"
)
logger.remove()
logger.add(sys.stderr, format=logger_format)


class custom_tqdm(tqdm):
    """
    Small extension to tqdm to make a few changes from default behavior.
    By default tqdm writes to stderr. Instead, we change it to write
    to stdout.
    """

    def __init__(self, *args, **kwargs):
        assert "file" not in kwargs
        super(custom_tqdm, self).__init__(*args, file=sys.stdout, **kwargs)


def log(message: str, color: str = "", ansi: str = ""):
    if ansi:
        logger.opt(colors=True).info(f"{ansi}{message}")
    elif color:
        logger.opt(colors=True).info(f"<{color}>{message}</{color}>")
    else:
        logger.info(message)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def add_prefix(dict, prefix):
    return {f"{prefix}/{key}": value for key, value in dict.items()}


def round_values(dict, n):
    return {key: round(value, n) for key, value in dict.items()}