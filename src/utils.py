import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data import load_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(log_dir: str | Path) -> logging.Logger:
    path = Path(log_dir)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("fake_news_detector")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(path / "training.log", encoding="utf-8")
        console_handler = logging.StreamHandler()
        for handler in (file_handler, console_handler):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger


__all__ = ["get_logger", "load_config", "set_seed"]
