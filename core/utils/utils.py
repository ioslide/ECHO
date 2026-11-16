"""Deterministic seeding and append-only experiment summaries."""
import fcntl
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch


def seed_everything(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed


def save_df(new_results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / ".save_df.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        frame = pd.DataFrame([new_results])
        if path.exists():
            frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
        frame.to_csv(path, index=False)


def check_isfile(path):
    return Path(path).is_file()
