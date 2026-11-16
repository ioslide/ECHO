import sys
from pathlib import Path
from loguru import logger as log


def set_logger(log_path, log_name):
    Path(log_path).mkdir(parents=True, exist_ok=True)
    log.add(str(Path(log_path) / (log_name + "_{time:YYYY-MM-DD_HH-mm-ss}.log")), level="INFO")
