from __future__ import annotations

import json
import logging
import os
import socket
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.distributed as dist


def create_log_target(target: str) -> logging.Logger:
    if target.lower() == "stdout":
        return create_log_target_stdout()
    return create_log_target_file(target)


def create_log_target_stdout() -> logging.Logger:
    return _create_logger_with_handler(f"{__name__}.stdout", logging.StreamHandler())


def create_log_target_file(directory: str) -> logging.Logger:
    import torch.distributed as dist

    os.makedirs(directory, exist_ok=True)
    hostname = socket.gethostname()
    rank = dist.get_rank() if dist.is_initialized() else 0
    filename = os.path.join(directory, f"{hostname}_{rank}.log")
    handler = TimedRotatingFileHandler(
        filename, when="H", backupCount=0, encoding="utf-8"
    )
    return _create_logger_with_handler(
        f"{__name__}.file.{directory}.{hostname}_{rank}", handler
    )


def _create_logger_with_handler(name: str, handler: logging.Handler) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


def log_json(logger: logging.Logger, event: str, data: dict) -> None:
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        **data,
    }
    logger.info(json.dumps(log_data, ensure_ascii=False))
