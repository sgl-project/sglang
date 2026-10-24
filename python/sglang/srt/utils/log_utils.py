from __future__ import annotations

import glob
import json
import logging
import os
import re
import socket
import sys
import threading
import time
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import List, Optional, Union

import torch.distributed as dist

_LOG_RETENTION_INTERVAL_SECONDS = 60 * 60
_ROTATED_LOG_SUFFIX_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}")
_LOG_RETENTION_SCHEDULES: set[tuple[str, int]] = set()
_LOG_RETENTION_SCHEDULES_LOCK = threading.Lock()


def create_log_targets(
    *,
    targets: Optional[List[str]],
    name_prefix: str,
    retention_days: Optional[int] = None,
) -> List[logging.Logger]:
    if not targets:
        return [_create_log_target_stdout(name_prefix)]
    return [_create_log_target(t, name_prefix, retention_days) for t in targets]


def _create_log_target(
    target: str, name_prefix: str, retention_days: Optional[int]
) -> logging.Logger:
    if target.lower() == "stdout":
        return _create_log_target_stdout(name_prefix)
    return _create_log_target_file(target, name_prefix, retention_days)


def _create_log_target_stdout(name_prefix: str) -> logging.Logger:
    return _create_logger_with_handler(
        f"{name_prefix}.stdout", logging.StreamHandler(sys.stdout)
    )


def _create_log_target_file(
    directory: str, name_prefix: str, retention_days: Optional[int] = None
) -> logging.Logger:
    os.makedirs(directory, exist_ok=True)
    hostname = socket.gethostname()
    rank = dist.get_rank() if dist.is_initialized() else 0
    filename = os.path.join(directory, f"{hostname}_{rank}.log")
    _start_log_retention_cleanup(filename, retention_days)
    handler = TimedRotatingFileHandler(
        filename, when="H", backupCount=0, encoding="utf-8"
    )
    return _create_logger_with_handler(
        f"{name_prefix}.file.{directory}.{hostname}_{rank}", handler
    )


def _get_rotated_log_files(filename: str) -> List[str]:
    """Return this logger's hourly rotations, excluding similarly prefixed files."""
    prefix = f"{filename}."
    return [
        path
        for path in glob.glob(glob.escape(prefix) + "*")
        if _ROTATED_LOG_SUFFIX_RE.fullmatch(path[len(prefix) :])
    ]


def _cleanup_rotated_log_files(
    filename: str, retention_days: int, now: Optional[float] = None
) -> None:
    cutoff = (time.time() if now is None else now) - retention_days * 24 * 60 * 60
    for path in _get_rotated_log_files(filename):
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            logging.getLogger(__name__).warning(
                "Unable to remove expired request log file %s", path, exc_info=True
            )


def _start_log_retention_cleanup(
    filename: str, retention_days: Optional[int]
) -> Optional[threading.Timer]:
    if retention_days is None:
        return None
    if retention_days <= 0:
        raise ValueError("log request retention days must be positive")

    _cleanup_rotated_log_files(filename, retention_days)
    schedule_key = (os.path.abspath(filename), retention_days)
    with _LOG_RETENTION_SCHEDULES_LOCK:
        if schedule_key in _LOG_RETENTION_SCHEDULES:
            return None
        _LOG_RETENTION_SCHEDULES.add(schedule_key)

    def cleanup_and_reschedule() -> None:
        try:
            _cleanup_rotated_log_files(filename, retention_days)
        finally:
            timer = threading.Timer(
                _LOG_RETENTION_INTERVAL_SECONDS, cleanup_and_reschedule
            )
            timer.daemon = True
            timer.start()

    timer = threading.Timer(_LOG_RETENTION_INTERVAL_SECONDS, cleanup_and_reschedule)
    timer.daemon = True
    timer.start()
    return timer


def _create_logger_with_handler(name: str, handler: logging.Handler) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


def log_json(
    loggers: Union[logging.Logger, List[logging.Logger]], event: str, data: dict
) -> None:
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        **data,
    }
    msg = json.dumps(log_data, ensure_ascii=False)

    if not isinstance(loggers, list):
        loggers = [loggers]

    for logger in loggers:
        logger.info(msg)
