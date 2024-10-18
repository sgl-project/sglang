import logging
from dataclasses import dataclass


def configure_logger(log_level, prefix: str = ""):
    # add level to the format
    format = f"[%(asctime)s{prefix}] %(levelname)s: %(message)s"

    logging.basicConfig(
        level=log_level,
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


@dataclass
class WorkerInfo:
    server_url: str
