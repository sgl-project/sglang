import logging


def configure_logger(log_level: str = "INFO"):
    format = "[%(asctime)s] %(name)s - %(levelname)s: %(message)s"

    logging.basicConfig(
        level=getattr(logging, log_level),
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
