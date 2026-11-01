"""Add ``warning_once`` and ``info_once`` to ``logging.Logger``.

SGLang code calls ``logger.warning_once`` / ``logger.info_once``, which are not
part of ``logging``: transformers adds them to ``logging.Logger`` when it is
imported. Installing them at startup keeps those calls working in code paths
that never import transformers.
"""

import functools
import logging


@functools.lru_cache(None)
def _warning_once(self, *args, **kwargs):
    self.warning(*args, **kwargs)


@functools.lru_cache(None)
def _info_once(self, *args, **kwargs):
    self.info(*args, **kwargs)


def install_logging_once() -> None:
    # Keep transformers' versions if it was imported first.
    if not hasattr(logging.Logger, "warning_once"):
        logging.Logger.warning_once = _warning_once
    if not hasattr(logging.Logger, "info_once"):
        logging.Logger.info_once = _info_once
