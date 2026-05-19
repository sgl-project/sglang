from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

import torch

logger = logging.getLogger(__name__)


def decoupled_spec_nvtx_enabled() -> bool:
    value = os.environ.get("SGLANG_DECOUPLED_SPEC_NVTX", "1").lower()
    return value not in ("0", "false", "no", "off")


@contextmanager
def decoupled_spec_nvtx_range(
    domain: str,
    name: str,
) -> Iterator[None]:
    if not decoupled_spec_nvtx_enabled() or not torch.cuda.is_available():
        yield
        return

    pushed = False
    try:
        torch.cuda.nvtx.range_push(f"{domain}::{name}")
        pushed = True
    except Exception:
        logger.debug("Failed to push decoupled spec NVTX range", exc_info=True)

    try:
        yield
    finally:
        if pushed:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                logger.debug("Failed to pop decoupled spec NVTX range", exc_info=True)
