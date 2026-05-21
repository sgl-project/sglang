"""Lightweight VLM profiling — gated by VLM_PROFILE=1 env var."""

import logging
import os

ENABLED = os.environ.get("VLM_PROFILE", "") == "1"

logger = logging.getLogger(__name__)


def log_stage(stage: str, **kwargs):
    if not ENABLED:
        return
    parts = [f"{stage}"]
    for k, v in kwargs.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.3f}s" if v > 0.1 else f"{k}={v * 1000:.2f}ms")
        else:
            parts.append(f"{k}={v}")
    logger.info("[VLM] " + " | ".join(parts))
