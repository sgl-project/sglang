# SPDX-License-Identifier: Apache-2.0
"""SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION for the process environment that
CUDA and PyTorch read (SGLang's own children are environ.py defaults): set during
server argument resolution, so every spawned process inherits it."""

from __future__ import annotations

import logging
import os

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

GB_PROCESS_ENVS: dict[str, str] = {
    "CUDA_MODULE_LOADING": "EAGER",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
# A key also counts as set by the user when its alias is set.
_ALIASES = {"PYTORCH_CUDA_ALLOC_CONF": "PYTORCH_ALLOC_CONF"}
_applied = False


def _user_set(key: str) -> bool:
    alias = _ALIASES.get(key)
    return key in os.environ or (alias is not None and alias in os.environ)


def apply_k31_gb_runtime_optimization() -> None:
    """Under the switch, set every unset GB_PROCESS_ENVS key; set keys win, 0 included."""
    global _applied
    if _applied or not envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION.get():
        return
    _applied = True
    applied, kept = [], []
    for key, value in GB_PROCESS_ENVS.items():
        if _user_set(key):
            kept.append(key)
        else:
            os.environ[key] = value
            applied.append(f"{key}={value}")
    logger.info(
        "SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION: applied %s; kept user-set %s",
        ", ".join(applied) or "none",
        ", ".join(kept) or "none",
    )
