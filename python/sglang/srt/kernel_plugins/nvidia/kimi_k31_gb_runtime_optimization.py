# SPDX-License-Identifier: Apache-2.0
"""The process env vars SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION sets."""

from __future__ import annotations

import logging
import os

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

GB_PROCESS_ENVS: dict[str, str] = {
    "CUDA_MODULE_LOADING": "EAGER",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
_ALIASES = {"PYTORCH_CUDA_ALLOC_CONF": "PYTORCH_ALLOC_CONF"}
_applied = False


def apply_k31_gb_runtime_optimization() -> None:
    global _applied
    if _applied or not envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION.get():
        return
    _applied = True
    applied, kept = [], []
    for key, value in GB_PROCESS_ENVS.items():
        if key in os.environ or _ALIASES.get(key, key) in os.environ:
            kept.append(key)
        else:
            os.environ[key] = value
            applied.append(f"{key}={value}")
    logger.info(
        "SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION: applied %s; kept user-set %s",
        ", ".join(applied) or "none",
        ", ".join(kept) or "none",
    )
