# SPDX-License-Identifier: Apache-2.0
"""Utilities for diffusion ``torch.compile`` configuration."""

from __future__ import annotations

import os
from typing import Any

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

DEFAULT_TORCH_COMPILE_MODE = "max-autotune-no-cudagraphs"
BLACKWELL_TORCH_COMPILE_MODE = "default"
TORCH_COMPILE_MODE_ENV = "SGLANG_TORCH_COMPILE_MODE"


def resolve_torch_compile_mode(
    model_config: Any | None,
    default: str = DEFAULT_TORCH_COMPILE_MODE,
) -> str:
    """Resolve DiT ``torch.compile`` mode with a Blackwell-safe default.

    PyTorch Inductor's max-autotune path can try Triton GEMM candidates that
    exceed SM100 resource limits and then fall back to ATen/cuBLAS anyway. Keep
    explicit user and model choices intact, but avoid that default autotune path
    on Blackwell when no override is provided.
    """
    env_mode = os.environ.get(TORCH_COMPILE_MODE_ENV)
    if env_mode:
        return env_mode

    mode = getattr(model_config, "torch_compile_mode", default)
    if mode == DEFAULT_TORCH_COMPILE_MODE and current_platform.is_blackwell():
        logger.info(
            "Using torch.compile mode %s on Blackwell instead of %s. "
            "Set %s to override.",
            BLACKWELL_TORCH_COMPILE_MODE,
            DEFAULT_TORCH_COMPILE_MODE,
            TORCH_COMPILE_MODE_ENV,
        )
        return BLACKWELL_TORCH_COMPILE_MODE

    return mode
