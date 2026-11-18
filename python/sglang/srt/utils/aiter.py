# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""AITER utility helpers."""

import functools
import logging
import os
from typing import Optional

from sglang.srt.environ import envs
from sglang.srt.utils.common import get_bool_env_var
from sglang.srt.utils.device import is_hip

logger = logging.getLogger(__name__)

_aiter_chip_info_cached = False


def maybe_pre_warm_aiter_chip_info() -> None:
    """Pre-populate AITER chip info env vars before CUDA graph capture.

    AITER chip info probes can shell out to rocminfo. During graph capture the
    GPU context is locked, so pre-caching CU_NUM/GPU_ARCHS avoids a hang.
    """
    if not (get_bool_env_var("SGLANG_USE_AITER") and is_hip()):
        return

    global _aiter_chip_info_cached
    if _aiter_chip_info_cached:
        return
    _aiter_chip_info_cached = True

    try:
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx
    except ImportError:
        logger.debug("Skip aiter chip info pre-warm because aiter is unavailable")
        return

    try:
        if not os.environ.get("CU_NUM"):
            cu_num = get_cu_num()
            os.environ["CU_NUM"] = str(cu_num)
            logger.info("Pre-warmed aiter CU_NUM=%s", cu_num)

        if not os.environ.get("GPU_ARCHS"):
            gfx = get_gfx()
            os.environ["GPU_ARCHS"] = gfx
            logger.info("Pre-warmed aiter GPU_ARCHS=%s", gfx)
    except Exception as e:
        logger.warning("Failed to pre-warm aiter chip info: %s", e)


@functools.lru_cache(maxsize=None)
def aiter_unified_attention_supports_non_causal() -> bool:
    """Whether the installed AITER ``unified_attention`` accepts ``causal=False``."""
    try:
        from aiter.ops.triton._triton_kernels.attention.unified_attention import (
            kernel_unified_attention_2d,
            kernel_unified_attention_3d,
        )

        # AITER added non-causal attention together with this kernel constexpr.
        return all(
            "CAUSAL" in kernel.arg_names
            for kernel in (kernel_unified_attention_2d, kernel_unified_attention_3d)
        )
    except (ImportError, AttributeError) as e:
        logger.debug("AITER unified_attention non-causal probe failed: %s", e)
        return False


def aiter_dspark_draft_reject_reason(
    *,
    draft_is_mla: bool,
    draft_has_sliding_window: bool,
    enable_unified_memory: bool,
    enable_deterministic_inference: bool,
) -> Optional[str]:
    """Why the aiter backend cannot run a DSpark draft, or None when it can.

    The aiter DSpark draft verifies its block bidirectionally on unified_attention
    (causal=False) over a 4D NHD pool. The draft-backend resolver, the Kimi-K3 hook
    and AiterAttnBackend all ask this one predicate.
    """
    if draft_is_mla:
        return "MLA draft (the aiter MLA verify is causal-only)"
    if draft_has_sliding_window:
        return (
            "sliding-window draft (unified_attention takes no window with causal=False)"
        )
    if not envs.SGLANG_AITER_UNIFIED_VERIFY.get():
        return "SGLANG_AITER_UNIFIED_VERIFY=0"
    if envs.SGLANG_AITER_KV_CACHE_LAYOUT.get().lower() != "nhd":
        return "SGLANG_AITER_KV_CACHE_LAYOUT is not nhd"
    if enable_unified_memory:
        return "unified memory"
    if enable_deterministic_inference:
        return "deterministic inference (unified_attention is not batch-invariant)"
    if not aiter_unified_attention_supports_non_causal():
        return (
            "the installed AITER unified_attention has no causal=False "
            "(ROCm/aiter#7244)"
        )
    return None
