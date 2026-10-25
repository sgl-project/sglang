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

import logging
import os

from sglang.srt.utils.common import get_bool_env_var, is_hip

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
