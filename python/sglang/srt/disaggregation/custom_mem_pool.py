# Copyright 2026 SGLang Team
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
"""Resolve transport-specific memory pools for PD disaggregation."""

from typing import Any, Callable, Dict, Optional, Tuple

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_disagg

PoolResult = Tuple[bool, Optional[Any], Optional[str]]


def _mori_pool(device: str) -> PoolResult:
    from sglang.srt.disaggregation.mori.custom_mem_pool import (
        init_mori_custom_mem_pool,
    )

    return init_mori_custom_mem_pool(device)


def _mooncake_pool(device: str) -> PoolResult:
    from sglang.srt.disaggregation.mooncake.utils import init_mooncake_custom_mem_pool

    return init_mooncake_custom_mem_pool(device)


# PD transfer backends that choose the KV pool allocator themselves.
_PD_BACKEND_POOLS: Dict[str, Callable[[str], PoolResult]] = {"mori": _mori_pool}


def maybe_init_custom_mem_pool(device: str) -> PoolResult:
    """Initialize the custom memory pool required by the active transport."""
    try:
        config = get_disagg()
    except ValueError:
        config = None
    if config is not None and config.disaggregation_mode in ("prefill", "decode"):
        resolve = _PD_BACKEND_POOLS.get(config.disaggregation_transfer_backend)
        if resolve is not None:
            return resolve(device)

    # Mooncake's pool stays env-driven, independent of the PD backend.
    if envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get() is not None:
        return _mooncake_pool(device)

    return False, None, None
