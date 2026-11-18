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
"""MoRI-specific utilities for custom memory pool management."""

import logging
from typing import Any, Optional, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

SUPPORTED_MORI_CUSTOM_MEM_POOL_TYPES = ["AMD_GEM"]

_logged_pool_types = set()


def check_mori_custom_mem_pool_enabled() -> Tuple[bool, Optional[str]]:
    """Return (enabled, pool type) from SGLANG_MORI_CUSTOM_MEM_POOL without importing mori."""
    custom_mem_pool_type = envs.SGLANG_MORI_CUSTOM_MEM_POOL.get()
    if custom_mem_pool_type is None:
        return False, None
    custom_mem_pool_type = custom_mem_pool_type.upper()
    if custom_mem_pool_type not in SUPPORTED_MORI_CUSTOM_MEM_POOL_TYPES:
        raise ValueError(
            f"Unsupported SGLANG_MORI_CUSTOM_MEM_POOL={custom_mem_pool_type!r}; "
            f"supported: {SUPPORTED_MORI_CUSTOM_MEM_POOL_TYPES}"
        )
    return True, custom_mem_pool_type


def init_mori_custom_mem_pool(
    device: str,
) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Initialize the custom memory pool selected by SGLANG_MORI_CUSTOM_MEM_POOL.

    Args:
        device: The device to allocate memory on

    Returns:
        Tuple of (enable_custom_mem_pool, custom_mem_pool, custom_mem_pool_type)
    """
    enabled, custom_mem_pool_type = check_mori_custom_mem_pool_enabled()
    if not enabled:
        return False, None, None

    # AMD_GEM: KV pools in amdgpu GEM BOs. amdgpu-dkms counts a hipMalloc buffer
    # that a NIC pins twice in the KFD allocation limit (30.x for dma-buf
    # registrations, 31.10 - 31.60 for all), so registering the KV cache leaves
    # no VRAM for later allocations; a GEM BO is counted once. Without the pool
    # the server would run into that OOM, so failing here beats falling back.
    try:
        from mori.io import HipGemAllocator
    except ImportError as e:
        raise RuntimeError(
            "SGLANG_MORI_CUSTOM_MEM_POOL=AMD_GEM needs a mori build that ships "
            f"mori.io.HipGemAllocator: {e}"
        ) from e

    allocator = HipGemAllocator.get_allocator(device)
    # MemPool binds to the current device; on a non-main thread that is device 0.
    with torch.cuda.device(device):
        custom_mem_pool = torch.cuda.MemPool(allocator.allocator())
    if custom_mem_pool_type not in _logged_pool_types:
        _logged_pool_types.add(custom_mem_pool_type)
        logger.info(
            f"MoRI custom memory pool {custom_mem_pool_type}: KV caches are "
            f"allocated as amdgpu GEM BOs (device {device})"
        )
    return True, custom_mem_pool, custom_mem_pool_type
