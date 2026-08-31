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
"""Allocator hooks the paged MLA attention backends need under the unified
memory pool.

Lives in its own module because three unrelated backend families consume it
(fa3, flashinfer_mla, and trtllm_mla with its cutedsl_mla / tokenspeed_mla
subclasses) and none of them should have to import another's module to get it.
"""

from __future__ import annotations

from typing import Callable, Optional

import msgspec
import torch


class UnifiedMLAHooks(msgspec.Struct, frozen=True):
    """Dense-view hooks for one KV allocator.

    All-``None``/1/``False`` for the statically-partitioned pool, where
    ``req_to_token`` already holds physical ids and no translation is needed.
    """

    # Page-level virtual->physical table, gathered through by block-table kernels.
    v2p_page_table: Optional[torch.Tensor]
    # Virtual token id -> DENSE kernel-facing id (tombstones clamped to the sink).
    translate_kv_loc_for_kernel: Optional[Callable[..., torch.Tensor]]
    # Dense page stride scale (= number of full-attention MLA layers).
    kernel_page_multiplier: int
    enabled: bool


_STATIC_POOL = UnifiedMLAHooks(
    v2p_page_table=None,
    translate_kv_loc_for_kernel=None,
    kernel_page_multiplier=1,
    enabled=False,
)


def unified_mla_hooks(allocator) -> UnifiedMLAHooks:
    """Probe ``allocator`` for the unified-pool per-layer-view hooks.

    Detection keys on the v2p table, NOT on ``kernel_page_multiplier > 1``: a
    rank owning exactly ONE full-attention layer has multiplier 1 while its
    ``req_to_token`` is still virtual. There the kernel-facing id collapses onto the
    physical id, so the v2p gather alone is the whole translation.
    """
    v2p = getattr(allocator, "full_v2p_page_table", None)
    if v2p is None:
        return _STATIC_POOL
    return UnifiedMLAHooks(
        v2p_page_table=v2p,
        translate_kv_loc_for_kernel=getattr(
            allocator, "translate_kv_loc_for_kernel", None
        ),
        kernel_page_multiplier=getattr(allocator, "kernel_page_multiplier", 1),
        enabled=True,
    )
