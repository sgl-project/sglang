"""CanaryBufferGroup and PoolKind. New API: holds tuples of RealKvSource (defined in jit_kernel)."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.jit_kernel.kv_cache_canary_verify import RealKvSource


class PoolKind(str, enum.Enum):
    """Which attention regime a canary group belongs to.

    - ``FULL`` covers ``[0, K_req)``. Attached to plain MHA/MLA pools and as one of the two canaries on
      every SWA system.
    - ``SWA`` covers ``[max(0, K_req - window), K_req)``. Attached as the second canary on every
      ``BaseSWAKVPool``.
    """

    FULL = "full"
    SWA = "swa"


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryBufferGroup:
    """Canary buffers + real-KV sources for one (PoolKind x K-half | V-half) on a pool.

    Each (head | tail) launch sees a single 2-D uint8 buf for the canary, plus a list of RealKvSource for
    the real-KV mixin. Head and tail use separate canary buffers so they can be staged at different points
    in the forward pass without overwriting each other.

    MLA-style pools have no V half (v_head / v_tail = None; real_kv_sources_v is empty). SWA pools have two
    CanaryBufferGroup instances (FULL sized to the full sub-pool, SWA sized to the swa sub-pool).

    ``swa_index_lut`` is the pool's ``full_to_swa_index_mapping`` (a ``[size_full + 1]`` int LUT whose
    final entry is ``-1`` sentinel) when this group lives on an SWA sub-pool whose slot index space
    differs from the FULL pool. ``None`` otherwise — FULL groups, MHA/MLA pools, and DSV4 (where full and
    swa share the same index space) all leave it unset and the plan consumes full-pool indices directly.

    Fields:
        kind: PoolKind.FULL or PoolKind.SWA.
        k_head: Head canary buffer for K-half launches, shape [num_slots, CANARY_SLOT_BYTES], uint8.
        k_tail: Tail canary buffer for K-half launches, same shape, uint8.
        v_head: Same for V-half, or None for MLA-style pools.
        v_tail: Same for V-half, or None.
        real_kv_sources_k: Real KV pieces folded into the K-half canary's real_kv_hash. Tuple length is
            pool-specific (1 for simple MHA, more for DSV4 / multi-layer / weird-layout pools). Empty
            tuple = real-KV mixin disabled for this half.
        real_kv_sources_v: Same for V-half. Empty tuple iff v_head is None or the mixin is disabled.
        swa_index_lut: SWA LUT for slot index translation, shape [full_pool_size + 1], int32, or None.
    """

    kind: PoolKind
    k_head: torch.Tensor
    k_tail: torch.Tensor
    v_head: Optional[torch.Tensor]
    v_tail: Optional[torch.Tensor]
    real_kv_sources_k: tuple[RealKvSource, ...] = ()
    real_kv_sources_v: tuple[RealKvSource, ...] = ()
    swa_index_lut: Optional[torch.Tensor] = None

    @property
    def has_v_half(self) -> bool:
        return self.v_head is not None
