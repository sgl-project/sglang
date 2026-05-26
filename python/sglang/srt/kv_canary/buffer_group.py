from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import torch

from sglang.jit_kernel.kv_canary.verify import RealKvSource


class PoolKind(IntEnum):
    """Which attention regime a canary group belongs to.

    - ``FULL`` covers ``[0, K_req)``. Attached to plain MHA/MLA pools and as one of the two canaries on
      every SWA system.
    - ``SWA`` covers ``[max(0, K_req - window), K_req)``. Attached as the second canary on every
      ``BaseSWAKVPool``.
    """

    FULL = 0
    SWA = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryBufferGroup:
    """Canary buffers + real-KV sources for one (PoolKind × K-half | V-half) on a pool.

    Each (head | tail) launch sees a single 2-D uint8 buf for the canary, plus a list of RealKvSource for the
    real-KV mixin. Head and tail use separate canary buffers so they can be staged at different points in the
    forward pass without overwriting each other.

    MLA-style pools have no V half (v_head / v_tail = None; real_kv_sources_v is empty). SWA pools have two
    CanaryBufferGroup instances (FULL sized to the full sub-pool, SWA sized to the swa sub-pool).

    Fields:
        kind: PoolKind.FULL or PoolKind.SWA.
        k_head: Head canary buffer for K-half launches, shape [num_slots, CANARY_SLOT_BYTES], uint8.
        k_tail: Tail canary buffer for K-half launches, same shape, uint8.
        v_head: Same for V-half, or None for MLA-style pools.
        v_tail: Same for V-half, or None.
        real_kv_sources_k: Real KV pieces folded into the K-half canary's real_kv_hash. Tuple length is
            pool-specific (1 for simple MHA, more for multi-layer / weird-layout pools). Empty tuple =
            real-KV mixin disabled for this half.
        real_kv_sources_v: Same for V-half. Empty tuple iff v_head is None or the mixin is disabled.
        swa_index_lut: SWA full-to-swa index mapping LUT, shape [full_pool_size + 1], int64, or None for FULL
            groups. Used by launch_canary_plan_kernels to translate verify/seed slot indices at plan time, and by
            launch_canary_write_kernel to translate write slots inline. None iff kind == PoolKind.FULL.
        kv_token_id_vs_position_offset: Logical-position offset between a canary slot and the source-of-truth token it
            fingerprints. 0 for target-style pools (slot ``p`` stores K/V for token at position ``p``); 1 for
            EAGLE draft pools where the input_ids rotation makes slot ``p`` store K/V for token at position
            ``p + 1``.
    """

    kind: PoolKind
    k_head: torch.Tensor
    k_tail: torch.Tensor
    v_head: Optional[torch.Tensor]
    v_tail: Optional[torch.Tensor]
    real_kv_sources_k: tuple[RealKvSource, ...]
    real_kv_sources_v: tuple[RealKvSource, ...]
    swa_index_lut: Optional[torch.Tensor]
    kv_token_id_vs_position_offset: int

    @property
    def has_v_half(self) -> bool:
        return self.v_head is not None
