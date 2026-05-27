"""NPU-only KV pool variant for DeepSeek-V4.

Subclasses :class:`DeepSeekV4TokenToKVPool` to swap the ring-buffered
:class:`CompressStatePool` for the paged :class:`NPUCompressStatePool` that
the on-NPU fused compressor kernel (``torch.ops.custom.compressor`` with
``cache_mode=1``) requires. Atlas A3 rejects ``cache_mode=2`` (ring) entirely,
so this is the only valid layout on that hardware.

Selected at pool construction time by
:meth:`ModelRunnerKVCacheMixin._init_pools` when the model is DSV4 AND the
device is NPU. CUDA continues to use the unchanged base class.

Only two methods are overridden:

  * ``_init_paged_compress_states`` — build :class:`NPUCompressStatePool`
    instances sized from :func:`npu_state_pool_size` instead of the
    ring-based formula the base class uses.
  * ``translate_kv_loc_to_compress_state_loc`` — raise loudly. The ring
    hash this method implements is meaningless on the paged kernel; callers
    must consume ``out_cache_loc_dsv4.out_c{4,128}_state_loc`` from the
    allocator bundle instead. Currently the only NPU caller that still
    invokes translate is the unfused Python compressor decode path
    (``layers/attention/dsv4/compressor.py``); with USE_FUSED_COMPRESSOR=1
    that path is dead. If someone disables the fused compressor, they hit
    the raise with a clear message.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from sglang.srt.hardware_backend.npu.dsv4_state_pool import (
    NPUCompressStatePool,
    npu_state_pool_size,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    ONLINE_C128,
)


class DSV4NPUTokenToKVPool(DeepSeekV4TokenToKVPool):
    """NPU-only DSV4 KV pool with paged compress-state buffers.

    Mirrors the CUDA :class:`DeepSeekV4TokenToKVPool` for full / SWA / c4 /
    c128 KV pools (those layouts already match the kernel). The only
    behavioral difference is the compress-state pool, which on NPU must be
    paged rather than ring-buffered.
    """

    def _init_paged_compress_states(self, enable_memory_saver: bool) -> None:
        # Same shape as the parent's loop (per ratio, build attention +
        # indexer state pools) but with NPUCompressStatePool instead of
        # the ring-based CompressStatePool.
        self.compress_state_pools: List[Optional[NPUCompressStatePool]] = []
        self.indexer_compress_state_pools: List[Optional[NPUCompressStatePool]] = []

        for ratio in self.compression_ratios:
            overlap = ratio == 4
            compress_state_pool: Optional[NPUCompressStatePool] = None
            indexer_compress_state_pool: Optional[NPUCompressStatePool] = None
            has_compress_state = ratio in (4, 128)

            if has_compress_state:
                # ONLINE_C128 (CUDA-only optimization) collapses c128 ring to
                # size 1; on NPU we always use the standard (kv, score) layout
                # because the fused compressor doesn't have an online mode.
                # Assert here to surface config mismatches early.
                assert not (ratio == 128 and ONLINE_C128), (
                    "SGLANG_OPT_USE_ONLINE_COMPRESS is incompatible with the "
                    "NPU fused compressor (no online mode in the kernel)."
                )
                size = (
                    self.c4_state_pool_size
                    if ratio == 4
                    else self.c128_state_pool_size
                )
                compress_state_pool = NPUCompressStatePool(
                    size=size,
                    overlap=overlap,
                    head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
                    dtype=self.state_dtype,
                    device=self.device,
                    enable_memory_saver=enable_memory_saver,
                    ratio=ratio,
                    page_size=self.swa_page_size,
                )

            if ratio == 4:
                # c4 indexer shares the c4 state pool size budget but has its
                # own slot_dim (indexer_head_dim vs attention head_dim).
                indexer_compress_state_pool = NPUCompressStatePool(
                    size=self.c4_state_pool_size,
                    overlap=overlap,
                    head_dim=self.indexer_head_dim,
                    device=self.device,
                    dtype=self.state_dtype,
                    enable_memory_saver=enable_memory_saver,
                    ratio=ratio,
                    page_size=self.swa_page_size,
                )

            self.compress_state_pools.append(compress_state_pool)
            self.indexer_compress_state_pools.append(indexer_compress_state_pool)

    def translate_kv_loc_to_compress_state_loc(
        self,
        kv_loc: torch.Tensor,
        compress_ratio: int,
    ) -> torch.Tensor:
        # The parent implementation computes
        # ``swa_page * ring_size + (swa_loc % ring_size)`` — a ring-buffer
        # hash that has no meaning under the NPU fused compressor's paged
        # cache_mode=1 contract. Allowing it to return a stale value would
        # silently misaddress the state_block_table and corrupt compressed
        # state across batches. Loud failure makes the misuse obvious.
        raise RuntimeError(
            "DSV4NPUTokenToKVPool.translate_kv_loc_to_compress_state_loc was "
            "called, but the NPU fused compressor kernel uses a paged state "
            "pool (cache_mode=1) and does not support ring-buffer state "
            "addressing (cache_mode=2 is explicitly unsupported on Atlas A3). "
            "Callers must consume out_cache_loc_dsv4.out_c{4,128}_state_loc "
            "from the allocator bundle (set during alloc_extend/alloc_decode) "
            "and read state_page_table from req_to_token_c{4,128}_state on "
            "the DSV4NPUReqToTokenPool instead. See "
            "hardware_backend/npu/dsv4_memory_pool.py for the rationale."
        )
