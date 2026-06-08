"""NPU-only KV pool variant for DeepSeek-V4.

Subclasses :class:`DeepSeekV4TokenToKVPool` to swap the ring-buffered
:class:`CompressStatePool` for the paged :class:`NPUCompressStatePool` that
the on-NPU fused compressor kernel (``torch.ops.custom.compressor`` with
``cache_mode=1``) requires. Atlas A3 rejects ``cache_mode=2`` (ring) entirely,
so this is the only valid layout on that hardware.

Selected at pool construction time by
:meth:`ModelRunnerKVCacheMixin._init_pools` when the model is DSV4 AND the
device is NPU. CUDA continues to use the unchanged base class.

The subclass overrides only:

  * ``_make_attn_state_pool`` / ``_make_indexer_state_pool`` — the per-ratio
    state-pool factories the base ``_init_paged_compress_states`` loop calls.
    Both return :class:`NPUCompressStatePool` (paged, ``cache_mode=1``)
    instead of the base's ring-buffered :class:`CompressStatePool`.
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

import math
from typing import Optional

import torch
import torch_npu

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4IndexerPool,
    DeepSeekV4TokenToKVPool,
    ONLINE_C128,
)


def npu_state_pool_size(
    *,
    ratio: int,
    page_size: int,
    max_num_reqs: int,
) -> int:
    """Per-pool state slot count for the NPU paged state pool's
    :class:`NPUPagedTokenToKVPoolAllocator`.

    Sizing formula::

        max(2, ceil(1.8 * ratio / page_size) + 1) * max_num_reqs * page_size

    Sized for steady-state during decode: each req keeps roughly the trailing
    ``sliding_window_size`` worth of state slots live at any one time (SWA
    eviction in :meth:`ScheduleBatch._evict_swa` frees state slots as it
    advances), and the 1.8x factor adds headroom for the tail-only allocation
    pattern across page boundaries.

    Prefill no longer drives sizing because allocation is tail-only — long
    prompts only allocate ``c{ratio}_alloc_len`` slots (``≤ tail + 128`` for
    c4, ``≤ tail`` for c128, where ``tail = seq_len % 128``), not the full raw
    seqlen. See :meth:`ScheduleBatch._compute_dsv4_state_lens_extend` for the
    per-req formula.

    Result is in TOKEN units (matches the SGLang allocator
    ``PagedTokenToKVPoolAllocator(size, ...)`` convention where
    ``num_pages = size // page_size`` is the count of USABLE pages handed out
    by ``free_pages = arange(1, num_pages+1)``). The BUFFER allocates one extra
    page (see :class:`NPUCompressStatePool`, sized ``(num_pages + 1) *
    page_size`` — page 0 is the kernel's skip-sentinel).
    """
    blocks_per_req = max(2, math.ceil(1.8 * ratio / page_size) + 1)
    num_usable_pages = blocks_per_req * max_num_reqs
    return num_usable_pages * page_size


class NPUCompressStatePool(CompressStatePool):
    """Paged compress-state pool for the NPU fused compressor kernel.

    ``torch.ops.custom.compressor`` (cache_mode=1) reads/writes the compress
    state via ``state_cache`` shape ``(block_num, page_size, 2*coff*head_dim)``
    indexed by a paged ``state_block_table`` (block ids from 1; value 0 means
    "skip this slot"). The CUDA :class:`CompressStatePool` sizes itself
    ring-style, which misaddresses slots under cache_mode=1 (ring is also
    unsupported on Atlas A3). This subclass keeps the parent's buffer layout
    (``(self._size, 2*coff*head_dim)`` flat; ``state_cache_3d`` reshapes to
    ``(num_blocks, page_size, 2*coff*head_dim)``) but replaces the size formula
    with a paged one derived from ``max_num_reqs``. Block 0 is reserved as the
    kernel's skip-sentinel (zero kv / -inf score) so any ``state_block_table``
    entry defaulting to 0 lands in a deterministic, attention-neutral place.

    NPU-only; CUDA keeps using the unchanged :class:`CompressStatePool`.
    """

    def __init__(
        self,
        *,
        size: int,
        overlap: bool,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
        enable_memory_saver: bool,
        ratio: int,
        page_size: int,
    ):
        # Bypass parent __init__ entirely — its sizing path is ring-based
        # (size + ring_size + 1, padded to lcm(ratio, page_size)) which is
        # incompatible with the kernel's paged block-id contract. Re-do
        # the buffer allocation ourselves; rest of the parent API
        # (``state_cache_3d`` property, ``kv_score_buffer`` attribute,
        # ``__getitem__`` semantics via KVAndScore) stays intact because
        # we set the same fields.
        assert ratio in (4, 128), f"NPUCompressStatePool only supports ratio in (4, 128); got {ratio}"
        assert page_size > 1, (
            "NPUCompressStatePool requires page_size>1 (kernel's "
            "state_cache_3d view is (block_num, page_size, slot_dim)). "
            "Got page_size=%d." % page_size
        )

        # ``size`` is the ALLOCATOR's size (num_usable_pages * page_size,
        # output of :func:`npu_state_pool_size`). The buffer needs one EXTRA
        # page so the allocator's free list ``arange(1, num_pages+1)`` can
        # index it without OOB (page 0 = dummy/skip sentinel, pages
        # 1..num_pages = real handed out by allocator). Mirrors the
        # DeepSeekV4SingleKVPool convention
        # ``num_pages = (size + page_size + 1) // page_size`` (= size/page_size + 1
        # when size % page_size == 0).
        num_usable_pages = (size + page_size - 1) // page_size
        num_buffer_pages = num_usable_pages + 1
        self._size = num_buffer_pages * page_size
        self.page_size = page_size
        # Mark this instance as "not ring-buffered" so any code that branches
        # on ring_size sees a benign 0. Parent uses ring_size to derive
        # state_loc via hashing; we replaced that with the paged allocator,
        # but keep the attribute so downstream isinstance / hasattr probes
        # don't break.
        self.ring_size = 0
        # online compress (3*head_dim slots) is a CUDA-only optimization
        # (SGLANG_OPT_USE_ONLINE_COMPRESS) and does not interact with the
        # NPU fused compressor; force off here so layout matches kernel
        # expectations.
        self.online = False

        # Slot dim = 2 * coff * head_dim = [kv | score] concatenated, where
        # coff = 1 (no overlap) or 2 (overlap). Matches CompressStatePool's
        # non-online layout exactly so KVAndScore.kv / .score halves work.
        self.last_dim = 2 * (1 + int(overlap)) * head_dim

        # Reuse the parent's buffer-allocation helper (memory-saver region +
        # custom mem pool + KVAndScore wrap); only ``self._size`` differs from
        # the ring-based parent path.
        self._alloc_kv_score_buffer(
            dtype=dtype, device=device, enable_memory_saver=enable_memory_saver
        )

        # Reserve block 0 as the kernel's skip-sentinel:
        # kv half zeroed, score half -inf so any softmax over it contributes 0.
        # The paged allocator's free list excludes block 0, so real reqs never
        # land here; only stale / unallocated state_block_table entries do.
        self.kv_score_buffer.kv[:page_size].zero_()
        self.kv_score_buffer.score[:page_size].fill_(float("-inf"))


class NPUDeepSeekV4IndexerPool(DeepSeekV4IndexerPool):
    """NPU c4-indexer pool. Keeps the base packed CUDA buffer (read by
    get_contiguous_buf_infos / NSA) and ADDS dedicated int8 K + float16 scale
    buffers in PA_ND layout at the global ``kernel_page_size``, written by
    ``torch_npu.npu_scatter_nd_update_`` and read by
    ``torch.ops.custom.npu_quant_lightning_indexer``.
    """

    def __init__(self, *args, kernel_page_size: int, **kwargs):
        # Set before super().__init__ — it calls _create_buffer().
        self._kernel_page_size = kernel_page_size
        super().__init__(*args, **kwargs)

    def _create_buffer(self):
        # Base allocates the packed CUDA index_k_with_scale_buffer (kept for
        # get_contiguous_buf_infos / NSA compat); then add the NPU buffers.
        super()._create_buffer()
        kp = self._kernel_page_size
        npu_num_pages = (self.size + kp + 1) // kp
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.index_k_buffer = [
                torch.zeros(
                    npu_num_pages, kp, 1, self.index_head_dim,
                    dtype=torch.int8, device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.index_scale_buffer = [
                torch.zeros(
                    npu_num_pages, kp, 1, 1,
                    dtype=torch.float16, device=self.device,
                )
                for _ in range(self.layer_num)
            ]

    @property
    def has_npu_storage(self) -> bool:
        return True

    def get_index_k(self, layer_id: int) -> torch.Tensor:
        return self.index_k_buffer[layer_id]

    def get_index_scale(self, layer_id: int) -> torch.Tensor:
        return self.index_scale_buffer[layer_id]

    def set_index_k_scale(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: Optional[torch.Tensor],
    ) -> None:
        # int8 K + fp16 scale come from _compressor_epilog_npu's npu_dynamic_quant
        # output (index_k: int8 [T, D], index_k_scale: fp16 [T, 1]).
        d = self.index_head_dim
        loc_long = loc.view(-1, 1).long()
        torch_npu.npu_scatter_nd_update_(
            self.index_k_buffer[layer_id].view(-1, 1, d),
            loc_long,
            index_k.to(torch.int8).view(-1, 1, d),
        )
        if index_k_scale is not None:
            torch_npu.npu_scatter_nd_update_(
                self.index_scale_buffer[layer_id].view(-1, 1, 1),
                loc_long,
                index_k_scale.to(torch.float16).view(-1, 1, 1),
            )


class DSV4NPUTokenToKVPool(DeepSeekV4TokenToKVPool):
    """NPU-only DSV4 KV pool with paged compress-state buffers.

    Mirrors the CUDA :class:`DeepSeekV4TokenToKVPool` for full / SWA / c4 /
    c128 KV pools (those layouts already match the kernel). The only
    behavioral difference is the compress-state pool, which on NPU must be
    paged rather than ring-buffered.
    """

    def _make_attn_state_pool(
        self, ratio: int, enable_memory_saver: bool
    ) -> NPUCompressStatePool:
        # ONLINE_C128 (CUDA-only optimization) collapses the c128 ring to
        # size 1; the NPU fused compressor has no online mode, so the
        # standard (kv, score) layout is the only valid one — assert the
        # config mismatch early.
        assert not (ratio == 128 and ONLINE_C128), (
            "SGLANG_OPT_USE_ONLINE_COMPRESS is incompatible with the "
            "NPU fused compressor (no online mode in the kernel)."
        )
        return NPUCompressStatePool(
            size=self._state_pool_size(ratio),
            overlap=ratio == 4,
            head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            dtype=self.state_dtype,
            device=self.device,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            page_size=self.swa_page_size,
        )

    def _make_indexer_state_pool(
        self, ratio: int, enable_memory_saver: bool
    ) -> NPUCompressStatePool:
        # c4 indexer shares the c4 state pool size budget but has its own
        # slot_dim (indexer_head_dim vs attention head_dim).
        return NPUCompressStatePool(
            size=self.c4_state_pool_size,
            overlap=ratio == 4,
            head_dim=self.indexer_head_dim,
            device=self.device,
            dtype=self.state_dtype,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            page_size=self.swa_page_size,
        )

    def _make_indexer_pool(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
    ) -> NPUDeepSeekV4IndexerPool:
        # NPU dedicated int8 K + fp16 scale buffers use the GLOBAL page_size
        # (= self.page_size) as kernel_page_size, matching ori_kv for the kernel.
        return NPUDeepSeekV4IndexerPool(
            size,
            page_size,
            dtype,
            index_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            kernel_page_size=self.page_size,
        )

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
