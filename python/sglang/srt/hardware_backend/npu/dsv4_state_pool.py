"""NPU-only paged compress-state pool for the DSV4 fused compressor kernel.

``torch.ops.custom.compressor`` (cann8.5.0-a3 wheel) reads/writes the
compress state via ``state_cache`` shape ``(block_num, page_size,
2*coff*head_dim)`` indexed by ``state_block_table[B, ceil(Smax/page_size)]``.
Per the kernel README (``cann/ops-transformer/experimental/attention/compressor``):

  * ``cache_mode=1`` (paged, default) — block_table values are real block
    ids starting from 1; value 0 means "skip this slot" (don't update).
  * ``cache_mode=2`` (ring) — explicitly **not supported on Atlas A3**.

The CUDA-side :class:`CompressStatePool` sizes itself ring-style
(``_size = swa_pages * ring_size + ring_size + 1``); on NPU with cache_mode=1
the kernel reads/writes by paged block id, so a ring layout misaddresses
slots (block ids hash-collide, and many derived ids are 0 = skip → writes
disappear). Symptom: long-output AIME decoded to garbage after raw seq_len
crossed the first c-pool page boundary, even with USE_FUSED_COMPRESSOR=1.

:class:`NPUCompressStatePool` keeps the parent's buffer layout
(``(_size, 2*coff*head_dim)`` flat, ``state_cache_3d`` reshapes to
``(num_blocks, page_size, 2*coff*head_dim)``) but replaces the size formula
with a paged-allocator-friendly one derived from ``max_num_reqs`` (mirrors
iforgetmyname/sglang dsv4_release ``c_state_max_total_num_tokens``). Block 0
is reserved as the kernel's skip-sentinel slot (zero kv / -inf score) so any
``state_block_table`` entry that defaults to 0 lands in a deterministic,
attention-neutral place.

This class is NPU-only; CUDA continues to use the unchanged
:class:`CompressStatePool` via :class:`DeepSeekV4TokenToKVPool`.
"""

from __future__ import annotations

import math

import torch

from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool


def npu_state_pool_size(
    *,
    ratio: int,
    page_size: int,
    max_num_reqs: int,
) -> int:
    """Per-pool state slot count for the NPU paged state pool's
    :class:`NPUPagedTokenToKVPoolAllocator`.

    Reference formula (iforgetmyname/sglang dsv4_release pool_configurator):

        ``max(2, ceil(1.8 * ratio / page_size) + 1) * max_num_reqs * page_size``

    Sized for steady-state during decode: each req keeps roughly the
    trailing ``sliding_window_size`` worth of state slots live at any one
    time (SWA eviction in :meth:`ScheduleBatch._evict_swa` frees state
    slots as it advances), and the 1.8x factor adds headroom for the
    tail-only allocation pattern across page boundaries.

    Prefill no longer drives sizing because allocation is now tail-only —
    long prompts only allocate ``c{ratio}_alloc_len`` slots
    (``≤ tail + 128`` for c4, ``≤ tail`` for c128, where ``tail =
    seq_len % 128``), not the full raw seqlen. See
    :meth:`ScheduleBatch._compute_dsv4_state_lens_extend` for the
    per-req formula.

    Result is in TOKEN units (matches the SGLang allocator
    ``PagedTokenToKVPoolAllocator(size, ...)`` convention where
    ``num_pages = size // page_size`` is the count of USABLE pages
    handed out by ``free_pages = arange(1, num_pages+1)``). The BUFFER
    must allocate one extra page slot (see :class:`NPUCompressStatePool`
    which sizes its buffer as ``(num_pages + 1) * page_size`` tokens —
    page 0 is the kernel's skip-sentinel).
    """
    blocks_per_req = max(2, math.ceil(1.8 * ratio / page_size) + 1)
    num_usable_pages = blocks_per_req * max_num_reqs
    return num_usable_pages * page_size


class NPUCompressStatePool(CompressStatePool):
    """Paged compress-state pool for the NPU fused compressor kernel.

    Inherits everything from :class:`CompressStatePool` except the size
    formula and the post-init block-0 sentinel fill. The buffer layout
    ``(self._size, 2*coff*head_dim)`` and the ``state_cache_3d`` reshape
    are unchanged, so :meth:`DeepSeekV4TokenToKVPool.get_state_cache`
    keeps returning the kernel-expected ``(num_blocks, page_size,
    2*coff*head_dim)`` view without further plumbing.
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
