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
from typing import List, Optional, Tuple

import torch
import torch_npu

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    ONLINE_C128,
    DeepSeekV4IndexerPool,
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
)


class NPUDeepSeekV4SingleKVPool(DeepSeekV4SingleKVPool):
    """NPU bf16 variant of the full / SWA / c4 / c128 single-KV pool.

    ``npu_sparse_attn_sharedkv`` reads KV in PA_ND layout
    ``(num_pages, kernel_page_size, num_kv_heads=1, dim)`` with ``dim`` packing
    K_nope + K_rope as bf16, and requires ``cmp_kv.shape[1] == ori_kv.shape[1]``.
    So the c4/c128 pools (whose token-level page_size is ``page_size // ratio``)
    are allocated at the GLOBAL ``kernel_page_size`` rather than their own
    per-ratio page_size; the SWA pool uses ``kernel_page_size == page_size``.
    The CUDA fp8-packed-bytes layout (the base ``create_buffer``) is untouched.
    """

    def __init__(self, *args, kernel_page_size: int, **kwargs):
        # Set before super().__init__ — it calls _create_buffers() ->
        # create_buffer(), which reads self.kernel_page_size.
        self.kernel_page_size = kernel_page_size
        super().__init__(*args, **kwargs)

    def create_buffer(self, *, num_pages: int):
        # Non-bf16 store dtype (shouldn't happen here) falls back to base layout.
        if self.store_dtype != torch.bfloat16:
            return super().create_buffer(num_pages=num_pages)
        kv_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_cache_total_dim = kv_dim
        # GLOBAL kernel_page_size keeps cmp_kv.shape[1] == ori_kv.shape[1]; writes
        # are flat-indexed by loc, so page granularity affects shape not location.
        npu_num_pages = (self.size + self.kernel_page_size + 1) // self.kernel_page_size
        return torch.zeros(
            npu_num_pages,
            self.kernel_page_size,
            1,
            kv_dim,
            dtype=torch.bfloat16,
            device=self.device,
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
        # Bypass parent __init__ — its ring-based sizing is incompatible with the
        # kernel's paged block-id contract. We redo buffer alloc and set the same
        # fields so the parent API (state_cache_3d, kv_score_buffer) stays intact.
        assert ratio in (
            4,
            128,
        ), f"NPUCompressStatePool only supports ratio in (4, 128); got {ratio}"
        assert page_size > 1, (
            "NPUCompressStatePool requires page_size>1 (kernel's "
            "state_cache_3d view is (block_num, page_size, slot_dim)). "
            "Got page_size=%d." % page_size
        )

        # ``size`` is the ALLOCATOR's size (npu_state_pool_size output). Buffer
        # needs one EXTRA page so the free list arange(1, num_pages+1) indexes it
        # without OOB (page 0 = skip sentinel; pages 1..num_pages handed out).
        num_usable_pages = (size + page_size - 1) // page_size
        num_buffer_pages = num_usable_pages + 1
        self._size = num_buffer_pages * page_size
        self.ratio = ratio
        self.page_size = page_size
        # ring_size=0 marks "not ring-buffered" (paged allocator replaces the
        # parent's ring hashing); kept so downstream hasattr probes don't break.
        self.ring_size = 0
        # online compress is a CUDA-only opt with no NPU fused-compressor support;
        # force off so layout matches kernel expectations.
        self.online = False

        # Slot dim = 2 * coff * head_dim = [kv | score]; coff = 1 (no overlap) or
        # 2 (overlap). Matches CompressStatePool non-online layout.
        self.last_dim = 2 * (1 + int(overlap)) * head_dim

        # Reuse parent's buffer-alloc helper; only self._size differs from the
        # ring-based parent path.
        self._alloc_kv_score_buffer(
            dtype=dtype, device=device, enable_memory_saver=enable_memory_saver
        )

        # Block 0 = kernel skip-sentinel: kv zeroed, score -inf (softmax → 0).
        # The free list excludes it; only stale state_block_table entries land here.
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
                    npu_num_pages,
                    kp,
                    1,
                    self.index_head_dim,
                    dtype=torch.int8,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.index_scale_buffer = [
                torch.zeros(
                    npu_num_pages,
                    kp,
                    1,
                    1,
                    dtype=torch.float16,
                    device=self.device,
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

    The full / SWA / c4 / c128 KV pools use the NPU bf16 PA_ND layout
    (:class:`NPUDeepSeekV4SingleKVPool`); the compress-state pool is paged
    (:class:`NPUCompressStatePool`) rather than ring-buffered; and the indexer
    pool adds dedicated int8 K + fp16 scale buffers
    (:class:`NPUDeepSeekV4IndexerPool`). The generic-accessor / port-hook
    methods at the bottom of this class are the NPU equivalents of the CUDA
    DSV4 store-cache chain — kept here, not in the community base, which raises
    ``NotImplementedError`` for them (CUDA goes through the radix / store_cache
    accessors instead).
    """

    def _make_kv_pool(
        self,
        *,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        global_page_size: int,
        cls: type = DeepSeekV4SingleKVPool,
    ) -> NPUDeepSeekV4SingleKVPool:
        # NPU does not use the HiSparse c4 device pool; fail loud if someone
        # enables it so the silent layout mismatch surfaces at init.
        assert cls is DeepSeekV4SingleKVPool, (
            "enable_hisparse is not supported on the NPU DSV4 KV pool "
            f"(got c4 pool class {cls.__name__})."
        )
        return NPUDeepSeekV4SingleKVPool(
            size,
            page_size,
            dtype,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            kernel_page_size=global_page_size,
        )

    def _get_state_pool(self, layer_id: int, from_indexer: bool) -> CompressStatePool:
        """Select this layer's attention vs c4-indexer compress-state pool.
        Wraps the community getters so the NPU port hooks below don't index the
        pool lists directly."""
        if from_indexer:
            return self.get_indexer_compress_states(layer_id)
        return self.get_attention_compress_states(layer_id)

    def _make_attn_state_pool(
        self, ratio: int, enable_memory_saver: bool
    ) -> NPUCompressStatePool:
        # ONLINE_C128 (CUDA-only) collapses the c128 ring to size 1; the NPU fused
        # compressor has no online mode, so assert the config mismatch early.
        assert not (ratio == 128 and ONLINE_C128), (
            "SGLANG_OPT_USE_ONLINE_COMPRESS is incompatible with the "
            "NPU fused compressor (no online mode in the kernel)."
        )
        return NPUCompressStatePool(
            size=self._state_pool_size(ratio),
            overlap=ratio == 4,
            head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            dtype=self.c4_state_dtype if ratio == 4 else self.c128_state_dtype,
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
            dtype=self.c4_state_dtype,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            page_size=self.swa_page_size,
        )

    def clear_unaccepted_c128_draft_states(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        accept_lens: torch.Tensor,
        num_draft_tokens: int,
    ) -> None:
        pass

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

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        # No full-token contiguous space on NPU; everything ships per-pool via
        # get_pd_state_components(), so the contiguous path is empty.
        return [], [], []

    def get_pd_state_components(
        self,
    ) -> List[Tuple[str, List[int], List[int], List[int]]]:
        """Ordered ``(AscendStateType, data_ptrs, data_lens, item_lens)`` per pool, in a
        fixed order so prefill and decode register identically (empty pools skipped)."""
        from sglang.srt.disaggregation.ascend.conn import AscendStateType

        components: List[Tuple[str, List[int], List[int], List[int]]] = []

        def kv_entry(bufs):
            return (
                [b.data_ptr() for b in bufs],
                [b.nbytes for b in bufs],
                [b[0].nbytes for b in bufs],
            )

        def state_entry(want_ratio: int, include_indexer: bool):
            ptrs: List[int] = []
            lens: List[int] = []
            ilens: List[int] = []

            def add(pool):
                t = pool.kv_score_buffer.kv_score
                ptrs.append(t.data_ptr())
                lens.append(t.nbytes)
                ilens.append(t[0].nbytes * pool.page_size)

            for ratio, pool in zip(self.compression_ratios, self.compress_state_pools):
                if pool is not None and ratio == want_ratio:
                    add(pool)
            if include_indexer:
                # indexer compress-state pools are all ratio 4 and share the
                # c4_state slot space.
                for pool in self.indexer_compress_state_pools:
                    if pool is not None:
                        add(pool)
            return ptrs, lens, ilens

        # KV pools (4D PA_ND).
        if self.swa_kv_pool is not None:
            components.append(
                (AscendStateType.DSV4_SWA, *kv_entry(self.swa_kv_pool.kv_buffer))
            )
        if self.c4_kv_pool is not None:
            components.append(
                (AscendStateType.DSV4_C4, *kv_entry(self.c4_kv_pool.kv_buffer))
            )
        if self.c128_kv_pool is not None:
            components.append(
                (AscendStateType.DSV4_C128, *kv_entry(self.c128_kv_pool.kv_buffer))
            )
        if self.c4_indexer_kv_pool is not None:
            idx_bufs = list(self.c4_indexer_kv_pool.index_k_buffer) + list(
                self.c4_indexer_kv_pool.index_scale_buffer
            )
            components.append((AscendStateType.DSV4_INDEXER, *kv_entry(idx_bufs)))

        # Compress-state pools (paged, flat 2D). c4_state bundles attn-c4-state +
        # indexer-c4-state (same req_to_token_c4_state slot space).
        components.append(
            (AscendStateType.DSV4_C4_STATE, *state_entry(4, include_indexer=True))
        )
        components.append(
            (AscendStateType.DSV4_C128_STATE, *state_entry(128, include_indexer=False))
        )

        # Drop empty components (e.g. a ratio with no layers) so every shipped
        # component has non-zero item_lens; the set is identical on both sides.
        return [c for c in components if c[1]]

    def get_state_cache(self, layer_id: int, from_indexer: bool) -> torch.Tensor:
        """fp32 ``[block_num, page_size, 2*coff*D]`` view of this layer's
        kv+score buffer — the fused compressor op
        (``torch.ops.custom.compressor``)'s ``state_cache`` argument."""
        return self._get_state_pool(layer_id, from_indexer).state_cache_3d

    # ------------------------------------------------------------------
    # Generic KV accessors (community base raises NotImplementedError; CUDA uses
    # store_cache). AscendAttnBackend reads KV through these, routed to the right
    # sub-pool by compression ratio.
    # ------------------------------------------------------------------

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        item = self.layer_mapping[layer_id]
        ratio = item.compress_ratio
        if ratio == 0:
            return self.swa_kv_pool.kv_buffer[item.compress_layer_id]
        if ratio == 4:
            return self.c4_kv_pool.kv_buffer[item.compress_layer_id]
        if ratio == 128:
            return self.c128_kv_pool.kv_buffer[item.compress_layer_id]
        raise ValueError(f"unsupported compress_ratio={ratio} for get_key_buffer")

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        # V4 uses MQA / latent attention — the K buffer doubles as V.
        return self.get_key_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        buf = self.get_key_buffer(layer_id)
        return buf, buf

    def get_swa_buffer(
        self, layer_id: int, loc: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return the SWA layer's KV cache in PA_ND layout
        (num_pages, page_size, num_kv_heads=1, dim). When ``loc`` is given,
        flatten across (num_pages, page_size) and gather the matching tokens —
        shape becomes (num_tokens, 1, dim).
        """
        # Index by RAW layer_id, not compress_layer_id (a per-bucket counter that
        # would collide across ratios). swa_kv_pool is sized layer_num=total_layers.
        kv = self.swa_kv_pool.kv_buffer[layer_id]
        if loc is not None:
            kv = kv.flatten(0, 1)[loc]
        return kv

    def get_compress_buffer(
        self,
        layer_id: int,
        from_indexer: bool = False,
        loc: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Return the compressed KV buffer for a c4 / c128 layer.

        Routes to c4 / c128 kv_pool by layer compression ratio. Returns
        ``None`` for ratio == 0 (no compress KV exists). The
        from_indexer=True branch returns the dedicated int8 K buffer that
        ``torch.ops.custom.npu_quant_lightning_indexer`` consumes.
        """
        item = self.layer_mapping[layer_id]
        if item.compress_ratio == 4:
            if from_indexer:
                kv = self.c4_indexer_kv_pool.get_index_k(item.compress_layer_id)
            else:
                kv = self.c4_kv_pool.kv_buffer[item.compress_layer_id]
        elif item.compress_ratio == 128:
            assert not from_indexer, "c128 has no indexer pool"
            kv = self.c128_kv_pool.kv_buffer[item.compress_layer_id]
        else:
            return None
        if loc is not None:
            kv = kv.flatten(0, 1)[loc]
        return kv

    def set_swa_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache: torch.Tensor,
    ) -> None:
        """Write ``cache`` into the SWA pool at flat token positions ``loc``.

        ``cache`` shape: (num_tokens, num_kv_heads=1, dim). The buffer view is
        (num_pages, page_size, 1, dim) so we flatten the first two dims and
        index_put.
        """
        # Index by raw layer_id (see get_swa_buffer) to avoid bucket collision.
        buf = self.swa_kv_pool.kv_buffer[layer_id]
        buf_flat = buf.flatten(0, 1)  # (num_pages * page_size, 1, dim)
        # Caller (V4 MQALayer) may hand us cache shaped (T, dim); the buffer has
        # an explicit num_kv_heads=1 axis, so insert it.
        if cache.ndim == buf_flat.ndim - 1:
            cache = cache.unsqueeze(1)
        buf_flat[loc] = cache.to(buf_flat.dtype)

    # ------------------------------------------------------------------
    # NPU port hooks — used by dsv4/{compressor,indexer}.py forward_npu.
    # CompressStatePool stores a fused [kv | score] tensor; split is a last-dim slice.
    # ------------------------------------------------------------------

    def set_state_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        kv: torch.Tensor,
        score: torch.Tensor,
        from_indexer: bool,
    ) -> None:
        # KVAndScore.kv_score is [..., 2*coff*head_dim] = [kv | score].
        kv_score = self._get_state_pool(layer_id, from_indexer).kv_score_buffer.kv_score
        last_dim = kv_score.shape[-1]
        half = last_dim // 2
        kv_view = kv.reshape(-1, half).to(kv_score.dtype)
        score_view = score.reshape(-1, half).to(kv_score.dtype)
        kv_score[loc, :half] = kv_view
        kv_score[loc, half:] = score_view

    def get_state_buffer(
        self,
        layer_id: int,
        from_indexer: bool,
        kv_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kv_score = self._get_state_pool(layer_id, from_indexer).kv_score_buffer.kv_score
        if kv_indices is not None:
            kv_score = kv_score[kv_indices]
        last_dim = kv_score.shape[-1]
        half = last_dim // 2
        kv = kv_score[..., :half].unsqueeze(-2)  # add num_kv_heads=1 axis
        score = kv_score[..., half:].unsqueeze(-2)
        return kv, score

    def set_compress_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        kv: torch.Tensor,
        kv_scale: Optional[torch.Tensor],
        from_indexer: bool,
    ) -> None:
        # Routes to c4_indexer (from_indexer) / c4_kv (ratio 4) / c128_kv (ratio
        # 128). NPU bypasses CUDA fused_store_cache with direct bf16 writes.
        ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        device_type = kv.device.type
        if from_indexer:
            assert ratio == 4, f"indexer only on c4 layers, got ratio={ratio}"
            if device_type == "npu":
                assert (
                    self.c4_indexer_kv_pool.has_npu_storage
                ), "NPU index buffers not allocated — pool was init'd on CUDA?"
                self.c4_indexer_kv_pool.set_index_k_scale(
                    compress_layer_id, loc, kv, kv_scale
                )
                return
            if kv_scale is None:
                self.c4_indexer_kv_pool.set_index_fused(compress_layer_id, loc, kv)
                return
            self.c4_indexer_kv_pool.set_index_k_scale_buffer(
                compress_layer_id, loc, kv, kv_scale
            )
            return
        compress_pool = self.c4_kv_pool if ratio == 4 else self.c128_kv_pool
        if device_type == "npu":
            # PA_ND layout: kv_buffer[layer_id] shape = (num_pages, page_size,
            # 1, kv_dim). Flatten (num_pages, page_size) and index by `loc`.
            buf = compress_pool.kv_buffer[compress_layer_id]
            buf_flat = buf.flatten(0, 1)
            kv_view = kv.to(buf_flat.dtype)
            if kv_view.ndim == buf_flat.ndim - 1:
                kv_view = kv_view.unsqueeze(1)
            buf_flat[loc] = kv_view
            return
        compress_pool.set_key_buffer_fused(compress_layer_id, loc, kv)

    def get_compress_dequant_scale_buffer(
        self,
        layer_id: int,
        from_indexer: bool,
    ) -> torch.Tensor:
        # Returns the float16 dequant scale buffer (NPU indexer pool's dedicated
        # scale buffer alongside the int8 K buffer).
        assert from_indexer, "only indexer compress pool has dequant scale"
        compress_layer_id = self.layer_mapping[layer_id].compress_layer_id
        return self.c4_indexer_kv_pool.get_index_scale(compress_layer_id)

    def translate_kv_loc_to_compress_state_loc(
        self,
        kv_loc: torch.Tensor,
        compress_ratio: int,
    ) -> torch.Tensor:
        # Parent's ring-buffer hash is meaningless under the paged cache_mode=1
        # contract; returning a stale value would silently corrupt state. Fail loud.
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
