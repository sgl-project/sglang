"""NPU-only KV pool variant for DeepSeek-V4.

The full/SWA/C4/C128 KV buffers keep their Ascend-specific PA_ND layout. The
Compressor state buffers, however, use the same ownership and flat ``state_loc``
rules as the GPU implementation:

* C4A/C4Li state follows SWA physical pages.
* C128A state follows ``req_pool_idx`` and absolute position.

``NPUCompressStatePool`` adds the contiguous 3-D view and positive dummy
location required by the Atlas fused compressor operators. arch22 uses explicit
locations; Ascend 950 uses the same ring storage through its request-bank (cycle) ABI.
There is no paged state allocator or ``cache_mode=1`` compatibility storage.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch_npu

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.hardware_backend.npu.dsv4.npu_kv_layout import NPUDSV4KVLayout
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.mem_cache.deepseek_v4_buffer_info import (
    DSV4IndexerBufferInfo,
    DSV4PagedBufferInfo,
)
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    ONLINE_C128,
    DeepSeekV4IndexerPool,
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
)
from sglang.srt.runtime_context import get_schedule


class NPUDeepSeekV4SingleKVPool(DeepSeekV4SingleKVPool):
    """NPU PA_ND variant of the full / SWA / c4 / c128 single-KV pool.

    ``npu_sparse_attn_sharedkv`` reads KV in PA_ND layout
    ``(num_pages, kernel_page_size, num_kv_heads=1, dim)`` with ``dim`` packing
    K_nope + K_rope as bf16 before arch35; arch35 uses packed FP8 KV rows (the row
    format comes from :class:`NPUDSV4KVLayout`). C4 uses its native page so
    its physical page id can be shared with the corresponding full page. C128
    uses its independently configured physical page size; Full/SWA use the
    global page size. The CUDA fp8-packed-bytes layout (the base
    ``create_buffer``) is untouched.
    """

    def __init__(
        self,
        *args,
        layout: NPUDSV4KVLayout,
        kernel_page_size: int,
        **kwargs,
    ):
        # Set before super().__init__ — it calls _create_buffers() ->
        # create_buffer(), which reads self.layout and self.kernel_page_size.
        self.layout = layout
        self.kernel_page_size = kernel_page_size
        # kernel_page_size is consumed here: the CUDA base __init__ has no
        # such parameter, so it must not be forwarded.
        super().__init__(*args, **kwargs)

    def paged_buffer_info(self) -> DSV4PagedBufferInfo:
        if self.store_dtype != torch.bfloat16:
            return super().paged_buffer_info()
        return DSV4PagedBufferInfo.from_buffers(
            self.kv_buffer, page_slots=self.kernel_page_size, layout="pa_nd"
        )

    def create_buffer(self, *, num_pages: int):
        # Non-bf16 store dtype (shouldn't happen here) falls back to base layout.
        if self.store_dtype != torch.bfloat16:
            return super().create_buffer(num_pages=num_pages)
        kv_dim, kv_dtype = self.layout.kv_info(
            self.qk_nope_head_dim, self.qk_rope_head_dim
        )
        self.kv_cache_total_dim = kv_dim
        # The HiCache assembler uses bytes_per_page_padded as host item_bytes.
        # For example: kernel_page_size * kv_dim * sizeof(bf16).
        self.bytes_per_page_padded = self.kernel_page_size * kv_dim * kv_dtype.itemsize
        # Writes are flat-indexed by loc; kernel_page_size controls the physical
        # page layout exposed to the NPU operators.
        npu_num_pages = (self.size + self.kernel_page_size + 1) // self.kernel_page_size
        return torch.zeros(
            npu_num_pages,
            self.kernel_page_size,
            1,
            kv_dim,
            dtype=kv_dtype,
            device=self.device,
        )


class DSV41NPUSingleKVPool(NPUDeepSeekV4SingleKVPool):
    """NPU DeepSeek-V4.1 KV pool.

    Rows are fully quantized by ``kv_compress_epilog_v2`` (no bf16 RoPE tail),
    so the row width and store dtype come from :class:`NPUDSV4KVLayout`.
    ``layout`` picks the side: ``V41_SWA_FP8`` is the model-wide window/main
    pool (FP8 E4M3 group 32, 544 B) and ``V41_CMP_FP4`` one ratio's compressed
    pool (FP4 E2M1 group 16, 320 B, whose page holds ``page_size // ratio``
    rows so its page id lines up with the full page id). The ratio-1
    compressed pool keeps the global page size like the window pool but must
    still take the compressed row format; it arrives pre-tagged through
    ``DSV4NPUTokenToKVPool.compressed_kv_layout()``.
    """

    def __init__(
        self,
        *args,
        layout: NPUDSV4KVLayout,
        kernel_page_size: int,
        **kwargs,
    ):
        assert layout in (
            NPUDSV4KVLayout.V41_SWA_FP8,
            NPUDSV4KVLayout.V41_CMP_FP4,
        ), layout
        super().__init__(
            *args, layout=layout, kernel_page_size=kernel_page_size, **kwargs
        )

    @property
    def dsv41_row_bytes(self) -> int:
        return self.kv_cache_total_dim

    def create_buffer(self, *, num_pages: int):
        kv_dim, kv_dtype = self.layout.kv_info(
            self.qk_nope_head_dim, self.qk_rope_head_dim
        )
        self.kv_cache_total_dim = kv_dim
        # The HiCache assembler uses bytes_per_page_padded as host item_bytes.
        self.bytes_per_page_padded = self.kernel_page_size * kv_dim * kv_dtype.itemsize
        npu_num_pages = (self.size + self.kernel_page_size + 1) // self.kernel_page_size
        return torch.zeros(
            npu_num_pages,
            self.kernel_page_size,
            1,
            kv_dim,
            dtype=kv_dtype,
            device=self.device,
        )


class NPUCompressStatePool(CompressStatePool):
    """Thin Atlas adapter over the shared GPU-style ring state pool.

    Allocation, sizing, ring ownership and address translation are inherited
    from :class:`CompressStatePool`. NPU only requests a contiguous 3-D view,
    enforces the FP32 state-cache contract and replaces invalid locations with
    a cleared positive dummy row for explicit-location callers.

    Location 0 is valid in explicit mode. Invalid/history-padding locations map
    to the final cleared row instead of ``-1`` because the arch22 kernel consumes
    unsigned offsets.
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
        ring_size: int,
        request_scoped: bool,
        swa_page_size: int,
    ):
        assert ratio in (
            4,
            128,
        ), f"NPUCompressStatePool only supports ratio in (4, 128); got {ratio}"
        assert dtype == torch.float32, (
            "arch22 npu.compressor requires FP32 state_cache, "
            f"but NPUCompressStatePool got {dtype}."
        )
        assert ring_size > 0, f"ring_size must be positive, got {ring_size}"
        super().__init__(
            size=size,
            ring_size=ring_size,
            overlap=overlap,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            online=False,
            request_scoped=request_scoped,
            swa_page_size=swa_page_size,
            state_cache_page_size=ring_size,
        )
        self.dummy_state_loc = self._size - 1

        # The shared pool initializes its dummy row. A cold C128 request bank
        # additionally needs every row initialized before its first partial use.
        if ratio == 128:
            self.kv_score_buffer.clear()

    def _replace_invalid_with_dummy(self, state_loc: torch.Tensor) -> torch.Tensor:
        return torch.where(
            state_loc < 0,
            torch.full_like(state_loc, self.dummy_state_loc),
            state_loc,
        )

    def translate_from_swa_loc_to_state_loc(
        self, swa_loc: torch.Tensor
    ) -> torch.Tensor:
        return self._replace_invalid_with_dummy(
            super().translate_from_swa_loc_to_state_loc(swa_loc)
        )

    def translate_from_req_position_to_state_loc(
        self, req_pool_indices: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return self._replace_invalid_with_dummy(
            super().translate_from_req_position_to_state_loc(
                req_pool_indices, positions
            )
        )


class NPUDeepSeekV4IndexerPool(DeepSeekV4IndexerPool):
    """NPU c4-indexer pool. Allocates only dedicated int8 K + float16 scale
    buffers in PA_ND layout at the native C4 ``kernel_page_size``, written by
    ``torch_npu.npu_scatter_nd_update_`` and read by
    ``torch.ops.custom.npu_quant_lightning_indexer``. The base packed CUDA
    ``index_k_with_scale_buffer`` is NOT allocated: every NPU consumer (the
    compute scatter/read path, the PD ``get_contiguous_buf_infos`` override,
    and the HiCache indexer regions) addresses the NPU buffers directly, so
    the CUDA packed buffer would be dead memory.
    """

    def __init__(self, *args, kernel_page_size: int, **kwargs):
        # Set before super().__init__ — it calls _create_buffer().
        self._kernel_page_size = kernel_page_size
        super().__init__(*args, **kwargs)

    def _create_buffer(self):
        # No super()._create_buffer(): the packed CUDA buffer is dead memory on
        # NPU; keep the attr defined (as None) for generic ``is not None`` checks.
        self.index_k_with_scale_buffer = None
        kp = self._kernel_page_size
        npu_num_pages = (self.size + kp + 1) // kp
        (
            k_dim,
            scale_dim,
            index_k_dtype,
            index_scale_dtype,
        ) = self._npu_index_row_dims()
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.index_k_buffer = [
                torch.zeros(
                    npu_num_pages,
                    kp,
                    1,
                    k_dim,
                    dtype=index_k_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.index_scale_buffer = [
                torch.zeros(
                    npu_num_pages,
                    kp,
                    1,
                    scale_dim,
                    dtype=index_scale_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

    def _npu_index_row_dims(self) -> Tuple[int, int, torch.dtype, torch.dtype]:
        """Last-dim sizes and dtypes of the NPU index K / scale PA_ND buffers.

        The c4 (fp8) pool stores ``index_head_dim`` fp8 values plus a single
        fp32 scale per token. The V4.1 low-ratio pools override this to store
        packed fp4 with one uint8 scale per 32 values (see
        :class:`DSV41NPUIndexerPool`).
        """
        layout = NPUDSV4KVLayout.V4_FP8 if is_npu_arch35() else NPUDSV4KVLayout.V4_BF16
        return layout.indexer_kv_info(self.index_head_dim)

    def indexer_buffer_info(self) -> DSV4IndexerBufferInfo:
        return DSV4IndexerBufferInfo(
            key=DSV4PagedBufferInfo.from_buffers(
                self.index_k_buffer,
                page_slots=self._kernel_page_size,
                layout="pa_nd",
            ),
            scale=DSV4PagedBufferInfo.from_buffers(
                self.index_scale_buffer,
                page_slots=self._kernel_page_size,
                layout="pa_nd",
            ),
        )

    @property
    def has_npu_storage(self) -> bool:
        return True

    def get_index_k(self, layer_id: int) -> torch.Tensor:
        return self.index_k_buffer[layer_id]

    def get_index_scale(self, layer_id: int) -> torch.Tensor:
        return self.index_scale_buffer[layer_id]

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError(
            "NPU indexer pools keep key and scale in separate PA_ND buffers; "
            "use get_index_k / get_index_scale instead"
        )

    def contiguous_page_row_buffers(self) -> List[torch.Tensor]:
        # NPU analogue of the base fp4 branch: expose the PA_ND K and scale
        # buffers as 2D uint8 page-row views for generic row consumers.
        return [
            buf.view(torch.uint8).flatten(1)
            for buf in (*self.index_k_buffer, *self.index_scale_buffer)
        ]

    def set_index_k_scale(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: Optional[torch.Tensor],
    ) -> None:
        d = self.index_head_dim
        loc_long = loc.view(-1, 1).long()
        index_k_cache = self.index_k_buffer[layer_id]
        torch_npu.npu_scatter_nd_update_(
            index_k_cache.view(-1, 1, d),
            loc_long,
            index_k.to(index_k_cache.dtype).view(-1, 1, d),
        )
        if index_k_scale is not None:
            index_scale_cache = self.index_scale_buffer[layer_id]
            torch_npu.npu_scatter_nd_update_(
                index_scale_cache.view(-1, 1, 1),
                loc_long,
                index_k_scale.to(index_scale_cache.dtype).view(-1, 1, 1),
            )


class DSV41NPUIndexerPool(NPUDeepSeekV4IndexerPool):
    """NPU DeepSeek-V4.1 low-ratio indexer K pool (packed fp4).

    ``indexer_prologue_k`` consumes two separate PA_ND buffers
    (vendor operator design notes, ``indexer_prologue_k_dsl`` README)::

        k_cache        uint8 [pages, page_size // ratio, 1, index_head_dim // 2]
        k_scale_cache  uint8 [pages, page_size // ratio, 1, index_head_dim // 32]

    i.e. 64 + 4 = 68 bytes per token at ``index_head_dim == 128``, matching the
    base ``get_dsv4_indexer_bytes_per_token(..., use_fp4_indexer=True)``. The
    inherited fp8-shaped buffers are replaced through
    :meth:`_npu_index_row_dims`.

    K is written by the operator itself (its ``cache_index`` is an INT64 loc
    whose ``-1`` entries are skipped), so the inherited torch scatter
    :meth:`set_index_k_scale` is not on this path.
    """

    def __init__(self, *args, kernel_page_size: int, **kwargs):
        super().__init__(*args, kernel_page_size=kernel_page_size, **kwargs)

    def _npu_index_row_dims(self) -> Tuple[int, int, torch.dtype, torch.dtype]:
        assert self.index_head_dim > 0, self.index_head_dim
        return NPUDSV4KVLayout.V41_CMP_FP4.indexer_kv_info(self.index_head_dim)


class DSV4NPUTokenToKVPool(DeepSeekV4TokenToKVPool):
    """NPU-only DSV4 KV pool with explicit-location ring state buffers.

    The full / SWA / c4 / c128 KV pools use the NPU bf16 PA_ND layout
    (:class:`NPUDeepSeekV4SingleKVPool`); :class:`NPUCompressStatePool`
    exposes the explicit-location ring view required by arch22;
    and the indexer pool adds dedicated int8 K + fp16 scale buffers
    (:class:`NPUDeepSeekV4IndexerPool`). The generic-accessor / port-hook
    methods at the bottom of this class are the NPU equivalents of the CUDA
    DSV4 store-cache chain — kept here, not in the community base, which raises
    ``NotImplementedError`` for them (CUDA goes through the radix / store_cache
    accessors instead).
    """

    def __init__(self, *args, **kwargs):
        c128_page_size = get_schedule().c128_page_size
        if c128_page_size <= 0 or c128_page_size % 16 != 0:
            raise ValueError(
                "c128_page_size must be a positive multiple of 16 for the NPU "
                f"sparse-attention operator, got {c128_page_size}"
            )
        self.c128_page_size = c128_page_size
        super().__init__(*args, **kwargs)

    def _is_v41_model(self) -> bool:
        return any(
            r in NPUDSV4KVLayout.COMPRESSED_RATIOS for r in self.compression_ratios
        )

    def compressed_kv_layout(self, compress_ratio: int):
        """NPU pools arrive at ``_make_kv_pool`` pre-tagged with their row
        format: the V4.1 low ratios compress through epilog_v2's fp4 rows
        unconditionally (the shared layout option does not apply here), and
        the V4 ratios take the arch35/arch22 row by part."""
        if compress_ratio in NPUDSV4KVLayout.COMPRESSED_RATIOS and self._is_v41_model():
            return NPUDSV4KVLayout.V41_CMP_FP4
        return NPUDSV4KVLayout.V4_FP8 if is_npu_arch35() else NPUDSV4KVLayout.V4_BF16

    def _window_kv_layout(self):
        """Row format of the non-compressed (full/swa/window) pools, resolved
        from the generation and the part; see the shared default."""
        if self._is_v41_model():
            return NPUDSV4KVLayout.V41_SWA_FP8
        return NPUDSV4KVLayout.V4_FP8 if is_npu_arch35() else NPUDSV4KVLayout.V4_BF16

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
        kv_layout: NPUDSV4KVLayout,
    ) -> NPUDeepSeekV4SingleKVPool:
        # NPU does not use the HiSparse c4 device pool; fail loud if someone
        # enables it so the silent layout mismatch surfaces at init.
        assert cls is DeepSeekV4SingleKVPool, (
            "enable_hisparse is not supported on the NPU DSV4 KV pool "
            f"(got c4 pool class {cls.__name__})."
        )
        pool_args = (
            size,
            page_size,
            dtype,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
        )
        # Every pool arrives pre-tagged by _window_kv_layout() or
        # compressed_kv_layout(); a raw shared value here is a caller bug.
        assert isinstance(kv_layout, NPUDSV4KVLayout), (
            f"NPU pools require an NPUDSV4KVLayout, got {kv_layout!r}"
        )
        if kv_layout in (
            NPUDSV4KVLayout.V41_SWA_FP8,
            NPUDSV4KVLayout.V41_CMP_FP4,
        ):
            # A V4.1 window page is the global page; a compressed page
            # holds global // ratio rows. Both keep their own page as
            # the kernel view.
            return DSV41NPUSingleKVPool(
                *pool_args,
                layout=kv_layout,
                kernel_page_size=page_size,
            )
        # A V4 window page is the global page; c4 keeps its native page and
        # c128 its independently configured page.
        if page_size * 4 == global_page_size:
            kernel_page_size = page_size
        elif page_size * 128 == global_page_size:
            kernel_page_size = self.c128_page_size
        else:
            kernel_page_size = global_page_size
        return NPUDeepSeekV4SingleKVPool(
            *pool_args,
            layout=kv_layout,
            kernel_page_size=kernel_page_size,
        )

    def _get_state_pool(self, layer_id: int, from_indexer: bool) -> CompressStatePool:
        """Select this layer's attention vs c4-indexer compress-state pool.
        Wraps the community getters so the NPU port hooks below don't index the
        pool lists directly."""
        if from_indexer:
            return self.get_indexer_compress_states(layer_id)
        return self.get_attention_compress_states(layer_id)

    def _make_compress_state_pool(
        self, ratio: int, *, head_dim: int, enable_memory_saver: bool
    ) -> NPUCompressStatePool:
        # ONLINE_C128 (CUDA-only) collapses the c128 ring to size 1; the NPU fused
        # compressor has no online mode, so assert the config mismatch early.
        assert not (ratio == 128 and ONLINE_C128), (
            "SGLANG_OPT_USE_ONLINE_COMPRESS is incompatible with the "
            "NPU fused compressor (no online mode in the kernel)."
        )
        config = self.compressed_pool_configs[ratio]
        ring_size = self.get_ring_size(ratio)
        # Ascend 950 cache_mode=2 addresses one ring bank per request.  The arch22
        # explicit-location path can share the smaller flat pool, but the Ascend 950
        # cycle ABI needs enough physical banks for every req_pool_idx.
        size = config.state_size
        if is_npu_arch35():
            size = max(size, self.num_req_slots * ring_size)
        return NPUCompressStatePool(
            size=size,
            ring_size=ring_size,
            overlap=ratio == 4,
            head_dim=head_dim,
            dtype=config.state_dtype,
            device=self.device,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            request_scoped=ratio == 128,
            swa_page_size=self.swa_page_size,
        )

    def _make_pair_state_pool(self, enable_memory_saver: bool) -> CompressStatePool:
        """Ratio-2 pending-pair state exposed as a physical
        ``[block_num, ring_size, 2*D]`` bank.

        The community factory leaves the flat ``page_size=1`` default, on which
        :attr:`CompressStatePool.state_cache_3d` raises — and that view is the
        ``state_cache`` the fused ``ds41.compressor`` gets on arch35 (one request
        bank per ``state_block_table`` row). Only the second-axis page size is
        turned on here; sizing, ring ownership and the PD transfer indices stay
        in :class:`CompressStatePool`.
        """
        ring_size = self.get_ring_size(2)
        return CompressStatePool(
            size=self.num_req_slots * ring_size,
            ring_size=ring_size,
            overlap=False,
            head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            dtype=torch.float32,
            device=self.device,
            enable_memory_saver=enable_memory_saver,
            ratio=2,
            request_scoped=True,
            online=False,
            state_cache_page_size=ring_size,
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
        force_fp4: bool = False,
        global_page_size: Optional[int] = None,
        index_k_page8: bool = False,
    ) -> NPUDeepSeekV4IndexerPool:
        assert not index_k_page8, (
            "index_k_page8 is a CUDA/AITER fp4 layout; the NPU pool expresses "
            "the same packing through indexer_prologue_k's combined_block_size."
        )
        is_v41 = self._is_v41_model()
        if force_fp4:
            assert is_v41, (
                "force_fp4 indexer pools on NPU are the DeepSeek-V4.1 low-ratio "
                f"pools, but compression_ratios={list(self.compression_ratios)}"
            )
            # Ratios 1 and 2 share the same packed-fp4 row shape, told apart
            # only by ``size``, so the pool needs no ratio-derived state.
            pool = DSV41NPUIndexerPool(
                size,
                page_size,
                dtype,
                index_head_dim,
                layer_num,
                device,
                enable_memory_saver,
                kernel_page_size=page_size,
                use_fp4_indexer=True,
                global_page_size=global_page_size,
            )
            # Same fp4 reference-rounding marker the base sets for `force_fp4`
            # pools; index_k_page8 is CUDA/AITER-only and stays False here.
            pool.index_k_rne = True
            return pool
        # Indexer shares C4 addresses and therefore uses the same native page.
        return NPUDeepSeekV4IndexerPool(
            size,
            page_size,
            dtype,
            index_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            kernel_page_size=page_size,
            global_page_size=global_page_size,
        )

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """Main PD buffers addressed by the full KV page id.

        One item is the whole content of one full page: a C4 / low-ratio
        compressed pool pages at ``page_size // ratio`` rows, i.e. exactly the
        rows one full page holds, which is why the two share a page id. The
        index pools page at ``DSV41_INDEX_PAGE_SIZE`` instead, so one item is
        the run of index pages tiling a full page.
        """
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        if self.c4_kv_pool is not None:
            indexer_pool = self._indexer_pool(4)
            for buf in (
                self.c4_kv_pool.kv_buffer
                + indexer_pool.index_k_buffer
                + indexer_pool.index_scale_buffer
            ):
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)

        # V4.1 has no C4 pool: its main component is the per-ratio compressed
        # KV + index K + index scale. Both PD peers walk the V4.1 low ratios
        # against the same compression config, so entries pair up positionally;
        # pools without such layers contribute nothing.
        for ratio in NPUDSV4KVLayout.COMPRESSED_RATIOS:
            kv_pool = self.kv_pools.get(ratio)
            if kv_pool is None:
                continue
            for buf in kv_pool.kv_buffer:
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)
            indexer_pool = self._indexer_pool(ratio)
            slots_per_full_page = self.page_size // ratio
            assert slots_per_full_page % indexer_pool.page_size == 0, (
                f"ratio-{ratio} index pages of {indexer_pool.page_size} slots do "
                f"not tile a full page of {slots_per_full_page} slots"
            )
            index_pages_per_full_page = slots_per_full_page // indexer_pool.page_size
            for buf in (
                *indexer_pool.index_k_buffer,
                *indexer_pool.index_scale_buffer,
            ):
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes * index_pages_per_full_page)

        return data_ptrs, data_lens, item_lens

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """GPU-compatible ``StateType.SWA`` component.

        On pre-Ascend 950 (EXPLICIT cache_mode), SWA KV, C4 attention state and C4
        indexer state retain separate buffers but share the same SWA page/state
        index.  On Ascend 950 (CYCLE cache_mode) the compressor addresses the C4 state
        ring by ``req_pool_idx`` instead of SWA page, so C4 state is excluded
        here and registered separately via :meth:`get_c4_state_buf_infos`.
        """
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        for buf in self.swa_kv_pool.kv_buffer:
            data_ptrs.append(buf.data_ptr())
            data_lens.append(buf.nbytes)
            item_lens.append(buf[0].nbytes)

        if not is_npu_arch35():
            for pools in (
                self.compress_state_pools,
                self.indexer_compress_state_pools,
            ):
                for pool in pools:
                    if pool is None or pool.ratio != 4:
                        continue
                    state = pool.kv_score_buffer.kv_score
                    data_ptrs.append(state.data_ptr())
                    data_lens.append(state.nbytes)
                    item_lens.append(state[0].nbytes * pool.ring_size)

        return data_ptrs, data_lens, item_lens

    def get_c4_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """C4 compress state ring (attention + indexer).

        Register one physical state row as an item.  PD peers can have
        different request-local ring sizes (for example prefill without MTP
        and decode with MTP), so payload indices map the same logical token
        positions into each peer's local ring independently.
        """
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        for pools in (self.compress_state_pools, self.indexer_compress_state_pools):
            for pool in pools:
                if pool is None or pool.ratio != 4:
                    continue
                state = pool.kv_score_buffer.kv_score
                data_ptrs.append(state.data_ptr())
                data_lens.append(state.nbytes)
                item_lens.append(state[0].nbytes)

        return data_ptrs, data_lens, item_lens

    def get_c128_kv_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        if self.c128_kv_pool is None:
            # DeepSeek-V4.1 has no ratio-128 compression.
            return [], [], []
        buffers = self.c128_kv_pool.kv_buffer
        return (
            [buf.data_ptr() for buf in buffers],
            [buf.nbytes for buf in buffers],
            [buf[0].nbytes for buf in buffers],
        )

    def get_state_cache(self, layer_id: int, from_indexer: bool) -> torch.Tensor:
        """FP32 ``[block_num, ring_size, 2*coff*D]`` view of this layer's
        kv+score buffer — the fused compressor op
        (``torch.ops.custom.compressor`` on Ascend 950 and ``torch.ops.npu.compressor``
        elsewhere)'s ``state_cache`` argument."""
        return self._get_state_pool(layer_id, from_indexer).state_cache_3d

    # ------------------------------------------------------------------
    # Generic KV accessors (community base raises NotImplementedError; CUDA uses
    # store_cache). AscendAttnBackend reads KV through these, routed to the right
    # sub-pool by compression ratio.
    # ------------------------------------------------------------------

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        ratio = item.compress_ratio
        if ratio == 0:
            return self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)]
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
        self.wait_layer_transfer(layer_id)
        # Index by PP-stage-local layer_id, not compress_layer_id (a per-bucket
        # counter that would collide across ratios). swa_kv_pool is sized
        # layer_num=stage_layer_num (only this PP stage's layers).
        kv = self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)]
        if loc is not None:
            kv = kv.flatten(0, 1)[loc]
        return kv

    def get_index_k(self, layer_id: int) -> torch.Tensor:
        """The layer's indexer-K payload buffer (PA_ND, packed fp4/fp8).

        ``DSV41NPUIndexerPool`` keeps K and its scales in two separate buffers
        per ratio, so they are reached through the layer's *ratio* index pool
        rather than through the token-to-KV pool itself.
        """
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        return self._indexer_pool(item.compress_ratio).get_index_k(
            item.compress_layer_id
        )

    def get_index_scale(self, layer_id: int) -> torch.Tensor:
        """The layer's indexer-K scale buffer (one uint8 E8M0 per 32 values)."""
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        return self._indexer_pool(item.compress_ratio).get_index_scale(
            item.compress_layer_id
        )

    def get_compress_buffer(
        self,
        layer_id: int,
        from_indexer: bool = False,
        loc: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Return the compressed KV buffer for a c4 / c128 layer.

        Routes to c4 / c128 kv_pool by layer compression ratio. Returns
        ``None`` for ratio == 0 (no compress KV exists). The
        from_indexer=True branch returns the dedicated quantized K buffer that
        ``torch.ops.custom.npu_quant_lightning_indexer`` consumes.
        """
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        if item.compress_ratio == 0:
            return None
        if from_indexer:
            indexer_pool = self._indexer_pool(item.compress_ratio)
            kv = indexer_pool.get_index_k(item.compress_layer_id)
        else:
            compress_pool = item.compress_kv_pool
            assert compress_pool is not None, "Missing compressed KV pool"
            kv = compress_pool.kv_buffer[item.compress_layer_id]
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
        # Index by PP-stage-local layer_id (see get_swa_buffer).
        pool = self.swa_kv_pool
        buf = pool.kv_buffer[self._swa_local_layer_id(layer_id)]
        if isinstance(pool, DSV41NPUSingleKVPool):
            # A V4.1 window row is FP8 E4M3 group 32 (544 B), not the V4 584-byte
            # arch35 row: the ds41 attention op reads ori_kv with all head_dim values
            # quantized and 16 BF16 scales per 512 (DSL README L184-L189).
            self._write_dsv41_packed_kv(
                buf=buf, loc=loc, cache=cache, layout=pool.layout
            )
            return
        if pool.layout is NPUDSV4KVLayout.V4_FP8:
            self._write_arch35_packed_kv(buf=buf, loc=loc, cache=cache)
            return
        buf_flat = buf.flatten(0, 1)  # (num_pages * page_size, 1, dim)
        # Caller (V4 MQALayer) may hand us cache shaped (T, dim); the buffer has
        # an explicit num_kv_heads=1 axis, so insert it.
        if cache.ndim == buf_flat.ndim - 1:
            cache = cache.unsqueeze(1)
        buf_flat[loc] = cache.to(buf_flat.dtype)

    def _write_arch35_packed_kv(
        self,
        *,
        buf: torch.Tensor,
        loc: torch.Tensor,
        cache: torch.Tensor,
    ) -> None:
        cache_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if cache.shape[-1] != cache_dim:
            raise RuntimeError(
                f"DSV4 Ascend 950 KV cache expects input last dim {cache_dim}, "
                f"got shape={tuple(cache.shape)}."
            )
        cache_2d = cache.reshape(-1, cache_dim).to(torch.bfloat16).contiguous()
        slot_mapping = loc.reshape(-1).contiguous()
        if cache_2d.shape[0] != slot_mapping.shape[0]:
            raise RuntimeError(
                "DSV4 Ascend 950 KV cache write expects one slot per token, got "
                f"{cache_2d.shape[0]} rows and {slot_mapping.shape[0]} slots."
            )
        if cache_2d.shape[0] == 0:
            return
        torch.ops.npu.kv_compress_epilog(
            buf.view(-1, 1, buf.shape[-1]),
            cache_2d,
            slot_mapping,
            quant_group_size=NPUDSV4KVLayout.V4_FP8.quant_group,
            quant_mode=NPUDSV4KVLayout.V4_FP8.quant_mode,
            round_scale_flag=True,
            layout=1,
        )

    def _write_dsv41_packed_kv(
        self,
        *,
        buf: torch.Tensor,
        loc: torch.Tensor,
        cache: torch.Tensor,
        layout: NPUDSV4KVLayout,
    ) -> None:
        """Quantize ``cache`` (bf16, last dim ``qk_nope + qk_rope``) into a V4.1
        pool row at flat token positions ``loc``.

        ``kv_compress_epilog_v2`` quantizes *every* head_dim value (there is no
        bf16 RoPE tail, unlike the V4 ``_write_arch35_packed_kv``), so RoPE must
        already have been applied by the caller. ``layout`` selects the side,
        matching :class:`NPUDSV4KVLayout` and the ds41 attention op's two
        arguments: window/``ori_kv`` = ``mxfp8_bf16`` group 32 (544 B),
        compressed/``cmp_kv`` = ``mxfp4_bf16`` group 16 (320 B).

        Reference calls: the vendor reference implementation
        ``modeling_deepseek.py::update_win_kv``
        (window) and ``compressor.py::update_compress_cache``.
        """
        cache_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if cache.shape[-1] != cache_dim:
            raise RuntimeError(
                f"DeepSeek-V4.1 KV cache expects input last dim {cache_dim}, "
                f"got shape={tuple(cache.shape)}."
            )
        cache_2d = cache.reshape(-1, cache_dim).to(torch.bfloat16).contiguous()
        slot_mapping = loc.reshape(-1).contiguous()
        if cache_2d.shape[0] != slot_mapping.shape[0]:
            raise RuntimeError(
                "DeepSeek-V4.1 KV cache write expects one slot per token, got "
                f"{cache_2d.shape[0]} rows and {slot_mapping.shape[0]} slots."
            )
        if cache_2d.shape[0] == 0:
            return
        quant_mode, quant_group_size = layout.quant
        torch.ops.custom.kv_compress_epilog_v2(
            buf.view(-1, buf.shape[-1]),
            cache_2d,
            slot_mapping,
            quant_group_size=quant_group_size,
            quant_mode=quant_mode,
            round_scale=True,
            x_scale=1.0,
        )

    def set_swa_key_buffer_radix_fused_norm_rope(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        kv: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        kv_out = torch_npu.npu_rms_norm(kv, kv_weight, eps)[0]

        rope_dim = freqs_cis.shape[-1] * 2

        from sglang.srt.hardware_backend.npu.dsv4.dsv4_rope import Dsv4NpuRoPE

        cos, sin = Dsv4NpuRoPE.for_freqs(freqs_cis).get_cos_sin(
            positions,
            kv_out.dtype,
            view_4d=True,
            allow_build=True,
            cache_dtype=torch.float32,
        )
        Dsv4NpuRoPE.apply_rotary_mul_inplace(
            kv_out.reshape(kv_out.shape[0], -1, kv_out.shape[-1]),
            None,
            cos,
            sin,
            qk_nope_dim=kv_out.shape[-1] - rope_dim,
        )

        safe_swa_loc = swa_loc.clamp_min(0).to(torch.int64)
        self.set_swa_buffer(
            layer_id,
            safe_swa_loc,
            kv_out,
        )

    def set_compress_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        kv: torch.Tensor,
        kv_scale: Optional[torch.Tensor],
        from_indexer: bool,
    ) -> None:
        # Routes to the indexer pool (from_indexer) or to the compressed KV pool
        # of this layer's ratio (4 / 128, plus 1 / 2 for DeepSeek-V4.1). NPU
        # bypasses CUDA fused_store_cache with direct writes into the PA_ND rows.
        item = self.layer_mapping[layer_id]
        ratio = item.compress_ratio
        compress_layer_id = item.compress_layer_id
        device_type = kv.device.type
        if from_indexer:
            indexer_pool = self._indexer_pool(ratio)
            if device_type == "npu":
                assert indexer_pool.has_npu_storage, (
                    "NPU index buffers not allocated — pool was init'd on CUDA?"
                )
                indexer_pool.set_index_k_scale(compress_layer_id, loc, kv, kv_scale)
                return
            if kv_scale is None:
                indexer_pool.set_index_fused(compress_layer_id, loc, kv)
                return
            indexer_pool.set_index_k_scale_buffer(compress_layer_id, loc, kv, kv_scale)
            return
        compress_pool = item.compress_kv_pool
        assert compress_pool is not None, "Missing compressed KV pool"
        if device_type == "npu":
            # PA_ND layout: kv_buffer[layer_id] shape = (num_pages, page_size,
            # 1, kv_dim). Flatten (num_pages, page_size) and index by `loc`.
            buf = compress_pool.kv_buffer[compress_layer_id]
            if isinstance(compress_pool, DSV41NPUSingleKVPool):
                # DeepSeek-V4.1 compressed KV: a bf16 latent goes in and a
                # fully-quantized row comes out, so no separate scale tensor.
                assert kv_scale is None, (
                    "the DeepSeek-V4.1 compressed KV is quantized by "
                    "kv_compress_epilog_v2 from a bf16 latent, not from a "
                    "separate scale tensor"
                )
                self._write_dsv41_packed_kv(
                    buf=buf, loc=loc, cache=kv, layout=compress_pool.layout
                )
                return
            if compress_pool.layout is NPUDSV4KVLayout.V4_FP8:
                self._write_arch35_packed_kv(buf=buf, loc=loc, cache=kv)
                return
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
        # The indexer scale is fp16 on pre-Ascend 950 parts and fp32 on Ascend 950.
        assert from_indexer, "only indexer compress pool has dequant scale"
        item = self.layer_mapping[layer_id]
        indexer_pool = self._indexer_pool(item.compress_ratio)
        return indexer_pool.get_index_scale(item.compress_layer_id)
