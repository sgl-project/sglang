from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Literal, NamedTuple, Optional, Tuple

import torch

from sglang.jit_kernel.deepseek_v4 import fused_store_cache
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4 import (
    index_buf_accessor as dsv4_index_buf_accessor,
)
from sglang.srt.layers.attention.dsv4.index_buf_accessor import NopeFp8RopeBf16Pack
from sglang.srt.layers.attention.nsa import index_buf_accessor
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import ceil_div, is_npu

_is_npu = is_npu()

logger = logging.getLogger(__name__)

ONLINE_C128 = envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()


def get_compress_state_ring_size(
    compress_ratio: int, is_speculative: bool = False
) -> int:
    assert compress_ratio in [4, 128], f"Unsupported {compress_ratio = }"
    # Online c128 keeps a single (max, sum, kv) state per index instead of a
    # 128-slot ring buffer of raw tokens, so ring_size collapses to 1. Online
    # is incompatible with speculative decode for now.
    if compress_ratio == 128 and ONLINE_C128:
        assert not is_speculative, "online c128 does not support MTP"
        return 1
    if is_speculative:
        return 16 if compress_ratio == 4 else 256
    else:
        return 8 if compress_ratio == 4 else 128


class DeepSeekV4SingleKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim

        self.scale_pad = 1
        self.quantize_block_size = 64
        self.rope_storage_dtype = torch.bfloat16
        self.k_with_scale_buffer_dtype = torch.int8
        self._create_buffers()

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.kv_buffer = [
                    self.create_buffer(
                        num_pages=(self.size + self.page_size + 1) // self.page_size,
                    )
                    for _ in range(self.layer_num)
                ]

    def get_bytes_per_token(self) -> int:
        dim_per_token = (
            self.qk_nope_head_dim
            + self.qk_rope_head_dim * self.rope_storage_dtype.itemsize
            + self.qk_nope_head_dim // self.quantize_block_size
            + self.scale_pad
        )
        return dim_per_token

    def create_buffer(self, *, num_pages: int):
        # NPU bf16 mode: allocate the same PA_ND layout that
        # iforgetmyname/dsv4_release uses for swa_kv_pool —
        # (num_pages, page_size, num_kv_heads=1, dim) where dim packs
        # K_nope (qk_nope_head_dim) and K_rope (qk_rope_head_dim) as bf16.
        # That's the shape npu_sparse_attn_sharedkv expects with
        # layout_kv="PA_ND". The CUDA fp8-packed-bytes layout is preserved
        # for non-NPU paths.
        is_npu_bf16 = _is_npu and self.store_dtype == torch.bfloat16
        if is_npu_bf16:
            kv_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            self.kv_cache_total_dim = kv_dim
            return torch.zeros(
                num_pages,
                self.page_size,
                1,
                kv_dim,
                dtype=torch.bfloat16,
                device=self.device,
            )

        bytes_per_token = self.get_bytes_per_token()
        self.kv_cache_total_dim = bytes_per_token
        bytes_per_page_non_padded = self.page_size * bytes_per_token
        self.bytes_per_page_padded = ceil_div(bytes_per_page_non_padded, 576) * 576

        assert bytes_per_token == 448 + 64 * 2 + 8, (
            "DSV4 KV layout: qk_nope_head_dim FP8 (448) + qk_rope_head_dim BF16 "
            "(64*2) + nope FP8 scales + scale_pad = 584 bytes/token"
        )
        assert self.store_dtype == torch.uint8

        return torch.zeros(
            num_pages,
            self.bytes_per_page_padded,
            dtype=self.store_dtype,
            device=self.device,
        )

    def set_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ):
        dsv4_index_buf_accessor.SetKAndS.execute(
            pool=self,
            buf=self.kv_buffer[layer_id],
            loc=loc,
            nope_fp8_rope_bf16_pack=cache_nope_fp8_rope_bf16_pack,
        )

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        return fused_store_cache(
            input=cache_k,
            cache=self.kv_buffer[layer_id],
            indices=loc,
            page_size=self.page_size,
            type="flashmla",
        )

    def get_key_buffer(self, layer_id: int):
        return self.kv_buffer[layer_id]

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError("Use get_key_buffer instead.")

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Use get_key_buffer instead.")


class HiSparseC4DevicePool(DeepSeekV4SingleKVPool):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: int | None = None,
        end_layer: int | None = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.compress_ratio = 4

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_from_full_to_compressed(self, full_indices: torch.Tensor):
        mask = (full_indices + 1) % self.compress_ratio == 0
        compressed_indices = full_indices[mask] // self.compress_ratio
        return compressed_indices

    def translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices].to(
            torch.int32
        )

    def _translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def translate_loc_from_full_to_hisparse_device(self, full_indices: torch.Tensor):
        return self._translate_loc_to_hisparse_device(
            self.translate_loc_from_full_to_compressed(full_indices)
        )

    def set_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_key_buffer(layer_id, loc, cache_nope_fp8_rope_bf16_pack)

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        loc = self.translate_loc_to_hisparse_device(loc)
        return super().set_key_buffer_fused(layer_id, loc, cache_k)

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseC4DevicePool does not support get_cpu_copy")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseC4DevicePool does not support load_cpu_copy")


class DeepSeekV4IndexerPool(KVCache):
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        kernel_page_size: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.index_head_dim = index_head_dim
        # Kernel-view page size — what npu_quant_lightning_indexer expects
        # cmp_kv.shape[1] to equal (= global page_size, typically 256).
        # main's V4 pool passes c4_page_size = page_size // 4 = 64 as
        # `page_size` here for the CUDA layout; on NPU we need page_size=256
        # to match ori_kv. If unset, default to self.page_size (CUDA backward
        # compat).
        self.kernel_page_size = (
            kernel_page_size if kernel_page_size is not None else page_size
        )

        self._create_buffer()

    def _create_buffer(self):
        num_scales_per_token = self.index_head_dim // self.quant_block_size
        page_bytes = self.page_size * self.index_head_dim
        page_bytes += self.page_size * num_scales_per_token * 4
        num_pages = (self.size + self.page_size + 1) // self.page_size
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.index_k_with_scale_buffer = [
                    torch.zeros(
                        num_pages,
                        page_bytes,
                        dtype=self.index_k_with_scale_buffer_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

        # NPU layout: separate int8 K buffer + float16 scale buffer per
        # layer, matching iforgetmyname/dsv4_release NPUSingleBufferTokenToKVPool
        # so torch_npu.npu_scatter_nd_update_ + torch.ops.custom.npu_quant_
        # lightning_indexer can read/write directly without unpacking the
        # CUDA-only uint8 packed layout. CUDA path keeps using
        # `index_k_with_scale_buffer` above; NPU path uses the buffers below.
        # ONLY allocate when SGLANG_DSV4_NPU_REAL_COMPRESSOR is on — these
        # buffers add ~570 MB total which would otherwise eat into the KV
        # pool budget for Tier 1 baseline launches.
        from sglang.srt.utils import is_npu as _is_npu_check
        from sglang.srt.environ import envs as _envs

        self._npu_buffers_present = (
            _is_npu_check()
            and _envs.SGLANG_DSV4_NPU_REAL_COMPRESSOR.get()
        )
        if self._npu_buffers_present:
            # NPU buffer uses GLOBAL kernel_page_size (= 256), not the
            # pool's per-ratio page_size (= 64 for c4 indexer pool). This
            # makes cmp_kv.shape[1] match ori_kv.shape[1] = global page_size,
            # which aclnnSparseAttnSharedkv / npu_quant_lightning_indexer
            # require. num_pages is recomputed for the kernel page size.
            npu_num_pages = (self.size + self.kernel_page_size + 1) // self.kernel_page_size
            self.npu_index_k_buffer = [
                torch.zeros(
                    npu_num_pages,
                    self.kernel_page_size,
                    1,
                    self.index_head_dim,
                    dtype=torch.int8,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.npu_index_scale_buffer = [
                torch.zeros(
                    npu_num_pages,
                    self.kernel_page_size,
                    1,
                    1,
                    dtype=torch.float16,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
        else:
            self.npu_index_k_buffer = None
            self.npu_index_scale_buffer = None

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        return self.index_k_with_scale_buffer[layer_id]

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        buf = self.index_k_with_scale_buffer[layer_id]
        return index_buf_accessor.GetKAndS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )

    def set_index_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        return fused_store_cache(
            input=cache_k,
            cache=self.index_k_with_scale_buffer[layer_id - self.start_layer],
            indices=loc,
            page_size=self.page_size,
            type="indexer",
        )


class DeepSeekV4LayerItem(NamedTuple):
    compress_ratio: Literal[0, 4, 128]
    compress_layer_id: int
    compress_kv_pool: Optional[DeepSeekV4SingleKVPool] = None


class DeepSeekV4TokenToKVPool(BaseSWAKVPool):

    def __init__(
        self,
        max_num_reqs: int,
        swa_size: int,
        c4_size: int,
        c128_size: int,
        c4_state_pool_size: int,
        c128_state_pool_size: int,
        page_size: int,
        swa_page_size: int,
        dtype: torch.dtype,
        state_dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        indexer_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        compression_ratios: List[int],
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_hisparse: bool = False,
    ):
        super().__init__(
            swa_size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        c4_logical_size = c128_size * 32

        logger.info(
            "Initialize DeepSeekV4TokenToKVPool with "
            f"{max_num_reqs=} {swa_size=} {c4_size=} "
            f"{c4_logical_size=} {c128_size=} "
            f"{c4_state_pool_size=} {c128_state_pool_size=}"
        )

        self.max_num_reqs = max_num_reqs
        self.c4_size = c4_size
        self.c4_logical_size = c4_logical_size
        self.c128_size = c128_size
        self.c4_state_pool_size = c4_state_pool_size
        self.c128_state_pool_size = c128_state_pool_size
        self.state_dtype = state_dtype
        self.compression_ratios = compression_ratios

        assert page_size % swa_page_size == 0

        self.swa_size = swa_size
        self.swa_window_size = swa_page_size
        self.swa_page_size = swa_page_size
        self.scale_pad = 1

        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.indexer_head_dim = indexer_head_dim

        c4_layer_num = sum(1 for r in compression_ratios if r == 4)
        c128_layer_num = sum(1 for r in compression_ratios if r == 128)
        c4_page_size = page_size // 4
        c128_page_size = page_size // 128
        self.swa_kv_pool = DeepSeekV4SingleKVPool(
            swa_size,
            swa_page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
        )

        c4_kv_pool_type = DeepSeekV4SingleKVPool
        if enable_hisparse:
            c4_kv_pool_type = HiSparseC4DevicePool
        self.c4_kv_pool = c4_kv_pool_type(
            c4_size,
            c4_page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            c4_layer_num,
            device,
            enable_memory_saver,
        )

        self.c128_kv_pool = DeepSeekV4SingleKVPool(
            c128_size,
            c128_page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            c128_layer_num,
            device,
            enable_memory_saver,
        )

        self.c4_indexer_kv_pool = DeepSeekV4IndexerPool(
            self.c4_logical_size,
            c4_page_size,
            dtype,
            indexer_head_dim,
            c4_layer_num,
            device,
            enable_memory_saver,
            kernel_page_size=page_size,  # global; NPU buffers use this
        )

        self._init_compressed_layer_mapping()

        self._init_paged_compress_states(enable_memory_saver)

        self._should_cache_swa = envs.SGLANG_OPT_CACHE_SWA_TRANSLATION.get()

        # Step-5c per-req c4/c128 slab allocator. Mirrors iforgetmyname's
        # req_to_token_c4 / req_to_token_c128 (own allocator that gives each
        # request its own contiguous slot range in c{N}_kv_pool, keyed by
        # compressed-seq position not raw-kv position) — but stored INSIDE
        # the V4 token-to-kv pool instead of the request pool to avoid
        # scheduler-side surgery. Each req_pool_idx i gets a fixed slab of
        # `max_pages_c{N}_per_req` pages starting at i * max_per_req.
        # Per-page granularity uses the global page_size (matches our
        # _forward_compressed cmp_kv reshape view).
        c4_n_pages_kernel = c4_size // page_size  # kernel-view num pages
        c128_n_pages_kernel = c128_size // page_size
        # Cap per-req max pages so all max_num_reqs reqs fit; round down.
        # NOTE: this is an *average* slab size. A single request whose
        # compressed token count exceeds max_pages_c{N}_per_req * page_size
        # will overflow its slab and corrupt or OOB-read neighbour slabs.
        # This is safe in the current sizing (c4_size / c128_size are
        # provisioned so that even at max concurrency a max-context req
        # fits), but low-concurrency long-context workloads (e.g.
        # max_num_reqs=1 with context_length close to swa_size) need a
        # real per-req allocator. Logged below + asserted at write time
        # so the failure mode is loud rather than silent corruption.
        self.max_pages_c4_per_req = max(1, c4_n_pages_kernel // max_num_reqs)
        self.max_pages_c128_per_req = max(1, c128_n_pages_kernel // max_num_reqs)
        logger.info(
            "DeepSeekV4TokenToKVPool per-req compressed-slab caps: "
            f"c4={self.max_pages_c4_per_req} pages "
            f"(={self.max_pages_c4_per_req * page_size} c4 tokens, "
            f"≈{self.max_pages_c4_per_req * page_size * 4} raw tokens), "
            f"c128={self.max_pages_c128_per_req} pages "
            f"(={self.max_pages_c128_per_req * page_size} c128 tokens, "
            f"≈{self.max_pages_c128_per_req * page_size * 128} raw tokens). "
            "A single request exceeding these limits will overflow its slab."
        )
        # req_to_token_c{N}_pages[req_idx, k] = kernel-view page index in
        # c{N}_kv_pool for the k-th compressed-token-page of request `req_idx`.
        self.req_to_token_c4_pages = (
            torch.arange(max_num_reqs * self.max_pages_c4_per_req, dtype=torch.int32)
            .view(max_num_reqs, self.max_pages_c4_per_req)
            .to(device)
        )
        self.req_to_token_c128_pages = (
            torch.arange(
                max_num_reqs * self.max_pages_c128_per_req, dtype=torch.int32
            )
            .view(max_num_reqs, self.max_pages_c128_per_req)
            .to(device)
        )

    def get_req_to_token_c_pages(self, compress_ratio: int) -> torch.Tensor:
        if compress_ratio == 4:
            return self.req_to_token_c4_pages
        if compress_ratio == 128:
            return self.req_to_token_c128_pages
        raise ValueError(f"unsupported compress_ratio={compress_ratio}")

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def get_ring_size(self, compress_ratio: int) -> int:
        server_args = get_global_server_args()
        is_speculative = server_args.speculative_algorithm is not None
        return get_compress_state_ring_size(compress_ratio, is_speculative)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None

        return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)

    def set_swa_loc(self, loc: torch.Tensor) -> None:
        # No-op: SWAKVPool's set_swa_loc precomputes SWA-translated loc once per
        # forward batch for set_kv_buffer to read via self.swa_loc. DSV4 has its
        # own equivalent cache via `_should_cache_swa + cached_loc` (in
        # set_swa_key_buffer_radix_fused), so we ignore main's precomputed loc.
        pass

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        for bufs in [
            self.c4_kv_pool.kv_buffer,
            self.c4_indexer_kv_pool.index_k_with_scale_buffer,
            self.c128_kv_pool.kv_buffer,
        ]:
            for buf in bufs:
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)

        return data_ptrs, data_lens, item_lens

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        for buf in self.swa_kv_pool.kv_buffer:
            assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
            data_ptrs.append(buf.data_ptr())
            data_lens.append(buf.nbytes)
            item_lens.append(buf[0].nbytes)

        for pools in [
            self.compress_state_pools,
            self.indexer_compress_state_pools,
        ]:
            for pool in pools:
                if pool is None:
                    continue
                t = pool.kv_score_buffer.kv_score
                assert t.ndim == 2, f"expected 2D buffer, got {t.ndim}D"
                data_ptrs.append(t.data_ptr())
                data_lens.append(t.nbytes)
                item_lens.append(t[0].nbytes * pool.ring_size)

        return data_ptrs, data_lens, item_lens

    def _init_paged_compress_states(self, enable_memory_saver: bool):
        c4_state_pool_size = self.c4_state_pool_size
        c128_state_pool_size = self.c128_state_pool_size
        self.compress_state_pools: List[CompressStatePool] = []
        self.indexer_compress_state_pools: List[CompressStatePool] = []

        for ratio in self.compression_ratios:
            overlap = ratio == 4
            compress_state_pool = indexer_compress_state_pool = None
            size = c4_state_pool_size if ratio == 4 else c128_state_pool_size
            # ratio == 1 (V4-Flash dense edge layers) is treated like
            # ratio == 0 here: no compress-state pool, no ring buffer.
            # get_compress_state_ring_size only handles 4 and 128.
            has_compress_state = ratio in (4, 128)
            ring_size = self.get_ring_size(ratio) if has_compress_state else 0
            if has_compress_state:
                compress_state_pool = CompressStatePool(
                    size=size,
                    ring_size=ring_size,
                    overlap=overlap,
                    head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
                    dtype=self.state_dtype,
                    device=self.device,
                    enable_memory_saver=enable_memory_saver,
                    ratio=ratio,
                    online=(ratio == 128 and ONLINE_C128),
                )

            if ratio == 4:
                indexer_compress_state_pool = CompressStatePool(
                    size=size,
                    ring_size=ring_size,
                    overlap=overlap,
                    head_dim=self.indexer_head_dim,
                    device=self.device,
                    dtype=self.state_dtype,
                    enable_memory_saver=enable_memory_saver,
                    ratio=ratio,
                )

            self.compress_state_pools.append(compress_state_pool)
            self.indexer_compress_state_pools.append(indexer_compress_state_pool)

    def _init_compressed_layer_mapping(self):
        c1_cnt, c4_cnt, c128_cnt = 0, 0, 0
        self.layer_mapping: List[DeepSeekV4LayerItem] = []

        for ratio in self.compression_ratios:
            # V4-Flash adds compress_ratio=1 for dense edge layers (e.g.
            # [1, 1, 4, 128, ..., 1] for the 43-layer Flash variant). Both
            # 0 and 1 mean "this layer has no compression / compressor", so
            # they share the same uncompressed mapping bucket.
            if ratio in (0, 1):
                self.layer_mapping.append(
                    DeepSeekV4LayerItem(
                        compress_ratio=ratio,
                        compress_layer_id=c1_cnt,
                    )
                )
                c1_cnt += 1
            elif ratio == 4:
                self.layer_mapping.append(
                    DeepSeekV4LayerItem(
                        compress_ratio=4,
                        compress_layer_id=c4_cnt,
                        compress_kv_pool=self.c4_kv_pool,
                    )
                )
                c4_cnt += 1
            elif ratio == 128:
                self.layer_mapping.append(
                    DeepSeekV4LayerItem(
                        compress_ratio=128,
                        compress_layer_id=c128_cnt,
                        compress_kv_pool=self.c128_kv_pool,
                    )
                )
                c128_cnt += 1
            else:
                raise ValueError(f"Unsupported compression ratio: {ratio}")

    def get_attention_compress_states(self, layer_id: int) -> CompressStatePool:
        compress_state_pool = self.compress_state_pools[layer_id]
        assert (
            compress_state_pool is not None
        ), "Only c4/c128 layers have attention states."
        return compress_state_pool

    def get_indexer_compress_states(self, layer_id: int) -> CompressStatePool:
        indexer_compress_state_pool = self.indexer_compress_state_pools[layer_id]
        assert (
            indexer_compress_state_pool is not None
        ), "Only c4 layers have indexer states."
        return indexer_compress_state_pool

    def get_swa_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.swa_kv_pool.get_key_buffer(layer_id)

    def set_swa_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        self.swa_kv_pool.set_key_buffer(layer_id, loc, cache_nope_fp8_rope_bf16_pack)

    def get_extra_key_buffer(self, layer_id: int) -> torch.Tensor | None:
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.get_key_buffer(compress_layer_id)

    def set_extra_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        compress_kv_pool.set_key_buffer(
            compress_layer_id, loc, cache_nope_fp8_rope_bf16_pack
        )

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.get_index_k_with_scale_buffer(compress_layer_id)

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.get_index_k_scale_buffer(
            compress_layer_id, seq_len, page_indices
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        self.c4_indexer_kv_pool.set_index_k_scale_buffer(
            compress_layer_id, loc, index_k, index_k_scale
        )

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        # Route to the appropriate sub-pool based on the layer's compression
        # ratio. The CUDA path uses set_swa_key_buffer_radix for c4/c128 +
        # store_cache shenanigans; AscendAttnBackend.forward_extend reads
        # KV through this generic accessor and would otherwise hit
        # NotImplementedError. ratio in (0, 1) -> swa pool (the dense /
        # uncompressed layers); ratio == 4 / 128 -> c4 / c128 pool.
        item = self.layer_mapping[layer_id]
        ratio = item.compress_ratio
        if ratio in (0, 1):
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
        """Mirrors iforgetmyname's SWAC4C128KVPool.get_swa_buffer.

        Returns the SWA layer's KV cache in PA_ND layout
        (num_pages, page_size, num_kv_heads=1, dim). When ``loc`` is given,
        flatten across (num_pages, page_size) and gather the matching
        tokens — shape becomes (num_tokens, 1, dim).
        """
        # HOTFIX: index swa_kv_pool.kv_buffer by raw layer_id, not by
        # item.compress_layer_id. compress_layer_id is a per-bucket counter
        # (c1_cnt / c4_cnt / c128_cnt each starting at 0), so writes from
        # different ratios COLLIDE on the same kv_buffer slot. Eg layer 0
        # (c1_cnt=0), layer 2 (c4_cnt=0), and layer 3 (c128_cnt=0) all hit
        # kv_buffer[0] and overwrite each other; decode then reads the last
        # writer's K instead of the layer's own K. swa_kv_pool is sized with
        # layer_num=total_layers so per-layer slots already exist; just
        # index them directly. Verified bit-perfect with iforgetmyname for
        # the first 5 generated tokens of `Four ` after this fix.
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
        """Mirrors iforgetmyname's SWAC4C128KVPool.get_compress_buffer.

        Routes to c4 / c128 kv_pool by layer compression ratio. Returns
        ``None`` for ratio in (0, 1) (no compress KV exists). With ``loc``
        gathers tokens like get_swa_buffer. The from_indexer=True branch
        on NPU returns the dedicated int8 K buffer that
        torch.ops.custom.npu_quant_lightning_indexer consumes; CUDA path
        keeps returning the packed uint8 buffer.
        """
        item = self.layer_mapping[layer_id]
        if item.compress_ratio == 4:
            if from_indexer:
                if (
                    self.c4_indexer_kv_pool.npu_index_k_buffer is not None
                ):  # NPU: return the dedicated int8 K buffer
                    kv = self.c4_indexer_kv_pool.npu_index_k_buffer[
                        item.compress_layer_id
                    ]
                else:
                    kv = self.c4_indexer_kv_pool.kv_buffer[item.compress_layer_id]
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

        ``cache`` shape: (num_tokens, num_kv_heads=1, dim) — same layout the
        attention kernel reads. The buffer view is (num_pages, page_size, 1,
        dim) so we flatten the first two dims and index_put.
        """
        # HOTFIX: see get_swa_buffer comment — index by raw layer_id, not
        # by item.compress_layer_id, otherwise c4/c128 bucket counters
        # collide with the c1 bucket and writes from different ratios
        # overwrite each other in swa_kv_pool.kv_buffer.
        buf = self.swa_kv_pool.kv_buffer[layer_id]
        buf_flat = buf.flatten(0, 1)  # (num_pages * page_size, 1, dim)
        # Caller (V4 MQALayer) hands us cache shaped (T, dim) — the kv tensor
        # before it splits heads. The buffer has an explicit num_kv_heads=1
        # axis, so insert it.
        if cache.ndim == buf_flat.ndim - 1:
            cache = cache.unsqueeze(1)
        buf_flat[loc] = cache.to(buf_flat.dtype)

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def set_swa_key_buffer_radix(
        self,
        layer_id: int,
        raw_loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        swa_loc = self.translate_loc_from_full_to_swa(raw_loc)
        self.swa_kv_pool.set_key_buffer(
            layer_id, swa_loc, cache_nope_fp8_rope_bf16_pack
        )

    def get_swa_key_buffer_radix(self, layer_id: int) -> torch.Tensor:
        return self.swa_kv_pool.get_key_buffer(layer_id)

    def set_swa_key_buffer_radix_fused(
        self,
        layer_id: int,
        raw_loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        if self._should_cache_swa:
            if layer_id == 0:
                self.cached_loc = self.translate_loc_from_full_to_swa(raw_loc)
            swa_loc = self.cached_loc
        else:
            swa_loc = self.translate_loc_from_full_to_swa(raw_loc)
        return self.swa_kv_pool.set_key_buffer_fused(layer_id, swa_loc, cache_k)

    def set_extra_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.set_key_buffer_fused(compress_layer_id, loc, cache_k)

    def set_index_k_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.set_index_fused(compress_layer_id, loc, cache_k)

    # ------------------------------------------------------------------
    # NPU port hooks — used by dsv4/{compressor,indexer}.py forward_npu
    # (gated by env SGLANG_DSV4_NPU_REAL_COMPRESSOR). Mirror the
    # iforgetmyname/dsv4_release SWAC4C128KVPool API on top of main's
    # CompressStatePool / DeepSeekV4SingleKVPool / DeepSeekV4IndexerPool.
    #
    # CompressStatePool stores a single fused [kv | score] tensor of shape
    # (size, 2*coff*head_dim); split + cat is just a last-dim slice.
    # ------------------------------------------------------------------

    def set_compress_state_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        kv: torch.Tensor,
        score: torch.Tensor,
        scale: Optional[torch.Tensor],
        from_indexer: bool,
    ) -> None:
        # iforgetmyname stores kv and score side-by-side in a fused buffer;
        # main's KVAndScore.kv_score is [..., 2*coff*head_dim] = [kv | score]
        # (kv first half, score second half). We accept (T, 1, coff*head_dim)
        # tensors and write them into the matching half-slices.
        _ = scale  # int8 scale not modelled — current state-pool layout is float32
        pool = (
            self.indexer_compress_state_pools[layer_id]
            if from_indexer
            else self.compress_state_pools[layer_id]
        )
        assert pool is not None, (
            f"layer_id={layer_id} has no {'indexer' if from_indexer else 'attention'} "
            f"compress state pool — only c4/c128 layers do"
        )
        kv_score = pool.kv_score_buffer.kv_score  # (size, 2*coff*head_dim)
        last_dim = kv_score.shape[-1]
        half = last_dim // 2
        # Caller hands (T, 1, coff*head_dim); coff*head_dim equals half.
        kv_view = kv.reshape(-1, half).to(kv_score.dtype)
        score_view = score.reshape(-1, half).to(kv_score.dtype)
        kv_score[loc, :half] = kv_view
        kv_score[loc, half:] = score_view

    def get_compress_state_buffer(
        self,
        layer_id: int,
        from_indexer: bool,
        kv_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pool = (
            self.indexer_compress_state_pools[layer_id]
            if from_indexer
            else self.compress_state_pools[layer_id]
        )
        assert pool is not None
        kv_score = pool.kv_score_buffer.kv_score
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
        # Routes to:
        #   from_indexer=True               → c4_indexer_kv_pool (int8 + scale)
        #   from_indexer=False, ratio=4     → c4_kv_pool
        #   from_indexer=False, ratio=128   → c128_kv_pool
        #
        # On NPU we bypass the CUDA fused_store_cache (tvm_ffi-compiled) and
        # do direct tensor writes against the bf16 buffer. The CUDA
        # set_key_buffer_fused / set_index_fused chain stays the default for
        # other devices.
        ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        device_type = kv.device.type
        if from_indexer:
            assert ratio == 4, f"indexer only on c4 layers, got ratio={ratio}"
            if device_type == "npu":
                # Step 5d: write int8 K + float16 scale into the dedicated
                # NPU buffer pair via torch_npu.npu_scatter_nd_update_
                # (mirrors iforgetmyname NPUSingleBufferTokenToKVPool.
                # set_kv_buffer). Both kv and kv_scale come from
                # _compressor_epilog_npu's torch_npu.npu_dynamic_quant
                # output (kv: int8 [T, dim], kv_scale: float16 [T, 1]).
                import torch_npu  # local: NPU only

                assert self.c4_indexer_kv_pool.npu_index_k_buffer is not None, (
                    "NPU index buffers not allocated — pool was init'd on CUDA?"
                )
                k_buf = self.c4_indexer_kv_pool.npu_index_k_buffer[
                    compress_layer_id
                ]
                scale_buf = self.c4_indexer_kv_pool.npu_index_scale_buffer[
                    compress_layer_id
                ]
                index_head_dim = self.c4_indexer_kv_pool.index_head_dim
                loc_long = loc.view(-1, 1).long()
                kv_view = kv.to(torch.int8).view(-1, 1, index_head_dim)
                torch_npu.npu_scatter_nd_update_(
                    k_buf.view(-1, 1, index_head_dim),
                    loc_long,
                    kv_view,
                )
                if kv_scale is not None:
                    scale_view = kv_scale.to(torch.float16).view(-1, 1, 1)
                    torch_npu.npu_scatter_nd_update_(
                        scale_buf.view(-1, 1, 1),
                        loc_long,
                        scale_view,
                    )
                return
            if kv_scale is None:
                self.c4_indexer_kv_pool.set_index_fused(
                    compress_layer_id, loc, kv
                )
                return
            self.c4_indexer_kv_pool.set_index_k_scale_buffer(
                compress_layer_id, loc, kv, kv_scale
            )
            return
        compress_pool = (
            self.c4_kv_pool if ratio == 4 else self.c128_kv_pool
        )
        if device_type == "npu":
            # PA_ND layout: kv_buffer[layer_id] shape = (num_pages, page_size,
            # 1, kv_dim). Flatten (num_pages, page_size) and index by `loc`
            # (a flat token slot inside the compress kv pool).
            buf = compress_pool.kv_buffer[compress_layer_id]
            buf_flat = buf.flatten(0, 1)
            kv_view = kv.to(buf_flat.dtype)
            if kv_view.ndim == buf_flat.ndim - 1:
                kv_view = kv_view.unsqueeze(1)
            buf_flat[loc] = kv_view
            return
        return compress_pool.set_key_buffer_fused(compress_layer_id, loc, kv)

    def translate_kv_loc_to_compress_state_loc(
        self,
        kv_loc: torch.Tensor,
        compress_ratio: int,
    ) -> torch.Tensor:
        # Same arithmetic as compressor.get_paged_compress_metadata.get_raw_loc
        # at compressor.py L226-233, but exposed as a callable so both the NPU
        # forward_npu paths (Compressor / C4Indexer) and the V4 NPU backend's
        # init_forward_metadata can compute a flat state-pool index from a
        # full-kv-pool slot id.
        swa_loc = self.translate_loc_from_full_to_swa(kv_loc)
        ring_size = self.get_ring_size(compress_ratio)
        swa_page = swa_loc // self.swa_page_size
        state_loc = swa_page * ring_size + (swa_loc % ring_size)
        return state_loc.to(torch.int32)

    def get_compress_dequant_scale_buffer(
        self,
        layer_id: int,
        from_indexer: bool,
    ) -> torch.Tensor:
        # Returns float16 dequant scale buffer. NPU has a dedicated
        # `npu_index_scale_buffer` allocated alongside the int8 K buffer
        # (matches iforgetmyname/dsv4_release NPUSingleBufferTokenToKVPool.
        # dequant_scale_buffer). CUDA still uses the packed uint8 layout.
        assert from_indexer, "only indexer compress pool has dequant scale"
        compress_layer_id = self.layer_mapping[layer_id].compress_layer_id
        if self.c4_indexer_kv_pool.npu_index_scale_buffer is not None:
            return self.c4_indexer_kv_pool.npu_index_scale_buffer[
                compress_layer_id
            ]
        return self.c4_indexer_kv_pool.get_index_k_with_scale_buffer(
            compress_layer_id
        )
