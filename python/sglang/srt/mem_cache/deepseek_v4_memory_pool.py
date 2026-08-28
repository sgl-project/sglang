from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, NamedTuple, Optional, Sequence, Tuple

import torch

from sglang.kernels.ops.attention.dsa import index_buf_accessor
from sglang.kernels.ops.attention.dsv4 import (
    clear_unaccepted_c128_draft_states,
    fused_k_norm_rope_flashmla,
    fused_store_cache,
)
from sglang.kernels.ops.attention.dsv4 import (
    index_buf_accessor as dsv4_index_buf_accessor,
)
from sglang.kernels.ops.attention.dsv4.index_buf_accessor import NopeFp8RopeBf16Pack
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import layout
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.runtime_context import get_exec, get_spec
from sglang.srt.utils import ceil_div, is_hip

logger = logging.getLogger(__name__)

_is_hip = is_hip()

ONLINE_C128 = not _is_hip and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()


def get_dsv4_indexer_bytes_per_token(index_head_dim: int, use_fp4_indexer: bool) -> int:
    """Return payload and quant-scale bytes for one compressed indexer token."""
    if use_fp4_indexer:
        return index_head_dim // 2 + index_head_dim // 32
    return index_head_dim + index_head_dim // 128 * 4


def get_compress_state_ring_size(
    compress_ratio: int, is_speculative: bool = False
) -> int:
    assert compress_ratio in [4, 128], f"Unsupported {compress_ratio = }"
    # Online C128 stores one (max, sum, kv) state per index;
    # speculative decoding requires the experimental online C128 MTP path.
    if compress_ratio == 128 and ONLINE_C128:
        if is_speculative and not envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get():
            raise AssertionError("online c128 does not support MTP")
        return 1
    if is_speculative:
        return 16 if compress_ratio == 4 else 256
    else:
        return 8 if compress_ratio == 4 else 128


def get_compress_state_write_pad(compress_ratio: int, ring_size: int) -> int:
    # Draft-token capacity must match mtp_pad in c_plan.cuh;
    # a non-speculative ring has no write padding.
    window_size = compress_ratio * (2 if compress_ratio == 4 else 1)
    return ring_size - window_size + 2 if ring_size > window_size else 0


def get_swa_ring_size(sliding_window: int, is_speculative: bool = False) -> int:
    # A verify batch writes its draft tokens ahead of the committed position.
    spec_extra = (get_spec().speculative_num_draft_tokens - 1) if is_speculative else 0
    return sliding_window + spec_extra


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
        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)

        return self.kv_buffer[layer_id]

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError("Use get_key_buffer instead.")

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Use get_key_buffer instead.")


class DeepSeekV4UniformFP8KVPool(DeepSeekV4SingleKVPool):
    """Uniform 512-dim FP8 (e4m3) variant of the DSv4 single-KV pool.

    Each token is 448 NoPE + 64 RoPE contiguous e4m3 values without in-cache
    scales or per-page padding. The backend supplies the dequant scale.
    """

    def get_bytes_per_token(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def create_buffer(self, *, num_pages: int):
        bytes_per_token = self.get_bytes_per_token()
        assert bytes_per_token == 512, (
            "DSV4 uniform-FP8 KV layout: qk_nope_head_dim (448) + "
            "qk_rope_head_dim (64), all e4m3 = 512 bytes/token"
        )
        self.kv_cache_total_dim = bytes_per_token
        self.bytes_per_page_padded = self.page_size * bytes_per_token

        return torch.zeros(
            num_pages,
            self.page_size * bytes_per_token,
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )

    def get_key_buffer(self, layer_id: int):
        return self.kv_buffer[layer_id]

    def set_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ):
        raise NotImplementedError(
            "The packed NopeFp8RopeBf16Pack store does not apply to the "
            "uniform-FP8 pool; use set_key_buffer_fused."
        )

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        """Store normed/roped rows as e4m3 with the backend's fixed unit scale.

        uint8 views work around index_put not supporting FP8 dtypes.
        """

        assert cache_k.dim() == 2 and cache_k.shape[1] == self.kv_cache_total_dim
        self.kv_buffer[layer_id].view(torch.uint8).view(-1, self.kv_cache_total_dim)[
            loc.long()
        ] = cache_k.to(torch.float8_e4m3fn).view(torch.uint8)


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

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        raise NotImplementedError("HiSparseC4DevicePool does not support get_cpu_copy")

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
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
        self.use_fp4_indexer = get_exec().kernel.enable_deepseek_v4_fp4_indexer
        self.uses_aiter_fp4_layout = _is_hip and self.use_fp4_indexer

        self._create_buffer()

    def get_bytes_per_token(self) -> int:
        return get_dsv4_indexer_bytes_per_token(
            self.index_head_dim, self.use_fp4_indexer
        )

    def _create_buffer(self):
        page_bytes = self.page_size * self.get_bytes_per_token()
        num_pages = (self.size + self.page_size + 1) // self.page_size
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                if self.uses_aiter_fp4_layout:
                    self.index_k_payload_buffer = [
                        torch.zeros(
                            (num_pages, 1, 4, self.page_size, 16),
                            dtype=torch.uint8,
                            device=self.device,
                        ).view(torch.float4_e2m1fn_x2)
                        for _ in range(self.layer_num)
                    ]
                    self.index_k_scale_buffer = [
                        torch.zeros(
                            (num_pages, 1, 4, self.page_size),
                            dtype=torch.uint8,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.index_k_with_scale_buffer = None
                    return

                self.index_k_with_scale_buffer = [
                    torch.zeros(
                        num_pages,
                        page_bytes,
                        dtype=self.index_k_with_scale_buffer_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

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

    def contiguous_page_row_buffers(self) -> List[torch.Tensor]:
        """Every indexer buffer as 2D page rows, for PD and HiCache transfer.

        FP8 keeps key and scale fused in one buffer per layer; the FP4 layout
        stores payload and scale separately, so it yields two buffers per layer.
        """
        if self.index_k_with_scale_buffer is not None:
            return self.index_k_with_scale_buffer
        return [
            buf.view(torch.uint8).flatten(1)
            for buf in (*self.index_k_payload_buffer, *self.index_k_scale_buffer)
        ]

    def get_index_k_fp4_payload_buffer(self, layer_id: int) -> torch.Tensor:
        return self.index_k_payload_buffer[layer_id]

    def get_index_k_fp4_scale_buffer(self, layer_id: int) -> torch.Tensor:
        return self.index_k_scale_buffer[layer_id]

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        buf = self.index_k_with_scale_buffer[layer_id]
        return index_buf_accessor.GetKAndS.execute(
            self,
            buf,
            page_indices=page_indices,
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
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

    def set_index_fp4(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            store_fp4_index_k_cache,
        )

        return store_fp4_index_k_cache(
            input=cache_k,
            cache=self.index_k_with_scale_buffer[layer_id - self.start_layer],
            loc=loc,
            page_size=self.page_size,
        )


class _CompressedPoolConfig(NamedTuple):
    kv_size: int
    state_size: int
    state_dtype: torch.dtype
    indexer_size: Optional[int] = None


class DeepSeekV4LayerItem(NamedTuple):
    compress_ratio: int
    compress_layer_id: int
    compress_kv_pool: Optional[DeepSeekV4SingleKVPool] = None


# re-exported: the pool allocates the rows, but the kernels that write them own the
# layout (see unified_kv_kernels/layout.py)
DSV4_FP8_NOPE_ROW_BYTES = layout.DSV4_FP8_NOPE_ROW_BYTES
DSV4_FP8_QUANT_TILE = layout.DSV4_FP8_QUANT_TILE


def dsv4_unified_row_bytes(
    qk_nope_head_dim: int, qk_rope_head_dim: int, fp8: bool
) -> int:
    """Bytes one unified_kv token occupies, summed over both pools."""
    if not fp8:
        return (qk_nope_head_dim + qk_rope_head_dim) * 2
    num_tiles = -(-qk_nope_head_dim // DSV4_FP8_QUANT_TILE)
    scale_bytes = 2 * num_tiles
    # not an assert: sizing runs under -O too, and a silently skipped check here
    # overreports capacity
    if qk_nope_head_dim + scale_bytes > DSV4_FP8_NOPE_ROW_BYTES:
        raise ValueError(
            f"fp8 nope row overflows: {qk_nope_head_dim} latent values at 1 B + "
            f"{scale_bytes} B scales > {DSV4_FP8_NOPE_ROW_BYTES} B stride"
        )
    return DSV4_FP8_NOPE_ROW_BYTES + qk_rope_head_dim * 2


# The following kv pool follows ATOM's unified_kv kernel layout.
class DeepSeekV4UnifiedKVPool:
    """
    Layout (bf16):
    unified_kv[L]: ``[swa_pages + padded_compress_rows, head_dim]`` bf16

    Layout (fp8, ``SGLANG_DSV4_UNIFIED_KV_FP8``) -- two parallel pools with the
    same row count, so a row index means the same thing in both. Named after the
    accessors, which under fp8 each return one half -- ``get_unified_kv`` the
    nope, ``get_unified_kv_rope`` the rope:
    unified_kv[L]      (nope): ``[rows, 512]`` fp8, see DSV4_FP8_NOPE_ROW_BYTES
    unified_kv_rope[L] (rope): ``[rows, qk_rope_head_dim]`` bf16, never quantized

    - rows ``[0, swa_pages)``   = SWA ring (``req_pool_indices * swa_window + pos % swa_window``)
    - rows ``[swa_pages, ...)`` = compressed (``swa_pages + page_index``)
    """

    K_PER_BLOCK = {0: 0, 4: 32, 128: 1}

    def __init__(
        self,
        *,
        stage_ratios: List[int],
        num_slots: int,
        num_blocks: int,
        page_size: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        device: str,
        memory_saver_adapter,
        custom_mem_pool,
        swa_ring_size: int,
        fp8: bool = False,
    ):
        self.swa_ring_size = swa_ring_size
        self.fp8 = fp8
        self.rope_head_dim = qk_rope_head_dim
        self.head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.num_slots = num_slots
        self.swa_pages = num_slots * self.swa_ring_size
        self.num_blocks = num_blocks
        self.page_size = page_size
        self.k_per_block = dict(self.K_PER_BLOCK)

        bufs = []
        rope_bufs = []
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(custom_mem_pool)
                if custom_mem_pool
                else nullcontext()
            ):
                for ratio in stage_ratios:
                    # Pad by one extra page. The KV pool reserves a null slot
                    # (token indices run 1..size).
                    compress_rows = self.num_blocks * self.k_per_block[ratio]
                    rows_per_page = self.page_size // ratio if ratio else 0
                    padded_compress_rows = compress_rows + rows_per_page
                    rows = self.swa_pages + padded_compress_rows
                    if self.fp8:
                        bufs.append(
                            torch.zeros(
                                rows,
                                DSV4_FP8_NOPE_ROW_BYTES,
                                dtype=torch.float8_e4m3fn,
                                device=device,
                            )
                        )
                        rope_bufs.append(
                            torch.zeros(
                                rows,
                                self.rope_head_dim,
                                dtype=torch.bfloat16,
                                device=device,
                            )
                        )
                    else:
                        bufs.append(
                            torch.zeros(
                                rows,
                                self.head_dim,
                                dtype=torch.bfloat16,
                                device=device,
                            )
                        )
                        rope_bufs.append(None)
        self.kv_buffer = bufs
        self.kv_buffer_rope = rope_bufs

    def get_unified_kv(self, local_layer_id: int) -> torch.Tensor:
        return self.kv_buffer[local_layer_id]

    def get_unified_kv_rope(self, local_layer_id: int) -> torch.Tensor:
        assert self.fp8, "rope pool only exists under SGLANG_DSV4_UNIFIED_KV_FP8"
        return self.kv_buffer_rope[local_layer_id]

    def get_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        if self.fp8:
            # same single-pool assumption as the outer get_contiguous_buf_infos:
            # one pointer and one row size per layer describes the nope pool only,
            # so whoever picks this up next would move half a row and not notice.
            # TODO(danli103): report both pools once a consumer needs them.
            raise NotImplementedError(
                "get_buf_infos describes one pool per layer; the fp8 rope pool "
                "would be dropped (SGLANG_DSV4_UNIFIED_KV_FP8=1)."
            )
        data_ptrs = [b.data_ptr() for b in self.kv_buffer]
        data_lens = [b.nbytes for b in self.kv_buffer]
        item_lens = [b[0].nbytes for b in self.kv_buffer]
        return data_ptrs, data_lens, item_lens


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
        c4_state_dtype: torch.dtype,
        c128_state_dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        indexer_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        compression_ratios: List[int],
        sliding_window: int = 128,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_hisparse: bool = False,
        online_mtp_max_draft_tokens: int = 0,
        num_req_slots: Optional[int] = None,
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
        # PD preallocation can exceed max_num_reqs;
        # the SWA ring must cover every addressable req_pool_idx.
        self.num_req_slots = (
            num_req_slots if num_req_slots is not None else max_num_reqs + 1
        )
        self.c4_logical_size = c4_logical_size
        self.c128_size = c128_size
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_fp8,
            is_unified_kv_triton,
        )

        # Resolve the unified-kv gate before any sizing so the two cannot drift.
        self._unified_kv = is_unified_kv_triton()
        self._unified_kv_fp8 = is_unified_kv_fp8()
        # Uniform 512-dim e4m3 layout for the trtllm attention backend
        self.uniform_fp8 = (
            not self._unified_kv
        ) and get_exec().kernel.dsv4_attn_backend == "trtllm"
        c4_ring_size = self.get_ring_size(4)
        if self._unified_kv:
            # Unified C4 state is request-addressed: one ring per req slot,
            # so the caller-supplied, SWA-scaled size does not apply here.
            c4_state_pool_size = self.num_req_slots * c4_ring_size
        # Non-unified (fp8) keeps the caller-supplied, SWA-addressed size.
        c128_ring_size = self.get_ring_size(128)
        if ONLINE_C128:
            # Request-scoped C128 state must also cover PD preallocation slots.
            c128_state_pool_size = max(c128_state_pool_size, self.num_req_slots)
        else:
            # Offline C128 keeps a per-request raw state ring.
            c128_state_pool_size = max(
                c128_state_pool_size, self.num_req_slots * c128_ring_size
            )
        self.compressed_pool_configs = {
            4: _CompressedPoolConfig(
                kv_size=c4_size,
                state_size=c4_state_pool_size,
                state_dtype=c4_state_dtype,
                indexer_size=c4_logical_size,
            ),
            128: _CompressedPoolConfig(
                kv_size=c128_size,
                state_size=c128_state_pool_size,
                state_dtype=c128_state_dtype,
            ),
        }
        self.compression_ratios = compression_ratios
        self.online_mtp_max_draft_tokens = online_mtp_max_draft_tokens
        self.online_c128_state_num_req_slots = c128_state_pool_size
        self.online_c128_mtp_pending_seq_lens: Optional[torch.Tensor] = None
        if ONLINE_C128 and envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get():
            self.online_c128_mtp_pending_seq_lens = torch.empty(
                self.online_c128_state_num_req_slots, dtype=torch.int64, device=device
            )

        # Determine this PP stage's absolute layer range
        if (
            start_layer is not None
            and end_layer is not None
            and len(compression_ratios) >= end_layer
        ):
            self._stage_start = start_layer
            self._stage_end = end_layer
        else:
            self._stage_start = 0
            self._stage_end = len(compression_ratios)
        stage_ratios = compression_ratios[self._stage_start : self._stage_end]

        assert page_size % swa_page_size == 0
        self.sliding_window = sliding_window

        self.swa_size = swa_size
        self.swa_page_size = swa_page_size

        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.indexer_head_dim = indexer_head_dim

        stage_layer_num = len(stage_ratios)
        kv_pool_cls: type = DeepSeekV4SingleKVPool

        if self._unified_kv:
            self.swa_kv_pool = None
            swa_ring_size = get_swa_ring_size(
                self.sliding_window, get_spec().speculative_algorithm is not None
            )
            self.unified_kv_pool = DeepSeekV4UnifiedKVPool(
                stage_ratios=stage_ratios,
                num_slots=self.num_req_slots,
                num_blocks=self.c128_size,
                page_size=page_size,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                device=device,
                memory_saver_adapter=self.memory_saver_adapter,
                custom_mem_pool=self.custom_mem_pool,
                swa_ring_size=swa_ring_size,
                fp8=self._unified_kv_fp8,
            )

            self.unified_swa_window = self.sliding_window
            self.unified_swa_ring_size = swa_ring_size
            self.unified_swa_pages = self.unified_kv_pool.swa_pages
            self.swa_req_ring_size = self.unified_swa_ring_size
        else:
            self.unified_kv_pool = None
            if self.uniform_fp8:
                assert dtype == torch.float8_e4m3fn, (
                    "--dsv4-attn-backend trtllm requires "
                    f"kv_cache_dtype=fp8_e4m3, got {dtype}"
                )
                kv_pool_cls = DeepSeekV4UniformFP8KVPool
            self.swa_kv_pool = self._make_kv_pool(
                size=swa_size,
                page_size=swa_page_size,
                dtype=dtype,
                layer_num=stage_layer_num,
                device=device,
                enable_memory_saver=enable_memory_saver,
                global_page_size=swa_page_size,
                cls=kv_pool_cls,
            )

        self._init_compressed_pools(
            stage_ratios=stage_ratios,
            page_size=page_size,
            dtype=dtype,
            device=device,
            enable_memory_saver=enable_memory_saver,
            enable_hisparse=enable_hisparse,
            kv_pool_cls=kv_pool_cls,
        )

        self._init_compressed_layer_mapping()

        self._init_paged_compress_states(enable_memory_saver)

    def get_unified_kv(self, layer_id: int) -> torch.Tensor:
        # Under HiCache the compressed region is loaded H->D per layer; wait for this
        # layer's transfer before attention reads it. No-op when HiCache is off.
        self.wait_layer_transfer(layer_id)
        return self.unified_kv_pool.get_unified_kv(layer_id - self._stage_start)

    def get_unified_kv_rope(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        return self.unified_kv_pool.get_unified_kv_rope(layer_id - self._stage_start)

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def get_ring_size(self, compress_ratio: int) -> int:
        is_speculative = get_spec().speculative_algorithm is not None
        return get_compress_state_ring_size(compress_ratio, is_speculative)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None
        return self.full_to_swa_index_mapping[kv_indices]

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        if self._unified_kv_fp8:
            # The page-block transfer below prices one row as buf[0].nbytes and
            # ships a single pointer per layer. Under fp8 that covers the nope
            # pool only -- the parallel bf16 rope pool would be dropped and the
            # remote side would decode rows against stale rope. Refuse instead.
            # TODO(danli103): ship the rope pool as a second per-layer entry.
            raise NotImplementedError(
                "PD disaggregation is not supported with "
                "SGLANG_DSV4_UNIFIED_KV_FP8=1 (the transfer assumes a single "
                "unified pool; the rope pool would be silently dropped)."
            )

        def append_page_buffer(buf: torch.Tensor) -> None:
            assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
            data_ptrs.append(buf.data_ptr())
            data_lens.append(buf.nbytes)
            item_lens.append(buf[0].nbytes)

        stage_ratios = self.compression_ratios[self._stage_start : self._stage_end]
        # Registration order defines the PD wire layout: C4 KV, C4 indexer, C128 KV.
        # Keep each indexer immediately after the KV buffers of the same ratio.
        for ratio, kv_pool in self.kv_pools.items():
            if self._unified_kv:
                # Unified buffers store token rows after the SWA ring. Transfer
                # compressed pages from the offset; SWA ships as StateType.SWA_RING.
                swa_pages = self.unified_kv_pool.swa_pages
                for local_layer_id, layer_ratio in enumerate(stage_ratios):
                    if layer_ratio != ratio:
                        continue
                    buf = self.unified_kv_pool.kv_buffer[local_layer_id]
                    assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                    row_bytes = buf[0].nbytes
                    rows_per_page = self.page_size // ratio
                    compress_rows = buf.shape[0] - swa_pages
                    data_ptrs.append(buf.data_ptr() + swa_pages * row_bytes)
                    data_lens.append(compress_rows * row_bytes)
                    item_lens.append(rows_per_page * row_bytes)
            else:
                for buf in kv_pool.kv_buffer:
                    append_page_buffer(buf)

            indexer_pool = self.index_pools.get(ratio)
            if indexer_pool is not None:
                for buf in indexer_pool.contiguous_page_row_buffers():
                    append_page_buffer(buf)

        return data_ptrs, data_lens, item_lens

    def get_unified_swa_ring_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        # StateType.SWA_RING transfers [0, swa_pages) of each unified_kv layer;
        # its indices address individual ring rows.
        # TODO(billishyahao): validate PP layer-slicing for SWA_RING.
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        if not self._unified_kv:
            return data_ptrs, data_lens, item_lens
        if self._unified_kv_fp8:
            # Other half of the PD path -- get_contiguous_buf_infos ships the
            # compressed region, this one the ring. Same single-pool assumption,
            # same silently dropped rope, same fix -- land them together.
            raise NotImplementedError(
                "PD disaggregation is not supported with "
                "SGLANG_DSV4_UNIFIED_KV_FP8=1 (the SWA_RING component assumes a "
                "single unified pool; the rope pool would be silently dropped)."
            )
        swa_pages = self.unified_kv_pool.swa_pages
        for buf in self.unified_kv_pool.kv_buffer:
            assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
            row_bytes = buf[0].nbytes
            data_ptrs.append(buf.data_ptr())
            data_lens.append(swa_pages * row_bytes)
            item_lens.append(row_bytes)
        return data_ptrs, data_lens, item_lens

    def unified_region_buffers(self, ratio: int) -> Tuple[List[torch.Tensor], int]:
        # HiCache expects byte rows containing whole pages;
        # the unified pool stores individual token rows after its SWA region.
        assert self._unified_kv, "unified_region_buffers requires unified_kv layout"
        assert ratio in (4, 128), f"unsupported compression ratio: {ratio}"
        if self._unified_kv_fp8:
            # item_bytes below prices kv_buffer alone, so the rope pool would never
            # be offloaded and a fetched page would carry stale rope -- wrong output,
            # no crash.
            # TODO(danli103): give rope its own host pool, the way C4_INDEXER
            # already parallels C4.
            raise NotImplementedError(
                "HiCache offload is not supported with "
                "SGLANG_DSV4_UNIFIED_KV_FP8=1 (the host pool assumes a single "
                "unified pool; the rope pool would never be offloaded)."
            )

        swa_pages = self.unified_kv_pool.swa_pages
        head_dim = self.unified_kv_pool.head_dim
        rows_per_page = self.page_size // ratio
        stage_ratios = self.compression_ratios[self._stage_start : self._stage_end]
        local_layer_ids = [i for i, r in enumerate(stage_ratios) if r == ratio]

        views: List[torch.Tensor] = []
        for local_layer_id in local_layer_ids:
            buf = self.unified_kv_pool.kv_buffer[local_layer_id]
            compress_rows = buf.shape[0] - swa_pages
            assert compress_rows % rows_per_page == 0, (
                f"compressed rows {compress_rows} not a multiple of "
                f"rows_per_page {rows_per_page} for ratio {ratio}"
            )
            num_pages = compress_rows // rows_per_page
            page_view = (
                buf.narrow(0, swa_pages, compress_rows)
                .reshape(num_pages, rows_per_page * head_dim)
                .view(torch.uint8)
            )
            views.append(page_view)

        item_bytes = (
            rows_per_page * head_dim * self.unified_kv_pool.kv_buffer[0].element_size()
        )
        return views, item_bytes

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        if not self._unified_kv:
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
                if pool.ratio == 128:
                    continue
                t = pool.kv_score_buffer.kv_score
                assert t.ndim == 2, f"expected 2D buffer, got {t.ndim}D"
                data_ptrs.append(t.data_ptr())
                data_lens.append(t.nbytes)
                item_lens.append(t[0].nbytes * pool.ring_size)

        return data_ptrs, data_lens, item_lens

    def get_request_state_buf_infos(
        self,
    ) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        for pool in self.compress_state_pools:
            if pool is None or pool.ratio != 128:
                continue
            t = pool.kv_score_buffer.kv_score
            assert t.ndim == 2, f"expected 2D buffer, got {t.ndim}D"
            data_ptrs.append(t.data_ptr())
            data_lens.append(t.nbytes)
            item_lens.append(t[0].nbytes if ONLINE_C128 else t[0].nbytes * 128)
        return data_ptrs, data_lens, item_lens

    def _init_compressed_pools(
        self,
        *,
        stage_ratios: Sequence[int],
        page_size: int,
        dtype: torch.dtype,
        device: str,
        enable_memory_saver: bool,
        enable_hisparse: bool,
        kv_pool_cls: type,
    ) -> None:
        configs = self.compressed_pool_configs
        layer_counts = {ratio: stage_ratios.count(ratio) for ratio in configs}
        # Keep empty pools and allocation order for PP stages without a given ratio.
        self.kv_pools: dict[int, Optional[DeepSeekV4SingleKVPool]] = {
            ratio: None for ratio in configs
        }

        if not self._unified_kv:
            for ratio, config in configs.items():
                pool_cls = kv_pool_cls
                if ratio == 4 and enable_hisparse:
                    assert not self.uniform_fp8, (
                        "enable_hisparse is not supported with --dsv4-attn-backend trtllm."
                    )
                    pool_cls = HiSparseC4DevicePool
                self.kv_pools[ratio] = self._make_kv_pool(
                    size=config.kv_size,
                    page_size=page_size // ratio,
                    dtype=dtype,
                    layer_num=layer_counts[ratio],
                    device=device,
                    enable_memory_saver=enable_memory_saver,
                    global_page_size=page_size,
                    cls=pool_cls,
                )

        self.index_pools: dict[int, DeepSeekV4IndexerPool] = {
            ratio: self._make_indexer_pool(
                config.indexer_size,
                page_size // ratio,
                dtype,
                self.indexer_head_dim,
                layer_counts[ratio],
                device,
                enable_memory_saver,
            )
            for ratio, config in configs.items()
            if config.indexer_size is not None
        }

        # HiCache and hardware backends still access the per-ratio attributes.
        self.c4_kv_pool = self.kv_pools[4]
        self.c128_kv_pool = self.kv_pools[128]
        self.c4_indexer_kv_pool = self.index_pools[4]

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
    ) -> DeepSeekV4SingleKVPool:
        """Build a full / SWA / c4 / c128 single-KV pool. ``global_page_size``
        is the model-wide page_size (== ``page_size`` for the SWA pool, larger
        for the per-ratio c4/c128 pools); the default CUDA pool ignores it.
        Overridden by :class:`DSV4NPUTokenToKVPool` to swap in the NPU bf16
        PA_ND variant, which needs ``global_page_size`` for its kernel view."""
        del global_page_size  # CUDA pools key only off their own page_size
        return cls(
            size,
            page_size,
            dtype,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
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
    ) -> DeepSeekV4IndexerPool:
        """Build the c4 lightning-indexer K pool (packed CUDA layout).
        Overridden by :class:`DSV4NPUTokenToKVPool` to swap in the
        dedicated-buffer NPU variant (int8 K + fp16 scale)."""
        return DeepSeekV4IndexerPool(
            size,
            page_size,
            dtype,
            index_head_dim,
            layer_num,
            device,
            enable_memory_saver,
        )

    def _make_compress_state_pool(
        self, ratio: int, *, head_dim: int, enable_memory_saver: bool
    ) -> CompressStatePool:
        """Build attention or indexer state; hardware backends override this factory."""
        config = self.compressed_pool_configs[ratio]
        return CompressStatePool(
            size=config.state_size,
            ring_size=self.get_ring_size(ratio),
            overlap=ratio == 4,
            head_dim=head_dim,
            dtype=config.state_dtype,
            device=self.device,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            online=(ratio == 128 and ONLINE_C128),
            swa_page_size=self.swa_page_size,
            online_mtp_max_draft_tokens=(
                self.online_mtp_max_draft_tokens if ratio == 128 else 0
            ),
        )

    def _init_paged_compress_states(self, enable_memory_saver: bool):
        total_L = len(self.compression_ratios)
        self.compress_state_pools: List[Optional[CompressStatePool]] = [None] * total_L
        self.indexer_compress_state_pools: List[Optional[CompressStatePool]] = [
            None
        ] * total_L

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio == 0:
                continue

            self.compress_state_pools[idx] = self._make_compress_state_pool(
                ratio,
                head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
                enable_memory_saver=enable_memory_saver,
            )

            if ratio in self.index_pools:
                self.indexer_compress_state_pools[idx] = self._make_compress_state_pool(
                    ratio,
                    head_dim=self.indexer_head_dim,
                    enable_memory_saver=enable_memory_saver,
                )

    def _init_compressed_layer_mapping(self):
        layer_counts = {0: 0, **{ratio: 0 for ratio in self.kv_pools}}
        total_L = len(self.compression_ratios)
        self.layer_mapping: List[Optional[DeepSeekV4LayerItem]] = [None] * total_L

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio not in layer_counts:
                raise ValueError(f"Unsupported compression ratio: {ratio}")
            self.layer_mapping[idx] = DeepSeekV4LayerItem(
                compress_ratio=ratio,
                compress_layer_id=layer_counts[ratio],
                compress_kv_pool=self.kv_pools.get(ratio),
            )
            layer_counts[ratio] += 1

    def wait_layer_transfer(self, layer_id: int) -> None:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_attention_compress_states(self, layer_id: int) -> CompressStatePool:
        self.wait_layer_transfer(layer_id)
        compress_state_pool = self.compress_state_pools[layer_id]
        assert compress_state_pool is not None, (
            "Only c4/c128 layers have attention states."
        )
        return compress_state_pool

    def get_online_c128_mtp_state_slot_offset(self) -> int:
        for pool in self.compress_state_pools:
            if pool is not None and pool.ratio == 128:
                return int(pool.online_mtp_state_slot_offset)
        return 0

    def get_online_c128_mtp_max_draft_tokens(self) -> int:
        for pool in self.compress_state_pools:
            if pool is not None and pool.ratio == 128:
                return int(pool.online_mtp_max_draft_tokens)
        return 0

    def get_online_c128_state_num_req_slots(self) -> int:
        return self.online_c128_state_num_req_slots

    def get_online_c128_mtp_pending_seq_lens(self) -> torch.Tensor:
        assert self.online_c128_mtp_pending_seq_lens is not None
        return self.online_c128_mtp_pending_seq_lens

    def clear_c4_req_states(self, req_pool_indices: Sequence[int]) -> None:
        if not self._unified_kv or not req_pool_indices:
            return

        pools = [
            pool
            for pool in self.compress_state_pools + self.indexer_compress_state_pools
            if pool is not None and pool.ratio == 4
        ]
        if not pools:
            return

        ring_size = self.get_ring_size(4)
        device = pools[0].kv_score_buffer.kv_score.device
        req_indices = torch.as_tensor(req_pool_indices, dtype=torch.long, device=device)
        state_locs = (
            req_indices[:, None] * ring_size
            + torch.arange(ring_size, dtype=torch.long, device=device)
        ).flatten()

        for pool in pools:
            state = pool.kv_score_buffer.kv_score
            half = state.shape[-1] // 2
            state[state_locs, :half] = 0
            state[state_locs, half:] = float("-inf")

    def clear_c128_req_state(self, req_pool_idx: int) -> None:
        """Reset request-scoped C128 state for one req slot."""
        for pool in self.compress_state_pools:
            if pool is None or pool.ratio != 128:
                continue

            state = pool.kv_score_buffer.kv_score
            if ONLINE_C128:
                row = state[req_pool_idx]
                head_dim = row.shape[-1] // 3
                row[:head_dim].fill_(float("-inf"))
                row[head_dim:].zero_()
            else:
                start = req_pool_idx * pool.ring_size
                rows = state[start : start + pool.ring_size]
                half = rows.shape[-1] // 2
                rows[:, :half].zero_()
                rows[:, half:].fill_(float("-inf"))

    def clear_unaccepted_c128_draft_states(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        accept_lens: torch.Tensor,
        num_draft_tokens: int,
    ) -> None:
        # C128 compression can read rejected draft slots at a boundary;
        # C4 overwrites its draft slots before reading them.
        if ONLINE_C128 or num_draft_tokens <= 1 or req_pool_indices.numel() == 0:
            return

        bs = req_pool_indices.numel()
        for pool in self.compress_state_pools:
            if pool is None or pool.ratio != 128:
                continue

            clear_unaccepted_c128_draft_states(
                pool.kv_score_buffer.kv_score,
                req_pool_indices,
                seq_lens,
                accept_lens,
                ring_size=pool.ring_size,
                num_draft_tokens=num_draft_tokens,
            )

    def get_indexer_compress_states(self, layer_id: int) -> CompressStatePool:
        self.wait_layer_transfer(layer_id)
        indexer_compress_state_pool = self.indexer_compress_state_pools[layer_id]
        assert indexer_compress_state_pool is not None, (
            "Only c4 layers have indexer states."
        )
        return indexer_compress_state_pool

    def _swa_local_layer_id(self, layer_id: int) -> int:
        """Convert absolute model layer_id to SWA-pool-local (PP-stage-local) index."""
        return layer_id - self._stage_start

    def get_swa_raw_buffer(self, layer_id: int) -> torch.Tensor:
        return self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)]

    def get_extra_key_page_size(self, layer_id: int) -> int:
        _, _, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.page_size

    def get_extra_key_buffer(self, layer_id: int) -> torch.Tensor | None:
        self.wait_layer_transfer(layer_id)
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

    def _indexer_pool(self, compress_ratio: int) -> DeepSeekV4IndexerPool:
        pool = self.index_pools.get(compress_ratio)
        assert pool is not None, (
            f"No indexer pool for compression ratio {compress_ratio}"
        )
        return pool

    def get_index_k_page_size(self, compress_ratio: int = 4) -> int:
        return self._indexer_pool(compress_ratio).page_size

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        return self._indexer_pool(compress_ratio).get_index_k_with_scale_buffer(
            compress_layer_id
        )

    def get_index_k_fp4_payload_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        return self._indexer_pool(compress_ratio).get_index_k_fp4_payload_buffer(
            compress_layer_id
        )

    def get_index_k_fp4_scale_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        return self._indexer_pool(compress_ratio).get_index_k_fp4_scale_buffer(
            compress_layer_id
        )

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.wait_layer_transfer(layer_id)
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        return self._indexer_pool(compress_ratio).get_index_k_scale_buffer(
            compress_layer_id,
            seq_len_tensor,
            page_indices,
            seq_len_sum,
            max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        self._indexer_pool(compress_ratio).set_index_k_scale_buffer(
            compress_layer_id, loc, index_k, index_k_scale
        )

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def set_swa_key_buffer_radix(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        self.swa_kv_pool.set_key_buffer(
            self._swa_local_layer_id(layer_id), swa_loc, cache_nope_fp8_rope_bf16_pack
        )

    def get_swa_key_buffer_radix(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        return self.swa_kv_pool.get_key_buffer(self._swa_local_layer_id(layer_id))

    def set_swa_key_buffer_radix_fused(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        return self.swa_kv_pool.set_key_buffer_fused(
            self._swa_local_layer_id(layer_id), swa_loc, cache_k
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
        if self.uniform_fp8:
            # Uniform-FP8 (trtllm-gen) layout: norm + RoPE with the existing
            # Triton kernel (in-place on kv; safe -- kv is not read again),
            # then a plain e4m3 cast + scatter in the pool setter (per-tensor
            # scale 1.0). Fusing the store is deferred to the perf phase.
            from sglang.kernels.ops.attention.deepseek_v4_rope import (
                fused_norm_rope_inplace_triton,
            )

            fused_norm_rope_inplace_triton(
                kv,
                kv_weight,
                eps,
                freqs_cis,
                positions=positions,
            )
            self.swa_kv_pool.set_key_buffer_fused(
                self._swa_local_layer_id(layer_id), swa_loc, kv
            )
            return
        fused_k_norm_rope_flashmla(
            kv=kv,
            kv_weight=kv_weight,
            eps=eps,
            freqs_cis=freqs_cis,
            positions=positions,
            out_loc=swa_loc,
            kvcache=self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)],
            page_size=self.swa_kv_pool.page_size,
        )

    def set_unified_key_buffer_radix_fused_norm_rope(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        kv: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """unified_kv counterpart of set_swa_key_buffer_radix_fused_norm_rope.

        Under unified_kv the (fp8, paged) swa_kv_pool is None -- SWA K lives in
        the shared bf16 unified_kv ring instead. Norm+RoPE the draft KV in place
        (the same freqs_cis path the main model uses via _compute_kv_bf16) and
        scatter it into ``unified_kv[swa_loc]``. Rows with swa_loc < 0
        (uncommitted verify tokens) are skipped by the scatter.
        """
        from sglang.kernels.ops.attention.dsv4 import fused_norm_rope_inplace
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import runtime

        fused_norm_rope_inplace(kv, kv_weight, eps, freqs_cis, positions)
        runtime.scatter_bf16_into_unified(
            kv=kv,
            loc=swa_loc,
            unified_kv=self.get_unified_kv(layer_id),
        )

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
        return self._indexer_pool(compress_ratio).set_index_fused(
            compress_layer_id, loc, cache_k
        )

    def set_index_k_fp4(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        return self._indexer_pool(compress_ratio).set_index_fp4(
            compress_layer_id, loc, cache_k
        )
