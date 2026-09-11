from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Literal, NamedTuple, Optional, Sequence, Tuple

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
    compress_ratio: int, is_speculative: bool = False, num_draft_tokens: int = 0
) -> int:
    assert compress_ratio in [2, 4, 128], f"Unsupported {compress_ratio = }"
    if compress_ratio == 2:
        # Ratio 2 keeps one pending even token per request, addressed by
        # position % ring_size; two positions are one pair. A speculative ring
        # must stay wider than the draft window so that regenerating a rejected
        # position overwrites its own slot before the position after it reads
        # the slot before it: smallest power of two >= 2 + draft tokens.
        if not is_speculative:
            return 2
        return 1 << (num_draft_tokens + 1).bit_length()
    # Online c128 keeps a single (max, sum, kv) state per index instead of a
    # 128-slot ring buffer of raw tokens, so ring_size collapses to 1. Online
    # is incompatible with speculative decode for now.
    if compress_ratio == 128 and ONLINE_C128:
        if is_speculative and not envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get():
            raise AssertionError("online c128 does not support MTP")
        return 1
    if is_speculative:
        return 16 if compress_ratio == 4 else 256
    else:
        return 8 if compress_ratio == 4 else 128


def get_compress_state_write_pad(compress_ratio: int, ring_size: int) -> int:
    """Largest draft-token count this ring can serve; mirrors `mtp_pad` in `c_plan.cuh`
    (the bound is derived there). Zero for a non-speculative ring, which is exactly one
    window wide. Ratio 2's window is the pair itself, so its speculative ring serves
    ring_size draft tokens."""
    window_size = compress_ratio * (2 if compress_ratio == 4 else 1)
    return ring_size - window_size + 2 if ring_size > window_size else 0


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


# Low-ratio indexer-K pool page, in compressed slots: the DeepGEMM indexer reads
# K in blocks of at most 128 and sglang's JIT metadata builder asserts 64.
DSV41_INDEX_PAGE_SIZE = 64


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
        use_fp4_indexer: Optional[bool] = None,
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
        if use_fp4_indexer is None:
            use_fp4_indexer = get_exec().kernel.enable_deepseek_v4_fp4_indexer
        self.use_fp4_indexer = use_fp4_indexer
        self.uses_aiter_fp4_layout = _is_hip and self.use_fp4_indexer
        # Low-ratio pools round to nearest even, as the reference does; c4 keeps
        # the threshold rounding.
        self.index_k_rne = False

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
            rne=self.index_k_rne,
        )

    def get_index_k_fp4(
        self, layer_id: int, slots: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Packed fp4 rows at `slots`: (payload int8 [n, 64], scales int32 [n]).
        Inverse of the store_fp4_index_k_cache page layout
        [page_size * 64 payload | page_size * 4 scale bytes]."""
        assert self.use_fp4_indexer, "packed readback only applies to the fp4 layout"
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        slots = slots.to(torch.int64)
        p = self.page_size
        page, off = (slots // p).unsqueeze(-1), slots % p
        payload_cols = (off * 64).unsqueeze(-1) + torch.arange(64, device=buf.device)
        scale_cols = (p * 64 + off * 4).unsqueeze(-1) + torch.arange(
            4, device=buf.device
        )
        payload = buf[page, payload_cols].view(torch.int8)  # [n, 64]
        scales = buf[page, scale_cols].contiguous().view(torch.int32).squeeze(-1)
        return payload, scales

    def get_index_k_dequant(
        self, layer_id: int, slots: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Dequantized bf16 [n, index_head_dim] index K at `slots` (every slot when
        None); pass the slots you need, the full table is pool-sized."""
        from sglang.srt.layers.quantization.fp8 import DSV4_DEQUANT_FP4_TABLE

        assert self.use_fp4_indexer, "dequant readback only applies to the fp4 layout"
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        if slots is None:
            slots = torch.arange(self.size, device=buf.device)
        slots = slots.to(torch.int64)
        # A page is [page_size * 64 payload bytes | page_size * 4 scale bytes]
        # (store_fp4_index_k_cache layout).
        p = self.page_size
        page, off = (slots // p).unsqueeze(-1), slots % p
        payload_cols = (off * 64).unsqueeze(-1) + torch.arange(64, device=buf.device)
        scale_cols = (p * 64 + off * 4).unsqueeze(-1) + torch.arange(
            4, device=buf.device
        )
        u = buf[page, payload_cols].view(torch.uint8)  # [n, 64]
        codes = torch.stack([u & 0x0F, (u >> 4) & 0x0F], dim=-1)  # [n, 64, 2]
        vals = DSV4_DEQUANT_FP4_TABLE.to(buf.device)[codes.long()].flatten(
            1
        )  # [n, 128]
        exps = buf[page, scale_cols].to(torch.int32) & 0xFF  # [n, 4]
        scales = torch.exp2(exps.float() - 127).repeat_interleave(32, dim=-1)
        return (vals * scales).to(torch.bfloat16)


class DeepSeekV4LayerItem(NamedTuple):
    compress_ratio: Literal[0, 1, 2, 4, 128]
    # Layer index inside compress_kv_pool. Ratios 1/2 share a pool layer across the
    # kv_source layer that writes it and the layers that read it.
    compress_layer_id: int
    compress_kv_pool: Optional[DeepSeekV4SingleKVPool] = None


# The following kv pool follows ATOM's unified_kv kernel layout.
class DeepSeekV4UnifiedKVPool:
    """
    Layout:
    unified_kv[L]: ``[swa_pages + padded_compress_rows, head_dim]`` bf16
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
    ):
        self.swa_ring_size = swa_ring_size
        self.head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.num_slots = num_slots
        self.swa_pages = num_slots * self.swa_ring_size
        self.num_blocks = num_blocks
        self.page_size = page_size
        self.k_per_block = dict(self.K_PER_BLOCK)

        bufs = []
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
                    bufs.append(
                        torch.zeros(
                            self.swa_pages + padded_compress_rows,
                            self.head_dim,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                    )
        self.kv_buffer = bufs

    def get_unified_kv(self, local_layer_id: int) -> torch.Tensor:
        return self.kv_buffer[local_layer_id]

    def get_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
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
        kv_source_layers: Sequence[int] = (),
        full_size: Optional[int] = None,
        is_draft_worker: bool = False,
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
        # SWA ring needs one slot per addressable req_pool_idx. PD decode inflates
        # req_to_token past max_num_reqs (pre-alloc), so the caller passes the real
        # capacity; sizing as max_num_reqs+1 overflows ("length out of range").
        self.num_req_slots = (
            num_req_slots if num_req_slots is not None else max_num_reqs + 1
        )
        self.c4_size = c4_size
        self.c4_logical_size = c4_logical_size
        self.c128_size = c128_size
        self.c4_state_pool_size = c4_state_pool_size
        c128_ring_size = self.get_ring_size(128)
        if ONLINE_C128:
            # Request-scoped online C128 state is indexed by req_pool_idx.
            # PD decode can allocate pre-transfer slots beyond
            # max_num_reqs, so size to the actual req_to_token row count.
            c128_state_pool_size = max(c128_state_pool_size, self.num_req_slots)
        else:
            # Offline C128 keeps a per-request raw state ring.
            c128_state_pool_size = max(
                c128_state_pool_size, self.num_req_slots * c128_ring_size
            )
        self.c128_state_pool_size = c128_state_pool_size
        self.c4_state_dtype = c4_state_dtype
        self.c128_state_dtype = c128_state_dtype
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
        self.swa_window_size = swa_page_size
        self.swa_page_size = swa_page_size
        self.scale_pad = 1

        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.indexer_head_dim = indexer_head_dim

        stage_layer_num = len(stage_ratios)

        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        self._unified_kv = is_unified_kv_triton()

        self.request_window = None
        encoder_replay = get_exec().features.enable_encoder_swa_bounded_replay
        # Note(Oasis-Git): Keep DSpark's paged SWA cache because removing its
        # history hurts speculative decoding acceptance length. The draft shares
        # the target's full-to-SWA mapping, so the target still needs the allocator.
        self.needs_paged_swa_allocator = (
            not encoder_replay
            or is_draft_worker
            or get_spec().speculative_algorithm is not None
        )
        if encoder_replay and not is_draft_worker:
            from sglang.srt.mem_cache.dsv41_request_window import RequestWindow

            def make_window_pool(size, layers):
                return self._make_kv_pool(
                    size=size,
                    page_size=swa_page_size,
                    dtype=dtype,
                    layer_num=layers,
                    device=device,
                    enable_memory_saver=enable_memory_saver,
                    global_page_size=swa_page_size,
                )

            self.swa_kv_pool = None
            self.unified_kv_pool = None
            from sglang.srt.runtime_context import get_schedule

            chunk = get_schedule().chunked_prefill_size or 0
            self.request_window = RequestWindow(
                make_window_pool,
                num_slots=self.num_req_slots,
                layers=stage_layer_num,
                page_size=swa_page_size,
                capacity=self.sliding_window + (online_mtp_max_draft_tokens or 0),
                workspace_rows=(self.num_req_slots + 1) * self.sliding_window
                + max(
                    chunk,
                    (self.num_req_slots + 1) * (1 + (online_mtp_max_draft_tokens or 0)),
                ),
            )
        elif self._unified_kv:
            self.swa_kv_pool = None
            spec_extra = (
                (get_spec().speculative_num_draft_tokens - 1)
                if get_spec().speculative_algorithm is not None
                else 0
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
                swa_ring_size=self.sliding_window + spec_extra,
            )

            self.unified_swa_window = self.sliding_window
            self.unified_swa_ring_size = self.sliding_window + spec_extra
            self.unified_swa_pages = self.unified_kv_pool.swa_pages
        else:
            self.unified_kv_pool = None
            self.swa_kv_pool = self._make_kv_pool(
                size=swa_size,
                page_size=swa_page_size,
                dtype=dtype,
                layer_num=stage_layer_num,
                device=device,
                enable_memory_saver=enable_memory_saver,
                global_page_size=swa_page_size,
            )

        logger.info(
            "DSV4 SWA storage: worker=%s, storage=%s, paged_allocator=%s",
            "draft" if is_draft_worker else "target",
            "request_window" if self.request_window is not None else "paged",
            self.needs_paged_swa_allocator,
        )
        self.kv_source_layers = list(kv_source_layers)
        self.sources_by_ratio = self._collect_sources_by_ratio()
        self._init_compressed_pools(
            c4_size=c4_size,
            c128_size=c128_size,
            full_size=full_size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            enable_memory_saver=enable_memory_saver,
            enable_hisparse=enable_hisparse,
        )
        # HiSparse, the HiCache pool assemblers and the NPU pool read the compressed
        # pools by these names; everything in this file goes through the registries.
        self.c4_kv_pool = self.kv_pools.get(4)
        self.c128_kv_pool = self.kv_pools.get(128)
        self.c4_indexer_kv_pool = self.index_pools.get(4)
        self._init_compressed_layer_mapping()

        self._init_paged_compress_states(enable_memory_saver)

    def get_unified_kv(self, layer_id: int) -> torch.Tensor:
        # Under HiCache the compressed region is loaded H->D per layer; wait for this
        # layer's transfer before attention reads it. No-op when HiCache is off.
        self.wait_layer_transfer(layer_id)
        return self.unified_kv_pool.get_unified_kv(layer_id - self._stage_start)

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def get_ring_size(self, compress_ratio: int) -> int:
        spec = get_spec()
        return get_compress_state_ring_size(
            compress_ratio,
            spec.speculative_algorithm is not None,
            spec.speculative_num_draft_tokens or 0,
        )

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None
        return self.full_to_swa_index_mapping[kv_indices]

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        if self._unified_kv:
            # Unified buffer per layer: [swa_pages + padded_compress_rows, head_dim].
            # Compressed region [swa_pages:] is page-contiguous (row swa_pages +
            # loc//ratio), so reuse the page-block PD transfer by offsetting the ptr
            # past the SWA ring and setting item_len = one page of rows. The SWA ring
            # ships separately as StateType.SWA_RING. Order [c4, c4_indexer, c128]
            # mirrors the non-unified kv_data layout (keeps PP ptr-slicing valid).
            stage_ratios = self.compression_ratios[self._stage_start : self._stage_end]
            swa_pages = self.unified_kv_pool.swa_pages

            def _append_compressed_entry(local_layer_id: int, ratio: int) -> None:
                buf = self.unified_kv_pool.kv_buffer[local_layer_id]
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                row_bytes = buf[0].nbytes
                rows_per_page = self.page_size // ratio
                compress_rows = buf.shape[0] - swa_pages
                data_ptrs.append(buf.data_ptr() + swa_pages * row_bytes)
                data_lens.append(compress_rows * row_bytes)
                item_lens.append(rows_per_page * row_bytes)

            c4_locals = [i for i, r in enumerate(stage_ratios) if r == 4]
            c128_locals = [i for i, r in enumerate(stage_ratios) if r == 128]

            for i in c4_locals:
                _append_compressed_entry(i, 4)
            for buf in self.c4_indexer_kv_pool.contiguous_page_row_buffers():
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)
            for i in c128_locals:
                _append_compressed_entry(i, 128)

            return data_ptrs, data_lens, item_lens

        # Fixed ratio order, so the receiver's PP ptr-slicing stays valid. The
        # transfer addresses every buffer by FULL page id, so one item is one
        # FULL page: a KV pool row already is one, while an index pool row is one
        # of its own pages (DSV41_INDEX_PAGE_SIZE slots for the low ratios), so
        # there a FULL page is the run of adjacent index pages that hold its
        # page_size // ratio slots.
        for ratio in (4, 128, 1, 2):
            if ratio not in self.kv_pools:
                continue
            for buf in self.kv_pools[ratio].kv_buffer:
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)
            index_pool = self.index_pools.get(ratio)
            if index_pool is None:
                continue
            slots_per_full_page = self.page_size // ratio
            assert slots_per_full_page % index_pool.page_size == 0, (
                f"ratio-{ratio} index pages of {index_pool.page_size} slots do not "
                f"tile a FULL page of {slots_per_full_page} slots"
            )
            index_pages_per_full_page = slots_per_full_page // index_pool.page_size
            for buf in index_pool.contiguous_page_row_buffers():
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes * index_pages_per_full_page)

        return data_ptrs, data_lens, item_lens

    def get_unified_swa_ring_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """SWA-ring region [0, swa_pages) of every unified_kv layer, addressed
        per-row by ring slot. Shipped as the StateType.SWA_RING PD component."""
        # TODO(billishyahao): validate PP layer-slicing for SWA_RING.
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        if not self._unified_kv:
            return data_ptrs, data_lens, item_lens
        swa_pages = self.unified_kv_pool.swa_pages
        for buf in self.unified_kv_pool.kv_buffer:
            assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
            row_bytes = buf[0].nbytes
            data_ptrs.append(buf.data_ptr())
            data_lens.append(swa_pages * row_bytes)
            item_lens.append(row_bytes)
        return data_ptrs, data_lens, item_lens

    def unified_region_buffers(self, ratio: int) -> Tuple[List[torch.Tensor], int]:
        """
        In unified_kv, swa/c4/c128 share one buffer with one slot per row. But the
        HiCache host pool transfers a whole page per indexed row, so we reshape the
        compressed region into the layout it expects: skip the SWA segment, reshape to
        one row per page, then cast to uint8.
        """
        assert self._unified_kv, "unified_region_buffers requires unified_kv layout"
        assert ratio in (4, 128), f"unsupported compression ratio: {ratio}"

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

        if self.swa_kv_pool is not None:
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
                # Request-scoped state ships as C128_STATE, not with the SWA ring.
                if pool.ratio in (2, 128):
                    continue
                t = pool.kv_score_buffer.kv_score
                assert t.ndim == 2, f"expected 2D buffer, got {t.ndim}D"
                data_ptrs.append(t.data_ptr())
                data_lens.append(t.nbytes)
                item_lens.append(t[0].nbytes * pool.ring_size)

        return data_ptrs, data_lens, item_lens

    def get_c128_state_buf_infos(
        self,
    ) -> Tuple[List[int], List[int], List[int]]:
        """The request-scoped state component, named after its first member: the
        c128 raw-token ring (or its single online row) and the ratio-2
        pending-pair ring. One item is one c128 page / one request's pair ring."""
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        for pool in self.compress_state_pools:
            if pool is None or pool.ratio not in (2, 128):
                continue
            t = pool.kv_score_buffer.kv_score
            assert t.ndim == 2, f"expected 2D buffer, got {t.ndim}D"
            data_ptrs.append(t.data_ptr())
            data_lens.append(t.nbytes)
            if pool.ratio == 2:
                item_lens.append(t[0].nbytes * pool.ring_size)
            else:
                item_lens.append(t[0].nbytes if ONLINE_C128 else t[0].nbytes * 128)
        return data_ptrs, data_lens, item_lens

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
        force_fp4: bool = False,
    ) -> DeepSeekV4IndexerPool:
        """Build the c4 lightning-indexer K pool (packed CUDA layout).
        Overridden by :class:`DSV4NPUTokenToKVPool` to swap in the
        dedicated-buffer NPU variant (int8 K + fp16 scale).

        ``force_fp4`` ignores the deployment flag: the low-ratio indexer is fp4 by
        design, and sizing it off the flag would mis-size the buffer."""
        if force_fp4:
            pool = DeepSeekV4IndexerPool(
                size,
                page_size,
                dtype,
                index_head_dim,
                layer_num,
                device,
                enable_memory_saver,
                use_fp4_indexer=True,
            )
            # The dsv41 low-ratio indexer rounds to nearest even (reference rounding).
            pool.index_k_rne = True
            return pool
        return DeepSeekV4IndexerPool(
            size,
            page_size,
            dtype,
            index_head_dim,
            layer_num,
            device,
            enable_memory_saver,
        )

    def _state_pool_size(self, ratio: int) -> int:
        return self.c4_state_pool_size if ratio == 4 else self.c128_state_pool_size

    def _make_attn_state_pool(
        self, ratio: int, enable_memory_saver: bool
    ) -> CompressStatePool:
        """Build the per-layer attention compress-state pool for ``ratio``
        (4 or 128). Overridden by :class:`DSV4NPUTokenToKVPool` to swap the
        ring-buffered pool for the NPU paged one."""
        return CompressStatePool(
            size=self._state_pool_size(ratio),
            ring_size=self.get_ring_size(ratio),
            overlap=ratio == 4,
            head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            dtype=self.c4_state_dtype if ratio == 4 else self.c128_state_dtype,
            device=self.device,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            online=(ratio == 128 and ONLINE_C128),
            swa_page_size=self.swa_page_size,
            online_mtp_max_draft_tokens=(
                self.online_mtp_max_draft_tokens if ratio == 128 else 0
            ),
        )

    def _make_indexer_state_pool(
        self, ratio: int, enable_memory_saver: bool
    ) -> CompressStatePool:
        """Build the per-layer indexer compress-state pool (c4 only)."""
        return CompressStatePool(
            size=self._state_pool_size(ratio),
            ring_size=self.get_ring_size(ratio),
            overlap=ratio == 4,
            head_dim=self.indexer_head_dim,
            device=self.device,
            dtype=self.c4_state_dtype,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            swa_page_size=self.swa_page_size,
        )

    def _make_pair_state_pool(self, enable_memory_saver: bool) -> CompressStatePool:
        """Ratio-2 pending-pair state: one position ring per request slot, holding
        the fp32 (kv, score) of an even token until its odd partner arrives. Sized
        from num_req_slots, not from a token budget -- the state is request-scoped."""
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
            online=False,
        )

    def _init_paged_compress_states(self, enable_memory_saver: bool):
        total_L = len(self.compression_ratios)
        self.compress_state_pools: List[Optional[CompressStatePool]] = [None] * total_L
        self.indexer_compress_state_pools: List[Optional[CompressStatePool]] = [
            None
        ] * total_L
        pair_sources = self.sources_by_ratio.get(2, [])

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio in (0, 1):
                continue

            if ratio == 2:
                # Only a kv_source layer compresses; the ratio-2 layers after it
                # read its latents and so need no pending-pair state of their own.
                if idx in pair_sources:
                    self.compress_state_pools[idx] = self._make_pair_state_pool(
                        enable_memory_saver
                    )
                continue

            self.compress_state_pools[idx] = self._make_attn_state_pool(
                ratio, enable_memory_saver
            )

            if ratio == 4:
                self.indexer_compress_state_pools[idx] = self._make_indexer_state_pool(
                    ratio, enable_memory_saver
                )

    def _collect_sources_by_ratio(self) -> dict[int, List[int]]:
        """Layers owning compressed storage, per ratio present in this PP stage:
        every layer of ratios 4/128, the kv_source layers of ratios 1/2."""
        stage = range(self._stage_start, self._stage_end)
        for idx in stage:
            ratio = self.compression_ratios[idx]
            if ratio not in (0, 1, 2, 4, 128):
                raise ValueError(f"Unsupported compression ratio: {ratio}")

        sources_by_ratio: dict[int, List[int]] = {}
        for ratio in (4, 128, 1, 2):
            if ratio in (1, 2):
                layers = [
                    l
                    for l in self.kv_source_layers
                    if l in stage and self.compression_ratios[l] == ratio
                ]
            else:
                layers = [l for l in stage if self.compression_ratios[l] == ratio]
            if layers:
                sources_by_ratio[ratio] = layers
        return sources_by_ratio

    def source_layer_of(self, layer_id: int) -> int:
        """The layer owning this layer's compressed storage: itself for ratios 4/128,
        the nearest preceding kv_source layer for ratios 1/2 -- the layer whose
        latents, index keys and pair state this layer reads."""
        ratio = self.compression_ratios[layer_id]
        sources = [l for l in self.sources_by_ratio[ratio] if l <= layer_id]
        assert sources, f"layer {layer_id} (ratio {ratio}) has no kv_source layer"
        return max(sources)

    def _init_compressed_pools(
        self,
        *,
        c4_size: int,
        c128_size: int,
        full_size: Optional[int],
        page_size: int,
        dtype: torch.dtype,
        device: str,
        enable_memory_saver: bool,
        enable_hisparse: bool,
    ) -> None:
        """One FlashMLA-layout KV pool per compress ratio present in this stage, plus
        the packed indexer-K pool every ratio but 128 carries: slot = full-pool token
        loc // ratio, page = page_size // ratio rows, so the pages line up with the
        full pool's."""
        self.kv_pools: dict[int, DeepSeekV4SingleKVPool] = {}
        self.index_pools: dict[int, DeepSeekV4IndexerPool] = {}
        if any(ratio in (1, 2) for ratio in self.sources_by_ratio):
            assert full_size is not None, "low compress ratios need the full pool size"
            assert not self._unified_kv, "unified_kv has no low compress ratio layout"

        kv_pool_size = {4: c4_size, 128: c128_size}
        if not self._unified_kv:
            for ratio, sources in self.sources_by_ratio.items():
                self.kv_pools[ratio] = self._make_kv_pool(
                    size=(
                        kv_pool_size[ratio]
                        if ratio in kv_pool_size
                        else full_size // ratio
                    ),
                    page_size=page_size // ratio,
                    dtype=dtype,
                    layer_num=len(sources),
                    device=device,
                    enable_memory_saver=enable_memory_saver,
                    global_page_size=page_size,
                    cls=(
                        HiSparseC4DevicePool
                        if ratio == 4 and enable_hisparse
                        else DeepSeekV4SingleKVPool
                    ),
                )

        for ratio, sources in self.sources_by_ratio.items():
            if ratio == 128:
                continue
            if ratio == 4:
                self.index_pools[ratio] = self._make_indexer_pool(
                    self.c4_logical_size,
                    page_size // 4,
                    dtype,
                    self.indexer_head_dim,
                    len(sources),
                    device,
                    enable_memory_saver,
                )
                continue
            # Slots remain loc // ratio, with DSV41_INDEX_PAGE_SIZE packed-buffer pages.
            # Reserved FULL page 0 extends real slots past full_size; one index padding
            # page is too small to cover that gap.
            self.index_pools[ratio] = self._make_indexer_pool(
                (full_size + page_size) // ratio,
                DSV41_INDEX_PAGE_SIZE,
                dtype,
                self.indexer_head_dim,
                len(sources),
                device,
                enable_memory_saver,
                force_fp4=True,
            )

    def _init_compressed_layer_mapping(self):
        full_cnt = 0
        total_L = len(self.compression_ratios)
        self.layer_mapping: List[Optional[DeepSeekV4LayerItem]] = [None] * total_L

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio == 0:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=0,
                    compress_layer_id=full_cnt,
                )
                full_cnt += 1
                continue

            sources = self.sources_by_ratio[ratio]
            self.layer_mapping[idx] = DeepSeekV4LayerItem(
                compress_ratio=ratio,
                compress_layer_id=sources.index(self.source_layer_of(idx)),
                compress_kv_pool=self.kv_pools.get(ratio),
            )

    def wait_layer_transfer(self, layer_id: int) -> None:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_attention_compress_states(self, layer_id: int) -> CompressStatePool:
        self.wait_layer_transfer(layer_id)
        compress_state_pool = self.compress_state_pools[layer_id]
        assert compress_state_pool is not None, (
            "Only c4/c128 layers and ratio-2 kv_source layers have attention states."
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

    def clear_c128_req_state(self, req_pool_idx: int) -> None:
        """Reset request-scoped state for one req slot: the C128 ring and the
        ratio-2 pending-pair ring."""
        for pool in self.compress_state_pools:
            if pool is None or pool.ratio not in (2, 128):
                continue

            if pool.ratio == 128 and ONLINE_C128:
                row = pool.kv_score_buffer.kv_score[req_pool_idx]
                head_dim = row.shape[-1] // 3
                row[:head_dim].fill_(float("-inf"))
                row[head_dim:].zero_()
                continue

            start = req_pool_idx * pool.ring_size
            pool.kv_score_buffer[start : start + pool.ring_size].clear()

    def clear_unaccepted_c128_draft_states(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        accept_lens: torch.Tensor,
        num_draft_tokens: int,
    ) -> None:
        """Clear offline C128 ring slots written for rejected speculative tokens."""
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
        if self.request_window is not None:
            return self.request_window.buffer(self._swa_local_layer_id(layer_id))
        return self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)]

    def get_swa_key_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        if self.request_window is not None:
            return self.get_swa_raw_buffer(layer_id).view(
                self.request_window.state.dtype
            )
        return self.swa_kv_pool.get_key_buffer(self._swa_local_layer_id(layer_id))

    def set_swa_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        if self.request_window is not None:
            dsv4_index_buf_accessor.SetKAndS.execute(
                pool=self.request_window.state,
                buf=self.get_swa_raw_buffer(layer_id),
                loc=loc,
                nope_fp8_rope_bf16_pack=cache_nope_fp8_rope_bf16_pack,
            )
        else:
            self.swa_kv_pool.set_key_buffer(
                self._swa_local_layer_id(layer_id), loc, cache_nope_fp8_rope_bf16_pack
            )

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
        assert pool is not None, f"no indexer pool for {compress_ratio = }"
        return pool

    def get_low_ratio_index_k_dequant(
        self, layer_id: int, slots: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Index-K rows at `slots` from the layer's latent source; see
        get_index_k_dequant."""
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        return self._indexer_pool(compress_ratio).get_index_k_dequant(
            compress_layer_id, slots
        )

    def get_low_ratio_index_k_fp4(
        self, layer_id: int, slots: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Packed fp4 index-K rows at `slots` from the layer's latent source:
        (payload int8 [n, 64], ue8m0 scales packed int32 [n]), the kernel input
        layout of quantize_fp4_indexer_tensor."""
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        return self._indexer_pool(compress_ratio).get_index_k_fp4(
            compress_layer_id, slots
        )

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
        self.set_swa_key_buffer(layer_id, swa_loc, cache_nope_fp8_rope_bf16_pack)

    def get_swa_key_buffer_radix(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        if self.request_window is not None:
            return self.get_swa_raw_buffer(layer_id).view(
                self.request_window.state.dtype
            )
        return self.swa_kv_pool.get_key_buffer(self._swa_local_layer_id(layer_id))

    def set_swa_key_buffer_radix_fused(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        return fused_store_cache(
            input=cache_k,
            cache=self.get_swa_raw_buffer(layer_id),
            indices=swa_loc,
            page_size=self.swa_page_size,
            type="flashmla",
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
        fused_k_norm_rope_flashmla(
            kv=kv,
            kv_weight=kv_weight,
            eps=eps,
            freqs_cis=freqs_cis,
            positions=positions,
            out_loc=swa_loc,
            kvcache=self.get_swa_raw_buffer(layer_id),
            page_size=self.swa_page_size,
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
