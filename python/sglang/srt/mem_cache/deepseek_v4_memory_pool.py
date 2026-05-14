from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Literal, NamedTuple, Optional, Tuple

import torch

from sglang.jit_kernel.deepseek_v4 import fused_k_norm_rope_flashmla, fused_store_cache
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
from sglang.srt.utils import ceil_div

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

        self._create_buffer()

    def _create_buffer(self):
        num_scales_per_token = self.index_head_dim // self.quant_block_size
        page_bytes = self.page_size * self.index_head_dim
        page_bytes += self.page_size * num_scales_per_token * 4
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.index_k_with_scale_buffer = [
                    torch.zeros(
                        (self.size + self.page_size + 1) // self.page_size,
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
        )

        self._init_compressed_layer_mapping()

        self._init_paged_compress_states(enable_memory_saver)

        self._should_cache_swa = envs.SGLANG_OPT_CACHE_SWA_TRANSLATION.get()

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
            ring_size = self.get_ring_size(ratio) if ratio != 0 else 0
            if ratio != 0:
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
            if ratio == 0:
                self.layer_mapping.append(
                    DeepSeekV4LayerItem(
                        compress_ratio=0,
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

    def get_extra_key_page_size(self, layer_id: int) -> int:
        _, _, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.page_size

    def get_extra_key_buffer(self, layer_id: int) -> torch.Tensor:
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

    def get_index_k_page_size(self) -> int:
        return self.c4_indexer_kv_pool.page_size

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

    def set_swa_key_buffer_radix_fused_norm_rope(
        self,
        layer_id: int,
        raw_loc: torch.Tensor,
        kv: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        if self._should_cache_swa:
            if layer_id == self.start_layer or self.cached_loc is None:
                self.cached_loc = self.translate_loc_from_full_to_swa(raw_loc)
            swa_loc = self.cached_loc
        else:
            swa_loc = self.translate_loc_from_full_to_swa(raw_loc)
        fused_k_norm_rope_flashmla(
            kv=kv,
            kv_weight=kv_weight,
            eps=eps,
            freqs_cis=freqs_cis,
            positions=positions,
            out_loc=swa_loc,
            kvcache=self.swa_kv_pool.kv_buffer[layer_id],
            page_size=self.swa_kv_pool.page_size,
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
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.set_index_fused(compress_layer_id, loc, cache_k)
