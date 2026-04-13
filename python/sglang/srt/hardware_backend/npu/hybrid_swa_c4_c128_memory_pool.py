import logging
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union

import torch

from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator
from sglang.srt.hardware_backend.npu.memory_pool_npu import (
    NPUMHATokenToKVPool,
    NPUSingleBufferTokenToKVPool,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool
from sglang.srt.model_executor.forward_batch_info import (
    ExtendNumTokens,
    KvLen,
    LastLoc,
    OutCacheLoc,
)

logger = logging.getLogger(__name__)
GB = 1024 * 1024 * 1024


class MappingIds(IntEnum):
    C4_COMPRESS = 0
    C128_COMPRESS = 1
    C4_STATE = 2
    C128_STATE = 3
    SWA = 4


class SWAC4C128KVPool(KVCache):
    """KV cache with separate pools for full and SWA attention layers."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        size_c4: int,
        size_c128: int,
        size_c4_state: int,
        size_c128_state: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        swa_attention_layer_ids: List[int],
        c4_attention_layer_ids: List[int],
        c128_attention_layer_ids: List[int],
        enable_kvcache_transpose: bool,
        device: str,
        token_to_kv_pool_class: KVCache = NPUSingleBufferTokenToKVPool,
        **kwargs,
    ):
        self.size = size
        self.size_c4 = size_c4
        self.size_c128 = size_c128
        self.size_swa = size_swa
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.device = device
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.c4_layer_nums = len(c4_attention_layer_ids)
        self.c128_layer_nums = len(c128_attention_layer_ids)
        self.start_layer = 0
        self.page_size = page_size
        self.swa_loc = None
        indexer_head_dim = kwargs.get("index_head_dim", 128)
        li_kv_dtype = kwargs.get("li_kv_dtype", "bf16")
        li_kv_dtype = torch.int8 if li_kv_dtype == "int8" else torch.bfloat16

        self.c4_state_size = size_c4_state
        self.c128_state_size = size_c128_state

        logger.info(
            f"[SWAC4C128KVPool] {size=}, {size_swa=}, {size_c4=}, {size_c128=}, {size_c4_state=}, {size_c128_state=}"
        )
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True
        assert not enable_kvcache_transpose

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )
        kwargs_ = {
            "page_size": page_size,
            "head_num": head_num,
            "enable_memory_saver": False,
            "device": device,
        }

        self.dummy_kv_pool = token_to_kv_pool_class(
            size=self.size,
            dtype=dtype,
            layer_num=self.swa_layer_nums,
            slot_dim=0,
            **kwargs_,
        )

        self.swa_kv_pool = token_to_kv_pool_class(
            size=size_swa,
            dtype=dtype,
            layer_num=self.swa_layer_nums,
            slot_dim=head_dim,
            **kwargs_,
        )

        self.c4_kv_pool = token_to_kv_pool_class(
            size=size_c4,
            dtype=dtype,
            layer_num=self.c4_layer_nums,
            slot_dim=head_dim,
            **kwargs_,
        )

        self.c4_index_kv_pool = token_to_kv_pool_class(
            size=size_c4,
            dtype=li_kv_dtype,
            layer_num=self.c4_layer_nums,
            slot_dim=indexer_head_dim,
            **kwargs_,
        )
        overlap_size = 2
        self.state_dtype = torch.float32
        self.c4_state_pool = NPUMHATokenToKVPool(
            size=size_c4_state,
            dtype=self.state_dtype,
            layer_num=self.c4_layer_nums,
            head_dim=head_dim * overlap_size,
            **kwargs_,
        )
        self.c4_index_state_pool = NPUMHATokenToKVPool(
            size=size_c4_state,
            dtype=self.state_dtype,
            layer_num=self.c4_layer_nums,
            head_dim=indexer_head_dim * overlap_size,
            **kwargs_,
        )
        self.c128_kv_pool = token_to_kv_pool_class(
            size=size_c128,
            dtype=dtype,
            layer_num=self.c128_layer_nums,
            slot_dim=head_dim,
            **kwargs_,
        )
        self.c128_state_pool = NPUMHATokenToKVPool(
            size=size_c128_state,
            dtype=self.state_dtype,
            layer_num=self.c128_layer_nums,
            head_dim=head_dim,
            **kwargs_,
        )
        # {layer_id: (index, is_c4, is_c8)}
        self.layers_mapping: Dict[int, Tuple[int, bool, bool]] = {}
        for c4_attn_layer_id, global_layer_id in enumerate(c4_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (c4_attn_layer_id, True, False)
        for c128_layer_id, global_layer_id in enumerate(c128_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (c128_layer_id, False, True)
        self.full_to_attn_index_mapping: Optional[torch.Tensor] = None

        cache_buffer = self.get_total_buffer_size_bytes()
        self.mem_usage = cache_buffer / GB
        logger.info(
            f"SWAC4C128KVPool mem usage: {self.mem_usage:.2f} GB. page_size: {self.page_size}"
        )

    def get_total_buffer_size_bytes(self):
        total_buffer_size = self.swa_kv_pool.get_kv_size_bytes()
        total_buffer_size += self.c4_kv_pool.get_kv_size_bytes()
        total_buffer_size += self.c4_index_kv_pool.get_kv_size_bytes()

        def kv_sum(x):
            if isinstance(x, (list, tuple)):
                return sum(x)
            return x

        total_buffer_size += kv_sum(self.c4_state_pool.get_kv_size_bytes())
        total_buffer_size += kv_sum(self.c4_index_state_pool.get_kv_size_bytes())
        total_buffer_size += self.c128_kv_pool.get_kv_size_bytes()
        total_buffer_size += kv_sum(self.c128_state_pool.get_kv_size_bytes())
        return total_buffer_size

    def get_contiguous_buf_infos(self):
        swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens = self._get_swa_buf_infos()
        c4_kv_data_ptrs, c4_kv_data_lens, c4_kv_item_lens = self._get_c4_buf_infos()
        c4_state_kv_data_ptrs, c4_state_kv_data_lens, c4_state_kv_item_lens = (
            self._get_c4_state_buf_infos()
        )
        c128_kv_data_ptrs, c128_kv_data_lens, c128_kv_item_lens = (
            self._get_c128_buf_infos()
        )
        c128_state_kv_data_ptrs, c128_state_kv_data_lens, c128_state_kv_item_lens = (
            self._get_c128_state_buf_infos()
        )
        kv_data_ptrs = (
            swa_kv_data_ptrs
            + c4_kv_data_ptrs
            + c4_state_kv_data_ptrs
            + c128_kv_data_ptrs
            + c128_state_kv_data_ptrs
        )
        kv_data_lens = (
            swa_kv_data_lens
            + c4_kv_data_lens
            + c4_state_kv_data_lens
            + c128_kv_data_lens
            + c128_state_kv_data_lens
        )
        kv_item_lens = (
            swa_kv_item_lens
            + c4_kv_item_lens
            + c4_state_kv_item_lens
            + c128_kv_item_lens
            + c128_state_kv_item_lens
        )
        # map
        buf_map = dict()
        buf_map["swa_begin"], buf_map["swa_end"] = 0, len(swa_kv_data_ptrs)
        buf_map["c4_begin"], buf_map["c4_end"] = buf_map["swa_end"], buf_map[
            "swa_end"
        ] + len(c4_kv_data_ptrs)
        buf_map["c4_state_begin"], buf_map["c4_state_end"] = buf_map["c4_end"], buf_map[
            "c4_end"
        ] + len(c4_state_kv_data_ptrs)
        buf_map["c128_begin"], buf_map["c128_end"] = buf_map["c4_state_end"], buf_map[
            "c4_state_end"
        ] + len(c128_kv_data_ptrs)
        buf_map["c128_state_begin"], buf_map["c128_state_end"] = buf_map[
            "c128_end"
        ], buf_map["c128_end"] + len(c128_state_kv_data_ptrs)
        return kv_data_ptrs, kv_data_lens, kv_item_lens, buf_map

    def _get_swa_buf_infos(self):
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.swa_kv_pool.get_contiguous_buf_infos()
        )
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def _get_c4_buf_infos(self):
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.c4_kv_pool.get_contiguous_buf_infos()
        )
        index_kv_data_ptrs, index_kv_data_lens, index_kv_item_lens = (
            self.c4_index_kv_pool.get_contiguous_buf_infos()
        )
        kv_data_ptrs += index_kv_data_ptrs
        kv_data_lens += index_kv_data_lens
        kv_item_lens += index_kv_item_lens
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def _get_c4_state_buf_infos(self):
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.c4_state_pool.get_contiguous_buf_infos()
        )
        index_kv_data_ptrs, index_kv_data_lens, index_kv_item_lens = (
            self.c4_index_state_pool.get_contiguous_buf_infos()
        )
        kv_data_ptrs += index_kv_data_ptrs
        kv_data_lens += index_kv_data_lens
        kv_item_lens += index_kv_item_lens
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def _get_c128_buf_infos(self):
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.c128_kv_pool.get_contiguous_buf_infos()
        )
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def _get_c128_state_buf_infos(self):
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.c128_state_pool.get_contiguous_buf_infos()
        )
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        pass

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        pass

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        pass

    def get_swa_buffer(self, layer_id: int, loc: torch.Tensor = None):
        kv = self.swa_kv_pool.get_kv_buffer(layer_id)
        if loc is not None:
            kv = kv.flatten(0, 1)
            kv = kv[loc]
        return kv

    def get_compress_buffer(
        self, layer_id: int, from_indexer: bool = False, loc: torch.Tensor = None
    ):
        if layer_id not in self.layers_mapping:
            return None
        layer_id_pool, is_c4_layer, is_c128_layer = self.layers_mapping[layer_id]
        if is_c4_layer:
            compress_kv_pool = (
                self.c4_index_kv_pool if from_indexer else self.c4_kv_pool
            )
            kv = compress_kv_pool.get_kv_buffer(layer_id_pool)
        else:
            kv = self.c128_kv_pool.get_kv_buffer(layer_id_pool)
        if loc is not None:
            kv = kv.flatten(0, 1)
            kv = kv[loc]
        return kv  # [num_tokens, head_num(1), dim]

    def get_compress_state_buffer(
        self, layer_id: int, from_indexer: bool = False, loc: torch.Tensor = None
    ):
        layer_id_pool, is_c4_layer, is_c128_layer = self.layers_mapping[layer_id]
        if is_c4_layer:
            state_pool = (
                self.c4_index_state_pool if from_indexer else self.c4_state_pool
            )
            kv_state, score_state = state_pool.get_kv_buffer(layer_id_pool)
        else:
            kv_state, score_state = self.c128_state_pool.get_kv_buffer(layer_id_pool)

        if loc is not None:
            kv_state = kv_state.flatten(0, 1)
            kv_state = kv_state[loc]
            score_state = score_state.flatten(0, 1)
            score_state = score_state[loc]
        return kv_state, score_state

    def get_compress_dequant_scale_buffer(
        self, layer_id: int, from_indexer: bool = False, loc: torch.Tensor = None
    ):
        if layer_id not in self.layers_mapping:
            return None
        layer_id_pool, is_c4_layer, is_c128_layer = self.layers_mapping[layer_id]
        assert is_c4_layer and from_indexer

        dequant_scale = self.c4_index_kv_pool.get_scale_buffer(layer_id_pool)
        if loc is not None:
            dequant_scale = dequant_scale.flatten(0, 1)
            dequant_scale = dequant_scale[loc]
        return dequant_scale  # [num_tokens, head_num(1), dim(1)] or [num_block, page_size, 1, 1]

    def set_swa_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache: torch.Tensor,
        scale: float = 1.0,
    ):
        self.swa_kv_pool.set_kv_buffer(layer, loc, cache, scale)

    def set_compress_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache: torch.Tensor,
        scale: torch.Tensor = None,
        from_indexer: bool = False,
    ):
        layer_id_pool, is_c4_layer, is_c128_layer = self.layers_mapping[layer_id]
        if is_c4_layer:
            compress_kv_pool = (
                self.c4_index_kv_pool if from_indexer else self.c4_kv_pool
            )
            compress_kv_pool.set_kv_buffer(
                None,
                loc,
                cache,
                scale,
                layer_id_override=layer_id_pool,
            )
        else:
            self.c128_kv_pool.set_kv_buffer(
                None,
                loc,
                cache,
                scale,
                layer_id_override=layer_id_pool,
            )

    def set_compress_state_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_score: torch.Tensor,
        scale: Optional[float] = None,
        from_indexer: bool = False,
    ):
        assert (
            cache_k.dim() == 3
            and cache_score.dim() == 3
            and cache_k.shape[1] == 1
            and cache_score.shape[1] == 1
        )
        layer_id_pool, is_c4_layer, is_c128_layer = self.layers_mapping[layer_id]
        if is_c4_layer:
            state_pool = (
                self.c4_state_pool if not from_indexer else self.c4_index_state_pool
            )
            state_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_score,
                layer_id_override=layer_id_pool,
            )
        else:
            self.c128_state_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_score,
                layer_id_override=layer_id_pool,
            )

    def get_cpu_copy(self, indices):
        pass

    def load_cpu_copy(self, kv_cache_cpu, indices):
        pass


class SWAC4C128TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Allocator for SWA hybrid KV cache."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        size_c4: int,
        size_c128: int,
        size_c4_state: int,
        size_c128_state: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: SWAC4C128KVPool,
        need_sort: bool,
    ):
        assert isinstance(kvcache, SWAC4C128KVPool)
        self._size = size
        self._size_c4 = size_c4
        self._size_c128 = size_c128
        self._size_swa = size_swa
        self._size_c4_state = size_c4_state
        self._size_c128_state = size_c128_state
        self.dtype = dtype
        self.device = device
        self.page_size = page_size
        self.sliding_window_size = 128
        assert size_c4_state == kvcache.c4_state_pool.size
        assert size_c128_state == kvcache.c128_state_pool.size

        kwargs = {"device": device, "need_sort": need_sort}
        if page_size == 1:
            allocator_cls = TokenToKVPoolAllocator
        else:
            allocator_cls = NPUPagedTokenToKVPoolAllocator
            kwargs.update({"page_size": page_size})

        self.dummy_attn_allocator = allocator_cls(
            size=size,
            dtype=dtype,
            kvcache=kvcache.dummy_kv_pool,
            **kwargs,
        )
        self.swa_attn_allocator = allocator_cls(
            size=size_swa,
            dtype=dtype,
            kvcache=kvcache.swa_kv_pool,
            **kwargs,
        )
        self.c4_attn_allocator = allocator_cls(
            size=size_c4,
            dtype=dtype,
            kvcache=kvcache.c4_kv_pool,
            **kwargs,
        )
        self.c4_state_allocator = allocator_cls(
            size=size_c4_state,
            dtype=kvcache.state_dtype,
            kvcache=kvcache.c4_state_pool,
            **kwargs,
        )
        self.c128_attn_allocator = allocator_cls(
            size=size_c128,
            dtype=dtype,
            kvcache=kvcache.c128_kv_pool,
            **kwargs,
        )
        self.c128_state_allocator = allocator_cls(
            size=size_c128_state,
            dtype=kvcache.state_dtype,
            kvcache=kvcache.c128_state_pool,
            **kwargs,
        )
        self._kvcache = kvcache
        self.full_to_attn_index_mapping = None

        self.need_sort = need_sort
        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

        self.clear()

    def available_size(self):
        return self.dummy_attn_allocator.available_size()

    def c4_available_size(self):
        return self.c4_attn_allocator.available_size()

    def c4_state_available_size(self):
        return self.c4_state_allocator.available_size()

    def c128_available_size(self):
        return self.c128_attn_allocator.available_size()

    def c128_state_available_size(self):
        return self.c128_state_allocator.available_size()

    def swa_available_size(self):
        return self.swa_attn_allocator.available_size()

    @property
    def size(self):
        return min(self._size_swa, self._size_c4, self._size_c128)

    @property
    def size_swa(self):
        return self._size_swa

    @property
    def size_c128(self):
        return self._size_c128

    @property
    def size_c4(self):
        return self._size_c4

    def debug_print(self) -> str:
        msg = ""
        msg += f"#swa-available-size: {self.swa_attn_allocator.available_size()}, "
        msg += f"#c4-attn-available-size: {self.c4_attn_allocator.available_size()}, "
        msg += (
            f"#c128-attn-available-size: {self.c128_attn_allocator.available_size()}, "
        )
        return msg

    def get_kvcache(self):
        return self._kvcache

    def alloc(
        self, need_size: int, seqlens: Union[list[int], torch.Tensor], is_prefill=True
    ):
        assert self.page_size == 1, "alloc pagesize=1 not implemented"

    def alloc_extend(
        self,
        prefix_lens: KvLen,
        prefix_lens_cpu: KvLen,
        seq_lens: KvLen,
        seq_lens_cpu: KvLen,
        last_loc: LastLoc,  # last_loc for full layers
        extend_num_tokens: ExtendNumTokens,
    ):
        assert self.page_size > 1
        num_tokens = (
            extend_num_tokens.full_extend_num_tokens
            + len(seq_lens.full_kv_len) * self.page_size
        )
        if num_tokens > self.dummy_attn_allocator.available_size():
            logger.info(f"{num_tokens=}, {self.dummy_attn_allocator.available_size()=}")
            return None

        # extend_lens_cpu = seq_lens_cpu - prefix_lens_cpu
        # extend_lens = seq_lens - prefix_lens
        if (
            extend_num_tokens.c4_extend_num_tokens
            > self.c4_attn_allocator.available_size()
        ):
            logger.info(
                f"{extend_num_tokens.c4_extend_num_tokens=}, {self.c4_attn_allocator.available_size()=}"
            )
            return None
        if (
            extend_num_tokens.c128_extend_num_tokens
            > self.c128_attn_allocator.available_size()
        ):
            logger.info(
                f"{extend_num_tokens.c128_extend_num_tokens=}, {self.c128_attn_allocator.available_size()=}"
            )
            return None

        alloc_full_indices = self.dummy_attn_allocator.alloc_extend(
            prefix_lens.full_kv_len,
            prefix_lens_cpu.full_kv_len,
            seq_lens.full_kv_len,
            seq_lens_cpu.full_kv_len,
            last_loc.last_loc,
            extend_num_tokens.full_extend_num_tokens,
        )
        assert alloc_full_indices is not None

        # swa prefill save 128 tokens, verify save extend_num
        alloc_swa_indices = self.swa_attn_allocator.alloc_extend(
            prefix_lens.swa_kv_len,
            prefix_lens_cpu.swa_kv_len,
            seq_lens.swa_kv_len,
            seq_lens_cpu.swa_kv_len,
            last_loc.last_swa_loc,
            extend_num_tokens.swa_extend_num_tokens,
        )
        assert alloc_swa_indices is not None

        if extend_num_tokens.c4_extend_num_tokens > 0:
            alloc_c4_indices = self.c4_attn_allocator.alloc_extend(
                prefix_lens.c4_kv_len,
                prefix_lens_cpu.c4_kv_len,
                seq_lens.c4_kv_len,
                seq_lens_cpu.c4_kv_len,
                last_loc.last_c4_loc,
                extend_num_tokens.c4_extend_num_tokens,
            )
            assert alloc_c4_indices is not None
        else:
            alloc_c4_indices = torch.empty((0,), dtype=torch.int64, device=self.device)

        if extend_num_tokens.c128_extend_num_tokens > 0:
            alloc_c128_indices = self.c128_attn_allocator.alloc_extend(
                prefix_lens.c128_kv_len,
                prefix_lens_cpu.c128_kv_len,
                seq_lens.c128_kv_len,
                seq_lens_cpu.c128_kv_len,
                last_loc.last_c128_loc,
                extend_num_tokens.c128_extend_num_tokens,
            )
            assert alloc_c128_indices is not None
        else:
            alloc_c128_indices = torch.empty(
                (0,), dtype=torch.int64, device=self.device
            )

        alloc_c4_state_indices = self.c4_state_allocator.alloc_extend(
            prefix_lens.c4_state_kv_len,
            prefix_lens_cpu.c4_state_kv_len,
            seq_lens.c4_state_kv_len,
            seq_lens_cpu.c4_state_kv_len,
            last_loc.last_c4_state_loc,
            extend_num_tokens.c4_state_extend_num_tokens,
        )
        assert alloc_c4_state_indices is not None

        alloc_c128_state_indices = self.c128_state_allocator.alloc_extend(
            prefix_lens.c128_state_kv_len,
            prefix_lens_cpu.c128_state_kv_len,
            seq_lens.c128_state_kv_len,
            seq_lens_cpu.c128_state_kv_len,
            last_loc.last_c128_state_loc,
            extend_num_tokens.c128_state_extend_num_tokens,
        )
        assert alloc_c128_state_indices is not None

        alloc_loc = OutCacheLoc(
            alloc_full_indices,
            alloc_swa_indices,
            alloc_c4_indices,
            alloc_c128_indices,
            alloc_c4_state_indices,
            alloc_c128_state_indices,
        )
        return alloc_loc

    def alloc_decode(
        self,
        prefix_lens: KvLen,
        prefix_lens_cpu: KvLen,
        seq_lens: KvLen,
        seq_lens_cpu: KvLen,
        last_loc: LastLoc,  # last_loc for full layers
        extend_num_tokens: ExtendNumTokens,
    ):
        assert self.page_size > 1

        if (
            extend_num_tokens.c4_extend_num_tokens
            > self.c4_attn_allocator.available_size()
        ):
            logger.info(
                f"{extend_num_tokens.c4_extend_num_tokens=}, {self.c4_attn_allocator.available_size()=}"
            )
            return None
        if (
            extend_num_tokens.c128_extend_num_tokens
            > self.c128_attn_allocator.available_size()
        ):
            logger.info(
                f"{extend_num_tokens.c128_extend_num_tokens=}, {self.c128_attn_allocator.available_size()=}"
            )
            return None

        alloc_full_indices = self.dummy_attn_allocator.alloc_decode(
            seq_lens.full_kv_len, seq_lens_cpu.full_kv_len, last_loc.last_loc
        )

        alloc_swa_indices = self.swa_attn_allocator.alloc_decode(
            seq_lens.swa_kv_len, seq_lens_cpu.swa_kv_len, last_loc.last_swa_loc
        )

        if extend_num_tokens.c4_extend_num_tokens > 0:
            alloc_c4_indices = self.c4_attn_allocator.alloc_extend(
                prefix_lens.c4_kv_len,
                prefix_lens_cpu.c4_kv_len,
                seq_lens.c4_kv_len,
                seq_lens_cpu.c4_kv_len,
                last_loc.last_c4_loc,
                extend_num_tokens.c4_extend_num_tokens,
            )
            assert alloc_c4_indices is not None
        else:
            alloc_c4_indices = torch.empty((0,), dtype=torch.int64, device=self.device)

        if extend_num_tokens.c128_extend_num_tokens > 0:
            alloc_c128_indices = self.c128_attn_allocator.alloc_extend(
                prefix_lens.c128_kv_len,
                prefix_lens_cpu.c128_kv_len,
                seq_lens.c128_kv_len,
                seq_lens_cpu.c128_kv_len,
                last_loc.last_c128_loc,
                extend_num_tokens.c128_extend_num_tokens,
            )
            assert alloc_c128_indices is not None
        else:
            alloc_c128_indices = torch.empty(
                (0,), dtype=torch.int64, device=self.device
            )

        alloc_c4_state_indices = self.c4_state_allocator.alloc_decode(
            seq_lens.c4_state_kv_len,
            seq_lens_cpu.c4_state_kv_len,
            last_loc.last_c4_state_loc,
        )
        assert (
            alloc_c4_state_indices is not None
        ), f"{seq_lens_cpu=}, {last_loc.last_c4_state_loc=}, {self.c4_state_allocator.free_pages=}, {self.c4_state_allocator.free_group=}, {self.c4_state_allocator.release_pages=}"

        alloc_c128_state_indices = self.c128_state_allocator.alloc_decode(
            seq_lens.c128_state_kv_len,
            seq_lens_cpu.c128_state_kv_len,
            last_loc.last_c128_state_loc,
        )
        assert (
            alloc_c128_state_indices is not None
        ), f"{seq_lens_cpu=}, {last_loc.last_c128_state_loc=}, {self.c128_state_allocator.free_pages=}, {self.c128_state_allocator.free_group=}, {self.c128_state_allocator.release_pages=}"

        alloc_loc = OutCacheLoc(
            alloc_full_indices,
            alloc_swa_indices,
            alloc_c4_indices,
            alloc_c128_indices,
            alloc_c4_state_indices,
            alloc_c128_state_indices,
        )
        return alloc_loc

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            indices = [torch.cat(item) for item in map(list, zip(*self.free_group))]
            self.swa_c4_c128_free(*indices)

    def free(self, free_index: torch.Tensor):
        raise NotImplementedError(
            "SWAC4C128TokenToKVPoolAllocator does not support original free Method"
        )

    # ensure all the indices input is right
    def swa_c4_c128_free(
        self,
        dummy_index,
        swa_kv_indices,
        c4_kv_indices,
        c128_kv_indices,
        c4_state_kv_indices,
        c128_state_kv_indices,
    ):
        if (
            dummy_index.numel() == 0
            and c4_state_kv_indices.numel() == 0
            and c128_state_kv_indices.numel() == 0
            and swa_kv_indices.numel() == 0
            and c4_kv_indices.numel() == 0
            and c128_kv_indices.numel() == 0
        ):
            return
        if self.is_not_in_free_group:
            self.free_swa(swa_kv_indices)
            self.free_compress(c4_kv_indices, "c4")
            self.free_compress(c128_kv_indices, "c128")
            self.free_compress_state(c4_state_kv_indices, "c4")
            self.free_compress_state(c128_state_kv_indices, "c128")
            self.dummy_attn_allocator.free(dummy_index)
        else:
            self.free_group.append(
                [
                    dummy_index,
                    swa_kv_indices,
                    c4_kv_indices,
                    c128_kv_indices,
                    c4_state_kv_indices,
                    c128_state_kv_indices,
                ]
            )

        assert self.swa_attn_allocator.available_size() <= self.swa_attn_allocator.size
        assert (
            self.c4_attn_allocator.available_size() <= self.c4_attn_allocator.size
        ), f"{self.c4_attn_allocator.size=}, {self.c4_attn_allocator.free_pages=}, {self.c4_attn_allocator.free_group=}, {len(self.c4_attn_allocator.free_pages)=} + {self.c4_attn_allocator.release_pages=}"
        assert (
            self.c128_attn_allocator.available_size() <= self.c128_attn_allocator.size
        ), f"{self.c128_attn_allocator.size=}, {self.c128_attn_allocator.free_pages=}, {self.c128_attn_allocator.free_group=}, {self.c128_attn_allocator.release_pages=}"

    def free_swa(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if self.swa_attn_allocator.debug_mode:
            assert torch.all(free_index >= self.page_size), f"{free_index=}"
        self.swa_attn_allocator.free(free_index)

    def free_compress_state(self, free_index: torch.Tensor, free_type=None):
        if free_index.numel() == 0:
            return
        if free_type is None:
            free_type = ["c4", "c128"]
        if "c4" in free_type:
            if self.c4_state_allocator.debug_mode:
                assert torch.all(free_index >= self.page_size), f"{free_index=}"
            self.c4_state_allocator.free(free_index)
        if "c128" in free_type:
            if self.c128_state_allocator.debug_mode:
                assert torch.all(free_index >= self.page_size), f"{free_index=}"
            self.c128_state_allocator.free(free_index)

    def free_compress(self, free_index: torch.Tensor, free_type=None):
        if free_index.numel() == 0:
            return
        if free_type is None:
            free_type = ["c4", "c128"]

        if "c4" in free_type:
            if self.c4_attn_allocator.debug_mode:
                assert torch.all(free_index >= self.page_size), f"{free_index=}"
            self.c4_attn_allocator.free(free_index)

        if "c128" in free_type:
            if self.c128_attn_allocator.debug_mode:
                assert torch.all(free_index >= self.page_size), f"{free_index=}"
            self.c128_attn_allocator.free(free_index)

    def backup_state(self):
        return [
            self.c4_attn_allocator.backup_state(),
            self.swa_attn_allocator.backup_state(),
        ]

    def restore_state(self, state):
        assert len(state) == 2
        self.c4_attn_allocator.restore_state(state[0])
        self.swa_attn_allocator.restore_state(state[1])

    def clear(self):
        self.dummy_attn_allocator.clear()
        self.swa_attn_allocator.clear()
        self.c4_attn_allocator.clear()
        self.c128_attn_allocator.clear()
        self.c4_state_allocator.clear()
        self.c128_state_allocator.clear()
        # Note: the last item is -1, we don't clear it, see the comment in __init__
        self.is_not_in_free_group = True
        self.free_group = []

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)
