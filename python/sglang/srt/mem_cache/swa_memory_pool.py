import logging
import weakref
from typing import Dict, List, Optional, Tuple, Union

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.deepseekv4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import KVCache, MHATokenToKVPool
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool
from sglang.srt.utils.common import get_num_new_pages

logger = logging.getLogger(__name__)
GB = 1024 * 1024 * 1024


class SWAKVPool(KVCache):
    """KV cache with separate pools for full and SWA attention layers."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        swa_attention_layer_ids: List[int],
        full_attention_layer_ids: List[int],
        enable_kvcache_transpose: bool,
        device: str,
        token_to_kv_pool_class: KVCache = MHATokenToKVPool,
        **kwargs,
    ):
        self.size = size
        self.size_swa = size_swa
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.device = device
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        self.start_layer = 0
        self.page_size = page_size
        self.swa_loc = None

        kwargs["page_size"] = page_size
        kwargs["enable_memory_saver"] = False
        kwargs["head_num"] = head_num
        kwargs["head_dim"] = head_dim
        kwargs["device"] = device
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True
        assert not enable_kvcache_transpose

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        self.swa_kv_pool = token_to_kv_pool_class(
            size=size_swa,
            dtype=dtype,
            layer_num=self.swa_layer_nums,
            **kwargs,
        )
        kwargs.pop("swa_head_num", None)
        kwargs.pop("swa_head_dim", None)
        kwargs.pop("swa_v_head_dim", None)
        self.full_kv_pool = token_to_kv_pool_class(
            size=size,
            dtype=dtype,
            layer_num=self.full_layer_nums,
            **kwargs,
        )
        # {layer_id: (index, is_swa_layer)}
        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        for full_attn_layer_id, global_layer_id in enumerate(full_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (full_attn_layer_id, False)
        for swa_layer_id, global_layer_id in enumerate(swa_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (swa_layer_id, True)
        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None

        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB
        logger.info(
            f"SWAKVPool mem usage: {self.mem_usage:.2f} GB, swa size: {self.size_swa}, full size: {self.size}"
        )

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def get_kv_size_bytes(self):
        k_size, v_size = self.full_kv_pool.get_kv_size_bytes()
        k_size_swa, v_size_swa = self.swa_kv_pool.get_kv_size_bytes()
        return k_size + k_size_swa, v_size + v_size_swa

    def get_contiguous_buf_infos(self):
        full_kv_data_ptrs, full_kv_data_lens, full_kv_item_lens = (
            self.full_kv_pool.get_contiguous_buf_infos()
        )
        return (
            full_kv_data_ptrs,
            full_kv_data_lens,
            full_kv_item_lens,
        )

    def get_state_buf_infos(self):
        swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens = (
            self.swa_kv_pool.get_contiguous_buf_infos()
        )

        return swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens

    def get_key_buffer(self, layer_id: int):
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_key_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_key_buffer(layer_id_pool)

    def get_value_buffer(self, layer_id: int):
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_value_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_value_buffer(layer_id_pool)

    def get_kv_buffer(self, layer_id: int):
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_kv_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_kv_buffer(layer_id_pool)

    def set_swa_loc(self, loc: torch.Tensor):
        self.swa_loc = loc

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None

        # Note: kv_indices could have -1 values (from alloc_extend), which will be mapped to -1
        # since the last item of full_to_swa_index_mapping is -1.
        return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):

        layer_id = layer.layer_id
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            if self.swa_loc is not None:
                loc = self.swa_loc
            else:
                if self.full_to_swa_index_mapping is not None:
                    loc = self.translate_loc_from_full_to_swa(loc)

            self.swa_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
            )
        else:
            self.full_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
            )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)
        tgt_loc_swa = self.translate_loc_from_full_to_swa(tgt_loc)
        src_loc_swa = self.translate_loc_from_full_to_swa(src_loc)
        self.swa_kv_pool.move_kv_cache(tgt_loc_swa, src_loc_swa)

    def _filter_swa_cpu_copy(self, swa_kv_cpu, row_mask: torch.Tensor):
        if swa_kv_cpu is None:
            return None
        if row_mask is None or bool(torch.all(row_mask).item()):
            return swa_kv_cpu

        chunk_size = getattr(
            self.swa_kv_pool, "cpu_offloading_chunk_size", len(row_mask)
        )
        filtered = []
        for layer_chunks in swa_kv_cpu:
            if len(layer_chunks) == 0:
                filtered.append([])
                continue

            k_cpu = torch.cat([chunk[0] for chunk in layer_chunks], dim=0)
            v_cpu = torch.cat([chunk[1] for chunk in layer_chunks], dim=0)
            k_cpu = k_cpu[row_mask]
            v_cpu = v_cpu[row_mask]

            filtered_layer = []
            for i in range(0, len(k_cpu), chunk_size):
                filtered_layer.append(
                    [k_cpu[i : i + chunk_size], v_cpu[i : i + chunk_size]]
                )
            filtered.append(filtered_layer)
        return filtered

    def get_cpu_copy(self, indices):
        # For SWA, we need to copy KV cache from both full and SWA pools
        # The indices are for the full pool, and we use mapping to get SWA indices
        full_kv_cpu = self.full_kv_pool.get_cpu_copy(indices)

        swa_mask = None
        if self.full_to_swa_index_mapping is not None:
            swa_indices = self.full_to_swa_index_mapping[indices]
            # Slot 0 is reserved as a dummy slot. Tail-only SWA allocations leave
            # the out-of-window full KV indices unmapped, so only copy mapped SWA
            # tokens and keep their positions for load_cpu_copy().
            swa_mask = swa_indices > 0
            if torch.any(swa_mask):
                swa_kv_cpu = self.swa_kv_pool.get_cpu_copy(swa_indices[swa_mask])
                swa_mask = swa_mask.cpu()
            else:
                swa_kv_cpu = None
        else:
            swa_kv_cpu = None

        return {"full": full_kv_cpu, "swa": swa_kv_cpu, "swa_mask": swa_mask}

    def load_cpu_copy(self, kv_cache_cpu, indices):
        # Load KV cache back from CPU to both full and SWA pools
        # Note: indices here are NEW indices (newly allocated), different from get_cpu_copy indices
        full_kv_cpu = kv_cache_cpu["full"]
        swa_kv_cpu = kv_cache_cpu["swa"]

        # Load full KV cache to the new indices
        self.full_kv_pool.load_cpu_copy(full_kv_cpu, indices)

        # Load SWA KV cache if it exists
        if swa_kv_cpu is not None and self.full_to_swa_index_mapping is not None:
            swa_indices = self.full_to_swa_index_mapping[indices]
            new_swa_mask = swa_indices > 0
            old_swa_mask = kv_cache_cpu.get("swa_mask")
            if old_swa_mask is not None:
                old_swa_mask = old_swa_mask.to(indices.device)
                row_mask = new_swa_mask[old_swa_mask].cpu()
                swa_indices = swa_indices[old_swa_mask][row_mask.to(indices.device)]
            else:
                row_mask = new_swa_mask.cpu()
                swa_indices = swa_indices[new_swa_mask]

            if swa_indices.numel() == 0:
                return

            swa_kv_cpu = self._filter_swa_cpu_copy(swa_kv_cpu, row_mask)
            self.swa_kv_pool.load_cpu_copy(swa_kv_cpu, swa_indices)


class SWATokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Allocator for SWA hybrid KV cache."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: Union[SWAKVPool, DeepSeekV4TokenToKVPool],
        need_sort: bool,
    ):
        assert isinstance(kvcache, (SWAKVPool, DeepSeekV4TokenToKVPool))
        self._size_full = size
        self._size_swa = size_swa
        self.dtype = dtype
        self.device = device
        self.page_size = page_size

        full_kv_pool = getattr(kvcache, "full_kv_pool", None)
        swa_kv_pool = getattr(kvcache, "swa_kv_pool", None)

        if page_size == 1:
            self.full_attn_allocator = TokenToKVPoolAllocator(
                size,
                dtype,
                device,
                full_kv_pool,
                need_sort,
            )
            self.swa_attn_allocator = TokenToKVPoolAllocator(
                size_swa,
                dtype,
                device,
                swa_kv_pool,
                need_sort,
            )
        else:
            self.full_attn_allocator = PagedTokenToKVPoolAllocator(
                size,
                page_size,
                dtype,
                device,
                full_kv_pool,
                need_sort,
            )
            self.swa_attn_allocator = PagedTokenToKVPoolAllocator(
                size_swa,
                page_size,
                dtype,
                device,
                swa_kv_pool,
                need_sort,
            )
        # Note: append one more item of value -1 in the end so -1 maps to -1.
        # It is needed for the last_loc in alloc_extend, where the first full_last_loc
        # is -1, and we need to map it to swa_last_loc -1 as well.
        self.full_to_swa_index_mapping = torch.cat(
            [
                torch.zeros(
                    size + self.page_size,
                    dtype=torch.int64,
                    device=device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=device),
            ]
        )

        self.need_sort = need_sort
        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

        self.clear()
        self._kvcache = kvcache
        self._kvcache.register_mapping(weakref.proxy(self.full_to_swa_index_mapping))

    def available_size(self):
        return min(
            self.full_attn_allocator.available_size(),
            self.swa_attn_allocator.available_size(),
        )

    def full_available_size(self):
        return self.full_attn_allocator.available_size()

    def swa_available_size(self):
        return self.swa_attn_allocator.available_size()

    @property
    def size(self):
        return min(self._size_full, self._size_swa)

    @property
    def size_swa(self):
        return self._size_swa

    @property
    def size_full(self):
        return self._size_full

    def debug_print(self) -> str:
        msg = ""
        msg += f"#swa-available-size: {self.swa_attn_allocator.available_size()}, "
        msg += (
            f"#full-attn-available-size: {self.full_attn_allocator.available_size()}, "
        )
        return msg

    def get_kvcache(self):
        return self._kvcache

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self._kvcache.full_to_swa_index_mapping is not None
        return self._kvcache.translate_loc_from_full_to_swa(kv_indices)

    def alloc(self, need_size: int):
        assert self.page_size == 1
        if need_size > self.full_attn_allocator.available_size():
            return None
        if need_size > self.swa_attn_allocator.available_size():
            return None

        alloc_full_indices = self.full_attn_allocator.alloc(need_size)
        alloc_swa_indices = self.swa_attn_allocator.alloc(need_size)
        assert alloc_full_indices is not None
        assert alloc_swa_indices is not None

        self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices
        return alloc_full_indices

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
        extend_num_tokens: int,
    ):
        assert self.page_size > 1

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        if num_new_pages > self.full_attn_allocator.available_size() // self.page_size:
            return None
        if num_new_pages > self.swa_attn_allocator.available_size() // self.page_size:
            return None

        swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)

        alloc_full_indices = self.full_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        alloc_swa_indices = self.swa_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            swa_last_loc,
            extend_num_tokens,
        )
        assert alloc_full_indices is not None
        assert alloc_swa_indices is not None

        self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices

        return alloc_full_indices

    def alloc_extend_swa_tail(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
        extend_num_tokens: int,
        swa_tail_len: int,
    ):
        """Allocate full KV for the whole extend and SWA KV only for the tail.

        This is used by disaggregated decode preallocation: decode receives full
        prompt KV for full-attention layers, but only the sliding-window state is
        transferred for SWA layers.
        """
        assert self.page_size > 1
        assert len(seq_lens_cpu) == 1, "SWA tail allocation currently supports bs=1"
        assert len(prefix_lens_cpu) == 1
        assert 0 <= swa_tail_len <= extend_num_tokens

        num_full_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        num_swa_pages = (swa_tail_len + self.page_size - 1) // self.page_size
        if num_full_pages > self.full_attn_allocator.available_size() // self.page_size:
            return None
        if num_swa_pages > self.swa_attn_allocator.available_size() // self.page_size:
            return None

        alloc_full_indices = self.full_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        assert alloc_full_indices is not None

        if swa_tail_len == 0:
            return alloc_full_indices

        device = self.device
        swa_prefix_lens = torch.zeros((1,), dtype=torch.int64, device=device)
        swa_prefix_lens_cpu = torch.zeros((1,), dtype=torch.int64)
        swa_seq_lens = torch.tensor([swa_tail_len], dtype=torch.int64, device=device)
        swa_seq_lens_cpu = torch.tensor([swa_tail_len], dtype=torch.int64)
        swa_last_loc = torch.tensor([-1], dtype=torch.int64, device=device)

        alloc_swa_indices = self.swa_attn_allocator.alloc_extend(
            swa_prefix_lens,
            swa_prefix_lens_cpu,
            swa_seq_lens,
            swa_seq_lens_cpu,
            swa_last_loc,
            swa_tail_len,
        )
        assert alloc_swa_indices is not None

        self.full_to_swa_index_mapping[alloc_full_indices[-swa_tail_len:]] = (
            alloc_swa_indices
        )
        return alloc_full_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
    ):
        assert self.page_size > 1
        swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)

        alloc_full_indices = self.full_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )
        alloc_swa_indices = self.swa_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, swa_last_loc
        )

        if alloc_full_indices is None or alloc_swa_indices is None:
            return None

        self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices

        return alloc_full_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        # NOTE: the API is not idempotent.
        if self.is_not_in_free_group:
            self.full_attn_allocator.free(free_index)
            self.free_swa(free_index)
        else:
            self.free_group.append(free_index)
        assert (
            self.full_attn_allocator.available_size() <= self.full_attn_allocator.size
        )
        assert self.swa_attn_allocator.available_size() <= self.swa_attn_allocator.size

    def free_swa(self, free_index: torch.Tensor):
        swa_indices = self.full_to_swa_index_mapping[free_index]
        swa_indices = swa_indices[swa_indices > 0]
        self.swa_attn_allocator.free(swa_indices)
        self.full_to_swa_index_mapping[free_index] = 0

    def backup_state(self):
        return [
            self.full_attn_allocator.backup_state(),
            self.swa_attn_allocator.backup_state(),
        ]

    def restore_state(self, state):
        assert len(state) == 2
        self.full_attn_allocator.restore_state(state[0])
        self.swa_attn_allocator.restore_state(state[1])

    def clear(self):
        self.swa_attn_allocator.clear()
        self.full_attn_allocator.clear()
        # Note: the last item is -1, we don't clear it, see the comment in __init__
        self.full_to_swa_index_mapping[:-1].fill_(0)
        self.is_not_in_free_group = True
        self.free_group = []

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)
