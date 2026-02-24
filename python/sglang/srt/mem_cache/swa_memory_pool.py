import logging
import weakref
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import KVCache, MHATokenToKVPool
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool

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

    def get_cpu_copy(self, indices):
        # For SWA, we need to copy KV cache from both full and SWA pools
        # The indices are for the full pool, and we use mapping to get SWA indices
        full_kv_cpu = self.full_kv_pool.get_cpu_copy(indices)

        # Get SWA indices through the mapping
        # Note: SWA allocation always creates 1:1 mapping, so no need to filter
        if self.full_to_swa_index_mapping is not None:
            swa_indices = self.full_to_swa_index_mapping[indices]
            swa_kv_cpu = self.swa_kv_pool.get_cpu_copy(swa_indices)
        else:
            swa_kv_cpu = None

        return {"full": full_kv_cpu, "swa": swa_kv_cpu}

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
        kvcache: SWAKVPool,
        need_sort: bool,
    ):
        assert isinstance(kvcache, SWAKVPool)
        self._size_full = size
        self._size_swa = size_swa
        self.dtype = dtype
        self.device = device
        self.page_size = page_size

        if page_size == 1:
            self.full_attn_allocator = TokenToKVPoolAllocator(
                size,
                dtype,
                device,
                kvcache.full_kv_pool,
                need_sort,
            )
            self.swa_attn_allocator = TokenToKVPoolAllocator(
                size_swa,
                dtype,
                device,
                kvcache.swa_kv_pool,
                need_sort,
            )
        else:
            self.full_attn_allocator = PagedTokenToKVPoolAllocator(
                size,
                page_size,
                dtype,
                device,
                kvcache.full_kv_pool,
                need_sort,
            )
            self.swa_attn_allocator = PagedTokenToKVPoolAllocator(
                size_swa,
                page_size,
                dtype,
                device,
                kvcache.swa_kv_pool,
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
        num_tokens = extend_num_tokens + len(seq_lens) * self.page_size
        if num_tokens > self.full_attn_allocator.available_size():
            return None
        if num_tokens > self.swa_attn_allocator.available_size():
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

    def free(self, free_index: torch.Tensor, swa_evicted_count: int = 0):
        """Free full-attention and SWA KV slots.

        SWA layers only attend to a sliding window, so as a request grows the
        scheduler calls _evict_swa() to reclaim SWA slots for tokens that have
        fallen outside the window.  _evict_swa() zeros their entries in
        full_to_swa_index_mapping to mark them as already freed.

        When the request eventually finishes, free() is called for ALL of its
        full-attention indices (evicted + live).  We must NOT pass the
        already-evicted indices to free_swa() because their mapping value is 0,
        and 0 is the dummy/padding slot reserved by the allocator - freeing it
        would put it back in the free-page list and corrupt future allocations.

        swa_evicted_count is the number of tokens at the START of free_index
        whose SWA slots were already freed mid-request.  We skip them by
        slicing with a plain Python int (CPU metadata, zero GPU sync), avoiding
        the alternative of a boolean-mask gather which forces a CPU-GPU sync to
        determine the output tensor shape.
        """
        if free_index.numel() == 0:
            return

        # NOTE: the API is not idempotent.
        if self.is_not_in_free_group:
            self.full_attn_allocator.free(free_index)
            self.free_swa(free_index[swa_evicted_count:])
        else:
            # Store swa_evicted_count alongside the indices so free_group_end
            # can replay both full and SWA frees with the correct split.
            self.free_group.append((free_index, swa_evicted_count))
        assert (
            self.full_attn_allocator.available_size() <= self.full_attn_allocator.size
        )
        assert self.swa_attn_allocator.available_size() <= self.swa_attn_allocator.size

    @staticmethod
    def _page_select(entry_lens, page_size, device):
        """Build a selection index tensor for batched paged free.

        Problem: PagedTokenToKVPoolAllocator.free() extracts page indices via
        (free_index // page_size)[::page_size], which assumes the input is a
        single contiguous allocation where every page contributes exactly
        page_size consecutive elements.  When free_group_end concatenates
        free_index tensors from multiple finished requests, sub-page tails
        (e.g. 3 tokens from the last partial page) break this uniform stride
        and cause the [::page_size] step to skip pages, leaking them.

        Solution: build a per-page stride array from the known entry lengths.
        Full pages contribute stride = page_size, sub-page residuals contribute
        stride = residual length.  An exclusive cumsum over this array gives the
        offset of the first element of each page in the concatenated tensor.
        Passing this as page_select to free() replaces the [::page_size] stride.

        Example with page_size=128, entries of length [256, 3, 384]:
          Entry 0 (256): 2 full pages  -> strides [128, 128]
          Entry 1 (3):   0 full + tail -> strides [3]
          Entry 2 (384): 3 full pages  -> strides [128, 128, 128]

          strides        = [128, 128,   3, 128, 128, 128]
          cumsum         = [128, 256, 259, 387, 515, 643]
          cumsum - strides = [  0, 128, 256, 259, 387, 515]  <- selection indices

        These indices pick position 0 (page 0 of entry 0), 128 (page 1 of
        entry 0), 256 (residual of entry 1), 259 (page 0 of entry 2), etc.
        Each picked position is the first element of its page, so
        (combined // page_size)[sel] gives the correct page IDs.

        The strides array is built on CPU from Python ints (tensor.numel() is
        always CPU), transferred to GPU via pinned memory + non_blocking copy,
        and cumsummed on GPU.  No GPU-to-CPU sync.
        """
        strides = []
        for length in entry_lens:
            num_full_pages = length // page_size
            residual = length % page_size
            strides.extend([page_size] * num_full_pages)
            if residual > 0:
                strides.append(residual)
        if not strides:
            return None
        strides_tensor = torch.tensor(strides, dtype=torch.int64, pin_memory=True)
        strides_gpu = strides_tensor.to(device, non_blocking=True)
        # Exclusive cumsum: cumsum gives end-of-block offsets, subtracting
        # the stride itself gives start-of-block (first element of each page).
        return strides_gpu.cumsum(0) - strides_gpu

    def free_group_end(self):
        """Flush deferred free() calls from the free_group batch.

        During extend/decode output processing, free() calls are deferred via
        free_group_begin/end to avoid interleaving frees with still-running
        kernels.  Each deferred entry stores (free_index, swa_evicted_count).

        We concatenate all entries and make one batched free() call per
        allocator (full and SWA), using _page_select to handle the paged
        stride correctly across entries with different lengths.  SWA entries
        are sliced by swa_evicted_count to exclude already-evicted tokens
        whose mapping was zeroed by _evict_swa mid-request.
        """
        self.is_not_in_free_group = True
        if self.free_group:
            # Full-attention: cat all free_index tensors and free in one call.
            full_parts = [free_index for free_index, _ in self.free_group]
            combined_full = torch.cat(full_parts)
            # page_select avoids torch.unique for paged allocators (page_size>1);
            # TokenToKVPoolAllocator (page_size=1) doesn't need it.
            if self.page_size > 1:
                full_page_select = self._page_select(
                    [free_index.numel() for free_index in full_parts],
                    self.page_size,
                    combined_full.device,
                )
                self.full_attn_allocator.free(combined_full, page_select=full_page_select)
            else:
                self.full_attn_allocator.free(combined_full)

            # SWA: slice off already-evicted prefix per entry, cat the live
            # portions, look up SWA indices via the mapping, and free in one call.
            swa_live_parts = [
                free_index[swa_evicted_count:]
                for free_index, swa_evicted_count in self.free_group
                if swa_evicted_count < free_index.numel()
            ]
            if swa_live_parts:
                swa_live_full = torch.cat(swa_live_parts)
                if self.page_size > 1:
                    swa_page_select = self._page_select(
                        [part.numel() for part in swa_live_parts],
                        self.page_size,
                        swa_live_full.device,
                    )
                    self.free_swa(swa_live_full, page_select=swa_page_select)
                else:
                    self.free_swa(swa_live_full)

    def free_swa(self, free_index: torch.Tensor, page_select: torch.Tensor = None):
        """Free SWA slots corresponding to the given full-pool indices.

        Args:
            free_index: full-pool indices whose SWA mapping is still live.
            page_select: optional selection indices for the paged SWA allocator,
                as returned by _page_select(). Passed through to
                swa_attn_allocator.free() to handle batched sub-page entries.
        """
        swa_indices = self.full_to_swa_index_mapping[free_index]
        # PagedTokenToKVPoolAllocator (page_size>1) accepts page_select;
        # TokenToKVPoolAllocator (page_size=1) does not.
        if page_select is not None:
            self.swa_attn_allocator.free(swa_indices, page_select=page_select)
        else:
            self.swa_attn_allocator.free(swa_indices)
        self.full_to_swa_index_mapping[free_index] = swa_indices.zero_()

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
