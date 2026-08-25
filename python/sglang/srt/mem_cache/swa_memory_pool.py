import logging
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    KVCache,
    MHATokenToKVPool,
    MLATokenToKVPool,
    unwrap_write_loc,
)
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool

logger = logging.getLogger(__name__)
GB = 1024 * 1024 * 1024


class SWAKVPool(BaseSWAKVPool):
    """Hybrid full/SWA cache composed from independently configurable pools.

    The default remains two MHA pools. Supplying ``full_kv_pool_class`` and
    ``swa_kv_pool_class`` enables other KV cache families, including MLA/DSA,
    without adding model-specific behavior to the pool selector.
    """

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
        device: str,
        token_to_kv_pool_class: KVCache = MHATokenToKVPool,
        full_kv_pool_class: Optional[type] = None,
        swa_kv_pool_class: Optional[type] = None,
        full_kv_pool_kwargs: Optional[dict] = None,
        swa_kv_pool_kwargs: Optional[dict] = None,
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
        self.layer_num = self.full_layer_nums + self.swa_layer_nums
        self.start_layer = 0
        self.page_size = page_size
        self.layer_transfer_counter = None

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        full_kv_pool_class = full_kv_pool_class or token_to_kv_pool_class
        swa_kv_pool_class = swa_kv_pool_class or token_to_kv_pool_class
        common_kwargs = {
            "page_size": page_size,
            "enable_memory_saver": False,
            "device": device,
        }
        if full_kv_pool_kwargs is None:
            full_kv_pool_kwargs = {
                **common_kwargs,
                "head_num": head_num,
                "head_dim": head_dim,
                "allocation_label": "Full",
                **kwargs,
            }
            full_kv_pool_kwargs.pop("swa_head_num", None)
            full_kv_pool_kwargs.pop("swa_head_dim", None)
            full_kv_pool_kwargs.pop("swa_v_head_dim", None)
        if swa_kv_pool_kwargs is None:
            swa_kv_pool_kwargs = {
                **common_kwargs,
                "head_num": head_num,
                "head_dim": head_dim,
                "allocation_label": "SWA",
                **kwargs,
            }

        self.full_kv_pool = full_kv_pool_class(
            size=size,
            dtype=dtype,
            layer_num=self.full_layer_nums,
            **full_kv_pool_kwargs,
        )
        self.swa_kv_pool = swa_kv_pool_class(
            size=size_swa,
            dtype=dtype,
            layer_num=self.swa_layer_nums,
            **swa_kv_pool_kwargs,
        )
        self.dsa_kv_cache_store_fp8 = False
        self.kv_cache_dim = None
        self.index_head_dim = None
        if isinstance(self.full_kv_pool, MLATokenToKVPool):
            self.dsa_kv_cache_store_fp8 = self.full_kv_pool.dsa_kv_cache_store_fp8
            self.kv_cache_dim = self.full_kv_pool.kv_cache_dim
        if isinstance(self.full_kv_pool, DSATokenToKVPool):
            self.index_head_dim = self.full_kv_pool.index_head_dim
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
            f"SWAKVPool {'VA upper bound' if self.post_capture_active else 'mem usage'}: {self.mem_usage:.2f} GB, swa size: {self.size_swa}, full size: {self.size}"
        )

    @property
    def post_capture_active(self) -> bool:
        """True iff the sub-pools took the post-capture VA-backed path (both share the flag)."""
        return self.full_kv_pool.post_capture_active

    @property
    def post_capture_backed_bytes(self) -> int:
        """Physically-backed KV bytes across both sub-pools (post-capture only)."""
        return (
            self.full_kv_pool.post_capture_backed_bytes
            + self.swa_kv_pool.post_capture_backed_bytes
        )

    def finalize_backing(self, config) -> None:
        """Back both sub-pools to their post-capture final sizes and record them."""
        self.full_kv_pool._finalize_backing_tokens(config.full_max_total_num_tokens)
        self.swa_kv_pool._finalize_backing_tokens(config.swa_max_total_num_tokens)
        self.size = int(config.full_max_total_num_tokens)
        self.size_swa = int(config.swa_max_total_num_tokens)

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def register_layer_transfer_counter(self, layer_transfer_counter):
        # Wait happens at this wrapper. Inner pools must not wait again.
        self.layer_transfer_counter = layer_transfer_counter
        self.full_kv_pool.register_layer_transfer_counter(None)
        self.swa_kv_pool.register_layer_transfer_counter(None)

    def _wait_for_layer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_kv_size_bytes(self):
        def split_size(pool):
            size = pool.get_kv_size_bytes()
            return size if isinstance(size, tuple) else (size, 0)

        k_size, v_size = split_size(self.full_kv_pool)
        k_size_swa, v_size_swa = split_size(self.swa_kv_pool)
        return k_size + k_size_swa, v_size + v_size_swa

    def is_mla(self) -> bool:
        return isinstance(self.full_kv_pool, MLATokenToKVPool) and isinstance(
            self.swa_kv_pool, MLATokenToKVPool
        )

    def get_contiguous_buf_infos(self):
        full_kv_data_ptrs, full_kv_data_lens, full_kv_item_lens = (
            self.full_kv_pool.get_contiguous_buf_infos()
        )
        return (
            full_kv_data_ptrs,
            full_kv_data_lens,
            full_kv_item_lens,
        )

    def get_kv_scale_buf_infos(self):
        return self.full_kv_pool.get_kv_scale_buf_infos()

    def get_swa_kv_scale_buf_infos(self):
        return self.swa_kv_pool.get_kv_scale_buf_infos()

    def get_state_buf_infos(self):
        swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens = (
            self.swa_kv_pool.get_contiguous_buf_infos()
        )

        return swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens

    def get_key_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_key_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_key_buffer(layer_id_pool)

    def get_value_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_value_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_value_buffer(layer_id_pool)

    def get_kv_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_kv_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_kv_buffer(layer_id_pool)

    def get_kv_scale_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_kv_scale_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_kv_scale_buffer(layer_id_pool)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        assert self.full_to_swa_index_mapping is not None
        # -1 in kv_indices maps to -1 via the sentinel appended to the mapping.
        return self.full_to_swa_index_mapping[kv_indices]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        # loc_info bundles the full loc and the pre-translated SWA loc.
        loc, swa_loc, _ = unwrap_write_loc(loc_info)
        layer_id = layer.layer_id
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa_layer else self.full_kv_pool
        if is_swa_layer:
            # swa_loc is the full->SWA translation, computed once per forward by
            # the attention backend; set_kv_buffer never translates internally.
            assert swa_loc is not None
            loc = swa_loc
        if isinstance(pool, MLATokenToKVPool):
            pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                layer_id_override=layer_id_pool,
            )
        else:
            pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
            )

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        loc, swa_loc, _ = unwrap_write_loc(loc_info)
        layer_id_pool, is_swa_layer = self.layers_mapping[layer.layer_id]
        pool = self.swa_kv_pool if is_swa_layer else self.full_kv_pool
        if is_swa_layer:
            assert swa_loc is not None
            loc = swa_loc
        if not isinstance(pool, MLATokenToKVPool):
            raise TypeError(f"Layer {layer.layer_id} is not backed by an MLA KV pool")
        pool.set_mla_kv_buffer(
            None,
            loc,
            cache_k_nope,
            cache_k_rope,
            layer_id_override=layer_id_pool,
        )

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        assert not is_swa_layer
        return self.full_kv_pool.get_index_k_with_scale_buffer(layer_id_pool)

    def get_index_k_continuous(self, layer_id: int, *args, **kwargs):
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        assert not is_swa_layer
        return self.full_kv_pool.get_index_k_continuous(layer_id_pool, *args, **kwargs)

    def get_index_k_scale_continuous(self, layer_id: int, *args, **kwargs):
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        assert not is_swa_layer
        return self.full_kv_pool.get_index_k_scale_continuous(
            layer_id_pool, *args, **kwargs
        )

    def get_index_k_scale_buffer(self, layer_id: int, *args, **kwargs):
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        assert not is_swa_layer
        return self.full_kv_pool.get_index_k_scale_buffer(
            layer_id_pool, *args, **kwargs
        )

    def set_index_k_scale_buffer(self, layer_id: int, *args, **kwargs):
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        assert not is_swa_layer
        return self.full_kv_pool.set_index_k_scale_buffer(
            layer_id_pool, *args, **kwargs
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

            # A chunk is whatever the sub-pool produced: k/v, plus the block
            # scales for a quantized pool. Filter every tensor it carries.
            num_tensors = len(layer_chunks[0])
            tensors = [
                torch.cat([chunk[t] for chunk in layer_chunks], dim=0)[row_mask]
                for t in range(num_tensors)
            ]

            filtered_layer = []
            for i in range(0, len(tensors[0]), chunk_size):
                filtered_layer.append([t[i : i + chunk_size] for t in tensors])
            filtered.append(filtered_layer)
        return filtered

    def get_cpu_copy(self, indices, mamba_indices=None):
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

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
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
