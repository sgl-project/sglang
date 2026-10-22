"""Paged HND KV storage for Cambricon attention kernels."""

from typing import TYPE_CHECKING

import torch
import torch_mlu_ops

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, unwrap_write_loc

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention


class MLUMHATokenToKVPool(MHATokenToKVPool):
    """MLU paged HND storage with shared allocation and slot movement."""

    def _create_buffers(self):
        if self.store_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("MLU MHA cache requires FP16 or BF16 storage.")
        self.use_hnd = True
        self.kv_cache_layout = "hnd"
        self.num_pages = self.size // self.page_size + 1
        super()._create_buffers()

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float | None = None,
        v_scale: float | None = None,
        layer_id_override: int | None = None,
        dcp_kv_mask: torch.Tensor | None = None,
    ):
        """Write MHA key/value tokens into the MLU paged cache."""
        loc, _, _ = unwrap_write_loc(loc_info)
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id

        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if dcp_kv_mask is not None:
            loc = loc[dcp_kv_mask]
            cache_k = cache_k[dcp_kv_mask]
            cache_v = cache_v[dcp_kv_mask]

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        slot_mapping = loc.to(torch.int32).contiguous()

        # kv cache shape: (block_num, head_num, block_size, head_size)
        torch_mlu_ops.reshape_paged_cache(
            k=cache_k,
            v=cache_v,
            k_cache=self.k_buffer[layer_id - self.start_layer].view(
                -1, self.head_num, self.page_size, self.head_dim
            ),
            v_cache=self.v_buffer[layer_id - self.start_layer].view(
                -1, self.head_num, self.page_size, self.v_head_dim
            ),
            slot_mapping=slot_mapping,
        )
