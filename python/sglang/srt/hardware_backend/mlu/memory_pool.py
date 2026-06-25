from typing import TYPE_CHECKING, Optional

import torch
import torch_mlu_ops
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention


class MLUMHATokenToKVPool(MHATokenToKVPool):
    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # Allocate one extra page beyond the logical token capacity. This
            # matches the base MHA pool's `size + page_size` guard capacity:
            # slot 0 is reserved for padded/dummy tokens, and the remaining
            # extra capacity keeps paged slot mapping in-bounds at page edges.
            allocated_page_num = self.size // self.page_size + 1
            # The layout of kv cache is changed from per-layer contiguous
            # token buffers to one contiguous paged buffer:
            # - [2, layer_num, allocated_page_num, head_num, page_size, head_dim]
            # Note: in vllm, the layout of kv cache is:
            # dict{layer_id: (2, page_num, head_num, page_size, head_dim)}
            # Continuous memory improves the efficiency of MLU transmission backend,
            # while other backends remain unchanged.
            self.kv_buffer = torch.zeros(
                (
                    2,
                    self.layer_num,
                    allocated_page_num,
                    self.head_num,
                    self.page_size,
                    self.head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.k_buffer = self.kv_buffer[0]
            self.v_buffer = self.kv_buffer[1]

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, head_num, page_size, head_dim]
        kv_data_ptrs = [
            self.get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self.get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self.get_key_buffer(i)[0].nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i)[0].nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id

        if self.store_dtype != self.dtype:
            cache_k = cache_k.to(self.store_dtype)
            cache_v = cache_v.to(self.store_dtype)

        # kv cache shape: (block_num, head_num, block_size, head_size)
        torch_mlu_ops.reshape_paged_cache(
            k=cache_k,
            v=cache_v,
            k_cache=self.k_buffer[layer_id - self.start_layer].view(
                -1, self.head_num, self.page_size, self.head_dim
            ),
            v_cache=self.v_buffer[layer_id - self.start_layer].view(
                -1, self.head_num, self.page_size, self.head_dim
            ),
            slot_mapping=loc.to(torch.int32),
        )
