"""Paged HND KV storage for Cambricon attention kernels."""

from typing import TYPE_CHECKING

import torch
import torch_mlu_ops

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, unwrap_write_loc
from sglang.srt.utils.log_utils import print_warning_once

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention


class MLUMHATokenToKVPool(MHATokenToKVPool):
    """MLU paged HND storage with shared allocation and slot movement."""

    def _create_buffers(self):
        if not self.use_hnd:
            raise NotImplementedError(
                "MLU MHA cache requires shared HND initialization, "
                f"got kv_cache_layout={self.kv_cache_layout!r}. "
                "Leave SGLANG_USE_HND_KVCACHE unset or set it to 1, "
                "and remove any explicit KV cache layout override."
            )
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(
                "MLU MHA cache currently supports only torch.float16 "
                f"and torch.bfloat16, got dtype={self.dtype}."
            )

        if envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
            print_warning_once(
                "MLU KV-write OOB checks are skipped despite "
                "SGLANG_ENABLE_ASYNC_ASSERT=1: torch._assert_async falls back "
                "to CPU on MLU."
            )

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
        if dcp_kv_mask is not None:
            raise NotImplementedError("dcp_kv_mask is not supported for MLU MHA cache.")

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

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

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
            slot_mapping=loc.to(torch.int32),
        )
