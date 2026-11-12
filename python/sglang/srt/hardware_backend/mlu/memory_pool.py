"""Paged HND KV storage for Cambricon attention kernels."""

from typing import TYPE_CHECKING

import torch
import torch_mlu_ops

from sglang.kernels.ops.kvcache.mla_buffer_mlu import get_mla_kv_buffer_mlu
from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    unwrap_write_loc,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.async_probe import maybe_detect_oob
from sglang.srt.utils.log_utils import print_warning_once

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.mem_cache.memory_pool import KVWriteLoc


class MLUMHATokenToKVPool(MHATokenToKVPool):
    """MLU paged HND storage with shared allocation and slot movement."""

    def _create_buffers(self) -> None:
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
        loc_info: "torch.Tensor | KVWriteLoc",
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float | None = None,
        v_scale: float | None = None,
        layer_id_override: int | None = None,
        dcp_kv_mask: torch.Tensor | None = None,
    ) -> None:
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


class MLUMLATokenToKVPool(MLATokenToKVPool):
    """Token-linear latent storage; TMO consumes a zero-copy paged view."""

    def _create_buffers(self) -> None:
        if self.dtype != torch.bfloat16 or self.use_dsa:
            raise NotImplementedError("MLU MLA requires BF16 dense latent KV cache.")
        super()._create_buffers()

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc_info: "torch.Tensor | KVWriteLoc",
        cache_k: torch.Tensor,
        cache_v: torch.Tensor | None,
        layer_id_override: int | None = None,
    ) -> None:
        loc, _, _ = unwrap_write_loc(loc_info)
        self._check_physical_write_loc(loc_info, "set_kv_buffer (MLA)")
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MLA)")
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        assert not self.dsa_kv_cache_store_fp8
        # No DCP-aware variant is possible: the two backends reaching this door
        # disagree on the loc space (flashinfer-MLA widened, Triton collapsed).
        assert (
            self.write_loc_is_dcp_resolved
            or self.dcp_replicated
            or not get_parallel().dcp_enabled
        ), (
            "MLATokenToKVPool.set_kv_buffer has no DCP-aware write path. Under "
            "--dcp-size > 1 the MLA write must go through set_mla_kv_buffer, "
            "whose kernel resolves the owner rule; reaching the combined-row "
            "door means an attention backend took a write path that never "
            "declared which loc space it emits."
        )
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)

        torch_mlu_ops.reshape_paged_cache(
            k=cache_k,
            v=None,
            k_cache=self.kv_buffer[layer_id - self.start_layer].view(
                -1, 1, self.page_size, self.kv_cache_dim
            ),
            v_cache=None,
            slot_mapping=loc.to(torch.int32),
        )

    def get_mla_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        dst_dtype: torch.dtype | None = None,
        layer_id_override: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # get k nope and k rope from the kv buffer, and optionally cast them to dst_dtype.
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        kv_buffer = self.get_key_buffer(layer_id)
        dst_dtype = dst_dtype or self.dtype
        cache_k_nope = torch.empty(
            (loc.shape[0], 1, self.kv_lora_rank),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        if self.qk_rope_head_dim == 0:
            cache_k_rope = None
        else:
            cache_k_rope = torch.empty(
                (loc.shape[0], 1, self.qk_rope_head_dim),
                dtype=dst_dtype,
                device=kv_buffer.device,
            )
        get_mla_kv_buffer_mlu(kv_buffer, loc, cache_k_nope, cache_k_rope)
        return cache_k_nope, cache_k_rope
