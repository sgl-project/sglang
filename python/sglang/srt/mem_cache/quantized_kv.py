from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass(frozen=True)
class RawFP4KVCache:
    """Raw packed FP4 KV tensors for one logical layer."""

    k: torch.Tensor
    v: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "k": self.k,
            "v": self.v,
            "k_scale": self.k_scale,
            "v_scale": self.v_scale,
        }


class QuantizedKVCacheAccess:
    """Narrow FP4 KV cache capability exposed by token-to-KV pools.

    The pool owns physical buffers and token/page layout. The quant method owns
    FP4 math. This access object is the small surface attention backends use for
    raw FP4 decode or FP8 prefill workspace preparation.
    """

    def __init__(self, pool):
        self.pool = pool

    @property
    def quant_method(self):
        return self.pool.quant_method

    @property
    def recipe(self) -> str:
        return self.quant_method.name

    def _local_layer_id(self, layer_id: int) -> int:
        return layer_id - self.pool.start_layer

    def _global_scale(self, global_layer_id: int, k_scale, v_scale):
        if k_scale is None and hasattr(self.quant_method, "k_scales_gpu"):
            k_scale = self.quant_method.k_scales_gpu[
                global_layer_id : global_layer_id + 1
            ]
            v_scale = self.quant_method.v_scales_gpu[
                global_layer_id : global_layer_id + 1
            ]
        return k_scale, v_scale

    def write(
        self,
        layer_id: int,
        global_layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale=None,
        v_scale=None,
        device_module=None,
        alt_stream=None,
    ) -> None:
        local_layer_id = self._local_layer_id(layer_id)
        k_scale, v_scale = self._global_scale(global_layer_id, k_scale, v_scale)
        self.quant_method.quantize_and_store(
            self.pool.k_buffer[local_layer_id],
            self.pool.v_buffer[local_layer_id],
            (
                self.pool.k_scale_buffer[local_layer_id]
                if self.pool.k_scale_buffer
                else None
            ),
            (
                self.pool.v_scale_buffer[local_layer_id]
                if self.pool.v_scale_buffer
                else None
            ),
            loc,
            cache_k,
            cache_v,
            k_scale,
            v_scale,
            device_module=device_module,
            alt_stream=alt_stream,
        )

    def raw_fp4_view(self, layer_id: int) -> RawFP4KVCache:
        local_layer_id = self._local_layer_id(layer_id)
        if self.pool.k_scale_buffer is None or self.pool.v_scale_buffer is None:
            raise RuntimeError("Raw FP4 KV cache requested from a non-FP4 KV pool.")
        return RawFP4KVCache(
            k=self.pool.k_buffer[local_layer_id],
            v=self.pool.v_buffer[local_layer_id],
            k_scale=self.pool.k_scale_buffer[local_layer_id].view(torch.float8_e4m3fn),
            v_scale=self.pool.v_scale_buffer[local_layer_id].view(torch.float8_e4m3fn),
        )

    def fp8_workspace(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pool.dq_k_buffer is None or self.pool.dq_v_buffer is None:
            raise RuntimeError(
                "Dequant workspace requested from a KV pool without FP4 dequant buffers."
            )
        return self.pool.dq_k_buffer, self.pool.dq_v_buffer

    def prepare_fp8_extend_workspace(
        self,
        layer_id: int,
        global_layer_id: int,
        req_to_token: torch.Tensor,
        req_pool_indices_cpu,
        extend_prefix_lens_cpu,
        extend_seq_lens_cpu,
        page_size: int,
        k_cur_fp8: Optional[torch.Tensor] = None,
        v_cur_fp8: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.raw_fp4_view(layer_id)
        dq_k, dq_v = self.fp8_workspace()

        cur_batch_start_loc_cpu = 0
        cur_token_idx_dq = page_size

        for i in range(len(req_pool_indices_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            prev_len = int(extend_prefix_lens_cpu[i])
            extend_len = int(extend_seq_lens_cpu[i])

            if prev_len > 0:
                prev_indices = req_to_token[req_idx, :prev_len]
                k_prev_fp8, v_prev_fp8 = self.quant_method.dequantize_prev_kv(
                    raw.k[prev_indices],
                    raw.k_scale[prev_indices],
                    raw.v[prev_indices],
                    raw.v_scale[prev_indices],
                    global_layer_id,
                )
                dq_k[cur_token_idx_dq : cur_token_idx_dq + prev_len] = k_prev_fp8
                dq_v[cur_token_idx_dq : cur_token_idx_dq + prev_len] = v_prev_fp8

            if k_cur_fp8 is not None:
                cur_end = cur_batch_start_loc_cpu + extend_len
                dst_start = cur_token_idx_dq + prev_len
                dst_end = dst_start + extend_len
                dq_k[dst_start:dst_end] = k_cur_fp8[cur_batch_start_loc_cpu:cur_end]
                dq_v[dst_start:dst_end] = v_cur_fp8[cur_batch_start_loc_cpu:cur_end]
                cur_batch_start_loc_cpu = cur_end

            cur_token_idx_dq = (
                (cur_token_idx_dq + prev_len + extend_len + page_size - 1)
                // page_size
                * page_size
            )

        return dq_k, dq_v


class LayerMappedQuantizedKVCacheAccess:
    """Layer-id adapter for hybrid pools that store full-attention layers densely."""

    def __init__(
        self,
        base: QuantizedKVCacheAccess,
        map_layer_id: Callable[[int], int],
        wait_for_layer: Callable[[int], None],
    ):
        self.base = base
        self.map_layer_id = map_layer_id
        self.wait_for_layer = wait_for_layer

    @property
    def quant_method(self):
        return self.base.quant_method

    @property
    def recipe(self) -> str:
        return self.base.recipe

    def _map(self, layer_id: int) -> int:
        self.wait_for_layer(layer_id)
        return self.map_layer_id(layer_id)

    def write(self, layer_id: int, global_layer_id: int, *args, **kwargs) -> None:
        local_layer_id = self._map(layer_id)
        self.base.write(local_layer_id, global_layer_id, *args, **kwargs)

    def raw_fp4_view(self, layer_id: int) -> RawFP4KVCache:
        return self.base.raw_fp4_view(self._map(layer_id))

    def fp8_workspace(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.base.fp8_workspace()

    def prepare_fp8_extend_workspace(
        self, layer_id: int, global_layer_id: int, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_layer_id = self._map(layer_id)
        return self.base.prepare_fp8_extend_workspace(
            local_layer_id, global_layer_id, *args, **kwargs
        )
