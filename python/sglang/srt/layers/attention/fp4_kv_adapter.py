from __future__ import annotations

from typing import Callable

import torch


class FlashInferNVFP4KVAdapter:
    """Prepare NVFP4 KV cache views for FlashInfer prefill/extend."""

    def __init__(self, token_to_kv_pool, req_to_token_pool, page_size: int):
        self.access = token_to_kv_pool.get_quantized_kv_access()
        self.req_to_token_pool = req_to_token_pool
        self.page_size = page_size

    def prepare_extend_kv_cache(
        self,
        layer,
        forward_batch,
        *,
        use_ragged: bool,
        dq_page_table,
        cpu_req_pool_indices,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dq_page_table is not None:
            # For paged path, attention reads prefix + current chunk from the
            # dequant workspace. Ragged path handles current chunk with raw k/v.
            transfer_cur_kv = not use_ragged
            k_cur_fp8 = (
                k.to(torch.float8_e4m3fn)
                if (k is not None and transfer_cur_kv)
                else None
            )
            v_cur_fp8 = (
                v.to(torch.float8_e4m3fn)
                if (v is not None and transfer_cur_kv)
                else None
            )
            self.access.prepare_fp8_extend_workspace(
                layer.layer_id,
                layer.layer_id,
                self.req_to_token_pool.req_to_token,
                cpu_req_pool_indices,
                forward_batch.extend_prefix_lens_cpu,
                forward_batch.extend_seq_lens_cpu,
                self.page_size,
                k_cur_fp8=k_cur_fp8,
                v_cur_fp8=v_cur_fp8,
            )

        k_buffer_dq, v_buffer_dq = self.access.fp8_workspace()
        return (
            k_buffer_dq.view(-1, layer.tp_k_head_num, layer.head_dim),
            v_buffer_dq.view(-1, layer.tp_v_head_num, layer.head_dim),
        )


class TRTLLMMHANVFP4KVAdapter:
    """Prepare raw NVFP4 KV cache views for TRTLLM MHA decode."""

    def __init__(self, token_to_kv_pool):
        self.access = token_to_kv_pool.get_quantized_kv_access()

    def bmm_scales(self, layer) -> tuple[float, float]:
        return self.access.quant_method.get_bmm_scales(layer.layer_id)

    def prepare_decode_kv_cache(
        self,
        layer,
        reshape_paged_kv_cache: Callable[
            [torch.Tensor, torch.Tensor, object, int],
            tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        raw = self.access.raw_fp4_view(layer.layer_id)
        kv_cache = reshape_paged_kv_cache(
            raw.k, raw.v, layer, layer.head_dim // 2
        )
        kv_cache_block_scales = reshape_paged_kv_cache(
            raw.k_scale, raw.v_scale, layer, layer.head_dim // 16
        )
        return kv_cache, kv_cache_block_scales
