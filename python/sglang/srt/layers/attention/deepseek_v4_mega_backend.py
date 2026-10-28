from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

HEAD_DIM = 512


def interleave_query_rows(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    return x.view(num_heads, 32, 16, -1).transpose(0, 1).reshape(x.shape)


def pack_query(q: torch.Tensor, padded_heads: int) -> torch.Tensor:
    tokens, heads, _ = q.shape
    out = q.new_zeros(tokens, 32, padded_heads, 16)
    out[:, :, :heads] = q.view(tokens, heads, 32, 16).transpose(1, 2)
    return out.view(tokens, padded_heads, HEAD_DIM)


def interleave_output_columns(w: torch.Tensor) -> torch.Tensor:
    return w.view(*w.shape[:-1], 8, 16, 32).transpose(-3, -2).reshape(w.shape)


def quantize_wo_a(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    groups, rank, _ = weight.shape
    return sglang_per_token_group_quant_fp8(
        weight.contiguous(),
        group_size=32,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
        masked_m=torch.full((groups,), rank, dtype=torch.int32, device=weight.device),
    )


class DeepseekV4MegaAttnBackend(DeepseekV4AttnBackend):
    mega_attn: bool = True

    def _mega_paged_prefill(
        self,
        forward_batch: ForwardBatch,
        *,
        num_qo_tokens: int,
        layer_id: int,
        compress_ratio: int,
        num_heads: int,
    ) -> bool:
        if (
            num_heads != 64
            or compress_ratio not in (1, 2)
            or num_qo_tokens > 1024
            or not forward_batch.forward_mode.is_extend_without_speculative()
            or forward_batch.seq_lens_cpu is None
            or self.token_to_kv_pool.get_extra_key_bytes_per_token(layer_id) != 288
        ):
            return False
        gather_tokens = int(forward_batch.seq_lens_cpu.sum()) // compress_ratio
        return gather_tokens >= max(32768, 192 * num_qo_tokens)

    def _forward_mega_decode(
        self,
        *,
        q: torch.Tensor,
        swa_k_cache: torch.Tensor,
        swa_page_indices: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        attn_sink: torch.Tensor,
        swa_topk_lengths: torch.Tensor,
        extra_k_cache: Optional[torch.Tensor],
        extra_indices: Optional[torch.Tensor],
        extra_topk_lengths: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from sgl_kernel.flash_mla_mega import flash_mla_mega_decode

        data, scale, _ = flash_mla_mega_decode(
            q,
            swa_k_cache,
            swa_page_indices,
            positions,
            cos_sin_cache,
            self.softmax_scale,
            attn_sink=attn_sink,
            topk_length=swa_topk_lengths,
            extra_kv=extra_k_cache,
            extra_indices=extra_indices,
            extra_topk_length=extra_topk_lengths,
        )
        return data, scale

    def _forward_mega_prefill(
        self,
        *,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_length: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from sgl_kernel.flash_mla_mega import flash_mla_mega_prefill

        data, scale, _, _ = flash_mla_mega_prefill(
            q,
            kv,
            indices,
            positions,
            cos_sin_cache,
            self.softmax_scale,
            attn_sink=attn_sink,
            topk_length=topk_length,
        )
        return data, scale
