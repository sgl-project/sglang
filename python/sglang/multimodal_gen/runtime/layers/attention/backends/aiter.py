# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import importlib
import logging
import os

import aiter
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.aiter import (
    USE_AITER_GFX95,
    USE_AITER_GFX942,
)

logger = logging.getLogger(__name__)

_use_fp8_attn = os.environ.get("SGLANG_DIFFUSION_AITER_FP8_ATTN", "0") == "1"
_fp8_dtype = torch.float8_e4m3fn

# fmha_fwd_hd128_fp8_gfx950 ASM kernel. Support full MHA with q/k/v head_dim == 128 -- e.g., Wan 2.2 self- and cross-attention.
_FMHA_FP8_HEAD_DIM = 128


if _use_fp8_attn:
    logger.info("DiT FP8 attention enabled via SGLANG_DIFFUSION_AITER_FP8_ATTN=1")


def _can_use_fmha_fp8_prefill(
    q_head_dim: int,
    k_head_dim: int,
    v_head_dim: int,
    num_heads: int,
    num_kv_heads: int,
) -> bool:
    """True if MHA q/k/v head_dim==128 on a gfx950-class arch."""
    if not USE_AITER_GFX95:
        return False
    if num_kv_heads != num_heads:
        return False
    return q_head_dim == k_head_dim == v_head_dim == _FMHA_FP8_HEAD_DIM


def _fmha_fp8_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    is_causal: bool,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> torch.Tensor:
    """
    FP8 FMHA prefill via aiter.flash_attn_fp8_pertensor_func.

    Expects q, k, v as (batch, seqlen, nheads, 128) FP8, contiguous.
    """

    def _ensure_fp8_descale(scale: torch.Tensor) -> torch.Tensor:
        """Per-tensor descale as shape (1,) float32 for flash_attn_fp8_pertensor_func."""
        return scale.to(dtype=torch.float32).reshape(1).contiguous()

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    q_descale = _ensure_fp8_descale(q_scale)
    k_descale = _ensure_fp8_descale(k_scale)
    v_descale = _ensure_fp8_descale(v_scale)

    return aiter.flash_attn_fp8_pertensor_func(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        causal=is_causal,
        softmax_scale=softmax_scale,
        window_size=(-1, -1, 0),
    )


class AITerBackend(AttentionBackend):
    """
    Backend for AITemplate attention implementation.
    """

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.AITER

    @staticmethod
    def get_impl_cls() -> type["AITerImpl"]:
        return AITerImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        # AITer backend does not require special metadata.
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError("AITer backend does not have a metadata builder.")


class AITerImpl(AttentionImpl):
    """
    Implementation of attention using AITemplate.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        dropout_p: float = 0.0,
        **extra_impl_args,
    ) -> None:
        # aiter's mha entry points take GQA/MQA K/V directly (they broadcast
        # each KV head across its group of query heads), so the only
        # requirement is an even split. The FP8 ASM path is MHA-only and
        # already routes grouped shapes back to BF16 below.
        if num_kv_heads is not None and num_heads % num_kv_heads != 0:
            raise ValueError(
                f"AITer backend requires num_heads ({num_heads}) to be a "
                f"multiple of num_kv_heads ({num_kv_heads})."
            )
        self.causal = causal
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale

    @torch.compiler.disable
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """
        Performs attention using one of:
          - _fmha_fp8_prefill_attention (FP8, SGLANG_DIFFUSION_AITER_FP8_ATTN=1 when eligible)
          - flash_attn_func (BF16, default or FP8 fallback for unsupported shapes)

        Args:
            query: Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            key: Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
            value: Value tensor of shape [batch_size, seq_len, num_heads, head_dim]
            attn_metadata: Metadata for the attention operation (unused).

        Returns:
            Output tensor of shape [batch_size, seq_len, num_heads, head_dim]
        """
        if _use_fp8_attn:
            if query.dtype != _fp8_dtype:
                q_fp8, q_scale = aiter.per_tensor_quant(query, quant_dtype=_fp8_dtype)
                k_fp8, k_scale = aiter.per_tensor_quant(key, quant_dtype=_fp8_dtype)
                v_fp8, v_scale = aiter.per_tensor_quant(value, quant_dtype=_fp8_dtype)
            else:
                q_fp8, k_fp8, v_fp8 = query, key, value
                one = torch.tensor(1.0, dtype=torch.float32, device=query.device)
                q_scale = k_scale = v_scale = one

            d_q = q_fp8.shape[-1]
            d_k = k_fp8.shape[-1]
            d_v = v_fp8.shape[-1]
            h_q = q_fp8.shape[2]
            h_kv = k_fp8.shape[2]

            if _can_use_fmha_fp8_prefill(d_q, d_k, d_v, h_q, h_kv):
                return _fmha_fp8_prefill_attention(
                    q_fp8,
                    k_fp8,
                    v_fp8,
                    softmax_scale=self.softmax_scale,
                    is_causal=self.causal,
                    q_scale=q_scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )

            logger.warning_once(
                "FP8 FMHA prefill unsupported for this shape (need gfx950-class AITER, "
                "full MHA, q/k/v head_dim=%d; got q=%d, k=%d, v=%d, num_heads=%d, "
                "num_kv_heads=%d). Falling back to BF16.",
                _FMHA_FP8_HEAD_DIM,
                d_q,
                d_k,
                d_v,
                h_q,
                h_kv,
            )

        # BF16 path
        output, _ = aiter.flash_attn_func(
            query,
            key,
            value,
            dropout_p=self.dropout_p,
            causal=self.causal,
            return_attn_probs=False,
            return_lse=True,
        )
        return output

    @torch.compiler.disable
    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        del cu_seqlens_host
        if USE_AITER_GFX942:
            # The grouped-varlen ASM kernel hangs on H3's ~64K packed
            # sequences on gfx942; AITER's Triton path handles this shape.
            attention_func = importlib.import_module(
                "aiter.ops.triton.attention.mha"
            ).flash_attn_varlen_func
        else:
            attention_func = aiter.flash_attn_varlen_func

        cu_seqlens = cu_seqlens.to(device=query.device, dtype=torch.int32).contiguous()
        output = attention_func(
            q=query.contiguous(),
            k=key.contiguous(),
            v=value.contiguous(),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
        return output[0] if isinstance(output, tuple) else output
