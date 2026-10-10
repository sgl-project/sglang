# SPDX-License-Identifier: Apache-2.0
# Attention module for the MiniMax H3 visual VAE (inference-only bundle).
import importlib.util
from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import logging
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)

from .vit_utils import _env_flag, apply_rotary_pos_emb_qk

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
_FORCE_ROCM_MATH_SDPA = current_platform.is_rocm() and "gfx95" in str(
    torch.cuda.get_device_properties(0).gcnArchName
)
# FA dispatches through FA3, which is CUDA-only, so AITer is the fused kernel
# available to the decoder's ViT blocks on ROCm.
_ROCM_AITER_AVAILABLE = (
    current_platform.is_rocm() and importlib.util.find_spec("aiter") is not None
)
_DEFAULT_ATTENTION_BACKEND = (
    AttentionBackendEnum.AITER
    if _ROCM_AITER_AVAILABLE
    else AttentionBackendEnum.TORCH_SDPA
)


def _sdpa_attention(query, key, value):
    context = sdpa_kernel([SDPBackend.MATH]) if _FORCE_ROCM_MATH_SDPA else nullcontext()
    with context:
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
        ).transpose(1, 2)


def _vit_norm_input(module, hidden_states):
    if _env_flag("MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM", "1"):
        return hidden_states.float()
    weight = module.weight
    return hidden_states.to(weight.dtype if weight is not None else hidden_states.dtype)


def _try_fused_qk_rmsnorm_rope(norm_q, norm_k, query, key, rotary_pos_emb):
    if not _env_flag("MINIMAX_H3_VAE_DECODER_FUSED_NORM", "1"):
        return None
    if (
        rotary_pos_emb is None
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or not isinstance(norm_q, nn.RMSNorm)
        or not isinstance(norm_k, nn.RMSNorm)
        or norm_q.weight is not None
        or norm_k.weight is not None
        or norm_q.eps != norm_k.eps
    ):
        return None
    cos, sin = rotary_pos_emb[:2]
    if cos.dim() != 4 or sin.shape != cos.shape or cos.shape[2] != 1:
        return None
    # Stacked tiles broadcast one rotary table. A real batch stride would mean
    # each tile has its own positions, which this kernel does not index.
    if cos.shape[0] != 1 and cos.stride(0) != 0:
        return None
    if sin.shape[0] != 1 and sin.stride(0) != 0:
        return None
    from sglang.kernels.ops.diffusion import h3_vae_qk_rmsnorm_rope

    return h3_vae_qk_rmsnorm_rope(
        query,
        key,
        norm_q.eps,
        cos[0, :, 0, :],
        sin[0, :, 0, :],
    )


def _apply_qk_norm(module, hidden_states):
    if (
        _env_flag("MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM", "1")
        and isinstance(module, (nn.LayerNorm, nn.RMSNorm))
        and module.weight is None
        and (not isinstance(module, nn.LayerNorm) or module.bias is None)
        and hidden_states.is_cuda
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
    ):
        # CUDA LayerNorm/RMSNorm accumulates half/bfloat16 inputs in FP32.
        # With no affine parameters its half output is bit-identical to the
        # released FP32-norm-then-cast recipe, without two full-tensor casts.
        with torch.autocast("cuda", enabled=False):
            return module(hidden_states)
    return module(_vit_norm_input(module, hidden_states)).to(hidden_states.dtype)


class Attention(nn.Module):
    def __init__(
        self,
        heads,
        dim_head,
        embed_dim: Optional[int] = None,
        qk_norm_type: Optional[str] = None,
        qk_norm_affine: bool = False,
        bias: bool = True,
        out_bias: Optional[bool] = None,
        eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.attn_inner_dim = dim_head * heads
        self.embed_dim = embed_dim if embed_dim is not None else self.attn_inner_dim

        out_bias = out_bias if out_bias is not None else bias

        if qk_norm_type is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm_type == "layer_norm":
            self.norm_q = nn.LayerNorm(
                dim_head, eps=eps, elementwise_affine=qk_norm_affine
            )
            self.norm_k = nn.LayerNorm(
                dim_head, eps=eps, elementwise_affine=qk_norm_affine
            )
        elif qk_norm_type == "rms_norm":
            self.norm_q = nn.RMSNorm(
                dim_head, eps=eps, elementwise_affine=qk_norm_affine
            )
            self.norm_k = nn.RMSNorm(
                dim_head, eps=eps, elementwise_affine=qk_norm_affine
            )
        else:
            raise ValueError(
                f"unknown qk_norm_type: {qk_norm_type}. Should be None,'layer_norm','rms_norm'"
            )

        self.to_qkv = nn.Linear(self.embed_dim, self.attn_inner_dim * 3, bias=bias)
        self.to_out = nn.Linear(self.attn_inner_dim, self.embed_dim, bias=out_bias)
        # Decode ranks process independent complete tiles. Reuse USPAttention's
        # backend dispatch, while deliberately bypassing its sequence collectives.
        self.attn = (
            USPAttention(
                num_heads=heads,
                head_size=dim_head,
                causal=False,
                supported_attention_backends={
                    AttentionBackendEnum.FA,
                    AttentionBackendEnum.AITER,
                    AttentionBackendEnum.TORCH_SDPA,
                },
                default_attention_backend=_DEFAULT_ATTENTION_BACKEND,
                skip_sequence_parallel=True,
            )
            if current_platform.is_cuda() or _ROCM_AITER_AVAILABLE
            else None
        )

        if len(kwargs) > 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            logger.warning(f"Unused kwargs: {kwargs}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.to_qkv(hidden_states)
        qkv = qkv.view(batch_size, seq_len, -1, 3 * self.dim_head)
        query, key, value = torch.chunk(qkv, 3, dim=-1)

        fused_qk = None
        if self.norm_q is not None and self.norm_k is not None:
            fused_qk = _try_fused_qk_rmsnorm_rope(
                self.norm_q, self.norm_k, query, key, rotary_pos_emb
            )
        if fused_qk is not None:
            query, key = fused_qk
        else:
            if self.norm_q is not None:
                query = _apply_qk_norm(self.norm_q, query)
            if self.norm_k is not None:
                key = _apply_qk_norm(self.norm_k, key)
            if rotary_pos_emb is not None:
                query, key = apply_rotary_pos_emb_qk(query, key, rotary_pos_emb)

        if self.attn is not None and query.dtype in (torch.float16, torch.bfloat16):
            hidden_states = self.attn(query, key, value)
        else:
            # FlashAttention kernels do not accept FP32. Preserve the explicit
            # no-autocast and MPS paths instead of making backend selection
            # change H3's supported precision contract.
            hidden_states = _sdpa_attention(query, key, value)

        hidden_states = hidden_states.reshape(batch_size, seq_len, -1)
        hidden_states = self.to_out(hidden_states)

        return hidden_states
