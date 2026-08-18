# SPDX-License-Identifier: Apache-2.0
# Attention module for the MiniMax H3 visual VAE (inference-only bundle).
from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import logging
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.platforms import current_platform

from .vit_utils import _env_flag, apply_rotary_pos_emb_qk

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
_FORCE_ROCM_MATH_SDPA = current_platform.is_rocm() and "gfx95" in str(
    torch.cuda.get_device_properties(0).gcnArchName
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
                skip_sequence_parallel=True,
            )
            if current_platform.is_cuda()
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
