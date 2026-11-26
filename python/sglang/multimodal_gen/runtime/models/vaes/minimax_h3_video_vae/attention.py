# SPDX-License-Identifier: Apache-2.0
# Attention module for the MiniMax H3 visual VAE (inference-only bundle).
import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import (
    BitExactFusionGate,
    aten_rms_norm_neox_rope_qk_rocm,
    aten_rms_norm_rocm,
    can_use_aten_rms_norm_neox_rope_qk_rocm,
    can_use_aten_rms_norm_rocm,
)
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from .vit_utils import apply_rotary_pos_emb_qk, native_rope_cache

logger = init_logger(__name__)
_FORCE_ROCM_MATH_SDPA = current_platform.is_rocm() and "gfx95" in str(
    torch.cuda.get_device_properties(0).gcnArchName
)
_rocm_flash_attn = None
if _FORCE_ROCM_MATH_SDPA:
    try:
        from aiter import flash_attn_func as _rocm_flash_attn
    except ImportError:
        pass


def _math_sdpa(query, key, value):
    """``F.scaled_dot_product_attention`` on ``SDPBackend.MATH``, bit-for-bit.

    Replays aten's ``_scaled_dot_product_attention_math`` for unmasked,
    non-causal inputs without the work this caller discards: ``_safe_softmax``'s
    all ``-inf`` row masking (no row is masked) and the downcast of the
    attention weights it also returns.
    """
    device = query.device.type
    # autocast casts the SDPA inputs once; aten's math ops then run uncast
    if torch.is_autocast_enabled(device):
        dtype = torch.get_autocast_dtype(device)
        query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)
    out_dtype = query.dtype
    if not torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed():
        query, key, value = query.float(), key.float(), value.float()
    # aten scales q and k by sqrt(1 / sqrt(head_dim)) each
    scale = math.sqrt(1.0 / math.sqrt(query.shape[-1]))
    with torch.autocast(device, enabled=False):
        attn = torch.matmul(query * scale, key.transpose(-2, -1) * scale)
        out = torch.matmul(torch.softmax(attn, -1), value)
    return out.to(out_dtype)


def _sdpa_attention(query, key, value):
    """query/key/value: [B, S, H, D]."""
    if _FORCE_ROCM_MATH_SDPA:
        # torch's fused SDPA corrupts this decoder on gfx950; half inputs on the
        # GPU take aiter's flash kernel (the fused class CUDA's SDPA picks, FP32
        # accumulation), FP32 and CPU inputs the exact math replica.
        device = query.device.type
        if torch.is_autocast_enabled(device):
            dtype = torch.get_autocast_dtype(device)
            query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)
        if (
            _rocm_flash_attn is not None
            and query.is_cuda
            and query.dtype in (torch.float16, torch.bfloat16)
        ):
            return _rocm_flash_attn(query, key, value, dropout_p=0.0, causal=False)
        query, key, value = (t.transpose(1, 2) for t in (query, key, value))
        return _math_sdpa(query, key, value).transpose(1, 2)
    query, key, value = (t.transpose(1, 2) for t in (query, key, value))
    out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
    return out.transpose(1, 2)


# ROCm runs these affine-free per-head RMSNorms one 256-thread block per
# 64-wide row after a contiguous copy; the Triton replica gives the same bytes
_ROCM_QK_RMS_NORM = BitExactFusionGate(
    "H3 VAE qk RMSNorm (ROCm aten replica)", per_signature=True
)


def _vit_norm_input(module, hidden_states):
    if envs.MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM:
        return hidden_states.float()
    weight = module.weight
    return hidden_states.to(weight.dtype if weight is not None else hidden_states.dtype)


def _apply_qk_norm(module, hidden_states):
    if (
        envs.MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM
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
        def eager():
            with torch.autocast("cuda", enabled=False):
                return module(hidden_states)

        if isinstance(module, nn.RMSNorm) and can_use_aten_rms_norm_rocm(hidden_states):
            sig = (hidden_states.shape[-1], hidden_states.dtype)
            if _ROCM_QK_RMS_NORM.can_attempt(sig):
                eps = module.eps
                if eps is None:
                    eps = torch.finfo(hidden_states.dtype).eps
                return _ROCM_QK_RMS_NORM.run(
                    lambda: aten_rms_norm_rocm(hidden_states, eps),
                    eager,
                    sig=sig,
                    logger=logger,
                )
        return eager()
    return module(_vit_norm_input(module, hidden_states)).to(hidden_states.dtype)


# The two qk RMSNorms and the NeoX rope as one launch, same bytes as the three
_ROCM_QK_NORM_ROPE = BitExactFusionGate(
    "H3 VAE qk RMSNorm + rope (ROCm)", per_signature=True
)


def _rocm_qk_norm_rope(attn, query, key, rotary_pos_emb):
    """``(query, key)`` normalized and rotated in one kernel, or None."""
    norm_q, norm_k = attn.norm_q, attn.norm_k
    if not (
        envs.MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM
        and rotary_pos_emb is not None
        and isinstance(norm_q, nn.RMSNorm)
        and isinstance(norm_k, nn.RMSNorm)
        and norm_q.weight is None
        and norm_k.weight is None
        and norm_q.eps == norm_k.eps
        and not torch.is_grad_enabled()
        and can_use_aten_rms_norm_rocm(query)
    ):
        return None
    native = native_rope_cache(query, key, rotary_pos_emb)
    if native is None:
        return None
    cache, positions = native
    if not can_use_aten_rms_norm_neox_rope_qk_rocm(query, key, cache, positions):
        return None
    sig = (query.shape[-1], cache.shape[-1], query.dtype)
    if not _ROCM_QK_NORM_ROPE.can_attempt(sig):
        return None
    eps = norm_q.eps if norm_q.eps is not None else torch.finfo(query.dtype).eps

    def reference():
        return apply_rotary_pos_emb_qk(
            _apply_qk_norm(norm_q, query), _apply_qk_norm(norm_k, key), rotary_pos_emb
        )

    return _ROCM_QK_NORM_ROPE.run(
        lambda: aten_rms_norm_neox_rope_qk_rocm(query, key, cache, positions, eps),
        reference,
        sig=sig,
        equal=lambda a, b: all(map(torch.equal, a, b)),
        logger=logger,
    )


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
                    AttentionBackendEnum.TORCH_SDPA,
                },
                default_attention_backend=AttentionBackendEnum.TORCH_SDPA,
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

        fused = _rocm_qk_norm_rope(self, query, key, rotary_pos_emb)
        if fused is not None:
            query, key = fused
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
