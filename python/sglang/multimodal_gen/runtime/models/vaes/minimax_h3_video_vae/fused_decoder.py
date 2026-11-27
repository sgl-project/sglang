# SPDX-License-Identifier: Apache-2.0
"""Fused dataflow for the MiniMax-H3 FP8 decoder.

Each block folds the pending residual into RMSNorm and per-token FP8
quantization, then fuses the gated FFN activation with quantization. The final
residual feeds LayerNorm. Admission is all-or-nothing: unsupported inputs use
eager FP8 blocks; kernel failures propagate.
"""

from typing import NamedTuple

import torch
from torch import nn

from sglang.kernels.ops.diffusion import (
    can_use_fused_inplace_qknorm_rope,
    can_use_minimax_h3_vae_rmsnorm_fp8,
    fused_inplace_qknorm_rope,
    minimax_h3_vae_residual_layernorm,
    minimax_h3_vae_residual_rmsnorm_fp8,
    minimax_h3_vae_rmsnorm_fp8,
    silu_mul_quant_fp8,
)
from sglang.multimodal_gen.runtime.layers.quantization.online_fp8_linear import (
    OnlineFP8Linear,
)

from .vit_utils import native_rope_cache


class ResidualPending(NamedTuple):
    """``residual + projected * scale`` left for the next norm kernel to apply."""

    residual: torch.Tensor
    projected: torch.Tensor
    scale: torch.Tensor


def _vector(value, size, device) -> bool:
    return (
        value is not None
        and value.shape == (size,)
        and value.dtype == torch.float32
        and value.device == device
        and value.is_contiguous()
    )


def _weightless_rmsnorm(module) -> bool:
    return (
        isinstance(module, nn.RMSNorm)
        and module.weight is None
        and module.eps is not None
    )


def _rope_ok(rope, *, batch, seq, dim_head, rope_dim, device) -> bool:
    """The shared fp16 rotary table that ``native_rope_cache`` accepts for the
    packed ``[batch, seq, heads, 3 * dim_head]`` QKV projection of every block."""
    if len(rope) != 4:
        return False
    cache, positions = rope[2], rope[3]
    return (
        cache.device == device
        and cache.dtype == torch.float16
        and cache.shape == (seq, rope_dim)
        and positions.device == device
        and positions.shape in ((seq,), (batch * seq,))
        and can_use_fused_inplace_qknorm_rope(
            dim_head, rope_dim, True, torch.float16, torch.float16, True
        )
    )


def _block_ok(block, *, width, dim_head, device) -> bool:
    if not (
        block.use_scale
        and _vector(block.scale1, width, device)
        and _vector(block.scale2, width, device)
    ):
        return False
    for norm in (block.norm1, block.norm2):
        if not (
            isinstance(norm, nn.RMSNorm)
            and _vector(norm.weight, width, device)
            and norm.eps > 0
        ):
            return False
    attn = block.attn
    if not (
        attn.dim_head == dim_head
        and attn.attn is not None
        and _weightless_rmsnorm(attn.norm_q)
        and _weightless_rmsnorm(attn.norm_k)
        and attn.norm_q.eps == attn.norm_k.eps
    ):
        return False
    if not (block.ff.use_gated and isinstance(block.ff.act_fn, nn.SiLU)):
        return False
    return all(
        isinstance(module, OnlineFP8Linear)
        for module in (attn.to_qkv, attn.to_out, block.ff.w1, block.ff.w2)
    )


def structure_reject_reason(decoder) -> str | None:
    """Why the installed decoder cannot run the fused producers, else None;
    the installer asks once and per-call admission only looks at the tensors."""
    if not isinstance(decoder.norm_out, nn.LayerNorm):
        return "output norm is not a LayerNorm"
    width = decoder.norm_out.normalized_shape[0]
    device = decoder.norm_out.weight.device
    if not (
        _vector(decoder.norm_out.weight, width, device)
        and _vector(decoder.norm_out.bias, width, device)
    ):
        return "output LayerNorm affine vectors are not fp32 contiguous"
    dim_head = decoder.config.dim_head
    for index, block in enumerate(decoder.transformer_blocks):
        if not _block_ok(block, width=width, dim_head=dim_head, device=device):
            return f"block {index} is not the released block layout"
    return None


def can_use_fused_decoder(decoder, value, rope) -> bool:
    if not (decoder.fp8_installed and decoder.fused_decoder_ready):
        return False
    if (
        torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    ):
        return False
    width = decoder.norm_out.normalized_shape[0]
    first_norm = decoder.transformer_blocks[0].norm1
    return (
        value.is_cuda
        and value.device == first_norm.weight.device
        and value.dtype == torch.float32
        and value.dim() == 3
        and value.shape[-1] == width
        and not decoder.mask_enabled
        and _rope_ok(
            rope,
            batch=value.shape[0],
            seq=value.shape[1],
            dim_head=decoder.config.dim_head,
            rope_dim=decoder.pos_embed.dim,
            device=value.device,
        )
        # Later block inputs are fresh kernel and GEMM outputs; only the
        # decoder input can violate the row layout the producers require.
        and can_use_minimax_h3_vae_rmsnorm_fp8(
            value, first_norm.weight, eps=first_norm.eps
        )
    )


def _quantized_input(value, norm):
    """fp32 residual plus the FP8 rows of ``RMSNorm(residual) * weight``."""
    if isinstance(value, ResidualPending):
        return minimax_h3_vae_residual_rmsnorm_fp8(
            value.residual, value.projected, value.scale, norm.weight, eps=norm.eps
        )
    q, scales = minimax_h3_vae_rmsnorm_fp8(value, norm.weight, eps=norm.eps)
    return value, q, scales


def _attention(attn, q, scales, shape, rope, qk_weight):
    """FP8 QKV GEMM, in-place weightless Q/K RMSNorm+RoPE, the block's attention."""
    batch, seq = shape[0], shape[1]
    packed = attn.to_qkv.forward_quantized(q, scales, shape).view(
        batch, seq, attn.heads, 3 * attn.dim_head
    )
    query, key, value = torch.chunk(packed, 3, dim=-1)
    native = native_rope_cache(query, key, rope)
    if native is None:
        raise RuntimeError(
            "MiniMax-H3 fused decoder admitted a rotary table that its Q/K kernel "
            "then refused for the packed QKV"
        )
    cache, positions = native
    # The packed projection is this block's fresh GEMM output and nothing else
    # reads it, so the strided Q/K views are normalized and rotated in place;
    # unit weights make the kernel the model's weightless RMSNorm.
    fused_inplace_qknorm_rope(
        query.flatten(0, 1),
        key.flatten(0, 1),
        qk_weight,
        qk_weight,
        cache,
        positions,
        is_neox=True,
        eps=attn.norm_q.eps,
        head_dim=attn.dim_head,
        rope_dim=cache.shape[1],
        round_norm_before_rope=True,
    )
    attended = attn.attn(query, key, value).reshape(batch, seq, -1)
    return attn.to_out(attended)


def block_forward(module, value, rope, qk_weight):
    """One decoder block; ``qk_weight`` is the fp16 unit weight of its Q/K norm."""
    residual, q, scales = _quantized_input(value, module.norm1)
    shape = tuple(residual.shape)
    projected = _attention(module.attn, q, scales, shape, rope, qk_weight)
    residual, q, scales = minimax_h3_vae_residual_rmsnorm_fp8(
        residual, projected, module.scale1, module.norm2.weight, eps=module.norm2.eps
    )
    expanded = module.ff.w1.forward_quantized(q, scales, shape)
    q, scales = silu_mul_quant_fp8(expanded)
    projected = module.ff.w2.forward_quantized(
        q, scales, (*shape[:-1], expanded.shape[-1] // 2)
    )
    # The last add stays pending: the next block folds it into its norm/quant
    # kernel and the final block hands it to the output LayerNorm.
    return ResidualPending(residual, projected, module.scale2)


def finish(value, norm):
    return minimax_h3_vae_residual_layernorm(
        value.residual,
        value.projected,
        value.scale,
        norm.weight,
        norm.bias,
        eps=norm.eps,
    )
