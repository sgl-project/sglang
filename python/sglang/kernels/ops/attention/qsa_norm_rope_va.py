# SPDX-License-Identifier: Apache-2.0
"""QSA normalization and MRoPE from the vLLM NPU plugin.

Source: the vLLM NPU plugin 24487f4f48ee99cf45d86d1c504e645549b8c1ef,
ops/triton/linearnorm/qsa_rmsnorm_mrope.py.
Only imports/custom-op registration are adapted for the SGLang caller.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _qsa_rmsnorm_mrope_kernel(
    input_ptr,
    weight_ptr,
    cos_sin_ptr,
    output_ptr,
    input_stride_token,
    input_stride_head,
    cos_sin_stride_axis,
    cos_sin_stride_token,
    output_stride_token,
    output_stride_head,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    eps: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    rope_dim: tl.constexpr,
    has_mrope: tl.constexpr,
    block_dim: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)
    offsets = tl.arange(0, block_dim)
    mask = offsets < head_dim
    input_row = input_ptr + token_id * input_stride_token + head_id * input_stride_head
    values = tl.load(input_row + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32) + 1.0
    variance = tl.sum(values * values, axis=0) / head_dim
    reciprocal_std = tl.rsqrt(variance + eps)
    normalized = (values * reciprocal_std * weight).to(tl.bfloat16)

    half_rope_dim: tl.constexpr = rope_dim // 2
    rotary_offsets = tl.arange(0, half_rope_dim)
    if has_mrope:
        h_mask = (rotary_offsets % 3 == 1) & (rotary_offsets < mrope_section_h * 3)
        w_mask = (rotary_offsets % 3 == 2) & (rotary_offsets < mrope_section_w * 3)
        cache_row = cos_sin_ptr + token_id * cos_sin_stride_token
        cos_t = tl.load(cache_row + rotary_offsets).to(tl.float32)
        cos_h = tl.load(cache_row + cos_sin_stride_axis + rotary_offsets).to(tl.float32)
        cos_w = tl.load(cache_row + 2 * cos_sin_stride_axis + rotary_offsets).to(tl.float32)
        sin_t = tl.load(cache_row + rotary_offsets + half_rope_dim).to(tl.float32)
        sin_h = tl.load(cache_row + cos_sin_stride_axis + rotary_offsets + half_rope_dim).to(tl.float32)
        sin_w = tl.load(cache_row + 2 * cos_sin_stride_axis + rotary_offsets + half_rope_dim).to(tl.float32)
        cos = tl.where(h_mask, cos_h, tl.where(w_mask, cos_w, cos_t))
        sin = tl.where(h_mask, sin_h, tl.where(w_mask, sin_w, sin_t))
    else:
        cache_row = cos_sin_ptr + token_id * cos_sin_stride_token
        cos = tl.load(cache_row + rotary_offsets).to(tl.float32)
        sin = tl.load(cache_row + rotary_offsets + half_rope_dim).to(tl.float32)
    first = tl.load(input_row + rotary_offsets).to(tl.float32)
    first_weight = tl.load(weight_ptr + rotary_offsets).to(tl.float32) + 1.0
    first = (first * reciprocal_std * first_weight).to(tl.bfloat16)
    second_offsets = rotary_offsets + half_rope_dim
    second = tl.load(input_row + second_offsets).to(tl.float32)
    second_weight = tl.load(weight_ptr + second_offsets).to(tl.float32) + 1.0
    second = (second * reciprocal_std * second_weight).to(tl.bfloat16)
    first_cos = (first * cos).to(tl.bfloat16)
    second_sin = (second * sin).to(tl.bfloat16)
    second_cos = (second * cos).to(tl.bfloat16)
    first_sin = (first * sin).to(tl.bfloat16)
    rotated_first = (first_cos - second_sin).to(tl.bfloat16)
    rotated_second = (second_cos + first_sin).to(tl.bfloat16)

    output_row = output_ptr + token_id * output_stride_token + head_id * output_stride_head
    tl.store(output_row + offsets, normalized, mask=mask)
    tl.store(output_row + rotary_offsets, rotated_first)
    tl.store(output_row + rotary_offsets + half_rope_dim, rotated_second)


def triton_qsa_rmsnorm_mrope(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    cos_sin: torch.Tensor,
    eps: float,
    mrope_section: list[int],
    rope_dim: int,
) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError("QSA norm+RoPE input must be [tokens, heads, head_dim]")
    if tensor.dtype != torch.bfloat16 or weight.dtype != tensor.dtype:
        raise ValueError("QSA norm+RoPE requires BF16 input and weight")
    if tensor.stride(-1) != 1 or weight.ndim != 1 or weight.shape[0] != tensor.shape[-1]:
        raise ValueError("QSA norm+RoPE requires contiguous head rows and matching weight")
    if cos_sin.ndim not in (2, 3) or cos_sin.shape[-1] != rope_dim:
        raise ValueError("QSA norm+RoPE received an invalid rotary cache view")
    if rope_dim <= 0 or rope_dim > tensor.shape[-1] or rope_dim % 2:
        raise ValueError("QSA norm+RoPE requires an even rotary width")
    if cos_sin.shape[-2] != tensor.shape[0]:
        raise ValueError("QSA norm+RoPE token and rotary rows must match")

    has_mrope = cos_sin.ndim == 3
    if has_mrope:
        if cos_sin.shape[0] != 3 or len(mrope_section) != 3:
            raise ValueError("QSA MRoPE requires three position axes and sections")
        cos_sin_stride_axis = cos_sin.stride(0)
    else:
        cos_sin_stride_axis = 0

    output = torch.empty_like(tensor)
    num_tokens, num_heads, head_dim = tensor.shape
    if not num_tokens:
        return output
    _qsa_rmsnorm_mrope_kernel[(num_tokens, num_heads)](
        tensor,
        weight,
        cos_sin,
        output,
        tensor.stride(0),
        tensor.stride(1),
        cos_sin_stride_axis,
        cos_sin.stride(-2),
        output.stride(0),
        output.stride(1),
        num_heads=num_heads,
        head_dim=head_dim,
        eps=eps,
        mrope_section_h=mrope_section[1] if has_mrope else 0,
        mrope_section_w=mrope_section[2] if has_mrope else 0,
        rope_dim=rope_dim,
        has_mrope=has_mrope,
        block_dim=triton.next_power_of_2(head_dim),
        num_warps=1,
        num_stages=1,
    )
    return output


def supports_qsa_norm_rope_va(tensor, positions, weight, rotary):
    """The measured Qwen Q/K ABI; other rotary modes keep their existing path."""
    cache = getattr(rotary, "cos_sin_cache", None)
    return bool(
        tensor.device.type == "npu"
        and tensor.ndim == 3
        and tensor.shape[-1] == 128
        and tensor.shape[1] in (1, 4)
        and tensor.stride(-1) == 1
        and tensor.dtype == weight.dtype == torch.bfloat16
        and weight.shape == (128,)
        and weight.device == tensor.device
        and rotary.rotary_dim == 64
        and rotary.is_neox_style
        and not getattr(rotary, "mrope_interleaved_glm", False)
        and cache is not None
        and cache.device == tensor.device
        and cache.dtype in (torch.bfloat16, torch.float32)
        and positions.device == tensor.device
        and positions.dtype in (torch.int32, torch.int64)
        and (
            (positions.ndim == 1 and positions.shape[0] == tensor.shape[0])
            or (
                positions.ndim == 2
                and positions.shape == (3, tensor.shape[0])
                and getattr(rotary, "mrope_interleaved", False)
                and list(getattr(rotary, "mrope_section", []) or []) == [11, 11, 10]
            )
        )
    )


def qsa_norm_rope_va(tensor, positions, weight, eps, rotary):
    """Use VA's local RoPE view without overwriting main-attention positions.

    QSA compressed keys have different positions from the main token queries.
    The rotary module is shared across layers; later main QKV prepare calls
    reuse its position_cos/sin fields. The indexer must not replace them.
    """
    if tensor.numel() == 0:
        return tensor
    positions = positions.long()
    if positions.ndim == 1:
        cos_sin = rotary.cos_sin_cache.index_select(0, positions)
    else:
        cos_sin = rotary.cos_sin_cache[positions]
    # Keep native apply_rotary_emb's cache-to-input dtype boundary.
    return triton_qsa_rmsnorm_mrope(
        tensor, weight, cos_sin.to(tensor.dtype), eps,
        list(getattr(rotary, "mrope_section", []) or []), rotary.rotary_dim,
    )
