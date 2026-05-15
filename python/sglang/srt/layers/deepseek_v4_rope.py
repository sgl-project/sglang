import math
from functools import lru_cache
from typing import Optional

import tilelang
import torch
import triton
import triton.language as tl

from sglang.srt.utils.common import maybe_torch_compile

tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


@lru_cache(2)
def precompute_freqs_cis(
    dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow
) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


@maybe_torch_compile
def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Adopted from DeepSeek's reference implementation, but adapted to sglang input formats:
        - 2D: x [bs, rope_dim], freqs_cis [bs, rope_dim // 2]
        - 3D: x [bs, n_heads, rope_dim], freqs_cis [bs, rope_dim // 2]

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        # x: [bs, n_heads, rope_dim // 2], freqs_cis: [bs, rope_dim // 2]
        # -> reshape freqs_cis to [bs, 1, rope_dim // 2] to broadcast over n_heads
        freqs_cis = freqs_cis.unsqueeze(1)
    # For 2D case should directly match: x [bs, rope_dim // 2], freqs_cis [bs, rope_dim // 2]
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


@triton.jit
def apply_rotary_emb_triton_kernel(
    x_ptr,
    freqs_ptr,
    positions_ptr,
    rope_dim,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_freq_pos,
    stride_freq_dim,
    USE_POS: tl.constexpr,
    IS_INVERSE: tl.constexpr,
    IS_3D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_dim = tl.program_id(2)

    # Get position: from tensor or directly use pid_batch
    if USE_POS:
        position = tl.load(positions_ptr + pid_batch)
    else:
        position = pid_batch

    if IS_3D:
        # [bs, n_heads, rope_dim]
        base_offset = pid_batch * stride_x_batch + pid_head * stride_x_head
    else:
        # [bs, rope_dim]
        base_offset = pid_batch * stride_x_batch

    offs_pair = pid_dim * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_pair < (rope_dim // 2)

    # real is even, imag is odd
    offs_x_real = base_offset + offs_pair * 2 * stride_x_dim
    offs_x_imag = base_offset + (offs_pair * 2 + 1) * stride_x_dim

    x_real = tl.load(x_ptr + offs_x_real, mask=mask, other=0.0).to(tl.float32)
    x_imag = tl.load(x_ptr + offs_x_imag, mask=mask, other=0.0).to(tl.float32)

    offs_freq_real = position * stride_freq_pos + offs_pair * 2 * stride_freq_dim
    offs_freq_imag = position * stride_freq_pos + (offs_pair * 2 + 1) * stride_freq_dim

    freq_real = tl.load(freqs_ptr + offs_freq_real, mask=mask, other=0.0)
    freq_imag = tl.load(freqs_ptr + offs_freq_imag, mask=mask, other=0.0)

    if IS_INVERSE:
        # (a + bi) * (c - di) = (ac + bd) + (bc - ad)i
        out_real = x_real * freq_real + x_imag * freq_imag
        out_imag = x_imag * freq_real - x_real * freq_imag
    else:
        # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        out_real = x_real * freq_real - x_imag * freq_imag
        out_imag = x_real * freq_imag + x_imag * freq_real

    tl.store(x_ptr + offs_x_real, out_real, mask=mask)
    tl.store(x_ptr + offs_x_imag, out_imag, mask=mask)


def apply_rotary_emb_triton(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
    inverse: bool = False,
) -> torch.Tensor:
    """
    Args:
        x: 2d [bs, rope_dim] or 3d [bs, n_heads, rope_dim]
        freqs_cis:
            - If positions is None: [bs, rope_dim // 2] (already indexed)
            - If positions is not None: [max_seqlen, rope_dim // 2] (full table)
        positions: Optional[bs], if provided will index into freqs_cis
        inverse: bool, if True, apply inverse rotation (conjugate)
    Returns:
        x with rotary embeddings applied (inplace)
    """
    is_3d = x.ndim == 3

    if is_3d:
        batch_size, n_heads, rope_dim = x.shape
    else:
        batch_size, rope_dim = x.shape
        n_heads = 1

    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)

    BLOCK_SIZE = 128

    num_blocks_dim = triton.cdiv(rope_dim // 2, BLOCK_SIZE)
    grid = (batch_size, n_heads if is_3d else 1, num_blocks_dim)

    if positions is not None:
        # use positions to index into freqs_cis
        assert positions.shape == (
            batch_size,
        ), f"positions shape {positions.shape} != ({batch_size},)"

        apply_rotary_emb_triton_kernel[grid](
            x,
            freqs_real,
            positions,
            rope_dim,
            x.stride(0),
            x.stride(1) if is_3d else 0,
            x.stride(-1),
            freqs_real.stride(0),
            freqs_real.stride(1),
            USE_POS=True,
            IS_INVERSE=inverse,
            IS_3D=is_3d,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # freqs_cis already indexed, use pid_batch as position
        assert (
            freqs_real.shape[0] == batch_size
        ), f"freqs_cis batch size {freqs_real.shape[0]} != x batch size {batch_size}"

        apply_rotary_emb_triton_kernel[grid](
            x,
            freqs_real,
            None,
            rope_dim,
            x.stride(0),
            x.stride(1) if is_3d else 0,
            x.stride(-1),
            freqs_real.stride(0),
            freqs_real.stride(1),
            USE_POS=False,
            IS_INVERSE=inverse,
            IS_3D=is_3d,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return x


@triton.jit
def _fused_norm_rope_kernel(
    x_ptr,
    weight_ptr,
    freqs_real_ptr,
    positions_ptr,
    eps,
    stride_x_row,
    stride_freq_row,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROPE_PAIR_BLOCK: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    USE_POS: tl.constexpr,
):
    # NOTE: avoids store-then-reload on the same kernel: rope-segment values
    # are loaded a 2nd time as (real, imag) pairs straight from the input,
    # rms_inv/weight applied in register, and all stores happen at the end.
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * stride_x_row

    offs = tl.arange(0, HEAD_BLOCK)
    mask = offs < HEAD_DIM
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    sum_sq = tl.sum(x * x, axis=0)
    rms_inv = tl.rsqrt(sum_sq / HEAD_DIM + eps)

    if HAS_WEIGHT:
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x_normed = x * rms_inv * w
    else:
        x_normed = x * rms_inv

    rope_start = HEAD_DIM - ROPE_DIM

    pair_offs = tl.arange(0, ROPE_PAIR_BLOCK)
    pair_mask = pair_offs < (ROPE_DIM // 2)

    x_real = tl.load(
        x_ptr + base + rope_start + 2 * pair_offs,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    x_imag = tl.load(
        x_ptr + base + rope_start + 2 * pair_offs + 1,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)

    if HAS_WEIGHT:
        w_real = tl.load(
            weight_ptr + rope_start + 2 * pair_offs,
            mask=pair_mask,
            other=1.0,
        ).to(tl.float32)
        w_imag = tl.load(
            weight_ptr + rope_start + 2 * pair_offs + 1,
            mask=pair_mask,
            other=1.0,
        ).to(tl.float32)
        x_real = x_real * rms_inv * w_real
        x_imag = x_imag * rms_inv * w_imag
    else:
        x_real = x_real * rms_inv
        x_imag = x_imag * rms_inv

    if USE_POS:
        position = tl.load(positions_ptr + pid).to(tl.int64)
    else:
        position = pid.to(tl.int64)

    freq_base = position * stride_freq_row
    f_real = tl.load(
        freqs_real_ptr + freq_base + 2 * pair_offs,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    f_imag = tl.load(
        freqs_real_ptr + freq_base + 2 * pair_offs + 1,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)

    out_real = x_real * f_real - x_imag * f_imag
    out_imag = x_real * f_imag + x_imag * f_real

    is_non_rope = offs < rope_start
    tl.store(
        x_ptr + base + offs,
        x_normed.to(x_ptr.dtype.element_ty),
        mask=mask & is_non_rope,
    )
    tl.store(
        x_ptr + base + rope_start + 2 * pair_offs,
        out_real.to(x_ptr.dtype.element_ty),
        mask=pair_mask,
    )
    tl.store(
        x_ptr + base + rope_start + 2 * pair_offs + 1,
        out_imag.to(x_ptr.dtype.element_ty),
        mask=pair_mask,
    )


@triton.jit
def _fused_softmax_pool_kernel(
    kv_score_ptr,
    out_ptr,
    stride_bs: tl.constexpr,
    stride_k: tl.constexpr,
    K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * stride_bs

    offs = tl.arange(0, HEAD_BLOCK)
    mask = offs < HEAD_DIM

    max_val = tl.full([HEAD_BLOCK], float("-inf"), dtype=tl.float32)
    for k in range(K):
        s = tl.load(
            kv_score_ptr + base + k * stride_k + HEAD_DIM + offs,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        max_val = tl.maximum(max_val, s)

    sum_exp = tl.zeros([HEAD_BLOCK], dtype=tl.float32)
    weighted = tl.zeros([HEAD_BLOCK], dtype=tl.float32)
    for k in range(K):
        s = tl.load(
            kv_score_ptr + base + k * stride_k + HEAD_DIM + offs,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        v = tl.load(
            kv_score_ptr + base + k * stride_k + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        w = tl.exp(s - max_val)
        sum_exp += w
        weighted += v * w

    result = weighted / sum_exp
    tl.store(
        out_ptr + pid * HEAD_DIM + offs, result.to(out_ptr.dtype.element_ty), mask=mask
    )


def fused_softmax_pool_triton(
    kv_score: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Fused softmax-weighted-sum: out = (kv * softmax(score, dim=1)).sum(dim=1).

    Replaces the generic cunn_SpatialSoftMaxForward + elementwise multiply + sum
    with a single Triton kernel.

    Args:
        kv_score: [bs, K, 2 * head_dim] where first head_dim is kv, second is score.
        head_dim: dimension of each of kv and score.
    Returns:
        output: [bs, head_dim]
    """
    assert kv_score.dim() == 3
    bs, K, last = kv_score.shape
    assert last == 2 * head_dim
    assert kv_score.is_contiguous()

    out = torch.empty(bs, head_dim, dtype=kv_score.dtype, device=kv_score.device)
    if bs == 0:
        return out

    HEAD_BLOCK = triton.next_power_of_2(head_dim)
    grid = (bs,)
    _fused_softmax_pool_kernel[grid](
        kv_score,
        out,
        stride_bs=kv_score.stride(0),
        stride_k=kv_score.stride(1),
        K=K,
        HEAD_DIM=head_dim,
        HEAD_BLOCK=HEAD_BLOCK,
    )
    return out


@triton.jit
def _fused_softmax_pool_split_kernel(
    kv_ptr,
    score_ptr,
    out_ptr,
    stride_kv_bs,
    stride_kv_k,
    stride_sc_bs,
    stride_sc_k,
    K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    kv_base = pid * stride_kv_bs
    sc_base = pid * stride_sc_bs

    offs = tl.arange(0, HEAD_BLOCK)
    mask = offs < HEAD_DIM

    max_val = tl.full([HEAD_BLOCK], float("-inf"), dtype=tl.float32)
    for k in range(K):
        s = tl.load(
            score_ptr + sc_base + k * stride_sc_k + offs,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        max_val = tl.maximum(max_val, s)

    sum_exp = tl.zeros([HEAD_BLOCK], dtype=tl.float32)
    weighted = tl.zeros([HEAD_BLOCK], dtype=tl.float32)
    for k in range(K):
        s = tl.load(
            score_ptr + sc_base + k * stride_sc_k + offs,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        v = tl.load(
            kv_ptr + kv_base + k * stride_kv_k + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        w = tl.exp(s - max_val)
        sum_exp += w
        weighted += v * w

    result = weighted / sum_exp
    tl.store(
        out_ptr + pid * HEAD_DIM + offs, result.to(out_ptr.dtype.element_ty), mask=mask
    )


def fused_softmax_pool_split_triton(
    kv: torch.Tensor,
    score: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Fused softmax-weighted-sum with separate kv and score tensors.

    Args:
        kv: [bs, K, head_dim]
        score: [bs, K, head_dim]
        head_dim: last dimension size.
    Returns:
        output: [bs, head_dim]
    """
    assert kv.dim() == 3 and score.dim() == 3
    bs, K, d = kv.shape
    assert d == head_dim
    assert score.shape == kv.shape

    out = torch.empty(bs, head_dim, dtype=kv.dtype, device=kv.device)
    if bs == 0:
        return out

    HEAD_BLOCK = triton.next_power_of_2(head_dim)
    grid = (bs,)
    _fused_softmax_pool_split_kernel[grid](
        kv,
        score,
        out,
        stride_kv_bs=kv.stride(0),
        stride_kv_k=kv.stride(1),
        stride_sc_bs=score.stride(0),
        stride_sc_k=score.stride(1),
        K=K,
        HEAD_DIM=head_dim,
        HEAD_BLOCK=HEAD_BLOCK,
    )
    return out


def fused_norm_rope_inplace_triton(
    kv: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    freqs_cis: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
) -> None:
    """Fused RMSNorm (over head_dim) + RoPE (on last rope_dim of head_dim), in-place.

    Equivalent to::

        kv = rms_normalize(kv, eps, weight)
        apply_rotary_emb_triton(kv[..., -rope_dim:], freqs_cis, positions=positions)

    Args:
        kv: [M, head_dim], any float dtype, contiguous along last dim. Modified in-place.
        weight: [head_dim] or None.
        eps: RMSNorm epsilon.
        freqs_cis: complex tensor.
            - If ``positions`` is None: shape [M, rope_dim // 2], one freq per token.
            - Else: shape [max_seq, rope_dim // 2], full table; indexed by ``positions``.
        positions: optional [M] int tensor, absolute positions to index into ``freqs_cis``.
    """
    assert kv.dim() == 2 and kv.stride(-1) == 1
    M, head_dim = kv.shape

    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    rope_dim = freqs_real.shape[-1]
    assert head_dim >= rope_dim and rope_dim % 2 == 0
    if weight is not None:
        assert weight.shape == (head_dim,)
    if positions is None:
        assert (
            freqs_real.shape[0] == M
        ), f"freqs_cis row count {freqs_real.shape[0]} != M={M}"
    else:
        assert positions.shape == (M,) and positions.dim() == 1

    if M == 0:
        return

    HEAD_BLOCK = triton.next_power_of_2(head_dim)
    ROPE_PAIR_BLOCK = max(triton.next_power_of_2(rope_dim // 2), 1)

    grid = (M,)
    _fused_norm_rope_kernel[grid](
        kv,
        weight,
        freqs_real,
        positions,
        eps,
        kv.stride(0),
        freqs_real.stride(0),
        HEAD_DIM=head_dim,
        ROPE_DIM=rope_dim,
        HEAD_BLOCK=HEAD_BLOCK,
        ROPE_PAIR_BLOCK=ROPE_PAIR_BLOCK,
        HAS_WEIGHT=(weight is not None),
        USE_POS=(positions is not None),
    )
