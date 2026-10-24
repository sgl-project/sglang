import logging
import math
from functools import lru_cache
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# This module is imported during model-registry discovery. Keep it free of
# TileLang imports so discovery does not load TileLang's native CUDA stubs.

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


@lru_cache(2)
def precompute_freqs_cis(
    dim,
    seqlen,
    original_seq_len,
    base,
    factor,
    beta_fast,
    beta_slow,
) -> torch.Tensor:

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


@lru_cache(2)
def precompute_cos_sin_table(
    dim,
    seqlen,
    original_seq_len,
    base,
    factor,
    beta_fast,
    beta_slow,
) -> torch.Tensor:
    freqs_cis = precompute_freqs_cis(
        dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow
    )
    return torch.view_as_real(freqs_cis).transpose(1, 2).reshape(seqlen, -1)


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

    if USE_POS:
        position = tl.load(positions_ptr + pid_batch)
    else:
        position = pid_batch

    if IS_3D:
        base_offset = pid_batch * stride_x_batch + pid_head * stride_x_head
    else:
        base_offset = pid_batch * stride_x_batch

    offs_pair = pid_dim * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_pair < (rope_dim // 2)

    offs_x_real = base_offset + offs_pair * 2 * stride_x_dim
    offs_x_imag = base_offset + (offs_pair * 2 + 1) * stride_x_dim

    x_real = tl.load(x_ptr + offs_x_real, mask=mask, other=0.0).to(tl.float32)
    x_imag = tl.load(x_ptr + offs_x_imag, mask=mask, other=0.0).to(tl.float32)

    offs_freq_real = position * stride_freq_pos + offs_pair * 2 * stride_freq_dim
    offs_freq_imag = position * stride_freq_pos + (offs_pair * 2 + 1) * stride_freq_dim

    freq_real = tl.load(freqs_ptr + offs_freq_real, mask=mask, other=0.0)
    freq_imag = tl.load(freqs_ptr + offs_freq_imag, mask=mask, other=0.0)

    if IS_INVERSE:
        out_real = x_real * freq_real + x_imag * freq_imag
        out_imag = x_imag * freq_real - x_real * freq_imag
    else:
        out_real = x_real * freq_real - x_imag * freq_imag
        out_imag = x_real * freq_imag + x_imag * freq_real

    tl.store(x_ptr + offs_x_real, out_real, mask=mask)
    tl.store(x_ptr + offs_x_imag, out_imag, mask=mask)


@triton.jit
def apply_rotary_emb_triton_kernel_batched(
    x_ptr,
    freqs_ptr,
    positions_ptr,
    rope_dim,
    n_tokens,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_freq_pos,
    stride_freq_dim,
    USE_POS: tl.constexpr,
    IS_INVERSE: tl.constexpr,
    IS_3D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # Batched variant: BLOCK_M tokens per program
    # which batches 32 tokens/program) to cut the per-token launch granularity of
    # the original (one program per token).
    pid_m = tl.program_id(0)
    pid_head = tl.program_id(1)

    tok = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tok_mask = tok < n_tokens
    pair = tl.arange(0, BLOCK_P)
    pair_mask = pair < (rope_dim // 2)
    m2 = tok_mask[:, None] & pair_mask[None, :]

    if USE_POS:
        position = tl.load(positions_ptr + tok, mask=tok_mask, other=0)
    else:
        position = tok

    if IS_3D:
        base = tok[:, None] * stride_x_batch + pid_head * stride_x_head
    else:
        base = tok[:, None] * stride_x_batch

    off_real = base + (pair[None, :] * 2) * stride_x_dim
    off_imag = base + (pair[None, :] * 2 + 1) * stride_x_dim

    x_real = tl.load(x_ptr + off_real, mask=m2, other=0.0).to(tl.float32)
    x_imag = tl.load(x_ptr + off_imag, mask=m2, other=0.0).to(tl.float32)

    off_f_real = (
        position[:, None] * stride_freq_pos + (pair[None, :] * 2) * stride_freq_dim
    )
    off_f_imag = (
        position[:, None] * stride_freq_pos + (pair[None, :] * 2 + 1) * stride_freq_dim
    )
    freq_real = tl.load(freqs_ptr + off_f_real, mask=m2, other=0.0)
    freq_imag = tl.load(freqs_ptr + off_f_imag, mask=m2, other=0.0)

    if IS_INVERSE:
        out_real = x_real * freq_real + x_imag * freq_imag
        out_imag = x_imag * freq_real - x_real * freq_imag
    else:
        out_real = x_real * freq_real - x_imag * freq_imag
        out_imag = x_real * freq_imag + x_imag * freq_real

    tl.store(x_ptr + off_real, out_real, mask=m2)
    tl.store(x_ptr + off_imag, out_imag, mask=m2)


@triton.jit
def apply_rotary_emb_tile_kernel(
    x_ptr,
    fr_ptr,
    pos_ptr,
    n_tokens,
    n_heads,
    sx_tok,
    sx_head,
    sx_d,
    sfr_pos,
    sfr_d,
    USE_POS: tl.constexpr,
    IS_INVERSE: tl.constexpr,
    RD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # same rounding as fused_q_norm_rope on HIP: round the sine product, then one fma
    tok = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    head = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    tmask = tok < n_tokens
    if USE_POS:
        pos = tl.load(pos_ptr + tok, mask=tmask, other=0)
    else:
        pos = tok
    d = tl.arange(0, RD)
    fr = tl.load(
        fr_ptr + pos.to(tl.int64)[:, None] * sfr_pos + d[None, :] * sfr_d,
        mask=tmask[:, None],
        other=0.0,
    )
    cos, sin = tl.split(tl.reshape(fr, (BLOCK_T, RD // 2, 2)))
    cos = cos[None, :, :]
    sin = sin[None, :, :]

    xo = (
        head[:, None, None] * sx_head
        + tok.to(tl.int64)[None, :, None] * sx_tok
        + d[None, None, :] * sx_d
    )
    mask = (head < n_heads)[:, None, None] & tmask[None, :, None]
    x = tl.load(x_ptr + xo, mask=mask, other=0.0).to(tl.float32)
    x_re, x_im = tl.split(tl.reshape(x, (BLOCK_H, BLOCK_T, RD // 2, 2)))
    if IS_INVERSE:
        out_re = tl.fma(x_re, cos, x_im * sin)
        out_im = tl.fma(x_im, cos, -(x_re * sin))
    else:
        out_re = tl.fma(x_re, cos, -(x_im * sin))
        out_im = tl.fma(x_im, cos, x_re * sin)
    out = tl.reshape(tl.join(out_re, out_im), (BLOCK_H, BLOCK_T, RD))
    tl.store(x_ptr + xo, out.to(x_ptr.dtype.element_ty), mask=mask)


def _rope_tile_config(n_tokens: int, n_heads: int):
    """(BLOCK_T, BLOCK_H, num_warps), tuned on MI355X."""
    block_h = min(triton.next_power_of_2(n_heads), 32)
    rows = 32 if n_tokens < 4096 else (64 if n_tokens < 16384 else 128)
    block_t = max(1, min(triton.next_power_of_2(n_tokens), rows // block_h))
    return block_t, block_h, max(1, block_t * block_h // 16)


# Use the batched / contiguous-load rope kernels (faster, coalesced) instead of the
# per-token kernel. Default OFF; DeepseekV4 enables it via set_batched_rope(True).
# The env var SGLANG_ROPE_BATCHED=1 still works as an override.
_USE_BATCHED_ROPE: bool = False


def set_batched_rope(enabled: bool = True) -> None:
    global _USE_BATCHED_ROPE
    _USE_BATCHED_ROPE = enabled


def apply_rotary_emb_triton(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
    inverse: bool = False,
) -> torch.Tensor:

    if _USE_BATCHED_ROPE:
        is_3d = x.ndim == 3
        if is_3d:
            batch_size, n_heads, rope_dim = x.shape
        else:
            batch_size, rope_dim = x.shape
            n_heads = 1
        freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
        if positions is not None:
            assert positions.shape == (batch_size,)
        else:
            assert freqs_real.shape[0] == batch_size
        BLOCK_M = 32
        # 3D (attention-output / q-k rope): (head, token) tiles.
        if is_3d:
            if batch_size == 0:
                return x
            RD = max(triton.next_power_of_2(rope_dim), 2)
            block_t, block_h, num_warps = _rope_tile_config(batch_size, n_heads)
            grid = (triton.cdiv(batch_size, block_t), triton.cdiv(n_heads, block_h))
            apply_rotary_emb_tile_kernel[grid](
                x,
                freqs_real,
                positions,
                batch_size,
                n_heads,
                x.stride(0),
                x.stride(1),
                x.stride(2),
                freqs_real.stride(0),
                freqs_real.stride(1),
                USE_POS=(positions is not None),
                IS_INVERSE=inverse,
                RD=RD,
                BLOCK_T=block_t,
                BLOCK_H=block_h,
                num_warps=num_warps,
            )
            return x
        BLOCK_P = max(triton.next_power_of_2(rope_dim // 2), 1)
        grid = (triton.cdiv(batch_size, BLOCK_M), 1)
        apply_rotary_emb_triton_kernel_batched[grid](
            x,
            freqs_real,
            positions,
            rope_dim,
            batch_size,
            x.stride(0),
            0,
            x.stride(-1),
            freqs_real.stride(0),
            freqs_real.stride(1),
            USE_POS=(positions is not None),
            IS_INVERSE=inverse,
            IS_3D=False,
            BLOCK_M=BLOCK_M,
            BLOCK_P=BLOCK_P,
        )
        return x

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
        assert positions.shape == (batch_size,), (
            f"positions shape {positions.shape} != ({batch_size},)"
        )

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
        assert freqs_real.shape[0] == batch_size, (
            f"freqs_cis batch size {freqs_real.shape[0]} != x batch size {batch_size}"
        )

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


# Not bit-exact with the JIT dsv4 fused_norm_rope_inplace (outputs differ by up to
# 1 bf16 ulp), so the two are kept separate.
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
        assert freqs_real.shape[0] == M, (
            f"freqs_cis row count {freqs_real.shape[0]} != M={M}"
        )
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
