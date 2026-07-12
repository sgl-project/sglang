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
def apply_rotary_emb_flat_kernel(
    x_ptr,
    fr_ptr,
    pos_ptr,
    n_rows,
    n_heads,
    sx_tok,
    sx_head,
    sx_d,
    sfr_pos,
    sfr_d,
    USE_POS: tl.constexpr,
    IS_INVERSE: tl.constexpr,
    RD: tl.constexpr,
    RDH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    # FLAT-row GPT-J rope: iterate over (token, head) pairs flattened as
    # row = token * n_heads + head, BLOCK_ROWS *consecutive* rows per program.
    # Consecutive rows are sx_head apart in memory (vs sx_tok == n_heads*sx_head
    # for the per-head contig kernel), so the read/write is far less scattered ->
    # ~2x higher achieved HBM bandwidth (cold) on the 128-head attention output
    # (production rope ~168us -> ~59us).
    pid = tl.program_id(0)
    row = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    rmask = row < n_rows
    tok = row // n_heads
    head = row % n_heads
    d = tl.arange(0, RD)
    base = tok[:, None] * sx_tok + head[:, None] * sx_head
    xo = base + d[None, :] * sx_d
    x = tl.load(x_ptr + xo, mask=rmask[:, None], other=0.0).to(tl.float32)

    if USE_POS:
        pos = tl.load(pos_ptr + tok, mask=rmask, other=0)
    else:
        pos = tok
    cos_idx = (d // 2) * 2
    cos = tl.load(
        fr_ptr + pos[:, None] * sfr_pos + cos_idx[None, :] * sfr_d,
        mask=rmask[:, None],
        other=0.0,
    )
    sin = tl.load(
        fr_ptr + pos[:, None] * sfr_pos + (cos_idx[None, :] + 1) * sfr_d,
        mask=rmask[:, None],
        other=0.0,
    )

    x_sin = x * sin
    even = (d % 2 == 0)[None, :]
    if IS_INVERSE:
        x_neg = tl.where(even, -x_sin, x_sin)
    else:
        x_neg = tl.where(even, x_sin, -x_sin)
    x_neg = tl.reshape(x_neg, (BLOCK_ROWS, RDH, 2))
    x_neg = tl.flip(x_neg, 2)
    x_rot = tl.reshape(x_neg, (BLOCK_ROWS, RD))

    out = x * cos + x_rot
    tl.store(x_ptr + xo, out.to(x_ptr.dtype.element_ty), mask=rmask[:, None])


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
        # 3D (attention-output / q-k rope): contiguous-load kernel.
        if is_3d:
            RD = max(triton.next_power_of_2(rope_dim), 2)
            # FLAT-row kernel: process (token, head) pairs flattened as
            # row = token*n_heads + head, BLOCK_ROWS consecutive rows per program.
            # The per-head contig kernel reads BLOCK_M tokens strided by
            # n_heads*head_dim (very scattered on the 128-head attention output) and
            # only reaches ~2.2 TB/s cold; the flat kernel's rows are head_dim apart
            # -> ~4.5 TB/s cold (~2x). Microbench (MI300, 8192x128x64,
            # cold): BLOCK_ROWS=16 + num_warps=1. Numerically bit-exact vs contig.
            FLAT_BLOCK_ROWS = 16
            n_rows = batch_size * n_heads
            grid = (triton.cdiv(n_rows, FLAT_BLOCK_ROWS),)
            apply_rotary_emb_flat_kernel[grid](
                x,
                freqs_real,
                positions,
                n_rows,
                n_heads,
                x.stride(0),
                x.stride(1),
                x.stride(2),
                freqs_real.stride(0),
                freqs_real.stride(1),
                USE_POS=(positions is not None),
                IS_INVERSE=inverse,
                RD=RD,
                RDH=RD // 2,
                BLOCK_ROWS=FLAT_BLOCK_ROWS,
                num_warps=1,
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


# Cache contiguous real/imag halves of each freqs_cis (its .real/.imag are
# strided views, stride=2 on the interleaved layout), keyed by id.
_NPU_ROPE_CONTIG_CACHE: dict[int, tuple] = {}


def _get_contig_freqs_real_imag(
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous (real, imag) halves of ``freqs_cis``, cached by id.

    Used by NPU rope paths to avoid the per-call StridedSlice materialization
    triggered by aclnnIndex over the strided ``.real`` / ``.imag`` views of
    the complex ``freqs_cis`` buffer. First call per freqs_cis pays the
    contiguous() once; later calls reuse the cached tensors.

    All callers within a single MQALayer (outer rope, indexer inner rope,
    compressor epilog rope) get the same freqs_cis instance, so each layer
    materializes at most one (real, imag) pair.
    """
    cache_key = id(freqs_cis)
    cached = _NPU_ROPE_CONTIG_CACHE.get(cache_key)
    if cached is None:
        cached = (freqs_cis.real.contiguous(), freqs_cis.imag.contiguous())
        _NPU_ROPE_CONTIG_CACHE[cache_key] = cached
    return cached


def get_fused_compressor_rope_cos_sin(
    freqs_cis: torch.Tensor,
    positions_cmp: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (cos, sin) tensors shaped ``[T, rope_head_dim]`` for the fused
    compressor op (``torch.ops.custom.compressor``).

    The op consumes ``rope_cos`` / ``rope_sin`` of shape
    ``[min(T, T//cmp_ratio + B), rope_head_dim]`` in bf16/fp16. We index
    the cached contig real/imag halves of the complex ``freqs_cis`` and
    interleave-double the last dim to match the kernel's expected layout
    (matches dsv4_release ``ComplexExpRotaryEmbedding.cos_cache``, which
    is built as ``complex_cache.real.repeat_interleave(2, dim=-1)``).

    Safe to call from inside a captured aclgraph: both ``index_select`` and
    ``repeat_interleave`` over a graph-input ``positions_cmp`` of fixed
    capture-time shape produce static-shape outputs. Identical to what the
    existing inplace_partial_rotary_mul fallback does at
    :func:`v4_rope_inplace_npu`, just without the inverse / 4D-view step.
    """
    real_contig, imag_contig = _get_contig_freqs_real_imag(freqs_cis)
    cos_half = real_contig.index_select(0, positions_cmp)
    sin_half = imag_contig.index_select(0, positions_cmp)
    cos = cos_half.repeat_interleave(2, dim=-1).to(dtype)
    sin = sin_half.repeat_interleave(2, dim=-1).to(dtype)
    return cos, sin


def v4_rope_inplace_npu(
    q_rope: torch.Tensor,
    kv_rope: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool = False,
) -> None:
    """In-place interleaved RoPE for V4 — torch fallback used on NPU.

    Mirrors main's CUDA `fused_rope` kernel: consecutive (even, odd) pairs
    of x form complex pairs, with `freqs_cis` a complex tensor where
    `freqs_cis.real[t, k]` = cos(theta_{t,k}), `freqs_cis.imag` = sin(...)
    indexed by frequency pair k in [0, rope_dim/2).

    NOTE on V4-Flash YARN `mscale`: when the model was trained with the
    YARN magnitude-scale `mscale` ≠ 1.0, the cos/sin values stored in
    `freqs_cis` MUST already be pre-multiplied by `mscale` at precompute
    time — see `precompute_freqs_cis`. This function
    just reads what's stored; it does NOT apply mscale here.

    Prefer the NPU-native `torch.ops.custom.inplace_partial_rotary_mul`:
    the torch fallback differs by ~1 ULP per element vs the kernel because
    torch does bf16*bf16 muls with bf16 accumulation while the NPU kernel
    accumulates in fp32; 43 layers × (Q + K) = 86 rope calls compound that
    drift enough to flip argmax on marginal prompts.
    """
    # Build cos/sin caches in the kernel's expected (T, 1, 1, rope_dim) layout,
    # each freq value repeated twice for the interleaved pairing convention.
    freqs_real_contig, freqs_imag_contig = _get_contig_freqs_real_imag(freqs_cis)
    cos_half = freqs_real_contig[positions]  # (T, rope_dim/2)
    sin_half = freqs_imag_contig[positions]
    if inverse:
        sin_half = -sin_half
    cos_full = cos_half.repeat_interleave(2, dim=-1).to(q_rope.dtype)
    sin_full = sin_half.repeat_interleave(2, dim=-1).to(q_rope.dtype)
    rope_dim = cos_full.shape[-1]
    # repeat_interleave produces a contiguous tensor, so the .view()
    # below already returns a contiguous result — no .contiguous() needed.
    cos4 = cos_full.view(-1, 1, 1, rope_dim)
    sin4 = sin_full.view(-1, 1, 1, rope_dim)
    # q_rope: (T, n_heads, rope_dim) → (T, 1, n_heads, rope_dim) view
    # kv_rope: (T, 1, rope_dim) → (T, 1, 1, rope_dim) view
    q_view = q_rope.unsqueeze(1)
    torch.ops.custom.inplace_partial_rotary_mul(
        q_view,
        cos4,
        sin4,
        rotary_mode="interleave",
        partial_slice=[0, rope_dim],
    )
    if kv_rope is not None:
        if kv_rope.dim() == 3:
            kv_view = kv_rope.unsqueeze(1)
        else:
            kv_view = kv_rope.view(-1, 1, 1, rope_dim)
        torch.ops.custom.inplace_partial_rotary_mul(
            kv_view,
            cos4,
            sin4,
            rotary_mode="interleave",
            partial_slice=[0, rope_dim],
        )
