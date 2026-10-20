# SPDX-License-Identifier: Apache-2.0
"""Per-token / per-channel FP8 GEMM for gfx1250.

gfx1250 has no tuned hipBLASLt kernels for the shapes diffusion backbones
serve: measured on an MI-class gfx1250 engineering sample at 2.4 GHz,
``torch.mm`` reaches ~70 TFLOPS BF16 and ``torch._scaled_mm`` ~500 TFLOPS FP8,
while a plain Triton tile reaches ~770 TFLOPS BF16 and ~1200 TFLOPS FP8.  The
library paths are therefore slower than the BF16 Inductor templates they would
replace, and FP8 only pays off through a Triton tile.  This module is that
tile.

Correctness note: on this target ``tl.dot`` with FP8 operands miscompiles for
``BLOCK_SIZE_K >= 128`` — the kernel silently returns NaN/zero rather than
failing.  Every config below keeps ``BLOCK_SIZE_K <= 64``, and
:func:`gfx1250_fp8_scaled_mm_supported` refuses shapes a safe config cannot
cover.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages).  BLOCK_K is capped at 64:
# see the module docstring.
_Config = tuple[int, int, int, int, int]

_BLOCK_K = 64


@triton.jit
def _fp8_scaled_mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Wrap the operand rows/columns instead of masking the loads: every address
    # stays in bounds, the tail tile recomputes rows it shares with the
    # previous tile, and only the store is masked.  Masked loads cost ~10% on
    # this target and the tail is at most one tile.
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc *= tl.load(a_scale_ptr + offs_m)[:, None]
    acc *= tl.load(b_scale_ptr + offs_n)[None, :]
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n)[None, :].to(tl.float32)
    c = acc.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def _select_config(M: int, N: int) -> _Config:
    """Tile chosen by an offline sweep over the FLUX.2 720p linear shapes.

    Tuned for a **row-major** ``[K, N]`` B (see the note on `b` in
    :func:`gfx1250_fp8_scaled_mm`); a column-major B is correct but prefers a
    different, slower tile. The sweep covered BLOCK_M/N in {64,128,256},
    BLOCK_K in {32,64}, num_warps in {4,8} and num_stages in {3,4} at M in
    {512, 3600, 4112}, N in {6144, 36864, 55296} and K in
    {6144, 15360, 18432, 24576}, gated on agreeing with a dequantized
    reference. A 128-row tile won every shape; N decides the warp count and,
    below 1024 rows, the tile width.
    """
    if M >= 256 or N >= 16384:
        return (128, 256, _BLOCK_K, 8 if N >= 16384 else 4, 4)
    return (128, 128, _BLOCK_K, 4, 4)


def gfx1250_fp8_scaled_mm_supported(M: int, N: int, K: int) -> bool:
    """Whether :func:`gfx1250_fp8_scaled_mm` can serve an ``[M, K] @ [K, N]``.

    The K loop is unmasked, so K must be a multiple of the (fixed) BLOCK_K.
    Small M is left to the caller's existing path, where quantization and
    launch cost dominate the GEMM anyway.
    """
    return M >= 64 and K % _BLOCK_K == 0 and N % 64 == 0


def gfx1250_fp8_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``(a * a_scale) @ (b * b_scale) + bias`` for FP8 ``a``/``b``.

    Args:
        a: ``[M, K]`` ``float8_e4m3fn`` activations, row-major.
        b: ``[K, N]`` ``float8_e4m3fn`` weights. Pass these **row-major**, i.e.
            contiguous along N. The usual transposed view of a ``[N, K]``
            weight is correct but 1.05-1.33x slower on this target, because a
            BLOCK_N slice of one k-row then reads N single bytes strided by K
            instead of BLOCK_N contiguous ones.
        a_scale: ``M`` per-token scales, any shape with ``M`` elements.
        b_scale: ``N`` per-channel scales, any shape with ``N`` elements.
        out_dtype: output dtype.
        bias: optional ``[N]`` bias, added after dequantization.

    Returns:
        ``[M, N]`` tensor of ``out_dtype``.
    """
    M, K = a.shape
    N = b.shape[1]
    a_scale = a_scale.reshape(-1).to(torch.float32)
    b_scale = b_scale.reshape(-1).to(torch.float32)
    assert a_scale.numel() == M, (
        f"expected {M} activation scales, got {a_scale.numel()}"
    )
    assert b_scale.numel() == N, f"expected {N} weight scales, got {b_scale.numel()}"

    block_m, block_n, block_k, num_warps, num_stages = _select_config(M, N)
    c = torch.empty((M, N), dtype=out_dtype, device=a.device)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    _fp8_scaled_mm_kernel[grid](
        a,
        b,
        c,
        a_scale,
        b_scale,
        bias,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


@triton.jit
def _per_token_quant_fp8_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    K,
    stride_xm,
    stride_qm,
    FP8_MAX: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.max(tl.abs(x)) / FP8_MAX
    # An all-zero row would divide by zero; its quantized values are zero
    # either way, so any positive scale reproduces it exactly.
    scale = tl.where(scale > 0.0, scale, 1.0)
    q = tl.clamp(x / scale, -FP8_MAX, FP8_MAX).to(q_ptr.dtype.element_ty)
    tl.store(q_ptr + row * stride_qm + offs, q, mask=mask)
    tl.store(scale_ptr + row, scale)


def per_token_quant_fp8_gfx1250(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[M, K]`` activations to ``float8_e4m3fn`` with one scale per row.

    Returns ``(q, scale)`` where ``scale`` is ``[M]`` float32 and
    ``q * scale[:, None]`` reconstructs ``x``.
    """
    assert x.dim() == 2
    x = x.contiguous()
    M, K = x.shape
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty(M, dtype=torch.float32, device=x.device)
    block_k = triton.next_power_of_2(K)
    _per_token_quant_fp8_kernel[(M,)](
        x,
        q,
        scale,
        K,
        x.stride(0),
        q.stride(0),
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return q, scale


@triton.jit
def _swiglu_quant_fp8_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    H,
    stride_xm,
    stride_qm,
    FP8_MAX: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    row = tl.program_id(axis=0)

    # The row's activated half is 18432 wide at the served shape; holding it in
    # registers to reach the per-row amax in one pass spills, so stream it
    # twice. The second pass mostly hits L2 -- the row is 72 KiB against a
    # 4 MiB L2 -- which is why this still beats the single-pass form by 1.8x.
    amax = 0.0
    for chunk in tl.static_range(NUM_CHUNKS):
        offs = chunk * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < H
        gate = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        up = tl.load(x_ptr + row * stride_xm + H + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        amax = tl.maximum(amax, tl.max(tl.abs(gate * tl.sigmoid(gate) * up)))

    scale = amax / FP8_MAX
    # An all-zero row would divide by zero; its quantized values are zero
    # either way, so any positive scale reproduces it exactly.
    scale = tl.where(scale > 0.0, scale, 1.0)
    inv_scale = 1.0 / scale

    for chunk in tl.static_range(NUM_CHUNKS):
        offs = chunk * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < H
        gate = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        up = tl.load(x_ptr + row * stride_xm + H + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        act = gate * tl.sigmoid(gate) * up * inv_scale
        tl.store(
            q_ptr + row * stride_qm + offs,
            tl.clamp(act, -FP8_MAX, FP8_MAX).to(q_ptr.dtype.element_ty),
            mask=mask,
        )
    tl.store(scale_ptr + row, scale)


def swiglu_quant_fp8_gfx1250(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``silu(x[:, :H]) * x[:, H:]`` quantized to FP8 with one scale per row.

    Replaces a separate SwiGLU kernel plus a per-token quantization pass: the
    bf16 activation is never written to HBM.

    Args:
        x: ``[M, 2H]`` bf16 gate-then-up projection output. Rows may be
            strided -- this is typically a column slice of a wider fused
            projection, and copying it to make it contiguous costs more than
            the fusion saves.

    Returns:
        ``(q, scale)`` where ``q`` is ``[M, H]`` float8_e4m3fn and ``scale`` is
        ``[M]`` float32, with ``q * scale[:, None]`` reconstructing the
        activation.
    """
    assert x.dim() == 2 and x.shape[1] % 2 == 0
    assert x.stride(1) == 1, "gate/up must be contiguous within a row"
    M, two_h = x.shape
    H = two_h // 2
    q = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty(M, dtype=torch.float32, device=x.device)
    # 2048 x 4 warps measured fastest over {1024, 2048, 4096} x {4, 8, 16}.
    block_h = 2048
    _swiglu_quant_fp8_kernel[(M,)](
        x,
        q,
        scale,
        H,
        x.stride(0),
        q.stride(0),
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        BLOCK_H=block_h,
        NUM_CHUNKS=triton.cdiv(H, block_h),
        num_warps=4,
    )
    return q, scale


@triton.jit
def _fp8_scaled_mm_2seg_kernel(
    a1_ptr,
    a2_ptr,
    b_ptr,
    c_ptr,
    a1_scale_ptr,
    a2_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    M,
    N,
    K1,
    K2,
    stride_a1m,
    stride_a1k,
    stride_a2m,
    stride_a2k,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_ptrs = a1_ptr + offs_m[:, None] * stride_a1m + offs_k[None, :] * stride_a1k
    for _ in range(0, tl.cdiv(K1, BLOCK_K)):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
        a_ptrs += BLOCK_K * stride_a1k
        b_ptrs += BLOCK_K * stride_bk

    # Both segments share one accumulator. Rebasing it onto the second
    # segment's scale here costs one tile multiply at the seam, against the
    # BLOCK_M x BLOCK_N second accumulator a two-scale sum would otherwise
    # need. Both scales are strictly positive (the quantizers map an all-zero
    # row to 1.0), so the ratio is finite.
    a1_scale = tl.load(a1_scale_ptr + offs_m)
    a2_scale = tl.load(a2_scale_ptr + offs_m)
    acc *= (a1_scale / a2_scale)[:, None]

    a_ptrs = a2_ptr + offs_m[:, None] * stride_a2m + offs_k[None, :] * stride_a2k
    for _ in range(0, tl.cdiv(K2, BLOCK_K)):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
        a_ptrs += BLOCK_K * stride_a2k
        b_ptrs += BLOCK_K * stride_bk

    acc *= a2_scale[:, None]
    acc *= tl.load(b_scale_ptr + offs_n)[None, :]
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n)[None, :].to(tl.float32)
    c = acc.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def gfx1250_fp8_scaled_mm_2seg(
    a1: torch.Tensor,
    a2: torch.Tensor,
    b: torch.Tensor,
    a1_scale: torch.Tensor,
    a2_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``concat(a1, a2) @ b`` without materializing the concatenation.

    Each activation segment keeps its own per-token scale, so a caller holding
    two separately quantized halves does not have to requantize them onto a
    shared scale to put them through one GEMM.

    Args:
        a1: ``[M, K1]`` ``float8_e4m3fn``, row-major.
        a2: ``[M, K2]`` ``float8_e4m3fn``, row-major.
        b: ``[K1 + K2, N]`` ``float8_e4m3fn``, row-major (see
            :func:`gfx1250_fp8_scaled_mm`), with the K1 rows first.
        a1_scale: ``M`` per-token scales for ``a1``, all positive.
        a2_scale: ``M`` per-token scales for ``a2``, all positive.
        b_scale: ``N`` per-channel scales.
        out_dtype: output dtype.
        bias: optional ``[N]`` bias, added after dequantization.

    Returns:
        ``[M, N]`` tensor of ``out_dtype``.
    """
    M, K1 = a1.shape
    K2 = a2.shape[1]
    N = b.shape[1]
    assert a2.shape[0] == M, f"segment rows disagree: {M} vs {a2.shape[0]}"
    assert b.shape[0] == K1 + K2, f"b has {b.shape[0]} rows, expected {K1 + K2}"
    a1_scale = a1_scale.reshape(-1).to(torch.float32)
    a2_scale = a2_scale.reshape(-1).to(torch.float32)
    b_scale = b_scale.reshape(-1).to(torch.float32)

    block_m, block_n, block_k, num_warps, num_stages = _select_config(M, N)
    c = torch.empty((M, N), dtype=out_dtype, device=a1.device)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    _fp8_scaled_mm_2seg_kernel[grid](
        a1,
        a2,
        b,
        c,
        a1_scale,
        a2_scale,
        b_scale,
        bias,
        M,
        N,
        K1,
        K2,
        a1.stride(0),
        a1.stride(1),
        a2.stride(0),
        a2.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


def gfx1250_fp8_2seg_supported(M: int, N: int, K1: int, K2: int) -> bool:
    """Whether :func:`gfx1250_fp8_scaled_mm_2seg` can serve these shapes.

    Both K loops are unmasked, so each segment must be a whole number of
    BLOCK_K steps.
    """
    return (
        gfx1250_fp8_scaled_mm_supported(M=M, N=N, K=K1 + K2)
        and K1 % _BLOCK_K == 0
        and K2 % _BLOCK_K == 0
    )
