"""
Packing / quantization utilities for MXFP4A8 (weight E2M1 + block=32 E8M0 scale,
activation FP8 e4m3) on the CUTLASS w4a8 backend.

These helpers are the MXFP4 counterparts of ``int4fp8_utils.py``. They deliberately
reuse the SAME 4-bit nibble packing layout (``order_map = [0, 2, 4, 6, 1, 3, 5, 7]``)
as the int4a8 path, because the kernel-side DirectConvert prmt-LUT for E2M1 mirrors
the int4 one bit-for-bit. The only differences from int4a8 are:

  1. the 4-bit code is an E2M1 code (sign + 3-bit magnitude index), not a
     two's-complement int4 value, and
  2. the K-wise group size is 32 (E8M0 block) instead of 128, and the E8M0
     power-of-2 scale is pre-expanded to bf16 on the host so the kernel's
     post-MMA bf16 group-scale path is reused unchanged.

The int4a8 path (``int4fp8_utils.py``) is left untouched; this is a parallel module.
"""

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

logger = logging.getLogger(__name__)

# E2M1 magnitudes indexed by the low-3-bit code. This ordering matches the
# kernel converter's POS_E4M3s LUT (mxfp4_numeric_conversion.hpp).
#   code: 0     1    2    3    4    5    6    7
#   mag : 0.0  0.5  1.0  1.5  2.0  3.0  4.0  6.0
_E2M1_MAGNITUDES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
_E2M1_MAX = 6.0
MXFP4_BLOCK_SIZE = 32
# Packed-scale interleave width. MUST equal the kernel's
# PackedScalesNum = TileK / GroupSize. mxfp4 uses TileK=128, GroupSize=32 -> 4.
# NOTE: TileK cannot be raised to 256 for mxfp4: that would require an
# Array<bf16,8> (128-bit) scale TMA element, but SM90's to_CUtensorMapDataType
# caps the TMA element at 64-bit (Array<bf16,4>). PackedScalesNum is therefore
# hard-locked to <=4. Every mxfp4 scale buffer -- weight AND activation -- is
# interleaved this wide.
MXFP4_PACKED_SCALES = 4


def _round_to_e2m1_code(x: torch.Tensor) -> torch.Tensor:
    """Map real values (already divided by their block scale) to the nearest
    E2M1 code (sign in bit 3, 3-bit magnitude index in bits 0..2). Returns an
    int8 tensor of nibble codes with the same shape as ``x``."""
    sign = (x < 0).to(torch.int8) << 3
    mag = x.abs().clamp(max=_E2M1_MAX)
    # nearest magnitude index
    grid = _E2M1_MAGNITUDES.to(x.device)  # [8]
    # |mag - grid| over last dim
    idx = (mag.unsqueeze(-1) - grid).abs().argmin(dim=-1).to(torch.int8)
    return (sign | idx).to(torch.int8)


def quantize_mxfp4_blockwise(
    w: torch.Tensor, block_size: int = MXFP4_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to MXFP4 (E2M1) with a per-block E8M0 scale.

    Args:
        w: weight tensor ``[..., K]``; K must be divisible by ``block_size``.
        block_size: K-wise block size (E8M0 block), default 32.

    Returns:
        codes: int8 tensor ``[..., K]`` of E2M1 nibble codes (sign|mag_idx).
        scale_e8m0: uint8 tensor ``[..., K // block_size]`` of E8M0 exponents
            (biased by 127, the standard E8M0 encoding).
    """
    orig_shape = w.shape
    k = orig_shape[-1]
    assert k % block_size == 0, f"K={k} not divisible by block_size={block_size}"
    wf = w.reshape(-1, k // block_size, block_size).float()

    # E8M0 scale: pick the power-of-2 that maps block absmax to <= E2M1_MAX.
    absmax = wf.abs().amax(dim=-1)  # [rows, nblocks]
    absmax = torch.clamp(absmax, min=1e-30)
    # exponent e such that absmax / 2^e <= E2M1_MAX  ->  e = ceil(log2(absmax / MAX))
    exp = torch.ceil(torch.log2(absmax / _E2M1_MAX))
    scale = torch.pow(2.0, exp)  # [rows, nblocks]

    scaled = wf / scale.unsqueeze(-1)
    codes = _round_to_e2m1_code(scaled).reshape(orig_shape)

    # E8M0 encoding: biased exponent (bias = 127), clamp to representable range.
    e8m0 = torch.clamp(exp + 127.0, 0.0, 255.0).to(torch.uint8)
    scale_e8m0 = e8m0.reshape(*orig_shape[:-1], k // block_size)
    return codes, scale_e8m0


def e8m0_to_bf16(scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Expand an E8M0 (biased-exponent) power-of-2 scale to an exact bf16 value.

    Because every E8M0 value is a pure power of two, the expansion is lossless in
    bf16 and lets the kernel reuse the int4a8 post-MMA bf16 group-scale path
    unchanged.
    """
    exp = scale_e8m0.to(torch.float32) - 127.0
    return torch.pow(2.0, exp).to(torch.bfloat16)


_FP8_E4M3_MAX = 448.0


def quantize_activation_mxfp8_blockwise(
    x: torch.Tensor, block_size: int = MXFP4_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize an activation tensor to mxfp8: FP8 (e4m3) data + a per-K-block
    scale, i.e. per-token (row) AND per-block (block_size along K).

    This is the activation-side counterpart of ``quantize_mxfp4_blockwise`` and
    is what makes the CUTLASS mxfp4a8 GEMM a true "per-token + per-block" (mxfp8)
    activation path on SM90, matching the SM100/SM120 native mxfp8 activation.

    Unlike the E8M0 weight scale, the activation block scale is a *general*
    fp32/bf16 amax-derived value (NOT restricted to a power of two): FP8 e4m3
    already carries a mantissa, so a real-valued per-block scale minimises the
    quantization error (this is how flashinfer / trtllm mxfp8 activation works in
    practice). The scale is emitted as bf16 so the kernel post-MMA path consumes
    the same bf16 element type as the (pre-expanded) weight block scale.

    Args:
        x: activation ``[M, K]`` (bf16/fp16/fp32); K must be divisible by
            ``block_size``.
        block_size: K-wise block size, default 32 (matches the E8M0 weight block).

    Returns:
        x_fp8: ``[M, K]`` float8_e4m3fn, the block-scaled activation.
        scale: ``[M, K // block_size]`` bf16, ``block_amax / 448`` per block.
            Dequant is ``x_fp8[m, k] * scale[m, k // block_size]``.
    """
    assert x.dim() == 2, "activation must be 2D [M, K]"
    m, k = x.shape
    assert k % block_size == 0, f"K={k} not divisible by block_size={block_size}"
    nblk = k // block_size

    xf = x.reshape(m, nblk, block_size).float()
    amax = xf.abs().amax(dim=-1)  # [M, nblk]
    amax = torch.clamp(amax, min=1e-12)
    scale = amax / _FP8_E4M3_MAX  # [M, nblk], real-valued
    xq = (xf / scale.unsqueeze(-1)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    x_fp8 = xq.reshape(m, k).to(torch.float8_e4m3fn)
    return x_fp8, scale.to(torch.bfloat16)


@triton.jit
def _fused_mxfp8_quant_kernel(
    x_ptr,
    xq_ptr,
    sc_ptr,
    K,
    nblk,
    FP8_MAX: tl.constexpr,
    BLK: tl.constexpr,
):
    """One program per (row, K-block): load a [1, BLK] block, compute the block
    amax, emit FP8 e4m3 data + the per-block bf16 scale (amax / FP8_MAX). This is
    a single-pass fused replacement for the pure-PyTorch multi-kernel
    ``quantize_activation_mxfp8_blockwise`` and is bit-exact with it."""
    row = tl.program_id(0)
    blk = tl.program_id(1)
    offs = row * K + blk * BLK + tl.arange(0, BLK)
    x = tl.load(x_ptr + offs).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), 1e-12)
    scale = amax / FP8_MAX
    xq = tl.minimum(tl.maximum(x / scale, -FP8_MAX), FP8_MAX)
    tl.store(xq_ptr + offs, xq.to(xq_ptr.dtype.element_ty))
    tl.store(sc_ptr + row * nblk + blk, scale.to(sc_ptr.dtype.element_ty))


def quantize_activation_mxfp8_native(
    x: torch.Tensor, block_size: int = MXFP4_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token + per-block mxfp8 activation quant via the native hand-tuned
    CUDA kernel ``sgl_per_token_group_quant_8bit`` (the same kernel DeepSeek-V4's
    block-fp8 path uses through ``sglang_per_token_group_quant_fp8``).

    Returns ``(x_fp8 [M, K], scale [M, K//block] fp32)``. Benchmarked 1.4x
    (decode) to 7.8x (prefill) faster than the Triton fallback at group=32 on
    H20; the scale is fp32 (higher precision than the Triton bf16 scale) and is
    down-cast to bf16 when scattered into the packed kernel buffer.

    Uses the tested high-level ``sglang_per_token_group_quant_fp8`` wrapper,
    which on CUDA routes to the live ``per_token_group_quant`` kernel and
    allocates the output-scale buffer with the correct row-major ``[M, nblk]``
    fp32 layout.
    """
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    assert x.dim() == 2, "activation must be 2D [M, K]"
    m, k = x.shape
    assert k % block_size == 0, f"K={k} not divisible by block_size={block_size}"
    if m == 0:
        nblk = k // block_size
        return (
            torch.empty(m, k, dtype=torch.float8_e4m3fn, device=x.device),
            torch.empty(m, nblk, dtype=torch.float32, device=x.device),
        )
    x_fp8, scale = sglang_per_token_group_quant_fp8(
        x.contiguous(), block_size, eps=1e-10
    )
    return x_fp8, scale


def silu_and_mul_mxfp8_quant_native(
    c1: torch.Tensor,
    n: int,
    block_size: int = MXFP4_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused SwiGLU (silu_and_mul) + per-token/per-block mxfp8 quant of the GEMM1
    output, via the native kernel's ``fuse_silu_and_mul`` path.

    Input ``c1 [M, 2n]`` (gate | up); returns ``(x_fp8 [M, n], scale [M, n//block]
    fp32)``. 1.5x-8.7x faster than the Triton fused kernel. NOTE: unlike the
    Triton path this native kernel has no swiglu clamp; it is used only when
    ``swiglu_limit`` is None (the clamp path falls back to Triton).

    Uses the tested high-level ``sglang_per_token_group_quant_fp8`` wrapper with
    ``fuse_silu_and_mul=True``, which allocates the correct ``[M, n//block]``
    fp32 output-scale and routes to the live ``per_token_group_quant`` kernel.
    """
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    m = c1.shape[0]
    assert n % block_size == 0, f"n={n} not divisible by block_size={block_size}"
    if m == 0:
        nblk = n // block_size
        return (
            torch.empty(m, n, dtype=torch.float8_e4m3fn, device=c1.device),
            torch.empty(m, nblk, dtype=torch.float32, device=c1.device),
        )
    x_fp8, scale = sglang_per_token_group_quant_fp8(
        c1.contiguous(), block_size, eps=1e-10, fuse_silu_and_mul=True
    )
    return x_fp8, scale


def quantize_activation_mxfp8_blockwise_fused(
    x: torch.Tensor, block_size: int = MXFP4_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused single-pass Triton mxfp8 activation quant.

    Numerically bit-exact with ``quantize_activation_mxfp8_blockwise`` (the FP8
    output bytes and the bf16 block scale match exactly), but issues a single
    kernel instead of the pure-PyTorch reshape/.float()/amax/div/clamp/.to()
    chain (which materialises an fp32 copy and launches 5-6 elementwise kernels).
    Verified bit-exact and 1.5-4x faster on H20. Kept as a portable fallback;
    the production path prefers ``quantize_activation_mxfp8_native``.
    """
    assert x.dim() == 2, "activation must be 2D [M, K]"
    m, k = x.shape
    assert k % block_size == 0, f"K={k} not divisible by block_size={block_size}"
    nblk = k // block_size
    if m == 0:
        return (
            torch.empty(0, k, dtype=torch.float8_e4m3fn, device=x.device),
            torch.empty(0, nblk, dtype=torch.bfloat16, device=x.device),
        )
    x_fp8 = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty(m, nblk, dtype=torch.bfloat16, device=x.device)
    _fused_mxfp8_quant_kernel[(m, nblk)](
        x, x_fp8, scale, k, nblk, FP8_MAX=_FP8_E4M3_MAX, BLK=block_size
    )
    return x_fp8, scale


@triton.jit
def _fused_silu_mul_mxfp8_quant_kernel(
    c1_ptr,
    xq_ptr,
    sc_ptr,
    N,
    twoN,
    nblk,
    HAS_LIM: tl.constexpr,
    LIM,
    FP8_MAX: tl.constexpr,
    BLK: tl.constexpr,
):
    """Fused SwiGLU (optional clamp) + silu_and_mul + per-block mxfp8 quant.

    Input ``c1[M, 2N]`` (gate = c1[:, :N], up = c1[:, N:]); one program per
    (row, K-block of N). Computes (optionally clamped) ``silu(gate) * up`` in
    fp32 with a bf16 round-trip to match the reference ``silu_and_mul`` bf16
    intermediate exactly, then emits FP8 e4m3 data + per-block bf16 scale.

    This fuses three ops (swiglu clamp, silu_and_mul writing a [M,N] bf16
    intermediate, and re-reading it for quant) into a single kernel, removing the
    bf16 intermediate write+read round-trip. Verified bit-exact (0 mismatch)
    with the ``silu_and_mul`` + ``quantize_activation_mxfp8_blockwise_fused``
    two-kernel path.
    """
    row = tl.program_id(0)
    blk = tl.program_id(1)
    col = blk * BLK + tl.arange(0, BLK)
    g = tl.load(c1_ptr + row * twoN + col).to(tl.float32)
    u = tl.load(c1_ptr + row * twoN + N + col).to(tl.float32)
    if HAS_LIM:
        g = tl.minimum(g, LIM)
        u = tl.minimum(tl.maximum(u, -LIM), LIM)
    act = (g / (1.0 + tl.exp(-g))) * u
    act = act.to(tl.bfloat16).to(tl.float32)  # match bf16 intermediate storage
    amax = tl.maximum(tl.max(tl.abs(act)), 1e-12)
    scale = amax / FP8_MAX
    xq = tl.minimum(tl.maximum(act / scale, -FP8_MAX), FP8_MAX)
    tl.store(xq_ptr + row * N + col, xq.to(xq_ptr.dtype.element_ty))
    tl.store(sc_ptr + row * nblk + blk, scale.to(sc_ptr.dtype.element_ty))


def silu_and_mul_mxfp8_quant_fused(
    c1: torch.Tensor,
    n: int,
    swiglu_limit: Optional[float] = None,
    block_size: int = MXFP4_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused SwiGLU + silu_and_mul + mxfp8 block quant of the GEMM1 output.

    Replaces the separate ``silu_and_mul(c1, intermediate)`` +
    ``quantize_activation_mxfp8_blockwise(intermediate)`` pair with a single
    Triton kernel, avoiding the ``[M, n]`` bf16 intermediate write+read.

    Args:
        c1: GEMM1 output ``[M, 2n]`` (gate | up), bf16.
        n: half the last dim (intermediate hidden size).
        swiglu_limit: if set, DeepSeek-V4 swiglu clamp (gate<=L, up in [-L, L]).
        block_size: K-wise block size (default 32).

    Returns:
        x_fp8: ``[M, n]`` float8_e4m3fn.
        scale: ``[M, n // block_size]`` bf16.
    """
    m = c1.shape[0]
    assert n % block_size == 0, f"n={n} not divisible by block_size={block_size}"
    nblk = n // block_size
    if m == 0:
        return (
            torch.empty(0, n, dtype=torch.float8_e4m3fn, device=c1.device),
            torch.empty(0, nblk, dtype=torch.bfloat16, device=c1.device),
        )
    x_fp8 = torch.empty(m, n, dtype=torch.float8_e4m3fn, device=c1.device)
    scale = torch.empty(m, nblk, dtype=torch.bfloat16, device=c1.device)
    _fused_silu_mul_mxfp8_quant_kernel[(m, nblk)](
        c1,
        x_fp8,
        scale,
        n,
        2 * n,
        nblk,
        HAS_LIM=(swiglu_limit is not None),
        LIM=(float(swiglu_limit) if swiglu_limit is not None else 0.0),
        FP8_MAX=_FP8_E4M3_MAX,
        BLK=block_size,
    )
    return x_fp8, scale


@triton.jit
def _fused_silu_mul_quant_scatter_kernel(
    c1_ptr,
    xq_ptr,
    as_ptr,
    eid_ptr,
    lrow_ptr,
    mpad_ptr,
    estart_ptr,
    N,
    twoN,
    HAS_LIM: tl.constexpr,
    LIM,
    FP8_MAX: tl.constexpr,
    BLK: tl.constexpr,
    A: tl.constexpr,
):
    """Fused SwiGLU(+clamp) + silu_and_mul + mxfp8 quant + scale scatter.

    Combines ``_fused_silu_mul_mxfp8_quant_kernel`` with the grouped
    activation-scale scatter, so the GEMM2 activation and its packed block-scale
    buffer are produced in a single pass -- no ``[M, nblk]`` scale tensor and no
    separate scatter launch. Bit-identical to the two-kernel eager path.
    """
    row = tl.program_id(0)
    blk = tl.program_id(1)
    e = tl.load(eid_ptr + row)
    lr = tl.load(lrow_ptr + row)
    mpad = tl.load(mpad_ptr + e)
    estart = tl.load(estart_ptr + e)
    col = blk * BLK + tl.arange(0, BLK)
    g = tl.load(c1_ptr + row * twoN + col).to(tl.float32)
    u = tl.load(c1_ptr + row * twoN + N + col).to(tl.float32)
    if HAS_LIM:
        g = tl.minimum(g, LIM)
        u = tl.minimum(tl.maximum(u, -LIM), LIM)
    act = (g / (1.0 + tl.exp(-g))) * u
    act = act.to(tl.bfloat16).to(tl.float32)  # match bf16 intermediate storage
    amax = tl.maximum(tl.max(tl.abs(act)), 1e-12)
    scale = amax / FP8_MAX
    xq = tl.minimum(tl.maximum(act / scale, -FP8_MAX), FP8_MAX)
    tl.store(xq_ptr + row * N + col, xq.to(xq_ptr.dtype.element_ty))
    dst = estart + (blk // A) * (mpad * A) + lr * A + (blk % A)
    tl.store(as_ptr + dst, scale.to(as_ptr.dtype.element_ty))


def silu_and_mul_quant_and_build_scale_fused(
    c1: torch.Tensor,
    n: int,
    expert_offsets: torch.Tensor,
    swiglu_limit: Optional[float] = None,
    block_size: int = MXFP4_BLOCK_SIZE,
):
    """Fused SwiGLU + silu_and_mul + mxfp8 quant + grouped scale build.

    Returns ``(x_fp8, as_packed, as_strides)`` for the GEMM2 activation, fusing
    ``silu_and_mul_mxfp8_quant_fused`` and the compact grouped-scale build into a
    single Triton pass. Bit-exact with that two-step eager path.
    """
    assert torch.is_tensor(expert_offsets)
    device = c1.device
    m = c1.shape[0]
    assert n % block_size == 0, f"n={n} not divisible by block_size={block_size}"
    nblk = n // block_size
    A = MXFP4_PACKED_SCALES
    assert nblk % A == 0, f"n//block={nblk} must be a multiple of {A}"

    off = expert_offsets.to(torch.int64)
    num_experts = off.numel() - 1
    counts = off[1:] - off[:-1]
    m_pads = ((counts + 1) // 2) * 2
    elem_per = m_pads * nblk
    estart = torch.zeros(num_experts, dtype=torch.int64, device=device)
    if num_experts > 1:
        estart[1:] = torch.cumsum(elem_per, 0)[:-1]
    total = (m + num_experts) * nblk

    x_fp8 = torch.empty(m, n, dtype=torch.float8_e4m3fn, device=device)
    as_packed = torch.zeros(total, dtype=torch.bfloat16, device=device)
    as_strides = torch.stack([m_pads, m_pads], dim=1).contiguous()
    if m == 0:
        return x_fp8, as_packed, as_strides

    row_ids = torch.arange(m, device=device, dtype=torch.int64)
    eid = torch.searchsorted(off[1:], row_ids, right=True)
    lrow = row_ids - off[eid]
    _fused_silu_mul_quant_scatter_kernel[(m, nblk)](
        c1,
        x_fp8,
        as_packed,
        eid.to(torch.int32),
        lrow.to(torch.int32),
        m_pads.to(torch.int32),
        estart,
        n,
        2 * n,
        HAS_LIM=(swiglu_limit is not None),
        LIM=(float(swiglu_limit) if swiglu_limit is not None else 0.0),
        FP8_MAX=_FP8_E4M3_MAX,
        BLK=block_size,
        A=A,
    )
    return x_fp8, as_packed, as_strides


def interleave_act_scale_mxfp8(
    scale: torch.Tensor, alignment: int = MXFP4_PACKED_SCALES
) -> torch.Tensor:
    """Interleave a per-token+per-block activation scale ``[M, K//block]`` into
    the kernel's physical activation-scale layout ``[K//(block*A), M*A]`` where
    ``A = alignment = PackedScalesNum``.

    This mirrors the weight-scale interleave (``interleave_scales`` in the
    int4a8 path) but tiled over tokens (M) instead of weight channels (N). The
    kernel's activation-scale TMA expects **token unit-stride** with the scale-K
    stride equal to M (tokens per expert), i.e. ``as_strides = M``. This exact
    layout was verified bit-exact (rel_mean = 0.0000) by the single-GEMM test
    ``tests/test_cutlass_mxfp4a8_moe_mm.py::run_case_mxfp8_act``.

    ``K//block`` must be a multiple of ``alignment`` (=8 for TileK=256); for K a
    multiple of 256 and block=32 this holds (K/32 multiple of 8).
    """
    m, nblk = scale.shape
    assert nblk % alignment == 0, f"K//block={nblk} not divisible by {alignment}"
    si = scale.reshape(m, nblk // alignment, alignment)  # [M, nblk/A, A]
    si = si.permute(1, 0, 2)  # [nblk/A, M, A]
    si = si.reshape(nblk // alignment, m * alignment)  # [nblk/A, M*A]
    return si.contiguous()


def quantize_activation_mxfp8_blockwise_grouped(
    x: torch.Tensor, block_size: int = MXFP4_BLOCK_SIZE, pad_even: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Grouped mxfp8 block-quant helper: quantize an activation tensor to FP8
    (e4m3) data + a per-token+per-block bf16 scale, padding the token dimension
    to an even count so the kernel's TMA layout constraints are satisfied.

    This is the grouped counterpart of ``quantize_activation_mxfp8_blockwise``:
    it applies the same per-token + per-block (block_size along K) real-valued
    ``amax / 448`` scale, but additionally pads M up to the next even value
    (matching the kernel's even-padding requirement) with zeros.

    Args:
        x: activation ``[M, K]`` (bf16/fp16/fp32); K must be divisible by
            ``block_size``.
        block_size: K-wise block size, default 32 (matches the E8M0 weight block).
        pad_even: if True (default), pad M up to the next even value with zeros.

    Returns:
        x_fp8: ``[M_padded, K]`` float8_e4m3fn, the block-scaled activation.
        scale: ``[M_padded, K // block_size]`` bf16, ``block_amax / 448`` per block.
    """
    assert x.dim() == 2, "activation must be 2D [M, K]"
    m, k = x.shape
    assert k % block_size == 0, f"K={k} not divisible by block_size={block_size}"

    if pad_even and (m % 2 != 0):
        pad = torch.zeros(1, k, dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad], dim=0)
        m = x.shape[0]

    x_fp8, scale = quantize_activation_mxfp8_blockwise(x, block_size=block_size)
    return x_fp8, scale


def _build_grouped_act_block_scale_compact(
    scale: torch.Tensor,
    expert_offsets_host,
    block_size: int = MXFP4_BLOCK_SIZE,
):
    """Compact builder used by the eager path.

    This keeps the buffer tightly packed by concatenating the per-expert blocks
    with an expert-local even padding ``M_pad``. It is numerically exact and
    memory-efficient, but it uses host-side expert offsets and dynamic-shape
    tensor construction, so it must stay out of CUDA graph capture.
    """
    device = scale.device
    num_experts = len(expert_offsets_host) - 1
    nblk = scale.shape[1]
    A = MXFP4_PACKED_SCALES
    assert nblk % A == 0, f"K//block={nblk} must be a multiple of {A}"

    blocks = []
    m_pads = []
    for e in range(num_experts):
        s0 = int(expert_offsets_host[e])
        s1 = int(expert_offsets_host[e + 1])
        se = scale[s0:s1].to(torch.bfloat16)  # [M_e, nblk]; kernel consumes bf16
        m_e = se.shape[0]
        m_pad = (m_e + 1) & ~1  # round up to even
        if m_pad != m_e:
            pad = torch.zeros(m_pad - m_e, nblk, dtype=se.dtype, device=device)
            se = torch.cat([se, pad], dim=0)
        # A-wide interleave over K-blocks: [M_pad, nblk] -> [nblk/A, M_pad*A]
        si = se.reshape(m_pad, nblk // A, A).permute(1, 0, 2).reshape(nblk // A, m_pad * A)
        blocks.append(si.reshape(-1).contiguous())
        m_pads.append(m_pad)

    as_packed = (
        torch.cat(blocks).contiguous()
        if blocks
        else torch.zeros(0, dtype=torch.bfloat16, device=device)
    )
    as_strides = torch.tensor(
        [[m_pads[e]] * 2 for e in range(num_experts)],
        dtype=torch.int64,
        device=device,
    )
    return as_packed, as_strides


@triton.jit
def _scatter_act_block_scale_kernel(
    sc_ptr,
    out_ptr,
    eid_ptr,
    lrow_ptr,
    mpad_ptr,
    estart_ptr,
    nblk,
    A: tl.constexpr,
    BLKN: tl.constexpr,
):
    """Scatter a per-token+per-block scale row into the A-wide interleaved,
    per-expert even-padded compact buffer, fully device-side.

    dst = expert_start + (blk // A) * (m_pad * A) + local_row * A + (blk % A)
    which is exactly the layout produced by ``_build_grouped_act_block_scale_compact``.
    (A = PackedScalesNum = TileK / GroupSize = 4 for the TileK=128 mxfp4 kernel;
    SM90 TMA caps the packed scale element at 64-bit = Array<bf16,4>, so TileK is
    hard-locked to 128 and PackedScalesNum to 4.)
    """
    row = tl.program_id(0)
    e = tl.load(eid_ptr + row)
    lr = tl.load(lrow_ptr + row)
    mpad = tl.load(mpad_ptr + e)
    estart = tl.load(estart_ptr + e)
    blk = tl.program_id(1) * BLKN + tl.arange(0, BLKN)
    mask = blk < nblk
    val = tl.load(sc_ptr + row * nblk + blk, mask=mask, other=0.0)
    dst = estart + (blk // A) * (mpad * A) + lr * A + (blk % A)
    tl.store(out_ptr + dst, val, mask=mask)


def _build_grouped_act_block_scale_compact_device(
    scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    block_size: int = MXFP4_BLOCK_SIZE,
):
    """Device-side equivalent of ``_build_grouped_act_block_scale_compact``.

    Produces a bit-identical packed buffer + strides but replaces the Python
    per-expert for-loop (a constant ~485us host cost for 32 experts, independent
    of batch size) with a cumulative-offset computation plus a single Triton
    scatter kernel. Verified bit-exact with the Python compact builder.

    Requires a tensor ``expert_offsets`` (device int). Used by the eager path;
    the capture-safe fixed-stride builder is unchanged.
    """
    assert torch.is_tensor(expert_offsets)
    device = scale.device
    nblk = scale.shape[1]
    A = MXFP4_PACKED_SCALES
    assert nblk % A == 0, f"K//block={nblk} must be a multiple of {A}"

    off = expert_offsets.to(torch.int64)
    num_experts = off.numel() - 1
    total_m = scale.shape[0]

    counts = off[1:] - off[:-1]
    m_pads = ((counts + 1) // 2) * 2  # round up to even
    elem_per = m_pads * nblk
    estart = torch.zeros(num_experts, dtype=torch.int64, device=device)
    if num_experts > 1:
        estart[1:] = torch.cumsum(elem_per, 0)[:-1]
    # Static upper bound for the packed buffer size, computed WITHOUT a device
    # sync: each expert rounds its token count up to even (adds at most +1 row),
    # so total padded rows <= total_m + num_experts. Over-allocating avoids the
    # ``.item()`` host-device sync that previously stalled the launch pipeline on
    # every MoE layer (a fixed ~200us cost, independent of batch). The trailing
    # region is never read: the GEMM indexes each expert by ``estart``/``m_pads``.
    total = (total_m + num_experts) * nblk

    # The kernel's post-MMA group-scale path consumes bf16, so the packed buffer
    # is always bf16. The scatter kernel casts on store, so ``scale`` may be
    # bf16 (Triton quant) OR fp32 (native sgl_per_token_group_quant_8bit).
    as_packed = torch.zeros(total, dtype=torch.bfloat16, device=device)
    if total_m > 0:
        row_ids = torch.arange(total_m, device=device, dtype=torch.int64)
        eid = torch.searchsorted(off[1:], row_ids, right=True)
        lrow = row_ids - off[eid]
        BLKN = triton.next_power_of_2(nblk)
        _scatter_act_block_scale_kernel[(total_m, triton.cdiv(nblk, BLKN))](
            scale,
            as_packed,
            eid.to(torch.int32),
            lrow.to(torch.int32),
            m_pads.to(torch.int32),
            estart,
            nblk,
            A=A,
            BLKN=BLKN,
        )
    as_strides = torch.stack([m_pads, m_pads], dim=1).contiguous()
    return as_packed, as_strides


@triton.jit
def _fused_quant_scatter_kernel(
    x_ptr,
    xq_ptr,
    as_ptr,
    eid_ptr,
    lrow_ptr,
    mpad_ptr,
    estart_ptr,
    K,
    FP8_MAX: tl.constexpr,
    BLK: tl.constexpr,
    A: tl.constexpr,
):
    """Fused mxfp8 quant + activation-scale scatter.

    One program per (row, K-block). Computes the FP8 e4m3 data AND the per-block
    bf16 scale (amax / FP8_MAX) exactly as ``_fused_mxfp8_quant_kernel``, but
    instead of writing the scale to a dense ``[M, nblk]`` tensor (later re-read
    and scattered by a second kernel), it scatters the scale straight into the
    kernel's A-wide, per-expert even-padded packed layout:

        dst = estart[e] + (blk // A) * (mpad[e] * A) + lrow * A + (blk % A)

    which is bit-identical to ``_build_grouped_act_block_scale_compact_device``.
    This removes the intermediate scale tensor, one full kernel launch, and the
    M*nblk scale write+read round-trip.
    """
    row = tl.program_id(0)
    blk = tl.program_id(1)
    e = tl.load(eid_ptr + row)
    lr = tl.load(lrow_ptr + row)
    mpad = tl.load(mpad_ptr + e)
    estart = tl.load(estart_ptr + e)
    offs = row * K + blk * BLK + tl.arange(0, BLK)
    x = tl.load(x_ptr + offs).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), 1e-12)
    scale = amax / FP8_MAX
    xq = tl.minimum(tl.maximum(x / scale, -FP8_MAX), FP8_MAX)
    tl.store(xq_ptr + offs, xq.to(xq_ptr.dtype.element_ty))
    dst = estart + (blk // A) * (mpad * A) + lr * A + (blk % A)
    tl.store(as_ptr + dst, scale.to(as_ptr.dtype.element_ty))


def quantize_activation_and_build_scale_fused(
    x: torch.Tensor,
    expert_offsets: torch.Tensor,
    block_size: int = MXFP4_BLOCK_SIZE,
):
    """Fused single-pass mxfp8 quant + grouped activation-scale build.

    Returns ``(x_fp8, as_packed, as_strides)`` -- exactly what the pair
    ``quantize_activation_mxfp8_blockwise_fused`` +
    ``_build_grouped_act_block_scale_compact_device`` produces, but the block
    scale is scattered into the packed buffer inside the quant kernel itself, so
    no intermediate ``[M, nblk]`` scale tensor is materialised and one full
    Triton launch (+ its M*nblk scale reload) is removed from every MoE GEMM.

    Bit-exact with the two-kernel eager path. Used by the eager/prefill path;
    the capture-safe decode builder is unchanged.
    """
    assert x.dim() == 2, "activation must be 2D [M, K]"
    assert torch.is_tensor(expert_offsets)
    device = x.device
    m, k = x.shape
    assert k % block_size == 0, f"K={k} not divisible by block_size={block_size}"
    nblk = k // block_size
    A = MXFP4_PACKED_SCALES
    assert nblk % A == 0, f"K//block={nblk} must be a multiple of {A}"

    off = expert_offsets.to(torch.int64)
    num_experts = off.numel() - 1

    counts = off[1:] - off[:-1]
    m_pads = ((counts + 1) // 2) * 2  # round up to even
    elem_per = m_pads * nblk
    estart = torch.zeros(num_experts, dtype=torch.int64, device=device)
    if num_experts > 1:
        estart[1:] = torch.cumsum(elem_per, 0)[:-1]
    # Static upper bound (no host sync): each expert adds at most +1 padded row.
    total = (m + num_experts) * nblk

    x_fp8 = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=device)
    as_packed = torch.zeros(total, dtype=torch.bfloat16, device=device)
    as_strides = torch.stack([m_pads, m_pads], dim=1).contiguous()
    if m == 0:
        return x_fp8, as_packed, as_strides

    row_ids = torch.arange(m, device=device, dtype=torch.int64)
    eid = torch.searchsorted(off[1:], row_ids, right=True)
    lrow = row_ids - off[eid]
    _fused_quant_scatter_kernel[(m, nblk)](
        x,
        x_fp8,
        as_packed,
        eid.to(torch.int32),
        lrow.to(torch.int32),
        m_pads.to(torch.int32),
        estart,
        k,
        FP8_MAX=_FP8_E4M3_MAX,
        BLK=block_size,
        A=A,
    )
    return x_fp8, as_packed, as_strides


def _build_grouped_act_block_scale_capture_safe(
    scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    block_size: int = MXFP4_BLOCK_SIZE,
):
    """Graph-safe builder for decode/spec CUDA graph capture.

    Unlike the eager compact builder, this path never calls ``.cpu()`` /
    ``.tolist()`` and never creates dynamic-shape outputs from per-expert token
    counts. Instead, every expert uses the SAME fixed even token stride
    ``M_stride = round_up_even(total_m)``, and the per-expert scale block is
    stored in a dense buffer of shape ``[E, K//(block*4), M_stride*4]``.

    This wastes some memory versus the compact eager layout, but for decode graph
    capture ``total_m`` is the captured batch size times ``topk`` (small and
    static), while the fixed even stride removes the original TMA alignment
    issue for odd per-expert token counts and makes the whole builder
    CUDA-graph-recordable.
    """
    assert torch.is_tensor(expert_offsets), "capture-safe path requires tensor expert_offsets"
    device = scale.device
    nblk = scale.shape[1]
    A = MXFP4_PACKED_SCALES
    assert nblk % A == 0, f"K//block={nblk} must be a multiple of {A}"

    num_experts = expert_offsets.numel() - 1
    total_m = scale.shape[0]
    m_stride = max(2, (total_m + 1) & ~1)

    # The kernel's post-MMA group-scale path consumes bf16, so the packed buffer
    # is always bf16. ``scale`` may be bf16 (Triton quant) or fp32 (native
    # sgl_per_token_group_quant_8bit); the assignment below casts fp32 -> bf16.
    grouped = torch.zeros(
        (num_experts, m_stride, nblk),
        dtype=torch.bfloat16,
        device=device,
    )

    if total_m > 0:
        row_ids = torch.arange(total_m, device=device, dtype=expert_offsets.dtype)
        expert_ids = torch.searchsorted(expert_offsets[1:], row_ids, right=True)
        row_offsets = expert_offsets.index_select(0, expert_ids)
        local_rows = (row_ids - row_offsets).to(torch.long)
        grouped[expert_ids.to(torch.long), local_rows, :] = scale.to(torch.bfloat16)

    packed = (
        grouped.reshape(num_experts, m_stride, nblk // A, A)
        .permute(0, 2, 1, 3)
        .reshape(num_experts, nblk // A, m_stride * A)
        .contiguous()
    )
    as_packed = packed.reshape(-1).contiguous()
    as_strides = torch.full(
        (num_experts, 2),
        m_stride,
        dtype=torch.int64,
        device=device,
    )
    return as_packed, as_strides


def build_grouped_act_block_scale(
    scale: torch.Tensor,
    expert_offsets,
    block_size: int = MXFP4_BLOCK_SIZE,
    capture_safe: Optional[bool] = None,
):
    """Build the activation block-scale buffer + strides consumed by CUTLASS.

    There are two layouts:

    1. eager compact layout: per-expert concatenation with expert-local even
       padding ``M_pad`` (minimal memory footprint).
    2. capture-safe layout: fixed even ``M_stride`` for every expert, fully
       device-side and static-shape so it is CUDA-graph-recordable.

    The capture-safe layout is selected automatically inside CUDA graph capture,
    or can be forced explicitly by passing ``capture_safe=True`` for testing.
    """
    if capture_safe is None:
        capture_safe = get_is_capture_mode()

    if capture_safe:
        return _build_grouped_act_block_scale_capture_safe(
            scale, expert_offsets, block_size=block_size
        )

    # Eager path: prefer the fully device-side compact builder (single Triton
    # scatter, bit-exact with the Python compact layout) which removes the
    # ~485us Python per-expert for-loop. Fall back to the Python builder only
    # when expert_offsets is already a host list (rare / non-tensor callers).
    if torch.is_tensor(expert_offsets):
        return _build_grouped_act_block_scale_compact_device(
            scale, expert_offsets, block_size=block_size
        )
    expert_offsets_host = expert_offsets
    return _build_grouped_act_block_scale_compact(
        scale, expert_offsets_host, block_size=block_size
    )


# E2M1 nibble interleave order used by the int4fp8 ``pack_*_to_int32`` helper.
# NOTE: the CUTLASS mxfp4a8 kernel does NOT expect this reorder for its packed
# int8 weight operand (see ``repack_hf_mxfp4_to_kernel`` below).
_ORDER_MAP = [0, 2, 4, 6, 1, 3, 5, 7]


def repack_hf_mxfp4_to_kernel(w_uint8: torch.Tensor) -> torch.Tensor:
    """Convert HF-packed MXFP4 (E2M1) bytes into the kernel's int8 weight layout.

    HF stores two E2M1 nibbles per byte in *natural* order (nibble ``2j`` in the
    low half of byte ``j``, nibble ``2j+1`` in the high half). This is EXACTLY
    the byte layout the CUTLASS mxfp4a8 grouped-GEMM expects for its packed int8
    weight operand ``b = [E, N, K//2]``.

    This was verified empirically with a single-GEMM bit-exact comparison
    (``tests/test_cutlass_mxfp4a8_moe_mm.py``): the natural nibble packing
    reproduces the bf16-dequant golden to rel_mean = 0, while applying the
    ``order_map = [0, 2, 4, 6, 1, 3, 5, 7]`` reorder produces garbage
    (rel_mean ~= 1.2). The ``order_map`` reorder only applies to the int4a8
    ``pack_int4_to_int32`` path (which packs into int32 words with a different
    prmt-LUT layout), NOT to this int8 grouped-GEMM path.

    Therefore this routine is a bit-preserving identity: the HF bytes are passed
    through unchanged (reinterpreted as int8 so two's-complement high bytes keep
    their bit pattern for the kernel's per-nibble decode).

    Args:
        w_uint8: HF-packed weights ``[..., cols]`` (uint8/int8), ``cols`` bytes
            per row, i.e. ``2*cols`` E2M1 codes along the last logical dim.

    Returns:
        int8 tensor of the same shape, bit-identical to the input bytes.
    """
    assert w_uint8.dtype in (torch.uint8, torch.int8)
    if w_uint8.dtype == torch.int8:
        return w_uint8.contiguous()
    # uint8 -> int8 is a pure bit reinterpretation (both 1 byte), which is what
    # the kernel's per-nibble decode operates on.
    return w_uint8.view(torch.int8).contiguous()


def pack_mxfp4_to_int32(to_pack: torch.Tensor, reorder: bool = True) -> torch.Tensor:
    """Pack E2M1 nibble codes into int32 words using the SAME interleave layout
    as ``pack_int4_to_int32``. ``to_pack`` holds 4-bit codes (0..15) as int8.
    """
    if to_pack.ndim > 2:
        raise ValueError("Pack: Only supports tensors with ndim <= 2.")

    order_map = [0, 2, 4, 6, 1, 3, 5, 7] if reorder else [0, 1, 2, 3, 4, 5, 6, 7]
    pack_num = 8
    if to_pack.ndim == 2:
        new_c = to_pack.shape[1] // pack_num
        packed = torch.zeros(
            to_pack.shape[0], new_c, dtype=torch.int32, device=to_pack.device
        )
        for c in range(new_c):
            for i in range(pack_num):
                col = (to_pack[:, c * pack_num + order_map[i]].to(torch.int32)) & 0x0F
                packed[:, c] = torch.bitwise_or(
                    packed[:, c], torch.bitwise_left_shift(col, i * 4)
                )
    elif to_pack.ndim == 0:
        packed = to_pack.to(torch.int32)
    else:
        new_c = to_pack.shape[0] // pack_num
        packed = torch.zeros(new_c, dtype=torch.int32, device=to_pack.device)
        for c in range(new_c):
            for i in range(pack_num):
                col = (to_pack[c * pack_num + order_map[i]].to(torch.int32)) & 0x0F
                packed[c] = torch.bitwise_or(
                    packed[c], torch.bitwise_left_shift(col, i * 4)
                )

    return packed.view(torch.uint32)
