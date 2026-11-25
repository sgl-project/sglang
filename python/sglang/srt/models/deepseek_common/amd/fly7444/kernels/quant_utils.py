# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared MX-format quantization IR helpers for FlyDSL kernels.

These functions emit MLIR/LLVM IR for the per-block E8M0 scale calculation
and f32 -> fp4 (e2m1) / fp6 (e2m3) conversions. They are *IR builders* -- you must
call them inside an active ``InsertionPoint`` (i.e. while the FlyDSL DSL
is mid-build of a kernel function), and they emit the same arith / LLVM
ops the kernels would otherwise emit inline.

The four E8M0 scale-rounding modes mirror PyTorch torchao's
``ScaleCalculationMode`` (``torchao/prototype/mx_formats/config.py``) 1:1
and are dtype-agnostic across the whole MX format family
(mxfp4 / mxfp6 / mxfp8 / mxint8) -- only the ``target_max_pow2`` /
``max_pos`` / ``mbits`` constants differ between dtypes. This matches both
PyTorch torchao's and the HIP-side ``MxScaleRoundMode`` design (see
``csrc/include/mx_quant_utils.h`` and ``aiter.ops.quant.MxScaleRoundMode``).

See ``aiter/utility/fp4_utils.py`` for cross-stack naming (PyTorch torchao /
NV Triton / DSv4 / FlashInfer / AMD Quark) and CPU torch reference
implementations.

This module imports only the *bare-int* mirrors
(:class:`MxScaleRoundModeInt` / :class:`MxDtypeInt`) from
:mod:`aiter.utility.mx_types`; the pybind11 :class:`MxScaleRoundMode` /
:class:`MxDtype` from the same module are loaded lazily and would
trigger a ``module_aiter_core`` JIT build on attribute access, which is
incompatible with wheel ``PREBUILD_KERNELS`` (FlyDSL AOT runs before the
HIP modules are built; see ``setup.py``). Callers may still pass either
view to :func:`emit_mx_e8m0_scale` -- both round-trip through
``int(...)``.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

from .tensor_shim import _to_raw as _raw
from aiter.utility.mx_types import (
    MX_DEFAULT_ROUND_MODE as _DEFAULT_MODE,
)

# Bare-int mirrors of MxScaleRoundMode / MxDtype (mx_quant_utils.h). Same
# numeric values as the pybind11 enum classes, but importing them is
# JIT-free, which is required at FlyDSL AOT time.
from aiter.utility.mx_types import (
    MxDtypeInt as _D,
)
from aiter.utility.mx_types import (
    MxScaleRoundModeInt as _M,
)

# Per-MX-dtype constants. Tuple form: (target_max_pow2, max_pos_inv_f32_bits, mbits)
# - target_max_pow2        = log2(largest pow2 <= max_normal(dtype))
# - max_pos_inv_f32_bits   = bit pattern of fp32(1.0 / max_normal(dtype)),
#                            used by RoundUp's ceil_pow2(amax / max_pos)
# - mbits                  = mantissa bits of the target dtype (EVEN only)
# Dict keyed by ``int(MxDtype.X)`` so callers may pass either bare int or
# pybind enum interchangeably.
_DTYPE_CFG = {
    _D.FP4_E2M1: (2, 0x3E2AAAAB, 1),  # max_pos=6.0,    1/6.0    ~= 0.16666667
    _D.FP8_E4M3: (
        8,
        0x3B124925,
        3,
    ),  # max_pos=448.0,  1/448.0  ~= 0.00223214 (OCP / H100 / gfx950+)
    _D.FP8_E4M3_FNUZ: (
        7,
        0x3B888889,
        3,
    ),  # max_pos=240.0,  1/240.0  ~= 0.00416667 (AMD gfx942)
}


def emit_mx_e8m0_scale(
    local_max,
    *,
    mode: int = _DEFAULT_MODE,
    dtype: int = _D.FP4_E2M1,
):
    """Emit IR computing the E8M0 block scale for an MX format.

    FlyDSL IR-builder analogue of PyTorch torchao ``to_mx(scaling_mode,
    elem_dtype)`` and the CPU torch ref
    :func:`aiter.utility.fp4_utils.f32_to_mx_e8m0_scale`. The four
    rounding formulas (FLOOR / RCEIL / CEIL / EVEN) are dtype-agnostic;
    ``dtype`` only selects ``target_max_pow2`` / ``max_pos`` / ``mbits``
    constants from :data:`_DTYPE_CFG`.

    See ``csrc/include/mx_quant_utils.h`` (``MxScaleRoundMode`` /
    ``MxDtype``) and :mod:`aiter.utility.mx_types` for the four formulas
    and cross-stack mapping (PyTorch torchao / NV / DSv4 / FlashInfer /
    AMD Quark naming).

    Args:
        local_max: f32 IR value, the (warp-reduced) ``max(|x|)`` of one
            block. Caller is responsible for the per-block reduction.
        mode: ``MxScaleRoundMode`` value -- accepts either the bare-int
            mirror :class:`aiter.utility.mx_types.MxScaleRoundModeInt`
            (recommended for FlyDSL kernel definitions) or the pybind11
            enum :class:`aiter.utility.mx_types.MxScaleRoundMode` (from
            user-facing code paths). Default ``RoundUp`` (industry
            consensus for MXFP4 and MXFP8).
        dtype: ``MxDtype`` value -- bare-int mirror or pybind11 enum.
            Default ``FP4_E2M1``.

    Returns:
        e8m0_biased: i32 IR value in the range ``[0, 0xFF]``. The caller
        derives ``quant_scale = (254 - e8m0_biased) << 23`` (bitcast to
        f32) for the multiplicative quant scale, and stores
        ``e8m0_biased`` as a ``uint8`` in the per-block scale tensor.
    """
    # Normalise int / pybind enum into a plain int -- pybind11 enum classes
    # don't auto-compare equal to ``int`` (unlike ``IntEnum``).
    local_max = fx.Float32(local_max)
    mode_int = int(mode)
    dtype_int = int(dtype)
    if dtype_int not in _DTYPE_CFG:
        raise ValueError(
            f"emit_mx_e8m0_scale: unsupported dtype {dtype!r}; "
            f"supported: {list(_DTYPE_CFG)}"
        )
    target_max_pow2, max_pos_inv_bits, mbits = _DTYPE_CFG[dtype_int]

    def _clamp_u8(x):
        # Defensive clamp into the E8M0 storage range [0, 0xFF]. Pathological
        # inputs (denormals, fp32 inf, mantissa bump from 0xFF -> 0x100) can
        # otherwise corrupt the stored uint8.
        return fx.min(fx.max(x, 0), 0xFF).ir_value()

    if mode_int == _M.RoundUp:
        # ceil_pow2(amax / max_pos): multiply by reciprocal of max_pos to get
        # the working value, then bump the exponent if any mantissa bit is
        # set. Bit-equivalent to HIP ``aiter::fp_f32_to_e8m0_scale<RoundUp,
        # FP4_E2M1>`` and to PyTorch torchao ``_to_mx_rceil`` (modulo
        # the GPU-vs-CPU fp32 ULP boundary effects documented in the PR).
        inv_max_pos_f32 = fx.Int32(max_pos_inv_bits).bitcast(fx.Float32)
        working = local_max * inv_max_pos_f32
        working_i32 = working.bitcast(fx.Int32)
        mantissa = working_i32 & 0x7FFFFF
        biased_exp = (working_i32 >> 23) & 0xFF
        mant_nonzero = mantissa != 0
        exp_field = mant_nonzero.select(biased_exp + 1, biased_exp)
        return _clamp_u8(exp_field)

    if mode_int == _M.RoundDown:
        # floor_pow2(amax) / 2^target_max_pow2: drop the f32 mantissa, then
        # subtract target_max_pow2 from the biased exponent.
        amax_i32 = local_max.bitcast(fx.Int32)
        biased_exp = (amax_i32 >> 23) & 0xFF
        return _clamp_u8(biased_exp - target_max_pow2)

    if mode_int == _M.Ceil:
        # ceil_pow2(amax) / 2^target_max_pow2: same as RoundDown but bump
        # the exponent if any mantissa bit is set.
        amax_i32 = local_max.bitcast(fx.Int32)
        mantissa = amax_i32 & 0x7FFFFF
        biased_exp = (amax_i32 >> 23) & 0xFF
        mant_nonzero = mantissa != 0
        biased_exp_bumped = mant_nonzero.select(biased_exp + 1, biased_exp)
        return _clamp_u8(biased_exp_bumped - target_max_pow2)

    if mode_int == _M.Even:
        # round_pow2_special(amax) / 2^target_max_pow2: add a half-step at
        # the "(mbits+1)-th-from-top" mantissa bit, then drop all mantissa
        # bits. ``val_to_add = 1 << (23 - mbits - 1)`` so that the carry
        # propagates exactly when amax >= 1.5 * 2^k (mbits=1, FP4) or
        # 1.0625 * 2^k (mbits=3, FP8 e4m3) etc. -- mantissa-precision-
        # aware ties-to-even on the power-of-2 lattice.
        val_to_add = 1 << (23 - mbits - 1)
        amax_i32 = local_max.bitcast(fx.Int32)
        amax_rounded = (amax_i32 + val_to_add) & 0xFF800000
        biased_exp = (amax_rounded >> 23) & 0xFF
        return _clamp_u8(biased_exp - target_max_pow2)

    raise ValueError(
        f"emit_mx_e8m0_scale: unknown mode int {mode_int} for {mode!r} "
        f"(expected one of MxScaleRoundModeInt: "
        f"RoundDown={_M.RoundDown}, RoundUp={_M.RoundUp}, "
        f"Even={_M.Even}, Ceil={_M.Ceil})"
    )


def emit_f32_to_e2m1(qx_f32):
    """Convert a scaled f32 value to FP4 (E2M1) as a 4-bit unsigned nibble.

    Matches:
    - CPU Python ref ``aiter.utility.fp4_utils.f32_to_mxfp4`` (round-to-
      nearest-even normal/denormal/saturate paths)
    - HIP gfx950 HW builtin ``v_cvt_pk_fp4_*`` (exact RNE)
    - HIP gfx942 SW fallback ``even_round_e2m1`` (algorithmically equivalent
      RHA; can differ from the CPU ref by <=1 ULP at FP4 round thresholds
      due to GPU vs CPU fp32 computation order)

    Args:
        qx_f32: f32 IR value, the already-scaled value ``act * quant_scale``.

    Returns:
        e2m1: i32 IR value with the 4-bit nibble in the low bits (sign at
        bit 3, magnitude at bits 0-2). Pack two nibbles into a byte for
        FP4x2 storage.
    """
    denorm_bias = fx.Int32(0x4A800000)

    qx = fx.Float32(qx_f32).bitcast(fx.Int32)
    s = qx & 0x80000000
    qx_abs = qx & 0x7FFFFFFF
    qx_abs_u = fx.Uint32(qx_abs)
    denormal_mask = qx_abs_u < 0x3F800000
    normal_mask = (qx_abs_u < 0x40C00000) & (qx_abs_u >= 0x3F800000)

    denorm_f32 = qx_abs.bitcast(fx.Float32) + denorm_bias.bitcast(fx.Float32)
    denormal_x = denorm_f32.bitcast(fx.Int32) - denorm_bias

    mant_odd = fx.Int32((qx_abs_u >> 22) & 1)
    normal_x = qx_abs + 0xC11FFFFF + mant_odd
    normal_x = fx.Uint32(normal_x) >> 22

    e2m1 = normal_mask.select(normal_x, 7)
    e2m1 = denormal_mask.select(denormal_x, e2m1)
    return ((fx.Uint32(s) >> 28) | fx.Uint32(e2m1)).ir_value()


def _emit_f32_to_fp6(qx_f32, *, mbits, bias, max_bits):
    shift = 23 - mbits
    cmin = fx.Uint32((128 - bias) << 23)
    cmax = fx.Uint32(max_bits)
    # Adding this power of two rounds subnormals directly onto their RNE grid.
    cden = fx.Uint32((151 - bias - mbits) << 23)
    cnorm = fx.Uint32((((bias - 127) << 23) + (1 << (shift - 1)) - 1) & 0xFFFFFFFF)
    bits = fx.Float32(qx_f32).bitcast(fx.Uint32)
    sign = (bits & fx.Uint32(0x80000000)) >> 26
    magnitude = bits & fx.Uint32(0x7FFFFFFF)
    denormal = magnitude < cmin
    normal = magnitude < cmax
    denorm_f32 = magnitude.bitcast(fx.Float32) + cden.bitcast(fx.Float32)
    denorm_code = denorm_f32.bitcast(fx.Uint32) - cden
    odd = (magnitude >> shift) & fx.Uint32(1)
    normal_code = (magnitude + cnorm + odd) >> shift
    code = normal.select(normal_code, fx.Uint32(31))
    return sign | denormal.select(denorm_code, code)


def emit_f32_to_e2m3(qx_f32):
    """Convert scaled f32 to OCP FP6 E2M3, RNE with finite saturation.

    Returns an i32 six-bit code (sign in bit 5), including signed zero.
    """
    return _emit_f32_to_fp6(qx_f32, mbits=3, bias=1, max_bits=0x40F00000)


# Typed FP8 encode/decode remain unlowered in FlyDSL 0.3.4.1.
def emit_f32_to_fp8(first, second):
    """Pack two f32 values into the low two bytes of an FP8 dword."""
    return fx.Int32(
        fx.rocdl.cvt_pk_fp8_f32(T.i32, _raw(first), _raw(second), _raw(fx.Int32(0)), 0)
    )


def emit_fp8_to_f32(word, high=False):
    """Decode the selected two FP8 bytes from a packed dword."""
    return fx.Vector(fx.rocdl.cvt_pk_f32_fp8(T.f32x2, _raw(word), high))


def emit_f32_to_e2m3_native(first, second, scale):
    """Convert two 16-value f32 vectors to six packed FP6 dwords on gfx950."""
    from flydsl._mlir.dialects import rocdl as rocdl_dialect

    return fx.Vector(
        rocdl_dialect.cvt_scalef32_2xpk16_fp6_f32(
            T.vec(6, T.i32), _raw(first), _raw(second), _raw(scale)
        )
    )


def emit_amax_e8m0_native_scale(
    all_vals, *, wave_size, dtype=_D.FP8_E4M3, amax_gain=None
):
    """Like :func:`emit_amax_e8m0_recip` but returns the *forward* scale
    ``2^(e8m0-127)`` needed by native gfx1250 ``v_cvt_scalef32_pk8_*``
    instructions (HW divides by the scale internally).

    ``amax_gain`` is an optional f32 the block amax is multiplied by before the
    scale is derived. It exists so a caller whose block is a uniform multiple of
    what it holds in registers -- an MoE tile row scaled by one route weight, say
    -- can pay a single multiply here instead of one per element, and fold the
    reciprocal into the ``v_cvt_scalef32`` scale operand. Applied AFTER the peer
    reduction: every lane in a block shares the gain, so the order is
    equivalent, and keeping it off the shuffle's input shortens that dependency
    chain. Must be non-negative; pass ``abs()`` of a signed factor. A gain above
    one can in principle push the amax past FLT_MAX; the e8m0 exponent clamp in
    :func:`emit_mx_e8m0_scale` absorbs that rather than this function paying a
    second clamp per block.

    Returns:
        ``(scale_f32, e8m0_byte)`` — f32 forward scale and i8 E8M0 byte.
    """
    if not all_vals:
        raise ValueError("emit_amax_e8m0_native_scale needs at least one value")
    c_flt_max = fx.Float32(3.4028234663852886e38)

    # A tree, not a linear accumulate: a chain of N maxes is N deep and every
    # step stalls on the last (one s_delay_alu each), while a tree is log-deep
    # with independent maxes per level for the scheduler to interleave.
    #
    # Ternary groups rather than pairs: the AMDGPU backend contracts
    # max(max(a, b), c) into one v_max3_f32, so a group of three costs the same
    # single instruction a pair does. That is 8 instructions for 16 values
    # instead of 15, and three levels instead of four. If the contraction ever
    # stops firing this degrades to the pairwise count, never worse.
    #
    # No 0.0 seed: abs() already floors the inputs at zero, so it only bought an
    # extra level.
    level = [abs(fx.Float32(v)) for v in all_vals]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 3):
            grp = level[i : i + 3]
            acc = grp[0]
            for v in grp[1:]:
                acc = fx.maxnumf(acc, v)
            nxt.append(acc)
        level = nxt
    block_amax = fx.minnumf(level[0], c_flt_max)
    peer = block_amax.shuffle_xor(16, wave_size)
    block_amax = fx.maxnumf(block_amax, peer)
    if amax_gain is not None:
        block_amax = block_amax * fx.Float32(amax_gain)

    e8m0 = emit_mx_e8m0_scale(block_amax, dtype=dtype)
    scale_f32 = (fx.Int32(e8m0) << 23).bitcast(fx.Float32).ir_value()
    e8m0_byte = fx.Uint8(e8m0).ir_value()
    return scale_f32, e8m0_byte


def emit_cvt_scalef32_pk8_fp8_f32(src_v8f32, scale_f32, *, v2i32_ty, rocdl):
    """Native gfx1250 ``v_cvt_scalef32_pk8_fp8_f32``: 8 f32 -> 8 fp8 e4m3.

    ``src_v8f32`` is a ``vector<8, f32>`` ir.Value, ``scale_f32`` an f32 whose
    exponent is the e8m0 block scale (2^(e8m0-127)); the HW divides each input
    by it and RNE-packs 8 fp8 e4m3 bytes into a ``vector<2, i32>``.
    No bf16 round-trip — full f32 precision into the scaled conversion.
    """
    return rocdl.cvt_scalef32_pk8_fp8_f32(
        v2i32_ty,
        _raw(src_v8f32),
        _raw(scale_f32),
    )


def emit_cvt_scalef32_pk8_fp4_f32(src_v8f32, scale_f32, *, i32_ty, rocdl):
    """Native gfx1250 ``v_cvt_scalef32_pk8_fp4_f32``: 8 f32 -> 8 fp4 e2m1.

    Same contract as the fp8 form, except the eight 4-bit values pack into one
    i32 instead of two. Also no bf16 round-trip, which the ``_bf16`` variant
    below would cost -- and fp4 has too little mantissa to spend on that.
    """
    return rocdl.cvt_scalef32_pk8_fp4_f32(
        i32_ty,
        _raw(src_v8f32),
        _raw(scale_f32),
    )


def emit_cvt_scalef32_pk8_fp4_bf16(src_v8bf16, scale_f32, *, i32_ty):
    """Native gfx1250 scaled conversion of eight bf16 values to packed FP4."""
    return rocdl.cvt_scalef32_pk8_fp4_bf16(
        i32_ty,
        _raw(src_v8bf16),
        _raw(scale_f32),
    )
