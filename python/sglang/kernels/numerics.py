# SPDX-License-Identifier: Apache-2.0
"""Numerical primitives shared by bit-exact Triton kernels."""

import triton  # type: ignore
import triton.language as tl  # type: ignore

_FLT_MIN = tl.constexpr(1.1754943508222875e-38)


@triton.jit
def round_bf16_to_fp32(value):
    """RNE-round fp32 to bf16 precision while keeping an fp32 register."""
    bits = value.to(tl.int32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & -65536
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit
def mul_rn_f32(x, y):
    """Opaque correctly-rounded fp32 multiply that blocks FMA contraction."""
    return tl.inline_asm_elementwise(
        asm="mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def div_rn_f32(x, y):
    """IEEE correctly-rounded fp32 division."""
    return tl.inline_asm_elementwise(
        asm="div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def rsqrt_approx_f32(x):
    return tl.inline_asm_elementwise(
        asm="rsqrt.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def cuda_rsqrtf(x):
    """Match CUDA ``rsqrtf``, including its subnormal rescaling path."""
    is_subnormal = tl.abs(x) < _FLT_MIN
    scaled = tl.where(is_subnormal, x * 16777216.0, x)
    result = rsqrt_approx_f32(scaled)
    return tl.where(is_subnormal, result * 4096.0, result)


@triton.jit
def silu_approx_f32(x):
    """Match the CUDA fast-math SiLU ``__fdividef(x, 1 + __expf(-x))``."""
    # __expf(-x) == ex2.approx(x * -log2(e)); folding the negation into
    # the constant (0fBFB8AA3B == -log2(e)) flips the sign exactly.
    exp_neg = tl.inline_asm_elementwise(
        asm="{ mul.ftz.f32 $0, $1, 0fBFB8AA3B; ex2.approx.ftz.f32 $0, $0; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return tl.inline_asm_elementwise(
        asm="div.approx.ftz.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[x, 1.0 + exp_neg],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _asm_f32(asm, constraints, args):
    return tl.inline_asm_elementwise(
        asm=asm,
        constraints=constraints,
        args=args,
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def cuda_expf(x):
    """Match CUDA ``expf`` (libdevice ``__nv_expf``) bit for bit.

    ``tl.exp`` lowers fp32 to ``ex2.approx(x * log2(e))``; aten's CUDA kernels
    call ``expf``, which splits ``x * log2(e)`` into an integer ``n`` in
    ``[-126, 126]`` and a remainder ``z`` with two fused multiply-adds and
    returns ``ex2.approx(z) * 2**n``.
    """
    # n + 126 = floor(252 * saturate(x * log2(e) / 252 + 0.5)), kept in the low
    # mantissa bits of r = 1.5 * 2**23 + 1 + (n + 126).
    t = _asm_f32("fma.rn.f32 $0, $1, 0f3BBB989D, 0f3F000000;", "=f,f", [x])
    t = _asm_f32("cvt.sat.f32.f32 $0, $1;", "=f,f", [t])
    r = _asm_f32("fma.rm.f32 $0, $1, 0f437C0000, 0f4B400001;", "=f,f", [t])
    neg_n = -(r - 12583039.0)
    # z = x * log2(e) - n, with log2(e) split into a high and a low word.
    z = _asm_f32("fma.rn.f32 $0, $1, 0f3FB8AA3B, $2;", "=f,f,f", [x, neg_n])
    z = _asm_f32("fma.rn.f32 $0, $1, 0f32A57060, $2;", "=f,f,f", [x, z])
    # r's low mantissa bits shifted into the exponent field are 2**n.
    scale = (r.to(tl.int32, bitcast=True) << 23).to(tl.float32, bitcast=True)
    return _asm_f32("ex2.approx.ftz.f32 $0, $1;", "=f,f", [z]) * scale


@triton.jit
def silu_f32(x):
    """Match aten's CUDA SiLU ``x / (1 + expf(-x))`` on fp32 bit for bit."""
    return tl.div_rn(x, 1.0 + cuda_expf(-x))


__all__ = [
    "cuda_expf",
    "cuda_rsqrtf",
    "div_rn_f32",
    "mul_rn_f32",
    "round_bf16_to_fp32",
    "rsqrt_approx_f32",
    "silu_approx_f32",
    "silu_f32",
]
