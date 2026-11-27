# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM vllm/models/deepseek_v41/nvidia/specialized_decode/moe/routed_expert_front_cute.py @ddd43a54
"""PTX helper ops of the DeepSeek-V4.1 low-batch MoE front that the
MiniMax-M3 low-batch FFN kernels reuse: FP32 add and max, BF16 pair
maxima, and the MXFP8 block scale, inverse scale and E4M3 conversion.
The DeepSeek-V4.1 front kernel and its top-k code are not vendored."""

from cutlass import Float32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


def _asm(result, operands: list, text: str, constraints: str, loc, ip):
    """A side-effect-free PTX inline assembly operation."""
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _f32(value, loc, ip):
    """IR value of a Float32."""
    return Float32(value).ir_value(loc=loc, ip=ip)


def _u32(value, loc, ip):
    """IR value of a Uint32."""
    return Uint32(value).ir_value(loc=loc, ip=ip)


@dsl_user_op
def add_f32(a, b, *, loc=None, ip=None) -> Float32:
    """`a + b`, round to nearest, no contraction."""
    operands = [_f32(a, loc, ip), _f32(b, loc, ip)]
    return Float32(_asm(T.f32(), operands, "add.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def abs_max_bf16x2(a, b, *, loc=None, ip=None) -> Uint32:
    """Lane-wise `max(|a|, |b|)` of two BF16 pairs."""
    return Uint32(
        _asm(
            T.i32(),
            [_u32(a, loc, ip), _u32(b, loc, ip)],
            "{\n.reg .b32 x, y;\nand.b32 x, $1, 0x7FFF7FFF;\n"
            "and.b32 y, $2, 0x7FFF7FFF;\nmax.bf16x2 $0, x, y;\n}",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def max_bf16x2(a, b, *, loc=None, ip=None) -> Uint32:
    """Lane-wise max of two BF16 pairs."""
    operands = [_u32(a, loc, ip), _u32(b, loc, ip)]
    return Uint32(_asm(T.i32(), operands, "max.bf16x2 $0, $1, $2;", "=r,r,r", loc, ip))


@dsl_user_op
def block_scale(amax, *, loc=None, ip=None) -> Uint32:
    """UE8M0 of amax / 448, rounded up (0 for a zero block)."""
    return Uint32(
        _asm(
            T.i32(),
            [_f32(amax, loc, ip)],
            "{\n.reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;\n"
            ".reg .u32 bits, exp_biased, mantissa, bump, result;\n"
            ".reg .f32 value;\nmul.f32 value, $1, 0f3B124925;\n"
            "setp.le.f32 p_zero, value, 0f00000000;\n"
            "mov.b32 bits, value;\nshr.b32 exp_biased, bits, 23;\n"
            "and.b32 exp_biased, exp_biased, 255;\n"
            "and.b32 mantissa, bits, 0x7FFFFF;\n"
            "setp.ne.u32 p_has_mant, mantissa, 0;\nselp.u32 bump, 1, 0, p_has_mant;\n"
            "setp.eq.u32 p_exp_zero, exp_biased, 0;\n"
            "setp.le.u32 p_tiny_sub, mantissa, 0x400000;\n"
            "and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;\n"
            "@p_tiny_sub mov.u32 bump, 0;\nadd.u32 result, exp_biased, bump;\n"
            "setp.gt.u32 p_ovf, result, 254;\nselp.u32 result, 254, result, p_ovf;\n"
            "selp.u32 $0, 0, result, p_zero;\n}",
            "=r,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def bf16x2_max_to_f32(x, *, loc=None, ip=None) -> Float32:
    """The larger of a BF16 pair, as FP32."""
    return Float32(
        _asm(
            T.f32(),
            [_u32(x, loc, ip)],
            "{\n.reg .b32 lo, hi;\n.reg .f32 f0, f1;\nand.b32 lo, $1, 0xFFFF;\n"
            "shr.b32 hi, $1, 16;\nshl.b32 lo, lo, 16;\nshl.b32 hi, hi, 16;\n"
            "mov.b32 f0, lo;\nmov.b32 f1, hi;\nmax.f32 $0, f0, f1;\n}",
            "=f,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def max_f32(a, b, *, loc=None, ip=None) -> Float32:
    """`max(a, b)` (no flush)."""
    operands = [_f32(a, loc, ip), _f32(b, loc, ip)]
    return Float32(_asm(T.f32(), operands, "max.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def inverse_scale(code, *, loc=None, ip=None) -> Float32:
    """2^(127 - code), or 0 for code 0."""
    return Float32(
        _asm(
            T.f32(),
            [_u32(code, loc, ip)],
            "{\n.reg .s32 new_exp;\n.reg .b32 float_bits;\n.reg .pred p_zero;\n"
            "setp.eq.u32 p_zero, $1, 0;\nsub.s32 new_exp, 254, $1;\n"
            "max.s32 new_exp, new_exp, 0;\nshl.b32 float_bits, new_exp, 23;\n"
            "mov.b32 $0, float_bits;\n@p_zero mov.b32 $0, 0;\n}",
            "=f,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def bf16x4_to_e4m3x4(a, b, inverse, *, loc=None, ip=None) -> Uint32:
    """Two BF16 pairs times the inverse scale as four saturating E4M3 bytes
    (`a`'s pair first)."""
    convert = (
        "and.b32 lo, {0}, 0xFFFF;\nshr.b32 hi, {0}, 16;\nshl.b32 lo, lo, 16;\n"
        "shl.b32 hi, hi, 16;\nmov.b32 f0, lo;\nmov.b32 f1, hi;\n"
        "mul.f32 f0, f0, $3;\nmul.f32 f1, f1, $3;\n"
        "min.f32 f0, f0, 0f43E00000;\nmax.f32 f0, f0, 0fC3E00000;\n"
        "min.f32 f1, f1, 0f43E00000;\nmax.f32 f1, f1, 0fC3E00000;\n"
        "cvt.rn.satfinite.e4m3x2.f32 {1}, f1, f0;\n"
    )
    return Uint32(
        _asm(
            T.i32(),
            [_u32(a, loc, ip), _u32(b, loc, ip), _f32(inverse, loc, ip)],
            "{\n.reg .b32 lo, hi;\n.reg .f32 f0, f1;\n.reg .b16 p0, p1;\n"
            + convert.format("$1", "p0")
            + convert.format("$2", "p1")
            + "mov.b32 $0, {p0, p1};\n}",
            "=r,r,r,f",
            loc,
            ip,
        )
    )
