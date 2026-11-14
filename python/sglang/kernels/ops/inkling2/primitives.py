from cutlass import Float32
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm
from cutlass._mlir.extras import types as mlir_types
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def fma_rn_f32(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            mlir_types.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fmul_rn_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            mlir_types.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fadd_rn_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            mlir_types.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def min_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        mlir_arith.minimumf(
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )
