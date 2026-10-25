# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/4868b542c9dfd166662eecc4bb8be3a36a3feaa2/vllm/cute_utils/cvt.py
from cutlass import Float32, Uint32, cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def fp32x2_to_bf16x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint32(out)


@dsl_user_op
def bf16x2_to_fp32x2(data, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    if isinstance(data, Uint32):
        out = llvm.inline_asm(
            llvm.StructType.get_literal([T.f32(), T.f32()]),
            [data.ir_value(loc=loc, ip=ip)],
            "shl.b32 $0, $2, 16;\n\tand.b32 $1, $2, 0xFFFF0000;",
            "=f,=f,r",
            has_side_effects=False,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
        return (
            Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip)),
        )

    elif isinstance(data, (cute.Tensor, cute.TensorSSA)):
        # NOTE: the output is always 1D
        size = cute.size(data.shape)
        out = cute.make_rmem_tensor(size * 2, Float32)
        for i in range(size):
            out[i * 2], out[i * 2 + 1] = bf16x2_to_fp32x2(data[i])
        return out

    else:
        raise ValueError(f"Unsupported type {type(data)}")
