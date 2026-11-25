# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM vllm/models/common/ops/moe_tail_fusion/primitives.py @ddd43a54
# Adapted from FlashInfer's flashinfer/comm/mnnvl_cutedsl/cute_dsl_primitives.py
# and runtime.py (flashinfer-ai/flashinfer@139af6f6).

"""CuTe DSL primitives the low-batch FFN kernels share: a 128-bit global
load, an FP32 to BF16 bit conversion and the current stream. This module
defines no kernel."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op


def _asm(result, operands, text, constraints, *, side_effects, loc, ip):
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_global_u32x4(
    pointer: cute.Pointer,
    *,
    volatile: cutlass.Constexpr[bool] = False,
    loc=None,
    ip=None,
):
    """Load one 128-bit fragment as four u32 registers.

    The volatile form is the Lamport polling load. Marking the asm
    side-effecting prevents loop-invariant motion and common-subexpression
    elimination across polling iterations.
    """
    address = pointer.toint(loc=loc, ip=ip)
    opcode = "ld.volatile.global.v4.u32" if volatile else "ld.global.v4.u32"
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        f"{opcode} {{$0, $1, $2, $3}}, [$4];",
        "=r,=r,=r,=r,l",
        side_effects=volatile,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [
            llvm.extractvalue(T.i32(), loaded, [index], loc=loc, ip=ip)
            for index in range(4)
        ],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def f32_to_bf16_bits(value: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [value.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 bits;\n\tcvt.rn.bf16.f32 bits, $1;\n\t"
            "cvt.u32.u16 $0, bits;\n\t}",
            "=r,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


def current_cu_stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)
