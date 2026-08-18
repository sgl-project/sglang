# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fixed-shape gfx950 Kimi-K3 MLA gate projection and epilogue."""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.arith import ArithValue, CmpFPredicate, CmpIPredicate
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops, vector
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_rsrc,
)

_HIDDEN = 7168
_OUTPUT = 1536
_WAVE_SIZE = 64
_ELEMENTS_PER_LOAD = 8
_K_PER_WAVE_ITERATION = _WAVE_SIZE * _ELEMENTS_PER_LOAD
_K_ITERATIONS = _HIDDEN // _K_PER_WAVE_ITERATION
_LOG2E = math.log2(math.e)


def _raw(value):
    return value.ir_value() if hasattr(value, "ir_value") else value


def build_kimi_k3_mla_gate_module(
    rows_per_block: int = 4,
    waves_per_eu: int = 2,
    weight_cache_modifier: int = 0,
):
    """Build the fixed B1 BF16 projection with one wave owning each output row."""

    if rows_per_block not in (1, 2, 4, 8):
        raise ValueError("rows_per_block must be 1, 2, 4, or 8")
    if waves_per_eu not in (0, 1, 2, 3, 4):
        raise ValueError("waves_per_eu must be between 0 and 4")
    if weight_cache_modifier not in (0, 1, 2, 3):
        raise ValueError("weight_cache_modifier must be between 0 and 3")

    block_threads = rows_per_block * _WAVE_SIZE
    kernel_name = (
        f"kimi_k3_mla_gate_b1_bf16_gfx950_r{rows_per_block}"
        f"_wpe{waves_per_eu}_wcm{weight_cache_modifier}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[block_threads, 1, 1])
    def gate_kernel(
        hidden: fx.Pointer,
        weight: fx.Pointer,
        attention: fx.Pointer,
        output: fx.Pointer,
    ):
        i1 = ir.IntegerType.get_signless(1)
        i32 = T.i32
        f32 = T.f32
        vec2_bf16 = T.vec(2, T.bf16)
        vec8_bf16 = T.vec(_ELEMENTS_PER_LOAD, T.bf16)

        tid = ArithValue(gpu.thread_idx.x)
        lane = tid % fx.Int32(_WAVE_SIZE)
        wave = tid // fx.Int32(_WAVE_SIZE)
        row = ArithValue(gpu.block_idx.x) * fx.Int32(rows_per_block) + wave
        hidden_rsrc = ptr_rsrc(hidden)
        weight_rsrc = ptr_rsrc(weight)
        attention_rsrc = ptr_rsrc(attention)
        output_rsrc = ptr_rsrc(output)

        def load_bf16x8(resource, element_index, cache_modifier=0):
            packed = buffer_ops.buffer_load(
                resource,
                element_index // fx.Int32(2),
                vec_width=4,
                dtype=i32,
                cache_modifier=cache_modifier,
            )
            return vector.bitcast(vec8_bf16, packed)

        local_dot = fx.Float32(0.0)
        row_weight_base = row * fx.Int32(_HIDDEN)
        for k_iteration in range_constexpr(_K_ITERATIONS):
            k = lane * fx.Int32(_ELEMENTS_PER_LOAD) + fx.Int32(
                k_iteration * _K_PER_WAVE_ITERATION
            )
            hidden_values = load_bf16x8(hidden_rsrc, k)
            weight_values = load_bf16x8(
                weight_rsrc,
                row_weight_base + k,
                weight_cache_modifier,
            )
            for pair_index in range_constexpr(_ELEMENTS_PER_LOAD // 2):
                hidden_pair = vector.from_elements(
                    vec2_bf16,
                    [
                        vector.extract(
                            hidden_values,
                            static_position=[pair_index * 2],
                            dynamic_position=[],
                        ),
                        vector.extract(
                            hidden_values,
                            static_position=[pair_index * 2 + 1],
                            dynamic_position=[],
                        ),
                    ],
                )
                weight_pair = vector.from_elements(
                    vec2_bf16,
                    [
                        vector.extract(
                            weight_values,
                            static_position=[pair_index * 2],
                            dynamic_position=[],
                        ),
                        vector.extract(
                            weight_values,
                            static_position=[pair_index * 2 + 1],
                            dynamic_position=[],
                        ),
                    ],
                )
                local_dot = ArithValue(
                    llvm.call_intrinsic(
                        f32,
                        "llvm.amdgcn.fdot2.f32.bf16",
                        [
                            hidden_pair,
                            weight_pair,
                            _raw(local_dot),
                            arith.constant(False, type=i1),
                        ],
                        [],
                        [],
                    )
                )

        dot = local_dot
        for offset in (32, 16, 8, 4, 2, 1):
            peer = dot.shuffle_xor(fx.Int32(offset), fx.Int32(_WAVE_SIZE))
            dot = dot + peer

        lane_zero = arith.cmpi(CmpIPredicate.eq, lane, fx.Int32(0))
        write_if = scf.IfOp(lane_zero, results_=[], has_else=False)
        with ir.InsertionPoint(write_if.then_block):
            # Match production rounding: projection->bf16, sigmoid->bf16,
            # bf16 attention multiply in fp32-equivalent arithmetic, result->bf16.
            projected_bf16 = arith.trunc_f(T.bf16, _raw(dot))
            projected = ArithValue(arith.extf(f32, projected_bf16))
            is_negative = arith.cmpf(CmpFPredicate.OLT, projected, fx.Float32(0.0))
            magnitude = ArithValue(
                arith.select(is_negative, _raw(-projected), _raw(projected))
            )
            exp_neg_abs = fx.math.exp2(-magnitude * fx.Float32(_LOG2E))
            denominator = fx.Float32(1.0) + exp_neg_abs
            sigmoid = ArithValue(
                arith.select(
                    is_negative,
                    _raw(exp_neg_abs / denominator),
                    _raw(fx.Float32(1.0) / denominator),
                )
            )
            sigmoid_bf16 = arith.trunc_f(T.bf16, _raw(sigmoid))
            sigmoid_f32 = ArithValue(arith.extf(f32, sigmoid_bf16))
            attention_bf16 = buffer_ops.buffer_load(
                attention_rsrc,
                row,
                vec_width=1,
                dtype=T.bf16,
            )
            attention_f32 = ArithValue(arith.extf(f32, attention_bf16))
            result = arith.trunc_f(T.bf16, _raw(sigmoid_f32 * attention_f32))
            buffer_ops.buffer_store(result, output_rsrc, row)
            scf.YieldOp([])

    @flyc.jit
    def launch_gate(
        hidden: fx.Pointer,
        weight: fx.Pointer,
        attention: fx.Pointer,
        output: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        context = CompilationContext.get_current()
        if const_expr(waves_per_eu > 0):
            for operation in context.gpu_module_body.operations:
                if (
                    hasattr(operation, "attributes")
                    and operation.OPERATION_NAME == "gpu.func"
                ):
                    operation.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                        T.i32,
                        waves_per_eu,
                    )
        gate_kernel(hidden, weight, attention, output).launch(
            grid=(_OUTPUT // rows_per_block, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch_gate.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_gate
