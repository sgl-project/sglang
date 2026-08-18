# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Persistent-wave Kimi-K3 B1 RMSNorm, FP8 GEMV, and shared-add kernel."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr

from aiter.ops.flydsl.kernels import buffer_ops, vector
from flydsl.expr import math as fmath
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.rocdl import cvt_pk_f32_fp8
from flydsl.expr.typing import T
from aiter.ops.flydsl.kernels.vector import ReductionOp

from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_rsrc,
)

_LATENT_DIM = 3584
_HIDDEN_DIM = 7168
_WAVE_SIZE = 64
_ELEMENTS_PER_LOAD = 8
_NORMALIZE_THREADS = _LATENT_DIM // _ELEMENTS_PER_LOAD
_NORMALIZE_WAVES = _NORMALIZE_THREADS // _WAVE_SIZE
_K_PER_WAVE_ITERATION = _WAVE_SIZE * _ELEMENTS_PER_LOAD


def _raw(value):
    return value.ir_value() if hasattr(value, "ir_value") else value


def build_b1_latent_moe_tail_fp8_persistent_module(
    rows_per_wave: int = 2,
    cu_count: int = 256,
    waves_per_eu: int = 0,
    weight_cache_modifier: int = 2,
):
    """Build a block-normalize/one-wave-per-output-group specialization."""

    if rows_per_wave not in (1, 2, 3, 4):
        raise ValueError("rows_per_wave must be 1, 2, 3, or 4")
    if not 1 <= cu_count <= 256:
        raise ValueError("cu_count must be between 1 and 256")
    if waves_per_eu < 0:
        raise ValueError("waves_per_eu must be non-negative")
    if weight_cache_modifier not in (0, 1, 2, 3):
        raise ValueError("weight_cache_modifier must be between 0 and 3")

    output_groups = (_HIDDEN_DIM + rows_per_wave - 1) // rows_per_wave
    waves_per_block = min(16, (output_groups + cu_count - 1) // cu_count)
    block_threads = waves_per_block * _WAVE_SIZE
    if block_threads < _NORMALIZE_THREADS:
        raise ValueError("schedule needs at least 448 threads for fixed RMS order")
    groups_per_grid = cu_count * waves_per_block
    persistent_iterations = (output_groups + groups_per_grid - 1) // groups_per_grid

    @fx.struct
    class SharedStorage:
        hidden: fx.Array[fx.BFloat16, _LATENT_DIM, 16]
        rms_sums: fx.Array[fx.Float32, _NORMALIZE_WAVES, 16]
        inverse_rms: fx.Array[fx.Float32, 1, 16]

    kernel_name = (
        f"latent_moe_tail_b1_bf16_fp8_persistent_gfx950"
        f"_rpw{rows_per_wave}_cu{cu_count}_wpb{waves_per_block}"
        f"_wpe{waves_per_eu}_wcm{weight_cache_modifier}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[block_threads, 1, 1])
    def tail_kernel(
        routed: fx.Pointer,
        shared: fx.Pointer,
        rms_weight: fx.Pointer,
        up_weight: fx.Pointer,
        up_scale: fx.Pointer,
        output: fx.Pointer,
        epsilon: fx.Float32,
    ):
        i1 = ir.IntegerType.get_signless(1)
        i32 = T.i32
        f32 = T.f32
        fm_fast = arith.FastMathFlags.fast
        tid = ArithValue(gpu.thread_idx.x)
        lane = tid % arith.constant(_WAVE_SIZE, type=i32)
        wave = tid // arith.constant(_WAVE_SIZE, type=i32)

        routed_rsrc = ptr_rsrc(routed)
        shared_rsrc = ptr_rsrc(shared)
        rms_weight_rsrc = ptr_rsrc(rms_weight)
        up_weight_rsrc = ptr_rsrc(up_weight)
        up_scale_rsrc = ptr_rsrc(up_scale)
        output_rsrc = ptr_rsrc(output)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        hidden_lds = lds.hidden.ptr
        rms_sums = lds.rms_sums.ptr
        inverse_rms = lds.inverse_rms.ptr

        vec2_f32 = T.vec(2, f32)
        vec2_bf16 = T.vec(2, T.bf16)
        vec8_bf16 = T.vec(_ELEMENTS_PER_LOAD, T.bf16)
        vec8_f32 = T.vec(_ELEMENTS_PER_LOAD, f32)
        zero_f32 = arith.constant(0.0, type=f32)

        def lds_load(ptr, index):
            return fx.ptr_load(ptr + fx.Int64(index))

        def lds_store(ptr, value, index):
            fx.ptr_store(value, ptr + fx.Int64(index))

        def load_bf16x8(resource, element_index):
            dwords = buffer_ops.buffer_load(
                resource,
                element_index // arith.constant(2, type=i32),
                vec_width=4,
                dtype=i32,
            )
            return vector.bitcast(vec8_bf16, dwords)

        def load_fp8x8_as_bf16(resource, element_index):
            packed = ArithValue(
                buffer_ops.buffer_load(
                    resource,
                    element_index // arith.constant(4, type=i32),
                    vec_width=2,
                    dtype=i32,
                    cache_modifier=weight_cache_modifier,
                )
            )
            packed0 = vector.extract(packed, static_position=[0], dynamic_position=[])
            packed1 = vector.extract(packed, static_position=[1], dynamic_position=[])
            weight0_lo = cvt_pk_f32_fp8(res=vec2_f32, src=packed0, word_sel=False)
            weight0_hi = cvt_pk_f32_fp8(res=vec2_f32, src=packed0, word_sel=True)
            weight1_lo = cvt_pk_f32_fp8(res=vec2_f32, src=packed1, word_sel=False)
            weight1_hi = cvt_pk_f32_fp8(res=vec2_f32, src=packed1, word_sel=True)
            weight_lo = weight0_lo.shuffle(weight0_hi, [0, 1, 2, 3])
            weight_hi = weight1_lo.shuffle(weight1_hi, [0, 1, 2, 3])
            weight_f32 = weight_lo.shuffle(weight_hi, [0, 1, 2, 3, 4, 5, 6, 7])
            return arith.trunc_f(vec8_bf16, _raw(weight_f32))

        def wave_reduce_add(value):
            reduced = _raw(value)
            for offset in (32, 16, 8, 4, 2, 1):
                peer = _raw(
                    ArithValue(reduced).shuffle_xor(
                        arith.constant(offset, type=i32),
                        arith.constant(_WAVE_SIZE, type=i32),
                    )
                )
                reduced = arith.AddFOp(reduced, peer, fastmath=fm_fast).result
            return reduced

        def dot_bf16x8(left, right, accumulator):
            dot = _raw(accumulator)
            for pair_index in range_constexpr(4):
                left_pair = vector.from_elements(
                    vec2_bf16,
                    [
                        vector.extract(
                            left,
                            static_position=[pair_index * 2],
                            dynamic_position=[],
                        ),
                        vector.extract(
                            left,
                            static_position=[pair_index * 2 + 1],
                            dynamic_position=[],
                        ),
                    ],
                )
                right_pair = vector.from_elements(
                    vec2_bf16,
                    [
                        vector.extract(
                            right,
                            static_position=[pair_index * 2],
                            dynamic_position=[],
                        ),
                        vector.extract(
                            right,
                            static_position=[pair_index * 2 + 1],
                            dynamic_position=[],
                        ),
                    ],
                )
                dot = llvm.call_intrinsic(
                    f32,
                    "llvm.amdgcn.fdot2.f32.bf16",
                    [
                        left_pair,
                        right_pair,
                        dot,
                        arith.constant(False, type=i1),
                    ],
                    [],
                    [],
                )
            return ArithValue(dot)

        normalize_thread = arith.cmpi(
            CmpIPredicate.ult,
            tid,
            arith.constant(_NORMALIZE_THREADS, type=i32),
        )
        normalize_if = scf.IfOp(normalize_thread, results_=[f32], has_else=True)
        with ir.InsertionPoint(normalize_if.then_block):
            element_index = tid * arith.constant(_ELEMENTS_PER_LOAD, type=i32)
            routed_bf16 = load_bf16x8(routed_rsrc, element_index)
            routed_f32 = ArithValue(routed_bf16).extf(vec8_f32)
            local_square_sum = (routed_f32 * routed_f32).reduce(
                ReductionOp.ADD, fastmath=fm_fast
            )
            scf.YieldOp([_raw(local_square_sum)])
        with ir.InsertionPoint(normalize_if.else_block):
            scf.YieldOp([zero_f32])
        wave_square_sum = wave_reduce_add(normalize_if.results[0])

        is_normalize_lane_zero = arith.andi(
            arith.cmpi(
                CmpIPredicate.eq,
                lane,
                arith.constant(0, type=i32),
            ),
            arith.cmpi(
                CmpIPredicate.ult,
                wave,
                arith.constant(_NORMALIZE_WAVES, type=i32),
            ),
        )
        lane_zero_if = scf.IfOp(is_normalize_lane_zero)
        with ir.InsertionPoint(lane_zero_if.then_block):
            lds_store(rms_sums, wave_square_sum, wave)
            scf.YieldOp([])
        gpu.barrier()

        is_thread_zero = arith.cmpi(CmpIPredicate.eq, tid, arith.constant(0, type=i32))
        thread_zero_if = scf.IfOp(is_thread_zero)
        with ir.InsertionPoint(thread_zero_if.then_block):
            total_square_sum = ArithValue(zero_f32)
            for wave_index in range_constexpr(_NORMALIZE_WAVES):
                total_square_sum = total_square_sum + lds_load(
                    rms_sums, arith.constant(wave_index, type=i32)
                )
            variance = total_square_sum * ArithValue(
                arith.constant(1.0 / _LATENT_DIM, type=f32)
            )
            inverse = fmath.rsqrt(variance + ArithValue(epsilon), fastmath=fm_fast)
            lds_store(inverse_rms, _raw(inverse), arith.constant(0, type=i32))
            scf.YieldOp([])
        gpu.barrier()

        normalize_store_if = scf.IfOp(normalize_thread)
        with ir.InsertionPoint(normalize_store_if.then_block):
            element_index = tid * arith.constant(_ELEMENTS_PER_LOAD, type=i32)
            routed_bf16 = load_bf16x8(routed_rsrc, element_index)
            gamma_bf16 = load_bf16x8(rms_weight_rsrc, element_index)
            routed_f32 = ArithValue(routed_bf16).extf(vec8_f32)
            gamma_f32 = ArithValue(gamma_bf16).extf(vec8_f32)
            inverse = ArithValue(lds_load(inverse_rms, arith.constant(0, type=i32)))
            normalized = (routed_f32 * gamma_f32 * inverse).truncf(vec8_bf16)
            fx.ptr_store(normalized, hidden_lds + element_index)
            scf.YieldOp([])
        gpu.barrier()

        first_group = (
            ArithValue(gpu.block_idx.x) * arith.constant(waves_per_block, type=i32)
            + wave
        )
        for persistent_index in range_constexpr(persistent_iterations):
            group = first_group + arith.constant(
                persistent_index * groups_per_grid, type=i32
            )
            row_base = group * arith.constant(rows_per_wave, type=i32)
            for row_offset in range_constexpr(rows_per_wave):
                row = row_base + arith.constant(row_offset, type=i32)
                row_in_range = arith.cmpi(
                    CmpIPredicate.ult,
                    row,
                    arith.constant(_HIDDEN_DIM, type=i32),
                )
                row_if = scf.IfOp(row_in_range)
                with ir.InsertionPoint(row_if.then_block):
                    local_dot = ArithValue(zero_f32)
                    for k_iteration in range_constexpr(
                        _LATENT_DIM // _K_PER_WAVE_ITERATION
                    ):
                        k_element = (
                            lane + arith.constant(k_iteration * _WAVE_SIZE, type=i32)
                        ) * arith.constant(_ELEMENTS_PER_LOAD, type=i32)
                        hidden_bf16 = fx.ptr_load(
                            hidden_lds + k_element,
                            result_type=vec8_bf16,
                        )
                        weight_element = (
                            row * arith.constant(_LATENT_DIM, type=i32) + k_element
                        )
                        weight_bf16 = load_fp8x8_as_bf16(up_weight_rsrc, weight_element)
                        local_dot = dot_bf16x8(hidden_bf16, weight_bf16, local_dot)

                    scale = buffer_ops.buffer_load(
                        up_scale_rsrc, row, vec_width=1, dtype=f32
                    )
                    reduced = ArithValue(wave_reduce_add(local_dot)) * ArithValue(scale)
                    is_lane_zero = arith.cmpi(
                        CmpIPredicate.eq,
                        lane,
                        arith.constant(0, type=i32),
                    )
                    write_if = scf.IfOp(is_lane_zero)
                    with ir.InsertionPoint(write_if.then_block):
                        # Match the BF16 Linear output boundary before the add.
                        projected_bf16 = arith.trunc_f(T.bf16, _raw(reduced))
                        projected_f32 = ArithValue(arith.extf(f32, projected_bf16))
                        shared_bf16 = buffer_ops.buffer_load(
                            shared_rsrc, row, vec_width=1, dtype=T.bf16
                        )
                        shared_f32 = ArithValue(arith.extf(f32, shared_bf16))
                        result = arith.trunc_f(T.bf16, _raw(projected_f32 + shared_f32))
                        buffer_ops.buffer_store(result, output_rsrc, row)
                        scf.YieldOp([])
                    scf.YieldOp([])

    @flyc.jit
    def launch_tail(
        routed: fx.Pointer,
        shared: fx.Pointer,
        rms_weight: fx.Pointer,
        up_weight: fx.Pointer,
        up_scale: fx.Pointer,
        output: fx.Pointer,
        epsilon: fx.Float32,
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
                        T.i32, int(waves_per_eu)
                    )
        tail_kernel(
            routed,
            shared,
            rms_weight,
            up_weight,
            up_scale,
            output,
            epsilon,
        ).launch(
            grid=(cu_count, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch_tail.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_tail
