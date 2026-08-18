# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 B1 mixed-precision MoE tri-projection for gfx950.

The kernel preserves the accepted pre-route projection precision boundaries:

* routed-down and shared gate/up use row-scaled OCP FP8 E4M3 weights and
  materialize BF16 outputs;
* router logits use BF16 weights, the accepted DPP reduction order, a BF16
  rounding boundary, and an FP32 output.

All three projections consume one BF16 hidden row from LDS in one wide grid.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as arith_dialect
from flydsl._mlir.dialects import llvm, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
)
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.rocdl import cvt_pk_f32_fp8
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops, vector
from aiter.ops.flydsl.kernels.vector import ReductionOp

from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_rsrc,
)

_HIDDEN_SIZE = 7168
_ROUTED_SIZE = 3584
_SHARED_UP_SIZE = 1536
_FP8_OUTPUT_SIZE = _ROUTED_SIZE + _SHARED_UP_SIZE
_ROUTER_SIZE = 896
_TOTAL_OUTPUT_SIZE = _FP8_OUTPUT_SIZE + _ROUTER_SIZE
_WAVE_SIZE = 64
_ELEMENTS_PER_LOAD = 8


def _raw(value):
    return value.ir_value() if hasattr(value, "ir_value") else value


def build_kimi_k3_b1_tri_projection_fp8_module(
    num_tokens: int = 1,
    rows_per_wave: int = 2,
    cu_count: int = 248,
    waves_per_eu: int = 0,
    weight_cache_modifier: int = 0,
    hidden_to_lds: bool = True,
    use_fp8_fdot2: bool = True,
):
    """Build the fixed mixed-precision tri-projection launcher."""

    if num_tokens not in (1, 2):
        raise ValueError("num_tokens must be 1 or 2")
    if rows_per_wave not in (1, 2, 3, 4, 5, 6, 8):
        raise ValueError("rows_per_wave must be 1, 2, 3, 4, 5, 6, or 8")
    if not 1 <= cu_count <= 256:
        raise ValueError("cu_count must be between 1 and 256")
    if waves_per_eu < 0:
        raise ValueError("waves_per_eu must be non-negative")
    if weight_cache_modifier not in (0, 1, 2, 3):
        raise ValueError("weight_cache_modifier must be between 0 and 3")

    output_groups = (_TOTAL_OUTPUT_SIZE + rows_per_wave - 1) // rows_per_wave
    waves_per_block = min(16, (output_groups + cu_count - 1) // cu_count)
    block_threads = waves_per_block * _WAVE_SIZE
    groups_per_grid = cu_count * waves_per_block
    persistent_iterations = (output_groups + groups_per_grid - 1) // groups_per_grid
    hidden_load_iterations = (
        _HIDDEN_SIZE + block_threads * _ELEMENTS_PER_LOAD - 1
    ) // (block_threads * _ELEMENTS_PER_LOAD)

    @fx.struct
    class SharedStorage:
        hidden: fx.Array[fx.BFloat16, _HIDDEN_SIZE, 16]

    kernel_name = (
        f"kimi_k3_b{num_tokens}_tri_projection_bf16_fp8_gfx950"
        f"_rpw{rows_per_wave}_cu{cu_count}_wpb{waves_per_block}"
        f"_wpe{waves_per_eu}_wcm{weight_cache_modifier}"
        f"_hlds{int(hidden_to_lds)}"
        f"_fdot2{int(use_fp8_fdot2)}"
    )

    @flyc.kernel(
        name=kernel_name,
        known_block_size=[block_threads, 1, 1],
    )
    def tri_projection_fp8_kernel(
        hidden: fx.Pointer,
        routed_weight: fx.Pointer,
        routed_scale: fx.Pointer,
        shared_weight: fx.Pointer,
        shared_scale: fx.Pointer,
        router_weight: fx.Pointer,
        routed_output: fx.Pointer,
        shared_output: fx.Pointer,
        router_output: fx.Pointer,
    ):
        i32 = T.i32
        f32 = T.f32
        fm_fast = arith.FastMathFlags.fast
        tid = ArithValue(gpu.thread_idx.x)
        lane = tid % arith.constant(_WAVE_SIZE, type=i32)
        wave = tid // arith.constant(_WAVE_SIZE, type=i32)

        hidden_rsrc = ptr_rsrc(hidden)
        routed_weight_rsrc = ptr_rsrc(routed_weight)
        routed_scale_rsrc = ptr_rsrc(routed_scale)
        shared_weight_rsrc = ptr_rsrc(shared_weight)
        shared_scale_rsrc = ptr_rsrc(shared_scale)
        router_weight_rsrc = ptr_rsrc(router_weight)
        routed_output_rsrc = ptr_rsrc(routed_output)
        shared_output_rsrc = ptr_rsrc(shared_output)
        router_output_rsrc = ptr_rsrc(router_output)
        token = ArithValue(gpu.block_idx.y)
        hidden_token_base = token * arith.constant(_HIDDEN_SIZE, type=i32)
        routed_token_base = token * arith.constant(_ROUTED_SIZE, type=i32)
        shared_token_base = token * arith.constant(_SHARED_UP_SIZE, type=i32)
        router_token_base = token * arith.constant(_ROUTER_SIZE, type=i32)
        hidden_lds = fx.SharedAllocator().allocate(SharedStorage).peek().hidden.ptr

        vec2_f32 = T.vec(2, f32)
        vec2_bf16 = T.vec(2, T.bf16)
        vec8_bf16 = T.vec(_ELEMENTS_PER_LOAD, T.bf16)
        vec8_f32 = T.vec(_ELEMENTS_PER_LOAD, f32)
        zero_i32 = arith.constant(0, type=i32)
        zero_f32 = arith.constant(0.0, type=f32)

        def load_bf16x8(resource, element_index):
            dwords = buffer_ops.buffer_load(
                resource,
                element_index // arith.constant(2, type=i32),
                vec_width=4,
                dtype=i32,
            )
            return vector.bitcast(vec8_bf16, dwords)

        def load_fp8x8_as_f32(resource, element_index):
            packed = buffer_ops.buffer_load(
                resource,
                element_index // arith.constant(4, type=i32),
                vec_width=2,
                dtype=i32,
                cache_modifier=weight_cache_modifier,
            )
            packed = ArithValue(packed)
            packed0 = vector.extract(packed, static_position=[0], dynamic_position=[])
            packed1 = vector.extract(packed, static_position=[1], dynamic_position=[])
            weight0_lo = cvt_pk_f32_fp8(res=vec2_f32, src=packed0, word_sel=False)
            weight0_hi = cvt_pk_f32_fp8(res=vec2_f32, src=packed0, word_sel=True)
            weight1_lo = cvt_pk_f32_fp8(res=vec2_f32, src=packed1, word_sel=False)
            weight1_hi = cvt_pk_f32_fp8(res=vec2_f32, src=packed1, word_sel=True)
            weight_lo = weight0_lo.shuffle(weight0_hi, [0, 1, 2, 3])
            weight_hi = weight1_lo.shuffle(weight1_hi, [0, 1, 2, 3])
            return weight_lo.shuffle(weight_hi, [0, 1, 2, 3, 4, 5, 6, 7])

        def wave_reduce_add(value):
            reduced = _raw(value)
            for offset in (32, 16, 8, 4, 2, 1):
                peer = _raw(
                    ArithValue(reduced).shuffle_xor(
                        arith.constant(offset, type=i32),
                        arith.constant(_WAVE_SIZE, type=i32),
                    )
                )
                reduced = arith_dialect.AddFOp(
                    reduced,
                    peer,
                    fastmath=fm_fast,
                ).result
            return reduced

        def dot_bf16x8(left, right, accumulator):
            dot = _raw(accumulator)
            for pair_index in range_constexpr(_ELEMENTS_PER_LOAD // 2):
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
                        arith.constant(False, type=ir.IntegerType.get_signless(1)),
                    ],
                    [],
                    [],
                )
            return ArithValue(dot)

        if const_expr(hidden_to_lds):
            for load_iteration in range_constexpr(hidden_load_iterations):
                element_index = (
                    tid + arith.constant(load_iteration * block_threads, type=i32)
                ) * arith.constant(_ELEMENTS_PER_LOAD, type=i32)
                can_load = arith.cmpi(
                    CmpIPredicate.ult,
                    element_index,
                    arith.constant(_HIDDEN_SIZE, type=i32),
                )
                load_if = scf.IfOp(can_load)
                with ir.InsertionPoint(load_if.then_block):
                    hidden_vector = load_bf16x8(
                        hidden_rsrc, hidden_token_base + element_index
                    )
                    fx.ptr_store(hidden_vector, hidden_lds + element_index)
                    scf.YieldOp([])
            gpu.barrier()

        first_group = (
            ArithValue(gpu.block_idx.x) * arith.constant(waves_per_block, type=i32)
            + wave
        )
        for persistent_index in range_constexpr(persistent_iterations):
            group = first_group + arith.constant(
                persistent_index * groups_per_grid,
                type=i32,
            )
            row_base = group * arith.constant(rows_per_wave, type=i32)
            for row_offset in range_constexpr(rows_per_wave):
                row = row_base + arith.constant(row_offset, type=i32)
                row_in_range = arith.cmpi(
                    CmpIPredicate.ult,
                    row,
                    arith.constant(_TOTAL_OUTPUT_SIZE, type=i32),
                )
                row_if = scf.IfOp(row_in_range)
                with ir.InsertionPoint(row_if.then_block):
                    is_fp8 = arith.cmpi(
                        CmpIPredicate.ult,
                        row,
                        arith.constant(_FP8_OUTPUT_SIZE, type=i32),
                    )
                    precision_if = scf.IfOp(is_fp8, has_else=True)
                    with ir.InsertionPoint(precision_if.then_block):
                        is_routed = arith.cmpi(
                            CmpIPredicate.ult,
                            row,
                            arith.constant(_ROUTED_SIZE, type=i32),
                        )
                        shared_row = row - arith.constant(_ROUTED_SIZE, type=i32)
                        scale_if = scf.IfOp(
                            is_routed,
                            results_=[f32],
                            has_else=True,
                        )
                        with ir.InsertionPoint(scale_if.then_block):
                            scale = buffer_ops.buffer_load(
                                routed_scale_rsrc,
                                row,
                                vec_width=1,
                                dtype=f32,
                            )
                            scf.YieldOp([_raw(scale)])
                        with ir.InsertionPoint(scale_if.else_block):
                            scale = buffer_ops.buffer_load(
                                shared_scale_rsrc,
                                shared_row,
                                vec_width=1,
                                dtype=f32,
                            )
                            scf.YieldOp([_raw(scale)])

                        local_dot = ArithValue(zero_f32)
                        for k_iteration in range_constexpr(
                            _HIDDEN_SIZE // (_WAVE_SIZE * _ELEMENTS_PER_LOAD)
                        ):
                            k_element = (
                                lane
                                + arith.constant(k_iteration * _WAVE_SIZE, type=i32)
                            ) * arith.constant(_ELEMENTS_PER_LOAD, type=i32)
                            if const_expr(hidden_to_lds):
                                hidden_bf16 = fx.ptr_load(
                                    hidden_lds + k_element,
                                    result_type=vec8_bf16,
                                )
                            else:
                                hidden_bf16 = load_bf16x8(
                                    hidden_rsrc, hidden_token_base + k_element
                                )
                            weight_if = scf.IfOp(
                                is_routed,
                                results_=[vec8_f32],
                                has_else=True,
                            )
                            with ir.InsertionPoint(weight_if.then_block):
                                weight_element = (
                                    row * arith.constant(_HIDDEN_SIZE, type=i32)
                                    + k_element
                                )
                                weight_f32 = load_fp8x8_as_f32(
                                    routed_weight_rsrc,
                                    weight_element,
                                )
                                scf.YieldOp([_raw(weight_f32)])
                            with ir.InsertionPoint(weight_if.else_block):
                                weight_element = (
                                    shared_row * arith.constant(_HIDDEN_SIZE, type=i32)
                                    + k_element
                                )
                                weight_f32 = load_fp8x8_as_f32(
                                    shared_weight_rsrc,
                                    weight_element,
                                )
                                scf.YieldOp([_raw(weight_f32)])
                            if const_expr(use_fp8_fdot2):
                                weight_bf16 = arith.trunc_f(
                                    vec8_bf16,
                                    weight_if.results[0],
                                )
                                local_dot = dot_bf16x8(
                                    hidden_bf16,
                                    weight_bf16,
                                    local_dot,
                                )
                            else:
                                hidden_f32 = ArithValue(hidden_bf16).extf(vec8_f32)
                                local_dot = local_dot + (
                                    hidden_f32 * ArithValue(weight_if.results[0])
                                ).reduce(ReductionOp.ADD, fastmath=fm_fast)

                        reduced = ArithValue(wave_reduce_add(local_dot)) * ArithValue(
                            scale_if.results[0]
                        )
                        is_lane_zero = arith.cmpi(
                            CmpIPredicate.eq,
                            lane,
                            arith.constant(0, type=i32),
                        )
                        write_if = scf.IfOp(is_lane_zero)
                        with ir.InsertionPoint(write_if.then_block):
                            result = arith.trunc_f(T.bf16, _raw(reduced))
                            output_if = scf.IfOp(
                                is_routed,
                                results_=[],
                                has_else=True,
                            )
                            with ir.InsertionPoint(output_if.then_block):
                                buffer_ops.buffer_store(
                                    result,
                                    routed_output_rsrc,
                                    routed_token_base + row,
                                )
                                scf.YieldOp([])
                            with ir.InsertionPoint(output_if.else_block):
                                buffer_ops.buffer_store(
                                    result,
                                    shared_output_rsrc,
                                    shared_token_base + shared_row,
                                )
                                scf.YieldOp([])
                            scf.YieldOp([])
                        scf.YieldOp([])

                    with ir.InsertionPoint(precision_if.else_block):
                        router_row = row - arith.constant(
                            _FP8_OUTPUT_SIZE,
                            type=i32,
                        )
                        accumulator = ArithValue(zero_f32)
                        for k_base in range_constexpr(0, _HIDDEN_SIZE, _WAVE_SIZE):
                            k = lane + arith.constant(k_base, type=i32)
                            if const_expr(hidden_to_lds):
                                hidden_bf16 = fx.ptr_load(hidden_lds + k)
                            else:
                                hidden_bf16 = buffer_ops.buffer_load(
                                    hidden_rsrc,
                                    hidden_token_base + k,
                                    vec_width=1,
                                    dtype=T.bf16,
                                )
                            hidden_f32 = ArithValue(arith.extf(f32, _raw(hidden_bf16)))
                            weight_index = (
                                router_row * arith.constant(_HIDDEN_SIZE, type=i32) + k
                            )
                            weight_bf16 = buffer_ops.buffer_load(
                                router_weight_rsrc,
                                weight_index,
                                vec_width=1,
                                dtype=T.bf16,
                            )
                            weight_f32 = ArithValue(arith.extf(f32, weight_bf16))
                            accumulator = accumulator + hidden_f32 * weight_f32

                        for dpp_control in (0xB1, 0x4E, 0x141, 0x140, 0x142, 0x143):
                            remote_i32 = rocdl.update_dpp(
                                i32,
                                zero_i32,
                                arith.bitcast(i32, _raw(accumulator)),
                                dpp_control,
                                0xF,
                                0xF,
                                True,
                            )
                            remote = ArithValue(arith.bitcast(f32, remote_i32))
                            accumulator = accumulator + remote

                        is_last_lane = arith.cmpi(
                            CmpIPredicate.eq,
                            lane,
                            arith.constant(_WAVE_SIZE - 1, type=i32),
                        )
                        store_if = scf.IfOp(is_last_lane)
                        with ir.InsertionPoint(store_if.then_block):
                            rounded = arith.trunc_f(T.bf16, _raw(accumulator))
                            projected = arith.extf(f32, rounded)
                            buffer_ops.buffer_store(
                                projected,
                                router_output_rsrc,
                                router_token_base + router_row,
                            )
                            scf.YieldOp([])
                        scf.YieldOp([])
                    scf.YieldOp([])

    @flyc.jit
    def launch_tri_projection_fp8(
        hidden: fx.Pointer,
        routed_weight: fx.Pointer,
        routed_scale: fx.Pointer,
        shared_weight: fx.Pointer,
        shared_scale: fx.Pointer,
        router_weight: fx.Pointer,
        routed_output: fx.Pointer,
        shared_output: fx.Pointer,
        router_output: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        ctx = CompilationContext.get_current()
        if const_expr(waves_per_eu > 0):
            for operation in ctx.gpu_module_body.operations:
                if (
                    hasattr(operation, "attributes")
                    and operation.OPERATION_NAME == "gpu.func"
                ):
                    operation.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                        T.i32,
                        int(waves_per_eu),
                    )
        tri_projection_fp8_kernel(
            hidden,
            routed_weight,
            routed_scale,
            shared_weight,
            shared_scale,
            router_weight,
            routed_output,
            shared_output,
            router_output,
        ).launch(
            grid=(cu_count, num_tokens, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch_tri_projection_fp8.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_tri_projection_fp8
