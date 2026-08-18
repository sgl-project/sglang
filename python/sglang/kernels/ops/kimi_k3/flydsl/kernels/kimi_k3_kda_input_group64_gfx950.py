# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fixed Kimi-K3 TP8 B1 KDA input projection for gfx950.

The schedule is inherited from the measured row-E4M3 projection.  The only
numerical change is a FP32 scale per output row and 64 input columns.  FP8
values are converted to BF16 for ``fdot2``; each sixteen-element partial is
scaled in FP32 and accumulated in FP32.  The final store is BF16.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as arith_dialect
from flydsl._mlir.dialects import llvm, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.rocdl import cvt_pk_f32_fp8
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops, vector
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_rsrc,
)

_INPUT_FEATURES = 7168
_PADDED_OUTPUT_FEATURES = 6288
_LOGICAL_OUTPUT_FEATURES = 6284
_GROUP_SIZE = 64
_GROUPS_PER_ROW = _INPUT_FEATURES // _GROUP_SIZE
_WAVE_SIZE = 64
_ELEMENTS_PER_LOAD = 16
_K_PER_WAVE_ITERATION = _WAVE_SIZE * _ELEMENTS_PER_LOAD


def _raw(value):
    return value.ir_value() if hasattr(value, "ir_value") else value


def build_kimi_k3_kda_input_group64_module(
    *,
    num_tokens: int = 1,
    rows_per_wave: int = 2,
    cu_count: int = 256,
    waves_per_eu: int = 0,
    weight_cache_modifier: int = 2,
    hidden_to_lds: bool = True,
):
    """Build the fixed ``[M,7168] @ [6284,7168].T -> [M,6288]`` kernel."""

    if num_tokens not in (1, 2):
        raise ValueError("num_tokens must be 1 or 2")
    if rows_per_wave not in (1, 2, 3, 4):
        raise ValueError("rows_per_wave must be 1, 2, 3, or 4")
    if not 1 <= cu_count <= 256:
        raise ValueError("cu_count must be between 1 and 256")
    if waves_per_eu < 0:
        raise ValueError("waves_per_eu must be non-negative")
    if weight_cache_modifier not in (0, 1, 2, 3):
        raise ValueError("weight_cache_modifier must be between 0 and 3")

    output_groups = (_PADDED_OUTPUT_FEATURES + rows_per_wave - 1) // rows_per_wave
    waves_per_block = min(16, (output_groups + cu_count - 1) // cu_count)
    block_threads = waves_per_block * _WAVE_SIZE
    groups_per_grid = cu_count * waves_per_block
    persistent_iterations = (output_groups + groups_per_grid - 1) // groups_per_grid
    hidden_load_iterations = (
        _INPUT_FEATURES + block_threads * _ELEMENTS_PER_LOAD - 1
    ) // (block_threads * _ELEMENTS_PER_LOAD)

    @fx.struct
    class SharedStorage:
        hidden: fx.Array[fx.BFloat16, _INPUT_FEATURES, 16]

    kernel_name = (
        f"kimi_k3_kda_input_m{num_tokens}_n6288_stored6284_k7168"
        f"_e4m3g64_gfx950_rpw{rows_per_wave}_cu{cu_count}"
        f"_wpb{waves_per_block}_wpe{waves_per_eu}"
        f"_wcm{weight_cache_modifier}_hlds{int(hidden_to_lds)}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[block_threads, 1, 1])
    def projection_kernel(
        hidden: fx.Pointer,
        weight: fx.Pointer,
        weight_scale: fx.Pointer,
        output: fx.Pointer,
    ):
        i1 = ir.IntegerType.get_signless(1)
        i32 = T.i32
        f32 = T.f32
        fm_fast = arith.FastMathFlags.fast
        tid = ArithValue(gpu.thread_idx.x)
        lane = tid % arith.constant(_WAVE_SIZE, type=i32)
        wave = tid // arith.constant(_WAVE_SIZE, type=i32)
        hidden_rsrc = ptr_rsrc(hidden)
        weight_rsrc = ptr_rsrc(weight)
        scale_rsrc = ptr_rsrc(weight_scale)
        output_rsrc = ptr_rsrc(output)
        token = ArithValue(gpu.block_idx.y)
        hidden_token_base = token * arith.constant(_INPUT_FEATURES, type=i32)
        output_token_base = token * arith.constant(
            _PADDED_OUTPUT_FEATURES, type=i32
        )
        hidden_lds = fx.SharedAllocator().allocate(SharedStorage).peek().hidden.ptr
        vec2_f32 = T.vec(2, f32)
        vec2_bf16 = T.vec(2, T.bf16)
        vec16_bf16 = T.vec(_ELEMENTS_PER_LOAD, T.bf16)
        zero_f32 = arith.constant(0.0, type=f32)

        def load_bf16x16(resource, element_index):
            dwords_lo = ArithValue(
                buffer_ops.buffer_load(
                    resource,
                    element_index // arith.constant(2, type=i32),
                    vec_width=4,
                    dtype=i32,
                )
            )
            dwords_hi = ArithValue(
                buffer_ops.buffer_load(
                    resource,
                    (element_index + arith.constant(_ELEMENTS_PER_LOAD // 2, type=i32))
                    // arith.constant(2, type=i32),
                    vec_width=4,
                    dtype=i32,
                )
            )
            dwords = vector.shuffle(dwords_lo, dwords_hi, list(range(8)))
            return vector.bitcast(vec16_bf16, dwords)

        def load_fp8x16_as_f32(resource, element_index):
            packed = ArithValue(
                buffer_ops.buffer_load(
                    resource,
                    element_index // arith.constant(4, type=i32),
                    vec_width=4,
                    dtype=i32,
                    cache_modifier=weight_cache_modifier,
                )
            )
            converted = []
            for packed_index in range_constexpr(4):
                packed_dword = vector.extract(
                    packed,
                    static_position=[packed_index],
                    dynamic_position=[],
                )
                weight_lo = cvt_pk_f32_fp8(
                    res=vec2_f32,
                    src=packed_dword,
                    word_sel=False,
                )
                weight_hi = cvt_pk_f32_fp8(
                    res=vec2_f32,
                    src=packed_dword,
                    word_sel=True,
                )
                converted.append(vector.shuffle(weight_lo, weight_hi, [0, 1, 2, 3]))
            weight_lo = vector.shuffle(
                converted[0],
                converted[1],
                list(range(8)),
            )
            weight_hi = vector.shuffle(
                converted[2],
                converted[3],
                list(range(8)),
            )
            return vector.shuffle(weight_lo, weight_hi, list(range(16)))

        def wave_reduce_add(value):
            reduced = _raw(value)
            for offset in (32, 16, 8, 4, 2, 1):
                peer = _raw(
                    ArithValue(reduced).shuffle_xor(
                        arith.constant(offset, type=i32),
                        arith.constant(_WAVE_SIZE, type=i32),
                    )
                )
                reduced = arith_dialect.AddFOp(reduced, peer, fastmath=fm_fast).result
            return reduced

        if const_expr(hidden_to_lds):
            for load_iteration in range_constexpr(hidden_load_iterations):
                element_index = (
                    tid + arith.constant(load_iteration * block_threads, type=i32)
                ) * arith.constant(_ELEMENTS_PER_LOAD, type=i32)
                can_load = arith.cmpi(
                    CmpIPredicate.ult,
                    element_index,
                    arith.constant(_INPUT_FEATURES, type=i32),
                )
                load_if = scf.IfOp(can_load)
                with ir.InsertionPoint(load_if.then_block):
                    fx.ptr_store(
                        load_bf16x16(
                            hidden_rsrc, hidden_token_base + element_index
                        ),
                        hidden_lds + element_index,
                    )
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
                    arith.constant(_PADDED_OUTPUT_FEATURES, type=i32),
                )
                row_if = scf.IfOp(row_in_range)
                with ir.InsertionPoint(row_if.then_block):
                    row_has_weight = arith.cmpi(
                        CmpIPredicate.ult,
                        row,
                        arith.constant(_LOGICAL_OUTPUT_FEATURES, type=i32),
                    )
                    weighted_if = scf.IfOp(row_has_weight, results_=[], has_else=True)
                    with ir.InsertionPoint(weighted_if.then_block):
                        local_dot = ArithValue(zero_f32)
                        for k_iteration in range_constexpr(
                            _INPUT_FEATURES // _K_PER_WAVE_ITERATION
                        ):
                            k_element = (
                                lane
                                + arith.constant(k_iteration * _WAVE_SIZE, type=i32)
                            ) * arith.constant(_ELEMENTS_PER_LOAD, type=i32)
                            if const_expr(hidden_to_lds):
                                hidden_bf16 = fx.ptr_load(
                                    hidden_lds + k_element,
                                    result_type=vec16_bf16,
                                )
                            else:
                                hidden_bf16 = load_bf16x16(
                                    hidden_rsrc, hidden_token_base + k_element
                                )
                            weight_element = (
                                row * arith.constant(_INPUT_FEATURES, type=i32)
                                + k_element
                            )
                            weight_f32 = load_fp8x16_as_f32(weight_rsrc, weight_element)
                            weight_bf16 = arith.trunc_f(vec16_bf16, _raw(weight_f32))
                            chunk_dot = ArithValue(zero_f32)
                            for pair_index in range_constexpr(_ELEMENTS_PER_LOAD // 2):
                                hidden_pair = vector.from_elements(
                                    vec2_bf16,
                                    [
                                        vector.extract(
                                            hidden_bf16,
                                            static_position=[pair_index * 2],
                                            dynamic_position=[],
                                        ),
                                        vector.extract(
                                            hidden_bf16,
                                            static_position=[pair_index * 2 + 1],
                                            dynamic_position=[],
                                        ),
                                    ],
                                )
                                weight_pair = vector.from_elements(
                                    vec2_bf16,
                                    [
                                        vector.extract(
                                            weight_bf16,
                                            static_position=[pair_index * 2],
                                            dynamic_position=[],
                                        ),
                                        vector.extract(
                                            weight_bf16,
                                            static_position=[pair_index * 2 + 1],
                                            dynamic_position=[],
                                        ),
                                    ],
                                )
                                chunk_dot = ArithValue(
                                    llvm.call_intrinsic(
                                        f32,
                                        "llvm.amdgcn.fdot2.f32.bf16",
                                        [
                                            hidden_pair,
                                            weight_pair,
                                            _raw(chunk_dot),
                                            arith.constant(False, type=i1),
                                        ],
                                        [],
                                        [],
                                    )
                                )
                            scale_index = row * arith.constant(
                                _GROUPS_PER_ROW, type=i32
                            ) + k_element // arith.constant(_GROUP_SIZE, type=i32)
                            scale = ArithValue(
                                buffer_ops.buffer_load(
                                    scale_rsrc,
                                    scale_index,
                                    vec_width=1,
                                    dtype=f32,
                                )
                            )
                            local_dot = local_dot + chunk_dot * scale

                        reduced = wave_reduce_add(local_dot)
                        is_lane_zero = arith.cmpi(
                            CmpIPredicate.eq,
                            lane,
                            arith.constant(0, type=i32),
                        )
                        write_if = scf.IfOp(is_lane_zero)
                        with ir.InsertionPoint(write_if.then_block):
                            result = arith.trunc_f(T.bf16, reduced)
                            buffer_ops.buffer_store(
                                result, output_rsrc, output_token_base + row
                            )
                            scf.YieldOp([])
                        scf.YieldOp([])
                    with ir.InsertionPoint(weighted_if.else_block):
                        is_lane_zero = arith.cmpi(
                            CmpIPredicate.eq,
                            lane,
                            arith.constant(0, type=i32),
                        )
                        zero_if = scf.IfOp(is_lane_zero)
                        with ir.InsertionPoint(zero_if.then_block):
                            buffer_ops.buffer_store(
                                arith.trunc_f(T.bf16, zero_f32),
                                output_rsrc,
                                output_token_base + row,
                            )
                            scf.YieldOp([])
                        scf.YieldOp([])
                    scf.YieldOp([])

    @flyc.jit
    def launch(
        hidden: fx.Pointer,
        weight: fx.Pointer,
        weight_scale: fx.Pointer,
        output: fx.Pointer,
        stream: fx.Stream,
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
        projection_kernel(hidden, weight, weight_scale, output).launch(
            grid=(cu_count, num_tokens, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": (AITER_FLYDSL_KERNARG_PRELOAD_COUNT),
        },
    }
    return launch
