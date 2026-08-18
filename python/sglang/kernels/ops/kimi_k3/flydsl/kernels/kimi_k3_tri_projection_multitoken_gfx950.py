# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Weight-reusing cooperative preactivated Kimi-K3 tri projection."""

import math
import os
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as arith_dialect
from flydsl._mlir.dialects import llvm, scf
from flydsl.compiler.extern_link import ExternFunction
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.rocdl import cvt_pk_f32_fp8
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops, vector
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_rsrc,
)

_HIDDEN = 7168
_ROUTED = 3584
_SHARED = 1536
_FP8_OUT = _ROUTED + _SHARED
_ROUTER = 896
_TOTAL = _FP8_OUT + _ROUTER
_WAVE = 64
_VEC = 8
_LOG2E = 1.4426950408889634


def _find_ocml_bitcode() -> str:
    for root in (
        os.environ.get("ROCM_PATH"),
        os.environ.get("ROCM_HOME"),
        "/opt/rocm",
    ):
        if not root:
            continue
        for pattern in (
            "amdgcn/bitcode/ocml.bc",
            "lib/llvm/lib/clang/*/lib/amdgcn/bitcode/ocml.bc",
        ):
            matches = sorted(Path(root).glob(pattern), reverse=True)
            if matches:
                return str(matches[0])
    raise RuntimeError("unable to locate ROCm OCML bitcode")


def _raw(value):
    return value.ir_value() if hasattr(value, "ir_value") else value


def build_kimi_k3_multitoken_tri_projection_module(
    *,
    num_tokens: int,
    token_tile: int = 8,
    cu_count: int = 248,
    waves_per_block: int = 4,
    waves_per_eu: int = 0,
    weight_cache_modifier: int = 2,
    interleaved_shared_pairs: bool = False,
    fast_situ: bool = False,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
):
    cooperative_preactivate_shared = True
    if num_tokens not in (2, 4):
        raise ValueError("num_tokens must be 2 or 4")
    if token_tile != num_tokens:
        raise ValueError("token_tile must match num_tokens")
    if waves_per_block not in (4, 8):
        raise ValueError("waves_per_block must be 4 or 8")
    if (
        not math.isfinite(situ_beta)
        or not math.isfinite(situ_linear_beta)
        or situ_beta <= 0.0
        or situ_linear_beta <= 0.0
    ):
        raise ValueError("SiTU beta values must be finite and positive")
    block_threads = waves_per_block * _WAVE
    groups_per_grid = cu_count * waves_per_block
    persistent_iterations = (_TOTAL + groups_per_grid - 1) // groups_per_grid
    hidden_elements = token_tile * _HIDDEN
    handoff_elements = (waves_per_block // 2) * token_tile * 2
    hidden_load_iterations = (
        hidden_elements + block_threads * _VEC - 1
    ) // (block_threads * _VEC)

    @fx.struct
    class SharedStorage:
        hidden: fx.Array[fx.BFloat16, hidden_elements, 16]
        handoff: fx.Array[fx.BFloat16, handoff_elements, 16]

    kernel_name = (
        f"kimi_k3_m{num_tokens}_mixed_tri_bf16_fp8_gfx950"
        f"_tt{token_tile}_cu{cu_count}_wpb{waves_per_block}"
        f"_wpe{waves_per_eu}_wcm{weight_cache_modifier}"
        f"_cooppreact768_fast{int(fast_situ)}"
        f"_interleaved{int(interleaved_shared_pairs)}"
    )
    ocml_bitcode = _find_ocml_bitcode()
    ocml_exp_f32 = ExternFunction(
        "__ocml_exp_f32",
        ["float32"],
        "float32",
        is_pure=True,
        bitcode_path=ocml_bitcode,
    )
    ocml_tanh_f32 = ExternFunction(
        "__ocml_tanh_f32",
        ["float32"],
        "float32",
        is_pure=True,
        bitcode_path=ocml_bitcode,
    )

    @flyc.kernel(name=kernel_name, known_block_size=[block_threads, 1, 1])
    def kernel(
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
        lane = tid % arith.constant(_WAVE, type=i32)
        wave = tid // arith.constant(_WAVE, type=i32)
        hidden_rsrc = ptr_rsrc(hidden)
        routed_weight_rsrc = ptr_rsrc(routed_weight)
        routed_scale_rsrc = ptr_rsrc(routed_scale)
        shared_weight_rsrc = ptr_rsrc(shared_weight)
        shared_scale_rsrc = ptr_rsrc(shared_scale)
        router_weight_rsrc = ptr_rsrc(router_weight)
        routed_output_rsrc = ptr_rsrc(routed_output)
        shared_output_rsrc = ptr_rsrc(shared_output)
        router_output_rsrc = ptr_rsrc(router_output)
        shared_storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        hidden_lds = shared_storage.hidden.ptr
        handoff_lds = shared_storage.handoff.ptr
        vec2_f32 = T.vec(2, f32)
        vec2_bf16 = T.vec(2, T.bf16)
        vec8_bf16 = T.vec(_VEC, T.bf16)
        vec8_f32 = T.vec(_VEC, f32)
        zero_i32 = arith.constant(0, type=i32)
        zero_f32 = arith.constant(0.0, type=f32)
        one_f32 = arith.constant(1.0, type=f32)
        beta_f32 = arith.constant(float(situ_beta), type=f32)
        inv_beta_f32 = arith.constant(1.0 / float(situ_beta), type=f32)
        linear_beta_f32 = arith.constant(float(situ_linear_beta), type=f32)
        inv_linear_beta_f32 = arith.constant(
            1.0 / float(situ_linear_beta), type=f32
        )

        def sigmoid(value):
            if const_expr(fast_situ):
                exponent = fx.math.exp2(
                    -value * fx.Float32(_LOG2E)
                )
            else:
                exponent = ocml_exp_f32(-value)
            return one_f32 / (one_f32 + exponent)

        def tanh(value):
            if const_expr(fast_situ):
                return 2.0 * sigmoid(2.0 * value) - 1.0
            return ocml_tanh_f32(value)

        def situ(gate, up):
            return (
                beta_f32
                * tanh(gate * inv_beta_f32)
                * sigmoid(gate)
                * linear_beta_f32
                * tanh(up * inv_linear_beta_f32)
            )

        def load_bf16x8(resource, element_index):
            packed = buffer_ops.buffer_load(
                resource,
                element_index // arith.constant(2, type=i32),
                vec_width=4,
                dtype=i32,
            )
            return vector.bitcast(vec8_bf16, packed)

        def load_fp8x8(resource, element_index):
            packed = ArithValue(
                buffer_ops.buffer_load(
                    resource,
                    element_index // arith.constant(4, type=i32),
                    vec_width=2,
                    dtype=i32,
                    cache_modifier=weight_cache_modifier,
                )
            )
            p0 = vector.extract(packed, static_position=[0], dynamic_position=[])
            p1 = vector.extract(packed, static_position=[1], dynamic_position=[])
            lo0 = cvt_pk_f32_fp8(res=vec2_f32, src=p0, word_sel=False)
            hi0 = cvt_pk_f32_fp8(res=vec2_f32, src=p0, word_sel=True)
            lo1 = cvt_pk_f32_fp8(res=vec2_f32, src=p1, word_sel=False)
            hi1 = cvt_pk_f32_fp8(res=vec2_f32, src=p1, word_sel=True)
            return lo0.shuffle(hi0, [0, 1, 2, 3]).shuffle(
                lo1.shuffle(hi1, [0, 1, 2, 3]), list(range(8))
            )

        def dot2x8(left, right, accumulator):
            dot = _raw(accumulator)
            for pair in range_constexpr(4):
                lp = vector.from_elements(
                    vec2_bf16,
                    [
                        vector.extract(left, static_position=[pair * 2], dynamic_position=[]),
                        vector.extract(left, static_position=[pair * 2 + 1], dynamic_position=[]),
                    ],
                )
                rp = vector.from_elements(
                    vec2_bf16,
                    [
                        vector.extract(right, static_position=[pair * 2], dynamic_position=[]),
                        vector.extract(right, static_position=[pair * 2 + 1], dynamic_position=[]),
                    ],
                )
                dot = llvm.call_intrinsic(
                    f32,
                    "llvm.amdgcn.fdot2.f32.bf16",
                    [lp, rp, dot, arith.constant(False, type=ir.IntegerType.get_signless(1))],
                    [],
                    [],
                )
            return ArithValue(dot)

        def wave_reduce(value):
            result = _raw(value)
            for offset in (32, 16, 8, 4, 2, 1):
                peer = _raw(
                    ArithValue(result).shuffle_xor(
                        arith.constant(offset, type=i32),
                        arith.constant(_WAVE, type=i32),
                    )
                )
                result = arith_dialect.AddFOp(
                    result, peer, fastmath=fm_fast
                ).result
            return ArithValue(result)

        first_group = (
            ArithValue(gpu.block_idx.x)
            * arith.constant(waves_per_block, type=i32)
            + wave
        )
        for tile_start in range_constexpr(0, num_tokens, token_tile):
            for load_iter in range_constexpr(hidden_load_iterations):
                element = (
                    tid + arith.constant(load_iter * block_threads, type=i32)
                ) * arith.constant(_VEC, type=i32)
                can_load = arith.cmpi(
                    CmpIPredicate.ult,
                    element,
                    arith.constant(hidden_elements, type=i32),
                )
                load_if = scf.IfOp(can_load)
                with ir.InsertionPoint(load_if.then_block):
                    token_local = element // arith.constant(_HIDDEN, type=i32)
                    hidden_col = element % arith.constant(_HIDDEN, type=i32)
                    source = (
                        arith.constant(tile_start, type=i32) + token_local
                    ) * arith.constant(_HIDDEN, type=i32) + hidden_col
                    fx.ptr_store(
                        load_bf16x8(hidden_rsrc, source),
                        hidden_lds + element,
                    )
                    scf.YieldOp([])
            gpu.barrier()

            for persistent_index in range_constexpr(persistent_iterations):
                group = first_group + arith.constant(
                    persistent_index * groups_per_grid, type=i32
                )
                row = group
                row_in_range = arith.cmpi(
                    CmpIPredicate.ult,
                    row,
                    arith.constant(_TOTAL, type=i32),
                )
                row_if = scf.IfOp(row_in_range)
                with ir.InsertionPoint(row_if.then_block):
                    is_fp8 = arith.cmpi(
                        CmpIPredicate.ult,
                        row,
                        arith.constant(_FP8_OUT, type=i32),
                    )
                    precision_if = scf.IfOp(is_fp8, has_else=True)
                    with ir.InsertionPoint(precision_if.then_block):
                        is_routed = arith.cmpi(
                            CmpIPredicate.ult,
                            row,
                            arith.constant(_ROUTED, type=i32),
                        )
                        shared_row = row - arith.constant(_ROUTED, type=i32)
                        shared_pair = shared_row // arith.constant(2, type=i32)
                        shared_role = shared_row % arith.constant(2, type=i32)
                        if const_expr(interleaved_shared_pairs):
                            shared_weight_row = shared_row
                        else:
                            shared_weight_row = (
                                shared_pair
                                + shared_role
                                * arith.constant(
                                    _SHARED // 2, type=i32
                                )
                            )
                        scale_if = scf.IfOp(
                            is_routed, results_=[f32], has_else=True
                        )
                        with ir.InsertionPoint(scale_if.then_block):
                            scale = buffer_ops.buffer_load(
                                routed_scale_rsrc, row, vec_width=1, dtype=f32
                            )
                            scf.YieldOp([_raw(scale)])
                        with ir.InsertionPoint(scale_if.else_block):
                            scale = buffer_ops.buffer_load(
                                shared_scale_rsrc,
                                shared_weight_row,
                                vec_width=1,
                                dtype=f32,
                            )
                            scf.YieldOp([_raw(scale)])
                        accumulators = [ArithValue(zero_f32) for _ in range(token_tile)]
                        for k_iter in range_constexpr(_HIDDEN // (_WAVE * _VEC)):
                            k_element = (
                                lane
                                + arith.constant(k_iter * _WAVE, type=i32)
                            ) * arith.constant(_VEC, type=i32)
                            weight_if = scf.IfOp(
                                is_routed, results_=[vec8_f32], has_else=True
                            )
                            with ir.InsertionPoint(weight_if.then_block):
                                weight = load_fp8x8(
                                    routed_weight_rsrc,
                                    row * arith.constant(_HIDDEN, type=i32)
                                    + k_element,
                                )
                                scf.YieldOp([_raw(weight)])
                            with ir.InsertionPoint(weight_if.else_block):
                                weight = load_fp8x8(
                                    shared_weight_rsrc,
                                    shared_weight_row
                                    * arith.constant(_HIDDEN, type=i32)
                                    + k_element,
                                )
                                scf.YieldOp([_raw(weight)])
                            weight_bf16 = arith.trunc_f(
                                vec8_bf16, weight_if.results[0]
                            )
                            for token_local in range_constexpr(token_tile):
                                h = fx.ptr_load(
                                    hidden_lds
                                    + arith.constant(
                                        token_local * _HIDDEN, type=i32
                                    )
                                    + k_element,
                                    result_type=vec8_bf16,
                                )
                                accumulators[token_local] = dot2x8(
                                    h, weight_bf16, accumulators[token_local]
                                )
                        lane_zero = arith.cmpi(
                            CmpIPredicate.eq, lane, arith.constant(0, type=i32)
                        )
                        for token_local in range_constexpr(token_tile):
                            reduced = wave_reduce(accumulators[token_local])
                            reduced = reduced * ArithValue(scale_if.results[0])
                            store_if = scf.IfOp(lane_zero)
                            with ir.InsertionPoint(store_if.then_block):
                                token = tile_start + token_local
                                result = arith.trunc_f(T.bf16, _raw(reduced))
                                output_if = scf.IfOp(
                                    is_routed, results_=[], has_else=True
                                )
                                with ir.InsertionPoint(output_if.then_block):
                                    buffer_ops.buffer_store(
                                        result,
                                        routed_output_rsrc,
                                        arith.constant(
                                            token * _ROUTED, type=i32
                                        )
                                        + row,
                                    )
                                    scf.YieldOp([])
                                with ir.InsertionPoint(output_if.else_block):
                                    pair_local = wave // arith.constant(
                                        2, type=i32
                                    )
                                    scratch_index = (
                                        pair_local
                                        * arith.constant(
                                            token_tile * 2, type=i32
                                        )
                                        + arith.constant(
                                            token_local * 2, type=i32
                                        )
                                        + shared_role
                                    )
                                    fx.ptr_store(
                                        result,
                                        handoff_lds + scratch_index,
                                    )
                                    scf.YieldOp([])
                                scf.YieldOp([])
                        scf.YieldOp([])

                    with ir.InsertionPoint(precision_if.else_block):
                        router_row = row - arith.constant(_FP8_OUT, type=i32)
                        accumulators = [ArithValue(zero_f32) for _ in range(token_tile)]
                        for k_iter in range_constexpr(
                            _HIDDEN // (_WAVE * _VEC)
                        ):
                            k = (
                                lane
                                + arith.constant(k_iter * _WAVE, type=i32)
                            ) * arith.constant(_VEC, type=i32)
                            weight_bf16 = load_bf16x8(
                                router_weight_rsrc,
                                router_row
                                * arith.constant(_HIDDEN, type=i32)
                                + k,
                            )
                            for token_local in range_constexpr(token_tile):
                                hidden_bf16 = fx.ptr_load(
                                    hidden_lds
                                    + arith.constant(
                                        token_local * _HIDDEN, type=i32
                                    )
                                    + k,
                                    result_type=vec8_bf16,
                                )
                                accumulators[token_local] = dot2x8(
                                    hidden_bf16,
                                    weight_bf16,
                                    accumulators[token_local],
                                )
                        last_lane = arith.cmpi(
                            CmpIPredicate.eq,
                            lane,
                            arith.constant(_WAVE - 1, type=i32),
                        )
                        for token_local in range_constexpr(token_tile):
                            accumulator = accumulators[token_local]
                            for dpp in (0xB1, 0x4E, 0x141, 0x140, 0x142, 0x143):
                                remote_i32 = rocdl.update_dpp(
                                    i32,
                                    zero_i32,
                                    arith.bitcast(i32, _raw(accumulator)),
                                    dpp,
                                    0xF,
                                    0xF,
                                    True,
                                )
                                accumulator = accumulator + ArithValue(
                                    arith.bitcast(f32, remote_i32)
                                )
                            store_if = scf.IfOp(last_lane)
                            with ir.InsertionPoint(store_if.then_block):
                                rounded = arith.trunc_f(
                                    T.bf16, _raw(accumulator)
                                )
                                buffer_ops.buffer_store(
                                    arith.extf(f32, rounded),
                                    router_output_rsrc,
                                    arith.constant(
                                        (tile_start + token_local) * _ROUTER,
                                        type=i32,
                                    )
                                    + router_row,
                                )
                                scf.YieldOp([])
                        scf.YieldOp([])
                    scf.YieldOp([])
                if const_expr(cooperative_preactivate_shared):
                    block_first_group = (
                        ArithValue(gpu.block_idx.x)
                        * arith.constant(waves_per_block, type=i32)
                        + arith.constant(
                            persistent_index * groups_per_grid, type=i32
                        )
                    )
                    shared_lower = arith.cmpi(
                        CmpIPredicate.uge,
                        block_first_group,
                        arith.constant(_ROUTED, type=i32),
                    )
                    shared_upper = arith.cmpi(
                        CmpIPredicate.ult,
                        block_first_group,
                        arith.constant(_FP8_OUT, type=i32),
                    )
                    shared_iteration = arith.andi(
                        shared_lower, shared_upper
                    )
                    shared_iteration_if = scf.IfOp(shared_iteration)
                    with ir.InsertionPoint(
                        shared_iteration_if.then_block
                    ):
                        gpu.barrier()
                        is_gate_wave = arith.cmpi(
                            CmpIPredicate.eq,
                            wave % arith.constant(2, type=i32),
                            arith.constant(0, type=i32),
                        )
                        gate_wave_if = scf.IfOp(is_gate_wave)
                        with ir.InsertionPoint(gate_wave_if.then_block):
                            pair_local = wave // arith.constant(2, type=i32)
                            shared_pair = (
                                row - arith.constant(_ROUTED, type=i32)
                            ) // arith.constant(2, type=i32)
                            for token_local in range_constexpr(token_tile):
                                scratch_base = (
                                    pair_local
                                    * arith.constant(
                                        token_tile * 2, type=i32
                                    )
                                    + arith.constant(
                                        token_local * 2, type=i32
                                    )
                                )
                                gate = ArithValue(
                                    arith.extf(
                                        f32,
                                        _raw(
                                            fx.ptr_load(
                                                handoff_lds + scratch_base,
                                                result_type=T.bf16,
                                            )
                                        ),
                                    )
                                )
                                up = ArithValue(
                                    arith.extf(
                                        f32,
                                        _raw(
                                            fx.ptr_load(
                                                handoff_lds
                                                + scratch_base
                                                + arith.constant(1, type=i32),
                                                result_type=T.bf16,
                                            )
                                        ),
                                    )
                                )
                                activated = arith.trunc_f(
                                    T.bf16, _raw(situ(gate, up))
                                )
                                buffer_ops.buffer_store(
                                    activated,
                                    shared_output_rsrc,
                                    arith.constant(
                                        (tile_start + token_local)
                                        * (_SHARED // 2),
                                        type=i32,
                                    )
                                    + shared_pair,
                                )
                            scf.YieldOp([])
                        gpu.barrier()
                        scf.YieldOp([])
            gpu.barrier()

    @flyc.jit
    def launch(
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
                        T.i32, int(waves_per_eu)
                    )
        kernel(
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
            grid=(cu_count, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        }
    }
    return launch


__all__ = ["build_kimi_k3_multitoken_tri_projection_module"]
