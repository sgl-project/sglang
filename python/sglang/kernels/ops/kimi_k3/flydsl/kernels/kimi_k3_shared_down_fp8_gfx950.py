# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused Kimi-K3 B1 SiTU and FP8-weight shared down projection for gfx950."""

import math
import os
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as arith_dialect
from flydsl._mlir.dialects import scf
from flydsl.compiler.extern_link import ExternFunction
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr

from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.rocdl import cvt_pk_f32_fp8
from flydsl.expr.typing import T
from aiter.ops.flydsl.kernels.vector import ReductionOp

from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_rsrc,
)

_SHARED_INTERMEDIATE_SIZE = 768
_SHARED_GATE_UP_SIZE = 2 * _SHARED_INTERMEDIATE_SIZE
_HIDDEN_SIZE = 7168
_WAVE_SIZE = 64
_ELEMENTS_PER_LOAD = 4


def _find_ocml_bitcode() -> str:
    """Find ROCm's device math library without pinning a ROCm release."""

    roots = [
        Path(value)
        for value in (
            os.environ.get("ROCM_PATH"),
            os.environ.get("ROCM_HOME"),
            "/opt/rocm",
        )
        if value
    ]
    patterns = (
        "amdgcn/bitcode/ocml.bc",
        "lib/llvm/lib/clang/*/lib/amdgcn/bitcode/ocml.bc",
    )
    for root in roots:
        for pattern in patterns:
            matches = sorted(root.glob(pattern), reverse=True)
            if matches:
                return str(matches[0])
    raise RuntimeError("unable to locate ROCm OCML bitcode")


def _raw(value):
    return value.ir_value() if hasattr(value, "ir_value") else value


def build_kimi_k3_b1_shared_down_fp8_module(
    num_tokens: int = 1,
    rows_per_wave: int = 2,
    cu_count: int = 256,
    waves_per_eu: int = 0,
    weight_cache_modifier: int = 2,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
):
    """Build the fused SiTU + row-scaled FP8-weight down projection."""

    if num_tokens != 1:
        raise ValueError("num_tokens must be 1")
    if rows_per_wave not in (1, 2, 3, 4, 5, 6, 8):
        raise ValueError("rows_per_wave must be 1, 2, 3, 4, 5, 6, or 8")
    if not 1 <= cu_count <= 256:
        raise ValueError("cu_count must be between 1 and 256")
    if waves_per_eu < 0:
        raise ValueError("waves_per_eu must be non-negative")
    if weight_cache_modifier not in (0, 1, 2, 3):
        raise ValueError("weight_cache_modifier must be between 0 and 3")
    if (
        not math.isfinite(situ_beta)
        or not math.isfinite(situ_linear_beta)
        or situ_beta <= 0.0
        or situ_linear_beta <= 0.0
    ):
        raise ValueError("SiTU beta values must be finite and positive")

    output_groups = (_HIDDEN_SIZE + rows_per_wave - 1) // rows_per_wave
    waves_per_block = min(16, (output_groups + cu_count - 1) // cu_count)
    block_threads = waves_per_block * _WAVE_SIZE
    groups_per_grid = cu_count * waves_per_block
    persistent_iterations = (output_groups + groups_per_grid - 1) // groups_per_grid
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

    @fx.struct
    class SharedStorage:
        activated: fx.Array[fx.BFloat16, _SHARED_INTERMEDIATE_SIZE, 16]

    beta_tag = f"{situ_beta:g}".replace(".", "p")
    linear_beta_tag = f"{situ_linear_beta:g}".replace(".", "p")
    kernel_name = (
        f"kimi_k3_b{num_tokens}_situ_shared_down_bf16_fp8_gfx950"
        f"_rpw{rows_per_wave}_cu{cu_count}_wpb{waves_per_block}"
        f"_wpe{waves_per_eu}_wcm{weight_cache_modifier}"
        f"_sb{beta_tag}_slb{linear_beta_tag}"
    )

    @flyc.kernel(
        name=kernel_name,
        known_block_size=[block_threads, 1, 1],
    )
    def shared_down_fp8_kernel(
        gate_up: fx.Pointer,
        weight: fx.Pointer,
        weight_scale: fx.Pointer,
        output: fx.Pointer,
    ):
        i32 = T.i32
        f32 = T.f32
        fm_fast = arith.FastMathFlags.fast
        tid = ArithValue(gpu.thread_idx.x)
        lane = tid % arith.constant(_WAVE_SIZE, type=i32)
        wave = tid // arith.constant(_WAVE_SIZE, type=i32)

        gate_up_rsrc = ptr_rsrc(gate_up)
        weight_rsrc = ptr_rsrc(weight)
        scale_rsrc = ptr_rsrc(weight_scale)
        output_rsrc = ptr_rsrc(output)
        token = ArithValue(gpu.block_idx.y)
        gate_up_token_base = token * arith.constant(
            2 * _SHARED_INTERMEDIATE_SIZE, type=i32
        )
        output_token_base = token * arith.constant(_HIDDEN_SIZE, type=i32)
        activated_lds = (
            fx.SharedAllocator().allocate(SharedStorage).peek().activated.ptr
        )

        vec2_f32 = T.vec(2, f32)
        vec4_bf16 = T.vec(_ELEMENTS_PER_LOAD, T.bf16)
        vec4_f32 = T.vec(_ELEMENTS_PER_LOAD, f32)
        zero_f32 = arith.constant(0.0, type=f32)
        one_f32 = arith.constant(1.0, type=f32)
        zero_i32 = arith.constant(0, type=i32)

        def sigmoid(value):
            # Match vLLM's production 1 / (1 + expf(-x)) evaluation.
            exponent = ocml_exp_f32(-value)
            return one_f32 / (one_f32 + exponent)

        def tanh(value):
            return ocml_tanh_f32(value)

        def load_fp8x4_as_f32(resource, element_index):
            packed = buffer_ops.buffer_load(
                resource,
                element_index // arith.constant(4, type=i32),
                vec_width=1,
                dtype=i32,
                cache_modifier=weight_cache_modifier,
            )
            weight_lo = cvt_pk_f32_fp8(
                res=vec2_f32,
                src=packed,
                word_sel=False,
            )
            weight_hi = cvt_pk_f32_fp8(
                res=vec2_f32,
                src=packed,
                word_sel=True,
            )
            return weight_lo.shuffle(weight_hi, [0, 1, 2, 3])

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

        # Every resident block computes the 768-element SiTU vector once and
        # retains it in LDS while that block processes its output-row tile.
        for activation_iteration in range_constexpr(
            (_SHARED_INTERMEDIATE_SIZE + block_threads - 1) // block_threads
        ):
            element = tid + arith.constant(
                activation_iteration * block_threads, type=i32
            )
            valid = arith.cmpi(
                CmpIPredicate.ult,
                element,
                arith.constant(_SHARED_INTERMEDIATE_SIZE, type=i32),
            )
            activation_if = scf.IfOp(valid)
            with ir.InsertionPoint(activation_if.then_block):
                gate_bf16 = buffer_ops.buffer_load(
                    gate_up_rsrc,
                    gate_up_token_base + element,
                    vec_width=1,
                    dtype=T.bf16,
                )
                up_bf16 = buffer_ops.buffer_load(
                    gate_up_rsrc,
                    gate_up_token_base
                    + element
                    + arith.constant(_SHARED_INTERMEDIATE_SIZE, type=i32),
                    vec_width=1,
                    dtype=T.bf16,
                )
                gate = ArithValue(arith.extf(f32, gate_bf16))
                up = ArithValue(arith.extf(f32, up_bf16))
                beta = arith.constant(float(situ_beta), type=f32)
                inv_beta = arith.constant(1.0 / float(situ_beta), type=f32)
                linear_beta = arith.constant(float(situ_linear_beta), type=f32)
                inv_linear_beta = arith.constant(
                    1.0 / float(situ_linear_beta),
                    type=f32,
                )
                activated = (
                    beta
                    * tanh(gate * inv_beta)
                    * sigmoid(gate)
                    * linear_beta
                    * tanh(up * inv_linear_beta)
                )
                fx.ptr_store(
                    arith.trunc_f(T.bf16, _raw(activated)),
                    activated_lds + element,
                )
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
                    arith.constant(_HIDDEN_SIZE, type=i32),
                )
                row_if = scf.IfOp(row_in_range)
                with ir.InsertionPoint(row_if.then_block):
                    row_scale = buffer_ops.buffer_load(
                        scale_rsrc,
                        row,
                        vec_width=1,
                        dtype=f32,
                    )
                    local_dot = ArithValue(zero_f32)
                    for k_iteration in range_constexpr(
                        _SHARED_INTERMEDIATE_SIZE // (_WAVE_SIZE * _ELEMENTS_PER_LOAD)
                    ):
                        k_element = (
                            lane + arith.constant(k_iteration * _WAVE_SIZE, type=i32)
                        ) * arith.constant(_ELEMENTS_PER_LOAD, type=i32)
                        activated_bf16 = fx.ptr_load(
                            activated_lds + k_element,
                            result_type=vec4_bf16,
                        )
                        activated_f32 = ArithValue(activated_bf16).extf(vec4_f32)
                        weight_element = (
                            row * arith.constant(_SHARED_INTERMEDIATE_SIZE, type=i32)
                            + k_element
                        )
                        weight_f32 = load_fp8x4_as_f32(weight_rsrc, weight_element)
                        local_dot = local_dot + (activated_f32 * weight_f32).reduce(
                            ReductionOp.ADD, fastmath=fm_fast
                        )

                    reduced = ArithValue(wave_reduce_add(local_dot)) * ArithValue(
                        row_scale
                    )
                    is_lane_zero = arith.cmpi(CmpIPredicate.eq, lane, zero_i32)
                    write_if = scf.IfOp(is_lane_zero)
                    with ir.InsertionPoint(write_if.then_block):
                        buffer_ops.buffer_store(
                            arith.trunc_f(T.bf16, _raw(reduced)),
                            output_rsrc,
                            output_token_base + row,
                        )
                        scf.YieldOp([])
                    scf.YieldOp([])

    @flyc.jit
    def launch_shared_down_fp8(
        gate_up: fx.Pointer,
        weight: fx.Pointer,
        weight_scale: fx.Pointer,
        output: fx.Pointer,
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
        shared_down_fp8_kernel(
            gate_up,
            weight,
            weight_scale,
            output,
        ).launch(
            grid=(cu_count, num_tokens, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch_shared_down_fp8.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_shared_down_fp8
