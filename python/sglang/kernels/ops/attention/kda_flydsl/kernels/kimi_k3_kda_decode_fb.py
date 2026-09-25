# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL Kimi-K3 KDA decode with a fused head-local f_b projection."""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import vector
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    GTensor,
    _to_raw,
)
from flydsl._mlir import ir
from flydsl._mlir.dialects import gpu as mlir_gpu
from flydsl._mlir.dialects import llvm, scf
from flydsl._mlir.dialects import vector as mlir_vector
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T

_HEADS = 12
_DIM = 128
_LOG2E = math.log2(math.e)
_SCALE = _DIM**-0.5
_BLOCK_THREADS = 256
_NUM_WARPS = 4
_WARP_SIZE = 64
_WARP_THREADS_K = 8
_VALUES_PER_THREAD_K = 4
_WARP_TILE_K = _WARP_THREADS_K * _VALUES_PER_THREAD_K
_K_ITERS = _DIM // _WARP_TILE_K
_WARP_THREADS_V = _WARP_SIZE // _WARP_THREADS_K
_V_GROUP_TILE = _NUM_WARPS * _WARP_THREADS_V
_V_ITERS = _DIM // _V_GROUP_TILE
_PROJECTION_VECTOR = 4
_PROJECTION_ITERS = _DIM // _PROJECTION_VECTOR
_DEFAULT_WAVES_PER_EU = 2


@functools.cache
def create_kimi_k3_kda_decode_fb_kernel(
    norm_eps: float,
    lower_bound: float,
    *,
    waves_per_eu: int = _DEFAULT_WAVES_PER_EU,
    cooperative_f_a: bool = False,
    parallel_front: bool = False,
    fused_norm_reduce: bool = False,
    projection_fdot2: bool = False,
):
    """Build the fixed gfx950 BF16 f_b plus KDA decode specialization."""
    conv_tid_offset = _DIM if parallel_front else 0
    conv_tid_upper = 2 * _DIM if parallel_front else _DIM

    if cooperative_f_a:

        @fx.struct
        class SharedStorage:
            f_a: fx.Array[fx.BFloat16, _DIM, 16]
            q: fx.Array[fx.BFloat16, _DIM, 16]
            k: fx.Array[fx.BFloat16, _DIM, 16]
            v: fx.Array[fx.BFloat16, _DIM, 16]
            gate: fx.Array[fx.BFloat16, _DIM, 16]
            recurrent_out: fx.Array[fx.BFloat16, _DIM, 16]
            norm_partial: fx.Array[fx.Float32, 4, 16]

    else:

        @fx.struct
        class SharedStorage:
            q: fx.Array[fx.BFloat16, _DIM, 16]
            k: fx.Array[fx.BFloat16, _DIM, 16]
            v: fx.Array[fx.BFloat16, _DIM, 16]
            gate: fx.Array[fx.BFloat16, _DIM, 16]
            recurrent_out: fx.Array[fx.BFloat16, _DIM, 16]
            norm_partial: fx.Array[fx.Float32, 4, 16]

    kernel_name = "kimi_k3_kda_decode_fb_bf16_gfx950"
    if (
        cooperative_f_a
        or parallel_front
        or fused_norm_reduce
        or projection_fdot2
        or waves_per_eu != _DEFAULT_WAVES_PER_EU
    ):
        kernel_name += (
            f"_wpe{waves_per_eu}_cfa{int(cooperative_f_a)}"
            f"_pf{int(parallel_front)}"
            f"_fnr{int(fused_norm_reduce)}"
            f"_fd2{int(projection_fdot2)}"
        )

    @flyc.kernel(
        name=kernel_name,
        known_block_size=[_BLOCK_THREADS, 1, 1],
    )
    def kernel(
        f_a_mem: fx.Tensor,
        f_b_weight_mem: fx.Tensor,
        x_mem: fx.Tensor,
        weight_mem: fx.Tensor,
        conv_state_mem: fx.Tensor,
        raw_beta_mem: fx.Tensor,
        A_log_mem: fx.Tensor,
        dt_bias_mem: fx.Tensor,
        state_mem: fx.Tensor,
        state_indices_mem: fx.Tensor,
        output_gate_mem: fx.Tensor,
        norm_weight_mem: fx.Tensor,
        out_mem: fx.Tensor,
        batch_size: fx.Int32,
        stride_f_a_token: fx.Int32,
        stride_f_b_head: fx.Int32,
        stride_f_b_output: fx.Int32,
        stride_x_token: fx.Int32,
        stride_weight_channel: fx.Int32,
        stride_weight_width: fx.Int32,
        stride_conv_slot: fx.Int32,
        stride_conv_channel: fx.Int32,
        stride_conv_width: fx.Int32,
        stride_beta_token: fx.Int32,
        stride_state_slot: fx.Int32,
        stride_gate_token: fx.Int32,
        stride_gate_head: fx.Int32,
        stride_out_token: fx.Int32,
        stride_out_head: fx.Int32,
    ):
        del batch_size

        f_a = GTensor(f_a_mem, dtype=T.bf16, shape=(-1,))
        f_b_weight = GTensor(f_b_weight_mem, dtype=T.bf16, shape=(-1,))
        x = GTensor(x_mem, dtype=T.bf16, shape=(-1,))
        weight = GTensor(weight_mem, dtype=T.f32, shape=(-1,))
        conv_state = GTensor(conv_state_mem, dtype=T.bf16, shape=(-1,))
        raw_beta = GTensor(raw_beta_mem, dtype=T.bf16, shape=(-1,))
        A_log = GTensor(A_log_mem, dtype=T.f32, shape=(-1,))
        dt_bias = GTensor(dt_bias_mem, dtype=T.f32, shape=(-1,))
        state = GTensor(state_mem, dtype=T.f32, shape=(-1,))
        state_indices = GTensor(state_indices_mem, dtype=T.i32, shape=(-1,))
        output_gate = GTensor(output_gate_mem, dtype=T.bf16, shape=(-1,))
        norm_weight = GTensor(norm_weight_mem, dtype=T.bf16, shape=(-1,))
        out = GTensor(out_mem, dtype=T.bf16, shape=(-1,))

        shared = fx.SharedAllocator().allocate(SharedStorage).peek()
        f_a_lds = shared.f_a.ptr if cooperative_f_a else shared.q.ptr
        q_lds = shared.q.ptr
        k_lds = shared.k.ptr
        v_lds = shared.v.ptr
        gate_lds = shared.gate.ptr
        out_lds = shared.recurrent_out.ptr
        norm_lds = shared.norm_partial.ptr

        tid = fx.thread_idx.x
        block = fx.block_idx.x
        batch = block // fx.Int32(_HEADS)
        head = block % fx.Int32(_HEADS)
        lane = tid % fx.Int32(_WARP_SIZE)
        warp = tid // fx.Int32(_WARP_SIZE)
        lane_k = lane % fx.Int32(_WARP_THREADS_K)

        state_idx = fx.Int32(state_indices[batch])
        valid = state_idx > fx.Int32(0)

        valid_if = scf.IfOp(_to_raw(valid), results_=[], has_else=True)
        with ir.InsertionPoint(valid_if.then_block):
            if const_expr(cooperative_f_a):
                f_a_load_if = scf.IfOp(
                    _to_raw(tid < fx.Int32(_DIM)),
                    results_=[],
                    has_else=False,
                )
                with ir.InsertionPoint(f_a_load_if.then_block):
                    fx.ptr_store(
                        fx.BFloat16(f_a[batch * stride_f_a_token + tid]),
                        f_a_lds + tid,
                    )
                    scf.YieldOp([])
                fx.gpu.barrier()

            # Threads 0..127 own one output each. Accumulation is FP32 and the
            # single BF16 store is the same numerical boundary as F.linear.
            projection_if = scf.IfOp(
                _to_raw(tid < fx.Int32(_DIM)),
                results_=[],
                has_else=False,
            )
            with ir.InsertionPoint(projection_if.then_block):
                i1 = ir.IntegerType.get_signless(1)
                vec_f32_projection = T.vec(_PROJECTION_VECTOR, T.f32)
                vec2_bf16 = T.vec(2, T.bf16)
                accum = fx.full(
                    _PROJECTION_VECTOR,
                    0.0,
                    fx.Float32,
                )
                local_dot = fx.Float32(0.0)
                f_a_base = batch * stride_f_a_token
                f_b_base = head * stride_f_b_head + tid * stride_f_b_output
                for projection_iter in range_constexpr(_PROJECTION_ITERS):
                    projection_offset = fx.Int32(projection_iter * _PROJECTION_VECTOR)
                    if const_expr(cooperative_f_a):
                        f_a_values = fx.ptr_load(
                            f_a_lds + projection_offset,
                            result_type=T.vec(_PROJECTION_VECTOR, T.bf16),
                        ).extf(vec_f32_projection)
                    else:
                        f_a_values = f_a.vec_load(
                            (f_a_base + projection_offset,),
                            _PROJECTION_VECTOR,
                        ).extf(vec_f32_projection)
                    weight_values = f_b_weight.vec_load(
                        (f_b_base + projection_offset,),
                        _PROJECTION_VECTOR,
                    ).extf(vec_f32_projection)
                    if const_expr(projection_fdot2):
                        f_a_bf16 = f_a_values.truncf(T.vec(_PROJECTION_VECTOR, T.bf16))
                        weight_bf16 = weight_values.truncf(
                            T.vec(_PROJECTION_VECTOR, T.bf16)
                        )
                        for pair_index in range_constexpr(_PROJECTION_VECTOR // 2):
                            f_a_pair = vector.from_elements(
                                vec2_bf16,
                                [
                                    vector.extract(
                                        f_a_bf16,
                                        static_position=[pair_index * 2],
                                        dynamic_position=[],
                                    ),
                                    vector.extract(
                                        f_a_bf16,
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
                            local_dot = ArithValue(
                                llvm.call_intrinsic(
                                    T.f32,
                                    "llvm.amdgcn.fdot2.f32.bf16",
                                    [
                                        f_a_pair,
                                        weight_pair,
                                        _to_raw(local_dot),
                                        arith.constant(False, type=i1),
                                    ],
                                    [],
                                    [],
                                )
                            )
                    else:
                        accum = mlir_vector.FMAOp(
                            f_a_values,
                            weight_values,
                            accum,
                        ).result
                if const_expr(projection_fdot2):
                    projected = local_dot
                else:
                    projected = mlir_vector.ReductionOp(
                        T.f32,
                        vector.CombiningKind.ADD,
                        accum,
                    ).dest
                fx.ptr_store(
                    fx.BFloat16(projected),
                    gate_lds + tid,
                )
                scf.YieldOp([])

            # A workgroup exclusively owns all three convolution channels for
            # its (batch, head), so every cache entry is shifted exactly once.
            conv_if = scf.IfOp(
                _to_raw(
                    (tid >= fx.Int32(conv_tid_offset))
                    & (tid < fx.Int32(conv_tid_upper))
                ),
                results_=[],
                has_else=False,
            )
            with ir.InsertionPoint(conv_if.then_block):
                channel_local = tid - fx.Int32(conv_tid_offset)
                q_channel = head * fx.Int32(_DIM) + channel_local
                k_channel = (
                    fx.Int32(_HEADS * _DIM) + head * fx.Int32(_DIM) + channel_local
                )
                v_channel = (
                    fx.Int32(2 * _HEADS * _DIM) + head * fx.Int32(_DIM) + channel_local
                )

                def convolve_channel(channel):
                    cs_base = (
                        state_idx * stride_conv_slot + channel * stride_conv_channel
                    )
                    c0 = fx.Float32(conv_state[cs_base])
                    c1 = fx.Float32(conv_state[cs_base + stride_conv_width])
                    c2 = fx.Float32(
                        conv_state[cs_base + fx.Int32(2) * stride_conv_width]
                    )
                    current = fx.BFloat16(x[batch * stride_x_token + channel])
                    current_f32 = fx.Float32(current)
                    w_base = channel * stride_weight_channel
                    acc = c0 * fx.Float32(weight[w_base])
                    acc = acc + c1 * fx.Float32(weight[w_base + stride_weight_width])
                    acc = acc + c2 * fx.Float32(
                        weight[w_base + fx.Int32(2) * stride_weight_width]
                    )
                    acc = acc + current_f32 * fx.Float32(
                        weight[w_base + fx.Int32(3) * stride_weight_width]
                    )
                    silu = acc / (
                        fx.Float32(1.0) + fx.math.exp2(-acc * fx.Float32(_LOG2E))
                    )
                    conv_state.store(
                        cs_base,
                        fx.BFloat16(c1),
                    )
                    conv_state.store(
                        cs_base + stride_conv_width,
                        fx.BFloat16(c2),
                    )
                    conv_state.store(
                        cs_base + fx.Int32(2) * stride_conv_width,
                        current,
                    )
                    return silu.to(fx.BFloat16)

                q_conv = convolve_channel(q_channel)
                k_conv = convolve_channel(k_channel)
                v_conv = convolve_channel(v_channel)
                fx.ptr_store(q_conv, q_lds + channel_local)
                fx.ptr_store(k_conv, k_lds + channel_local)
                fx.ptr_store(v_conv, v_lds + channel_local)
                scf.YieldOp([])

            # Both projection and convolution LDS values must be visible
            # before the recurrent core begins.
            fx.gpu.barrier()

            # Four waves split V into 32-row groups. Eight-lane subgroups
            # reduce K; each lane issues one aligned f32x4 state transaction.
            k_vec_start = lane_k * fx.Int32(_VALUES_PER_THREAD_K)
            global_v_start = warp * fx.Int32(_WARP_THREADS_V) + lane // fx.Int32(
                _WARP_THREADS_K
            )
            vec_f32 = T.vec(_VALUES_PER_THREAD_K, T.f32)
            vec_bf16 = T.vec(_VALUES_PER_THREAD_K, T.bf16)
            zero_vec = fx.full(
                _VALUES_PER_THREAD_K,
                0.0,
                fx.Float32,
            )

            q_vecs = []
            k_vecs = []
            decay_vecs = []
            sum_q_partial = fx.Float32(0.0)
            sum_k_partial = fx.Float32(0.0)
            a = fx.math.exp2(fx.Float32(A_log[head]) * fx.Float32(_LOG2E))

            for ki in range_constexpr(_K_ITERS):
                k_base = k_vec_start + fx.Int32(ki * _WARP_TILE_K)
                q_bf16 = fx.ptr_load(
                    q_lds + k_base,
                    result_type=vec_bf16,
                )
                k_bf16 = fx.ptr_load(
                    k_lds + k_base,
                    result_type=vec_bf16,
                )
                q_f32 = q_bf16.extf(vec_f32)
                k_f32 = k_bf16.extf(vec_f32)
                q_vecs.append(q_f32)
                k_vecs.append(k_f32)
                sum_q_vec = q_f32 * q_f32
                sum_k_vec = k_f32 * k_f32
                sum_q_partial = (
                    sum_q_partial
                    + mlir_vector.ReductionOp(
                        T.f32,
                        vector.CombiningKind.ADD,
                        sum_q_vec,
                    ).dest
                )
                sum_k_partial = (
                    sum_k_partial
                    + mlir_vector.ReductionOp(
                        T.f32,
                        vector.CombiningKind.ADD,
                        sum_k_vec,
                    ).dest
                )

                # The projection is rounded in LDS before the lower-bound gate.
                gate_bf16 = fx.ptr_load(
                    gate_lds + k_base,
                    result_type=vec_bf16,
                )
                gate_f32 = gate_bf16.extf(vec_f32)
                dt = dt_bias.vec_load(
                    (head * fx.Int32(_DIM) + k_base,),
                    _VALUES_PER_THREAD_K,
                )
                sigmoid_arg = (gate_f32 + dt) * a
                gate = fx.Float32(lower_bound) / (
                    fx.Float32(1.0) + fx.math.exp2(-sigmoid_arg * fx.Float32(_LOG2E))
                )
                decay_vecs.append(fx.math.exp2(gate * fx.Float32(_LOG2E)))

            width = fx.Int32(_WARP_SIZE)
            for offset in (1, 2, 4):
                sum_q_partial = (
                    sum_q_partial
                    + mlir_gpu.ShuffleOp(
                        _to_raw(sum_q_partial),
                        _to_raw(fx.Int32(offset)),
                        _to_raw(width),
                        mode="xor",
                    ).shuffleResult
                )
                sum_k_partial = (
                    sum_k_partial
                    + mlir_gpu.ShuffleOp(
                        _to_raw(sum_k_partial),
                        _to_raw(fx.Int32(offset)),
                        _to_raw(width),
                        mode="xor",
                    ).shuffleResult
                )

            subgroup_leader = (lane // fx.Int32(_WARP_THREADS_K)) * fx.Int32(
                _WARP_THREADS_K
            )
            norm_q = mlir_gpu.ShuffleOp(
                _to_raw(sum_q_partial),
                _to_raw(subgroup_leader),
                _to_raw(width),
                mode="idx",
            ).shuffleResult
            norm_k = mlir_gpu.ShuffleOp(
                _to_raw(sum_k_partial),
                _to_raw(subgroup_leader),
                _to_raw(width),
                mode="idx",
            ).shuffleResult
            inv_q = fx.math.rsqrt(fx.Float32(norm_q) + fx.Float32(1e-6))
            inv_k = fx.math.rsqrt(fx.Float32(norm_k) + fx.Float32(1e-6))

            for ki in range_constexpr(_K_ITERS):
                q_vecs[ki] = q_vecs[ki] * fx.Float32(inv_q) * fx.Float32(_SCALE)
                k_vecs[ki] = k_vecs[ki] * fx.Float32(inv_k)

            dot_kq_vec = zero_vec
            for ki in range_constexpr(_K_ITERS):
                dot_kq_vec = mlir_vector.FMAOp(
                    k_vecs[ki],
                    q_vecs[ki],
                    dot_kq_vec,
                ).result
            dot_kq = mlir_vector.ReductionOp(
                T.f32,
                vector.CombiningKind.ADD,
                dot_kq_vec,
            ).dest
            for offset in (1, 2, 4):
                dot_kq = (
                    dot_kq
                    + mlir_gpu.ShuffleOp(
                        _to_raw(dot_kq),
                        _to_raw(fx.Int32(offset)),
                        _to_raw(width),
                        mode="xor",
                    ).shuffleResult
                )

            beta_value = fx.Float32(raw_beta[batch * stride_beta_token + head])
            beta = fx.Float32(1.0) / (
                fx.Float32(1.0) + fx.math.exp2(-beta_value * fx.Float32(_LOG2E))
            )
            state_head_base = state_idx * stride_state_slot + head * fx.Int32(
                _DIM * _DIM
            )

            def process_state_row(vi, row_state_vecs):
                global_v = global_v_start + fx.Int32(vi * _V_GROUP_TILE)
                sum_hk_vec = zero_vec
                sum_hq_vec = zero_vec
                for ki in range_constexpr(_K_ITERS):
                    decayed = row_state_vecs[ki] * decay_vecs[ki]
                    row_state_vecs[ki] = decayed
                    sum_hk_vec = mlir_vector.FMAOp(
                        decayed,
                        k_vecs[ki],
                        sum_hk_vec,
                    ).result
                    sum_hq_vec = mlir_vector.FMAOp(
                        decayed,
                        q_vecs[ki],
                        sum_hq_vec,
                    ).result

                sum_hk = mlir_vector.ReductionOp(
                    T.f32,
                    vector.CombiningKind.ADD,
                    sum_hk_vec,
                ).dest
                sum_hq = mlir_vector.ReductionOp(
                    T.f32,
                    vector.CombiningKind.ADD,
                    sum_hq_vec,
                ).dest
                for offset in (1, 2, 4):
                    sum_hk = (
                        sum_hk
                        + mlir_gpu.ShuffleOp(
                            _to_raw(sum_hk),
                            _to_raw(fx.Int32(offset)),
                            _to_raw(width),
                            mode="xor",
                        ).shuffleResult
                    )
                    sum_hq = (
                        sum_hq
                        + mlir_gpu.ShuffleOp(
                            _to_raw(sum_hq),
                            _to_raw(fx.Int32(offset)),
                            _to_raw(width),
                            mode="xor",
                        ).shuffleResult
                    )

                conv_v = fx.Float32(fx.ptr_load(v_lds + global_v))
                v_new = (conv_v - fx.Float32(sum_hk)) * beta
                v_new = mlir_gpu.ShuffleOp(
                    _to_raw(v_new),
                    _to_raw(subgroup_leader),
                    _to_raw(width),
                    mode="idx",
                ).shuffleResult
                recurrent_value = fx.Float32(sum_hq) + fx.Float32(v_new) * fx.Float32(
                    dot_kq
                )
                v_new_vec = mlir_vector.BroadcastOp(
                    vec_f32,
                    _to_raw(v_new),
                ).vector

                for ki in range_constexpr(_K_ITERS):
                    updated = mlir_vector.FMAOp(
                        k_vecs[ki],
                        v_new_vec,
                        row_state_vecs[ki],
                    ).result
                    k_base = k_vec_start + fx.Int32(ki * _WARP_TILE_K)
                    state_off = state_head_base + global_v * fx.Int32(_DIM) + k_base
                    state.vec_store(
                        (state_off,),
                        updated,
                        _VALUES_PER_THREAD_K,
                    )

                if lane_k == fx.Int32(0):
                    fx.ptr_store(
                        fx.BFloat16(recurrent_value),
                        out_lds + global_v,
                    )
                rounded = fx.BFloat16(recurrent_value)
                rounded_f32 = fx.Float32(rounded)
                return rounded_f32 * rounded_f32

            norm_accum = fx.Float32(0.0)
            state_vecs = []
            for vi in range_constexpr(_V_ITERS):
                global_v = global_v_start + fx.Int32(vi * _V_GROUP_TILE)
                for ki in range_constexpr(_K_ITERS):
                    k_base = k_vec_start + fx.Int32(ki * _WARP_TILE_K)
                    state_off = state_head_base + global_v * fx.Int32(_DIM) + k_base
                    state_vecs.append(state.vec_load((state_off,), 4))
            for vi in range_constexpr(_V_ITERS):
                norm_accum = norm_accum + process_state_row(
                    vi,
                    state_vecs[vi * _K_ITERS : (vi + 1) * _K_ITERS],
                )

            if const_expr(fused_norm_reduce):
                for offset in (32, 16, 8, 4, 2, 1):
                    norm_accum = (
                        norm_accum
                        + mlir_gpu.ShuffleOp(
                            _to_raw(norm_accum),
                            _to_raw(fx.Int32(offset)),
                            _to_raw(width),
                            mode="xor",
                        ).shuffleResult
                    )
                if lane == fx.Int32(0):
                    fx.ptr_store(
                        norm_accum * fx.Float32(1.0 / _WARP_THREADS_K),
                        norm_lds + warp,
                    )

            fx.gpu.barrier()

            # Preserve the model's BF16 boundary before RMSNorm and gating.
            if const_expr(not fused_norm_reduce):
                output_if = scf.IfOp(
                    _to_raw(tid < fx.Int32(_DIM)),
                    results_=[],
                    has_else=False,
                )
                with ir.InsertionPoint(output_if.then_block):
                    recurrent_bf16 = fx.ptr_load(out_lds + tid)
                    recurrent_f32 = fx.Float32(recurrent_bf16)
                    square = recurrent_f32 * recurrent_f32
                    for offset in (32, 16, 8, 4, 2, 1):
                        square = (
                            square
                            + mlir_gpu.ShuffleOp(
                                _to_raw(square),
                                _to_raw(fx.Int32(offset)),
                                _to_raw(width),
                                mode="xor",
                            ).shuffleResult
                        )
                    if lane == fx.Int32(0):
                        fx.ptr_store(square, norm_lds + warp)
                    scf.YieldOp([])

                fx.gpu.barrier()

            output_store_if = scf.IfOp(
                _to_raw(tid < fx.Int32(_DIM)),
                results_=[],
                has_else=False,
            )
            with ir.InsertionPoint(output_store_if.then_block):
                norm_sum = fx.Float32(fx.ptr_load(norm_lds))
                norm_sum = norm_sum + fx.Float32(fx.ptr_load(norm_lds + fx.Int32(1)))
                if const_expr(fused_norm_reduce):
                    norm_sum = norm_sum + fx.Float32(
                        fx.ptr_load(norm_lds + fx.Int32(2))
                    )
                    norm_sum = norm_sum + fx.Float32(
                        fx.ptr_load(norm_lds + fx.Int32(3))
                    )
                inv_rms = fx.math.rsqrt(
                    norm_sum * fx.Float32(1.0 / _DIM) + fx.Float32(norm_eps)
                )
                recurrent_f32 = fx.Float32(fx.ptr_load(out_lds + tid))
                norm_w = fx.Float32(norm_weight[tid])
                gate_value = fx.Float32(
                    output_gate[
                        batch * stride_gate_token + head * stride_gate_head + tid
                    ]
                )
                output_sigmoid = fx.Float32(1.0) / (
                    fx.Float32(1.0) + fx.math.exp2(-gate_value * fx.Float32(_LOG2E))
                )
                result = recurrent_f32 * inv_rms * norm_w * output_sigmoid
                out.store(
                    batch * stride_out_token + head * stride_out_head + tid,
                    result.to(fx.BFloat16),
                )
                scf.YieldOp([])
            scf.YieldOp([])
        with ir.InsertionPoint(valid_if.else_block):
            zero_if = scf.IfOp(
                _to_raw(tid < fx.Int32(_DIM)),
                results_=[],
                has_else=False,
            )
            with ir.InsertionPoint(zero_if.then_block):
                out.store(
                    batch * stride_out_token + head * stride_out_head + tid,
                    fx.BFloat16(0.0),
                )
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch(
        f_a_mem: fx.Tensor,
        f_b_weight_mem: fx.Tensor,
        x_mem: fx.Tensor,
        weight_mem: fx.Tensor,
        conv_state_mem: fx.Tensor,
        raw_beta_mem: fx.Tensor,
        A_log_mem: fx.Tensor,
        dt_bias_mem: fx.Tensor,
        state_mem: fx.Tensor,
        state_indices_mem: fx.Tensor,
        output_gate_mem: fx.Tensor,
        norm_weight_mem: fx.Tensor,
        out_mem: fx.Tensor,
        batch_size: fx.Int32,
        stride_f_a_token: fx.Int32,
        stride_f_b_head: fx.Int32,
        stride_f_b_output: fx.Int32,
        stride_x_token: fx.Int32,
        stride_weight_channel: fx.Int32,
        stride_weight_width: fx.Int32,
        stride_conv_slot: fx.Int32,
        stride_conv_channel: fx.Int32,
        stride_conv_width: fx.Int32,
        stride_beta_token: fx.Int32,
        stride_state_slot: fx.Int32,
        stride_gate_token: fx.Int32,
        stride_gate_head: fx.Int32,
        stride_out_token: fx.Int32,
        stride_out_head: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(
            f_a_mem,
            f_b_weight_mem,
            x_mem,
            weight_mem,
            conv_state_mem,
            raw_beta_mem,
            A_log_mem,
            dt_bias_mem,
            state_mem,
            state_indices_mem,
            output_gate_mem,
            norm_weight_mem,
            out_mem,
            batch_size,
            stride_f_a_token,
            stride_f_b_head,
            stride_f_b_output,
            stride_x_token,
            stride_weight_channel,
            stride_weight_width,
            stride_conv_slot,
            stride_conv_channel,
            stride_conv_width,
            stride_beta_token,
            stride_state_slot,
            stride_gate_token,
            stride_gate_head,
            stride_out_token,
            stride_out_head,
        ).launch(
            grid=(batch_size * fx.Int32(_HEADS), 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch.compile_hints = {
        "waves_per_eu": waves_per_eu,
        "llvm_options": {
            "amdgpu-expert-scheduling-mode": True,
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch


__all__ = ["create_kimi_k3_kda_decode_fb_kernel"]
