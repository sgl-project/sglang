# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL kernel for the fused Kimi-K3 KDA decode path on gfx950."""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import vector
from aiter.ops.flydsl.kernels.tensor_shim import GTensor, _to_raw
from flydsl._mlir import ir
from flydsl._mlir.dialects import gpu as mlir_gpu
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects import vector as mlir_vector
from flydsl.expr import range_constexpr
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
_WAVES_PER_EU = 3


@functools.cache
def create_kimi_k3_kda_decode_kernel(norm_eps: float, lower_bound: float):
    """Build the fixed gfx950 BF16 Kimi-K3 decode specialization."""

    @fx.struct
    class SharedStorage:
        q: fx.Array[fx.BFloat16, _DIM, 16]
        k: fx.Array[fx.BFloat16, _DIM, 16]
        v: fx.Array[fx.BFloat16, _DIM, 16]
        recurrent_out: fx.Array[fx.BFloat16, _DIM, 16]
        norm_partial: fx.Array[fx.Float32, 2, 16]

    @flyc.kernel(
        name="kimi_k3_kda_decode_bf16_gfx950",
        known_block_size=[_BLOCK_THREADS, 1, 1],
    )
    def kernel(
        x_mem: fx.Tensor,
        weight_mem: fx.Tensor,
        conv_state_mem: fx.Tensor,
        raw_g_mem: fx.Tensor,
        raw_beta_mem: fx.Tensor,
        A_log_mem: fx.Tensor,
        dt_bias_mem: fx.Tensor,
        state_mem: fx.Tensor,
        state_indices_mem: fx.Tensor,
        output_gate_mem: fx.Tensor,
        norm_weight_mem: fx.Tensor,
        out_mem: fx.Tensor,
        batch_size: fx.Int32,
        stride_x_token: fx.Int32,
        stride_weight_channel: fx.Int32,
        stride_weight_width: fx.Int32,
        stride_conv_slot: fx.Int32,
        stride_conv_channel: fx.Int32,
        stride_conv_width: fx.Int32,
        stride_g_token: fx.Int32,
        stride_beta_token: fx.Int32,
        stride_state_slot: fx.Int32,
        stride_gate_token: fx.Int32,
        stride_gate_head: fx.Int32,
        stride_out_token: fx.Int32,
        stride_out_head: fx.Int32,
    ):
        del batch_size

        x = GTensor(x_mem, dtype=T.bf16, shape=(-1,))
        weight = GTensor(weight_mem, dtype=T.f32, shape=(-1,))
        conv_state = GTensor(conv_state_mem, dtype=T.bf16, shape=(-1,))
        raw_g = GTensor(raw_g_mem, dtype=T.bf16, shape=(-1,))
        raw_beta = GTensor(raw_beta_mem, dtype=T.bf16, shape=(-1,))
        A_log = GTensor(A_log_mem, dtype=T.f32, shape=(-1,))
        dt_bias = GTensor(dt_bias_mem, dtype=T.f32, shape=(-1,))
        state = GTensor(state_mem, dtype=T.f32, shape=(-1,))
        state_indices = GTensor(state_indices_mem, dtype=T.i32, shape=(-1,))
        output_gate = GTensor(output_gate_mem, dtype=T.bf16, shape=(-1,))
        norm_weight = GTensor(norm_weight_mem, dtype=T.bf16, shape=(-1,))
        out = GTensor(out_mem, dtype=T.bf16, shape=(-1,))

        shared = fx.SharedAllocator().allocate(SharedStorage).peek()
        q_lds = shared.q.ptr
        k_lds = shared.k.ptr
        v_lds = shared.v.ptr
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
            # A workgroup exclusively owns all three convolution channels for
            # its (batch, head), so every cache entry is shifted exactly once.
            conv_if = scf.IfOp(
                _to_raw(tid < fx.Int32(_DIM)),
                results_=[],
                has_else=False,
            )
            with ir.InsertionPoint(conv_if.then_block):
                channel_local = tid
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
                fx.ptr_store(q_conv, q_lds + tid)
                fx.ptr_store(k_conv, k_lds + tid)
                fx.ptr_store(v_conv, v_lds + tid)
                scf.YieldOp([])

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

                gate_bf16 = raw_g.vec_load(
                    (batch * stride_g_token + head * fx.Int32(_DIM) + k_base,),
                    _VALUES_PER_THREAD_K,
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

            state_vecs = []
            for vi in range_constexpr(_V_ITERS):
                global_v = global_v_start + fx.Int32(vi * _V_GROUP_TILE)
                for ki in range_constexpr(_K_ITERS):
                    k_base = k_vec_start + fx.Int32(ki * _WARP_TILE_K)
                    state_off = state_head_base + global_v * fx.Int32(_DIM) + k_base
                    state_vecs.append(
                        state.vec_load(
                            (state_off,),
                            _VALUES_PER_THREAD_K,
                        )
                    )

            for vi in range_constexpr(_V_ITERS):
                global_v = global_v_start + fx.Int32(vi * _V_GROUP_TILE)
                sum_hk_vec = zero_vec
                sum_hq_vec = zero_vec
                for ki in range_constexpr(_K_ITERS):
                    state_pos = vi * _K_ITERS + ki
                    decayed = state_vecs[state_pos] * decay_vecs[ki]
                    state_vecs[state_pos] = decayed
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
                    state_pos = vi * _K_ITERS + ki
                    updated = mlir_vector.FMAOp(
                        k_vecs[ki],
                        v_new_vec,
                        state_vecs[state_pos],
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

            fx.gpu.barrier()

            # Preserve the model's BF16 boundary before RMSNorm and gating.
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
        x_mem: fx.Tensor,
        weight_mem: fx.Tensor,
        conv_state_mem: fx.Tensor,
        raw_g_mem: fx.Tensor,
        raw_beta_mem: fx.Tensor,
        A_log_mem: fx.Tensor,
        dt_bias_mem: fx.Tensor,
        state_mem: fx.Tensor,
        state_indices_mem: fx.Tensor,
        output_gate_mem: fx.Tensor,
        norm_weight_mem: fx.Tensor,
        out_mem: fx.Tensor,
        batch_size: fx.Int32,
        stride_x_token: fx.Int32,
        stride_weight_channel: fx.Int32,
        stride_weight_width: fx.Int32,
        stride_conv_slot: fx.Int32,
        stride_conv_channel: fx.Int32,
        stride_conv_width: fx.Int32,
        stride_g_token: fx.Int32,
        stride_beta_token: fx.Int32,
        stride_state_slot: fx.Int32,
        stride_gate_token: fx.Int32,
        stride_gate_head: fx.Int32,
        stride_out_token: fx.Int32,
        stride_out_head: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(
            x_mem,
            weight_mem,
            conv_state_mem,
            raw_g_mem,
            raw_beta_mem,
            A_log_mem,
            dt_bias_mem,
            state_mem,
            state_indices_mem,
            output_gate_mem,
            norm_weight_mem,
            out_mem,
            batch_size,
            stride_x_token,
            stride_weight_channel,
            stride_weight_width,
            stride_conv_slot,
            stride_conv_channel,
            stride_conv_width,
            stride_g_token,
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
        "waves_per_eu": _WAVES_PER_EU,
        "llvm_options": {
            "amdgpu-expert-scheduling-mode": True,
        },
    }
    return launch


__all__ = ["create_kimi_k3_kda_decode_kernel"]
