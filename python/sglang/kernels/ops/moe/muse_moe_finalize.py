# SPDX-License-Identifier: MIT
"""Vectorized Muse per-expert RMSNorm followed by routing-weight reduction."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw


def build_splitk_silu(I, split_k):
    """Reduce FP32 gate/up K-split partials and materialize BF16 SiLU*up."""
    threads = 256
    vec = 4
    block_n = threads * vec
    nblocks = (I + block_n - 1) // block_n

    @flyc.kernel(name="muse_moe_splitk_silu_flydsl")
    def kernel(partials: fx.Tensor, out: fx.Tensor, rows: fx.Int32):
        pid = fx.block_idx.x
        tid = fx.thread_idx.x
        row = pid // nblocks
        col = (pid % nblocks) * block_n + tid * vec
        pr = buffer_ops.create_buffer_resource(
            partials,
            num_records_bytes=_to_raw(split_k * rows * 2 * I * 4),
        )
        outr = buffer_ops.create_buffer_resource(
            out, num_records_bytes=_to_raw(rows * I * 2)
        )
        gate = [arith.constant(0.0, type=T.f32) for _ in range(vec)]
        up = [arith.constant(0.0, type=T.f32) for _ in range(vec)]
        for sk in range_constexpr(split_k):
            base = sk * rows * (2 * I) + row * (2 * I) + col
            gate_raw = buffer_ops.buffer_load(
                pr, base, vec_width=vec, dtype=T.f32, mask=_to_raw(col < I)
            )
            up_raw = buffer_ops.buffer_load(
                pr, base + I, vec_width=vec, dtype=T.f32, mask=_to_raw(col < I)
            )
            gate_vec = fx.Vector(gate_raw, shape=vec, dtype=fx.Float32)
            up_vec = fx.Vector(up_raw, shape=vec, dtype=fx.Float32)
            gate = [arith.addf(gate[j], _to_raw(gate_vec[j])) for j in range(vec)]
            up = [arith.addf(up[j], _to_raw(up_vec[j])) for j in range(vec)]
        values = []
        for j in range_constexpr(vec):
            exp = fx.rocdl.exp2(
                T.f32,
                arith.mulf(
                    gate[j], arith.constant(-1.4426950408889634, type=T.f32)
                ),
            )
            sigmoid = arith.divf(
                arith.constant(1.0, type=T.f32),
                arith.addf(arith.constant(1.0, type=T.f32), exp),
            )
            values.append(
                arith.truncf(T.bf16, arith.mulf(arith.mulf(gate[j], sigmoid), up[j]))
            )
        packed = _to_raw(fx.Vector.from_elements(values, fx.BFloat16))
        buffer_ops.buffer_store(
            packed, outr, row * I + col, mask=_to_raw(col < I)
        )

    @flyc.jit
    def launch(
        partials: fx.Tensor,
        out: fx.Tensor,
        rows: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(partials, out, rows).launch(
            grid=(fx.Int64(grid_x), 1, 1), block=(threads, 1, 1), stream=stream
        )

    return launch


def build_finalize(H, topk, has_norm, has_scale, weight_dtype="bf16", split_k=1):
    threads = 256
    vec = 4 if H % 4 == 0 else 1
    chunks = (H + threads * vec - 1) // (threads * vec)
    elems = chunks * vec
    weight_bytes = 4 if weight_dtype == "fp32" else 2

    @fx.struct
    class Shared:
        # Give every routing slot its own four-wave reduction cells. This
        # removes the second barrier that was only needed before reusing the
        # same cells for the next slot.
        sums: fx.Array[fx.Float32, topk * 4, 16]

    def wave_sum(value):
        for shift in (32, 16, 8, 4, 2, 1):
            peer = _to_raw(ArithValue(value).shuffle_xor(shift, 64))
            value = arith.addf(value, peer)
        return value

    @flyc.kernel(name="muse_moe_finalize_vector_flydsl")
    def kernel(
        y: fx.Tensor,
        gamma: fx.Tensor,
        scores: fx.Tensor,
        ids: fx.Tensor,
        out: fx.Tensor,
        local_start: fx.Int32,
        num_local: fx.Int32,
        eps: fx.Float32,
        gain: fx.Float32,
        rows: fx.Int32,
    ):
        m = fx.block_idx.x
        tid = fx.thread_idx.x
        lane, wave = tid % 64, tid // 64
        zero = arith.constant(0.0, type=T.f32)
        one = arith.constant(1.0, type=T.f32)
        yr = buffer_ops.create_buffer_resource(
            y,
            num_records_bytes=_to_raw(
                split_k * rows * topk * H * (4 if split_k > 1 else 2)
            ),
        )
        wr = buffer_ops.create_buffer_resource(
            gamma, num_records_bytes=H * weight_bytes
        )
        sr = buffer_ops.create_buffer_resource(scores, max_size=True)
        ir = buffer_ops.create_buffer_resource(ids, max_size=True)
        outr = buffer_ops.create_buffer_resource(
            out, num_records_bytes=_to_raw(rows * H * 2)
        )
        lds = fx.SharedAllocator().allocate(Shared).peek()
        scales = []
        if fx.const_expr(has_norm and has_scale):
            wtype = (
                T.f32
                if weight_dtype == "fp32"
                else (T.f16 if weight_dtype == "fp16" else T.bf16)
            )
            numeric = (
                fx.Float32
                if weight_dtype == "fp32"
                else (fx.Float16 if weight_dtype == "fp16" else fx.BFloat16)
            )
            for chunk in range_constexpr(chunks):
                h = tid * vec + chunk * threads * vec
                raw = buffer_ops.buffer_load(
                    wr, h, vec_width=vec, dtype=wtype, mask=_to_raw(h < H)
                )
                values = fx.Vector(raw, shape=vec, dtype=numeric) if vec > 1 else [raw]
                for j in range_constexpr(vec):
                    v = _to_raw(values[j])
                    if fx.const_expr(weight_dtype != "fp32"):
                        v = arith.extf(T.f32, v)
                    v = arith.addf(v, _to_raw(gain))
                    if fx.const_expr(weight_dtype != "fp32"):
                        v = arith.extf(T.f32, arith.truncf(wtype, v))
                    scales.append(v)
        for slot_i, state in range(0, topk, 1, init=[zero for _ in range(elems)]):
            slot = m * topk + fx.Int32(arith.index_cast(T.i32, _to_raw(slot_i)))
            e = fx.Int32(buffer_ops.buffer_load(ir, slot, vec_width=1, dtype=T.i32))
            updated = _to_raw(fx.Vector.from_elements(state, fx.Float32))
            if (e >= local_start) & (e < local_start + num_local):
                sc = buffer_ops.buffer_load(sr, slot, vec_width=1, dtype=T.f32)
                values = []
                ss = zero
                for chunk in range_constexpr(chunks):
                    h = tid * vec + chunk * threads * vec
                    if fx.const_expr(split_k > 1):
                        partial = [zero for _ in range(vec)]
                        for sk in range_constexpr(split_k):
                            raw = buffer_ops.buffer_load(
                                yr,
                                sk * rows * topk * H + slot * H + h,
                                vec_width=vec,
                                dtype=T.f32,
                                mask=_to_raw(h < H),
                            )
                            vector = fx.Vector(raw, shape=vec, dtype=fx.Float32)
                            partial = [
                                arith.addf(partial[j], _to_raw(vector[j]))
                                for j in range(vec)
                            ]
                        vector = partial
                    else:
                        raw = buffer_ops.buffer_load(
                            yr,
                            slot * H + h,
                            vec_width=vec,
                            dtype=T.bf16,
                            mask=_to_raw(h < H),
                        )
                        vector = (
                            fx.Vector(raw, shape=vec, dtype=fx.BFloat16)
                            if vec > 1
                            else [raw]
                        )
                    for j in range_constexpr(vec):
                        value = (
                            arith.extf(
                                T.f32, arith.truncf(T.bf16, _to_raw(vector[j]))
                            )
                            if split_k > 1
                            else arith.extf(T.f32, _to_raw(vector[j]))
                        )
                        if fx.const_expr(has_norm and has_scale):
                            value = arith.extf(
                                T.f32,
                                arith.truncf(
                                    T.bf16, arith.mulf(value, scales[chunk * vec + j])
                                ),
                            )
                        values.append(value)
                        ss = arith.addf(ss, arith.mulf(value, value))
                if fx.const_expr(has_norm):
                    ss = wave_sum(ss)
                    slot_base = fx.Int32(
                        arith.index_cast(T.i32, _to_raw(slot_i))
                    ) * 4
                    if lane == 0:
                        lds.sums[slot_base + wave] = fx.Float32(ss)
                    fx.gpu.barrier()
                    total = zero
                    for w in range_constexpr(4):
                        total = arith.addf(total, _to_raw(lds.sums[slot_base + w]))
                    mean = arith.addf(
                        arith.mulf(total, arith.constant(1.0 / H, type=T.f32)),
                        _to_raw(eps),
                    )
                    sc = arith.mulf(sc, arith.divf(one, fx.rocdl.sqrt(T.f32, mean)))
                updated = _to_raw(
                    fx.Vector.from_elements(
                        [
                            arith.addf(state[j], arith.mulf(sc, values[j]))
                            for j in range(elems)
                        ],
                        fx.Float32,
                    )
                )
            next_values = [
                _to_raw(fx.Vector(updated, shape=elems, dtype=fx.Float32)[j])
                for j in range(elems)
            ]
            final = yield next_values
        for chunk in range_constexpr(chunks):
            h = tid * vec + chunk * threads * vec
            packed = [arith.truncf(T.bf16, final[chunk * vec + j]) for j in range(vec)]
            value = (
                _to_raw(fx.Vector.from_elements(packed, fx.BFloat16))
                if vec > 1
                else packed[0]
            )
            buffer_ops.buffer_store(value, outr, m * H + h, mask=_to_raw(h < H))

    @flyc.jit
    def launch(
        y: fx.Tensor,
        gamma: fx.Tensor,
        scores: fx.Tensor,
        ids: fx.Tensor,
        out: fx.Tensor,
        local_start: fx.Int32,
        num_local: fx.Int32,
        eps: fx.Float32,
        gain: fx.Float32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            y, gamma, scores, ids, out, local_start, num_local, eps, gain, grid_x
        ).launch(grid=(fx.Int64(grid_x), 1, 1), block=(threads, 1, 1), stream=stream)

    return launch
