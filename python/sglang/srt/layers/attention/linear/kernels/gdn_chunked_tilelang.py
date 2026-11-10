# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""SM70 TileLang kernels for the chunked gated-delta-rule forward pass.

The implementation follows the public FlashQLA block algorithm (MIT, Qwen
team) at the level of equations: solve the 64-token lower-triangular delta
system, then carry the recurrent state between chunks while tensor cores do
the dense intra-chunk work.  The kernels and integration here are native
SGLang code and have no FlashQLA or 1Cat runtime/build dependency.

Unlike the Hopper schedule that motivated the algorithm, this implementation
uses ordinary CTA synchronization, explicit shared-memory transposes and SM70
MMA shapes.  Persistent state is read and written directly in SGLang's indexed
``[slot, Hv, V, K]`` cache layout.
"""

import os
from functools import lru_cache

import tilelang
import tilelang.language as T
import torch
from sglang.kernels.ops.attention.fla.chunk_fwd import (
    chunk_gated_delta_rule_fwd_intra,
)
from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
from sglang.kernels.ops.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd

CHUNK_SIZE = 64
KKT_SHARED_ROWS = 96
KEY_DIM = 128
VALUE_DIM = 128
VALUE_BLOCK = 16
THREADS = 128
KKT_THREADS = 128
_LOG2_E = 1.4426950408889634
# The Cython adapter builds a normal CUDA shared object and calls it with raw
# tensor pointers.  On Volta this avoids roughly 0.2 ms of TVM-FFI launch
# overhead per helper, which is material for the multi-kernel chunked path.
_EXECUTION_BACKEND = "cython"

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    _PASS_CONFIGS[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    _PASS_CONFIGS[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False

_GATE_PASS_CONFIGS = dict(_PASS_CONFIGS)
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    _GATE_PASS_CONFIGS[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = False
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    _GATE_PASS_CONFIGS[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = True


@tilelang.jit(
    out_idx=[1, 2],
    execution_backend=_EXECUTION_BACKEND,
    pass_configs=_PASS_CONFIGS,
)
def _packed_qk_norm_kernel(
    q_heads: int,
    value_heads: int,
    block_tokens: int = 8,
):
    """Normalize row-strided packed Q/K into compact chunk workspaces."""

    tokens = T.dynamic("tokens")
    mixed_dim = 2 * q_heads * KEY_DIM + value_heads * VALUE_DIM

    @T.prim_func
    def main(
        MixedQKV: T.Tensor([tokens, mixed_dim], T.float16),
        Q: T.Tensor([1, tokens, q_heads, KEY_DIM], T.float16),
        K: T.Tensor([1, tokens, q_heads, KEY_DIM], T.float16),
    ):
        with T.Kernel(
            T.ceildiv(tokens, block_tokens),
            q_heads,
            threads=THREADS,
        ) as (token_block, head):
            q_values = T.alloc_fragment([block_tokens, KEY_DIM], T.float32)
            k_values = T.alloc_fragment([block_tokens, KEY_DIM], T.float32)
            q_squares = T.alloc_fragment([block_tokens, KEY_DIM], T.float32)
            k_squares = T.alloc_fragment([block_tokens, KEY_DIM], T.float32)
            q_norm = T.alloc_fragment([block_tokens], T.float32)
            k_norm = T.alloc_fragment([block_tokens], T.float32)
            token_start = token_block * block_tokens

            for row, d in T.Parallel(block_tokens, KEY_DIM):
                token = token_start + row
                q_value = T.if_then_else(
                    token < tokens,
                    T.cast(MixedQKV[token, head * KEY_DIM + d], T.float32),
                    0,
                )
                k_value = T.if_then_else(
                    token < tokens,
                    T.cast(
                        MixedQKV[
                            token,
                            q_heads * KEY_DIM + head * KEY_DIM + d,
                        ],
                        T.float32,
                    ),
                    0,
                )
                q_values[row, d] = q_value
                k_values[row, d] = k_value
                q_squares[row, d] = q_value * q_value
                k_squares[row, d] = k_value * k_value
            T.reduce_sum(q_squares, q_norm, dim=1)
            T.reduce_sum(k_squares, k_norm, dim=1)
            for row, d in T.Parallel(block_tokens, KEY_DIM):
                token = token_start + row
                if token < tokens:
                    Q[0, token, head, d] = T.cast(
                        q_values[row, d] / T.sqrt(q_norm[row] + 1e-6),
                        T.float16,
                    )
                    K[0, token, head, d] = T.cast(
                        k_values[row, d] / T.sqrt(k_norm[row] + 1e-6),
                        T.float16,
                    )

    return main


@tilelang.jit(
    out_idx=[1],
    execution_backend=_EXECUTION_BACKEND,
    pass_configs=_PASS_CONFIGS,
)
def _packed_v_copy_kernel(
    q_heads: int,
    value_heads: int,
    block_tokens: int = 8,
):
    """Copy row-strided packed V into the compact chunk workspace."""

    tokens = T.dynamic("tokens")
    mixed_dim = 2 * q_heads * KEY_DIM + value_heads * VALUE_DIM

    @T.prim_func
    def main(
        MixedQKV: T.Tensor([tokens, mixed_dim], T.float16),
        V: T.Tensor([1, tokens, value_heads, VALUE_DIM], T.float16),
    ):
        with T.Kernel(
            T.ceildiv(tokens, block_tokens),
            value_heads,
            threads=THREADS,
        ) as (token_block, head):
            token_start = token_block * block_tokens
            for row, d in T.Parallel(block_tokens, VALUE_DIM):
                token = token_start + row
                if token < tokens:
                    V[0, token, head, d] = MixedQKV[
                        token,
                        2 * q_heads * KEY_DIM + head * VALUE_DIM + d,
                    ]

    return main


@tilelang.jit(
    out_idx=[6, 7],
    execution_backend=_EXECUTION_BACKEND,
    pass_configs=_GATE_PASS_CONFIGS,
)
def _packed_gate_cumsum_kernel(value_heads: int, num_sequences: int):
    """Fuse sigmoid/softplus gating with a 64-token inclusive scan."""

    tokens = T.dynamic("tokens")
    sequences = num_sequences
    chunks = T.dynamic("chunks")

    @T.prim_func
    def main(
        GateA: T.Tensor([tokens, value_heads], T.float16),
        GateB: T.Tensor([tokens, value_heads], T.float16),
        ALog: T.Tensor([value_heads], T.float32),
        DtBias: T.Tensor([value_heads], T.float16),
        CuSeqLens: T.Tensor([sequences + 1], T.int32),
        ChunkIndices: T.Tensor([chunks, 2], T.int32),
        GateCumsum: T.Tensor([1, tokens, value_heads], T.float32),
        Beta: T.Tensor([1, tokens, value_heads], T.float32),
    ):
        with T.Kernel(chunks, value_heads, threads=64) as (
            chunk_id,
            head,
        ):
            gate_value = T.alloc_local([1], T.float32)
            shuffled = T.alloc_local([1], T.float32)
            first_warp_total = T.alloc_shared([1], T.float32)
            lane = T.get_thread_binding()
            warp_lane = lane % 32
            sequence = ChunkIndices[chunk_id, 0]
            local_chunk = ChunkIndices[chunk_id, 1]
            begin = CuSeqLens[sequence] + local_chunk * CHUNK_SIZE
            end = CuSeqLens[sequence + 1]
            token = begin + lane
            if token < end:
                gate_x = T.cast(GateA[token, head], T.float32) + T.cast(
                    DtBias[head], T.float32
                )
                softplus = T.if_then_else(
                    gate_x <= 20,
                    T.log(1.0 + T.exp(gate_x)),
                    gate_x,
                )
                gate_value[0] = -T.exp(ALog[head]) * softplus
                beta_x = T.cast(GateB[token, head], T.float32)
                # The established GDN contract rounds sigmoid(beta) through
                # the FP16 gate dtype before storing FP32.
                Beta[0, token, head] = T.cast(
                    T.cast(1.0 / (1.0 + T.exp(-beta_x)), T.float16),
                    T.float32,
                )
            else:
                gate_value[0] = 0

            # Each warp scans one 32-token half. One shared scalar carries the
            # first half's total into the second half, halving the dependent
            # transcendental/shuffle work per lane versus a one-warp schedule.
            for stage in T.unroll(5):
                offset = 1 << stage
                shuffled[0] = T.shfl_up(gate_value[0], offset)
                if warp_lane >= offset:
                    gate_value[0] += shuffled[0]
            if lane == 31:
                first_warp_total[0] = gate_value[0]
            T.sync_threads()
            if lane >= 32:
                gate_value[0] += first_warp_total[0]
            if token < end:
                GateCumsum[0, token, head] = gate_value[0]

    return main


@tilelang.jit(
    out_idx=[4, 5],
    execution_backend=_EXECUTION_BACKEND,
    pass_configs=_GATE_PASS_CONFIGS,
)
def _packed_gate_raw_kernel(value_heads: int):
    """Compute raw log-decay and beta with a dense elementwise launch."""

    tokens = T.dynamic("tokens")
    threads = 256

    @T.prim_func
    def main(
        GateA: T.Tensor([tokens, value_heads], T.float16),
        GateB: T.Tensor([tokens, value_heads], T.float16),
        ALog: T.Tensor([value_heads], T.float32),
        DtBias: T.Tensor([value_heads], T.float16),
        Gate: T.Tensor([1, tokens, value_heads], T.float32),
        Beta: T.Tensor([1, tokens, value_heads], T.float32),
    ):
        with T.Kernel(
            T.ceildiv(tokens * value_heads, threads), threads=threads
        ) as block:
            index = block * threads + T.get_thread_binding()
            if index < tokens * value_heads:
                token = T.floordiv(index, value_heads)
                head = index % value_heads
                gate_x = T.cast(GateA[token, head], T.float32) + T.cast(
                    DtBias[head], T.float32
                )
                softplus = T.if_then_else(
                    gate_x <= 20,
                    T.log(1.0 + T.exp(gate_x)),
                    gate_x,
                )
                Gate[0, token, head] = -T.exp(ALog[head]) * softplus
                beta_x = T.cast(GateB[token, head], T.float32)
                Beta[0, token, head] = T.cast(
                    T.cast(1.0 / (1.0 + T.exp(-beta_x)), T.float16),
                    T.float32,
                )

    return main


@tilelang.jit(
    out_idx=[10],
    execution_backend=_EXECUTION_BACKEND,
    pass_configs=_GATE_PASS_CONFIGS,
)
def _packed_recurrent_kernel(
    q_heads: int,
    value_heads: int,
    num_sequences: int,
    state_slots: int,
    column_groups_per_block: int,
    store_checkpoints: bool,
    state_fp32: bool,
):
    """Persistent 16-lane SM70 GDN recurrence over four value columns."""

    tokens = T.dynamic("tokens")
    chunks = T.dynamic("chunks")
    mixed_dim = 2 * q_heads * KEY_DIM + value_heads * VALUE_DIM
    subgroup_width = 16
    columns = 4
    rows_per_lane = KEY_DIM // subgroup_width
    threads = 32 * column_groups_per_block
    subgroups_per_cta = threads // subgroup_width
    value_ctas = (VALUE_DIM + subgroups_per_cta * columns - 1) // (
        subgroups_per_cta * columns
    )
    heads_per_key = value_heads // q_heads
    state_dtype = T.float32 if state_fp32 else T.float16

    @T.prim_func
    def main(
        Q: T.Tensor([1, tokens, q_heads, KEY_DIM], T.float16),
        K: T.Tensor([1, tokens, q_heads, KEY_DIM], T.float16),
        MixedQKV: T.Tensor([tokens, mixed_dim], T.float16),
        Gate: T.Tensor([1, tokens, value_heads], T.float32),
        Beta: T.Tensor([1, tokens, value_heads], T.float32),
        State: T.Tensor([state_slots, value_heads, VALUE_DIM, KEY_DIM], state_dtype),
        StateIndices: T.Tensor([num_sequences], T.int32),
        CuSeqLens: T.Tensor([num_sequences + 1], T.int32),
        ChunkOffsets: T.Tensor([num_sequences + 1], T.int32),
        Scale: T.float32,
        Output: T.Tensor([1, tokens, value_heads, VALUE_DIM], T.float16),
        Checkpoints: T.Tensor([1, chunks, value_heads, VALUE_DIM, KEY_DIM], T.float16),
    ):
        with T.Kernel(
            value_ctas,
            value_heads,
            num_sequences,
            threads=threads,
        ) as (value_cta, value_head, sequence):
            state = T.alloc_local([columns, rows_per_lane], T.float32)
            q_values = T.alloc_local([rows_per_lane], T.float32)
            k_values = T.alloc_local([rows_per_lane], T.float32)
            projection = T.alloc_local([columns], T.float32)
            delta = T.alloc_local([columns], T.float32)
            result = T.alloc_local([columns], T.float32)
            shuffled = T.alloc_local([1], T.float32)
            decay_value = T.alloc_local([1], T.float32)
            beta_value = T.alloc_local([1], T.float32)

            thread = T.get_thread_binding()
            subgroup = T.floordiv(thread, subgroup_width)
            lane = thread % subgroup_width
            column_group = value_cta * subgroups_per_cta + subgroup
            value_start = column_group * columns
            key_head = T.floordiv(value_head, heads_per_key)
            slot = StateIndices[sequence]
            begin = CuSeqLens[sequence]
            end = CuSeqLens[sequence + 1]
            T.assume(begin >= 0)
            T.assume(end >= begin)
            T.assume(end <= tokens)

            for column in T.unroll(columns):
                for row_iter in T.unroll(rows_per_lane):
                    row = row_iter * subgroup_width + lane
                    state[column, row_iter] = T.if_then_else(
                        (slot >= 0) & (value_start + column < VALUE_DIM),
                        State[
                            slot,
                            value_head,
                            value_start + column,
                            row,
                        ],
                        0,
                    )

            if slot >= 0:
                for local_token in T.serial(end - begin):
                    token = begin + local_token
                    if store_checkpoints:
                        if local_token % CHUNK_SIZE == 0:
                            checkpoint = ChunkOffsets[sequence] + T.floordiv(
                                local_token, CHUNK_SIZE
                            )
                            for column in T.unroll(columns):
                                for row_iter in T.unroll(rows_per_lane):
                                    if value_start + column < VALUE_DIM:
                                        Checkpoints[
                                            0,
                                            checkpoint,
                                            value_head,
                                            value_start + column,
                                            row_iter * subgroup_width + lane,
                                        ] = T.cast(state[column, row_iter], T.float16)

                    for row_iter in T.unroll(rows_per_lane):
                        row = row_iter * subgroup_width + lane
                        q_values[row_iter] = T.cast(
                            Q[0, token, key_head, row], T.float32
                        )
                        k_values[row_iter] = T.cast(
                            K[0, token, key_head, row], T.float32
                        )

                    # Gate/beta are identical for every value-column subgroup.
                    # Load and evaluate them once per physical warp, then
                    # broadcast.  On Volta, redundantly issuing exp2 and two
                    # global loads from every lane lengthens the dependency
                    # chain enough to dominate this recurrent schedule.
                    if thread % 32 == 0:
                        decay_value[0] = T.exp2(Gate[0, token, value_head] * _LOG2_E)
                        beta_value[0] = Beta[0, token, value_head]
                    decay_value[0] = T.shfl_sync(0xFFFFFFFF, decay_value[0], 0, 32)
                    beta_value[0] = T.shfl_sync(0xFFFFFFFF, beta_value[0], 0, 32)

                    # Accumulate all four value columns while each K element
                    # is live.  This ordering creates independent FFMA chains
                    # for Volta to interleave and avoids repeatedly walking the
                    # Q/K register vectors once per column.
                    for column in T.unroll(columns):
                        projection[column] = 0
                    for row_iter in T.unroll(rows_per_lane):
                        for column in T.unroll(columns):
                            projection[column] += (
                                state[column, row_iter] * k_values[row_iter]
                            )
                    for column in T.unroll(columns):
                        for stage in T.unroll(4):
                            offset = 8 >> stage
                            shuffled[0] = T.shfl_down(projection[column], offset)
                            if lane < offset:
                                projection[column] += shuffled[0]
                        if lane == 0:
                            delta[column] = T.if_then_else(
                                value_start + column < VALUE_DIM,
                                (
                                    T.cast(
                                        MixedQKV[
                                            token,
                                            2 * q_heads * KEY_DIM
                                            + value_head * VALUE_DIM
                                            + value_start
                                            + column,
                                        ],
                                        T.float32,
                                    )
                                    - decay_value[0] * projection[column]
                                )
                                * beta_value[0],
                                0,
                            )
                        delta[column] = T.shfl_sync(
                            0xFFFFFFFF,
                            delta[column],
                            0,
                            subgroup_width,
                        )
                    for column in T.unroll(columns):
                        result[column] = 0
                    for row_iter in T.unroll(rows_per_lane):
                        for column in T.unroll(columns):
                            state[column, row_iter] = (
                                decay_value[0] * state[column, row_iter]
                                + delta[column] * k_values[row_iter]
                            )
                            result[column] += (
                                state[column, row_iter] * q_values[row_iter]
                            )
                    for column in T.unroll(columns):
                        for stage in T.unroll(4):
                            offset = 8 >> stage
                            shuffled[0] = T.shfl_down(result[column], offset)
                            if lane < offset:
                                result[column] += shuffled[0]
                        if (lane == 0) & (value_start + column < VALUE_DIM):
                            Output[
                                0,
                                token,
                                value_head,
                                value_start + column,
                            ] = T.cast(result[column] * Scale, T.float16)

                for column in T.unroll(columns):
                    for row_iter in T.unroll(rows_per_lane):
                        if value_start + column < VALUE_DIM:
                            State[
                                slot,
                                value_head,
                                value_start + column,
                                row_iter * subgroup_width + lane,
                            ] = state[column, row_iter]
            else:
                if lane == 0:
                    for local_token in T.serial(end - begin):
                        for column in T.unroll(columns):
                            if value_start + column < VALUE_DIM:
                                Output[
                                    0,
                                    begin + local_token,
                                    value_head,
                                    value_start + column,
                                ] = 0

    return main


@tilelang.jit(
    out_idx=[4],
    execution_backend=_EXECUTION_BACKEND,
    pass_configs=_PASS_CONFIGS,
)
def _kkt_raw_kernel(q_heads: int, value_heads: int, num_sequences: int):
    """Build the raw strictly-lower ``beta * K K^T`` matrix."""

    tokens = T.dynamic("tokens")
    sequences = num_sequences
    chunks = T.dynamic("chunks")
    heads_per_key = value_heads // q_heads

    @T.prim_func
    def main(
        K: T.Tensor([1, tokens, q_heads, KEY_DIM], T.float16),
        Beta: T.Tensor([1, tokens, value_heads], T.float32),
        CuSeqLens: T.Tensor([sequences + 1], T.int32),
        ChunkIndices: T.Tensor([chunks, 2], T.int32),
        Raw: T.Tensor([1, tokens, value_heads, CHUNK_SIZE], T.float32),
    ):
        with T.Kernel(chunks, value_heads, threads=KKT_THREADS) as (
            chunk_id,
            value_head,
        ):
            k_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
            dot = T.alloc_fragment([CHUNK_SIZE, CHUNK_SIZE], T.float32)

            sequence = ChunkIndices[chunk_id, 0]
            local_chunk = ChunkIndices[chunk_id, 1]
            begin = CuSeqLens[sequence] + local_chunk * CHUNK_SIZE
            end = CuSeqLens[sequence + 1]
            key_head = T.floordiv(value_head, heads_per_key)

            for i, d in T.Parallel(CHUNK_SIZE, KEY_DIM):
                if begin + i < end:
                    k_shared[i, d] = K[0, begin + i, key_head, d]
                else:
                    k_shared[i, d] = 0
            T.clear(dot)
            T.gemm(
                k_shared,
                k_shared,
                dot,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )

            for i, j in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                if begin + i < end:
                    Raw[0, begin + i, value_head, j] = T.if_then_else(
                        j < i,
                        T.cast(Beta[0, begin + i, value_head], T.float32) * dot[i, j],
                        0,
                    )

    return main


@tilelang.jit(
    out_idx=[4],
    execution_backend=_EXECUTION_BACKEND,
    pass_configs=_PASS_CONFIGS,
)
def _kkt_inverse_kernel(q_heads: int, value_heads: int, num_sequences: int):
    """Fuse KKT construction and SM70-friendly 16/32/64 inversion."""

    tokens = T.dynamic("tokens")
    sequences = num_sequences
    chunks = T.dynamic("chunks")
    heads_per_key = value_heads // q_heads

    @T.macro
    def merge_diagonal_pair(
        base,
        lower_shared,
        inverse_shared,
        left_shared,
        right_shared,
        offdiag_shared,
        middle_shared,
        first,
        second,
    ):
        T.clear(left_shared)
        T.clear(right_shared)
        T.clear(offdiag_shared)
        for i, j in T.Parallel(16, 16):
            left_shared[i, j] = T.cast(inverse_shared[base + i, base + j], T.float16)
            right_shared[i, j] = T.cast(
                inverse_shared[base + 16 + i, base + 16 + j],
                T.float16,
            )
            offdiag_shared[i, j] = T.cast(
                lower_shared[base + 16 + i, base + j], T.float16
            )
        T.sync_threads()
        T.clear(first)
        T.gemm(
            right_shared,
            offdiag_shared,
            first,
            policy=T.GemmWarpPolicy.FullRow,
        )
        for i, j in T.Parallel(32, 32):
            middle_shared[i, j] = T.cast(first[i, j], T.float16)
        T.sync_threads()
        T.clear(second)
        T.gemm(
            middle_shared,
            left_shared,
            second,
            policy=T.GemmWarpPolicy.FullRow,
        )
        for i, j in T.Parallel(16, 16):
            inverse_shared[base + 16 + i, base + j] = -second[i, j]
        T.sync_threads()

    @T.prim_func
    def main(
        K: T.Tensor([1, tokens, q_heads, KEY_DIM], T.float16),
        Beta: T.Tensor([1, tokens, value_heads], T.float32),
        CuSeqLens: T.Tensor([sequences + 1], T.int32),
        ChunkIndices: T.Tensor([chunks, 2], T.int32),
        Inverse: T.Tensor([1, tokens, value_heads, CHUNK_SIZE], T.float16),
    ):
        with T.Kernel(chunks, value_heads, threads=KKT_THREADS) as (
            chunk_id,
            value_head,
        ):
            k_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
            # TileLang's SM70 linear-layout lowering vectorizes some 32x32
            # fragment transfers beyond the logical 64-row tile. Keep an
            # explicit physical tail so those accesses cannot alias the next
            # shared allocation. Only the first CHUNK_SIZE rows are logical.
            lower_shared = T.alloc_shared([KKT_SHARED_ROWS, CHUNK_SIZE + 1], T.float32)
            inverse_shared = T.alloc_shared(
                [KKT_SHARED_ROWS, CHUNK_SIZE + 1], T.float32
            )
            left_shared = T.alloc_shared([32, 32], T.float16)
            right_shared = T.alloc_shared([32, 32], T.float16)
            offdiag_shared = T.alloc_shared([32, 32], T.float16)
            middle_shared = T.alloc_shared([32, 32], T.float16)
            dot = T.alloc_fragment([CHUNK_SIZE, CHUNK_SIZE], T.float32)
            first = T.alloc_fragment([32, 32], T.float32)
            second = T.alloc_fragment([32, 32], T.float32)
            T.annotate_layout(
                {
                    lower_shared: tilelang.layout.make_linear_layout(lower_shared),
                    inverse_shared: tilelang.layout.make_linear_layout(inverse_shared),
                }
            )

            sequence = ChunkIndices[chunk_id, 0]
            local_chunk = ChunkIndices[chunk_id, 1]
            begin = CuSeqLens[sequence] + local_chunk * CHUNK_SIZE
            end = CuSeqLens[sequence + 1]
            key_head = T.floordiv(value_head, heads_per_key)
            for i, d in T.Parallel(CHUNK_SIZE, KEY_DIM):
                k_shared[i, d] = T.if_then_else(
                    begin + i < end,
                    K[0, begin + i, key_head, d],
                    0,
                )
            T.clear(dot)
            T.gemm(
                k_shared,
                k_shared,
                dot,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for i, j in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                lower_shared[i, j] = T.if_then_else(
                    (j < i) & (begin + i < end),
                    T.cast(Beta[0, begin + i, value_head], T.float32) * dot[i, j],
                    0,
                )
                inverse_shared[i, j] = 0
            T.sync_threads()

            # Four independent 16x16 forward substitutions keep the scalar
            # dependency chain short. The remaining work is tensor-core block
            # multiplication, where Volta is strongest.
            for block, column in T.Parallel(4, 16):
                inverse_shared[block * 16, block * 16 + column] = T.if_then_else(
                    column == 0, 1, 0
                )
            T.sync_threads()
            for row in T.serial(1, 16):
                for block, column in T.Parallel(4, 16):
                    inverse_shared[block * 16 + row, block * 16 + column] = 0
                T.sync_threads()
                for inner in T.serial(row):
                    for block, column in T.Parallel(4, 16):
                        inverse_shared[block * 16 + row, block * 16 + column] -= (
                            lower_shared[block * 16 + row, block * 16 + inner]
                            * inverse_shared[block * 16 + inner, block * 16 + column]
                        )
                    T.sync_threads()
                for block, column in T.Parallel(4, 16):
                    inverse_shared[block * 16 + row, block * 16 + column] = (
                        T.if_then_else(
                            column < row,
                            inverse_shared[block * 16 + row, block * 16 + column],
                            T.if_then_else(column == row, 1, 0),
                        )
                    )
                T.sync_threads()

            # Merge 16->32 independently for the upper and lower halves.
            merge_diagonal_pair(
                0,
                lower_shared,
                inverse_shared,
                left_shared,
                right_shared,
                offdiag_shared,
                middle_shared,
                first,
                second,
            )
            merge_diagonal_pair(
                32,
                lower_shared,
                inverse_shared,
                left_shared,
                right_shared,
                offdiag_shared,
                middle_shared,
                first,
                second,
            )

            # Merge the two complete 32x32 halves. For a block-lower matrix,
            # the off-diagonal inverse is -R^-1 * A_rl * L^-1.
            for i, j in T.Parallel(32, 32):
                left_shared[i, j] = T.cast(inverse_shared[i, j], T.float16)
                right_shared[i, j] = T.cast(inverse_shared[32 + i, 32 + j], T.float16)
                offdiag_shared[i, j] = T.cast(lower_shared[32 + i, j], T.float16)
            T.sync_threads()
            T.clear(first)
            T.gemm(
                right_shared,
                offdiag_shared,
                first,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for i, j in T.Parallel(32, 32):
                middle_shared[i, j] = T.cast(first[i, j], T.float16)
            T.sync_threads()
            T.clear(second)
            T.gemm(
                middle_shared,
                left_shared,
                second,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for i, j in T.Parallel(32, 32):
                inverse_shared[32 + i, j] = -second[i, j]
            T.sync_threads()

            for i, j in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                if begin + i < end:
                    Inverse[0, begin + i, value_head, j] = T.cast(
                        inverse_shared[i, j], T.float16
                    )

    return main


@tilelang.jit(
    execution_backend=_EXECUTION_BACKEND,
    pass_configs=_PASS_CONFIGS,
)
def _chunk_forward_kernel(
    q_heads: int,
    value_heads: int,
    num_sequences: int,
    state_slots: int,
    store_checkpoints: bool,
    state_fp32: bool,
    value_block: int = VALUE_BLOCK,
    packed_v: bool = False,
    k_reuse_mode: int = 0,
):
    if value_block not in (16, 32, 64) or VALUE_DIM % value_block != 0:
        raise ValueError("value_block must be 16, 32, or 64 and divide VALUE_DIM")
    if k_reuse_mode not in (0, 1, 2, 3, 4):
        raise ValueError("k_reuse_mode must be between 0 and 4")
    tokens = T.dynamic("tokens")
    sequences = num_sequences
    chunks = T.dynamic("chunks")
    heads_per_key = value_heads // q_heads
    state_dtype = T.float32 if state_fp32 else T.float16
    mixed_dim = 2 * q_heads * KEY_DIM + value_heads * VALUE_DIM
    v_shape = [tokens, mixed_dim] if packed_v else [1, tokens, value_heads, VALUE_DIM]

    @T.prim_func
    def main(
        Q: T.Tensor([1, tokens, q_heads, KEY_DIM], T.float16),
        K: T.Tensor([1, tokens, q_heads, KEY_DIM], T.float16),
        V: T.Tensor(v_shape, T.float16),
        Inverse: T.Tensor([1, tokens, value_heads, CHUNK_SIZE], T.float16),
        GateCumsum: T.Tensor([1, tokens, value_heads], T.float32),
        Beta: T.Tensor([1, tokens, value_heads], T.float32),
        State: T.Tensor([state_slots, value_heads, VALUE_DIM, KEY_DIM], state_dtype),
        StateIndices: T.Tensor([sequences], T.int32),
        CuSeqLens: T.Tensor([sequences + 1], T.int32),
        ChunkOffsets: T.Tensor([sequences + 1], T.int32),
        Scale: T.float32,
        Output: T.Tensor([1, tokens, value_heads, VALUE_DIM], T.float16),
        Checkpoints: T.Tensor([1, chunks, value_heads, VALUE_DIM, KEY_DIM], T.float16),
    ):
        with T.Kernel(
            T.ceildiv(VALUE_DIM, value_block),
            value_heads,
            sequences,
            threads=THREADS,
        ) as (value_tile, value_head, sequence):
            q_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
            if k_reuse_mode == 0:
                k_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
                k_dot_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
                k_update_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
            elif k_reuse_mode == 1:
                k_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
                k_dot_shared = k_shared
                k_update_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
            elif k_reuse_mode == 2:
                k_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
                k_dot_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
                k_update_shared = k_shared
            elif k_reuse_mode == 3:
                k_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
                k_dot_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
                k_update_shared = k_dot_shared
            else:
                k_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
                k_dot_shared = k_shared
                k_update_shared = k_shared
            value_shared = T.alloc_shared([CHUNK_SIZE, value_block], T.float16)
            inverse_shared = T.alloc_shared([CHUNK_SIZE, CHUNK_SIZE], T.float16)
            state_shared = T.alloc_shared([KEY_DIM, value_block], T.float16)
            delta_shared = T.alloc_shared([CHUNK_SIZE, value_block], T.float16)
            delta_t_shared = T.alloc_shared([value_block, CHUNK_SIZE], T.float16)
            scores_shared = T.alloc_shared([CHUNK_SIZE, CHUNK_SIZE], T.float16)
            gate_shared = T.alloc_shared([CHUNK_SIZE], T.float32)
            beta_shared = T.alloc_shared([CHUNK_SIZE], T.float32)

            state = T.alloc_fragment([value_block, KEY_DIM], T.float32)
            prediction = T.alloc_fragment([CHUNK_SIZE, value_block], T.float32)
            corrected = T.alloc_fragment([CHUNK_SIZE, value_block], T.float32)
            output = T.alloc_fragment([CHUNK_SIZE, value_block], T.float32)
            scores = T.alloc_fragment([CHUNK_SIZE, CHUNK_SIZE], T.float32)

            slot = StateIndices[sequence]
            begin = CuSeqLens[sequence]
            end = CuSeqLens[sequence + 1]
            num_chunks = T.ceildiv(end - begin, CHUNK_SIZE)
            key_head = T.floordiv(value_head, heads_per_key)
            value_start = value_tile * value_block

            for dv, d in T.Parallel(value_block, KEY_DIM):
                state[dv, d] = T.if_then_else(
                    (slot >= 0) & (value_start + dv < VALUE_DIM),
                    State[slot, value_head, value_start + dv, d],
                    0,
                )

            if slot >= 0:
                for chunk in T.serial(num_chunks):
                    chunk_begin = begin + chunk * CHUNK_SIZE
                    valid = T.min(CHUNK_SIZE, end - chunk_begin)

                    if store_checkpoints:
                        checkpoint = ChunkOffsets[sequence] + chunk
                        for dv, d in T.Parallel(value_block, KEY_DIM):
                            if value_start + dv < VALUE_DIM:
                                Checkpoints[
                                    0,
                                    checkpoint,
                                    value_head,
                                    value_start + dv,
                                    d,
                                ] = T.cast(state[dv, d], T.float16)

                    for i, d in T.Parallel(CHUNK_SIZE, KEY_DIM):
                        if i < valid:
                            q_shared[i, d] = Q[0, chunk_begin + i, key_head, d]
                            k_shared[i, d] = K[0, chunk_begin + i, key_head, d]
                            if k_reuse_mode not in (1, 4):
                                k_dot_shared[i, d] = K[0, chunk_begin + i, key_head, d]
                            if k_reuse_mode not in (2, 3, 4):
                                k_update_shared[i, d] = K[
                                    0, chunk_begin + i, key_head, d
                                ]
                        else:
                            q_shared[i, d] = 0
                            k_shared[i, d] = 0
                            if k_reuse_mode not in (1, 4):
                                k_dot_shared[i, d] = 0
                            if k_reuse_mode not in (2, 3, 4):
                                k_update_shared[i, d] = 0
                    for i, dv in T.Parallel(CHUNK_SIZE, value_block):
                        if packed_v:
                            value_shared[i, dv] = T.if_then_else(
                                (i < valid) & (value_start + dv < VALUE_DIM),
                                V[
                                    chunk_begin + i,
                                    2 * q_heads * KEY_DIM
                                    + value_head * VALUE_DIM
                                    + value_start
                                    + dv,
                                ],
                                0,
                            )
                        else:
                            value_shared[i, dv] = T.if_then_else(
                                (i < valid) & (value_start + dv < VALUE_DIM),
                                V[
                                    0,
                                    chunk_begin + i,
                                    value_head,
                                    value_start + dv,
                                ],
                                0,
                            )
                    for i, j in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                        inverse_shared[i, j] = T.if_then_else(
                            i < valid,
                            Inverse[0, chunk_begin + i, value_head, j],
                            0,
                        )
                    for i in T.Parallel(CHUNK_SIZE):
                        gate_shared[i] = T.if_then_else(
                            i < valid,
                            GateCumsum[0, chunk_begin + i, value_head],
                            GateCumsum[0, chunk_begin + valid - 1, value_head],
                        )
                        beta_shared[i] = T.if_then_else(
                            i < valid,
                            T.cast(
                                Beta[0, chunk_begin + i, value_head],
                                T.float32,
                            ),
                            0,
                        )
                    for d, dv in T.Parallel(KEY_DIM, value_block):
                        state_shared[d, dv] = T.cast(state[dv, d], T.float16)
                    T.sync_threads()

                    # W = V - exp(g) K H.
                    T.clear(prediction)
                    T.gemm(
                        k_dot_shared,
                        state_shared,
                        prediction,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, dv in T.Parallel(CHUNK_SIZE, value_block):
                        prediction[i, dv] = (
                            T.cast(value_shared[i, dv], T.float32)
                            - T.exp2(gate_shared[i] * _LOG2_E) * prediction[i, dv]
                        )
                        delta_shared[i, dv] = T.cast(prediction[i, dv], T.float16)
                    T.sync_threads()

                    # Apply the solved intra-chunk delta transform.
                    for i, j in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                        scores_shared[i, j] = T.cast(
                            T.if_then_else(
                                (j <= i) & (i < valid),
                                T.cast(inverse_shared[i, j], T.float32)
                                * T.exp2((gate_shared[i] - gate_shared[j]) * _LOG2_E)
                                * beta_shared[j],
                                0,
                            ),
                            T.float16,
                        )
                    T.sync_threads()
                    T.clear(corrected)
                    T.gemm(
                        scores_shared,
                        delta_shared,
                        corrected,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, dv in T.Parallel(CHUNK_SIZE, value_block):
                        delta_shared[i, dv] = T.cast(corrected[i, dv], T.float16)
                    T.sync_threads()

                    # O = scale * (exp(g) QH + (G * QK^T) V_delta).
                    T.clear(output)
                    T.gemm(
                        q_shared,
                        state_shared,
                        output,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.clear(scores)
                    T.gemm(
                        q_shared,
                        k_update_shared,
                        scores,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, j in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                        scores[i, j] = T.if_then_else(
                            (j <= i) & (i < valid),
                            scores[i, j]
                            * T.exp2((gate_shared[i] - gate_shared[j]) * _LOG2_E)
                            * Scale,
                            0,
                        )
                        scores_shared[i, j] = T.cast(scores[i, j], T.float16)
                    for i, dv in T.Parallel(CHUNK_SIZE, value_block):
                        output[i, dv] *= Scale * T.exp2(gate_shared[i] * _LOG2_E)
                    T.sync_threads()
                    T.gemm(
                        scores_shared,
                        delta_shared,
                        output,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, dv in T.Parallel(CHUNK_SIZE, value_block):
                        if (i < valid) & (value_start + dv < VALUE_DIM):
                            Output[
                                0,
                                chunk_begin + i,
                                value_head,
                                value_start + dv,
                            ] = T.cast(output[i, dv], T.float16)

                    # H' = exp(g_last) H + V_delta^T diag(exp(g_last-g)) K.
                    for dv, i in T.Parallel(value_block, CHUNK_SIZE):
                        delta_t_shared[dv, i] = T.cast(
                            T.cast(delta_shared[i, dv], T.float32)
                            * T.exp2(
                                (gate_shared[valid - 1] - gate_shared[i]) * _LOG2_E
                            ),
                            T.float16,
                        )
                    last_decay = T.exp2(gate_shared[valid - 1] * _LOG2_E)
                    for dv, d in T.Parallel(value_block, KEY_DIM):
                        state[dv, d] *= last_decay
                    T.gemm(
                        delta_t_shared,
                        k_shared,
                        state,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for dv, d in T.Parallel(value_block, KEY_DIM):
                    if value_start + dv < VALUE_DIM:
                        State[slot, value_head, value_start + dv, d] = state[dv, d]

    return main


@tilelang.jit(
    execution_backend=_EXECUTION_BACKEND,
    pass_configs=_PASS_CONFIGS,
)
def _state_output_kernel(
    q_heads: int,
    value_heads: int,
    num_sequences: int,
    state_slots: int,
    store_checkpoints: bool,
    state_fp32: bool,
    value_block: int = VALUE_BLOCK,
):
    """Fuse the inter-chunk state recurrence with the output projection.

    ``W`` and ``U`` use the standard FLA WY representation.  Combining the
    formerly separate state and output kernels keeps each state tile resident
    and avoids writing the per-chunk state tensor to HBM unless prefix-cache
    checkpointing actually asks for it.
    """

    tokens = T.dynamic("tokens")
    sequences = num_sequences
    chunks = T.dynamic("chunks")
    heads_per_key = value_heads // q_heads
    state_dtype = T.float32 if state_fp32 else T.float16

    @T.prim_func
    def main(
        Q: T.Tensor([1, tokens, q_heads, KEY_DIM], T.float16),
        K: T.Tensor([1, tokens, q_heads, KEY_DIM], T.float16),
        W: T.Tensor([1, tokens, value_heads, KEY_DIM], T.float16),
        U: T.Tensor([1, tokens, value_heads, VALUE_DIM], T.float16),
        GateCumsum: T.Tensor([1, tokens, value_heads], T.float32),
        State: T.Tensor([state_slots, value_heads, VALUE_DIM, KEY_DIM], state_dtype),
        StateIndices: T.Tensor([sequences], T.int32),
        CuSeqLens: T.Tensor([sequences + 1], T.int32),
        ChunkOffsets: T.Tensor([sequences + 1], T.int32),
        Scale: T.float32,
        Output: T.Tensor([1, tokens, value_heads, VALUE_DIM], T.float16),
        Checkpoints: T.Tensor([1, chunks, value_heads, VALUE_DIM, KEY_DIM], T.float16),
    ):
        with T.Kernel(
            T.ceildiv(VALUE_DIM, value_block),
            value_heads,
            sequences,
            threads=THREADS,
        ) as (value_tile, value_head, sequence):
            q_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
            w_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
            k_dot_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
            k_update_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
            state_shared = T.alloc_shared([KEY_DIM, value_block], T.float16)
            value_shared = T.alloc_shared([CHUNK_SIZE, value_block], T.float16)
            value_t_shared = T.alloc_shared([value_block, CHUNK_SIZE], T.float16)
            scores_shared = T.alloc_shared([CHUNK_SIZE, CHUNK_SIZE], T.float16)
            gate_shared = T.alloc_shared([CHUNK_SIZE], T.float32)

            state = T.alloc_fragment([value_block, KEY_DIM], T.float32)
            value_new = T.alloc_fragment([CHUNK_SIZE, value_block], T.float32)
            output = T.alloc_fragment([CHUNK_SIZE, value_block], T.float32)
            scores = T.alloc_fragment([CHUNK_SIZE, CHUNK_SIZE], T.float32)

            slot = StateIndices[sequence]
            begin = CuSeqLens[sequence]
            end = CuSeqLens[sequence + 1]
            num_chunks = T.ceildiv(end - begin, CHUNK_SIZE)
            key_head = T.floordiv(value_head, heads_per_key)
            value_start = value_tile * value_block

            for dv, d in T.Parallel(value_block, KEY_DIM):
                state[dv, d] = T.if_then_else(
                    (slot >= 0) & (value_start + dv < VALUE_DIM),
                    State[slot, value_head, value_start + dv, d],
                    0,
                )

            if slot >= 0:
                for chunk in T.serial(num_chunks):
                    chunk_begin = begin + chunk * CHUNK_SIZE
                    valid = T.min(CHUNK_SIZE, end - chunk_begin)

                    if store_checkpoints:
                        checkpoint = ChunkOffsets[sequence] + chunk
                        for dv, d in T.Parallel(value_block, KEY_DIM):
                            if value_start + dv < VALUE_DIM:
                                Checkpoints[
                                    0,
                                    checkpoint,
                                    value_head,
                                    value_start + dv,
                                    d,
                                ] = T.cast(state[dv, d], T.float16)

                    for i, d in T.Parallel(CHUNK_SIZE, KEY_DIM):
                        if i < valid:
                            q_shared[i, d] = Q[0, chunk_begin + i, key_head, d]
                            w_shared[i, d] = W[0, chunk_begin + i, value_head, d]
                            k_dot_shared[i, d] = K[0, chunk_begin + i, key_head, d]
                            k_update_shared[i, d] = K[0, chunk_begin + i, key_head, d]
                        else:
                            q_shared[i, d] = 0
                            w_shared[i, d] = 0
                            k_dot_shared[i, d] = 0
                            k_update_shared[i, d] = 0
                    for i, dv in T.Parallel(CHUNK_SIZE, value_block):
                        value_shared[i, dv] = T.if_then_else(
                            (i < valid) & (value_start + dv < VALUE_DIM),
                            U[
                                0,
                                chunk_begin + i,
                                value_head,
                                value_start + dv,
                            ],
                            0,
                        )
                    for i in T.Parallel(CHUNK_SIZE):
                        gate_shared[i] = T.if_then_else(
                            i < valid,
                            GateCumsum[0, chunk_begin + i, value_head],
                            GateCumsum[0, chunk_begin + valid - 1, value_head],
                        )
                    for d, dv in T.Parallel(KEY_DIM, value_block):
                        state_shared[d, dv] = T.cast(state[dv, d], T.float16)
                    T.sync_threads()

                    # V_new = U - W H_before.
                    T.clear(value_new)
                    T.gemm(
                        w_shared,
                        state_shared,
                        value_new,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, dv in T.Parallel(CHUNK_SIZE, value_block):
                        value_new[i, dv] = (
                            T.cast(value_shared[i, dv], T.float32) - value_new[i, dv]
                        )
                        value_shared[i, dv] = T.cast(value_new[i, dv], T.float16)
                    T.sync_threads()

                    # O = scale * (exp(g) QH + causal(G * QK^T) V_new).
                    T.clear(output)
                    T.gemm(
                        q_shared,
                        state_shared,
                        output,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.clear(scores)
                    T.gemm(
                        q_shared,
                        k_dot_shared,
                        scores,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, j in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                        scores[i, j] = T.if_then_else(
                            (j <= i) & (i < valid),
                            scores[i, j]
                            * T.exp2((gate_shared[i] - gate_shared[j]) * _LOG2_E)
                            * Scale,
                            0,
                        )
                        scores_shared[i, j] = T.cast(scores[i, j], T.float16)
                    for i, dv in T.Parallel(CHUNK_SIZE, value_block):
                        output[i, dv] *= Scale * T.exp2(gate_shared[i] * _LOG2_E)
                    T.sync_threads()
                    T.gemm(
                        scores_shared,
                        value_shared,
                        output,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, dv in T.Parallel(CHUNK_SIZE, value_block):
                        if (i < valid) & (value_start + dv < VALUE_DIM):
                            Output[
                                0,
                                chunk_begin + i,
                                value_head,
                                value_start + dv,
                            ] = T.cast(output[i, dv], T.float16)

                    # H' = exp(g_last) H + V_new^T diag(exp(g_last-g)) K.
                    for dv, i in T.Parallel(value_block, CHUNK_SIZE):
                        value_t_shared[dv, i] = T.cast(
                            value_new[i, dv]
                            * T.exp2(
                                (gate_shared[valid - 1] - gate_shared[i]) * _LOG2_E
                            ),
                            T.float16,
                        )
                    last_decay = T.exp2(gate_shared[valid - 1] * _LOG2_E)
                    for dv, d in T.Parallel(value_block, KEY_DIM):
                        state[dv, d] *= last_decay
                    T.gemm(
                        value_t_shared,
                        k_update_shared,
                        state,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for dv, d in T.Parallel(value_block, KEY_DIM):
                    if value_start + dv < VALUE_DIM:
                        State[slot, value_head, value_start + dv, d] = state[dv, d]

    return main


@lru_cache(maxsize=None)
def _get_kkt_raw(q_heads: int, value_heads: int, num_sequences: int):
    return _kkt_raw_kernel(q_heads, value_heads, num_sequences)


@lru_cache(maxsize=None)
def _get_packed_qk_norm(q_heads: int, value_heads: int):
    return _packed_qk_norm_kernel(q_heads, value_heads)


@lru_cache(maxsize=None)
def _get_packed_v_copy(q_heads: int, value_heads: int):
    return _packed_v_copy_kernel(q_heads, value_heads)


@lru_cache(maxsize=None)
def _get_packed_gate_cumsum(value_heads: int, num_sequences: int):
    return _packed_gate_cumsum_kernel(value_heads, num_sequences)


@lru_cache(maxsize=None)
def _get_packed_gate_raw(value_heads: int):
    return _packed_gate_raw_kernel(value_heads)


@lru_cache(maxsize=None)
def _get_packed_recurrent(
    q_heads: int,
    value_heads: int,
    num_sequences: int,
    state_slots: int,
    column_groups_per_block: int,
    store_checkpoints: bool,
    state_fp32: bool,
):
    return _packed_recurrent_kernel(
        q_heads,
        value_heads,
        num_sequences,
        state_slots,
        column_groups_per_block,
        store_checkpoints,
        state_fp32,
    )


@lru_cache(maxsize=None)
def _get_kkt_inverse(q_heads: int, value_heads: int, num_sequences: int):
    return _kkt_inverse_kernel(q_heads, value_heads, num_sequences)


@lru_cache(maxsize=None)
def _get_chunk_forward(
    q_heads: int,
    value_heads: int,
    num_sequences: int,
    state_slots: int,
    store_checkpoints: bool,
    state_fp32: bool,
    value_block: int = VALUE_BLOCK,
    packed_v: bool = False,
    k_reuse_mode: int = 0,
):
    return _chunk_forward_kernel(
        q_heads,
        value_heads,
        num_sequences,
        state_slots,
        store_checkpoints,
        state_fp32,
        value_block,
        packed_v,
        k_reuse_mode,
    )


@lru_cache(maxsize=None)
def _get_state_output(
    q_heads: int,
    value_heads: int,
    num_sequences: int,
    state_slots: int,
    store_checkpoints: bool,
    state_fp32: bool,
    value_block: int = VALUE_BLOCK,
):
    return _state_output_kernel(
        q_heads,
        value_heads,
        num_sequences,
        state_slots,
        store_checkpoints,
        state_fp32,
        value_block,
    )


def _column_groups_per_block(
    tokens: int,
    q_heads: int,
    value_heads: int,
) -> int:
    override = os.environ.get("SGLANG_V100_GDN_COLUMN_GROUPS_PER_BLOCK")
    if override is not None:
        groups = int(override)
        if groups not in (1, 2, 4, 8):
            raise ValueError(
                "SGLANG_V100_GDN_COLUMN_GROUPS_PER_BLOCK must be 1, 2, 4, or 8"
            )
        return groups
    if value_heads == 12:
        return 2 if tokens >= 1024 else 1
    if value_heads == 8:
        return 1 if tokens <= 1024 else 4
    if value_heads == 16:
        return 2 if q_heads == 8 or tokens <= 1024 else 1
    if value_heads in (24, 48):
        return 2
    if value_heads >= 32:
        return 1
    return 2


def packed_recurrent_gdn_sm70(
    mixed_qkv: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    *,
    q_heads: int,
    value_heads: int,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    store_checkpoints: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the persistent subgroup GDN path on row-strided packed QKV."""

    if mixed_qkv.device.type != "cuda" or torch.cuda.get_device_capability(
        mixed_qkv.device
    ) != (7, 0):
        raise ValueError("The packed recurrent GDN path requires an SM70 GPU")
    if mixed_qkv.dtype != torch.float16 or not mixed_qkv.is_contiguous():
        raise ValueError("Packed SM70 GDN requires contiguous FP16 mixed QKV")
    if gate_a.dtype != torch.float16 or gate_b.dtype != torch.float16:
        raise ValueError("Packed SM70 GDN requires FP16 gate inputs")
    if (
        gate_a.shape != (mixed_qkv.shape[0], value_heads)
        or gate_b.shape != gate_a.shape
    ):
        raise ValueError("Packed gate inputs must have shape [tokens, value_heads]")
    if state.dtype not in (torch.float16, torch.float32) or state.shape[1:] != (
        value_heads,
        VALUE_DIM,
        KEY_DIM,
    ):
        raise ValueError(
            "Packed SM70 GDN requires [slots, Hv, 128, 128] FP16/FP32 state"
        )
    expected_dim = 2 * q_heads * KEY_DIM + value_heads * VALUE_DIM
    if mixed_qkv.shape[1] != expected_dim or value_heads % q_heads != 0:
        raise ValueError("mixed_qkv does not match the requested Q/V head geometry")
    if a_log.shape != (value_heads,) or a_log.dtype != torch.float32:
        raise ValueError("a_log must be FP32 with one value per value head")
    if dt_bias.shape != (value_heads,) or dt_bias.dtype != torch.float16:
        raise ValueError("dt_bias must be FP16 with one value per value head")
    if cu_seqlens.numel() != state_indices.numel() + 1:
        raise ValueError("cu_seqlens must contain one boundary per sequence")

    cu_seqlens = cu_seqlens.to(dtype=torch.int32).contiguous()
    state_indices = state_indices.to(dtype=torch.int32).contiguous()
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE).to(dtype=torch.int32)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE).to(dtype=torch.int32)
    q, k = _get_packed_qk_norm(q_heads, value_heads)(mixed_qkv)
    gate, beta = _get_packed_gate_raw(value_heads)(
        gate_a.contiguous(),
        gate_b.contiguous(),
        a_log,
        dt_bias,
    )
    if store_checkpoints:
        checkpoints = torch.empty(
            (
                1,
                int(chunk_indices.shape[0]),
                value_heads,
                VALUE_DIM,
                KEY_DIM,
            ),
            device=mixed_qkv.device,
            dtype=torch.float16,
        )
    else:
        checkpoints = torch.empty(
            (1, 0, value_heads, VALUE_DIM, KEY_DIM),
            device=mixed_qkv.device,
            dtype=torch.float16,
        )
    groups = _column_groups_per_block(mixed_qkv.shape[0], q_heads, value_heads)
    output = _get_packed_recurrent(
        q_heads,
        value_heads,
        state_indices.numel(),
        state.shape[0],
        groups,
        store_checkpoints,
        state.dtype == torch.float32,
    )(
        q,
        k,
        mixed_qkv,
        gate,
        beta,
        state,
        state_indices,
        cu_seqlens,
        chunk_offsets,
        float(scale),
        checkpoints,
    )
    return output, checkpoints if store_checkpoints else None


def packed_chunked_gdn_sm70(
    mixed_qkv: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    *,
    q_heads: int,
    value_heads: int,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    store_checkpoints: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the tensor-core chunked path directly from packed projection rows."""

    # Reuse the strict packed-contract validation, but do not launch the
    # recurrent kernel merely to validate. The remaining checks are identical
    # to the public chunked entry point below.
    if mixed_qkv.device.type != "cuda" or torch.cuda.get_device_capability(
        mixed_qkv.device
    ) != (7, 0):
        raise ValueError("The packed chunked GDN path requires an SM70 GPU")
    if mixed_qkv.dtype != torch.float16 or not mixed_qkv.is_contiguous():
        raise ValueError("Packed SM70 GDN requires contiguous FP16 mixed QKV")
    if (
        gate_a.shape != (mixed_qkv.shape[0], value_heads)
        or gate_b.shape != gate_a.shape
    ):
        raise ValueError("Packed gate inputs must have shape [tokens, value_heads]")
    if gate_a.dtype != torch.float16 or gate_b.dtype != torch.float16:
        raise ValueError("Packed SM70 GDN requires FP16 gate inputs")
    if state.dtype not in (torch.float16, torch.float32) or state.shape[1:] != (
        value_heads,
        VALUE_DIM,
        KEY_DIM,
    ):
        raise ValueError(
            "Packed SM70 GDN requires [slots, Hv, 128, 128] FP16/FP32 state"
        )
    if mixed_qkv.shape[1] != 2 * q_heads * KEY_DIM + value_heads * VALUE_DIM:
        raise ValueError("mixed_qkv does not match the requested Q/V head geometry")
    if cu_seqlens.numel() != state_indices.numel() + 1:
        raise ValueError("cu_seqlens must contain one boundary per sequence")

    cu_seqlens = cu_seqlens.to(dtype=torch.int32).contiguous()
    state_indices = state_indices.to(dtype=torch.int32).contiguous()
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, CHUNK_SIZE).to(dtype=torch.int32).contiguous()
    )
    packed_v_direct_value = (
        os.environ.get("SGLANG_V100_GDN_PACKED_V_DIRECT", "1").strip().lower()
    )
    if packed_v_direct_value not in (
        "0",
        "false",
        "off",
        "no",
        "1",
        "true",
        "on",
        "yes",
    ):
        raise ValueError("SGLANG_V100_GDN_PACKED_V_DIRECT must be a boolean value")
    packed_v_direct = packed_v_direct_value in ("1", "true", "on", "yes")
    q, k, v, gate_cumsum, beta = prepare_packed_gdn_sm70(
        mixed_qkv,
        gate_a,
        gate_b,
        q_heads=q_heads,
        value_heads=value_heads,
        a_log=a_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        copy_v=not packed_v_direct,
    )
    return chunked_gdn_sm70(
        q,
        k,
        v,
        gate_cumsum,
        beta,
        scale=scale,
        state=state,
        state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        store_checkpoints=store_checkpoints,
        qk_normalized=True,
        gate_is_cumulative=True,
        chunk_indices=chunk_indices,
        packed_v=packed_v_direct,
    )


def prepare_packed_gdn_sm70(
    mixed_qkv: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    *,
    q_heads: int,
    value_heads: int,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    copy_v: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare row-strided mixed QKV and gates using only TileLang kernels."""

    if mixed_qkv.dtype != torch.float16 or not mixed_qkv.is_contiguous():
        raise ValueError("Packed SM70 GDN preparation requires contiguous FP16 QKV")
    if gate_a.dtype != torch.float16 or gate_b.dtype != torch.float16:
        raise ValueError("Packed SM70 GDN preparation requires FP16 gate inputs")
    gate_a = gate_a.contiguous()
    gate_b = gate_b.contiguous()
    q, k = _get_packed_qk_norm(q_heads, value_heads)(mixed_qkv)
    # The native chunk forward kernel can consume V directly from the
    # row-strided projection output.  Keep the compact copy available for the
    # generic/Triton control path and external callers.
    v = _get_packed_v_copy(q_heads, value_heads)(mixed_qkv) if copy_v else mixed_qkv
    gate_cumsum, beta = _get_packed_gate_cumsum(value_heads, cu_seqlens.numel() - 1)(
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        cu_seqlens,
        chunk_indices,
    )
    return q, k, v, gate_cumsum, beta


def chunked_gdn_sm70(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    store_checkpoints: bool = False,
    qk_normalized: bool = False,
    gate_is_cumulative: bool = False,
    chunk_indices: torch.Tensor | None = None,
    packed_v: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run normalized, variable-length GDN prefill on SM70."""

    if torch.cuda.get_device_capability(q.device) != (7, 0):
        raise ValueError("The TileLang chunked GDN kernel is specialized for SM70.")
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("The SM70 TileLang chunked GDN kernel requires FP16 QKV.")
    if q.shape[0] != 1 or k.shape[0] != 1 or (not packed_v and v.shape[0] != 1):
        raise ValueError(
            "Variable-length QKV must be flattened with batch dimension 1."
        )
    _, tokens, q_heads, key_dim = q.shape
    if packed_v:
        value_heads = state.shape[-3]
        value_dim = state.shape[-1]
        expected_dim = 2 * q_heads * key_dim + value_heads * value_dim
        if v.ndim != 2 or v.shape != (tokens, expected_dim) or not v.is_contiguous():
            raise ValueError(
                "Packed V must be contiguous row-strided mixed QKV with the "
                "requested Q/V head geometry."
            )
    else:
        value_heads, value_dim = v.shape[-2:]
    if key_dim != KEY_DIM or value_dim != VALUE_DIM:
        raise ValueError("The SM70 TileLang chunked GDN path supports K=V=128.")
    if state.dtype not in (torch.float16, torch.float32):
        raise ValueError(
            "The SM70 TileLang chunked GDN path requires FP16 or FP32 state."
        )

    if not qk_normalized:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)
    cu_seqlens = cu_seqlens.to(dtype=torch.int32).contiguous()
    state_indices = state_indices.to(dtype=torch.int32).contiguous()
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    chunk_indices = chunk_indices.to(dtype=torch.int32).contiguous()
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE).to(dtype=torch.int32)
    if gate_is_cumulative:
        g_cumsum = g
    else:
        g_cumsum = chunk_local_cumsum(
            g,
            chunk_size=CHUNK_SIZE,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    output = torch.empty(
        (1, tokens, value_heads, value_dim),
        dtype=torch.float16,
        device=q.device,
    )
    num_chunks = int(chunk_indices.shape[0])
    if store_checkpoints:
        checkpoints = torch.empty(
            (1, num_chunks, value_heads, value_dim, key_dim),
            dtype=torch.float16,
            device=q.device,
        )
    else:
        # A zero-sized placeholder keeps the compiled ABI stable.
        checkpoints = torch.empty(
            (1, 0, value_heads, value_dim, key_dim),
            dtype=torch.float16,
            device=q.device,
        )

    full_tilelang = os.environ.get("SGLANG_V100_GDN_FULL_TILELANG", "1").strip().lower()
    if full_tilelang not in ("0", "false", "off", "no", "1", "true", "on", "yes"):
        raise ValueError("SGLANG_V100_GDN_FULL_TILELANG must be a boolean value")
    if full_tilelang in ("1", "true", "on", "yes"):
        num_sequences = state_indices.numel()
        state_slots = state.shape[0]
        value_block = int(os.environ.get("SGLANG_V100_GDN_VALUE_BLOCK", "32"))
        k_reuse_mode = int(os.environ.get("SGLANG_V100_GDN_K_REUSE_MODE", "3"))
        if value_block not in (16, 32):
            raise ValueError("SGLANG_V100_GDN_VALUE_BLOCK must be 16 or 32")
        if k_reuse_mode not in (0, 3):
            raise ValueError("SGLANG_V100_GDN_K_REUSE_MODE must be 0 or 3")
        if value_block == 32 and k_reuse_mode != 3:
            raise ValueError("The 32-column SM70 GDN schedule requires K reuse mode 3")
        inverse = _get_kkt_inverse(q_heads, value_heads, num_sequences)(
            k,
            beta,
            cu_seqlens,
            chunk_indices,
        )
        _get_chunk_forward(
            q_heads,
            value_heads,
            num_sequences,
            state_slots,
            store_checkpoints,
            state.dtype == torch.float32,
            value_block=value_block,
            packed_v=packed_v,
            k_reuse_mode=k_reuse_mode,
        )(
            q,
            k,
            v,
            inverse,
            g_cumsum,
            beta,
            state,
            state_indices,
            cu_seqlens,
            chunk_offsets,
            float(scale),
            output,
            checkpoints,
        )
    else:
        # Explicit rollback/control: SGLang Triton prepares WY while TileLang
        # retains the recurrent state/output fusion.
        if packed_v:
            v = _get_packed_v_copy(q_heads, value_heads)(v)
        w, u, _ = chunk_gated_delta_rule_fwd_intra(
            k=k,
            v=v,
            g=g_cumsum,
            beta=beta,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        _get_state_output(
            q_heads,
            value_heads,
            state_indices.numel(),
            state.shape[0],
            store_checkpoints,
            state.dtype == torch.float32,
        )(
            q,
            k,
            w,
            u,
            g_cumsum,
            state,
            state_indices,
            cu_seqlens,
            chunk_offsets,
            float(scale),
            output,
            checkpoints,
        )
    return output, checkpoints if store_checkpoints else None
