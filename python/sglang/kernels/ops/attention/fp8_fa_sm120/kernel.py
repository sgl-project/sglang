# SPDX-License-Identifier: Apache-2.0
"""FP8 flash-attention forward for SM120 in CuTe-DSL: E4M3 QK and PV, FP32 softmax.

    Q, K   [H, padded_S, 128]  E4M3, queries padded to 128 rows, keys to 32
    V^T    [H, 128, padded_S]  E4M3, keys stored in key_order_for_positions() order
    scales [3, H]              FP32 per-head descale of Q, K, V
    O      [H, S, 128]         BF16
    LSE    [H, S]              FP32

One CTA covers 128 queries x 32 keys on 128 threads: 4 warps, 32 query rows per warp.
Every thread keeps the online-softmax state (running max m, running sum l) of its four
query rows in registers; the max is reduced across the quad with two butterfly shuffles
per key tile, the sum once after the key loop.

Per key tile: K and V^T arrive in one of two shared-memory stages and the next tile is
requested right after the barrier, so the copy overlaps the math. QK runs as four E4M3
mma.sync blocks into FP32 score registers. The probabilities are scaled by 256, cast to
E4M3 and packed in registers from the score fragment into the A fragment of PV; they
never go through shared memory. The A fragment holds the keys of every 16-key group in
a fixed permuted order, which is why V^T is stored in that order. PV accumulates into
FP32 O registers that are rescaled to the new max on every tile and normalized by l at
the end.

Non-causal only. No dropout, no autograd. Inputs must be finite.
"""

import math

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils


def _make_gmem_tiled_copy(atom_copy, dtype, copy_bits, minor_size, num_threads):
    copy_elems = copy_bits // dtype.width
    shape_dim_1 = minor_size // copy_elems
    thread_layout = cute.make_layout(
        (num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
    )
    value_layout = cute.make_layout((1, copy_elems))
    return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)


def _make_smem_layout_fp8(dtype, copy_bits, smem_tiler):
    major_size = smem_tiler[1]
    row_bytes = major_size * dtype.width // 8
    chunk_bytes = copy_bits // 8
    swizzle_bits = min(int(math.log2(row_bytes // chunk_bytes)), 3)
    base_bits = int(math.log2(chunk_bytes))
    shift_bits = int(math.log2(128 // chunk_bytes))
    swizzle = cute.make_swizzle(swizzle_bits, base_bits, shift_bits)
    atom = cute.make_layout((8, major_size), stride=(major_size, 1))
    layout = cute.tile_to_shape(atom, smem_tiler, (0, 1, 2))
    return layout, swizzle


@cute.kernel
def _fp8_attention_sm120(
    softmax_scale: cutlass.Float32,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    mLSE: cute.Tensor,
    mScales: cute.Tensor,
    sQ_layout: cute.Layout,
    sK_layout: cute.Layout,
    sV_layout: cute.Layout,
    sQ_swizzle: cute.Swizzle,
    sK_swizzle: cute.Swizzle,
    sV_swizzle: cute.Swizzle,
    tiled_copy_Q: cute.TiledCopy,
    tiled_copy_K: cute.TiledCopy,
    tiled_copy_V: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    cta_tiler: cutlass.Constexpr = (128, 32, 128),
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, bidz = cute.arch.block_idx()

    qk_scale = mScales[0, bidz] * mScales[1, bidz] * softmax_scale
    pv_scale = mScales[2, bidz] / 256.0
    sequence = mLSE.shape[0]

    m = cute.make_rmem_tensor((4,), cutlass.Float32)
    m.fill(-cutlass.Float32.inf)

    l = cute.make_rmem_tensor((4,), cutlass.Float32)
    l.fill(0.0)

    gQ = cute.local_tile(
        mQ[None, None, bidz], cta_tiler, (bidx, None, 0), proj=(1, None, 1)
    )
    gK = cute.local_tile(
        mK[None, None, bidz], cta_tiler, (None, None, 0), proj=(None, 1, 1)
    )
    gV = cute.local_tile(mV[None, None, bidz], (128, 32), (0, None))
    gO = cute.local_tile(mO[None, None, bidz], (128, 128), (bidx, 0))

    @cute.struct
    class SharedStorageQKV:
        q: cute.struct.Align[
            cute.struct.MemRange[mQ.element_type, cute.cosize(sQ_layout)], 16
        ]
        k: cute.struct.Align[
            cute.struct.MemRange[mK.element_type, cute.cosize(sK_layout)], 16
        ]
        v: cute.struct.Align[
            cute.struct.MemRange[mV.element_type, cute.cosize(sV_layout)], 16
        ]

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorageQKV.size_in_bytes(), byte_alignment=16)
    sQ = SharedStorageQKV(storage).q.get_tensor(sQ_layout, swizzle=sQ_swizzle)
    sK = SharedStorageQKV(storage).k.get_tensor(sK_layout, swizzle=sK_swizzle)
    sV = SharedStorageQKV(storage).v.get_tensor(sV_layout, swizzle=sV_swizzle)

    thr_copy_Q = tiled_copy_Q.get_slice(tidx)
    thr_copy_K = tiled_copy_K.get_slice(tidx)
    thr_copy_V = tiled_copy_V.get_slice(tidx)

    tQgQ = thr_copy_Q.partition_S(gQ)
    tKgK = thr_copy_K.partition_S(gK)
    tVgV = thr_copy_V.partition_S(gV)

    tQsQ = thr_copy_Q.partition_D(sQ)
    tKsK = thr_copy_K.partition_D(sK)
    tVsV = thr_copy_V.partition_D(sV)

    k_tile_count = cute.size(tKgK, mode=[3])

    thr_mma = tiled_mma.get_slice(tidx)
    tCgO = thr_mma.partition_C(gO)

    tCsQ = thr_mma.partition_A(sQ)
    tCsK = thr_mma.partition_B(sK)
    tCsV = thr_mma.partition_B(sV)

    tCrQ = tiled_mma.make_fragment_A(tCsQ[None, None, None, 0])
    tCrK = tiled_mma.make_fragment_B(tCsK[None, None, None, 0])
    tCrV = tiled_mma.make_fragment_B(tCsV[None, None, None, 0])

    acc_shape = thr_mma.partition_shape_C((128, 32))
    tCrC = cute.make_rmem_tensor(acc_shape, cutlass.Float32)

    acc_shape_O = thr_mma.partition_shape_C((128, 128))
    tCrO = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
    tCrO.fill(0.0)

    tCrP = cute.make_rmem_tensor(
        cute.make_layout(((4, 2, 2), 2, 1), stride=((1, 4, 8), 16, 0)),
        mQ.element_type,
    )
    num_k_block_PV = cute.size(tCrP, mode=[2])

    pack_shape = ((2, 2), 2, (2, 2))
    tCrC_as_pack = cute.make_tensor(
        tCrC.iterator,
        cute.make_layout(pack_shape, stride=((1, 2), 4, (8, 16))),
    )
    tCrP_as_pack = cute.make_tensor(
        tCrP.iterator,
        cute.make_layout(pack_shape, stride=((1, 4), 16, (2, 8))),
    )

    atom_copy_s2r_Q = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x16x8bOp(transpose=False, num_matrices=4),
        mQ.element_type,
    )
    atom_copy_s2r_K = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x16x8bOp(transpose=False, num_matrices=4),
        mK.element_type,
    )
    atom_copy_s2r_V = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x16x8bOp(transpose=False, num_matrices=4),
        mV.element_type,
    )

    tiled_copy_s2r_Q = cute.make_tiled_copy_A(atom_copy_s2r_Q, tiled_mma)
    tiled_copy_s2r_K = cute.make_tiled_copy_B(atom_copy_s2r_K, tiled_mma)
    tiled_copy_s2r_V = cute.make_tiled_copy_B(atom_copy_s2r_V, tiled_mma)

    ldmatrix_Q = tiled_copy_s2r_Q.get_slice(tidx)
    ldmatrix_K = tiled_copy_s2r_K.get_slice(tidx)
    ldmatrix_V = tiled_copy_s2r_V.get_slice(tidx)

    tCsQ_copy_view = ldmatrix_Q.partition_S(sQ)
    tCrQ_copy_view = ldmatrix_Q.retile(tCrQ)
    tCsK_copy_view = ldmatrix_K.partition_S(sK)
    tCrK_copy_view = ldmatrix_K.retile(tCrK)
    tCsV_copy_view = ldmatrix_V.partition_S(sV)
    tCrV_copy_view = ldmatrix_V.retile(tCrV)

    num_k_block = cute.size(tCrQ, mode=[2])

    tCsQ_p = tCsQ_copy_view[None, None, None, 0]

    cute.copy(tiled_copy_Q, tQgQ[None, None, None], tQsQ[None, None, None, 0])
    cute.copy(tiled_copy_K, tKgK[None, None, None, 0], tKsK[None, None, None, 0])
    cute.copy(tiled_copy_V, tVgV[None, None, None, 0], tVsV[None, None, None, 0])
    cute.arch.cp_async_commit_group()

    for k_tile in range(k_tile_count):
        stage = k_tile % 2
        next_stage = 1 - stage

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        if k_tile + 1 < k_tile_count:
            cute.copy(
                tiled_copy_K,
                tKgK[None, None, None, k_tile + 1],
                tKsK[None, None, None, next_stage],
            )
            cute.copy(
                tiled_copy_V,
                tVgV[None, None, None, k_tile + 1],
                tVsV[None, None, None, next_stage],
            )
        cute.arch.cp_async_commit_group()

        tCsK_p = tCsK_copy_view[None, None, None, stage]
        tCsV_p = tCsV_copy_view[None, None, None, stage]

        tCrC.fill(0.0)

        for k_block in cutlass.range(num_k_block, unroll_full=True):
            cute.copy(
                tiled_copy_s2r_Q,
                tCsQ_p[None, None, k_block],
                tCrQ_copy_view[None, None, k_block],
            )
            cute.copy(
                tiled_copy_s2r_K,
                tCsK_p[None, None, k_block],
                tCrK_copy_view[None, None, k_block],
            )
            cute.gemm(
                tiled_mma,
                tCrC,
                tCrQ[None, None, k_block],
                tCrK[None, None, k_block],
                tCrC,
            )

        for m_tile in cutlass.range_constexpr(2):
            for row in cutlass.range_constexpr(2):
                state = m_tile * 2 + row
                tCrC_row = tCrC[(None, row), m_tile, None]
                row_scores = tCrC_row.load() * qk_scale

                if cutlass.const_expr(mK.shape[0] != sequence):
                    tCrC_row.store(row_scores)
                    for column_group in cutlass.range_constexpr(4):
                        for column_pair in cutlass.range_constexpr(2):
                            key_row = (
                                k_tile * 32
                                + (tidx % 4) * 2
                                + column_group * 8
                                + column_pair
                            )
                            if key_row >= sequence:
                                tCrC_row[
                                    column_pair, column_group
                                ] = -cutlass.Float32.inf
                    row_scores = tCrC_row.load()

                local_max = row_scores.reduce(
                    cute.ReductionOp.MAX,
                    -cutlass.Float32.inf,
                    0,
                )

                neighbor_max = cute.arch.shuffle_sync_bfly(local_max, offset=1)
                pair_max = cute.arch.fmax(local_max, neighbor_max)

                neighbor_max = cute.arch.shuffle_sync_bfly(pair_max, offset=2)
                tile_max = cute.arch.fmax(pair_max, neighbor_max)

                m_new = cute.arch.fmax(m[state], tile_max)

                alpha = cute.math.exp2(
                    (m[state] - m_new) * math.log2(math.e),
                    fastmath=True,
                )
                p = cute.math.exp2(
                    (row_scores - m_new) * math.log2(math.e),
                    fastmath=True,
                )

                p_partial_sums = cute.make_rmem_tensor((4,), cutlass.Float32)
                for column_group in cutlass.range_constexpr(4):
                    p_partial_sums[column_group] = (
                        p[0, column_group] + p[1, column_group]
                    )

                for level in cutlass.range_constexpr(2):
                    for sum_index in cutlass.range_constexpr(2 >> level):
                        p_partial_sums[sum_index] = (
                            p_partial_sums[2 * sum_index]
                            + p_partial_sums[2 * sum_index + 1]
                        )
                tile_sum = p_partial_sums[0]

                l[state] = alpha * l[state] + tile_sum
                m[state] = m_new

                tCrC_row.store(p)

                tCrO_row = tCrO[(None, row), m_tile, None]
                tCrO_row.store(tCrO_row.load() * alpha)

        tCrP_as_pack.store((tCrC_as_pack.load() * 256.0).to(mQ.element_type))

        for k_block in cutlass.range(num_k_block_PV, unroll_full=True):
            cute.copy(
                tiled_copy_s2r_V,
                tCsV_p[None, None, k_block],
                tCrV_copy_view[None, None, k_block],
            )
            cute.gemm(
                tiled_mma,
                tCrO,
                tCrP[None, None, k_block],
                tCrV[None, None, k_block],
                tCrO,
            )

    warp_id = tidx // 32
    lane_id = tidx % 32
    quad_id = lane_id // 4

    for m_tile in cutlass.range_constexpr(2):
        for row in cutlass.range_constexpr(2):
            state = m_tile * 2 + row
            row_sum = l[state]

            neighbor_sum = cute.arch.shuffle_sync_bfly(row_sum, offset=1)
            row_sum += neighbor_sum

            neighbor_sum = cute.arch.shuffle_sync_bfly(row_sum, offset=2)
            row_sum += neighbor_sum

            l[state] = row_sum
            lse = m[state] + cute.math.log(row_sum)

            query_row = bidx * 128 + m_tile * 64 + warp_id * 16 + quad_id + row * 8

            valid_query = True
            if cutlass.const_expr(sequence % 128 != 0):
                valid_query = query_row < sequence

            if valid_query and lane_id % 4 == 0:
                mLSE[query_row, bidz] = lse

            tCrO_row = tCrO[(None, row), m_tile, None]
            inverse_row_sum = pv_scale / row_sum
            tCrO_row.store(tCrO_row.load() * inverse_row_sum)

            if valid_query:
                tCrO_row_out = cute.make_fragment_like(tCrO_row, mO.element_type)
                tCrO_row_out.store(tCrO_row.load().to(mO.element_type))
                cute.autovec_copy(tCrO_row_out, tCgO[(None, row), m_tile, None])


@cute.jit
def fp8_attention_host(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    mLSE: cute.Tensor,
    mScales: cute.Tensor,
    softmax_scale: cutlass.Float32,
    stream,
):
    mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[1, 2, 0]))
    mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=[1, 2, 0]))
    mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=[1, 2, 0]))
    mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=[1, 2, 0]))
    mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=[1, 0]))

    mma_op = cute.nvgpu.warp.MmaFP8Op(
        mQ.element_type,
        cutlass.Float32,
        (16, 8, 32),
    )
    tiled_mma = cute.make_tiled_mma(
        mma_op,
        (4, 1, 1),
        permutation_mnk=(128, 16, 32),
    )

    copy_bits = 128
    num_threads = 128

    sQ_layout, sQ_swizzle = _make_smem_layout_fp8(
        mQ.element_type,
        copy_bits,
        (128, 128, 1),
    )
    sK_layout, sK_swizzle = _make_smem_layout_fp8(
        mK.element_type,
        copy_bits,
        (32, 128, 2),
    )
    sV_layout, sV_swizzle = _make_smem_layout_fp8(
        mV.element_type,
        copy_bits,
        (128, 32, 2),
    )

    atom_copy_g2s = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
        mQ.element_type,
        num_bits_per_copy=copy_bits,
    )
    tiled_copy_Q = _make_gmem_tiled_copy(
        atom_copy_g2s,
        mQ.element_type,
        copy_bits,
        128,
        num_threads,
    )
    tiled_copy_K = _make_gmem_tiled_copy(
        atom_copy_g2s,
        mK.element_type,
        copy_bits,
        128,
        num_threads,
    )
    tiled_copy_V = _make_gmem_tiled_copy(
        atom_copy_g2s,
        mV.element_type,
        copy_bits,
        32,
        num_threads,
    )

    _fp8_attention_sm120(
        softmax_scale,
        mQ,
        mK,
        mV,
        mO,
        mLSE,
        mScales,
        sQ_layout,
        sK_layout,
        sV_layout,
        sQ_swizzle,
        sK_swizzle,
        sV_swizzle,
        tiled_copy_Q,
        tiled_copy_K,
        tiled_copy_V,
        tiled_mma,
    ).launch(
        grid=(cute.ceil_div(mQ.shape[0], 128), 1, mQ.shape[2]),
        block=(num_threads, 1, 1),
        stream=stream,
    )


def key_order_for_positions(positions):
    """Key held at each stored V^T position.

    Inside every 16-key group, position 4t + i holds key 8*(i//2) + 2t + i%2:
    the k order of the PV A fragment packed straight from the score fragment.
    """
    group = positions // 16 * 16
    quad = (positions % 16) // 4
    element = positions % 4
    return group + 8 * (element // 2) + 2 * quad + element % 2
