# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
#
# MExt R1 pair-fold direct-store epilogue for the SM100/SM103 CuTeDSL
# block-scaled GEMM. Implemented against the public CuTeDSL helper API
# (cutlass.utils.gemm.sm100); the calling kernel is the BSD-3-Clause
# FlashInfer port of CUTLASS's sm103 dense block-scaled example.

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import Boolean, Int32, const_expr
from cutlass.utils.gemm.sm100 import (
    epilogue_tmem_copy_and_partition,
    transform_partitioned_tensor_layout,
)


@cute.jit
def mext_fold_epilogue(
    gemm_kernel,
    epi_tidx: Int32,
    tCtAcc_base: cute.Tensor,
    tCgC_base: cute.Tensor,
    epi_tile: cute.Tile,
    epilogue_op: cutlass.Constexpr,
    mma_tile_coord_mnl,
    acc_consumer_state: pipeline.PipelineState,
    acc_pipeline: pipeline.PipelineAsync,
    tCcC_base: cute.Tensor = None,
    mC_mnl: cute.Tensor = None,
) -> pipeline.PipelineState:
    """Direct-store epilogue with in-register MExt row-pair folding.

    Contract: the GEMM's logical C is (n_w, m2, l) where m2 = 2*m_tok and the
    activation (GEMM B) rows are row-pair interleaved [base0, res0, base1,
    res1, ...]. The caller builds gC/cC from a pair-collapsed view of the
    physical output D[n_w, m_tok]: logical shape (n_w, (2, m_tok)) with
    stride (m_tok, (0, 1)), so both members of a (base, residue) pair alias
    one output address, and the m2 direction is memory-contiguous so the
    TMEM-load fragment's value mode runs along it.

    The fold sums each adjacent accumulator pair in fp32 before output dtype
    conversion; both aliased lanes then carry the sum, making the double store
    benign.
    """
    tCgC = transform_partitioned_tensor_layout(tCgC_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)

    tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, tCtAcc, tCgC, epi_tile, gemm_kernel.use_2cta_instrs
    )

    gC_epi = cute.flat_divide(tCgC, epi_tile)
    thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
    tTR_gC_partitioned = thr_copy_t2r.partition_D(gC_epi)
    tTR_rC = cute.make_rmem_tensor(
        tTR_gC_partitioned[(None, None, None, 0, 0, 0, 0, 0)].shape,
        gemm_kernel.c_dtype,
    )

    use_predication = tCcC_base is not None and mC_mnl is not None
    if const_expr(use_predication):
        tCcC = transform_partitioned_tensor_layout(tCcC_base)
        cC_epi = cute.flat_divide(tCcC, epi_tile)
        tTR_cC_partitioned = thr_copy_t2r.partition_D(cC_epi)

    tTR_gC = tTR_gC_partitioned[(None, None, None, None, None, *mma_tile_coord_mnl)]
    if const_expr(use_predication):
        tTR_cC = tTR_cC_partitioned[(None, None, None, None, None, *mma_tile_coord_mnl)]
        tTR_cC = cute.group_modes(tTR_cC, 3, cute.rank(tTR_cC))

    if const_expr(gemm_kernel.overlapping_accum):
        acc_stage_index = acc_consumer_state.phase
        reverse_subtile = Boolean(True) if acc_stage_index == 0 else Boolean(False)
    else:
        acc_stage_index = acc_consumer_state.index
    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

    acc_pipeline.consumer_wait(acc_consumer_state)

    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
    tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))

    # Pair members must be adjacent within the fragment's value mode.
    value_mode_size = cute.size(tTR_rAcc.shape, mode=[0])
    assert (
        value_mode_size % 2 == 0
    ), f"MExt fold requires an even TMEM-load value mode, got {value_mode_size}"
    frag_size = cute.size(tTR_rAcc.shape)

    # Probe (at trace time) whether the fragment profile supports the
    # vectorized pair-fold path; vec_ok is a Python constant, so the branch
    # below is compile-time static (no dynamic type-join in the DSL).
    _PAIR_EVEN = (((0, None), None), None, None)
    _PAIR_ODD = (((1, None), None), None, None)
    tTR_rC_half = None
    try:
        _gC_half_probe = tTR_gC[(None, None, None, 0)][_PAIR_EVEN]
        _rAcc_probe = tTR_rAcc[_PAIR_EVEN]
        tTR_rC_half = cute.make_rmem_tensor(_gC_half_probe.shape, gemm_kernel.c_dtype)
    except Exception:
        tTR_rC_half = None
    vec_ok = tTR_rC_half is not None

    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
    for subtile_idx in range(subtile_cnt):
        real_subtile_idx = subtile_idx
        if const_expr(gemm_kernel.overlapping_accum):
            # Phase 0 drains in reverse so the columns shared with the other
            # pseudo-buffer are consumed first (see kernel ctor comment).
            if reverse_subtile:
                real_subtile_idx = subtile_cnt - 1 - subtile_idx
        tTR_gC_subtile = tTR_gC[(None, None, None, real_subtile_idx)]
        tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

        # Release the accumulator buffer as soon as the shared-column region
        # (overlapping) / the whole buffer (strict) has been read from TMEM.
        if const_expr(gemm_kernel.overlapping_accum):
            release_at = gemm_kernel.iter_acc_early_release_in_epilogue
        else:
            release_at = subtile_cnt - 1
        if subtile_idx == release_at:
            cute.arch.fence_view_async_tmem_load()
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_consumer_state)
            acc_consumer_state.advance()

        # MExt fold. Vector path: the pair submode is the first submode of
        # the value mode by construction, so slicing it at 0/1 yields
        # even/odd half-fragments; one TensorSSA add folds all pairs, and the
        # even-lane gC slice drops the stride-0 alias mode, leaving
        # m_tok-contiguous addresses for a vectorized copy. Partitioned
        # identity coordinates grow monotonically per mode, so one
        # corner check covers a whole subtile; only ragged M2-edge subtiles
        # pay per-element guards. vec_ok is trace-time static.
        half_size = frag_size // 2
        if const_expr(vec_ok):
            folded_half = epilogue_op(
                tTR_rAcc[_PAIR_EVEN].load().to(cutlass.Float32)
                + tTR_rAcc[_PAIR_ODD].load().to(cutlass.Float32)
            )
            tTR_rC_half.store(folded_half.to(gemm_kernel.c_dtype))
            gC_half = tTR_gC_subtile[_PAIR_EVEN]
            if const_expr(use_predication):
                tTR_cC_subtile = tTR_cC[(None, None, None, real_subtile_idx)]
                if cute.elem_less(tTR_cC_subtile[frag_size - 1], mC_mnl.shape):
                    cute.autovec_copy(tTR_rC_half, gC_half)
                else:
                    cC_half = tTR_cC_subtile[_PAIR_EVEN]
                    for i in range(half_size):
                        if cute.elem_less(cC_half[i], mC_mnl.shape):
                            gC_half[i] = tTR_rC_half[i]
            else:
                cute.autovec_copy(tTR_rC_half, gC_half)
        else:
            tTR_rFold = cute.make_rmem_tensor(tTR_rAcc.shape, cutlass.Float32)
            for i in range(0, frag_size, 2):
                pair_sum = tTR_rAcc[i] + tTR_rAcc[i + 1]
                tTR_rFold[i] = pair_sum
                tTR_rFold[i + 1] = pair_sum
            acc_vec = epilogue_op(tTR_rFold.load()).to(gemm_kernel.c_dtype)
            tTR_rC.store(acc_vec)
            if const_expr(use_predication):
                tTR_cC_subtile = tTR_cC[(None, None, None, real_subtile_idx)]
                if cute.elem_less(tTR_cC_subtile[frag_size - 1], mC_mnl.shape):
                    for i in range(0, frag_size, 2):
                        tTR_gC_subtile[i] = tTR_rC[i]
                else:
                    for i in range(0, frag_size, 2):
                        if cute.elem_less(tTR_cC_subtile[i], mC_mnl.shape):
                            tTR_gC_subtile[i] = tTR_rC[i]
            else:
                for i in range(0, frag_size, 2):
                    tTR_gC_subtile[i] = tTR_rC[i]

    return acc_consumer_state


@cute.jit
def mext_fold_epilogue_tma(
    gemm_kernel,
    epi_tidx: Int32,
    warp_idx: Int32,
    tma_atom_c,
    tCtAcc_base: cute.Tensor,
    sC: cute.Tensor,
    tCgC_alias: cute.Tensor,
    gC_fold_mnl: cute.Tensor,
    epi_tile: cute.Tile,
    num_tiles_executed: Int32,
    epilogue_op: cutlass.Constexpr,
    mma_tile_coord_mnl,
    acc_consumer_state: pipeline.PipelineState,
    acc_pipeline: pipeline.PipelineAsync,
    c_pipeline,
) -> pipeline.PipelineState:
    """TMA-store fold epilogue.

    Accumulator side is stock geometry: t2r partitions against the stride-0
    alias view (tCgC_alias, m2-wide) exactly like the direct-store variant.
    After the in-register pair fold the fragment is half-width
    (one row per thread, m_tok-contiguous), so each thread writes its row of
    a plain half-width smem tile and the TMA engine bulk-stores that tile to
    the compact D_t (gC_fold_mnl). TMA clamps at the tensor extent, so no
    predication is needed. Sequencing mirrors the BSD-3 CUTLASS
    dense_blockscaled_gemm_persistent example's inline epilogue.
    """
    tCgC = transform_partitioned_tensor_layout(tCgC_alias)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)

    tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, tCtAcc, tCgC, epi_tile, gemm_kernel.use_2cta_instrs
    )

    epilog_sync_barrier = pipeline.NamedBarrier(
        barrier_id=gemm_kernel.epilog_sync_bar_id,
        num_threads=32 * len(gemm_kernel.epilogue_warp_id),
    )

    _PAIR_EVEN = (((0, None), None), None, None)
    _PAIR_ODD = (((1, None), None), None, None)
    _rAcc_half_probe = tTR_rAcc[_PAIR_EVEN]
    tTR_rC_half = cute.make_rmem_tensor(_rAcc_half_probe.shape, gemm_kernel.c_dtype)

    # TMA partition over the compact fold output. sC: (EPI_M, EPI_N2, STAGE).
    gC_fold_epi = cute.flat_divide(gC_fold_mnl, gemm_kernel.epi_tile_c)
    bSG_sC, bSG_gC_all = cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        cute.group_modes(sC, 0, 2),
        cute.group_modes(gC_fold_epi, 0, 2),
    )
    bSG_gC = bSG_gC_all[(None, None, None, *mma_tile_coord_mnl)]
    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

    if const_expr(gemm_kernel.overlapping_accum):
        acc_stage_index = acc_consumer_state.phase
        reverse_subtile = Boolean(True) if acc_stage_index == 0 else Boolean(False)
    else:
        acc_stage_index = acc_consumer_state.index
    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]
    acc_pipeline.consumer_wait(acc_consumer_state)
    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
    num_prev_subtiles = num_tiles_executed * subtile_cnt

    for subtile_idx in range(subtile_cnt):
        real_subtile_idx = subtile_idx
        if const_expr(gemm_kernel.overlapping_accum):
            # Phase 0 drains in reverse so the shared columns free first.
            if reverse_subtile:
                real_subtile_idx = subtile_cnt - 1 - subtile_idx
        tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

        if const_expr(gemm_kernel.overlapping_accum):
            release_at = gemm_kernel.iter_acc_early_release_in_epilogue
        else:
            release_at = subtile_cnt - 1
        if subtile_idx == release_at:
            cute.arch.fence_view_async_tmem_load()
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_consumer_state)
            acc_consumer_state.advance()

        folded_half = epilogue_op(
            tTR_rAcc[_PAIR_EVEN].load().to(cutlass.Float32)
            + tTR_rAcc[_PAIR_ODD].load().to(cutlass.Float32)
        )
        tTR_rC_half.store(folded_half.to(gemm_kernel.c_dtype))

        # Each thread owns one row of the half-width epi tile (the fragment
        # has no M/N repeat modes); write it straight into plain smem.
        c_buffer = (num_prev_subtiles + subtile_idx) % gemm_kernel.num_c_stage
        sC_row = sC[(epi_tidx, None, c_buffer)]
        # Transposed staging makes the destination strided (n_w-contiguous
        # smem): scalar element writes, rank-agnostic via linear indexing.
        for i in range(cute.size(sC_row.shape)):
            sC_row[i] = tTR_rC_half[i]

        cute.arch.fence_proxy("async.shared", space="cta")
        epilog_sync_barrier.arrive_and_wait()

        if warp_idx == gemm_kernel.epilogue_warp_id[0]:
            cute.copy(
                tma_atom_c,
                bSG_sC[(None, c_buffer)],
                bSG_gC[(None, real_subtile_idx)],
            )
            c_pipeline.producer_commit()
            c_pipeline.producer_acquire()
        epilog_sync_barrier.arrive_and_wait()

    return acc_consumer_state
