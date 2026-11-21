# Copyright (c) 2025 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
# Vendored from flashinfer-ai/flashinfer#9003 (epilogue_utils.py) at d999616a3 for FlashInfer 0.7.2;
# drop it once sglang pins a FlashInfer release with bmm_mxfp8(out_scale=...).

from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir.dialects import math as mlir_math
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import Constexpr, Int32
from cutlass.utils.gemm.sm100 import (
    epilogue_smem_copy_and_partition,
    epilogue_tmem_copy_and_partition,
    transform_partitioned_tensor_layout,
)


@cute.jit
def epilogue_tma_store_mxfp8(
    gemm_kernel,
    epi_tidx: Int32,
    warp_idx: Int32,
    tma_atom_c: cute.CopyAtom,
    # Input of epilogue
    tCtAcc_base: cute.Tensor,
    # Staging of epilogue
    sC: cute.Tensor,
    # Output of epilogue
    tCgC_base: cute.Tensor,
    tCgSFC_base: cute.Tensor,
    epi_tile: cute.Tile,
    num_tiles_executed: Int32,
    epilogue_op: Constexpr,
    alpha_value: cutlass.Float32,
    mma_tile_coord_mnl: Tuple[Int32, Int32, Int32],
    acc_consumer_state: pipeline.PipelineState,
    acc_pipeline: pipeline.PipelineAsync,
    c_pipeline: pipeline.PipelineTmaStore,
) -> pipeline.PipelineState:
    """
    Per-tile epilogue that writes MXFP8: Float8E4M3FN values plus one UE8M0 scale per 32 output columns.

    Each value is scaled by alpha in Float32, rounded to bf16 and passed through ``epilogue_op``
    (as the bf16 epilogue applies it), then quantized exactly as
    ``mxfp8_quantize`` would quantize that bf16 tensor (round-up UE8M0 of absmax / 448, then
    a round-to-nearest saturating e4m3 conversion), so the bytes match a bf16 GEMM followed by
    ``mxfp8_quantize(..., is_sf_swizzled_layout=True)``. ``tCgSFC_base`` is the scale tensor
    partitioned like ``tCgC_base``; its stride is 0 across each 32-column block.
    """
    from flashinfer.quantization.quantization_cute_dsl_utils import (
        float_to_ue8m0_fast,
        ue8m0_to_inv_scale_fast,
    )

    tCgC = transform_partitioned_tensor_layout(tCgC_base)
    tCgSFC = transform_partitioned_tensor_layout(tCgSFC_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)

    tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, tCtAcc, tCgC, epi_tile, gemm_kernel.use_2cta_instrs
    )

    tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, gemm_kernel.c_dtype)
    tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
        gemm_kernel, tiled_copy_t2r, tTR_rC, epi_tidx, sC
    )

    tCgC_epi = cute.flat_divide(tCgC, epi_tile)
    bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        cute.group_modes(sC, 0, 2),
        cute.group_modes(tCgC_epi, 0, 2),
    )
    thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
    tTR_gSFC = thr_copy_t2r.partition_D(cute.flat_divide(tCgSFC, epi_tile))[
        (None, None, None, None, None, *mma_tile_coord_mnl)
    ]
    tTR_gSFC = cute.group_modes(tTR_gSFC, 3, cute.rank(tTR_gSFC))

    epilog_sync_barrier = pipeline.NamedBarrier(
        barrier_id=gemm_kernel.epilog_sync_bar_id,
        num_threads=32 * len(gemm_kernel.epilogue_warp_id),
    )

    bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

    acc_pipeline.consumer_wait(acc_consumer_state)

    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
    num_prev_subtiles = num_tiles_executed * subtile_cnt
    for subtile_idx in range(subtile_cnt):
        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
        tTR_rAcc.store(
            epilogue_op((tTR_rAcc.load() * alpha_value).to(cutlass.BFloat16)).to(
                cutlass.Float32
            )
        )

        acc_frg = cute.logical_divide(tTR_rAcc, cute.make_layout(32))
        sfc_frg = cute.logical_divide(
            tTR_gSFC[(None, None, None, subtile_idx)], cute.make_layout(32)
        )
        acc_ssa = acc_frg.load()
        abs_ssa = type(acc_ssa)(
            mlir_math.absf(acc_ssa.ir_value()), acc_ssa.shape, acc_ssa.dtype
        )
        for vi in cutlass.range_constexpr(cute.size(acc_frg, mode=[1])):
            amax = abs_ssa[None, vi].reduce(
                cute.ReductionOp.MAX, cutlass.Float32(0.0), 0
            )
            code = float_to_ue8m0_fast(amax * cutlass.Float32(1.0 / 448.0))
            sfc_frg[0, vi] = code.to(cutlass.Uint8)
            inv_scale = ue8m0_to_inv_scale_fast(code)
            vec = acc_frg[None, vi]
            for ei in cutlass.range_constexpr(32):
                vec[ei] = vec[ei] * inv_scale

        tRS_rC.store(tiled_copy_r2s.retile(tTR_rAcc).load().to(gemm_kernel.c_dtype))

        c_buffer = (num_prev_subtiles + subtile_idx) % gemm_kernel.num_c_stage
        cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
        cute.arch.fence_proxy("async.shared", space="cta")
        epilog_sync_barrier.arrive_and_wait()

        if warp_idx == gemm_kernel.epilogue_warp_id[0]:
            cute.copy(
                tma_atom_c,
                bSG_sC[(None, c_buffer)],
                bSG_gC[(None, subtile_idx)],
            )
            c_pipeline.producer_commit()
            c_pipeline.producer_acquire()
        epilog_sync_barrier.arrive_and_wait()

    epilog_sync_barrier.arrive_and_wait()

    with cute.arch.elect_one():
        acc_pipeline.consumer_release(acc_consumer_state)
    acc_consumer_state.advance()
    return acc_consumer_state
