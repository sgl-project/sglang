"""CuTe DSL Fused Sigmoid Gating Delta Rule Kernel for GDN Decode."""

import logging
from typing import Dict, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

logger = logging.getLogger(__name__)

_compiled_kernels: Dict[Tuple, object] = {}
_cu_seqlens_cache: Dict[Tuple, torch.Tensor] = {}
TILE_K = 128
TILE_V = 32
TILE_V_PADDED = 36
TILE_V_SMALL = 16
TILE_V_SMALL_PADDED = 20
NUM_STAGES = 2
NUM_THREADS = 128
NUM_BLOCKS_PER_STATE_SMALL = 8
NUM_THREADS_LARGE = 256
NUM_WARPS_LARGE = 8
V_PER_WARP = 4
ROWS_PER_ITER = 8
NUM_K_ITERS = TILE_K // ROWS_PER_ITER
SMALL_BATCH_THRESHOLD = 32


def _define_kernels():
    """Define CuTe DSL kernels for normal and varlen decode modes."""

    NUM_WARPS_SMALL = 4
    V_PER_WARP_SMALL = TILE_V_SMALL // NUM_WARPS_SMALL
    ROWS_PER_ITER_SMALL = 32 // V_PER_WARP_SMALL
    NUM_K_ITERS_SMALL = TILE_K // ROWS_PER_ITER_SMALL

    @cute.kernel
    def gdn_kernel_small_batch(
        tiled_copy_load: cute.TiledCopy,
        h0_source: cute.Tensor,
        smem_layout_staged: cute.Layout,
        num_v_tiles: cutlass.Constexpr[int],
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        o: cute.Tensor,
        h0_indices: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
    ):
        """Small batch kernel for (N, 1, ...) format."""
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        block_idx, _, _ = cute.arch.block_idx()

        batch_idx = block_idx // NUM_BLOCKS_PER_STATE_SMALL
        batch_inner = block_idx % NUM_BLOCKS_PER_STATE_SMALL
        num_v_tiles_per_block = num_v_tiles // NUM_BLOCKS_PER_STATE_SMALL
        start_v_tile = batch_inner * num_v_tiles_per_block

        i_n = batch_idx // HV
        i_hv = batch_idx % HV
        i_h = i_hv // (HV // H)

        pool_idx = h0_indices[i_n]

        if pool_idx >= 0:
            k_local = in_warp_tid // V_PER_WARP_SMALL
            v_local = in_warp_tid % V_PER_WARP_SMALL
            v_base = warp_idx * V_PER_WARP_SMALL
            v_idx = v_base + v_local

            smem = cutlass.utils.SmemAllocator()
            sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
            smem_o_layout = cute.make_layout((TILE_V_SMALL,), stride=(1,))
            smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
            smem_k_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_q_layout = cute.make_layout((TILE_K,), stride=(1,))
            sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
            sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)

            if tidx < TILE_K:
                sK[tidx] = cutlass.Float32(k[i_n, 0, i_h, tidx])
                sQ[tidx] = cutlass.Float32(q[i_n, 0, i_h, tidx])

            gSrc_batch = h0_source[(pool_idx, i_hv, None, None)]
            gSrc = cute.local_tile(gSrc_batch, (TILE_K, TILE_V_SMALL), (0, None))
            thr_copy_load = tiled_copy_load.get_slice(tidx)

            prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles_per_block)
            for v_tile_offset in range(prefetch_count):
                v_tile = start_v_tile + v_tile_offset
                stage = v_tile_offset % NUM_STAGES
                gSrc_tile = gSrc[(None, None, v_tile)]
                sData_stage = sData[(None, None, stage)]
                thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
                thr_sData = thr_copy_load.partition_D(sData_stage)
                cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                cute.arch.cp_async_commit_group()

            r_A_log = cutlass.Float32(A_log[i_hv])
            r_dt_bias = cutlass.Float32(dt_bias[i_hv])
            r_a = cutlass.Float32(a[i_n, 0, i_hv])
            r_b = cutlass.Float32(b[i_n, 0, i_hv])

            r_g = 0.0
            r_beta = 0.0
            if in_warp_tid == 0:
                x = r_a + r_dt_bias
                beta_x = softplus_beta * x
                softplus_x = 0.0
                if beta_x <= softplus_threshold:
                    exp_beta_x = cute.exp(beta_x)
                    log_input = cutlass.Float32(1.0 + exp_beta_x)
                    log_result = cutlass.Float32(cute.log(log_input))
                    softplus_x = cutlass.Float32(
                        (cutlass.Float32(1.0) / softplus_beta) * log_result
                    )
                else:
                    softplus_x = x
                r_g_value = -cute.exp(r_A_log) * softplus_x
                r_beta = 1.0 / (1.0 + cute.exp(-r_b))
                r_g = cute.exp(r_g_value)

            r_g = cute.arch.shuffle_sync(r_g, 0)
            r_beta = cute.arch.shuffle_sync(r_beta, 0)

            cute.arch.barrier()

            if use_qk_l2norm:
                sum_q_partial = 0.0
                sum_k_partial = 0.0
                if tidx < TILE_K:
                    q_val = sQ[tidx]
                    k_val = sK[tidx]
                    sum_q_partial = q_val * q_val
                    sum_k_partial = k_val * k_val

                for offset in [16, 8, 4, 2, 1]:
                    sum_q_partial += cute.arch.shuffle_sync_bfly(
                        sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    sum_k_partial += cute.arch.shuffle_sync_bfly(
                        sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31
                    )

                if in_warp_tid == 0:
                    smem_o[warp_idx] = sum_q_partial
                    smem_o[warp_idx + 4] = sum_k_partial
                cute.arch.barrier()

                inv_norm_q = 0.0
                inv_norm_k = 0.0
                if warp_idx == 0:
                    local_sum_q = 0.0
                    local_sum_k = 0.0
                    if in_warp_tid < NUM_WARPS_SMALL:
                        local_sum_q = smem_o[in_warp_tid]
                        local_sum_k = smem_o[in_warp_tid + 4]
                    for offset in [2, 1]:
                        local_sum_q += cute.arch.shuffle_sync_bfly(
                            local_sum_q, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        local_sum_k += cute.arch.shuffle_sync_bfly(
                            local_sum_k, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    if in_warp_tid == 0:
                        smem_o[0] = cute.rsqrt(local_sum_q + 1e-6)
                        smem_o[1] = cute.rsqrt(local_sum_k + 1e-6)
                cute.arch.barrier()

                inv_norm_q = smem_o[0]
                inv_norm_k = smem_o[1]

                if tidx < TILE_K:
                    sK[tidx] = sK[tidx] * inv_norm_k
                    sQ[tidx] = sQ[tidx] * scale * inv_norm_q
                cute.arch.barrier()
            else:
                if tidx < TILE_K:
                    sQ[tidx] = sQ[tidx] * scale
                cute.arch.barrier()

            for v_tile_offset in range(num_v_tiles_per_block):
                v_tile = start_v_tile + v_tile_offset
                stage = v_tile_offset % NUM_STAGES

                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                next_v_tile_offset = v_tile_offset + prefetch_count
                if next_v_tile_offset < num_v_tiles_per_block:
                    next_v_tile = start_v_tile + next_v_tile_offset
                    next_stage = next_v_tile_offset % NUM_STAGES
                    gSrc_next = gSrc[(None, None, next_v_tile)]
                    sData_next = sData[(None, None, next_stage)]
                    thr_gSrc = thr_copy_load.partition_S(gSrc_next)
                    thr_sData = thr_copy_load.partition_D(sData_next)
                    cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                    cute.arch.cp_async_commit_group()

                v_global = v_tile * TILE_V_SMALL + v_idx
                r_v = cutlass.Float32(v[i_n, 0, i_hv, v_global])

                sum_hk = 0.0
                for k_iter in cutlass.range_dynamic(NUM_K_ITERS_SMALL, unroll=16):
                    k_base = k_iter * ROWS_PER_ITER_SMALL
                    k_idx = k_base + k_local
                    h_val = sData[(k_idx, v_idx, stage)] * r_g
                    r_k_val = sK[k_idx]
                    sum_hk += h_val * r_k_val

                for offset in [4, 2, 1]:
                    sum_hk += cute.arch.shuffle_sync_bfly(
                        sum_hk,
                        offset=offset * V_PER_WARP_SMALL,
                        mask=-1,
                        mask_and_clamp=31,
                    )

                v_new = (r_v - sum_hk) * r_beta
                v_new = cute.arch.shuffle_sync(v_new, v_local)

                sum_hq = 0.0
                for k_iter in cutlass.range_dynamic(NUM_K_ITERS_SMALL, unroll=16):
                    k_base = k_iter * ROWS_PER_ITER_SMALL
                    k_idx = k_base + k_local
                    h_old = sData[(k_idx, v_idx, stage)] * r_g
                    r_k_val = sK[k_idx]
                    r_q_val = sQ[k_idx]
                    h_new = h_old + r_k_val * v_new
                    sData[(k_idx, v_idx, stage)] = h_new
                    sum_hq += h_new * r_q_val

                for offset in [4, 2, 1]:
                    sum_hq += cute.arch.shuffle_sync_bfly(
                        sum_hq,
                        offset=offset * V_PER_WARP_SMALL,
                        mask=-1,
                        mask_and_clamp=31,
                    )

                if k_local == 0:
                    v_global_out = v_tile * TILE_V_SMALL + v_idx
                    o[(i_n, 0, i_hv, v_global_out)] = cutlass.BFloat16(sum_hq)

                cute.arch.barrier()

                for k_iter in range(NUM_K_ITERS_SMALL):
                    flat_idx = tidx + k_iter * 128
                    k_write = flat_idx // TILE_V_SMALL
                    v_write = flat_idx % TILE_V_SMALL
                    if k_write < TILE_K:
                        h_val = sData[(k_write, v_write, stage)]
                        v_global_write = v_tile * TILE_V_SMALL + v_write
                        h0_source[(pool_idx, i_hv, k_write, v_global_write)] = h_val

                cute.arch.barrier()

    @cute.kernel
    def gdn_kernel_small_batch_varlen(
        tiled_copy_load: cute.TiledCopy,
        h0_source: cute.Tensor,
        smem_layout_staged: cute.Layout,
        num_v_tiles: cutlass.Constexpr[int],
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        o: cute.Tensor,
        h0_indices: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
    ):
        """Small batch kernel for varlen decode (1, N, ...) format."""
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        block_idx, _, _ = cute.arch.block_idx()

        batch_idx = block_idx // NUM_BLOCKS_PER_STATE_SMALL
        batch_inner = block_idx % NUM_BLOCKS_PER_STATE_SMALL
        num_v_tiles_per_block = num_v_tiles // NUM_BLOCKS_PER_STATE_SMALL
        start_v_tile = batch_inner * num_v_tiles_per_block

        i_n = batch_idx // HV
        i_hv = batch_idx % HV
        i_h = i_hv // (HV // H)

        pool_idx = h0_indices[i_n]

        if pool_idx >= 0:
            k_local = in_warp_tid // V_PER_WARP_SMALL
            v_local = in_warp_tid % V_PER_WARP_SMALL
            v_base = warp_idx * V_PER_WARP_SMALL
            v_idx = v_base + v_local

            smem = cutlass.utils.SmemAllocator()
            sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
            smem_o_layout = cute.make_layout((TILE_V_SMALL,), stride=(1,))
            smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
            smem_k_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_q_layout = cute.make_layout((TILE_K,), stride=(1,))
            sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
            sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)

            if tidx < TILE_K:
                sK[tidx] = cutlass.Float32(k[0, i_n, i_h, tidx])
                sQ[tidx] = cutlass.Float32(q[0, i_n, i_h, tidx])

            gSrc_batch = h0_source[(pool_idx, i_hv, None, None)]
            gSrc = cute.local_tile(gSrc_batch, (TILE_K, TILE_V_SMALL), (0, None))
            thr_copy_load = tiled_copy_load.get_slice(tidx)

            prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles_per_block)
            for v_tile_offset in range(prefetch_count):
                v_tile = start_v_tile + v_tile_offset
                stage = v_tile_offset % NUM_STAGES
                gSrc_tile = gSrc[(None, None, v_tile)]
                sData_stage = sData[(None, None, stage)]
                thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
                thr_sData = thr_copy_load.partition_D(sData_stage)
                cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                cute.arch.cp_async_commit_group()

            r_A_log = cutlass.Float32(A_log[i_hv])
            r_dt_bias = cutlass.Float32(dt_bias[i_hv])
            r_a = cutlass.Float32(a[i_n, i_hv])
            r_b = cutlass.Float32(b[i_n, i_hv])

            r_g = 0.0
            r_beta = 0.0
            if in_warp_tid == 0:
                x = r_a + r_dt_bias
                beta_x = softplus_beta * x
                softplus_x = 0.0
                if beta_x <= softplus_threshold:
                    exp_beta_x = cute.exp(beta_x)
                    log_input = cutlass.Float32(1.0 + exp_beta_x)
                    log_result = cutlass.Float32(cute.log(log_input))
                    softplus_x = cutlass.Float32(
                        (cutlass.Float32(1.0) / softplus_beta) * log_result
                    )
                else:
                    softplus_x = x
                r_g_value = -cute.exp(r_A_log) * softplus_x
                r_beta = 1.0 / (1.0 + cute.exp(-r_b))
                r_g = cute.exp(r_g_value)

            r_g = cute.arch.shuffle_sync(r_g, 0)
            r_beta = cute.arch.shuffle_sync(r_beta, 0)

            cute.arch.barrier()

            if use_qk_l2norm:
                sum_q_partial = 0.0
                sum_k_partial = 0.0
                if tidx < TILE_K:
                    q_val = sQ[tidx]
                    k_val = sK[tidx]
                    sum_q_partial = q_val * q_val
                    sum_k_partial = k_val * k_val

                for offset in [16, 8, 4, 2, 1]:
                    sum_q_partial += cute.arch.shuffle_sync_bfly(
                        sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    sum_k_partial += cute.arch.shuffle_sync_bfly(
                        sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31
                    )

                if in_warp_tid == 0:
                    smem_o[warp_idx] = sum_q_partial
                    smem_o[warp_idx + 4] = sum_k_partial
                cute.arch.barrier()

                inv_norm_q = 0.0
                inv_norm_k = 0.0
                if warp_idx == 0:
                    local_sum_q = 0.0
                    local_sum_k = 0.0
                    if in_warp_tid < NUM_WARPS_SMALL:
                        local_sum_q = smem_o[in_warp_tid]
                        local_sum_k = smem_o[in_warp_tid + 4]
                    for offset in [2, 1]:
                        local_sum_q += cute.arch.shuffle_sync_bfly(
                            local_sum_q, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        local_sum_k += cute.arch.shuffle_sync_bfly(
                            local_sum_k, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    if in_warp_tid == 0:
                        smem_o[0] = cute.rsqrt(local_sum_q + 1e-6)
                        smem_o[1] = cute.rsqrt(local_sum_k + 1e-6)
                cute.arch.barrier()

                inv_norm_q = smem_o[0]
                inv_norm_k = smem_o[1]

                if tidx < TILE_K:
                    sK[tidx] = sK[tidx] * inv_norm_k
                    sQ[tidx] = sQ[tidx] * scale * inv_norm_q
                cute.arch.barrier()
            else:
                if tidx < TILE_K:
                    sQ[tidx] = sQ[tidx] * scale
                cute.arch.barrier()

            for v_tile_offset in range(num_v_tiles_per_block):
                v_tile = start_v_tile + v_tile_offset
                stage = v_tile_offset % NUM_STAGES

                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                next_v_tile_offset = v_tile_offset + prefetch_count
                if next_v_tile_offset < num_v_tiles_per_block:
                    next_v_tile = start_v_tile + next_v_tile_offset
                    next_stage = next_v_tile_offset % NUM_STAGES
                    gSrc_next = gSrc[(None, None, next_v_tile)]
                    sData_next = sData[(None, None, next_stage)]
                    thr_gSrc = thr_copy_load.partition_S(gSrc_next)
                    thr_sData = thr_copy_load.partition_D(sData_next)
                    cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                    cute.arch.cp_async_commit_group()

                v_global = v_tile * TILE_V_SMALL + v_idx
                r_v = cutlass.Float32(v[0, i_n, i_hv, v_global])

                sum_hk = 0.0
                for k_iter in cutlass.range_dynamic(NUM_K_ITERS_SMALL, unroll=16):
                    k_base = k_iter * ROWS_PER_ITER_SMALL
                    k_idx = k_base + k_local
                    h_val = sData[(k_idx, v_idx, stage)] * r_g
                    r_k_val = sK[k_idx]
                    sum_hk += h_val * r_k_val

                for offset in [4, 2, 1]:
                    sum_hk += cute.arch.shuffle_sync_bfly(
                        sum_hk,
                        offset=offset * V_PER_WARP_SMALL,
                        mask=-1,
                        mask_and_clamp=31,
                    )

                v_new = (r_v - sum_hk) * r_beta
                v_new = cute.arch.shuffle_sync(v_new, v_local)

                sum_hq = 0.0
                for k_iter in cutlass.range_dynamic(NUM_K_ITERS_SMALL, unroll=16):
                    k_base = k_iter * ROWS_PER_ITER_SMALL
                    k_idx = k_base + k_local
                    h_old = sData[(k_idx, v_idx, stage)] * r_g
                    r_k_val = sK[k_idx]
                    r_q_val = sQ[k_idx]
                    h_new = h_old + r_k_val * v_new
                    sData[(k_idx, v_idx, stage)] = h_new
                    sum_hq += h_new * r_q_val

                for offset in [4, 2, 1]:
                    sum_hq += cute.arch.shuffle_sync_bfly(
                        sum_hq,
                        offset=offset * V_PER_WARP_SMALL,
                        mask=-1,
                        mask_and_clamp=31,
                    )

                if k_local == 0:
                    v_global_out = v_tile * TILE_V_SMALL + v_idx
                    o[(0, i_n, i_hv, v_global_out)] = cutlass.BFloat16(sum_hq)

                cute.arch.barrier()

                for k_iter in range(NUM_K_ITERS_SMALL):
                    flat_idx = tidx + k_iter * 128
                    k_write = flat_idx // TILE_V_SMALL
                    v_write = flat_idx % TILE_V_SMALL
                    if k_write < TILE_K:
                        h_val = sData[(k_write, v_write, stage)]
                        v_global_write = v_tile * TILE_V_SMALL + v_write
                        h0_source[(pool_idx, i_hv, k_write, v_global_write)] = h_val

                cute.arch.barrier()

    @cute.kernel
    def gdn_kernel_large_batch(
        tiled_copy_load: cute.TiledCopy,
        h0_source: cute.Tensor,
        smem_layout_staged: cute.Layout,
        num_v_tiles: cutlass.Constexpr[int],
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        o: cute.Tensor,
        h0_indices: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
    ):
        """Large batch kernel for (N, 1, ...) format."""
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        batch_idx, _, _ = cute.arch.block_idx()
        i_n = batch_idx // HV
        i_hv = batch_idx % HV
        i_h = i_hv // (HV // H)

        pool_idx = h0_indices[i_n]

        if pool_idx >= 0:
            k_local = in_warp_tid // V_PER_WARP
            v_local = in_warp_tid % V_PER_WARP
            v_base = warp_idx * V_PER_WARP
            v_idx = v_base + v_local

            smem = cutlass.utils.SmemAllocator()
            sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
            smem_o_layout = cute.make_layout((TILE_V,), stride=(1,))
            smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
            smem_k_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_q_layout = cute.make_layout((TILE_K,), stride=(1,))
            sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
            sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)

            if tidx < TILE_K:
                sK[tidx] = cutlass.Float32(k[i_n, 0, i_h, tidx])
                sQ[tidx] = cutlass.Float32(q[i_n, 0, i_h, tidx])

            gSrc_batch = h0_source[(pool_idx, i_hv, None, None)]
            gSrc = cute.local_tile(gSrc_batch, (TILE_K, TILE_V), (0, None))
            thr_copy_load = tiled_copy_load.get_slice(tidx)

            prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles)
            for v_tile in range(prefetch_count):
                stage = v_tile % NUM_STAGES
                gSrc_tile = gSrc[(None, None, v_tile)]
                sData_stage = sData[(None, None, stage)]
                thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
                thr_sData = thr_copy_load.partition_D(sData_stage)
                cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                cute.arch.cp_async_commit_group()

            r_A_log = cutlass.Float32(A_log[i_hv])
            r_dt_bias = cutlass.Float32(dt_bias[i_hv])
            r_a = cutlass.Float32(a[i_n, 0, i_hv])
            r_b = cutlass.Float32(b[i_n, 0, i_hv])

            r_g = 0.0
            r_beta = 0.0
            if in_warp_tid == 0:
                x = r_a + r_dt_bias
                beta_x = softplus_beta * x
                softplus_x = 0.0
                if beta_x <= softplus_threshold:
                    exp_beta_x = cute.exp(beta_x)
                    log_input = cutlass.Float32(1.0 + exp_beta_x)
                    log_result = cutlass.Float32(cute.log(log_input))
                    softplus_x = cutlass.Float32(
                        (cutlass.Float32(1.0) / softplus_beta) * log_result
                    )
                else:
                    softplus_x = x
                r_g_value = -cute.exp(r_A_log) * softplus_x
                r_beta = 1.0 / (1.0 + cute.exp(-r_b))
                r_g = cute.exp(r_g_value)

            r_g = cute.arch.shuffle_sync(r_g, 0)
            r_beta = cute.arch.shuffle_sync(r_beta, 0)

            cute.arch.barrier()

            if use_qk_l2norm:
                sum_q_partial = 0.0
                sum_k_partial = 0.0
                if tidx < TILE_K:
                    q_val = sQ[tidx]
                    k_val = sK[tidx]
                    sum_q_partial = q_val * q_val
                    sum_k_partial = k_val * k_val

                for offset in [16, 8, 4, 2, 1]:
                    sum_q_partial += cute.arch.shuffle_sync_bfly(
                        sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    sum_k_partial += cute.arch.shuffle_sync_bfly(
                        sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31
                    )

                if in_warp_tid == 0:
                    smem_o[warp_idx] = sum_q_partial
                    smem_o[warp_idx + 8] = sum_k_partial
                cute.arch.barrier()

                inv_norm_q = 0.0
                inv_norm_k = 0.0
                if warp_idx == 0:
                    local_sum_q = 0.0
                    local_sum_k = 0.0
                    if in_warp_tid < NUM_WARPS_LARGE:
                        local_sum_q = smem_o[in_warp_tid]
                        local_sum_k = smem_o[in_warp_tid + 8]
                    for offset in [4, 2, 1]:
                        local_sum_q += cute.arch.shuffle_sync_bfly(
                            local_sum_q, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        local_sum_k += cute.arch.shuffle_sync_bfly(
                            local_sum_k, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    if in_warp_tid == 0:
                        smem_o[0] = cute.rsqrt(local_sum_q + 1e-6)
                        smem_o[1] = cute.rsqrt(local_sum_k + 1e-6)
                cute.arch.barrier()

                inv_norm_q = smem_o[0]
                inv_norm_k = smem_o[1]

                if tidx < TILE_K:
                    sK[tidx] = sK[tidx] * inv_norm_k
                    sQ[tidx] = sQ[tidx] * scale * inv_norm_q
                cute.arch.barrier()
            else:
                if tidx < TILE_K:
                    sQ[tidx] = sQ[tidx] * scale
                cute.arch.barrier()

            for v_tile in range(num_v_tiles):
                stage = v_tile % NUM_STAGES

                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                next_v_tile = v_tile + prefetch_count
                if next_v_tile < num_v_tiles:
                    next_stage = next_v_tile % NUM_STAGES
                    gSrc_next = gSrc[(None, None, next_v_tile)]
                    sData_next = sData[(None, None, next_stage)]
                    thr_gSrc = thr_copy_load.partition_S(gSrc_next)
                    thr_sData = thr_copy_load.partition_D(sData_next)
                    cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                    cute.arch.cp_async_commit_group()

                v_global = v_tile * TILE_V + v_idx
                r_v = cutlass.Float32(v[i_n, 0, i_hv, v_global])

                sum_hk = 0.0
                for k_iter in cutlass.range_dynamic(NUM_K_ITERS, unroll=8):
                    k_base = k_iter * ROWS_PER_ITER
                    k_idx = k_base + k_local
                    h_val = sData[(k_idx, v_idx, stage)] * r_g
                    r_k_val = sK[k_idx]
                    sum_hk += h_val * r_k_val

                for offset in [4, 2, 1]:
                    sum_hk += cute.arch.shuffle_sync_bfly(
                        sum_hk, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31
                    )

                v_new = (r_v - sum_hk) * r_beta
                v_new = cute.arch.shuffle_sync(v_new, v_local)

                sum_hq = 0.0
                for k_iter in cutlass.range_dynamic(NUM_K_ITERS, unroll=8):
                    k_base = k_iter * ROWS_PER_ITER
                    k_idx = k_base + k_local
                    h_old = sData[(k_idx, v_idx, stage)] * r_g
                    r_k_val = sK[k_idx]
                    r_q_val = sQ[k_idx]
                    h_new = h_old + r_k_val * v_new
                    sData[(k_idx, v_idx, stage)] = h_new
                    sum_hq += h_new * r_q_val

                for offset in [4, 2, 1]:
                    sum_hq += cute.arch.shuffle_sync_bfly(
                        sum_hq, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31
                    )

                if k_local == 0:
                    v_global_out = v_tile * TILE_V + v_idx
                    o[(i_n, 0, i_hv, v_global_out)] = cutlass.BFloat16(sum_hq)

                cute.arch.barrier()

                for k_iter in range(NUM_K_ITERS):
                    flat_idx = tidx + k_iter * 256
                    k_write = flat_idx // TILE_V
                    v_write = flat_idx % TILE_V
                    if k_write < TILE_K:
                        h_val = sData[(k_write, v_write, stage)]
                        v_global_write = v_tile * TILE_V + v_write
                        h0_source[(pool_idx, i_hv, k_write, v_global_write)] = h_val

                cute.arch.barrier()

    @cute.kernel
    def gdn_kernel_large_batch_varlen(
        tiled_copy_load: cute.TiledCopy,
        h0_source: cute.Tensor,
        smem_layout_staged: cute.Layout,
        num_v_tiles: cutlass.Constexpr[int],
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        o: cute.Tensor,
        h0_indices: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
    ):
        """Large batch kernel for varlen decode (1, N, ...) format."""
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        batch_idx, _, _ = cute.arch.block_idx()
        i_n = batch_idx // HV
        i_hv = batch_idx % HV
        i_h = i_hv // (HV // H)

        pool_idx = h0_indices[i_n]

        if pool_idx >= 0:
            k_local = in_warp_tid // V_PER_WARP
            v_local = in_warp_tid % V_PER_WARP
            v_base = warp_idx * V_PER_WARP
            v_idx = v_base + v_local

            smem = cutlass.utils.SmemAllocator()
            sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
            smem_o_layout = cute.make_layout((TILE_V,), stride=(1,))
            smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
            smem_k_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_q_layout = cute.make_layout((TILE_K,), stride=(1,))
            sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
            sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)

            if tidx < TILE_K:
                sK[tidx] = cutlass.Float32(k[0, i_n, i_h, tidx])
                sQ[tidx] = cutlass.Float32(q[0, i_n, i_h, tidx])

            gSrc_batch = h0_source[(pool_idx, i_hv, None, None)]
            gSrc = cute.local_tile(gSrc_batch, (TILE_K, TILE_V), (0, None))
            thr_copy_load = tiled_copy_load.get_slice(tidx)

            prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles)
            for v_tile in range(prefetch_count):
                stage = v_tile % NUM_STAGES
                gSrc_tile = gSrc[(None, None, v_tile)]
                sData_stage = sData[(None, None, stage)]
                thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
                thr_sData = thr_copy_load.partition_D(sData_stage)
                cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                cute.arch.cp_async_commit_group()

            r_A_log = cutlass.Float32(A_log[i_hv])
            r_dt_bias = cutlass.Float32(dt_bias[i_hv])
            r_a = cutlass.Float32(a[i_n, i_hv])
            r_b = cutlass.Float32(b[i_n, i_hv])

            r_g = 0.0
            r_beta = 0.0
            if in_warp_tid == 0:
                x = r_a + r_dt_bias
                beta_x = softplus_beta * x
                softplus_x = 0.0
                if beta_x <= softplus_threshold:
                    exp_beta_x = cute.exp(beta_x)
                    log_input = cutlass.Float32(1.0 + exp_beta_x)
                    log_result = cutlass.Float32(cute.log(log_input))
                    softplus_x = cutlass.Float32(
                        (cutlass.Float32(1.0) / softplus_beta) * log_result
                    )
                else:
                    softplus_x = x
                r_g_value = -cute.exp(r_A_log) * softplus_x
                r_beta = 1.0 / (1.0 + cute.exp(-r_b))
                r_g = cute.exp(r_g_value)

            r_g = cute.arch.shuffle_sync(r_g, 0)
            r_beta = cute.arch.shuffle_sync(r_beta, 0)

            cute.arch.barrier()

            if use_qk_l2norm:
                sum_q_partial = 0.0
                sum_k_partial = 0.0
                if tidx < TILE_K:
                    q_val = sQ[tidx]
                    k_val = sK[tidx]
                    sum_q_partial = q_val * q_val
                    sum_k_partial = k_val * k_val

                for offset in [16, 8, 4, 2, 1]:
                    sum_q_partial += cute.arch.shuffle_sync_bfly(
                        sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    sum_k_partial += cute.arch.shuffle_sync_bfly(
                        sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31
                    )

                if in_warp_tid == 0:
                    smem_o[warp_idx] = sum_q_partial
                    smem_o[warp_idx + 8] = sum_k_partial
                cute.arch.barrier()

                inv_norm_q = 0.0
                inv_norm_k = 0.0
                if warp_idx == 0:
                    local_sum_q = 0.0
                    local_sum_k = 0.0
                    if in_warp_tid < NUM_WARPS_LARGE:
                        local_sum_q = smem_o[in_warp_tid]
                        local_sum_k = smem_o[in_warp_tid + 8]
                    for offset in [4, 2, 1]:
                        local_sum_q += cute.arch.shuffle_sync_bfly(
                            local_sum_q, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        local_sum_k += cute.arch.shuffle_sync_bfly(
                            local_sum_k, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    if in_warp_tid == 0:
                        smem_o[0] = cute.rsqrt(local_sum_q + 1e-6)
                        smem_o[1] = cute.rsqrt(local_sum_k + 1e-6)
                cute.arch.barrier()

                inv_norm_q = smem_o[0]
                inv_norm_k = smem_o[1]

                if tidx < TILE_K:
                    sK[tidx] = sK[tidx] * inv_norm_k
                    sQ[tidx] = sQ[tidx] * scale * inv_norm_q
                cute.arch.barrier()
            else:
                if tidx < TILE_K:
                    sQ[tidx] = sQ[tidx] * scale
                cute.arch.barrier()

            for v_tile in range(num_v_tiles):
                stage = v_tile % NUM_STAGES

                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                next_v_tile = v_tile + prefetch_count
                if next_v_tile < num_v_tiles:
                    next_stage = next_v_tile % NUM_STAGES
                    gSrc_next = gSrc[(None, None, next_v_tile)]
                    sData_next = sData[(None, None, next_stage)]
                    thr_gSrc = thr_copy_load.partition_S(gSrc_next)
                    thr_sData = thr_copy_load.partition_D(sData_next)
                    cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                    cute.arch.cp_async_commit_group()

                v_global = v_tile * TILE_V + v_idx
                r_v = cutlass.Float32(v[0, i_n, i_hv, v_global])

                sum_hk = 0.0
                for k_iter in cutlass.range_dynamic(NUM_K_ITERS, unroll=8):
                    k_base = k_iter * ROWS_PER_ITER
                    k_idx = k_base + k_local
                    h_val = sData[(k_idx, v_idx, stage)] * r_g
                    r_k_val = sK[k_idx]
                    sum_hk += h_val * r_k_val

                for offset in [4, 2, 1]:
                    sum_hk += cute.arch.shuffle_sync_bfly(
                        sum_hk, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31
                    )

                v_new = (r_v - sum_hk) * r_beta
                v_new = cute.arch.shuffle_sync(v_new, v_local)

                sum_hq = 0.0
                for k_iter in cutlass.range_dynamic(NUM_K_ITERS, unroll=8):
                    k_base = k_iter * ROWS_PER_ITER
                    k_idx = k_base + k_local
                    h_old = sData[(k_idx, v_idx, stage)] * r_g
                    r_k_val = sK[k_idx]
                    r_q_val = sQ[k_idx]
                    h_new = h_old + r_k_val * v_new
                    sData[(k_idx, v_idx, stage)] = h_new
                    sum_hq += h_new * r_q_val

                for offset in [4, 2, 1]:
                    sum_hq += cute.arch.shuffle_sync_bfly(
                        sum_hq, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31
                    )

                if k_local == 0:
                    v_global_out = v_tile * TILE_V + v_idx
                    o[(0, i_n, i_hv, v_global_out)] = cutlass.BFloat16(sum_hq)

                cute.arch.barrier()

                for k_iter in range(NUM_K_ITERS):
                    flat_idx = tidx + k_iter * 256
                    k_write = flat_idx // TILE_V
                    v_write = flat_idx % TILE_V
                    if k_write < TILE_K:
                        h_val = sData[(k_write, v_write, stage)]
                        v_global_write = v_tile * TILE_V + v_write
                        h0_source[(pool_idx, i_hv, k_write, v_global_write)] = h_val

                cute.arch.barrier()

    return (
        gdn_kernel_small_batch,
        gdn_kernel_small_batch_varlen,
        gdn_kernel_large_batch,
        gdn_kernel_large_batch_varlen,
    )


def _create_jit_functions():
    """Create JIT-compiled launcher functions for all kernel variants."""

    gdn_small, gdn_small_varlen, gdn_large, gdn_large_varlen = _define_kernels()

    @cute.jit
    def run_small_batch(
        cu_seqlens: cute.Tensor,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        h0_source: cute.Tensor,
        h0_indices: cute.Tensor,
        o: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        B: cutlass.Constexpr[int],
        T: cutlass.Constexpr[int],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        use_initial_state: cutlass.Constexpr[bool],
        use_qk_l2norm: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        pool_size, hv_dim, k_dim, v_dim = h0_source.layout.shape
        n_indices = h0_indices.layout.shape[0]
        batch_size = n_indices * hv_dim

        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32,
            num_bits_per_copy=128,
        )
        num_v_tiles_small = cute.ceil_div(v_dim, TILE_V_SMALL)
        smem_layout_small = cute.make_layout(
            (TILE_K, TILE_V_SMALL, NUM_STAGES),
            stride=(TILE_V_SMALL_PADDED, 1, TILE_K * TILE_V_SMALL_PADDED),
        )
        thread_layout_small = cute.make_layout((32, 4), stride=(4, 1))
        val_layout_small = cute.make_layout((1, 4))
        tiled_copy_load_small = cute.make_tiled_copy_tv(
            copy_atom, thread_layout_small, val_layout_small
        )
        smem_bytes_small = (
            4 * TILE_K * TILE_V_SMALL_PADDED * NUM_STAGES
            + 4 * TILE_V_SMALL
            + 4 * TILE_K * 2
            + 64
        )

        gdn_small(
            tiled_copy_load_small,
            h0_source,
            smem_layout_small,
            num_v_tiles_small,
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            o,
            h0_indices,
            softplus_beta,
            softplus_threshold,
            scale,
            H,
            HV,
            use_qk_l2norm,
        ).launch(
            grid=(batch_size * NUM_BLOCKS_PER_STATE_SMALL, 1, 1),
            block=[NUM_THREADS, 1, 1],
            smem=smem_bytes_small,
            stream=stream,
        )

    @cute.jit
    def run_small_batch_varlen(
        cu_seqlens: cute.Tensor,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        h0_source: cute.Tensor,
        h0_indices: cute.Tensor,
        o: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        B: cutlass.Constexpr[int],
        T: cutlass.Constexpr[int],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        use_initial_state: cutlass.Constexpr[bool],
        use_qk_l2norm: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        pool_size, hv_dim, k_dim, v_dim = h0_source.layout.shape
        n_indices = h0_indices.layout.shape[0]
        batch_size = n_indices * hv_dim

        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32,
            num_bits_per_copy=128,
        )
        num_v_tiles_small = cute.ceil_div(v_dim, TILE_V_SMALL)
        smem_layout_small = cute.make_layout(
            (TILE_K, TILE_V_SMALL, NUM_STAGES),
            stride=(TILE_V_SMALL_PADDED, 1, TILE_K * TILE_V_SMALL_PADDED),
        )
        thread_layout_small = cute.make_layout((32, 4), stride=(4, 1))
        val_layout_small = cute.make_layout((1, 4))
        tiled_copy_load_small = cute.make_tiled_copy_tv(
            copy_atom, thread_layout_small, val_layout_small
        )
        smem_bytes_small = (
            4 * TILE_K * TILE_V_SMALL_PADDED * NUM_STAGES
            + 4 * TILE_V_SMALL
            + 4 * TILE_K * 2
            + 64
        )

        gdn_small_varlen(
            tiled_copy_load_small,
            h0_source,
            smem_layout_small,
            num_v_tiles_small,
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            o,
            h0_indices,
            softplus_beta,
            softplus_threshold,
            scale,
            H,
            HV,
            use_qk_l2norm,
        ).launch(
            grid=(batch_size * NUM_BLOCKS_PER_STATE_SMALL, 1, 1),
            block=[NUM_THREADS, 1, 1],
            smem=smem_bytes_small,
            stream=stream,
        )

    @cute.jit
    def run_large_batch(
        cu_seqlens: cute.Tensor,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        h0_source: cute.Tensor,
        h0_indices: cute.Tensor,
        o: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        B: cutlass.Constexpr[int],
        T: cutlass.Constexpr[int],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        use_initial_state: cutlass.Constexpr[bool],
        use_qk_l2norm: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        pool_size, hv_dim, k_dim, v_dim = h0_source.layout.shape
        n_indices = h0_indices.layout.shape[0]
        batch_size = n_indices * hv_dim

        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32,
            num_bits_per_copy=128,
        )
        num_v_tiles = cute.ceil_div(v_dim, TILE_V)
        base_smem_layout = cute.make_layout(
            (TILE_K, TILE_V, NUM_STAGES),
            stride=(TILE_V_PADDED, 1, TILE_K * TILE_V_PADDED),
        )
        thread_layout = cute.make_layout((32, 8), stride=(8, 1))
        val_layout = cute.make_layout((1, 4))
        tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)
        smem_bytes = (
            4 * TILE_K * TILE_V_PADDED * NUM_STAGES + 4 * TILE_V + 4 * TILE_K * 2 + 64
        )

        gdn_large(
            tiled_copy_load,
            h0_source,
            base_smem_layout,
            num_v_tiles,
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            o,
            h0_indices,
            softplus_beta,
            softplus_threshold,
            scale,
            H,
            HV,
            use_qk_l2norm,
        ).launch(
            grid=(batch_size, 1, 1),
            block=[NUM_THREADS_LARGE, 1, 1],
            smem=smem_bytes,
            stream=stream,
        )

    @cute.jit
    def run_large_batch_varlen(
        cu_seqlens: cute.Tensor,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        h0_source: cute.Tensor,
        h0_indices: cute.Tensor,
        o: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        B: cutlass.Constexpr[int],
        T: cutlass.Constexpr[int],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        use_initial_state: cutlass.Constexpr[bool],
        use_qk_l2norm: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        pool_size, hv_dim, k_dim, v_dim = h0_source.layout.shape
        n_indices = h0_indices.layout.shape[0]
        batch_size = n_indices * hv_dim

        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32,
            num_bits_per_copy=128,
        )
        num_v_tiles = cute.ceil_div(v_dim, TILE_V)
        base_smem_layout = cute.make_layout(
            (TILE_K, TILE_V, NUM_STAGES),
            stride=(TILE_V_PADDED, 1, TILE_K * TILE_V_PADDED),
        )
        thread_layout = cute.make_layout((32, 8), stride=(8, 1))
        val_layout = cute.make_layout((1, 4))
        tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)
        smem_bytes = (
            4 * TILE_K * TILE_V_PADDED * NUM_STAGES + 4 * TILE_V + 4 * TILE_K * 2 + 64
        )

        gdn_large_varlen(
            tiled_copy_load,
            h0_source,
            base_smem_layout,
            num_v_tiles,
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            o,
            h0_indices,
            softplus_beta,
            softplus_threshold,
            scale,
            H,
            HV,
            use_qk_l2norm,
        ).launch(
            grid=(batch_size, 1, 1),
            block=[NUM_THREADS_LARGE, 1, 1],
            smem=smem_bytes,
            stream=stream,
        )

    return (
        run_small_batch,
        run_small_batch_varlen,
        run_large_batch,
        run_large_batch_varlen,
    )


_jit_functions = None


def _get_jit_functions():
    global _jit_functions
    if _jit_functions is None:
        _jit_functions = _create_jit_functions()
    return _jit_functions


def _get_compiled_kernel(N, H, HV, K, V, pool_size, use_small_batch, is_varlen_decode):
    """Get or compile the kernel for given dimensions."""
    global _compiled_kernels

    key = (N, H, HV, K, V, pool_size, use_small_batch, is_varlen_decode)
    if key in _compiled_kernels:
        return _compiled_kernels[key]

    cu_seqlens = torch.zeros(N + 1, dtype=torch.int32, device="cuda")

    if is_varlen_decode:
        q = torch.zeros(1, N, H, K, dtype=torch.bfloat16, device="cuda")
        k = torch.zeros(1, N, H, K, dtype=torch.bfloat16, device="cuda")
        v = torch.zeros(1, N, HV, V, dtype=torch.bfloat16, device="cuda")
        a = torch.zeros(N, HV, dtype=torch.bfloat16, device="cuda")
        b = torch.zeros(N, HV, dtype=torch.bfloat16, device="cuda")
        o = torch.zeros(1, N, HV, V, dtype=torch.bfloat16, device="cuda")
    else:
        q = torch.zeros(N, 1, H, K, dtype=torch.bfloat16, device="cuda")
        k = torch.zeros(N, 1, H, K, dtype=torch.bfloat16, device="cuda")
        v = torch.zeros(N, 1, HV, V, dtype=torch.bfloat16, device="cuda")
        a = torch.zeros(N, 1, HV, dtype=torch.bfloat16, device="cuda")
        b = torch.zeros(N, 1, HV, dtype=torch.bfloat16, device="cuda")
        o = torch.zeros(N, 1, HV, V, dtype=torch.bfloat16, device="cuda")

    A_log = torch.zeros(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(HV, dtype=torch.bfloat16, device="cuda")
    h0_source = torch.zeros(pool_size, HV, K, V, dtype=torch.float32, device="cuda")
    h0_indices = torch.zeros(N, dtype=torch.int32, device="cuda")

    cu_seqlens_tensor = from_dlpack(cu_seqlens, assumed_align=16)
    q_tensor = from_dlpack(q, assumed_align=16)
    k_tensor = from_dlpack(k, assumed_align=16)
    v_tensor = from_dlpack(v, assumed_align=16)
    a_tensor = from_dlpack(a, assumed_align=16)
    b_tensor = from_dlpack(b, assumed_align=16)
    A_log_tensor = from_dlpack(A_log, assumed_align=16)
    dt_bias_tensor = from_dlpack(dt_bias, assumed_align=16)
    h0_source_tensor = from_dlpack(h0_source, assumed_align=16)
    h0_indices_tensor = from_dlpack(h0_indices, assumed_align=16)
    o_tensor = from_dlpack(o, assumed_align=16)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    run_small, run_small_varlen, run_large, run_large_varlen = _get_jit_functions()

    if use_small_batch:
        kernel_func = run_small_varlen if is_varlen_decode else run_small
    else:
        kernel_func = run_large_varlen if is_varlen_decode else run_large

    scale = K**-0.5
    softplus_beta = 1.0
    softplus_threshold = 20.0

    B_compile = 1 if is_varlen_decode else N
    T_compile = N if is_varlen_decode else 1

    compiled_kernel = cute.compile(
        kernel_func,
        cu_seqlens_tensor,
        q_tensor,
        k_tensor,
        v_tensor,
        a_tensor,
        b_tensor,
        A_log_tensor,
        dt_bias_tensor,
        h0_source_tensor,
        h0_indices_tensor,
        o_tensor,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        B=B_compile,
        T=T_compile,
        H=H,
        K=K,
        V=V,
        HV=HV,
        use_initial_state=True,
        use_qk_l2norm=True,
        stream=stream,
    )

    _compiled_kernels[key] = compiled_kernel
    logger.info(
        f"CuTe DSL GDN kernel compiled: N={N}, H={H}, HV={HV}, K={K}, V={V}, pool_size={pool_size}, small_batch={use_small_batch}, varlen={is_varlen_decode}"
    )

    return compiled_kernel


def cutedsl_fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
) -> torch.Tensor:
    """CuTe DSL implementation of fused sigmoid gating delta rule update."""

    B_q, T_q, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    N = initial_state_indices.shape[0]

    is_varlen_decode = B_q == 1 and T_q == N and N > 1
    if scale is None:
        scale = K**-0.5

    use_small_batch = N < SMALL_BATCH_THRESHOLD

    if initial_state_source.dim() == 1:
        pool_size = initial_state_source.numel() // (HV * K * V)
        h0_source = initial_state_source.view(pool_size, HV, K, V)
    elif initial_state_source.dim() == 4:
        pool_size = initial_state_source.shape[0]
        h0_source = initial_state_source
    else:
        raise ValueError(
            f"Unexpected initial_state_source shape: {initial_state_source.shape}"
        )

    if is_varlen_decode:
        if a.dim() == 3:
            a = a.squeeze(0)
        if b.dim() == 3:
            b = b.squeeze(0)
        o = q.new_empty(1, N, HV, V, dtype=torch.bfloat16)
    else:
        if a.dim() == 2:
            a = a.unsqueeze(1)
        if b.dim() == 2:
            b = b.unsqueeze(1)
        o = q.new_empty(N, 1, HV, V, dtype=torch.bfloat16)

    q, k, v = [t.contiguous() for t in (q, k, v)]

    global _cu_seqlens_cache
    if cu_seqlens is not None:
        cu_seqlens_to_use = cu_seqlens
    else:
        cache_key = (N, str(q.device))
        if cache_key not in _cu_seqlens_cache:
            _cu_seqlens_cache[cache_key] = torch.arange(
                N + 1, dtype=torch.int32, device=q.device
            )
        cu_seqlens_to_use = _cu_seqlens_cache[cache_key]

    cu_seqlens_tensor = from_dlpack(
        cu_seqlens_to_use.detach(), assumed_align=16
    ).mark_layout_dynamic(leading_dim=0)
    q_tensor = from_dlpack(q.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=q.ndim - 1
    )
    k_tensor = from_dlpack(k.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=k.ndim - 1
    )
    v_tensor = from_dlpack(v.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=v.ndim - 1
    )
    a_tensor = from_dlpack(a.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=a.ndim - 1
    )
    b_tensor = from_dlpack(b.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=b.ndim - 1
    )
    A_log_tensor = from_dlpack(A_log.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=0
    )
    dt_bias_tensor = from_dlpack(
        dt_bias.detach(), assumed_align=16
    ).mark_layout_dynamic(leading_dim=0)
    h0_source_tensor = from_dlpack(
        h0_source.detach(), assumed_align=16
    ).mark_layout_dynamic(leading_dim=h0_source.ndim - 1)
    h0_indices_tensor = from_dlpack(
        initial_state_indices.detach(), assumed_align=16
    ).mark_layout_dynamic(leading_dim=0)
    o_tensor = from_dlpack(o.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=o.ndim - 1
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled_kernel = _get_compiled_kernel(
        N, H, HV, K, V, pool_size, use_small_batch, is_varlen_decode
    )

    compiled_kernel(
        cu_seqlens_tensor,
        q_tensor,
        k_tensor,
        v_tensor,
        a_tensor,
        b_tensor,
        A_log_tensor,
        dt_bias_tensor,
        h0_source_tensor,
        h0_indices_tensor,
        o_tensor,
        stream,
    )

    return o
