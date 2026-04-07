"""CuTe DSL Fused Sigmoid Gating Delta Rule Kernel for KDA Decode.

This version uses production / Triton-compatible VK state layout:
    state.shape == (pool_size, HV, V, K)

The kernel still computes on a logical (K, V) matrix in shared memory. Global
state loads/stores therefore explicitly map:
    global(V, K) <-> shared(K, V)

Notes:
- This is a correctness-first implementation for decode.
- It keeps the original small-batch / large-batch split.
- It preserves the previous PAD semantics: if pool_idx < 0 the block does not
  load / update / write output or state, consistent with the earlier CuTe path.
"""

import logging
from typing import Dict, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
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
    """Define CuTe DSL kernels for KDA normal and varlen decode modes."""

    NUM_WARPS_SMALL = 4
    V_PER_WARP_SMALL = TILE_V_SMALL // NUM_WARPS_SMALL
    ROWS_PER_ITER_SMALL = 32 // V_PER_WARP_SMALL
    NUM_K_ITERS_SMALL = TILE_K // ROWS_PER_ITER_SMALL

    @cute.kernel
    def kda_kernel_small_batch(
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
        """Small batch KDA kernel for dense decode: q/k/v shapes (N, 1, ...)."""
        del tiled_copy_load
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
            smem_g_layout = cute.make_layout((TILE_K,), stride=(1,))
            sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
            sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)
            sG = smem.allocate_tensor(cutlass.Float32, smem_g_layout, 128)

            if tidx < TILE_K:
                sK[tidx] = cutlass.Float32(k[i_n, 0, i_h, tidx])
                sQ[tidx] = cutlass.Float32(q[i_n, 0, i_h, tidx])

            r_A_log = cutlass.Float32(A_log[i_hv])
            r_exp_A = cute.exp(r_A_log)
            if tidx < TILE_K:
                r_a_k = cutlass.Float32(a[i_n, 0, i_hv, tidx])
                r_dt_bias_k = cutlass.Float32(dt_bias[i_hv, tidx])
                x = r_a_k + r_dt_bias_k
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
                sG[tidx] = cute.exp(-r_exp_A * softplus_x)

            r_beta = 0.0
            if in_warp_tid == 0:
                r_b = cutlass.Float32(b[i_n, 0, i_hv])
                r_beta = 1.0 / (1.0 + cute.exp(-r_b))
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
                stage = v_tile_offset % NUM_STAGES
                v_tile = start_v_tile + v_tile_offset

                for k_iter in range(NUM_K_ITERS_SMALL):
                    flat_idx = tidx + k_iter * NUM_THREADS
                    k_load = flat_idx // TILE_V_SMALL
                    v_load = flat_idx % TILE_V_SMALL
                    if k_load < TILE_K:
                        v_global_load = v_tile * TILE_V_SMALL + v_load
                        h_val = 0.0
                        if v_global_load < v.shape[3]:
                            h_val = cutlass.Float32(
                                h0_source[(pool_idx, i_hv, v_global_load, k_load)]
                            )
                        sData[(k_load, v_load, stage)] = h_val

                cute.arch.barrier()

                v_global = v_tile * TILE_V_SMALL + v_idx
                r_v = 0.0
                if v_global < v.shape[3]:
                    r_v = cutlass.Float32(v[i_n, 0, i_hv, v_global])

                sum_hk = 0.0
                for k_iter in range(NUM_K_ITERS_SMALL):
                    k_base = k_iter * ROWS_PER_ITER_SMALL
                    k_idx = k_base + k_local
                    sum_hk += sData[(k_idx, v_idx, stage)] * sG[k_idx] * sK[k_idx]

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
                for k_iter in range(NUM_K_ITERS_SMALL):
                    k_base = k_iter * ROWS_PER_ITER_SMALL
                    k_idx = k_base + k_local
                    h_old = sData[(k_idx, v_idx, stage)] * sG[k_idx]
                    h_new = h_old + sK[k_idx] * v_new
                    sData[(k_idx, v_idx, stage)] = h_new
                    sum_hq += h_new * sQ[k_idx]

                for offset in [4, 2, 1]:
                    sum_hq += cute.arch.shuffle_sync_bfly(
                        sum_hq,
                        offset=offset * V_PER_WARP_SMALL,
                        mask=-1,
                        mask_and_clamp=31,
                    )

                if k_local == 0 and v_global < v.shape[3]:
                    o[(i_n, 0, i_hv, v_global)] = cutlass.BFloat16(sum_hq)

                cute.arch.barrier()

                for k_iter in range(NUM_K_ITERS_SMALL):
                    flat_idx = tidx + k_iter * NUM_THREADS
                    k_write = flat_idx // TILE_V_SMALL
                    v_write = flat_idx % TILE_V_SMALL
                    if k_write < TILE_K:
                        v_global_write = v_tile * TILE_V_SMALL + v_write
                        if v_global_write < v.shape[3]:
                            h0_source[(pool_idx, i_hv, v_global_write, k_write)] = (
                                sData[(k_write, v_write, stage)]
                            )

                cute.arch.barrier()

    @cute.kernel
    def kda_kernel_small_batch_varlen(
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
        """Small batch KDA kernel for varlen decode: q/k/v shapes (1, N, ...)."""
        del tiled_copy_load
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
            smem_g_layout = cute.make_layout((TILE_K,), stride=(1,))
            sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
            sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)
            sG = smem.allocate_tensor(cutlass.Float32, smem_g_layout, 128)

            if tidx < TILE_K:
                sK[tidx] = cutlass.Float32(k[0, i_n, i_h, tidx])
                sQ[tidx] = cutlass.Float32(q[0, i_n, i_h, tidx])

            r_A_log = cutlass.Float32(A_log[i_hv])
            r_exp_A = cute.exp(r_A_log)
            if tidx < TILE_K:
                r_a_k = cutlass.Float32(a[i_n, i_hv, tidx])
                r_dt_bias_k = cutlass.Float32(dt_bias[i_hv, tidx])
                x = r_a_k + r_dt_bias_k
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
                sG[tidx] = cute.exp(-r_exp_A * softplus_x)

            r_beta = 0.0
            if in_warp_tid == 0:
                r_b = cutlass.Float32(b[i_n, i_hv])
                r_beta = 1.0 / (1.0 + cute.exp(-r_b))
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
                stage = v_tile_offset % NUM_STAGES
                v_tile = start_v_tile + v_tile_offset

                for k_iter in range(NUM_K_ITERS_SMALL):
                    flat_idx = tidx + k_iter * NUM_THREADS
                    k_load = flat_idx // TILE_V_SMALL
                    v_load = flat_idx % TILE_V_SMALL
                    if k_load < TILE_K:
                        v_global_load = v_tile * TILE_V_SMALL + v_load
                        h_val = 0.0
                        if v_global_load < v.shape[3]:
                            h_val = cutlass.Float32(
                                h0_source[(pool_idx, i_hv, v_global_load, k_load)]
                            )
                        sData[(k_load, v_load, stage)] = h_val

                cute.arch.barrier()

                v_global = v_tile * TILE_V_SMALL + v_idx
                r_v = 0.0
                if v_global < v.shape[3]:
                    r_v = cutlass.Float32(v[0, i_n, i_hv, v_global])

                sum_hk = 0.0
                for k_iter in range(NUM_K_ITERS_SMALL):
                    k_base = k_iter * ROWS_PER_ITER_SMALL
                    k_idx = k_base + k_local
                    sum_hk += sData[(k_idx, v_idx, stage)] * sG[k_idx] * sK[k_idx]

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
                for k_iter in range(NUM_K_ITERS_SMALL):
                    k_base = k_iter * ROWS_PER_ITER_SMALL
                    k_idx = k_base + k_local
                    h_old = sData[(k_idx, v_idx, stage)] * sG[k_idx]
                    h_new = h_old + sK[k_idx] * v_new
                    sData[(k_idx, v_idx, stage)] = h_new
                    sum_hq += h_new * sQ[k_idx]

                for offset in [4, 2, 1]:
                    sum_hq += cute.arch.shuffle_sync_bfly(
                        sum_hq,
                        offset=offset * V_PER_WARP_SMALL,
                        mask=-1,
                        mask_and_clamp=31,
                    )

                if k_local == 0 and v_global < v.shape[3]:
                    o[(0, i_n, i_hv, v_global)] = cutlass.BFloat16(sum_hq)

                cute.arch.barrier()

                for k_iter in range(NUM_K_ITERS_SMALL):
                    flat_idx = tidx + k_iter * NUM_THREADS
                    k_write = flat_idx // TILE_V_SMALL
                    v_write = flat_idx % TILE_V_SMALL
                    if k_write < TILE_K:
                        v_global_write = v_tile * TILE_V_SMALL + v_write
                        if v_global_write < v.shape[3]:
                            h0_source[(pool_idx, i_hv, v_global_write, k_write)] = (
                                sData[(k_write, v_write, stage)]
                            )

                cute.arch.barrier()

    @cute.kernel
    def kda_kernel_large_batch(
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
        """Large batch KDA kernel for dense decode: q/k/v shapes (N, 1, ...)."""
        del tiled_copy_load
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
            smem_g_layout = cute.make_layout((TILE_K,), stride=(1,))
            sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
            sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)
            sG = smem.allocate_tensor(cutlass.Float32, smem_g_layout, 128)

            if tidx < TILE_K:
                sK[tidx] = cutlass.Float32(k[i_n, 0, i_h, tidx])
                sQ[tidx] = cutlass.Float32(q[i_n, 0, i_h, tidx])

            r_A_log = cutlass.Float32(A_log[i_hv])
            r_exp_A = cute.exp(r_A_log)
            if tidx < TILE_K:
                r_a_k = cutlass.Float32(a[i_n, 0, i_hv, tidx])
                r_dt_bias_k = cutlass.Float32(dt_bias[i_hv, tidx])
                x = r_a_k + r_dt_bias_k
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
                sG[tidx] = cute.exp(-r_exp_A * softplus_x)

            r_beta = 0.0
            if in_warp_tid == 0:
                r_b = cutlass.Float32(b[i_n, 0, i_hv])
                r_beta = 1.0 / (1.0 + cute.exp(-r_b))
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

                for k_iter in range(NUM_K_ITERS):
                    flat_idx = tidx + k_iter * NUM_THREADS_LARGE
                    k_load = flat_idx // TILE_V
                    v_load = flat_idx % TILE_V
                    if k_load < TILE_K:
                        v_global_load = v_tile * TILE_V + v_load
                        h_val = 0.0
                        if v_global_load < v.shape[3]:
                            h_val = cutlass.Float32(
                                h0_source[(pool_idx, i_hv, v_global_load, k_load)]
                            )
                        sData[(k_load, v_load, stage)] = h_val

                cute.arch.barrier()

                v_global = v_tile * TILE_V + v_idx
                r_v = 0.0
                if v_global < v.shape[3]:
                    r_v = cutlass.Float32(v[i_n, 0, i_hv, v_global])

                sum_hk = 0.0
                for k_iter in range(NUM_K_ITERS):
                    k_base = k_iter * ROWS_PER_ITER
                    k_idx = k_base + k_local
                    sum_hk += sData[(k_idx, v_idx, stage)] * sG[k_idx] * sK[k_idx]

                for offset in [4, 2, 1]:
                    sum_hk += cute.arch.shuffle_sync_bfly(
                        sum_hk,
                        offset=offset * V_PER_WARP,
                        mask=-1,
                        mask_and_clamp=31,
                    )

                v_new = (r_v - sum_hk) * r_beta
                v_new = cute.arch.shuffle_sync(v_new, v_local)

                sum_hq = 0.0
                for k_iter in range(NUM_K_ITERS):
                    k_base = k_iter * ROWS_PER_ITER
                    k_idx = k_base + k_local
                    h_old = sData[(k_idx, v_idx, stage)] * sG[k_idx]
                    h_new = h_old + sK[k_idx] * v_new
                    sData[(k_idx, v_idx, stage)] = h_new
                    sum_hq += h_new * sQ[k_idx]

                for offset in [4, 2, 1]:
                    sum_hq += cute.arch.shuffle_sync_bfly(
                        sum_hq,
                        offset=offset * V_PER_WARP,
                        mask=-1,
                        mask_and_clamp=31,
                    )

                if k_local == 0 and v_global < v.shape[3]:
                    o[(i_n, 0, i_hv, v_global)] = cutlass.BFloat16(sum_hq)

                cute.arch.barrier()

                for k_iter in range(NUM_K_ITERS):
                    flat_idx = tidx + k_iter * NUM_THREADS_LARGE
                    k_write = flat_idx // TILE_V
                    v_write = flat_idx % TILE_V
                    if k_write < TILE_K:
                        v_global_write = v_tile * TILE_V + v_write
                        if v_global_write < v.shape[3]:
                            h0_source[(pool_idx, i_hv, v_global_write, k_write)] = (
                                sData[(k_write, v_write, stage)]
                            )

                cute.arch.barrier()

    @cute.kernel
    def kda_kernel_large_batch_varlen(
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
        """Large batch KDA kernel for varlen decode: q/k/v shapes (1, N, ...)."""
        del tiled_copy_load
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
            smem_g_layout = cute.make_layout((TILE_K,), stride=(1,))
            sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
            sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)
            sG = smem.allocate_tensor(cutlass.Float32, smem_g_layout, 128)

            if tidx < TILE_K:
                sK[tidx] = cutlass.Float32(k[0, i_n, i_h, tidx])
                sQ[tidx] = cutlass.Float32(q[0, i_n, i_h, tidx])

            r_A_log = cutlass.Float32(A_log[i_hv])
            r_exp_A = cute.exp(r_A_log)
            if tidx < TILE_K:
                r_a_k = cutlass.Float32(a[i_n, i_hv, tidx])
                r_dt_bias_k = cutlass.Float32(dt_bias[i_hv, tidx])
                x = r_a_k + r_dt_bias_k
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
                sG[tidx] = cute.exp(-r_exp_A * softplus_x)

            r_beta = 0.0
            if in_warp_tid == 0:
                r_b = cutlass.Float32(b[i_n, i_hv])
                r_beta = 1.0 / (1.0 + cute.exp(-r_b))
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

                for k_iter in range(NUM_K_ITERS):
                    flat_idx = tidx + k_iter * NUM_THREADS_LARGE
                    k_load = flat_idx // TILE_V
                    v_load = flat_idx % TILE_V
                    if k_load < TILE_K:
                        v_global_load = v_tile * TILE_V + v_load
                        h_val = 0.0
                        if v_global_load < v.shape[3]:
                            h_val = cutlass.Float32(
                                h0_source[(pool_idx, i_hv, v_global_load, k_load)]
                            )
                        sData[(k_load, v_load, stage)] = h_val

                cute.arch.barrier()

                v_global = v_tile * TILE_V + v_idx
                r_v = 0.0
                if v_global < v.shape[3]:
                    r_v = cutlass.Float32(v[0, i_n, i_hv, v_global])

                sum_hk = 0.0
                for k_iter in range(NUM_K_ITERS):
                    k_base = k_iter * ROWS_PER_ITER
                    k_idx = k_base + k_local
                    sum_hk += sData[(k_idx, v_idx, stage)] * sG[k_idx] * sK[k_idx]

                for offset in [4, 2, 1]:
                    sum_hk += cute.arch.shuffle_sync_bfly(
                        sum_hk,
                        offset=offset * V_PER_WARP,
                        mask=-1,
                        mask_and_clamp=31,
                    )

                v_new = (r_v - sum_hk) * r_beta
                v_new = cute.arch.shuffle_sync(v_new, v_local)

                sum_hq = 0.0
                for k_iter in range(NUM_K_ITERS):
                    k_base = k_iter * ROWS_PER_ITER
                    k_idx = k_base + k_local
                    h_old = sData[(k_idx, v_idx, stage)] * sG[k_idx]
                    h_new = h_old + sK[k_idx] * v_new
                    sData[(k_idx, v_idx, stage)] = h_new
                    sum_hq += h_new * sQ[k_idx]

                for offset in [4, 2, 1]:
                    sum_hq += cute.arch.shuffle_sync_bfly(
                        sum_hq,
                        offset=offset * V_PER_WARP,
                        mask=-1,
                        mask_and_clamp=31,
                    )

                if k_local == 0 and v_global < v.shape[3]:
                    o[(0, i_n, i_hv, v_global)] = cutlass.BFloat16(sum_hq)

                cute.arch.barrier()

                for k_iter in range(NUM_K_ITERS):
                    flat_idx = tidx + k_iter * NUM_THREADS_LARGE
                    k_write = flat_idx // TILE_V
                    v_write = flat_idx % TILE_V
                    if k_write < TILE_K:
                        v_global_write = v_tile * TILE_V + v_write
                        if v_global_write < v.shape[3]:
                            h0_source[(pool_idx, i_hv, v_global_write, k_write)] = (
                                sData[(k_write, v_write, stage)]
                            )

                cute.arch.barrier()

    return (
        kda_kernel_small_batch,
        kda_kernel_small_batch_varlen,
        kda_kernel_large_batch,
        kda_kernel_large_batch_varlen,
    )


def _create_jit_functions():
    """Create JIT-compiled launcher functions for all KDA kernel variants."""

    kda_small, kda_small_varlen, kda_large, kda_large_varlen = _define_kernels()

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
        del cu_seqlens, B, T, K, use_initial_state
        _, hv_dim, v_dim, _ = h0_source.layout.shape
        n_indices = h0_indices.layout.shape[0]
        batch_size = n_indices * hv_dim

        num_v_tiles_small = cute.ceil_div(v_dim, TILE_V_SMALL)
        smem_layout_small = cute.make_layout(
            (TILE_K, TILE_V_SMALL, NUM_STAGES),
            stride=(TILE_V_SMALL_PADDED, 1, TILE_K * TILE_V_SMALL_PADDED),
        )
        smem_bytes_small = (
            4 * TILE_K * TILE_V_SMALL_PADDED * NUM_STAGES
            + 4 * TILE_V_SMALL
            + 4 * TILE_K * 2
            + 4 * TILE_K
            + 64
        )

        kda_small(
            None,
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
        del cu_seqlens, B, T, K, use_initial_state
        _, hv_dim, v_dim, _ = h0_source.layout.shape
        n_indices = h0_indices.layout.shape[0]
        batch_size = n_indices * hv_dim

        num_v_tiles_small = cute.ceil_div(v_dim, TILE_V_SMALL)
        smem_layout_small = cute.make_layout(
            (TILE_K, TILE_V_SMALL, NUM_STAGES),
            stride=(TILE_V_SMALL_PADDED, 1, TILE_K * TILE_V_SMALL_PADDED),
        )
        smem_bytes_small = (
            4 * TILE_K * TILE_V_SMALL_PADDED * NUM_STAGES
            + 4 * TILE_V_SMALL
            + 4 * TILE_K * 2
            + 4 * TILE_K
            + 64
        )

        kda_small_varlen(
            None,
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
        del cu_seqlens, B, T, K, use_initial_state
        _, hv_dim, v_dim, _ = h0_source.layout.shape
        n_indices = h0_indices.layout.shape[0]
        batch_size = n_indices * hv_dim

        num_v_tiles = cute.ceil_div(v_dim, TILE_V)
        smem_layout = cute.make_layout(
            (TILE_K, TILE_V, NUM_STAGES),
            stride=(TILE_V_PADDED, 1, TILE_K * TILE_V_PADDED),
        )
        smem_bytes = (
            4 * TILE_K * TILE_V_PADDED * NUM_STAGES
            + 4 * TILE_V
            + 4 * TILE_K * 2
            + 4 * TILE_K
            + 64
        )

        kda_large(
            None,
            h0_source,
            smem_layout,
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
        del cu_seqlens, B, T, K, use_initial_state
        _, hv_dim, v_dim, _ = h0_source.layout.shape
        n_indices = h0_indices.layout.shape[0]
        batch_size = n_indices * hv_dim

        num_v_tiles = cute.ceil_div(v_dim, TILE_V)
        smem_layout = cute.make_layout(
            (TILE_K, TILE_V, NUM_STAGES),
            stride=(TILE_V_PADDED, 1, TILE_K * TILE_V_PADDED),
        )
        smem_bytes = (
            4 * TILE_K * TILE_V_PADDED * NUM_STAGES
            + 4 * TILE_V
            + 4 * TILE_K * 2
            + 4 * TILE_K
            + 64
        )

        kda_large_varlen(
            None,
            h0_source,
            smem_layout,
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
    """Get or compile the KDA kernel for given dimensions."""
    global _compiled_kernels

    key = (N, H, HV, K, V, pool_size, use_small_batch, is_varlen_decode)
    if key in _compiled_kernels:
        return _compiled_kernels[key]

    cu_seqlens = torch.zeros(N + 1, dtype=torch.int32, device="cuda")

    if is_varlen_decode:
        q = torch.zeros(1, N, H, K, dtype=torch.bfloat16, device="cuda")
        k = torch.zeros(1, N, H, K, dtype=torch.bfloat16, device="cuda")
        v = torch.zeros(1, N, HV, V, dtype=torch.bfloat16, device="cuda")
        a = torch.zeros(N, HV, K, dtype=torch.bfloat16, device="cuda")
        b = torch.zeros(N, HV, dtype=torch.bfloat16, device="cuda")
        o = torch.zeros(1, N, HV, V, dtype=torch.bfloat16, device="cuda")
    else:
        q = torch.zeros(N, 1, H, K, dtype=torch.bfloat16, device="cuda")
        k = torch.zeros(N, 1, H, K, dtype=torch.bfloat16, device="cuda")
        v = torch.zeros(N, 1, HV, V, dtype=torch.bfloat16, device="cuda")
        a = torch.zeros(N, 1, HV, K, dtype=torch.bfloat16, device="cuda")
        b = torch.zeros(N, 1, HV, dtype=torch.bfloat16, device="cuda")
        o = torch.zeros(N, 1, HV, V, dtype=torch.bfloat16, device="cuda")

    A_log = torch.zeros(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(HV, K, dtype=torch.bfloat16, device="cuda")
    h0_source = torch.zeros(pool_size, HV, V, K, dtype=torch.float32, device="cuda")
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
        softplus_beta=1.0,
        softplus_threshold=20.0,
        scale=K**-0.5,
        B=1 if is_varlen_decode else N,
        T=N if is_varlen_decode else 1,
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
        "CuTe DSL KDA kernel compiled: "
        f"N={N}, H={H}, HV={HV}, K={K}, V={V}, pool_size={pool_size}, "
        f"small_batch={use_small_batch}, varlen={is_varlen_decode}"
    )
    return compiled_kernel


def _normalize_A_log(A_log: torch.Tensor, HV: int) -> torch.Tensor:
    if A_log.numel() != HV:
        raise ValueError(f"Unexpected A_log shape: {A_log.shape}; expected numel={HV}")
    return A_log.reshape(HV).contiguous()


def _normalize_dt_bias(dt_bias: torch.Tensor, HV: int, K: int) -> torch.Tensor:
    if dt_bias.numel() != HV * K:
        raise ValueError(
            f"Unexpected dt_bias shape: {dt_bias.shape}; expected numel={HV * K}"
        )
    return dt_bias.reshape(HV, K).contiguous()


def _normalize_kda_a(a, *, is_varlen_decode, N, HV, K):
    """Normalize `a` to match the compile-time shape expected by the kernel.

    varlen kernel compiled shape: (N, HV, K)  -- 3D
    dense kernel compiled shape:  (N, 1, HV, K) -- 4D
    """
    if is_varlen_decode:
        # Target: (N, HV, K) -- 3D
        if a.dim() == 2 and a.shape == (N, HV * K):
            return a.view(N, HV, K)
        if a.dim() == 3 and a.shape == (N, HV, K):
            return a  # already correct
        if a.dim() == 4 and a.shape == (1, N, HV, K):
            return a.squeeze(0)  # remove leading dim
        raise ValueError(f"Unexpected a shape for varlen: {a.shape}")
    else:
        # Target: (N, 1, HV, K) -- 4D
        if a.dim() == 2 and a.shape == (N, HV * K):
            return a.view(N, 1, HV, K)
        if a.dim() == 3 and a.shape == (N, HV, K):
            return a.unsqueeze(1)
        if a.dim() == 4 and a.shape == (N, 1, HV, K):
            return a
        raise ValueError(f"Unexpected a shape for dense: {a.shape}")


def cutedsl_fused_sigmoid_gating_kda_update(
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
    """CuTe DSL implementation of fused sigmoid gating KDA update.

    State layout contract:
        initial_state_source.shape == (pool_size, HV, V, K)

    Dense decode:
        q/k: (N, 1, H, K)
        v:   (N, 1, HV, V)
        a:   (N, 1, HV, K)
        b:   (N, 1, HV)

    Varlen decode:
        q/k: (1, N, H, K)
        v:   (1, N, HV, V)
        a:   (N, HV, K) or (1, N, HV, K)
        b:   (N, HV) or (1, N, HV)
    """

    A_log = A_log.contiguous()

    B_q, T_q, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    N = initial_state_indices.shape[0]

    assert K == TILE_K, f"Current CuTe DSL KDA kernel requires K={TILE_K}, got {K}"
    assert (
        V % TILE_V_SMALL == 0
    ), f"Current CuTe DSL KDA kernel requires V % {TILE_V_SMALL} == 0, got V={V}"
    assert (
        V % TILE_V == 0
    ), f"Current CuTe DSL KDA kernel requires V % {TILE_V} == 0, got V={V}"
    assert (V // TILE_V_SMALL) % NUM_BLOCKS_PER_STATE_SMALL == 0, (
        "Small-batch KDA kernel requires num_v_tiles_small divisible by "
        f"{NUM_BLOCKS_PER_STATE_SMALL}, got V={V}"
    )

    is_varlen_decode = B_q == 1 and T_q == N and N > 1
    if scale is None:
        scale = K**-0.5
    else:
        assert scale > 0, f"scale must be positive, got {scale}"

    use_small_batch = N < SMALL_BATCH_THRESHOLD

    if initial_state_source.dim() == 1:
        pool_size = initial_state_source.numel() // (HV * V * K)
        h0_source = initial_state_source.view(pool_size, HV, V, K)
    elif initial_state_source.dim() == 4:
        pool_size = initial_state_source.shape[0]
        h0_source = initial_state_source
    else:
        raise ValueError(
            f"Unexpected initial_state_source shape: {initial_state_source.shape}"
        )

    a = _normalize_kda_a(a, is_varlen_decode=is_varlen_decode, N=N, HV=HV, K=K)

    if is_varlen_decode:
        # varlen b compiled: (N, HV) -- 2D
        if b.dim() == 3:
            b = b.squeeze(0)  # (1, N, HV) -> (N, HV)
        # b should be 2D (N, HV)
        o = q.new_empty(1, N, HV, V, dtype=torch.bfloat16)
    else:
        # dense b compiled: (N, 1, HV) -- 3D
        if b.dim() == 2:
            b = b.unsqueeze(1)
        # b should be 3D (N, 1, HV)
        o = q.new_empty(N, 1, HV, V, dtype=torch.bfloat16)

    q, k, v, a, b = [t.contiguous() for t in (q, k, v, a, b)]
    dt_bias = dt_bias.contiguous()

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

    A_log = _normalize_A_log(A_log, HV)
    dt_bias = _normalize_dt_bias(dt_bias, HV, K)

    h0_source = h0_source.contiguous()

    initial_state_indices = initial_state_indices.contiguous()
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.contiguous()

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
    ).mark_layout_dynamic(leading_dim=dt_bias.ndim - 1)
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
