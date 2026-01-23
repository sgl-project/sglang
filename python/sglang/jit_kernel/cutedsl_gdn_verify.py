# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
CuTe DSL GDN Verify Kernel (K-last optimized)

This file implements a GDN verify kernel (MTP mode) that extends the decode kernel to:
1. Process multiple tokens (T > 1) in sequence
2. Cache intermediate states at each time step for verification/rollback
3. Keep gate computation fused inside kernel
4. Support disable_state_update option (for speculative decoding verify)

Key optimization: 
- Uses K-last layout [HV, V, K] for efficient memory access in MTP kernel
- Loop order is (v_tiles outer, time_steps inner)
- Each v_tile's h stays in shared memory across all time steps
- Only write to global memory for intermediate state cache (when needed)
- Avoids expensive global memory round-trip between time steps

For topk=1 case only (retrieve_parent_token is None).
"""

import logging
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Lazy import globals
_cutlass_available = None
_cute = None
_cutlass = None
_cpasync = None
_from_dlpack = None
_cuda = None
_compiled_kernels: Dict[Tuple, object] = {}
_cu_seqlens_cache: Dict[Tuple, torch.Tensor] = {}

# Global configuration
TILE_V = 8
TILE_K = 128
NUM_STAGES = 2
NUM_THREADS = 128  # 4 warps


def _check_cutlass_available():
    """Check if cutlass/cute is available and import if needed."""
    global _cutlass_available, _cute, _cutlass, _cpasync, _from_dlpack, _cuda

    if _cutlass_available is not None:
        return _cutlass_available

    try:
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.nvgpu import cpasync
        from cutlass.cute.runtime import from_dlpack

        _cutlass = cutlass
        _cute = cute
        _cpasync = cpasync
        _from_dlpack = from_dlpack
        _cuda = cuda
        _cutlass_available = True
        logger.info("CuTe DSL GDN Verify kernel: cutlass/cute available")
    except ImportError as e:
        _cutlass_available = False
        logger.warning(f"CuTe DSL GDN Verify kernel: cutlass/cute not available: {e}")

    return _cutlass_available


def is_cutedsl_gdn_verify_available() -> bool:
    """Check if CuTe DSL GDN Verify kernel is available."""
    return _check_cutlass_available()


def _define_kernels():
    """Define CuTe DSL kernels for verify mode."""
    cute = _cute
    cutlass = _cutlass
    cpasync = _cpasync

    @cute.kernel
    def sglang_gdn_verify_kernel(
        tiled_copy_load: cute.TiledCopy,     # TiledCopy for G2S load (passed from JIT launcher)
        h0_source: cute.Tensor,              # [pool_size * HV, V, K] - initial state pool (K-last)
        intermediate_states: cute.Tensor,    # [pool_size * T * HV, V, K] - intermediate state cache
        smem_layout_staged: cute.ComposedLayout,  # Swizzled layout to avoid bank conflicts
        vec_size: cutlass.Constexpr[int],
        num_v_tiles: cutlass.Constexpr[int],
        A_log: cute.Tensor,      # [HV]
        a: cute.Tensor,          # [B, T, HV]
        dt_bias: cute.Tensor,    # [HV]
        g_log_in: cute.Tensor,   # [B, T, HV] (optional precomputed g_log)
        g_decay_in: cute.Tensor, # [B, T, HV] (optional precomputed exp(g_log))
        q: cute.Tensor,          # [B, T, H, K]
        k: cute.Tensor,          # [B, T, H, K]
        v: cute.Tensor,          # [B, T, HV, V]
        b: cute.Tensor,          # [B, T, HV]
        beta_in: cute.Tensor,    # [B, T, HV] (optional precomputed beta, fp32 but quantized)
        o: cute.Tensor,          # [B, T, HV, V] - output
        h0_indices: cute.Tensor, # [B] - initial state indices
        cu_seqlens: cute.Tensor, # [B+1] - cumulative sequence lengths (for varlen)
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        HV: cutlass.Constexpr[int],
        B: cutlass.Constexpr[int],
        T: cutlass.Constexpr[int],
        H: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        use_initial_state: cutlass.Constexpr[bool],
        use_qk_l2norm: cutlass.Constexpr[bool],
        is_varlen: cutlass.Constexpr[bool],
        act_is_bf16: cutlass.Constexpr[bool],
        use_precomputed_g_beta: cutlass.Constexpr[bool],
        use_precomputed_g_decay: cutlass.Constexpr[bool],
        disable_state_update: cutlass.Constexpr[bool],
        cache_intermediate_states: cutlass.Constexpr[bool],
    ):
        """
        Verify kernel with optimized loop order: (v_tiles outer, time_steps inner)
        
        Uses cute.make_rmem_tensor for register arrays (same style as decode kernel).
        Store-compute overlap: cache data in registers, store during next iteration.
        
        Fixed precision settings (per ablation study):
        - Beta is always quantized to activation dtype (critical for accuracy)
        - Output is always bf16 (critical for accuracy)
        - L2norm uses division form (no measurable difference, kept for simplicity)
        """

        tidx, _, _ = cute.arch.thread_idx()
        lane_id = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        batch_idx, _, _ = cute.arch.block_idx()
        i_n = batch_idx // HV
        i_hv = batch_idx % HV
        i_h = i_hv // (HV // H)

        # Load A_log and dt_bias once (they don't vary with time)
        r_A_log = cutlass.Float32(A_log[i_hv])
        r_dt_bias = cutlass.Float32(dt_bias[i_hv])

        smem = cutlass.utils.SmemAllocator()

        # ===================================================================
        # Allocate shared memory with padding to avoid bank conflicts
        # Padding: add 8 elements per row so different rows hit different banks
        # ===================================================================
        sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
        sOutput = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, V)), 16)
        # Pre-computed shared memory for all time steps
        sQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, K), stride=(K, 1)), 16)
        sK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, K), stride=(K, 1)), 16)
        sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, V), stride=(V, 1)), 16)
        sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T,)), 16)
        sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T,)), 16)

        # Allocate register tensors (same style as decode kernel)
        r_q = cute.make_rmem_tensor(
            cute.make_layout((vec_size,), stride=(1,)),
            cutlass.Float32
        )
        r_k = cute.make_rmem_tensor(
            cute.make_layout((vec_size,), stride=(1,)),
            cutlass.Float32
        )
        r_h = cute.make_rmem_tensor(
            cute.make_layout((vec_size,), stride=(1,)),
            cutlass.Float32
        )
        
        # Initialize output accumulator to zero
        for i_t in range(T):
            sOutput[(i_t, tidx)] = 0.0
        
        cute.arch.barrier()

        # Get initial state index for this batch
        cache_idx = h0_indices[i_n]

        # ===================================================================
        # Early exit optimization: skip pre-computation for padding slots
        # This saves significant overhead for CUDA Graph padding entries
        # ===================================================================
        if cache_idx >= 0:
            # ===================================================================
            # Pre-compute q, k, v, g, beta for all time steps (outside v_tiles loop)
            # ===================================================================
            for i_t in range(T):
                # Load q, k into register arrays
                for i in range(vec_size):
                    r_q[i] = cutlass.Float32(q[i_n, i_t, i_h, i * 32 + lane_id])
                    r_k[i] = cutlass.Float32(k[i_n, i_t, i_h, i * 32 + lane_id])
                    # Load v for all V elements
                    sV[(i_t, i * 32 + lane_id)] = cutlass.Float32(v[i_n, i_t, i_hv, i * 32 + lane_id])

                # Compute g and beta
                r_g = 0.0
                r_beta = 0.0
                if use_precomputed_g_beta:
                    if lane_id == 0:
                        r_g_value = cutlass.Float32(g_log_in[i_n, i_t, i_hv])
                        r_beta = cutlass.Float32(beta_in[i_n, i_t, i_hv])
                        if use_precomputed_g_decay:
                            r_g = cutlass.Float32(g_decay_in[i_n, i_t, i_hv])
                        else:
                            r_g = cute.exp(r_g_value)  # decay = exp(g_log)
                    r_g = cute.arch.shuffle_sync(r_g, 0)
                    r_beta = cute.arch.shuffle_sync(r_beta, 0)
                else:
                    r_a = cutlass.Float32(a[i_n, i_t, i_hv])
                    r_b = cutlass.Float32(b[i_n, i_t, i_hv])
                    if lane_id == 0:
                        x = r_a + r_dt_bias
                        beta_x = softplus_beta * x
                        softplus_x = 0.0

                        if beta_x <= softplus_threshold:
                            exp_beta_x = cute.exp(beta_x)
                            log_input = cutlass.Float32(1.0 + exp_beta_x)
                            log_result = cutlass.Float32(cute.log(log_input))
                            softplus_x = cutlass.Float32((cutlass.Float32(1.0) / softplus_beta) * log_result)
                        else:
                            softplus_x = x

                        r_g_value = -cute.exp(r_A_log) * softplus_x
                        r_beta = 1.0 / (1.0 + cute.exp(-r_b))
                        # Match fused_gdn_gating behavior: beta is quantized to activation dtype
                        # (fp16/bf16) before being consumed by the recurrent update.
                        # This is critical for accuracy (ablation: 0.26% argmax flip without it).
                        if act_is_bf16:
                            r_beta = cutlass.Float32(cutlass.BFloat16(r_beta))
                        else:
                            r_beta = cutlass.Float32(cutlass.Float16(r_beta))
                        r_g = cute.exp(r_g_value)

                    r_g = cute.arch.shuffle_sync(r_g, 0)
                    r_beta = cute.arch.shuffle_sync(r_beta, 0)

                # Apply L2 normalization
                if use_qk_l2norm:
                    sum_q = 0.0
                    sum_k = 0.0
                    for i in range(vec_size):
                        sum_q += r_q[i] * r_q[i]
                        sum_k += r_k[i] * r_k[i]

                    for offset in [16, 8, 4, 2, 1]:
                        sum_q += cute.arch.shuffle_sync_bfly(
                            sum_q, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        sum_k += cute.arch.shuffle_sync_bfly(
                            sum_k, offset=offset, mask=-1, mask_and_clamp=31
                        )

                    # Match Triton (tl.sqrt + division) for exactness.
                    norm_q = cute.sqrt(sum_q + 1e-6)
                    norm_k = cute.sqrt(sum_k + 1e-6)
                    for i in range(vec_size):
                        r_q[i] = r_q[i] / norm_q
                        r_k[i] = r_k[i] / norm_k

                # Apply scaling to q
                for i in range(vec_size):
                    r_q[i] = r_q[i] * scale

                # Store pre-computed values to shared memory
                for i in range(vec_size):
                    sQ[(i_t, i * 32 + lane_id)] = r_q[i]
                    sK[(i_t, i * 32 + lane_id)] = r_k[i]
                
                # Store g and beta (only one thread needs to do this)
                if tidx == 0:
                    sG[i_t] = r_g
                    sBeta[i_t] = r_beta

        # All threads must participate in barrier (CUDA requirement)
        cute.arch.barrier()

        # Main computation only for valid batch entries
        if cache_idx >= 0:
            # Setup source tensor for initial state loading
            gSrc_batch = h0_source[(cache_idx * HV + i_hv, None, None)]
            gDst_h0 = cute.local_tile(h0_source, (1, TILE_V, TILE_K), (cache_idx * HV + i_hv, None, 0))
            gSrc = cute.local_tile(gSrc_batch, (TILE_V, TILE_K), (None, 0))
            
            thr_copy_load = tiled_copy_load.get_slice(tidx)

            # ===================================================================
            # Main loop: v_tiles (outer) x time_steps (inner)
            # With store-compute overlap optimization
            # ===================================================================
            prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles)
            
            # Prefetch first v_tile(s)
            for v_tiles in range(prefetch_count):
                stage = v_tiles % NUM_STAGES
                gSrc_tile = gSrc[(None, None, v_tiles)]
                sData_stage = sData[(None, None, stage)]
                thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
                thr_sData = thr_copy_load.partition_D(sData_stage)
                cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                cute.arch.cp_async_commit_group()

            # Process each v_tile
            for v_tiles in range(num_v_tiles):
                stage = v_tiles % NUM_STAGES
                
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
                
                # Prefetch next v_tile
                next_v_tiles = v_tiles + prefetch_count
                if next_v_tiles < num_v_tiles:
                    next_stage = next_v_tiles % NUM_STAGES
                    gSrc_next = gSrc[(None, None, next_v_tiles)]
                    sData_next = sData[(None, None, next_stage)]
                    thr_gSrc = thr_copy_load.partition_S(gSrc_next)
                    thr_sData = thr_copy_load.partition_D(sData_next)
                    cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                    cute.arch.cp_async_commit_group()

                # Inner loop: all time steps for this v_tile
                for i_t in range(T):
                    # Load pre-computed values from shared memory
                    # todo: put q,k compute into main loop instread of pre-compute. and also to reduce shared memory usage.
                    for i in range(vec_size):
                        r_q[i] = sQ[(i_t, i * 32 + lane_id)]
                        r_k[i] = sK[(i_t, i * 32 + lane_id)]
                    
                    r_g = sG[i_t]
                    r_beta = sBeta[i_t]

                    # Compute delta rule for this v_tile
                    for row in range(0, TILE_V, 4):
                        row_offset = tidx // 32
                        
                        # Load h from sData, apply decay
                        for i in range(vec_size):
                            r_h[i] = sData[(row + row_offset, i * 32 + lane_id, stage)] * r_g
                        
                        # Compute sum_hk = h @ k
                        sum_hk = 0.0
                        for i in range(vec_size):
                            sum_hk += r_h[i] * r_k[i]

                        for offset in [16, 8, 4, 2, 1]:
                            sum_hk += cute.arch.shuffle_sync_bfly(
                                sum_hk, offset=offset, mask=-1, mask_and_clamp=31
                            )

                        # Delta rule update
                        v_idx = v_tiles * TILE_V + row + row_offset
                        v_new = sV[(i_t, v_idx)] - sum_hk
                        v_new = v_new * r_beta

                        # Update h and write back to sData
                        for i in range(vec_size):
                            r_h[i] += r_k[i] * v_new
                            sData[(row + row_offset, i * 32 + lane_id, stage)] = r_h[i]
                        
                        # Store intermediate state INSIDE row loop (overlap store with sum_hq compute)
                        # Note: Use i_n (batch index 0 to B-1) for intermediate states, not cache_idx
                        # because intermediate_state_cache is indexed sequentially per request
                        if cache_intermediate_states:
                            flat_idx = i_n * T * HV + i_t * HV + i_hv
                            if v_idx < V:
                                for i in range(vec_size):
                                    intermediate_states[(flat_idx, v_idx, i * 32 + lane_id)] = r_h[i]
                        
                        # Compute sum_hq = h @ q (overlaps with store above)
                        sum_hq = 0.0
                        for i in range(vec_size):
                            sum_hq += r_h[i] * r_q[i]

                        for offset in [16, 8, 4, 2, 1]:
                            sum_hq += cute.arch.shuffle_sync_bfly(
                                sum_hq, offset=offset, mask=-1, mask_and_clamp=31
                            )

                        o_idx = v_tiles * TILE_V + row + row_offset
                        if lane_id == 0 and o_idx < V:
                            sOutput[(i_t, o_idx)] = cutlass.Float32(sum_hq)

                # Write final h for this v_tile to h0_source
                if not disable_state_update:
                    for row in range(0, TILE_V, 4):
                        row_offset = tidx // 32
                        for i in range(vec_size):
                            gDst_h0[(0, row + row_offset, i * 32 + lane_id, v_tiles)] = sData[(row + row_offset, i * 32 + lane_id, stage)]


            # Final writeback (always bf16 - critical for accuracy, ablation: 0.30% argmax flip with fp16)
            cute.arch.barrier()
            
            for i_t in range(T):
                o[(i_n, i_t, i_hv, tidx)] = cutlass.BFloat16(sOutput[(i_t, tidx)])
        # Note: padding slots (cache_idx < 0) skip all computation and output writeback
        # Their output values are unused since they're CUDA Graph padding

    return sglang_gdn_verify_kernel


def _create_jit_function():
    """Create JIT-compiled launcher function for verify kernel."""
    cute = _cute
    cutlass = _cutlass
    cpasync = _cpasync
    cuda = _cuda

    sglang_gdn_verify_kernel = _define_kernels()

    @cute.jit
    def run_verify_kernel(
        h0_source: cute.Tensor,
        intermediate_states: cute.Tensor,
        A_log: cute.Tensor,
        a: cute.Tensor,
        dt_bias: cute.Tensor,
        g_log_in: cute.Tensor,
        g_decay_in: cute.Tensor,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        b: cute.Tensor,
        beta_in: cute.Tensor,
        o: cute.Tensor,
        h0_indices: cute.Tensor,
        cu_seqlens: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        HV: cutlass.Constexpr[int],
        B: cutlass.Constexpr[int],
        T: cutlass.Constexpr[int],
        H: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        use_initial_state: cutlass.Constexpr[bool],
        use_qk_l2norm: cutlass.Constexpr[bool],
        is_varlen: cutlass.Constexpr[bool],
        act_is_bf16: cutlass.Constexpr[bool],
        use_precomputed_g_beta: cutlass.Constexpr[bool],
        use_precomputed_g_decay: cutlass.Constexpr[bool],
        disable_state_update: cutlass.Constexpr[bool],
        cache_intermediate_states: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        _, v_dim, k_dim = h0_source.layout.shape[0], h0_source.layout.shape[1], h0_source.layout.shape[2]

        num_v_tiles = cute.ceil_div(v_dim, TILE_V)
        batch_size = B * HV
        vec_size = TILE_K // 32  # Each thread in a warp processes this many elements

        # =========================================================================
        # Composed Swizzle Layout to avoid shared memory bank conflicts
        # Base layout: (TILE_V, TILE_K, NUM_STAGES) = (8, 128, 2), stride=(128, 1, 1024)
        #   addr = v * 128 + k
        #   Bank = k % 32 (row doesn't affect bank!)
        #
        # Problem: Different rows with same column hit same bank
        # Solution: Swizzle<2, 3, 7> for (V=8, K=128) layout
        #   - B=2: swizzle 2 bits (4 patterns)
        #   - M=3: start at bit 3 (skip 8 bytes = 2 floats)
        #   - S=7: extract from bit 7+ (row index in V*K stride)
        #   addr' = addr XOR ((addr >> 7) & 3) << 3
        # =========================================================================
        base_smem_layout = cute.make_layout(
            (TILE_V, TILE_K, NUM_STAGES),
            stride=(TILE_K, 1, TILE_V * TILE_K)
        )
        swizzle = cute.make_swizzle(2, 3, 7)  # Swizzle<2, 3, 7>
        smem_layout_staged = cute.make_composed_layout(swizzle, 0, base_smem_layout)

        # =========================================================================
        # Create tiled copy for G2S load (in JIT launcher, passed to kernel)
        # This fixes the ptr alignment issue when using lazy imports
        # =========================================================================
        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32,
            num_bits_per_copy=128
        )
        thread_layout = cute.make_layout((4, 32), stride=(32, 1))
        val_layout = cute.make_layout((1, 4))
        tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)

        # smem: sData + sOutput + sQ + sK + sV + sG + sBeta + padding
        smem_bytes = (
            4 * TILE_V * TILE_K * NUM_STAGES
            + 4 * T * v_dim
            + 4 * T * k_dim
            + 4 * T * k_dim
            + 4 * T * v_dim
            + 4 * T
            + 4 * T
            + 128
        )

        sglang_gdn_verify_kernel(
            tiled_copy_load,
            h0_source, intermediate_states, smem_layout_staged,
            vec_size, num_v_tiles,
            A_log, a, dt_bias, g_log_in, g_decay_in, q, k, v, b, beta_in, o,
            h0_indices, cu_seqlens,
            softplus_beta, softplus_threshold, scale,
            HV, B, T, H, K, V,
            use_initial_state,
            use_qk_l2norm,
            is_varlen,
            act_is_bf16,
            use_precomputed_g_beta,
            use_precomputed_g_decay,
            disable_state_update,
            cache_intermediate_states,
        ).launch(
            grid=(batch_size, 1, 1),
            block=[NUM_THREADS, 1, 1],
            smem=smem_bytes,
            stream=stream
        )

    return run_verify_kernel


_jit_function = None


def _get_jit_function():
    global _jit_function
    if _jit_function is None:
        _jit_function = _create_jit_function()
    return _jit_function


def _get_compiled_kernel(
    B,
    T,
    H,
    HV,
    K,
    V,
    pool_size,
    cache_steps,
    disable_state_update,
    cache_intermediate_states,
    use_qk_l2norm,
    act_dtype,
    use_precomputed_g_beta,
    use_precomputed_g_decay,
):
    """Get or compile the kernel for given dimensions.
    
    Fixed precision settings (per ablation study):
    - Beta is always quantized to activation dtype
    - Output is always bf16
    - L2norm uses division form
    """
    if not _check_cutlass_available():
        raise RuntimeError("CuTe DSL GDN Verify kernel requires cutlass/cute")

    global _compiled_kernels

    key = (
        B,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        cache_steps,
        disable_state_update,
        cache_intermediate_states,
        use_qk_l2norm,
        str(act_dtype),
        use_precomputed_g_beta,
        use_precomputed_g_decay,
    )
    if key in _compiled_kernels:
        return _compiled_kernels[key]

    cute = _cute
    cuda = _cuda
    from_dlpack = _from_dlpack

    # Create dummy tensors for compilation with actual sizes
    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device="cuda")
    q = torch.zeros(B, T, H, K, dtype=act_dtype, device="cuda")
    k = torch.zeros(B, T, H, K, dtype=act_dtype, device="cuda")
    v = torch.zeros(B, T, HV, V, dtype=act_dtype, device="cuda")
    a = torch.zeros(B, T, HV, dtype=act_dtype, device="cuda")
    b = torch.zeros(B, T, HV, dtype=act_dtype, device="cuda")
    A_log = torch.zeros(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(HV, dtype=torch.float32, device="cuda")
    g_log_in = torch.zeros(B, T, HV, dtype=torch.float32, device="cuda")
    g_decay_in = torch.zeros(B, T, HV, dtype=torch.float32, device="cuda")
    beta_in = torch.zeros(B, T, HV, dtype=torch.float32, device="cuda")
    # h0_source is already K-last: [pool_size * HV, V, K]
    h0_source = torch.zeros(pool_size * HV, V, K, dtype=torch.float32, device="cuda")
    h0_indices = torch.zeros(B, dtype=torch.int32, device="cuda")
    # Output is always bf16 (critical for accuracy per ablation study)
    o = torch.zeros(B, T, HV, V, dtype=torch.bfloat16, device="cuda")

    if cache_intermediate_states:
        # intermediate_states is K-last: [pool_size * cache_steps * HV, V, K]
        intermediate_states = torch.zeros(
            pool_size * cache_steps * HV, V, K,
            dtype=torch.float32, device="cuda"
        )
    else:
        intermediate_states = torch.zeros(1, 1, 1, dtype=torch.float32, device="cuda")

    cu_seqlens_tensor = from_dlpack(cu_seqlens, assumed_align=16)
    q_tensor = from_dlpack(q, assumed_align=16)
    k_tensor = from_dlpack(k, assumed_align=16)
    v_tensor = from_dlpack(v, assumed_align=16)
    a_tensor = from_dlpack(a, assumed_align=16)
    b_tensor = from_dlpack(b, assumed_align=16)
    A_log_tensor = from_dlpack(A_log, assumed_align=16)
    dt_bias_tensor = from_dlpack(dt_bias, assumed_align=16)
    g_log_in_tensor = from_dlpack(g_log_in, assumed_align=16)
    g_decay_in_tensor = from_dlpack(g_decay_in, assumed_align=16)
    h0_source_tensor = from_dlpack(h0_source, assumed_align=16)
    h0_indices_tensor = from_dlpack(h0_indices, assumed_align=16)
    o_tensor = from_dlpack(o, assumed_align=16)
    intermediate_states_tensor = from_dlpack(intermediate_states, assumed_align=16)
    beta_in_tensor = from_dlpack(beta_in, assumed_align=16)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    run_verify_kernel = _get_jit_function()

    scale = K ** -0.5
    softplus_beta = 1.0
    softplus_threshold = 20.0
    act_is_bf16 = act_dtype == torch.bfloat16

    compiled_kernel = cute.compile(
        run_verify_kernel,
        h0_source_tensor,
        intermediate_states_tensor,
        A_log_tensor, a_tensor, dt_bias_tensor,
        g_log_in_tensor,
        g_decay_in_tensor,
        q_tensor, k_tensor, v_tensor, b_tensor, beta_in_tensor, o_tensor,
        h0_indices_tensor, cu_seqlens_tensor,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        HV=HV, B=B, T=T, H=H, K=K, V=V,
        use_initial_state=True,
        use_qk_l2norm=use_qk_l2norm,
        is_varlen=False,
        act_is_bf16=act_is_bf16,
        use_precomputed_g_beta=use_precomputed_g_beta,
        use_precomputed_g_decay=use_precomputed_g_decay,
        disable_state_update=disable_state_update,
        cache_intermediate_states=cache_intermediate_states,
        stream=stream,
    )

    _compiled_kernels[key] = compiled_kernel
    logger.info(
        f"CuTe DSL GDN Verify kernel compiled: B={B}, T={T}, H={H}, HV={HV}, K={K}, V={V}, "
        f"pool_size={pool_size}, cache_steps={cache_steps}, use_qk_l2norm={use_qk_l2norm}, act_dtype={act_dtype}"
    )

    return compiled_kernel


def cutedsl_gdn_verify_k_last(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,  # K-last: [pool_size, HV, V, K]
    initial_state_indices: torch.Tensor,
    intermediate_states_buffer: torch.Tensor,  # K-last: [layers, pool_size, cache_steps, HV, V, K]
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    disable_state_update: bool = True,
    cache_steps: Optional[int] = None,
    g_log_in: Optional[torch.Tensor] = None,
    beta_in: Optional[torch.Tensor] = None,
    g_decay_in: Optional[torch.Tensor] = None,
):
    """
    CuTe DSL implementation of GDN verify kernel (K-last optimized).
    
    This version accepts K-last layout directly (no internal transpose):
    - initial_state_source: [pool_size, HV, V, K]
    - intermediate_states_buffer: [pool_size, cache_steps, HV, V, K]
    """
    if not _check_cutlass_available():
        raise RuntimeError("CuTe DSL GDN Verify kernel requires cutlass/cute to be installed")

    from_dlpack = _from_dlpack
    cuda = _cuda

    B, T, H, K_dim = q.shape
    V_dim = v.shape[-1]
    HV = v.shape[2]
    pool_size = initial_state_source.shape[0]
    
    if scale is None:
        scale = K_dim ** -0.5
    
    if cache_steps is None:
        cache_steps = T
    
    # initial_state_source is already K-last: [pool_size, HV, V, K]
    # Reshape to [pool_size * HV, V, K] for kernel
    # IMPORTANT: Convert to float32 to match kernel compilation (kernel uses Float32 internally)
    h0_source = initial_state_source.to(torch.float32).reshape(pool_size * HV, V_dim, K_dim)
    
    # intermediate_states_buffer is already K-last: [buffer_size, cache_steps, HV, V, K]
    # Note: buffer_size may differ from pool_size (e.g., spec_state_size + 1)
    # Reshape to [buffer_size * cache_steps * HV, V, K] for kernel
    # Kernel uses i_n (0 to B-1) to index, so buffer_size >= B is sufficient
    cache_intermediate_states = intermediate_states_buffer is not None
    if cache_intermediate_states:
        buffer_size = intermediate_states_buffer.shape[0]
        intermediate_states = intermediate_states_buffer.to(torch.float32).reshape(
            buffer_size * cache_steps * HV, V_dim, K_dim
        )
    else:
        intermediate_states = torch.zeros(1, 1, 1, dtype=torch.float32, device=initial_state_source.device)
    
    # Output is always bf16 (critical for accuracy per ablation study - 0.30% argmax flip with fp16)
    o = torch.empty(B, T, HV, V_dim, dtype=torch.bfloat16, device=v.device)

    # cu_seqlens caching
    global _cu_seqlens_cache
    if cu_seqlens is not None:
        cu_seqlens_to_use = cu_seqlens
    else:
        cache_key = (B, str(q.device))
        if cache_key not in _cu_seqlens_cache:
            _cu_seqlens_cache[cache_key] = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
        cu_seqlens_to_use = _cu_seqlens_cache[cache_key]
    
    # Preserve activation dtype (fp16 or bf16) to avoid unnecessary precision loss.
    # Note: kernel compilation cache key now includes act_dtype.
    act_dtype = v.dtype if v.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    q_act = q.to(act_dtype).detach().contiguous()
    k_act = k.to(act_dtype).detach().contiguous()
    v_act = v.to(act_dtype).detach().contiguous()
    a_act = a.to(act_dtype).detach().contiguous()
    b_act = b.to(act_dtype).detach().contiguous()

    use_precomputed_g_beta = (g_log_in is not None) and (beta_in is not None)
    if use_precomputed_g_beta:
        g_log_act = g_log_in.detach().contiguous().to(torch.float32)
        beta_act = beta_in.detach().contiguous().to(torch.float32)
        if g_decay_in is None:
            # Use torch.exp to precompute decay (ablation showed no measurable difference vs kernel exp)
            g_decay_act = torch.exp(g_log_act)
            use_precomputed_g_decay = True
        else:
            g_decay_act = g_decay_in.detach().contiguous().to(torch.float32)
            use_precomputed_g_decay = True
    else:
        # Dummy tensors to satisfy kernel signature (won't be used).
        g_log_act = torch.zeros(B, T, HV, dtype=torch.float32, device=v.device)
        g_decay_act = torch.zeros(B, T, HV, dtype=torch.float32, device=v.device)
        beta_act = torch.zeros(B, T, HV, dtype=torch.float32, device=v.device)
        use_precomputed_g_decay = False
    
    # Convert to CuTe tensors with dynamic layout marking
    h0_source_tensor = from_dlpack(h0_source.detach(), assumed_align=16)
    intermediate_states_tensor = from_dlpack(intermediate_states.detach(), assumed_align=16)
    A_log_tensor = from_dlpack(A_log.detach().contiguous(), assumed_align=16)
    a_tensor = from_dlpack(a_act, assumed_align=16)
    dt_bias_tensor = from_dlpack(dt_bias.detach().contiguous(), assumed_align=16)
    g_log_in_tensor = from_dlpack(g_log_act, assumed_align=16)
    g_decay_in_tensor = from_dlpack(g_decay_act, assumed_align=16)
    q_tensor = from_dlpack(q_act, assumed_align=16)
    k_tensor = from_dlpack(k_act, assumed_align=16)
    v_tensor = from_dlpack(v_act, assumed_align=16)
    b_tensor = from_dlpack(b_act, assumed_align=16)
    beta_in_tensor = from_dlpack(beta_act, assumed_align=16)
    o_tensor = from_dlpack(o.detach(), assumed_align=16)
    h0_indices_tensor = from_dlpack(initial_state_indices.detach().contiguous(), assumed_align=16)
    cu_seqlens_tensor = from_dlpack(cu_seqlens_to_use.detach(), assumed_align=16)
    
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled_kernel = _get_compiled_kernel(
        B, T, H, HV, K_dim, V_dim, pool_size, cache_steps,
        disable_state_update,
        cache_intermediate_states,
        use_qk_l2norm_in_kernel,
        act_dtype,
        use_precomputed_g_beta,
        use_precomputed_g_decay,
    )

    compiled_kernel(
        h0_source_tensor,
        intermediate_states_tensor,
        A_log_tensor, a_tensor, dt_bias_tensor,
        g_log_in_tensor,
        g_decay_in_tensor,
        q_tensor, k_tensor, v_tensor, b_tensor,
        beta_in_tensor,
        o_tensor,
        h0_indices_tensor, cu_seqlens_tensor,
        stream
    )
    
    return o
