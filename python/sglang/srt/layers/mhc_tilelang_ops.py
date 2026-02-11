"""
Fused CUDA kernels for multi-head composite (mHC) operators.

This module provides high-performance fused kernel implementations
for mHC operations using TileLang, including normalization, linear projection,
sigmoid activation, aggregation, Sinkhorn-Knopp normalization, and residual merging.
"""
import logging
import torch
import tilelang
import tilelang.language as T
from typing import Tuple
import math
try:
    import deep_gemm
except ImportError:
    deep_gemm = None

tilelang.set_log_level("ERROR")

for logger_name in list(logging.Logger.manager.loggerDict.keys()) + ["tilelang", "TileLang"]:
    if 'tilelang' in logger_name.lower():
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}


@tilelang.jit(pass_configs=pass_configs)
def norm_linear_splitk_kernel(
    M_pad, K, 
    M, N,  # Real dimensions for boundary checks
    threads, split_k, block_M, block_N, block_K
):
    """
    TileLang kernel for fused RMS normalization and linear projection with Split-K.
    Uses Atomic Add for accumulation.
    """
    in_dtype = "bfloat16"
    weight_dtype = "float32"
    accum_dtype = "float32"
    
    # Define symbolic variables for dynamic shape
    # Use variable shadowing for cleaner syntax
    M_pad = T.dynamic("M_pad")
    M = T.dynamic("M")

    @T.prim_func
    def norm_linear_splitk_func(
        X: T.Tensor[(M_pad, K), in_dtype], 
        W: T.Tensor[(N, K), weight_dtype],
        Y_accum: T.Tensor[(M, N), accum_dtype],      # Global Accumulator (FP32) - Real Shape
        SumSqX_accum: T.Tensor[(M, ), accum_dtype],  # Global Accumulator - Real Shape
    ):
        # Grid: [M_pad/block_M, N/block_N, split_k]
        # Note: N/block_N handles ceiling automatically in ceildiv
        with T.Kernel(
            T.ceildiv(M_pad, block_M),
            T.ceildiv(N, block_N),
            split_k,
            threads=threads
        ) as (pid_m, pid_n, pid_k):
            X_shared = T.alloc_shared((block_M, block_K), in_dtype)
            # Load FP32 Weight into BF16 Shared Memory for GEMM
            W_shared = T.alloc_shared((block_N, block_K), in_dtype)
            
            # Local accumulators
            Y_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            SumSqX_local = T.alloc_shared((block_M, ), accum_dtype)
            
            T.clear(Y_local)
            T.clear(SumSqX_local)
            
            T.sync_threads()

            total_k_blocks = T.ceildiv(K, block_K)
            k_blocks_per_split = T.ceildiv(total_k_blocks, split_k)
            k_start = pid_k * k_blocks_per_split
            k_end = T.min((pid_k + 1) * k_blocks_per_split, total_k_blocks)
            
            # Loop over K chunks
            # Use num_stages=2 for Split-K kernel as loop count is small
            for k_idx in T.Pipelined(k_end - k_start, num_stages=2):
                k = k_start + k_idx
                T.copy(X[pid_m * block_M, k * block_K], X_shared)
                
                # Load W (NormW is assumed fused into W by caller or ignored if identity)
                # Since user requested "proj input use original x", we just load W directly.
                for i, j in T.Parallel(block_N, block_K):
                    n_idx = pid_n * block_N + i
                    k_val_idx = k * block_K + j
                    
                    # Boundary check for W loading (Handle N not divisible by block_N)
                    if n_idx < N:
                        # Load W (FP32) and cast to BF16
                        W_shared[i, j] = T.cast(W[n_idx, k_val_idx], in_dtype)
                    else:
                        W_shared[i, j] = T.cast(0.0, in_dtype)

                X_frag = T.alloc_fragment((block_M, block_K), in_dtype)
                
                T.copy(X_shared, X_frag)
                
                # Compute Stats (SumSqX only)
                vec_size = 8
                for i in T.Parallel(block_M):
                    for j in T.serial(block_K // vec_size):
                        for v in T.vectorized(vec_size):
                            idx = j * vec_size + v
                            val = T.cast(X_frag[i, idx], accum_dtype)
                            SumSqX_local[i] += val * val
                
                # Compute GEMM using original X and W
                T.gemm(X_shared, W_shared, Y_local, transpose_B=True)
            
            T.sync_threads()
            # Atomic Accumulate Y
            for i, j in T.Parallel(block_M, block_N):
                idx_m = pid_m * block_M + i
                idx_n = pid_n * block_N + j
                if idx_m < M and idx_n < N:
                    T.atomic_add(Y_accum[idx_m, idx_n], Y_local[i, j])
            
            # Atomic Accumulate Stats
            if pid_n == 0:
                for i in T.Parallel(block_M):
                    idx_m = pid_m * block_M + i
                    if idx_m < M:
                        T.atomic_add(SumSqX_accum[idx_m], SumSqX_local[i])
    return norm_linear_splitk_func


@tilelang.jit(pass_configs=pass_configs)
def reduce_finalize_kernel(M, num_splits, threads, block_M):
    """
    Fused kernel to reduce sum_sq_x_split and compute RMS.
    Replaces: sum(dim=0) -> sqrt -> div -> copy
    """
    fp32 = "float32"
    M = T.dynamic("M")
    
    @T.prim_func
    def reduce_finalize_func(
        SumSqX_Split: T.Tensor[(num_splits, M), fp32],
        R_Out: T.Tensor[(M, 1), fp32],
        K_val: T.float32,
        eps: T.float32
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as pid_m:
            for i in T.Parallel(block_M):
                m_idx = pid_m * block_M + i
                if m_idx < M:
                    # Reduce across splits
                    # Use alloc_fragment for mutable scalar accumulation
                    sum_sq = T.alloc_fragment((1,), fp32)
                    sum_sq[0] = 0.0
                    for s in T.serial(num_splits):
                        sum_sq[0] += SumSqX_Split[s, m_idx]
                    
                    # Compute RMS
                    # R = sqrt(sum_sq / K + eps)
                    r_val = T.sqrt(sum_sq[0] / K_val + eps)
                    
                    # Store result
                    R_Out[m_idx, 0] = r_val
    return reduce_finalize_func


@tilelang.jit(pass_configs=pass_configs)
def norm_linear_finalize_kernel(M, N, threads, block_M, block_N):
    """
    Finalize kernel to compute R and cast Y.
    """
    out_dtype = "float32"
    accum_dtype = "float32"
    r_dtype = "float32"

    M = T.dynamic("M")
    
    @T.prim_func
    def norm_linear_finalize_func(
        Y_accum: T.Tensor[(M, N), accum_dtype],
        SumSqX_accum: T.Tensor[(M, ), accum_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        R: T.Tensor[(M, ), r_dtype],
        K_val: T.float32,
        eps: T.float32
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (pid_m, pid_n):
            # Finalize Y: Just cast Accum to Output (No scaling)
            # Scaling will be handled by map_sigmoid later using 1/R
            for i, j in T.Parallel(block_M, block_N):
                m_idx = pid_m * block_M + i
                n_idx = pid_n * block_N + j
                if m_idx < M and n_idx < N:
                    Y[m_idx, n_idx] = T.cast(Y_accum[m_idx, n_idx], out_dtype)
            
            # Finalize R: Compute RMS(X) (Only if pid_n == 0)
            if pid_n == 0:
                for i in T.Parallel(block_M):
                    m_idx = pid_m * block_M + i
                    if m_idx < M:
                        sum_sq_x = SumSqX_accum[m_idx]
                        # R = RMS(X) = sqrt(mean(X^2) + eps)
                        R[m_idx] = T.sqrt(sum_sq_x / K_val + eps)
    return norm_linear_finalize_func


def mhc_tilelang_norm_linear(
    x: torch.Tensor,
    weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused RMS normalization and linear projection operation using Split-K strategy.
    Args:
        x: [M, n_streams, D], bf16
        weight: [2 * n_streams + n_streams * n_streams, n_streams * D], fp32
    Returns:
        r: [M, 1], fp32
        proj: [M, 2 * n_streams + n_streams * n_streams], fp32
    """
    x_in = x.flatten(1)
    M, K = x_in.shape
    N, K_w = weight.shape

    out_y = torch.empty((M, N), dtype=torch.float32, device=x.device)
    out_r = torch.empty((M, 1), dtype=torch.float32, device=x.device)

    out_y_in = out_y
    out_r_in = out_r
    assert K == K_w, f"Shape mismatch"
    
    if not x_in.is_contiguous():
        x_in = x_in.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    
    # Based on tuning results:
    # Small Batch (M=128): Split-K Path, Threads=128, BM=64, BN=32, SplitK=128
    # Large Batch (M=5120): Split-K Path, Threads=128, BM=128, BN=32, SplitK=16
    
    if M <= 128:
        threads = 256
        BLOCK_M = 64
        BLOCK_N = 32
        TARGET_SPLIT_K = 64
    else:
        threads = 128
        BLOCK_M = 128
        BLOCK_N = 32
        TARGET_SPLIT_K = 32

    BLOCK_K = 128
    
    # --- Split-K Path ---
    
    # Padding (Recalculate for Split Config)
    # Optimization: Weight padding is handled inside kernel via masking
    weight_in = weight
        
    pad_m = 0
    if M % BLOCK_M != 0:
        pad_m = BLOCK_M - (M % BLOCK_M)
        x_padded = torch.nn.functional.pad(x_in, (0, 0, 0, pad_m))
    else:
        x_padded = x_in
        
    M_pad = x_padded.shape[0]
    
    # Split-K Logic
    # Directly use tuned TARGET_SPLIT_K, capped by physical K blocks
    max_split_by_k = max(1, K // BLOCK_K)
    split_k = min(TARGET_SPLIT_K, max_split_by_k)
    
    # Ensure weight is FP32 as per requirement
    eps = 1e-6
    
    # Alloc Accumulators
    # Optimization: Directly use out_y as accumulator (must be zeroed)
    # Since we added boundary checks in kernel, we can pass non-padded out_y directly
    
    y_accum = out_y_in

    y_accum.zero_()
    
    r_out_real = out_r_in.view(M)

    sum_sq_x_accum = torch.zeros((M, ), dtype=torch.float32, device=x.device)
    
    kernel_split = norm_linear_splitk_kernel(
        M_pad, K, 
        M, N, 
        threads, split_k, BLOCK_M, BLOCK_N, BLOCK_K
    )
    # Input is now BF16, supported by kernel
    kernel_split(x_padded, weight_in, y_accum, sum_sq_x_accum)
    
    # Finalize kernel computes R and optionally casts Y
    # Since y_accum IS out_y (FP32), we technically don't need to cast Y, 
    # but we run finalize to compute R and ensure consistency.
    # We pass Real M, N to finalize kernel
    kernel_final = norm_linear_finalize_kernel(M, N, threads, BLOCK_M, BLOCK_N)
    kernel_final(y_accum, sum_sq_x_accum, y_accum, r_out_real, float(K), eps)

    return out_r, out_y


def mhc_deepgemm_norm_linear(
    x: torch.Tensor,
    weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DeepGEMM accelerated RMS normalization and linear projection operation.
    Args:
        x: [M, n_streams, D], bf16
        weight: [2 * n_streams + n_streams * n_streams, n_streams * D], fp32
    Returns:
        r: [M, 1], fp32
        proj: [M, 2 * n_streams + n_streams * n_streams], fp32
    """
    if deep_gemm is None or not hasattr(deep_gemm, "tf32_hc_prenorm_gemm"):
        raise RuntimeError("deep_gemm.tf32_hc_prenorm_gemm not found. Please install or update DeepGEMM library.")

    x_in = x.flatten(1)
    M, K = x_in.shape
    N, K_w = weight.shape

    out_y = torch.empty((M, N), dtype=torch.float32, device=x.device)
    out_r = torch.empty((M, 1), dtype=torch.float32, device=x.device)

    out_y_in = out_y
    out_r_in = out_r
    assert K == K_w, f"Shape mismatch"
    
    if not x_in.is_contiguous():
        x_in = x_in.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    # When num_splits is set (e.g. 16), outputs must be splitted tensors:
    # sum_sq_x: (num_splits, M)
    # proj_out: (num_splits, M, N)
    num_splits = 16
    sum_sq_x_split = torch.empty((num_splits, M), device=x.device, dtype=torch.float32)
    proj_out_split = torch.empty((num_splits, M, N), device=x.device, dtype=torch.float32)
    
    # DeepGEMM optimization: Fused GEMM + Pre-Norm Stats
    deep_gemm.tf32_hc_prenorm_gemm(x_in, weight, proj_out_split, sum_sq_x_split, num_splits)
    
    # Reduce Split-K results
    # Optimization: Use out= parameter to avoid intermediate buffer allocation and copy
    # for y_accum if output tensor is contiguous.
    torch.sum(proj_out_split, dim=0, out=out_y_in)

    # Compute r from sum_sq_x
    # Optimization: Fused kernel to reduce overhead
    # r = sqrt(sum(sum_sq_x_split) / K)
    
    # Kernel config
    # Optimization: Use smaller blocks for small batch to avoid thread idleness
    if M <= 64:
        threads = 64
        block_M = 64
    else:
        threads = 128
        block_M = 128
    eps = 1e-6
    
    finalize_kernel = reduce_finalize_kernel(M, num_splits, threads, block_M)
    finalize_kernel(sum_sq_x_split, out_r_in, float(K), eps)

    return out_r, out_y


@tilelang.jit(pass_configs=pass_configs)
def map_sigmoid_kernel(M, Dim, N, threads, block_M, block_Dim):
    """
    TileLang kernel for fused map and sigmoid operations.
    
    Args:
        M: Batch dimension.
        Dim: Feature dimension.
        N: Number of streams.
        threads: Number of threads per block.
        block_M: Block size for M dimension.
        block_Dim: Block size for Dim dimension.
        
    Returns:
        Compiled kernel function.
    """
    fp32 = "float32"
    
    M = T.dynamic("M")

    @T.prim_func
    def map_sigmoid_func(
        R: T.Tensor[(M, 1), fp32],
        Proj: T.Tensor[(M, Dim), fp32],
        Alpha: T.Tensor[(Dim, ), fp32],
        Bias: T.Tensor[(Dim, ), fp32],
        H_pre: T.Tensor[(M, N), fp32],
        H_post: T.Tensor[(M, N), fp32],
        H_res: T.Tensor[(M, N, N), fp32]
    ):
        T.Assert(block_M * block_Dim <= threads, "block_M * block_Dim must be <= threads")
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(Dim, block_Dim), threads=threads) as (pid_m, pid_d):
            for tid in T.Parallel(threads):
                if tid < block_M * block_Dim:
                    m_idx = pid_m * block_M + (tid // block_Dim)
                    d_idx = pid_d * block_Dim + (tid % block_Dim)
                    if m_idx < M and d_idx < Dim:
                        r_val = R[m_idx, 0]
                        proj_val = Proj[m_idx, d_idx]
                        alpha_val = Alpha[d_idx]
                        bias_val = Bias[d_idx]
                        h_val = (1.0 / (r_val + 1e-8)) * proj_val * alpha_val + bias_val
                        if d_idx < N:
                            H_pre[m_idx, d_idx] = T.cast(T.sigmoid(h_val), fp32)
                        elif d_idx < 2 * N:
                            H_post[m_idx, d_idx - N] = T.cast(T.sigmoid(h_val) * 2.0, fp32)
                        else:
                            res_idx = d_idx - 2 * N
                            row = T.floordiv(res_idx, N)
                            col = T.floormod(res_idx, N)
                            H_res[m_idx, row, col] = T.cast(h_val, fp32)
    return map_sigmoid_func


def mhc_tilelang_map_sigmoid(r, proj, bias, alpha_pre, alpha_post, alpha_res, n_streams):
    """
    Fused map and sigmoid operations.
    Args:
        r: [M, 1], fp32
        proj: [M, 2 * n_streams + n_streams * n_streams, fp32
        bias: [2 * n_streams + n_streams * n_streams, ], fp32
        alpha_pre: [1, ], fp32
        alpha_post: [1, ], fp32
        alpha_res: [1, ], fp32
        n_streams: int
    Returns:
        h_pre: [M, n_streams], fp32
        h_post: [M, n_streams], fp32
        h_res: [M, n_streams, n_streams], fp32
    """
    # Expand and concatenate alpha components
    alpha = torch.cat([
        alpha_pre.expand(n_streams),
        alpha_post.expand(n_streams),
        alpha_res.expand(n_streams * n_streams)
    ], dim=-1).contiguous()
    
    OutDim = proj.shape[-1]
    M, _ = r.shape
    r_in = r
    proj_in = proj

    if not r_in.is_contiguous():
        r_in = r_in.contiguous()
    if not proj_in.is_contiguous():
        proj_in = proj_in.contiguous()

    n = n_streams
    
    h_pre = torch.empty((M, n), dtype=torch.float32, device=r.device)
    h_post = torch.empty((M, n), dtype=torch.float32, device=r.device)
    h_res = torch.empty((M, n, n), dtype=torch.float32, device=r.device)

    # Tuned for M=128 and M=5120 on H800
    if M <= 128:
        threads = 128
        block_M = 1
        block_Dim = 32
    else:
        threads = 256
        block_M = 2
        block_Dim = 32
    
    kernel = map_sigmoid_kernel(M, OutDim, n, threads, block_M, block_Dim)
    kernel(r_in, proj_in, alpha, bias, h_pre, h_post, h_res)

    return h_pre, h_post, h_res

@tilelang.jit(pass_configs=pass_configs)
def aggregate_kernel(M, n_streams, Dim, threads, block_M, block_D):
    """
    TileLang kernel for fused aggregation operation.
    
    Args:
        M: Batch dimension, batch * seq.
        n_streams: Number of streams.
        Dim: Feature dimension.
        threads: Number of threads per block.
        block_M: Block size for M dimension.
        block_D: Block size for D dimension.
        
    Returns:
        Compiled kernel function.
    """
    
    bf16 = "bfloat16"
    fp32 = "float32"
    vec_size = 8  # Vectorize 8 elements (128 bits for BF16)
    
    block_D_threads = block_D // vec_size 
    
    M = T.dynamic("M")

    @T.prim_func
    def aggregate_func(
        Res: T.Tensor[(M, n_streams, Dim), bf16],
        H_pre: T.Tensor[(M, n_streams), fp32],
        Out: T.Tensor[(M, Dim), bf16]
    ):
        with T.Kernel(
            T.ceildiv(M, block_M), 
            T.ceildiv(Dim, block_D),
            threads=threads
        ) as (pid_m, pid_d):
            
            # Shared memory cache for H_pre
            H_shared = T.alloc_shared([block_M, n_streams], fp32)
            
            # Cooperative load of H_pre
            for tid in T.Parallel(threads):
                if tid < block_M * n_streams:
                    row = tid // n_streams
                    col = tid % n_streams
                    global_row = pid_m * block_M + row
                    if global_row < M:
                        H_shared[row, col] = H_pre[global_row, col]
            
            T.sync_threads()
            
            # Compute - vectorized
            # Use T.thread_binding instead of T.Parallel to support internal vectorized instructions
            for tid in T.thread_binding(threads, thread="threadIdx.x"):
                m_offset = tid // block_D_threads
                d_vec_idx = tid % block_D_threads
                d_base = d_vec_idx * vec_size
                
                m_idx = pid_m * block_M + m_offset
                d_idx_base = pid_d * block_D + d_base
                
                # Only execute when the entire vector is within bounds
                if m_idx < M and d_idx_base < Dim:
                    
                    # Load H into registers
                    h0 = H_shared[m_offset, 0]
                    h1 = H_shared[m_offset, 1]
                    h2 = H_shared[m_offset, 2]
                    h3 = H_shared[m_offset, 3]

                    # Vectorized read/write and compute
                    for v in T.vectorized(vec_size):
                        d_idx = d_idx_base + v
                        
                        val = (
                            T.cast(Res[m_idx, 0, d_idx], fp32) * h0 +
                            T.cast(Res[m_idx, 1, d_idx], fp32) * h1 +
                            T.cast(Res[m_idx, 2, d_idx], fp32) * h2 +
                            T.cast(Res[m_idx, 3, d_idx], fp32) * h3
                        )
                        Out[m_idx, d_idx] = T.cast(val, bf16)
    
    return aggregate_func

def mhc_tilelang_aggregate(residuals, h_pre):
    """
    Fused aggregation operation.
    Args:
        residuals: [M, n_streams, D], bf16
        h_pre: [M, n_streams], fp32
    Returns:
        out: [M, D], bf16
    """
    M, N, D = residuals.shape
    res_in = residuals
    h_pre_in = h_pre

    # Based on kernel, Output is BF16
    out_residuals = torch.empty((M, D), dtype=torch.bfloat16, device=residuals.device)
    out = out_residuals

    if not res_in.is_contiguous():
        res_in = res_in.contiguous()
    if not h_pre_in.is_contiguous():
        h_pre_in = h_pre_in.contiguous()

    if N != 4:
        raise ValueError(f"aggregate only supports n_streams=4, got {N}")
    
    # Tuned for M=128 and M=5120 on H800
    if M <= 128:
        threads = 128
        block_M = 8
        block_D_threads = 16
    else:
        threads = 256
        block_M = 1
        block_D_threads = 256
    
    vec_size = 8
    block_D = block_D_threads * vec_size
    
    kernel = aggregate_kernel(M, N, D, threads, block_M, block_D)
    kernel(res_in, h_pre_in, out)

    return out_residuals


@tilelang.jit(pass_configs=pass_configs)
def sinkhorn_kernel(M, n_streams, threads, block_M):
    """
    TileLang kernel for fused Sinkhorn-Knopp normalization.
    
    Args:
        M: Batch dimension.
        n_streams: Number of streams (must be 4).
        threads: Number of threads per block.
        block_M: Block size for M dimension.
        
    Returns:
        Compiled kernel function.
    """
    # Use Shared Memory to store state
    fp32 = "float32"
    accum_dtype = "float32"
    N = n_streams
    iters = 20

    M = T.dynamic("M")

    @T.prim_func
    def sinkhorn_func(Logits: T.Tensor[(M, N, N), fp32], Out: T.Tensor[(M, N, N), fp32]):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as pid_m:
            # Shared memory for the whole block: [block_M, N, N]
            mat_shared = T.alloc_shared((block_M, N, N), accum_dtype)
            # Auxiliary shared memory for sums
            row_sum_shared = T.alloc_shared((block_M, N), accum_dtype)
            col_sum_shared = T.alloc_shared((block_M, N), accum_dtype)
            
            # Load phase
            for i in T.Parallel(block_M):
                m_idx = pid_m * block_M + i
                if m_idx < M:
                    # Load raw logits first to find max for numerical stability
                    # row 0
                    l00 = Logits[m_idx, 0, 0]
                    l01 = Logits[m_idx, 0, 1]
                    l02 = Logits[m_idx, 0, 2]
                    l03 = Logits[m_idx, 0, 3]
                    # row 1
                    l10 = Logits[m_idx, 1, 0]
                    l11 = Logits[m_idx, 1, 1]
                    l12 = Logits[m_idx, 1, 2]
                    l13 = Logits[m_idx, 1, 3]
                    # row 2
                    l20 = Logits[m_idx, 2, 0]
                    l21 = Logits[m_idx, 2, 1]
                    l22 = Logits[m_idx, 2, 2]
                    l23 = Logits[m_idx, 2, 3]
                    # row 3
                    l30 = Logits[m_idx, 3, 0]
                    l31 = Logits[m_idx, 3, 1]
                    l32 = Logits[m_idx, 3, 2]
                    l33 = Logits[m_idx, 3, 3]

                    # Find max value per row (Row Max)
                    rm0 = T.max(T.max(l00, l01), T.max(l02, l03))
                    rm1 = T.max(T.max(l10, l11), T.max(l12, l13))
                    rm2 = T.max(T.max(l20, l21), T.max(l22, l23))
                    rm3 = T.max(T.max(l30, l31), T.max(l32, l33))
                    
                    # Compute exp(logits - row_max) to prevent overflow and ensure row stability
                    # row 0
                    mat_shared[i, 0, 0] = T.exp(l00 - rm0)
                    mat_shared[i, 0, 1] = T.exp(l01 - rm0)
                    mat_shared[i, 0, 2] = T.exp(l02 - rm0)
                    mat_shared[i, 0, 3] = T.exp(l03 - rm0)
                    # row 1
                    mat_shared[i, 1, 0] = T.exp(l10 - rm1)
                    mat_shared[i, 1, 1] = T.exp(l11 - rm1)
                    mat_shared[i, 1, 2] = T.exp(l12 - rm1)
                    mat_shared[i, 1, 3] = T.exp(l13 - rm1)
                    # row 2
                    mat_shared[i, 2, 0] = T.exp(l20 - rm2)
                    mat_shared[i, 2, 1] = T.exp(l21 - rm2)
                    mat_shared[i, 2, 2] = T.exp(l22 - rm2)
                    mat_shared[i, 2, 3] = T.exp(l23 - rm2)
                    # row 3
                    mat_shared[i, 3, 0] = T.exp(l30 - rm3)
                    mat_shared[i, 3, 1] = T.exp(l31 - rm3)
                    mat_shared[i, 3, 2] = T.exp(l32 - rm3)
                    mat_shared[i, 3, 3] = T.exp(l33 - rm3)
            
            # Ite, 3]rations loop (serial) outside Parallel
            for iter_idx in T.serial(iters):
                # Standard Order: Row Norm first, then Col Norm
                # This ensures the final operation is Col Norm, guaranteeing col_sum == 1.0

                # Row Norm Phase
                for i in T.Parallel(block_M):
                    m_idx = pid_m * block_M + i
                    if m_idx < M:
                        # Row 0
                        row_sum_shared[i, 0] = T.max(
                            mat_shared[i, 0, 0] + mat_shared[i, 0, 1] +
                            mat_shared[i, 0, 2] + mat_shared[i, 0, 3], 1e-8)
                        mat_shared[i, 0, 0] /= row_sum_shared[i, 0]
                        mat_shared[i, 0, 1] /= row_sum_shared[i, 0]
                        mat_shared[i, 0, 2] /= row_sum_shared[i, 0]
                        mat_shared[i, 0, 3] /= row_sum_shared[i, 0]
                        
                        # Row 1
                        row_sum_shared[i, 1] = T.max(
                            mat_shared[i, 1, 0] + mat_shared[i, 1, 1] +
                            mat_shared[i, 1, 2] + mat_shared[i, 1, 3], 1e-8)
                        mat_shared[i, 1, 0] /= row_sum_shared[i, 1]
                        mat_shared[i, 1, 1] /= row_sum_shared[i, 1]
                        mat_shared[i, 1, 2] /= row_sum_shared[i, 1]
                        mat_shared[i, 1, 3] /= row_sum_shared[i, 1]

                        # Row 2
                        row_sum_shared[i, 2] = T.max(
                            mat_shared[i, 2, 0] + mat_shared[i, 2, 1] +
                            mat_shared[i, 2, 2] + mat_shared[i, 2, 3], 1e-8)
                        mat_shared[i, 2, 0] /= row_sum_shared[i, 2]
                        mat_shared[i, 2, 1] /= row_sum_shared[i, 2]
                        mat_shared[i, 2, 2] /= row_sum_shared[i, 2]
                        mat_shared[i, 2, 3] /= row_sum_shared[i, 2]
                        
                        # Row 3
                        row_sum_shared[i, 3] = T.max(
                            mat_shared[i, 3, 0] + mat_shared[i, 3, 1] +
                            mat_shared[i, 3, 2] + mat_shared[i, 3, 3], 1e-8)
                        mat_shared[i, 3, 0] /= row_sum_shared[i, 3]
                        mat_shared[i, 3, 1] /= row_sum_shared[i, 3]
                        mat_shared[i, 3, 2] /= row_sum_shared[i, 3]
                        mat_shared[i, 3, 3] /= row_sum_shared[i, 3]

                # Col Norm Phase
                for i in T.Parallel(block_M):
                    m_idx = pid_m * block_M + i
                    if m_idx < M:
                        # Col 0
                        col_sum_shared[i, 0] = T.max(
                            mat_shared[i, 0, 0] + mat_shared[i, 1, 0] +
                            mat_shared[i, 2, 0] + mat_shared[i, 3, 0], 1e-8)
                        mat_shared[i, 0, 0] /= col_sum_shared[i, 0]
                        mat_shared[i, 1, 0] /= col_sum_shared[i, 0]
                        mat_shared[i, 2, 0] /= col_sum_shared[i, 0]
                        mat_shared[i, 3, 0] /= col_sum_shared[i, 0]

                        # Col 1
                        col_sum_shared[i, 1] = T.max(
                            mat_shared[i, 0, 1] + mat_shared[i, 1, 1] +
                            mat_shared[i, 2, 1] + mat_shared[i, 3, 1], 1e-8)
                        mat_shared[i, 0, 1] /= col_sum_shared[i, 1]
                        mat_shared[i, 1, 1] /= col_sum_shared[i, 1]
                        mat_shared[i, 2, 1] /= col_sum_shared[i, 1]
                        mat_shared[i, 3, 1] /= col_sum_shared[i, 1]

                        # Col 2
                        col_sum_shared[i, 2] = T.max(
                            mat_shared[i, 0, 2] + mat_shared[i, 1, 2] +
                            mat_shared[i, 2, 2] + mat_shared[i, 3, 2], 1e-8)
                        mat_shared[i, 0, 2] /= col_sum_shared[i, 2]
                        mat_shared[i, 1, 2] /= col_sum_shared[i, 2]
                        mat_shared[i, 2, 2] /= col_sum_shared[i, 2]
                        mat_shared[i, 3, 2] /= col_sum_shared[i, 2]
                        
                        # Col 3
                        col_sum_shared[i, 3] = T.max(
                            mat_shared[i, 0, 3] + mat_shared[i, 1, 3] +
                            mat_shared[i, 2, 3] + mat_shared[i, 3, 3], 1e-8)
                        mat_shared[i, 0, 3] /= col_sum_shared[i, 3]
                        mat_shared[i, 1, 3] /= col_sum_shared[i, 3]
                        mat_shared[i, 2, 3] /= col_sum_shared[i, 3]
                        mat_shared[i, 3, 3] /= col_sum_shared[i, 3]            
            # Store Phase
            for i in T.Parallel(block_M):
                m_idx = pid_m * block_M + i
                if m_idx < M:
                    # Row 0
                    Out[m_idx, 0, 0] = T.cast(mat_shared[i, 0, 0], fp32)
                    Out[m_idx, 0, 1] = T.cast(mat_shared[i, 0, 1], fp32)
                    Out[m_idx, 0, 2] = T.cast(mat_shared[i, 0, 2], fp32)
                    Out[m_idx, 0, 3] = T.cast(mat_shared[i, 0, 3], fp32)
                    # Row 1
                    Out[m_idx, 1, 0] = T.cast(mat_shared[i, 1, 0], fp32)
                    Out[m_idx, 1, 1] = T.cast(mat_shared[i, 1, 1], fp32)
                    Out[m_idx, 1, 2] = T.cast(mat_shared[i, 1, 2], fp32)
                    Out[m_idx, 1, 3] = T.cast(mat_shared[i, 1, 3], fp32)
                    # Row 2
                    Out[m_idx, 2, 0] = T.cast(mat_shared[i, 2, 0], fp32)
                    Out[m_idx, 2, 1] = T.cast(mat_shared[i, 2, 1], fp32)
                    Out[m_idx, 2, 2] = T.cast(mat_shared[i, 2, 2], fp32)
                    Out[m_idx, 2, 3] = T.cast(mat_shared[i, 2, 3], fp32)
                    # Row 3
                    Out[m_idx, 3, 0] = T.cast(mat_shared[i, 3, 0], fp32)
                    Out[m_idx, 3, 1] = T.cast(mat_shared[i, 3, 1], fp32)
                    Out[m_idx, 3, 2] = T.cast(mat_shared[i, 3, 2], fp32)
                    Out[m_idx, 3, 3] = T.cast(mat_shared[i, 3, 3], fp32)

    return sinkhorn_func


def mhc_tilelang_sinkhorn(logits: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """
    Fused Sinkhorn-Knopp normalization.
    Args:
        logits: [M, n_streams, n_streams], fp32
        n_iters: int, default 20
    Returns:
        matrix: [M, n_streams, n_streams], fp32
    """
    M, N, _ = logits.shape
    logits_in = logits

    out_h_res = torch.empty((M, N, N), dtype=torch.float32, device=logits.device)
    out = out_h_res

    if not logits_in.is_contiguous():
        logits_in = logits_in.contiguous()

    if N != 4:
        raise ValueError(f"sinkhorn only supports n_streams=4, got {N}")
    
    # No cast needed, accept FP32
    
    BLOCK_M = 128
    m_main = (M // BLOCK_M) * BLOCK_M
    m_tail = M % BLOCK_M
    
    if M <= 128:
        threads = 32
        block_M_kernel = 32
    else:
        threads = 32
        block_M_kernel = 64

    # 1. Main aligned part - no padding needed
    if m_main > 0:
        logits_main = logits_in[:m_main]
        out_main = out[:m_main]
        kernel_main = sinkhorn_kernel(m_main, N, threads, block_M_kernel)
        kernel_main(logits_main, out_main)
        
    # 2. Tail part
    if m_tail > 0:
        logits_tail = logits_in[m_main:]
        out_tail = out[m_main:]

        # Optimization: Remove padding for all cases to improve performance
        kernel_tail = sinkhorn_kernel(m_tail, N, threads, block_M_kernel)
        kernel_tail(logits_tail, out_tail)

    return out_h_res

@tilelang.jit(pass_configs=pass_configs)
def expand_merge_kernel(M, n_streams, HiddenDim, threads, block_M, block_D):
    """
    TileLang kernel for fused expand and merge operations.
    
    Args:
        M: Batch dimension.
        n_streams: Number of streams (must be 4).
        HiddenDim: Hidden dimension size.
        threads: Number of threads per block.
        block_M: Block size for M dimension.
        block_D: Block size for D dimension.
        
    Returns:
        Compiled kernel function.
    """
    
    bf16 = "bfloat16"
    fp32 = "float32"
    N = n_streams
    
    M = T.dynamic("M")

    @T.prim_func
    def expand_merge_func(
        H_res: T.Tensor[(M, N, N), fp32],
        Residuals: T.Tensor[(M, N, HiddenDim), bf16],
        LayerOut: T.Tensor[(M, HiddenDim), bf16],
        H_post: T.Tensor[(M, N), fp32],
        Out: T.Tensor[(M, N, HiddenDim), bf16]
    ):
        with T.Kernel(
            T.ceildiv(M, block_M),
            T.ceildiv(HiddenDim, block_D),
            threads=threads
        ) as (pid_m, pid_d):
            
            # Shared memory cache for H_res and H_post
            H_res_shared = T.alloc_shared([block_M, N * N], fp32)
            H_post_shared = T.alloc_shared([block_M, N], fp32)
            
            # Cooperative load of H_res
            for tid in T.Parallel(threads):
                if tid < block_M * N * N:
                    row = tid // (N * N)
                    col = tid % (N * N)
                    global_row = pid_m * block_M + row
                    if global_row < M:
                        # Flatten H_res load for shared memory packing
                        # row=local_m, col=flattened_NN
                        n_row = col // N
                        n_col = col % N
                        H_res_shared[row, col] = H_res[global_row, n_row, n_col]
            
            # Cooperative load of H_post
            for tid in T.Parallel(threads):
                if tid < block_M * N:
                    row = tid // N
                    col = tid % N
                    global_row = pid_m * block_M + row
                    if global_row < M:
                        H_post_shared[row, col] = H_post[global_row, col]
            
            T.sync_threads()
            
            # Main computation
            for tid in T.Parallel(threads):
                m_offset = tid // block_D
                d_offset = tid % block_D
                
                m_idx = pid_m * block_M + m_offset
                d_idx = pid_d * block_D + d_offset
                
                if m_idx < M and d_idx < HiddenDim:
                    # Read H_post from shared memory
                    hp0 = H_post_shared[m_offset, 0]
                    hp1 = H_post_shared[m_offset, 1]
                    hp2 = H_post_shared[m_offset, 2]
                    hp3 = H_post_shared[m_offset, 3]
                    
                    # Read Residuals
                    res0 = T.cast(Residuals[m_idx, 0, d_idx], fp32)
                    res1 = T.cast(Residuals[m_idx, 1, d_idx], fp32)
                    res2 = T.cast(Residuals[m_idx, 2, d_idx], fp32)
                    res3 = T.cast(Residuals[m_idx, 3, d_idx], fp32)
                    
                    # Read LayerOut
                    l_out = T.cast(LayerOut[m_idx, d_idx], fp32)
                    
                    # Compute Out[m, 0, d]
                    out0_0 = l_out * hp0
                    out0_1 = out0_0 + H_res_shared[m_offset, 0] * res0
                    out0_2 = out0_1 + H_res_shared[m_offset, 1] * res1
                    out0_3 = out0_2 + H_res_shared[m_offset, 2] * res2
                    out0_4 = out0_3 + H_res_shared[m_offset, 3] * res3
                    Out[m_idx, 0, d_idx] = T.cast(out0_4, bf16)
                    
                    # Compute Out[m, 1, d]
                    out1_0 = l_out * hp1
                    out1_1 = out1_0 + H_res_shared[m_offset, 4] * res0
                    out1_2 = out1_1 + H_res_shared[m_offset, 5] * res1
                    out1_3 = out1_2 + H_res_shared[m_offset, 6] * res2
                    out1_4 = out1_3 + H_res_shared[m_offset, 7] * res3
                    Out[m_idx, 1, d_idx] = T.cast(out1_4, bf16)
                    
                    # Compute Out[m, 2, d]
                    out2_0 = l_out * hp2
                    out2_1 = out2_0 + H_res_shared[m_offset, 8] * res0
                    out2_2 = out2_1 + H_res_shared[m_offset, 9] * res1
                    out2_3 = out2_2 + H_res_shared[m_offset, 10] * res2
                    out2_4 = out2_3 + H_res_shared[m_offset, 11] * res3
                    Out[m_idx, 2, d_idx] = T.cast(out2_4, bf16)
                    
                    # Compute Out[m, 3, d]
                    out3_0 = l_out * hp3
                    out3_1 = out3_0 + H_res_shared[m_offset, 12] * res0
                    out3_2 = out3_1 + H_res_shared[m_offset, 13] * res1
                    out3_3 = out3_2 + H_res_shared[m_offset, 14] * res2
                    out3_4 = out3_3 + H_res_shared[m_offset, 15] * res3
                    Out[m_idx, 3, d_idx] = T.cast(out3_4, bf16)
    
    return expand_merge_func

def mhc_tilelang_expand_merge(residuals, layer_output, h_res, h_post):
    """
    Fused expand and merge operations.
    Args:
        residuals: [M, n_streams, D], bf16
        layer_output: [M, D], bf16
        h_res: [M, n_streams, n_streams], fp32
        h_post: [M, n_streams], fp32
    Returns:
        mixed: [M, n_streams, D], bf16
    """
    M, N, D = residuals.shape
    h_res_in = h_res
    res_in = residuals
    l_out_in = layer_output
    h_post_in = h_post

    # Kernel uses BF16 for output
    out_residuals = torch.empty((M, N, D), dtype=torch.bfloat16, device=residuals.device)
    out = out_residuals

    if not h_res_in.is_contiguous():
        h_res_in = h_res_in.contiguous()
    if not res_in.is_contiguous():
        res_in = res_in.contiguous()
    if not l_out_in.is_contiguous():
        l_out_in = l_out_in.contiguous()
    if not h_post_in.is_contiguous():
        h_post_in = h_post_in.contiguous()

    if N != 4:
        raise ValueError(f"expand_merge only supports n_streams=4, got {N}")
    
    # Tuned for M=128 and M=5120 on H800
    # Note: Ensure threads // block_D <= block_M to avoid shared memory OOB
    if M <= 128:
        threads = 128
        block_M = 2
        block_D = 64
    else:
        threads = 128
        block_M = 1
        block_D = 128

    kernel = expand_merge_kernel(M, N, D, threads, block_M, block_D)
    kernel(h_res_in, res_in, l_out_in, h_post_in, out)

    return out_residuals