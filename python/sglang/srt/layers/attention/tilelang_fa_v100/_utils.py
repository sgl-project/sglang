"""Shared utilities for TileLang FA V100 kernels."""


def _is_valid_gemm2(block_M, dim, num_warps):
    """Check if GEMM 2 (M=block_M, N=dim) is valid for SM70 with Square policy."""
    max_m_warps = block_M // 16
    max_n_warps = dim // 16
    for m in range(1, min(max_m_warps, num_warps) + 1):
        if num_warps % m != 0:
            continue
        n = num_warps // m
        if n <= max_n_warps:
            return True
    return False


def _smem_forward(block_M, block_N, dim):
    """Forward SMEM in bytes: Q_shared + K_shared + V_shared + P_shared."""
    return (block_M * dim + 2 * block_N * dim + block_M * block_N) * 2


def _smem_backward(block_M, block_N, dim):
    """Backward SMEM in bytes: Q_shared + K_shared + V_shared + P_shared (no dO_shared re-use)."""
    return 4 * dim * (block_M + block_N) + 2 * block_M * block_N


MAX_SMEM = 86000
