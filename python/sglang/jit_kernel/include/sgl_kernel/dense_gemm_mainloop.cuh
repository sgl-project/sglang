/// Extracted reusable utilities from kimi_k3/comm/gemm_ar.cuh.

#pragma once

#include <cuda_runtime.h>

// ================= recipes/dense_gemm_mainloop/kernel.cuh =================
// (reduced at assembly: this recipe header contributes only
// group_n_swizzle to the single file)
namespace dense_gemm_mainloop {
// ============================================================================
// GROUP_N L2-stripe swizzle primitive — group_n_swizzle<CTA_GROUP, GROUP_N>.
// ============================================================================
//
// Per-tile (linear → (bid_m, bid_n)) raster used by the entire 5-kernel
// fused_gemm family (1cta dense, 2cta cluster, 1cta SF, 2cta SF, grouped
// dense) to walk an output GEMM grid in N-stripes of width `GROUP_N`. The
// stripe walk groups B tiles that share the same `bid_n` band so the same
// chunk of B sits resident in L2 across all M-rows in the stripe — cite
// `recipes/scheduler_picking/` §3 (DeepGEMM's `get_swizzled_block_idx`).
//
// CTA_GROUP=1 path (1cta dense / 1cta SF):
//     bid_m = in_group / num_n_in_group
//     bid_n = first_n + in_group % num_n_in_group
//     (`linear` = tile index in row-major order; `crank` is ignored)
//
// CTA_GROUP=2 path (2cta cluster MMA, 2cta SF, grouped dense):
//     c             = linear / 2          (cluster slot)
//     r             = linear & 1          (intra-cluster M lane)
//     cluster_bid_m = in_group / num_n_in_group
//     cluster_bid_n = first_n + in_group % num_n_in_group
//     bid_m         = cluster_bid_m * 2 + r
//     bid_n         = cluster_bid_n
//
// Caller-side conventions (cite recipes/gemm_epi/transpose_picking/fused_gemm_2cta/kernel.cu
// and kernels/gemm/dense_1cta/kernel.cu:290-301):
//   - `cluster_grid_m` is the M-axis tile count in CLUSTER units
//     (= grid_m / CTA_GROUP); for CTA_GROUP=1 the caller passes grid_m.
//   - `grid_n` is the N-axis tile count (NOT halved by CTA_GROUP).
//   - The tail-stripe is shorter than `GROUP_N` when `grid_n % GROUP_N != 0`;
//     `num_n_in_group = min(GROUP_N, grid_n - first_n)` handles that.
//
// Byte-id proof vs the inline 11-line lambda the 5 kernels each carry:
// the math is character-for-character identical (modulo function-call
// abstraction). `__forceinline__` + identical operand types collapses the
// call to the same SASS as the inline lambda; the 4 `tile_mn` lambdas
// in fused_gemm_2cta_sf alone are 8 LoC × 4 = 32 LoC of duplication that
// distill to ONE `group_n_swizzle<2, GROUP_N>(...)` call site.
template <int CTA_GROUP, int GROUP_N>
__device__ __forceinline__ int2 group_n_swizzle(
        int linear, int crank, int cluster_grid_m, int grid_n) {
    if constexpr (CTA_GROUP == 1) {
        const int num_blocks_per_group = cluster_grid_m * GROUP_N;
        const int group_idx = linear / num_blocks_per_group;
        const int first_n   = group_idx * GROUP_N;
        const int in_group  = linear - group_idx * num_blocks_per_group;
        const int num_n_in_group = grid_n - first_n < GROUP_N
                                       ? grid_n - first_n
                                       : GROUP_N;
        const int bid_m = in_group / num_n_in_group;
        const int bid_n = first_n + (in_group % num_n_in_group);
        (void)crank;
        return {bid_m, bid_n};
    } else {
        const int c = linear / CTA_GROUP;
        const int r = linear & (CTA_GROUP - 1);
        const int num_blocks_per_group = cluster_grid_m * GROUP_N;
        const int group_idx = c / num_blocks_per_group;
        const int first_n = group_idx * GROUP_N;
        const int in_group = c - group_idx * num_blocks_per_group;
        const int num_n_in_group = grid_n - first_n < GROUP_N
                                       ? grid_n - first_n
                                       : GROUP_N;
        const int cluster_bid_m = in_group / num_n_in_group;
        const int cluster_bid_n = first_n + (in_group % num_n_in_group);
        const int bid_m = cluster_bid_m * CTA_GROUP + r;
        const int bid_n = cluster_bid_n;
        (void)crank;  // `r` already encodes the intra-cluster lane.
        return {bid_m, bid_n};
    }
}
}  // namespace dense_gemm_mainloop
