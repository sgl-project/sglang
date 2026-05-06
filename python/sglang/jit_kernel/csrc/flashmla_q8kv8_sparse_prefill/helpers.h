#pragma once

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

namespace sm90 {

__forceinline__ __device__ void cp_async_cacheglobal_l2_prefetch_256B(const void* src, void* dst) {
  uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst);
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], %2;\n" ::"r"(dst_addr), "l"(src), "n"(16));
}

__forceinline__ __device__ void
cp_async_cacheglobal_l2_prefetch_256B(const void* src, void* dst, bool pred, int64_t cache_policy) {
  uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst);
  asm volatile(
      "cp.async.cg.shared.global.L2::cache_hint.L2::256B [%0], [%1], 16, %2, %3;\n" ::"r"(dst_addr),
      "l"(src),
      "r"(pred ? 16 : 0),
      "l"(cache_policy));
}

__forceinline__ __device__ int64_t createpolicy_evict_last() {
  int64_t res;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0; \n\t" : "=l"(res) :);
  return res;
}

__forceinline__ __device__ int64_t createpolicy_evict_first() {
  int64_t res;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0; \n\t" : "=l"(res) :);
  return res;
}

__forceinline__ __device__ int get_AorC_row_idx(int local_row_idx, int idx_in_warpgroup) {
  // In the layout of fragment A and fragment C during WGMMA, the data each thread holds resides in two particular rows.
  // This function converts the local_row_idx (0~2) to the actual row_idx You may refer to this link for the detailed
  // layout: https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-a
  int row_idx = (idx_in_warpgroup / 32) * 16 + local_row_idx * 8 + (idx_in_warpgroup % 32 / 4);
  return row_idx;
}

__forceinline__ __device__ int get_AorC_col_idx(int local_elem_idx, int idx_in_warpgroup) {
  int col_idx = 8 * (local_elem_idx / 4) + (idx_in_warpgroup % 4) * 2 + (local_elem_idx & 1);
  return col_idx;
}

// Adapted from
// https://github.com/Dao-AILab/flash-attention/blob/cdaf2de6e95cb05400959b5ab984f66e4c7df317/hopper/utils.h
// * Copyright (c) 2024, Tri Dao.
template <
    bool zero_init = false,
    int wg_wait = 0,
    bool arrive = true,
    bool commit = true,
    typename Tensor0,
    typename Tensor1,
    typename Tensor2,
    typename TiledMma>
__forceinline__ __device__ void gemm(TiledMma& tiled_mma, Tensor0 const& tCrA, Tensor1 const& tCrB, Tensor2& tCrC) {
  using namespace cute;
  constexpr bool Is_RS = !cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;
  // Need to cast away const on tCrA since warpgroup_fence_operand doesn't take const
  if constexpr (Is_RS) {
    cute::warpgroup_fence_operand(const_cast<Tensor0&>(tCrA));
  }
  warpgroup_fence_operand(tCrC);
  if constexpr (arrive) {
    warpgroup_arrive();
  }
  if constexpr (zero_init) {
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    // Unroll the K mode manually to set scale D to 1
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
  } else {
    // cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
    // Unroll the K mode manually to set scale D to 1
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
  }
  if constexpr (commit) {
    warpgroup_commit_batch();
  }
  if constexpr (wg_wait >= 0) {
    warpgroup_wait<wg_wait>();
  }
  warpgroup_fence_operand(tCrC);
  if constexpr (Is_RS) {
    warpgroup_fence_operand(const_cast<Tensor0&>(tCrA));
  }
}

// A simpler version of gemm
template <typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
__forceinline__ __device__ void gemm_ss(
    bool clear_accum,
    TiledMma tiled_mma,
    Tensor0 const& sA,
    Tensor1 const& sB,
    Tensor2& rC_frag,
    int idx_in_warpgroup) {
  using namespace cute;
  ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
  Tensor sA_frag = thr_mma.partition_fragment_A(sA);
  Tensor sB_frag = thr_mma.partition_fragment_B(sB);
  static_assert(size<2>(sA_frag) == size<2>(sB_frag));

  warpgroup_fence_operand(rC_frag);
  warpgroup_arrive();
  tiled_mma.accumulate_ = clear_accum ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
  CUTLASS_PRAGMA_UNROLL
  for (int k = 0; k < size<2>(sA_frag); ++k) {
    cute::gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), rC_frag);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
  }
  warpgroup_fence_operand(rC_frag);
}

template <typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
__forceinline__ __device__ void gemm_rs(
    bool clear_accum, TiledMma tiled_mma, Tensor0 rA_frag, Tensor1 const& sB, Tensor2& rC_frag, int idx_in_warpgroup) {
  using namespace cute;
  ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
  Tensor sB_frag = thr_mma.partition_fragment_B(sB);
  static_assert(size<2>(rA_frag) == size<2>(sB_frag));

  warpgroup_fence_operand(const_cast<Tensor0&>(rA_frag));
  warpgroup_fence_operand(rC_frag);
  warpgroup_arrive();
  tiled_mma.accumulate_ = clear_accum ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
  CUTLASS_PRAGMA_UNROLL
  for (int k = 0; k < size<2>(rA_frag); ++k) {
    cute::gemm(tiled_mma, rA_frag(_, _, k), sB_frag(_, _, k), rC_frag);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
  }
  warpgroup_fence_operand(rC_frag);
  warpgroup_fence_operand(const_cast<Tensor0&>(rA_frag));
}

__forceinline__ __device__ uint32_t get_sm_id() {
  uint32_t ret;
  asm("mov.u32 %0, %%smid;" : "=r"(ret));
  return ret;
}

static constexpr int PEER_ADDR_MASK =
    16777216;  // peer_addr = my_addr ^ PEER_ADDR_MASK. Not sure if this number is the same on all GPUs.
template <typename T>
CUTE_DEVICE T* get_peer_addr(const T* p) {
  return (T*)((int64_t)(p) ^ PEER_ADDR_MASK);
}

template <typename TMA, typename Tensor0, typename Tensor1>
CUTE_DEVICE void launch_tma_copy(
    const TMA& tma_copy,
    Tensor0 src,
    Tensor1 dst,
    cutlass::arch::ClusterTransactionBarrier& bar,
    const cute::TMA::CacheHintSm90& cache_hint = cute::TMA::CacheHintSm90::EVICT_NORMAL) {
  auto thr_tma = tma_copy.get_slice(cute::_0{});
  cute::copy(
      tma_copy.with(
          reinterpret_cast<typename cutlass::arch::ClusterTransactionBarrier::ValueType&>(bar), 0, cache_hint),
      thr_tma.partition_S(src),
      thr_tma.partition_D(dst));
}

}  // namespace sm90
