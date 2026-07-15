#pragma once

#include <cooperative_groups.h>

#include <cute/tensor.hpp>

#include "../config.h"

using namespace cute;

namespace sm90::decode::sparse_nvfp4_dsv4 {

// In the layout of fragment A and fragment C during WGMMA, data each thread holds resides in two particular rows. This
// function converts the local_row_idx (0~1) to the actual row_idx You may refer to this link for the detailed layout:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-a
__forceinline__ __device__ int get_AorC_row_idx(int local_row_idx, int idx_in_warpgroup) {
  int row_idx = (idx_in_warpgroup / 32) * 16 + local_row_idx * 8 + (idx_in_warpgroup % 32 / 4);
  return row_idx;
}

// Adapted from
// https://github.com/Dao-AILab/flash-attention/blob/cdaf2de6e95cb05400959b5ab984f66e4c7df317/hopper/utils.h
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

template <typename TMA, typename Tensor0, typename Tensor1>
CUTE_DEVICE void launch_tma_copy(
    const TMA& tma_copy,
    const Tensor0& src,
    Tensor1& dst,
    transac_bar_t& bar,
    const cute::TMA::CacheHintSm90& cache_hint = cute::TMA::CacheHintSm90::EVICT_NORMAL,
    const uint16_t& multicast_mask = 0) {
  auto thr_tma = tma_copy.get_slice(_0{});
  cute::copy(
      tma_copy.with(reinterpret_cast<typename transac_bar_t::ValueType&>(bar), multicast_mask, cache_hint),
      thr_tma.partition_S(src),
      thr_tma.partition_D(dst));
}

template <typename T>
CUTE_DEVICE static void st_async_128b(void* dst_ptr, const T& data, const transac_bar_t* mbar_ptr) {
  long2 data_long2 = *reinterpret_cast<const long2*>(&data);
  uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst_ptr);
  uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(mbar_ptr);
  asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.s64 [%0], {%1, %2}, [%3]; \n"
               :
               : "r"(dst_addr), "l"(data_long2.x), "l"(data_long2.y), "r"(mbar_addr));
}

CUTE_DEVICE
static void cp_async_bulk_shared_cta_shared_cluster(void* dst_ptr, void* src_ptr, int size, transac_bar_t* mbar_ptr) {
  uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst_ptr);
  uint32_t src_addr = cute::cast_smem_ptr_to_uint(src_ptr);
  uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(mbar_ptr);
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3]; \n"
               :
               : "r"(dst_addr), "r"(src_addr), "r"(size), "r"(mbar_addr));
}

static constexpr int PEER_ADDR_MASK = 16777216;  // peer_addr = my_addr ^ PEER_ADDR_MASK.
template <typename T>
CUTE_DEVICE T* get_peer_addr(T* p) {
  return (T*)((int64_t)(p) ^ PEER_ADDR_MASK);
}

}  // namespace sm90::decode::sparse_nvfp4_dsv4
