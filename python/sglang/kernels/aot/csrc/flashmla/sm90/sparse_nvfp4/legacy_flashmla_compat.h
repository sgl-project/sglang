/*
 * Copyright (c) 2026 SGLang Team
 *
 * Portions adapted from FlashMLA, commit
 * be055fb7df0090fde45f08e9cb5b8b4c0272da73.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>

using bf16 = cutlass::bfloat16_t;
using fp8 = cutlass::float_e4m3_t;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;

using namespace cute;

static constexpr int NUM_THREADS = 128 * 3;
static constexpr int BLOCK_M = 64;
static constexpr int TOPK_BLOCK_SIZE = 64;
static constexpr int PAGE_BLOCK_SIZE = 64;
static constexpr int QUANT_TILE_SIZE = 128;

static constexpr int HEAD_DIM_K = 576;
static constexpr int HEAD_DIM_V = 512;
static constexpr int HEAD_DIM_NOPE = HEAD_DIM_V;
static constexpr int HEAD_DIM_ROPE = HEAD_DIM_K - HEAD_DIM_V;
static constexpr int NUM_SCALES = HEAD_DIM_NOPE / QUANT_TILE_SIZE;
static constexpr int NUM_BYTES_PER_TOKEN = HEAD_DIM_NOPE + NUM_SCALES * sizeof(float) + HEAD_DIM_ROPE * sizeof(bf16);

static constexpr int NUM_K_BUFS = 2;

using SmemLayoutQTile =
    decltype(tile_to_shape(GMMA::Layout_SW128_Atom<bf16, GMMA::Major::K>{}, Shape<Int<BLOCK_M>, Int<64>>{}));

template <int NUM_TILES>
using SmemLayoutQTiles =
    decltype(tile_to_shape(SmemLayoutQTile{}, Shape<Int<BLOCK_M>, Int<64 * NUM_TILES>>{}, Step<_1, _2>{}));

using SmemLayoutQ = SmemLayoutQTiles<9>;

using SmemLayoutKTile = decltype(tile_to_shape(
    GMMA::Layout_INTER_Atom<bf16, GMMA::Major::K>{}, Shape<Int<TOPK_BLOCK_SIZE>, _64>{}, Step<_1, _2>{}));

template <int NUM_TILES>
using SmemLayoutKTiles =
    decltype(tile_to_shape(SmemLayoutKTile{}, Shape<Int<TOPK_BLOCK_SIZE>, Int<64 * NUM_TILES>>{}, Step<_1, _2>{}));

template <int NUM_TILES>
using SmemLayoutKTilesTransposed = decltype(composition(
    SmemLayoutKTiles<NUM_TILES>{},
    Layout<Shape<Int<64 * NUM_TILES>, Int<TOPK_BLOCK_SIZE>>, Stride<Int<TOPK_BLOCK_SIZE>, _1>>{}));

using SmemLayoutOBuf =
    decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{}));

using SmemLayoutOAccumBuf = Layout<Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>, Stride<Int<520>, _1>>;

using SmemLayoutK = SmemLayoutKTiles<9>;
using SmemLayoutV = SmemLayoutKTilesTransposed<8>;
using SmemLayoutHalfV = SmemLayoutKTilesTransposed<4>;

using SmemLayoutS =
    decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}));

struct SharedMemoryPlan {
  array_aligned<bf16, cosize_v<SmemLayoutQ>> q;
  union {
    array_aligned<bf16, cosize_v<SmemLayoutK>> k[NUM_K_BUFS];
    array_aligned<bf16, cosize_v<SmemLayoutOBuf>> oBuf;
    array_aligned<float, cosize_v<SmemLayoutOAccumBuf>> oAccumBuf;
  } u;
  array_aligned<bf16, cosize_v<SmemLayoutS>> s;
  bool is_kv_valid[NUM_K_BUFS][TOPK_BLOCK_SIZE];

  float sM[BLOCK_M], sL[BLOCK_M], sScale[BLOCK_M];
  transac_bar_t bar_q, bar_k_local_ready[NUM_K_BUFS], bar_k_remote_ready[NUM_K_BUFS], bar_k_avail[NUM_K_BUFS];
};

template <typename Shape_Q, typename TMA_Q, typename Shape_O, typename TMA_O>
struct TmaParams {
  Shape_Q shape_Q;
  TMA_Q tma_Q;
  Shape_O shape_O;
  TMA_O tma_O;
};

using TiledMMA_QK = decltype(make_tiled_mma(
    GMMA::MMA_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}, Layout<Shape<_1, _1, _1>>{}));

using TiledMMA_QK_rQ = decltype(make_tiled_mma(
    GMMA::MMA_64x64x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{}, Layout<Shape<_1, _1, _1>>{}));

using TiledMMA_PV_LocalP = decltype(make_tiled_mma(
    GMMA::MMA_64x256x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}, Layout<Shape<_1, _1, _1>>{}));

using TiledMMA_PV_RemoteP = decltype(make_tiled_mma(
    GMMA::MMA_64x256x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::MN>{}, Layout<Shape<_1, _1, _1>>{}));

enum NamedBarriers : uint32_t {
  sScale_and_sS_ready = 0,
  sScale_and_sS_free = 1,
  oBuf_free_and_sL_ready = 2,
  epilogue_r2s_ready = 3,
  batch_loop_sync = 4,
  warpgroup0_sync = 5
};

__forceinline__ __device__ int get_AorC_row_idx(int local_row_idx, int idx_in_warpgroup) {
  return (idx_in_warpgroup / 32) * 16 + local_row_idx * 8 + (idx_in_warpgroup % 32 / 4);
}

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
  if constexpr (Is_RS) {
    cute::warpgroup_fence_operand(const_cast<Tensor0&>(tCrA));
  }
  warpgroup_fence_operand(tCrC);
  if constexpr (arrive) {
    warpgroup_arrive();
  }
  if constexpr (zero_init) {
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
  }
  CUTLASS_PRAGMA_UNROLL
  for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
    cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
  }
  if constexpr (commit) {
    warpgroup_commit_batch();
  }
  if constexpr (wg_wait >= 0) {
    warpgroup_wait<wg_wait>();
  }
  warpgroup_fence_operand(tCrC);
  if constexpr (Is_RS) {
    cute::warpgroup_fence_operand(const_cast<Tensor0&>(tCrA));
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

static constexpr int PEER_ADDR_MASK = 16777216;

template <typename T>
CUTE_DEVICE T* get_peer_addr(const T* p) {
  return reinterpret_cast<T*>(reinterpret_cast<int64_t>(p) ^ PEER_ADDR_MASK);
}

template <bool IS_NO_SPLIT, typename TMAParams, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3>
__forceinline__ __device__ void store_o(
    Tensor0& rO,
    Tensor1& gOorAccum,
    Tensor2& sOutputBuf,
    Tensor3& sOutputAccumBuf,
    float rL[2],
    TMAParams& tma_params,
    int batch_idx,
    int s_q_idx,
    int head_block_idx,
    int num_valid_seq_q,
    int warpgroup_idx,
    int idx_in_warpgroup) {
  using cutlass::arch::NamedBarrier;
  if constexpr (IS_NO_SPLIT) {
    Tensor rOb = make_tensor_like<bf16>(rO);
    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < size(rO); ++idx) {
      rOb(idx) = static_cast<bf16>(rO(idx) / rL[idx % 4 >= 2]);
    }

    Tensor sMyOutputBuf = local_tile(sOutputBuf, Shape<_64, _256>{}, make_coord(_0{}, warpgroup_idx));
    TiledCopy r2s_tiled_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, bf16>{}, TiledMMA_PV_LocalP{});
    ThrCopy r2s_thr_copy = r2s_tiled_copy.get_slice(idx_in_warpgroup);
    Tensor r2s_thr_copy_rOb = r2s_thr_copy.retile_S(rOb);
    Tensor r2s_thr_copy_sMyOutputBuf = r2s_thr_copy.partition_D(sMyOutputBuf);
    cute::copy(r2s_tiled_copy, r2s_thr_copy_rOb, r2s_thr_copy_sMyOutputBuf);
    cutlass::arch::fence_view_async_shared();

    NamedBarrier::arrive_and_wait(256, NamedBarriers::epilogue_r2s_ready);

    if (threadIdx.x == 0) {
      Tensor tma_gO = tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, s_q_idx, batch_idx);
      auto thr_tma = tma_params.tma_O.get_slice(_0{});
      Tensor my_tma_gO = flat_divide(tma_gO, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{})(_, _, head_block_idx, _0{});
      cute::copy(tma_params.tma_O, thr_tma.partition_S(sOutputBuf), thr_tma.partition_D(my_tma_gO));
      cute::tma_store_arrive();
    }
  } else {
    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < size(rO); idx += 2) {
      int row = (idx_in_warpgroup / 32) * 16 + (idx_in_warpgroup % 32 / 4) + (idx % 4 >= 2 ? 8 : 0);
      int col = warpgroup_idx * 256 + (idx_in_warpgroup % 4) * 2 + idx / 4 * 8;
      *reinterpret_cast<float2*>(&(sOutputAccumBuf(row, col))) =
          float2{rO(idx) / rL[idx % 4 >= 2], rO(idx + 1) / rL[idx % 4 >= 2]};
    }
    cutlass::arch::fence_view_async_shared();

    NamedBarrier::arrive_and_wait(256, NamedBarriers::epilogue_r2s_ready);

    if (elect_one_sync()) {
      CUTLASS_PRAGMA_UNROLL
      for (int local_row = 0; local_row < BLOCK_M / (256 / 32); ++local_row) {
        int row = local_row * (256 / 32) + (threadIdx.x / 32);
        if (row < num_valid_seq_q) {
          SM90_BULK_COPY_S2G::copy(&sOutputAccumBuf(row, _0{}), &gOorAccum(row, _0{}), HEAD_DIM_V * sizeof(float));
        }
      }
      cute::tma_store_arrive();
    }
  }
}
