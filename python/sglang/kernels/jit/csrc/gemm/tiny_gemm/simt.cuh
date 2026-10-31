// CUDA-core tiny GEMM (CUDA and ROCm). A lane holds 16 elements of each row of x, so a lane group of
// kLanes = K / 16 lanes covers one row of w. A CTA owns kNTile rows of w (in registers), kRowsPerLane per
// lane group. A group reduces over its lanes with a shuffle tree, and over its warps in order through shared
// memory when kLanes > 32. A CTA serves up to kMaxMTile rows of x; larger batches copy each weight tile into
// the fewest CTAs along blockIdx.y, with the rows spread evenly. A row's summation order depends on neither
// the tile nor the other rows, so its result does not depend on the batch.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/gemm/dot_product.cuh>

#include "common.cuh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace sglang {

template <uint32_t N_, uint32_t K_, typename Weight_, uint32_t kNTile_, uint32_t kMaxMTile_, uint32_t kRowsPerLane_>
struct SIMTTrait {
  using Weight = Weight_;
  static constexpr uint32_t N = N_;
  static constexpr uint32_t K = K_;
  static constexpr uint32_t kNTile = kNTile_;              // rows of w per CTA
  static constexpr uint32_t kMaxMTile = kMaxMTile_;        // rows of x per CTA
  static constexpr uint32_t kRowsPerLane = kRowsPerLane_;  // rows of w per lane group
  static constexpr uint32_t kElems = 16;                   // elements of x per lane and row of x
  static_assert(K % kElems == 0, "K must be a multiple of 16");
  static constexpr uint32_t kLanes = K / kElems;  // lanes covering one row of w
  static constexpr bool kCrossWarp = kLanes > 32;
  static_assert(kCrossWarp ? kLanes % 32 == 0 : 32 % kLanes == 0, "K / 16 must divide or be a multiple of 32");
  static_assert(kNTile % kRowsPerLane == 0, "N_TILE must be a multiple of the rows per lane group");
  static constexpr uint32_t kGroups = kNTile / kRowsPerLane;
  static constexpr uint32_t kBlockSize = kGroups * kLanes;
  static_assert(N % kNTile == 0, "N must be a multiple of N_TILE");
  static_assert(kBlockSize % 32 == 0 && kBlockSize <= 1024, "block size must be whole warps, at most 1024");
  static_assert(!kCrossWarp || kRowsPerLane * kMaxMTile <= kLanes, "one thread of a group finishes one output");
  static constexpr uint32_t kNumBlocks = N / kNTile;
  // The widest vector of w; x follows its element count. Lane l reads vectors l, l + L, ... of a row.
  static constexpr uint32_t kVecSize = device::kMaxVecBytes / sizeof(Weight);
  static constexpr uint32_t kUnroll = kElems / kVecSize;
  static constexpr uint32_t kXAlign = kVecSize * sizeof(bf16_t);
  static constexpr uint32_t kWAlign = kVecSize * sizeof(Weight);
};

// M_TILE rows of x per CTA, tiled over blockIdx.y. The last tile ends at the last row, so it may overlap the
// one before; the overlapping rows are computed twice and stored with the same bits.
template <typename Trait, uint32_t M_TILE, typename Out, bool kUsePDL>
__global__ __launch_bounds__(Trait::kBlockSize, 1) void tiny_simt_kernel(const TinyGEMMParams params) {
  using namespace device;
  using Weight = typename Trait::Weight;
  constexpr uint32_t N = Trait::N;
  constexpr uint32_t K = Trait::K;
  constexpr uint32_t kLanes = Trait::kLanes;
  constexpr uint32_t kRows = Trait::kRowsPerLane;
  constexpr uint32_t kVecSize = Trait::kVecSize;
  constexpr uint32_t kUnroll = Trait::kUnroll;
  constexpr uint32_t kOutputs = M_TILE * kRows;  // per lane group
  using vec_t = AlignedVector<bf16x2_t, kVecSize / 2>;
  using weight_vec_t = AlignedVector<packed_t<Weight>, kVecSize / 2>;

  const uint32_t lane = threadIdx.x % kLanes;
  const uint32_t group = threadIdx.x / kLanes;
  const uint32_t n_begin = blockIdx.x * Trait::kNTile + group * kRows;
  const uint32_t m_begin = min(blockIdx.y * M_TILE, params.num_tokens - M_TILE);
  const auto w_tile = static_cast<const Weight*>(params.w) + n_begin * K;

  // The weight does not depend on the producer: load it before the PDL wait.
  weight_vec_t wv[kRows][kUnroll];
#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      wv[r][u].load(w_tile + r * K, lane + u * kLanes);
    }
  }

  PDLWaitPrimary<kUsePDL>();

  const auto x_tile = params.x + m_begin * params.stride_x;
  vec_t xv[M_TILE][kUnroll];
#pragma unroll
  for (uint32_t m = 0; m < M_TILE; ++m) {
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      xv[m][u].load(tiny_gemm::assume_global(x_tile + m * params.stride_x), lane + u * kLanes);
    }
  }

  const auto out = static_cast<Out*>(params.out);
  const auto store = [&](uint32_t o, float value) {
    out[(m_begin + o / kRows) * N + n_begin + o % kRows] = cast<Out>(value);
  };
  if constexpr (!Trait::kCrossWarp) {
#pragma unroll
    for (uint32_t o = 0; o < kOutputs; ++o) {
      float acc = 0.0f;
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        dot_product_vec(xv[o / kRows][u], wv[o % kRows][u], acc);
      }
      const auto sum = warp::reduce_sum<kLanes>(acc);
      // Every lane holds the sum; the lanes of a group take turns storing.
      if (lane == o % kLanes) store(o, sum);
    }
    PDLTriggerSecondary<kUsePDL>();
  } else {
    constexpr uint32_t kGroupWarps = kLanes / kWarpThreads;
    __shared__ float s_acc[Trait::kGroups][kGroupWarps][kOutputs];
    const uint32_t group_warp = lane / kWarpThreads;
#pragma unroll
    for (uint32_t o = 0; o < kOutputs; ++o) {
      float acc = 0.0f;
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        dot_product_vec(xv[o / kRows][u], wv[o % kRows][u], acc);
      }
      s_acc[group][group_warp][o] = warp::reduce_sum(acc);
    }
    PDLTriggerSecondary<kUsePDL>();
    __syncthreads();

    // Lane o of a group sums the group's output o over its warps, in order.
    if (lane < kOutputs) {
      float acc[kGroupWarps];
#pragma unroll
      for (uint32_t i = 0; i < kGroupWarps; ++i) {
        acc[i] = s_acc[group][i][lane];
      }
#pragma unroll
      for (uint32_t i = 1; i < kGroupWarps; ++i) {
        acc[0] += acc[i];
      }
      store(lane, acc[0]);
    }
  }
}

template <
    uint32_t N,
    uint32_t K,
    typename Weight,
    typename Out,
    uint32_t kNTile,
    uint32_t kMaxMTile,
    uint32_t kRowsPerLane,
    bool kUsePDL>
struct TinySIMTKernel {
  using Trait = SIMTTrait<N, K, Weight, kNTile, kMaxMTile, kRowsPerLane>;
  using KernelFn = void (*)(TinyGEMMParams);

  template <std::size_t... I>
  static constexpr auto make_table(std::index_sequence<I...>) {
    return std::array<KernelFn, kMaxMTile>{tiny_simt_kernel<Trait, I + 1, Out, kUsePDL>...};
  }
  // One kernel per rows-per-CTA count up to kMaxMTile.
  static constexpr auto kTable = make_table(std::make_index_sequence<kMaxMTile>{});

  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView w, const tvm::ffi::TensorView out) {
    using namespace host;
    DLDevice device;
    const auto params = tiny_gemm::make_params<N, K, Weight, Out>(x, w, out, Trait::kXAlign, Trait::kWAlign, device);
    const uint32_t m = params.num_tokens;
    if (m == 0) return;
    const uint32_t grid_y = div_ceil(m, kMaxMTile);
    const uint32_t m_tile = div_ceil(m, grid_y);
    LaunchKernel(dim3(Trait::kNumBlocks, grid_y), Trait::kBlockSize, device)
        .enable_pdl(kUsePDL)(kTable[m_tile - 1], params);
  }
};

}  // namespace sglang
