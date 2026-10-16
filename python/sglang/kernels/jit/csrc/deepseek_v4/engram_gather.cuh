// DeepSeek-V4.1 Engram table gather across the ranks' shards.
//
// The table is sharded over the TP ranks on hash-column boundaries, so an id's
// owner follows from its column alone: column c of a token lives in shard
// c / (kCols / world). Shards are peer-visible (symmetric memory, or one host
// mapping as a world of one) and read in place; no collective is involved.
#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/stl.h>

#include <cstdint>
#include <ios>
#include <vector>

namespace sglang {

namespace device::engram {

inline constexpr uint32_t kMaxWorld = 8;

// A table row is one MXFP8 head vector: 256 e4m3 bytes, 8 e8m0 block scales.
inline constexpr uint32_t kHeadDim = 256;
inline constexpr uint32_t kScalesPerRow = kHeadDim / 32;
inline constexpr uint32_t kCols = 24;            // hash columns per token (3 n-grams x 8 heads)
inline constexpr uint32_t K = kCols * kHeadDim;  // 6144 values per token

using vec_t = device::AlignedVector<bf16x2_t, 4>;
using table_vec_t = device::AlignedVector<uint64_t, 2>;

/// \brief An e8m0 scale as fp32: 2^(e - 127), exactly (0 is the subnormal 2^-127).
SGL_DEVICE float e8m0_to_float(uint8_t e) {
  return __uint_as_float(e ? (static_cast<uint32_t>(e) << 23) : 0x00400000u);
}

}  // namespace device::engram

namespace engram_gather {

using namespace device::engram;

/// \brief Each warp moves kRowsPerWarp table rows of one token, 16 bytes a lane.
template <uint32_t kWarpsPerCTA_, uint32_t kRowsPerWarp_>
struct GatherShape {
  static constexpr uint32_t kBytesPerLane = sizeof(table_vec_t);
  static constexpr uint32_t kLanesPerRow = kHeadDim / kBytesPerLane;
  static constexpr uint32_t kRowsPerSweep = device::kWarpThreads / kLanesPerRow;
  static constexpr uint32_t kRowsPerWarp = kRowsPerWarp_;
  static constexpr uint32_t kNumSweeps = kRowsPerWarp / kRowsPerSweep;
  static constexpr uint32_t kWarpsPerToken = kCols / kRowsPerWarp;
  static_assert(kRowsPerWarp % kRowsPerSweep == 0 && kCols % kRowsPerWarp == 0);
  static constexpr uint32_t kWarpsPerCTA = kWarpsPerCTA_;
  static constexpr uint32_t kCTASize = kWarpsPerCTA * device::kWarpThreads;
};

// Measured on H200: the 4-row warp never loses, and wins 7-14% at 1K-16K rows.
using GatherShapeBF16 = GatherShape<1, 4>;

template <uint32_t kWorldSize>
struct GatherBf16Params {
  const int32_t* __restrict__ ids;  // [num_tokens, kCols], row stride ids_stride
  bf16_t* __restrict__ out;         // [num_tokens, K]
  // Shard r's bases, pre-biased by its first global row so a global id indexes them.
  const uint8_t* w[kWorldSize];
  const uint8_t* s[kWorldSize];
  int64_t ids_stride;
  uint32_t num_tokens;
};

template <uint32_t kWorldSize, typename Shape>
__global__ __launch_bounds__(Shape::kCTASize)  //
    void gather_bf16_kernel(const __grid_constant__ GatherBf16Params<kWorldSize> params) {
  using namespace device;
  constexpr uint32_t kColsPerRank = kCols / kWorldSize;
  constexpr uint32_t kLanesPerRow = Shape::kLanesPerRow;
  constexpr uint32_t kBytesPerLane = Shape::kBytesPerLane;
  static_assert(kCols % kWorldSize == 0, "the shards cut on hash-column boundaries");
  const auto tx = threadIdx.x;
  const auto lane_id = tx % kWarpThreads;
  const auto global_warp_id = blockIdx.x * Shape::kWarpsPerCTA + tx / kWarpThreads;
  const auto m = global_warp_id / Shape::kWarpsPerToken;
  const auto col = (global_warp_id % Shape::kWarpsPerToken) * Shape::kRowsPerWarp;
  const auto byte_of_row = (lane_id % kLanesPerRow) * kBytesPerLane;
  const auto row_lane_id = lane_id / kLanesPerRow;
  if (m >= params.num_tokens) return;
  uint32_t my_id = 0;
  if (lane_id < Shape::kRowsPerWarp) {
    my_id = params.ids[static_cast<int64_t>(m) * params.ids_stride + col + lane_id];
  }
  const auto out_row = params.out + static_cast<uint64_t>(m) * K + col * kHeadDim;
  table_vec_t payload[Shape::kNumSweeps];
  uint8_t scale_e8m0[Shape::kNumSweeps];
  const auto load = [&](const uint32_t i) {
    const uint32_t row = i * Shape::kRowsPerSweep + row_lane_id;
    const uint32_t id = __shfl_sync(0xffffffffu, my_id, row);
    const auto owner = (col + row) / kColsPerRank;
    payload[i].load(params.w[owner] + static_cast<uint64_t>(id) * kHeadDim + byte_of_row);
    // A lane's 16 bytes sit inside one 32-value scale block.
    scale_e8m0[i] = params.s[owner][static_cast<uint64_t>(id) * kScalesPerRow + (byte_of_row >> 5)];
  };
  const auto store = [&](const uint32_t i) {
    const uint32_t row = i * Shape::kRowsPerSweep + row_lane_id;
    const float scale = e8m0_to_float(scale_e8m0[i]);
    const auto bytes = reinterpret_cast<const fp8x2_e4m3_t*>(&payload[i]);
    vec_t lo, hi;
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
      const auto a = cast<fp32x2_t>(bytes[e]);
      const auto b = cast<fp32x2_t>(bytes[e + 4]);
      lo[e] = cast<bf16x2_t>(fp32x2_t{a.x * scale, a.y * scale});
      hi[e] = cast<bf16x2_t>(fp32x2_t{b.x * scale, b.y * scale});
    }
    lo.store(out_row + row * kHeadDim + byte_of_row);
    hi.store(out_row + row * kHeadDim + byte_of_row + 8);
  };
  // Issue every row's load before the first store: a load cannot move above a store
  // that may alias it, and a host-memory read takes microseconds.
#pragma unroll
  for (uint32_t i = 0; i < Shape::kNumSweeps; ++i)
    load(i);
#pragma unroll
  for (uint32_t i = 0; i < Shape::kNumSweeps; ++i)
    store(i);
}

}  // namespace engram_gather

/**
 * \brief Gather every token's kCols table rows, dequantized to bf16, from the
 * ranks' column shards; exact, since e8m0 scales are powers of two.
 *
 * \tparam kWorldSize Number of shards; must divide kCols.
 */
template <uint32_t kWorldSize>
struct EngramGather {
  static_assert(kWorldSize >= 1 && kWorldSize <= device::engram::kMaxWorld);

  /**
   * \param ids          [M, kCols] int32 global row ids, unit column stride
   * \param out          [M, K] bf16
   * \param w_ptrs       Shard r's fp8 rows base, 32-byte aligned
   * \param s_ptrs       Shard r's e8m0 scales base, 8-byte aligned
   * \param shard_starts Shard r's first global row; shard 0 starts at 0
   */
  static void gather_bf16(
      const tvm::ffi::TensorView ids,
      const tvm::ffi::TensorView out,
      const std::vector<int64_t> w_ptrs,
      const std::vector<int64_t> s_ptrs,
      const std::vector<int64_t> shard_starts) {
    using namespace host;
    using namespace engram_gather;

    auto M = SymbolicSize{"num_tokens"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({M, static_cast<int64_t>(kCols)})  //
        .with_strides({-1, 1})
        .with_dtype<int32_t>()
        .with_device(dev)
        .verify(ids);
    TensorMatcher({M, static_cast<int64_t>(K)})  //
        .with_dtype<bf16_t>()
        .with_device(dev)
        .ensure_alignment(16)
        .verify(out);
    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    CHECK_HOST(w_ptrs.size() == kWorldSize && s_ptrs.size() == kWorldSize && shard_starts.size() == kWorldSize);
    CHECK_HOST(shard_starts[0] == 0);
    for (uint32_t r = 0; r < kWorldSize; ++r) {
      CHECK_HOST(w_ptrs[r] != 0 && w_ptrs[r] % 32 == 0 && s_ptrs[r] != 0 && s_ptrs[r] % 8 == 0)
          << "shard " << r << " " << std::hex << w_ptrs[r] << " " << s_ptrs[r] << std::dec;
    }
    if (num_tokens == 0) return;
    GatherBf16Params<kWorldSize> params{
        .ids = static_cast<const int32_t*>(ids.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .ids_stride = ids.stride(0),
        .num_tokens = num_tokens,
    };
    // The bias is a whole number of rows, so the 32B / 8B alignment survives.
    for (uint32_t r = 0; r < kWorldSize; ++r) {
      params.w[r] = reinterpret_cast<const uint8_t*>(w_ptrs[r]) - shard_starts[r] * kHeadDim;
      params.s[r] = reinterpret_cast<const uint8_t*>(s_ptrs[r]) - shard_starts[r] * kScalesPerRow;
    }
    using Shape = GatherShapeBF16;
    const auto num_warps = num_tokens * Shape::kWarpsPerToken;
    LaunchKernel(div_ceil(num_warps, Shape::kWarpsPerCTA), Shape::kCTASize, dev.unwrap())(
        gather_bf16_kernel<kWorldSize, Shape>, params);
  }
};

}  // namespace sglang
