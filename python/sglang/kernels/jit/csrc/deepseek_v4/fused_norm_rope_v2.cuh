#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/compress_v2.cuh>
#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using PlanC = device::compress::CompressPlan;
using PlanD = device::compress::DecodePlan;
using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

SGL_DEVICE uint8_t quant_fp4_e2m1(float x) {
  const float ax = fminf(fabsf(x), 6.0f);
  uint8_t idx = 0;
  idx += ax > 0.25f;
  idx += ax > 0.75f;
  idx += ax > 1.25f;
  idx += ax > 1.75f;
  idx += ax > 2.5f;
  idx += ax > 3.5f;
  idx += ax > 5.0f;
  if (x < 0.0f && idx != 0) idx |= 0x8;
  return idx;
}

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
// FlashMLA ILP: a block processes this many tokens back-to-back. Resolving all
// K plans, then issuing all K input loads before any is consumed, keeps many
// independent global loads in flight to hide the ~hundreds-of-cycles load
// latency the 1-token baseline stalls on (long_scoreboard). The per-token
// reduction tree and store (fp8 quant OR bf16) are unchanged, so output is
// bit-identical to the 1-token kernel. Small num_tokens is grid-starved at K=4
// (too few blocks for the SMs), so the launcher drops to K=1 below the cutoff.
constexpr uint32_t kFlashmlaTokensPerBlock = 4;
constexpr uint32_t kFlashmlaSmallNTokensPerBlock = 1;
constexpr uint32_t kFlashmlaSmallNCutoff = 2048;

struct FusedNormRopeStoreParams {
  void* __restrict__ input;
  const void* __restrict__ handle;  // plan decode / compress
  const void* __restrict__ weight;
  const float* __restrict__ freqs_cis;
  const int64_t* __restrict__ out_loc;
  uint8_t* __restrict__ kvcache;
  float eps;
  uint32_t compress_ratio;
  uint32_t num_tokens;
};

enum class ForwardMode : bool {
  CompressExtend = 0,
  CompressDecode = 1,
};

#define INDEXER_KERNEL __global__ __launch_bounds__(kBlockSize, 8)
#define FLASHMLA_KERNEL __global__ __launch_bounds__(kBlockSize, 8)

// ----------------------------------------------------------------------------
// Indexer variant: kHeadDim = 128, 1 token per *warp* (8 tokens per block).
// Each warp's 32 lanes cover the full 128-elem head_dim (kVecSize = 4 each).
// Cache layout: 132 bytes/token (128 fp8 nope + 4 fp32 scale).
// ----------------------------------------------------------------------------
template <typename DType, ForwardMode kMode, int32_t kPageBits, bool kUsePDL, int32_t kPreshuffleSize = 0>
INDEXER_KERNEL void fused_norm_rope_indexer(const __grid_constant__ FusedNormRopeStoreParams params) {
  using namespace device;
  using enum ForwardMode;

  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  constexpr int64_t kPageBytes = 132ll << kPageBits;
  static_assert(kHeadDim == kWarpThreads * kVecSize);
  static_assert(kRopeDim == kWarpThreads * 2);
  static_assert(kRopeSize <= kWarpThreads);
  using Storage = AlignedVector<DType, kVecSize>;
  using Float4 = AlignedVector<float, kVecSize>;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto work_id = blockIdx.x * kNumWarps + warp_id;
  // Lanes whose 4-elem pack lies in the rope tail (= last `kRopeSize` packs).
  const bool is_rope_lane = lane_id >= kWarpThreads - kRopeSize;

  if (work_id >= params.num_tokens) return;

  const auto input = static_cast<DType*>(params.input) + work_id * kHeadDim;
  int32_t position;
  int64_t out_loc;
  if constexpr (kMode == CompressExtend) {
    const auto plan = static_cast<const PlanC*>(params.handle)[work_id];
    if (plan.is_invalid()) return;
    position = plan.seq_len - params.compress_ratio;
    out_loc = params.out_loc[plan.ragged_id];
  } else if constexpr (kMode == CompressDecode) {
    const auto plan = static_cast<const PlanD*>(params.handle)[work_id];
    if (plan.seq_len % params.compress_ratio != 0) return;
    position = plan.seq_len - params.compress_ratio;
    out_loc = params.out_loc[work_id];
  } else {
    static_assert(host::dependent_false_v<DType>, "Unsupported Mode");
  }
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  Float4 data, freq;

  // part 1: norm
  {
    Storage input_vec, weight_vec;
    input_vec.load(input, lane_id);
    weight_vec.load(params.weight, lane_id);
    if (is_rope_lane) freq.load(freqs_cis, lane_id - (kWarpThreads - kRopeSize));

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_input = cast<float>(input_vec[i]);
      sum_of_squares += fp32_input * fp32_input;
    }

    sum_of_squares = warp::reduce_sum(sum_of_squares);
    const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_input = cast<float>(input_vec[i]);
      const auto fp32_weight = cast<float>(weight_vec[i]);
      data[i] = fp32_input * norm_factor * fp32_weight;
    }
  }

  // part 2: rope (rope-lane only, 4 elems per lane = 2 (real, imag) pairs)
  if (is_rope_lane) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto y_real = data[2];
    const auto y_imag = data[3];
    const auto freq_x_real = freq[0];
    const auto freq_x_imag = freq[1];
    const auto freq_y_real = freq[2];
    const auto freq_y_imag = freq[3];
    data[0] = x_real * freq_x_real - x_imag * freq_x_imag;
    data[1] = x_real * freq_x_imag + x_imag * freq_x_real;
    data[2] = y_real * freq_y_real - y_imag * freq_y_imag;
    data[3] = y_real * freq_y_imag + y_imag * freq_y_real;
  }

  // part 3: hadamard transform
  {
    // Stage 1: butterfly (data[0], data[1]) and (data[2], data[3]).
    {
      const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a1;
      data[1] = a0 - a1;
      data[2] = a2 + a3;
      data[3] = a2 - a3;
    }
    // Stage 2: butterfly (data[0], data[2]) and (data[1], data[3]).
    {
      const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a2;
      data[1] = a1 + a3;
      data[2] = a0 - a2;
      data[3] = a1 - a3;
    }
    // Stages 3..7: cross-lane butterflies. Lower-lane (mask bit clear) keeps
    // the sum, upper-lane (mask bit set) keeps the difference. shfl_xor is
    // unsynchronized across early-returned lanes, but invalid-plan returns
    // happen above for *all* lanes of a warp (work_id is warp-uniform), so
    // the warp is intact here.
#pragma unroll
    for (uint32_t mask = 1; mask < kWarpThreads; mask <<= 1) {
#pragma unroll
      for (int i = 0; i < kVecSize; ++i) {
#ifndef USE_ROCM
        const float other = __shfl_xor_sync(kFullMask, data[i], mask, kWarpThreads);
#else
        const float other = __shfl_xor(data[i], mask, kWarpThreads);
#endif
        data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
      }
    }
    const float kHadamardScale = math::rsqrt(static_cast<float>(kHeadDim));
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      data[i] *= kHadamardScale;
  }

  // part 4: per-warp UE8M0 quant + store. The whole warp emits one fp8 group
  // (= 128 elements) plus a single fp32 scale, matching the indexer cache
  // layout (`fused_store_indexer_cache`).
  {
    using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;
    float local_max = math::abs(data[0]);
#pragma unroll
    for (int i = 1; i < kVecSize; ++i) {
      local_max = math::max(local_max, math::abs(data[i]));
    }
    const auto abs_max = warp::reduce_max(local_max);
    const auto scale = fmaxf(1e-4f, abs_max) / kFP8E4M3Max;
    const auto inv_scale = 1.0f / scale;
    const int64_t page = out_loc >> kPageBits;
    const int64_t offset = out_loc & ((1 << kPageBits) - 1);
    const auto page_ptr = params.kvcache + page * kPageBytes;
    const auto value_ptr = page_ptr + offset * 128;
    const auto scale_ptr = page_ptr + (128 << kPageBits) + offset * 4;
    OutStorage result;
    result[0] = pack_fp8(data[0] * inv_scale, data[1] * inv_scale);
    result[1] = pack_fp8(data[2] * inv_scale, data[3] * inv_scale);
    PDLTriggerSecondary<kUsePDL>();
    if constexpr (kPreshuffleSize != 0) {
      constexpr int32_t kTile = kPreshuffleSize;
      const int32_t dim_base = lane_id * kVecSize;
      const int32_t token_tile_id = offset / kTile;
      const int32_t token_in_tile = offset % kTile;
      const int32_t col_tile_id = dim_base / kTile;
      const int32_t col_in_tile = dim_base % kTile;
      const int32_t value_offset = token_tile_id * (kTile * static_cast<int32_t>(kHeadDim)) +
                                   col_tile_id * (kTile * kTile) + token_in_tile * kTile + col_in_tile;
      result.store(page_ptr + value_offset, 0);
    } else {
      result.store(value_ptr, lane_id);
    }
    // The single fp32 scale is identical across all lanes -- write from any lane.
    if (lane_id == 0) reinterpret_cast<float*>(scale_ptr)[0] = scale;
  }
}

template <typename DType, ForwardMode kMode, int32_t kPageBits, bool kUsePDL>
INDEXER_KERNEL void fused_norm_rope_indexer_fp4(const __grid_constant__ FusedNormRopeStoreParams params) {
  using namespace device;
  using enum ForwardMode;

  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  constexpr int64_t kPageBytes = 68ll << kPageBits;
  static_assert(kHeadDim == kWarpThreads * kVecSize);
  static_assert(kRopeDim == kWarpThreads * 2);
  static_assert(kRopeSize <= kWarpThreads);
  using Storage = AlignedVector<DType, kVecSize>;
  using Float4 = AlignedVector<float, kVecSize>;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto work_id = blockIdx.x * kNumWarps + warp_id;
  const bool is_rope_lane = lane_id >= kWarpThreads - kRopeSize;

  if (work_id >= params.num_tokens) return;

  const auto input = static_cast<DType*>(params.input) + work_id * kHeadDim;
  int32_t position;
  int64_t out_loc;
  if constexpr (kMode == CompressExtend) {
    const auto plan = static_cast<const PlanC*>(params.handle)[work_id];
    if (plan.is_invalid()) return;
    position = plan.seq_len - params.compress_ratio;
    out_loc = params.out_loc[plan.ragged_id];
  } else if constexpr (kMode == CompressDecode) {
    const auto plan = static_cast<const PlanD*>(params.handle)[work_id];
    if (plan.seq_len % params.compress_ratio != 0) return;
    position = plan.seq_len - params.compress_ratio;
    out_loc = params.out_loc[work_id];
  } else {
    static_assert(host::dependent_false_v<DType>, "Unsupported Mode");
  }
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  Float4 data, freq;

  {
    Storage input_vec, weight_vec;
    input_vec.load(input, lane_id);
    weight_vec.load(params.weight, lane_id);
    if (is_rope_lane) freq.load(freqs_cis, lane_id - (kWarpThreads - kRopeSize));

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_input = cast<float>(input_vec[i]);
      sum_of_squares += fp32_input * fp32_input;
    }

    sum_of_squares = warp::reduce_sum(sum_of_squares);
    const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_input = cast<float>(input_vec[i]);
      const auto fp32_weight = cast<float>(weight_vec[i]);
      data[i] = fp32_input * norm_factor * fp32_weight;
    }
  }

  if (is_rope_lane) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto y_real = data[2];
    const auto y_imag = data[3];
    const auto freq_x_real = freq[0];
    const auto freq_x_imag = freq[1];
    const auto freq_y_real = freq[2];
    const auto freq_y_imag = freq[3];
    data[0] = x_real * freq_x_real - x_imag * freq_x_imag;
    data[1] = x_real * freq_x_imag + x_imag * freq_x_real;
    data[2] = y_real * freq_y_real - y_imag * freq_y_imag;
    data[3] = y_real * freq_y_imag + y_imag * freq_y_real;
  }

  {
    {
      const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a1;
      data[1] = a0 - a1;
      data[2] = a2 + a3;
      data[3] = a2 - a3;
    }
    {
      const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a2;
      data[1] = a1 + a3;
      data[2] = a0 - a2;
      data[3] = a1 - a3;
    }
#pragma unroll
    for (uint32_t mask = 1; mask < kWarpThreads; mask <<= 1) {
#pragma unroll
      for (int i = 0; i < kVecSize; ++i) {
#ifndef USE_ROCM
        const float other = __shfl_xor_sync(kFullMask, data[i], mask, kWarpThreads);
#else
        const float other = __shfl_xor(data[i], mask, kWarpThreads);
#endif
        data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
      }
    }
    const float kHadamardScale = math::rsqrt(static_cast<float>(kHeadDim));
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      data[i] *= kHadamardScale;
  }

  {
    float local_max = math::abs(data[0]);
#pragma unroll
    for (int i = 1; i < kVecSize; ++i) {
      local_max = math::max(local_max, math::abs(data[i]));
    }
    local_max = warp::reduce_max<8>(local_max);

    const auto scale_raw = fmaxf(1e-4f, local_max) / 6.0f;
    const auto scale_ue8m0 = static_cast<uint8_t>(cast_to_ue8m0(scale_raw));
    const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);

    const uint8_t packed0 = quant_fp4_e2m1(data[0] * inv_scale) | (quant_fp4_e2m1(data[1] * inv_scale) << 4);
    const uint8_t packed1 = quant_fp4_e2m1(data[2] * inv_scale) | (quant_fp4_e2m1(data[3] * inv_scale) << 4);
    const uint16_t packed = static_cast<uint16_t>(packed0) | (static_cast<uint16_t>(packed1) << 8);

    const int64_t page = out_loc >> kPageBits;
    const int64_t offset = out_loc & ((1 << kPageBits) - 1);
    const auto page_ptr = params.kvcache + page * kPageBytes;
    const auto value_ptr = page_ptr + offset * 64;
    const auto scale_ptr = page_ptr + (64 << kPageBits) + offset * 4;

    PDLTriggerSecondary<kUsePDL>();
    reinterpret_cast<uint16_t*>(value_ptr)[lane_id] = packed;
    if ((lane_id & 7) == 0) static_cast<uint8_t*>(scale_ptr)[lane_id >> 3] = scale_ue8m0;
  }
}

// ----------------------------------------------------------------------------
// FlashMLA variant: kHeadDim = 512, kTokensPerBlock tokens per *block* (256
// threads; each thread owns kVecSize=2 elems -> 256 threads cover one token's
// 512 dims, and the block loops over its tokens).
// Cache layout: 584 bytes/token = 448 fp8 nope + 64 (=32 bf16x2) rope + 8 scale.
//
// ILP: resolve all K plans, then issue all K input loads before consuming any,
// so the K independent global loads stay in flight together and hide the load
// latency the 1-token baseline stalls on. The per-token reduction tree and
// store (fp8 quant OR bf16) are byte-for-byte the 1-token path, so output is
// bit-identical to the original kernel.
// ----------------------------------------------------------------------------
template <
    typename DType,
    ForwardMode kMode,
    int32_t kPageBits,
    bool kUsePDL,
    bool kBf16Store = false,
    uint32_t kTokensPerBlockT = kFlashmlaTokensPerBlock>
FLASHMLA_KERNEL void fused_norm_rope_flashmla(const __grid_constant__ FusedNormRopeStoreParams params) {
  using namespace device;
  using enum ForwardMode;

  constexpr int64_t kHeadDim = 512;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 2;
  // Last warp owns the rope tail. The remaining 7 warps each emit one
  // 64-element fp8 group (own UE8M0 scale).
  constexpr uint32_t kRopeWarp = kNumWarps - 1;
  constexpr uint32_t kTokensPerBlock = kTokensPerBlockT;
  // kBf16Store: write the whole head_dim as plain BF16 (no fp8 / no scale) into a
  // [num_slots, head_dim] bf16 cache (page_size==1) at row out_loc
  constexpr int64_t kPageBytes =
      kBf16Store ? ((kHeadDim * 2ll) << kPageBits) : host::div_ceil(584ll << kPageBits, 576) * 576;
  static_assert(kHeadDim == kBlockSize * kVecSize);
  static_assert(kRopeDim == kWarpThreads * kVecSize);
  static_assert(kHeadDim - kRopeDim == kRopeWarp * kWarpThreads * kVecSize);
  using Storage = AlignedVector<DType, kVecSize>;
  using Float2 = AlignedVector<float, kVecSize>;
  // One 2-bf16 input pack == one 32-bit word; used for the streaming __ldcs load.
  using LoadWord = typename details::sized_int<Storage>;

  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;
  const auto work_base = blockIdx.x * kTokensPerBlock;

  // Per-token state. valid[t] is block-uniform (depends only on blockIdx.x and
  // the plan), so branching on it around __syncthreads and the warp-wide fp8
  // reduce_max never splits a warp.
  bool valid[kTokensPerBlock];
  int32_t position_arr[kTokensPerBlock];
  int64_t out_loc_arr[kTokensPerBlock];
  Storage input_vec[kTokensPerBlock];
  Float2 freq[kTokensPerBlock];
  Storage weight_vec;  // shared by every token -> loaded once

  __shared__ float partial_sums[kTokensPerBlock][kNumWarps];

  PDLWaitPrimary<kUsePDL>();
  weight_vec.load(params.weight, tx);

  // Stage A: resolve every token's plan first (K independent 16B plan loads in
  // flight) and stash position / out_loc; do NOT yet touch input / freqs.
#pragma unroll
  for (uint32_t t = 0; t < kTokensPerBlock; ++t) {
    const auto work_id = work_base + t;
    bool ok = (work_id < params.num_tokens);
    int32_t position = 0;
    int64_t out_loc = 0;
    if (ok) {
      if constexpr (kMode == CompressExtend) {
        const auto plan = static_cast<const PlanC*>(params.handle)[work_id];
        if (plan.is_invalid()) {
          ok = false;
        } else {
          position = plan.seq_len - params.compress_ratio;
          out_loc = params.out_loc[plan.ragged_id];
        }
      } else if constexpr (kMode == CompressDecode) {
        const auto plan = static_cast<const PlanD*>(params.handle)[work_id];
        if (plan.seq_len % params.compress_ratio != 0) {
          ok = false;
        } else {
          position = plan.seq_len - params.compress_ratio;
          out_loc = params.out_loc[work_id];
        }
      } else {
        static_assert(host::dependent_false_v<DType>, "Unsupported Mode");
      }
    }
    valid[t] = ok;
    position_arr[t] = position;
    out_loc_arr[t] = out_loc;
  }

  // Stage B: issue all input (+ freqs) loads back-to-back. Addresses are
  // resolved, so the K loads have no dependency and stay in flight together.
  // Input is streamed (read once) via __ldcs (evict-first, read-only path) so
  // it doesn't evict the reused weight/freqs from L1 -- this is the second half
  // of the speedup (drives long_scoreboard down further than K-ILP alone).
#pragma unroll
  for (uint32_t t = 0; t < kTokensPerBlock; ++t) {
    if (!valid[t]) continue;
    const auto work_id = work_base + t;
    const auto input = static_cast<DType*>(params.input) + work_id * kHeadDim;
    const auto word = __ldcs(reinterpret_cast<const LoadWord*>(input) + tx);
    *reinterpret_cast<LoadWord*>(&input_vec[t]) = word;
    if (warp_id == kRopeWarp) freq[t].load(params.freqs_cis + position_arr[t] * kRopeDim, lane_id);
  }

  // part 1: norm -- per-token sum of squares, warp reduce, write partial.
#pragma unroll
  for (uint32_t t = 0; t < kTokensPerBlock; ++t) {
    if (!valid[t]) continue;
    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_input = cast<float>(input_vec[t][i]);
      sum_of_squares += fp32_input * fp32_input;
    }
    const auto warp_sum = warp::reduce_sum(sum_of_squares);
    if (lane_id == 0) partial_sums[t][warp_id] = warp_sum;
  }
  __syncthreads();

  PDLTriggerSecondary<kUsePDL>();

  // part 2: per token -- cross-warp reduce -> normalize -> rope + store.
#pragma unroll
  for (uint32_t t = 0; t < kTokensPerBlock; ++t) {
    if (!valid[t]) continue;
    // Replicate the per-warp partial sums to a full warp and reduce. Every
    // lane-group of `kNumWarps` lanes ends up with the global sum.
    const auto sum_of_squares = warp::reduce_sum<kNumWarps>(partial_sums[t][lane_id % kNumWarps]);
    const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);

    Float2 data;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_input = cast<float>(input_vec[t][i]);
      const auto fp32_weight = cast<float>(weight_vec[i]);
      data[i] = fp32_input * norm_factor * fp32_weight;
    }

    const int64_t out_loc = out_loc_arr[t];
    const int64_t page = out_loc >> kPageBits;
    const int64_t offset = out_loc & ((1 << kPageBits) - 1);
    const auto page_ptr = params.kvcache + page * kPageBytes;
    const auto value_ptr = page_ptr + offset * (kBf16Store ? (kHeadDim * 2) : 576);

    if constexpr (kBf16Store) {
      Float2 d = data;
      if (warp_id == kRopeWarp) {
        const auto x_real = data[0];
        const auto x_imag = data[1];
        const auto freq_real = freq[t][0];
        const auto freq_imag = freq[t][1];
        // Explicit fma pins the fp-contraction so the unrolled K-loop compiles
        // to the same rounding as the 1-token baseline (nvcc fuses a*b-c*d into
        // fma(a,b,-(c*d)); pin it so unrolling can't pick a different form).
        d[0] = __fmaf_rn(x_real, freq_real, -(x_imag * freq_imag));
        d[1] = __fmaf_rn(x_real, freq_imag, x_imag * freq_real);
      }
      reinterpret_cast<bf16x2_t*>(value_ptr)[tx] = cast<bf16x2_t>(fp32x2_t{d[0], d[1]});
    } else if (warp_id == kRopeWarp) {
      // Each rope-warp lane owns exactly one (real, imag) pair within the rope
      // tail. Apply rotation, downcast to BF16, write to the slot's rope region.
      const auto x_real = data[0];
      const auto x_imag = data[1];
      const auto freq_real = freq[t][0];
      const auto freq_imag = freq[t][1];
      data[0] = __fmaf_rn(x_real, freq_real, -(x_imag * freq_imag));
      data[1] = __fmaf_rn(x_real, freq_imag, x_imag * freq_real);
      const auto result = cast<bf16x2_t>(fp32x2_t{data[0], data[1]});
      const auto rope_ptr = value_ptr + 448;
      reinterpret_cast<bf16x2_t*>(rope_ptr)[lane_id] = result;
    } else {
      // Non-rope warp: per-warp UE8M0 group (64 elems -> 64 fp8 + 1 scale byte).
      // BF16 round-trip to match the precision of the non-fused path
      // (which goes through quant_to_nope_fp8_rope_bf16_pack_triton with bf16 input).
      const auto x = cast<float>(cast<bf16_t>(data[0]));
      const auto y = cast<float>(cast<bf16_t>(data[1]));
      const auto abs_max = warp::reduce_max(fmaxf(fabs(x), fabs(y)));
      const auto scale_raw = fmaxf(1e-4f, abs_max) / kFP8E4M3Max;
      const auto scale_ue8m0 = cast_to_ue8m0(scale_raw);
      const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);
      const auto result = pack_fp8(x * inv_scale, y * inv_scale);
      const auto scale_ptr = page_ptr + (576 << kPageBits) + offset * 8;
      reinterpret_cast<fp8x2_e4m3_t*>(value_ptr)[tx] = result;
      // All lanes in this warp produce the same scale byte; let lane 0 publish.
      if (lane_id == 0) static_cast<uint8_t*>(scale_ptr)[warp_id] = scale_ue8m0;
    }
  }
}

template <
    typename DType,
    int64_t kHeadDim,
    int64_t kRopeDim,
    uint32_t kPageSize,
    bool kUsePDL,
    int32_t kPreshuffleSize = 0,
    bool kBf16Store = false>
struct FusedNormRopeKernel {
  static constexpr int32_t kLogPageSize = std::countr_zero(kPageSize);
  static constexpr bool kIsIndexer = (kHeadDim == 128);
  static_assert(!(kIsIndexer && kBf16Store), "bf16 store only for flashmla head_dim=512");
  static constexpr int64_t kIndexerBytes = 132 * kPageSize;
  static constexpr int64_t kFlashMLABytes = host::div_ceil(584 * kPageSize, 576) * 576;
  static constexpr int64_t kBf16Bytes = kHeadDim * 2 * kPageSize;  // plain bf16 cache
  static constexpr int64_t kPageBytes = kBf16Store ? kBf16Bytes : (kIsIndexer ? kIndexerBytes : kFlashMLABytes);

  /// TODO: Let's fix the config for now.
  static_assert(kRopeDim == 64 && (kHeadDim == 128 || kHeadDim == 512));
  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");

  template <ForwardMode kMode, uint32_t kTPW = kFlashmlaTokensPerBlock>
  static constexpr auto select_kernel() {
    if constexpr (kIsIndexer) {
      return fused_norm_rope_indexer<DType, kMode, kLogPageSize, kUsePDL, kPreshuffleSize>;
    } else {
      return fused_norm_rope_flashmla<DType, kMode, kLogPageSize, kUsePDL, kBf16Store, kTPW>;
    }
  }

  template <ForwardMode kMode>
  static constexpr auto select_fp4_kernel() {
    static_assert(kIsIndexer, "FP4 fused store is only defined for the indexer");
    return fused_norm_rope_indexer_fp4<DType, kMode, kLogPageSize, kUsePDL>;
  }

  static void forward(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView plan,
      const tvm::ffi::TensorView weight,
      const float eps,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView kvcache,
      const bool is_decode,
      const uint32_t compress_ratio) {
    using namespace host;
    using enum ForwardMode;

    const auto mode = static_cast<ForwardMode>(is_decode);

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    TensorMatcher({N, kHeadDim})  // input
        .with_dtype<DType>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({kHeadDim})  // weight
        .with_dtype<DType>()
        .with_device(device_)
        .verify(weight);
    TensorMatcher({-1, kRopeDim})  // freqs_cis
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs_cis);
    TensorMatcher({-1})  // out_loc
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(out_loc);
    TensorMatcher({-1, -1})  // cache
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(kvcache);

    switch (mode) {
      case CompressExtend:
        compress::verify_plan_c(plan, N, device_);
        RuntimeCheck(out_loc.size(0) >= N.unwrap());
        break;
      case CompressDecode:
        compress::verify_plan_d(plan, N, device_);
        RuntimeCheck(out_loc.size(0) == N.unwrap());
        break;
    }

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_tokens == 0) return;
    const auto params = FusedNormRopeStoreParams{
        .input = input.data_ptr(),
        .handle = plan.data_ptr(),
        .weight = weight.data_ptr(),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .out_loc = static_cast<const int64_t*>(out_loc.data_ptr()),
        .kvcache = static_cast<uint8_t*>(kvcache.data_ptr()),
        .eps = eps,
        .compress_ratio = compress_ratio,
        .num_tokens = num_tokens,
    };
    const auto device = device_.unwrap();
    if constexpr (kIsIndexer) {
      // Indexer packs `kNumWarps` tokens per block (warp-major); unchanged.
      const uint32_t num_blocks = div_ceil(num_tokens, kNumWarps);
      const auto kernel = mode == CompressExtend ? select_kernel<CompressExtend>() : select_kernel<CompressDecode>();
      LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(kernel, params);
    } else {
      // FlashMLA: K tokens/block (ILP hides load latency). Small num_tokens is
      // grid-starved at the K=4 large-N default (too few blocks for the SMs),
      // so drop to K=1 there. Per-token math/store are identical across K, so
      // output is bit-identical either way.
      const bool small_n = num_tokens < kFlashmlaSmallNCutoff;
      constexpr uint32_t kBigK = kFlashmlaTokensPerBlock;
      constexpr uint32_t kSmallK = kFlashmlaSmallNTokensPerBlock;
      const uint32_t k = small_n ? kSmallK : kBigK;
      const uint32_t num_blocks = div_ceil(num_tokens, k);
      const auto kernel =
          mode == CompressExtend
              ? (small_n ? select_kernel<CompressExtend, kSmallK>() : select_kernel<CompressExtend, kBigK>())
              : (small_n ? select_kernel<CompressDecode, kSmallK>() : select_kernel<CompressDecode, kBigK>());
      LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(kernel, params);
    }
  }

  static void forward_fp4(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView plan,
      const tvm::ffi::TensorView weight,
      const float eps,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView kvcache,
      const bool is_decode,
      const uint32_t compress_ratio) {
    using namespace host;
    using enum ForwardMode;

    static_assert(kIsIndexer, "FP4 fused store is only defined for the indexer");
    constexpr int64_t kFp4PageBytes = 68 * kPageSize;
    const auto mode = static_cast<ForwardMode>(is_decode);

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({N, kHeadDim}).with_dtype<DType>().with_device(device_).verify(input);
    TensorMatcher({kHeadDim}).with_dtype<DType>().with_device(device_).verify(weight);
    TensorMatcher({-1, kRopeDim}).with_dtype<float>().with_device(device_).verify(freqs_cis);
    TensorMatcher({-1}).with_dtype<int64_t>().with_device(device_).verify(out_loc);
    TensorMatcher({-1, -1}).with_strides({kFp4PageBytes, 1}).with_dtype<uint8_t>().with_device(device_).verify(kvcache);

    switch (mode) {
      case CompressExtend:
        compress::verify_plan_c(plan, N, device_);
        RuntimeCheck(out_loc.size(0) >= N.unwrap());
        break;
      case CompressDecode:
        compress::verify_plan_d(plan, N, device_);
        RuntimeCheck(out_loc.size(0) == N.unwrap());
        break;
    }

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_tokens == 0) return;
    const auto params = FusedNormRopeStoreParams{
        .input = input.data_ptr(),
        .handle = plan.data_ptr(),
        .weight = weight.data_ptr(),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .out_loc = static_cast<const int64_t*>(out_loc.data_ptr()),
        .kvcache = static_cast<uint8_t*>(kvcache.data_ptr()),
        .eps = eps,
        .compress_ratio = compress_ratio,
        .num_tokens = num_tokens,
    };
    const uint32_t num_blocks = div_ceil(num_tokens, kNumWarps);
    const auto device = device_.unwrap();
    const auto kernel =
        mode == CompressExtend ? select_fp4_kernel<CompressExtend>() : select_fp4_kernel<CompressDecode>();
    LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
