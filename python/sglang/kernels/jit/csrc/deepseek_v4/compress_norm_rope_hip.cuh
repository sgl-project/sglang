// HIP C4 / C128 compression fused with its norm + rope + cache store. 512-dim V4 fp8 or bf16 latent: one block per
// plan row (C4) or per plan row and 64-column split (C128), thread `tx` owns elements [2 * tx, 2 * tx + 2).
// 128-dim C4 indexer: one warp per plan row. Same softmax pooling as flash_c{4,128}_*, but the state rows and
// ape are read in fp32 as stored, followed by the fused_norm_rope_{flashmla,indexer} epilogue in fp32 registers.

#include "c128_v2.cuh"
// c4_v2.cuh defines its own WRITE_KERNEL; the C128 kernels above are already declared.
#undef WRITE_KERNEL
#include "c4_v2.cuh"
#include "fused_norm_rope_v2.cuh"
#include <type_traits>

namespace sglang {

struct CompressNormRopeParams {
  void* __restrict__ kv_buffer;
  const void* __restrict__ kv_input;
  const void* __restrict__ score_bias;
  float* __restrict__ scratch;      // C128: [rows, 512]
  uint32_t* __restrict__ arrivals;  // C128: [rows]
  FusedNormRopeStoreParams store;   // handle = plan_c / plan_d, num_tokens = plan rows
};

/// \brief Loads the `idx`-th group of N elements and widens it to fp32.
template <int32_t N, typename Float>
SGL_DEVICE device::AlignedVector<float, N> load_fp32(const Float* ptr, const uint32_t idx) {
  using namespace device;
  AlignedVector<Float, N> data;
  data.load(ptr, idx);
  AlignedVector<float, N> result;
#pragma unroll
  for (int32_t i = 0; i < N; ++i) {
    result[i] = cast<float>(data[i]);
  }
  return result;
}

template <typename InputFloat, typename BufferFloat>
struct C4Window {
  const InputFloat* kv_src;
  const BufferFloat* kv_buf_0;
  const BufferFloat* kv_buf_1;
  bool need_overlap;
  int32_t buffer_len;
};

/// \brief Resolves the row's C4 window; a decode row first hands its state row to `write_state(dst, src)`.
template <typename Trait, ForwardMode kMode, typename InputFloat, typename BufferFloat, typename WriteState>
SGL_DEVICE bool c4_window(
    const void* handle,
    const uint32_t work_id,
    const InputFloat* kv_input,
    BufferFloat* kv_buffer,
    C4Window<InputFloat, BufferFloat>& window,
    const WriteState& write_state) {
  constexpr auto kPage = Trait::kPageElementSize;
  const auto resolve = [&](const auto& plan, const InputFloat* src, const int32_t len) {
    window = {src, kv_buffer + plan.read_page_0 * kPage, kv_buffer + plan.read_page_1 * kPage, plan.seq_len > 4, len};
  };
  if constexpr (kMode == ForwardMode::CompressExtend) {
    const auto plan = static_cast<const PlanC*>(handle)[work_id];
    if (plan.is_invalid()) return false;
    resolve(plan, kv_input + plan.ragged_id * Trait::kElementSize, plan.buffer_len);
  } else {
    const auto plan = static_cast<const PlanD*>(handle)[work_id];
    const auto kv_src = kv_input + work_id * Trait::kElementSize;
    write_state(kv_buffer + plan.write_loc * Trait::kElementSize, kv_src);
    resolve(plan, kv_src, 8);
  }
  return true;
}

/// \brief Softmax pool of the `idx`-th group of N elements over the 8 window positions.
template <int32_t N, typename Trait, typename InputFloat, typename BufferFloat, typename BiasFloat>
SGL_DEVICE device::AlignedVector<float, N>
c4_pool(const C4Window<InputFloat, BufferFloat>& w, const BiasFloat* score_bias, const uint32_t idx) {
  using namespace device;
  AlignedVector<float, N> kv[8], score[8];
  const auto load = [&](const auto* base, const int32_t j) {
    kv[j] = load_fp32<N>(base, idx);
    score[j] = load_fp32<N>(base + Trait::kScoreOffset, idx);
  };
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    if (w.need_overlap && i < w.buffer_len) {
      load(w.kv_buf_0 + i * Trait::kElementSize, i);
    } else if (w.need_overlap) {
      load(w.kv_src + (i - 7) * Trait::kElementSize, i);
    } else {
      kv[i].fill(0.0f);
      score[i].fill(-FLT_MAX);
    }
  }
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    if (i + 4 < w.buffer_len) {
      load(w.kv_buf_1 + i * Trait::kElementSize + Trait::kOverlapOffset, i + 4);
    } else {
      load(w.kv_src + (i - 3) * Trait::kElementSize + Trait::kOverlapOffset, i + 4);
    }
  }
#pragma unroll
  for (int32_t j = 0; j < 8; ++j) {
    const auto bias = load_fp32<N>(score_bias + j * Trait::kHeadDim, idx);
#pragma unroll
    for (int32_t k = 0; k < N; ++k) {
      score[j][k] = score[j][k] + bias[k];
    }
  }

  AlignedVector<float, N> result;
#pragma unroll
  for (int32_t k = 0; k < N; ++k) {
    float max_value = score[0][k];
#pragma unroll
    for (int32_t j = 1; j < 8; ++j) {
      max_value = fmaxf(max_value, score[j][k]);
    }
    float sum_product = 0.0f;
    float sum_exp_value = 0.0f;
#pragma unroll
    for (int32_t j = 0; j < 8; ++j) {
      const auto exp_score = __expf(score[j][k] - max_value);
      sum_product += kv[j][k] * exp_score;
      sum_exp_value += exp_score;
    }
    result[k] = sum_product / sum_exp_value;
  }
  return result;
}

/// \brief c128_forward with fp32 state/ape reads; writes the block's kTileDim pooled values to `out`.
template <typename Trait, bool kUsePDL, typename BufferFloat, typename InputFloat, typename BiasFloat>
SGL_DEVICE void c128_pool(
    const BufferFloat* kv_buf,  // [128n, 128n + 127]
    const InputFloat* kv_src,   // ragged pointer at position = 128n + 127
    float* out,
    const BiasFloat* score_bias,
    const int32_t buffer_len) {
  using namespace device;
  constexpr uint32_t kTileElements = Trait::kTileElements;
  constexpr uint32_t kElementsPerWarp = Trait::kElementsPerWarp;
  constexpr uint32_t kNumWarps = Trait::kNumWarps;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const int32_t warp_offset = warp_id * kElementsPerWarp;
  const auto kv_start = kv_src - 127 * Trait::kElementSize;  // point to start

  AlignedVector<float, kTileElements> kv[kElementsPerWarp], score[kElementsPerWarp];
#pragma unroll
  for (int32_t i = 0; i < kElementsPerWarp; ++i) {
    const int32_t j = i + warp_offset;
    __builtin_assume(j < 128);
    const auto bias = load_fp32<kTileElements>(score_bias + j * Trait::kHeadDim, lane_id);
    const auto load = [&](const auto* src) {
      kv[i] = load_fp32<kTileElements>(src, lane_id);
      score[i] = load_fp32<kTileElements>(src + Trait::kScoreOffset, lane_id);
    };
    if (j < buffer_len) {
      load(kv_buf + j * Trait::kElementSize);
    } else {
      load(kv_start + j * Trait::kElementSize);
    }
#pragma unroll
    for (int32_t k = 0; k < kTileElements; ++k) {
      score[i][k] = score[i][k] + bias[k];
    }
  }

  using SharedBuffer = typename Trait::SharedBuffer;
  using TmpStorage = typename Trait::SharedStorage;
  __shared__ SharedBuffer s_local_val_max;
  __shared__ SharedBuffer s_local_exp_sum;
  __shared__ SharedBuffer s_local_product;
  TmpStorage tmp_val_max, tmp_exp_sum, tmp_product;

#pragma unroll
  for (int32_t k = 0; k < kTileElements; ++k) {
    float max_value = score[0][k];
#pragma unroll
    for (int32_t j = 1; j < kElementsPerWarp; ++j) {
      max_value = fmaxf(max_value, score[j][k]);
    }
    float sum_exp_value = 0.0f;
    float sum_product = 0.0f;
#pragma unroll
    for (int32_t j = 0; j < kElementsPerWarp; ++j) {
      const auto exp_score = __expf(score[j][k] - max_value);
      sum_product += kv[j][k] * exp_score;
      sum_exp_value += exp_score;
    }
    tmp_val_max[k] = max_value;
    tmp_exp_sum[k] = sum_exp_value;
    tmp_product[k] = sum_product;
  }
  s_local_val_max[warp_id][lane_id] = tmp_val_max;
  s_local_exp_sum[warp_id][lane_id] = tmp_exp_sum;
  s_local_product[warp_id][lane_id] = tmp_product;
  __syncthreads();
  PDLTriggerSecondary<kUsePDL>();

  if (warp_id < kTileElements) {
    const uint32_t tx = threadIdx.x;
    const uint32_t local_lane_id = tx / kTileElements;
    const uint32_t local_tile_id = tx % kTileElements;
    float global_max = s_local_val_max[0][local_lane_id][local_tile_id];
#pragma unroll
    for (uint32_t i = 1; i < kNumWarps; ++i) {
      global_max = fmaxf(global_max, s_local_val_max[i][local_lane_id][local_tile_id]);
    }
    float global_exp_sum = 0.0f;
    float global_product = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kNumWarps; ++i) {
      const auto exp_val = __expf(s_local_val_max[i][local_lane_id][local_tile_id] - global_max);
      global_exp_sum += s_local_exp_sum[i][local_lane_id][local_tile_id] * exp_val;
      global_product += s_local_product[i][local_lane_id][local_tile_id] * exp_val;
    }
    out[tx] = global_product / global_exp_sum;
  }
}

/// \brief Plan row lookup of fused_norm_rope_*; false = nothing to store.
template <ForwardMode kMode>
SGL_DEVICE bool
store_target(const FusedNormRopeStoreParams& params, const uint32_t work_id, int32_t& position, int64_t& out_loc) {
  if constexpr (kMode == ForwardMode::CompressExtend) {
    const auto plan = static_cast<const PlanC*>(params.handle)[work_id];
    if (plan.is_invalid()) return false;
    position = plan.seq_len - params.compress_ratio;
    out_loc = params.out_loc[plan.ragged_id];
  } else {
    const auto plan = static_cast<const PlanD*>(params.handle)[work_id];
    if (plan.seq_len % params.compress_ratio != 0) return false;
    position = plan.seq_len - params.compress_ratio;
    out_loc = params.out_loc[work_id];
  }
  return true;
}

/// \brief fused_norm_rope_indexer on fp32 registers; lane `l` owns elements [4 * l, 4 * l + 4).
template <int32_t kPageBits, bool kUsePDL, int32_t kPreshuffleSize>
SGL_DEVICE void norm_rope_store_indexer(
    const FusedNormRopeStoreParams& params, const float (&input)[4], const int32_t position, const int64_t out_loc) {
  using namespace device;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  constexpr int64_t kPageBytes = 132ll << kPageBits;
  using Float4 = AlignedVector<float, kVecSize>;

  const auto lane_id = threadIdx.x % kWarpThreads;
  const bool is_rope_lane = lane_id >= kWarpThreads - kRopeSize;
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;
  Float4 data, freq, weight;
  weight.load(params.weight, lane_id);
  if (is_rope_lane) freq.load(freqs_cis, lane_id - (kWarpThreads - kRopeSize));

  float sum_of_squares = 0.0f;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    sum_of_squares += input[i] * input[i];
  }
  sum_of_squares = warp::reduce_sum(sum_of_squares);
  const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    data[i] = input[i] * norm_factor * weight[i];
  }

  if (is_rope_lane) {
    const auto x_real = data[0], x_imag = data[1], y_real = data[2], y_imag = data[3];
    data[0] = x_real * freq[0] - x_imag * freq[1];
    data[1] = x_real * freq[1] + x_imag * freq[0];
    data[2] = y_real * freq[2] - y_imag * freq[3];
    data[3] = y_real * freq[3] + y_imag * freq[2];
  }

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
      const float other = __shfl_xor(data[i], mask, kWarpThreads);
      data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
    }
  }
  const float kHadamardScale = math::rsqrt(static_cast<float>(kHeadDim));
#pragma unroll
  for (int i = 0; i < kVecSize; ++i)
    data[i] *= kHadamardScale;

  float local_max = math::abs(data[0]);
#pragma unroll
  for (int i = 1; i < kVecSize; ++i) {
    local_max = math::max(local_max, math::abs(data[i]));
  }
  const auto abs_max = warp::reduce_max(local_max);
  const auto scale = fmaxf(1e-4f, abs_max) / kFP8E4M3Max;
  const auto inv_scale = 1.0f / scale;
  const int64_t offset = out_loc & ((1 << kPageBits) - 1);
  const auto page_ptr = params.kvcache + (out_loc >> kPageBits) * kPageBytes;
  AlignedVector<fp8x2_e4m3_t, 2> result;
  result[0] = pack_fp8(data[0] * inv_scale, data[1] * inv_scale);
  result[1] = pack_fp8(data[2] * inv_scale, data[3] * inv_scale);
  PDLTriggerSecondary<kUsePDL>();
  if constexpr (kPreshuffleSize != 0) {
    constexpr int32_t kTile = kPreshuffleSize;
    const int32_t dim_base = lane_id * kVecSize;
    const int32_t value_offset = offset / kTile * (kTile * static_cast<int32_t>(kHeadDim)) +
                                 dim_base / kTile * (kTile * kTile) + offset % kTile * kTile + dim_base % kTile;
    result.store(page_ptr + value_offset, 0);
  } else {
    result.store(page_ptr + offset * 128, lane_id);
  }
  if (lane_id == 0) reinterpret_cast<float*>(page_ptr + (128 << kPageBits) + offset * 4)[0] = scale;
}

/// \brief fused_norm_rope_flashmla (V4 fp8 / bf16) on fp32 registers; thread `tx` owns [2 * tx, 2 * tx + 2).
template <int32_t kPageBits, bool kBf16Store, bool kUsePDL>
SGL_DEVICE void norm_rope_store_flashmla(
    const FusedNormRopeStoreParams& params, const float (&input)[2], const int32_t position, const int64_t out_loc) {
  using namespace device;
  constexpr int64_t kHeadDim = 512;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 2;
  constexpr uint32_t kRopeWarp = kNumWarps - 1;
  using Float2 = AlignedVector<float, kVecSize>;

  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;
  Float2 data, freq;
  {
    __shared__ float partial_sums[kNumWarps];
    Float2 weight;
    weight.load(params.weight, tx);
    if (warp_id == kRopeWarp) freq.load(freqs_cis, lane_id);
    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      sum_of_squares += input[i] * input[i];
    }
    const auto warp_sum = warp::reduce_sum(sum_of_squares);
    if (lane_id == 0) partial_sums[warp_id] = warp_sum;
    __syncthreads();
    sum_of_squares = warp::reduce_sum<kNumWarps>(partial_sums[lane_id % kNumWarps]);
    const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      data[i] = input[i] * norm_factor * weight[i];
    }
  }

  const auto row = deepseek_v4::PagedKV<deepseek_v4::KVLayout::V4, kPageBits>::row(params.kvcache, out_loc);
  PDLTriggerSecondary<kUsePDL>();
  if constexpr (kBf16Store) {
    Float2 d = data;
    if (warp_id == kRopeWarp) {
      const auto x_real = data[0];
      const auto x_imag = data[1];
      d[0] = x_real * freq[0] - x_imag * freq[1];
      d[1] = x_real * freq[1] + x_imag * freq[0];
    }
    reinterpret_cast<bf16x2_t*>(params.kvcache + out_loc * (kHeadDim * 2))[tx] = cast<bf16x2_t>(fp32x2_t{d[0], d[1]});
  } else if (warp_id == kRopeWarp) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    data[0] = x_real * freq[0] - x_imag * freq[1];
    data[1] = x_real * freq[1] + x_imag * freq[0];
    reinterpret_cast<bf16x2_t*>(row.data + 448)[lane_id] = cast<bf16x2_t>(fp32x2_t{data[0], data[1]});
  } else {
    // bf16 round-trip as in the unfused quant_to_nope_fp8_rope_bf16_pack_triton
    const auto x = cast<float>(cast<bf16_t>(data[0]));
    const auto y = cast<float>(cast<bf16_t>(data[1]));
    const auto abs_max = warp::reduce_max(fmaxf(fabs(x), fabs(y)));
    const auto scale_ue8m0 = cast_to_ue8m0(fmaxf(1e-4f, abs_max) / kFP8E4M3Max);
    const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);
    reinterpret_cast<fp8x2_e4m3_t*>(row.data)[tx] = pack_fp8(x * inv_scale, y * inv_scale);
    if (lane_id == 0) static_cast<uint8_t*>(row.scale)[warp_id] = scale_ue8m0;
  }
}

template <
    typename BufferFloat,
    typename InputFloat,
    typename BiasFloat,
    ForwardMode kMode,
    int32_t kPageBits,
    bool kBf16Store,
    bool kUsePDL>
FLASHMLA_KERNEL void fused_c4_norm_rope_flashmla(const __grid_constant__ CompressNormRopeParams params) {
  using namespace device;
  using Trait = C4Trait<512>;

  const auto& store = params.store;
  const auto work_id = blockIdx.x;
  if (work_id >= store.num_tokens) return;

  const int64_t elem = threadIdx.x * 2;
  const auto kv_input = static_cast<const InputFloat*>(params.kv_input) + elem;
  const auto kv_buffer = static_cast<BufferFloat*>(params.kv_buffer) + elem;
  const auto score_bias = static_cast<const BiasFloat*>(params.score_bias) + elem;

  PDLWaitPrimary<kUsePDL>();
  // The state row is written by the thread that reads these elements back below.
  const auto write_state = [](BufferFloat* kv_dst, const InputFloat* kv_src) {
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
#pragma unroll
      for (int32_t k = 0; k < 2; ++k) {
        kv_dst[i * Trait::kHeadDim + k] = cast<BufferFloat>(kv_src[i * Trait::kHeadDim + k]);
      }
    }
  };
  C4Window<InputFloat, BufferFloat> w;
  if (!c4_window<Trait, kMode>(store.handle, work_id, kv_input, kv_buffer, w, write_state)) return;
  int32_t position;
  int64_t out_loc;
  if (!store_target<kMode>(store, work_id, position, out_loc)) return;

  const auto pooled = c4_pool<2, Trait>(w, score_bias, 0);
  const float input[2] = {pooled[0], pooled[1]};
  norm_rope_store_flashmla<kPageBits, kBf16Store, kUsePDL>(store, input, position, out_loc);
}

template <
    typename BufferFloat,
    typename InputFloat,
    typename BiasFloat,
    ForwardMode kMode,
    int32_t kPageBits,
    int32_t kPreshuffleSize,
    bool kUsePDL>
C4_KERNEL void fused_c4_norm_rope_indexer(const __grid_constant__ CompressNormRopeParams params) {
  using namespace device;
  using Compress = FlashCompress4Kernel<128, BufferFloat, InputFloat, float, kUsePDL>;
  using Trait = typename Compress::Trait;
  static_assert(Trait::kNumSplit == 1);

  const auto& store = params.store;
  const auto work_id = blockIdx.x * Compress::kWarpsPerBlock + threadIdx.x / kWarpThreads;
  if (work_id >= store.num_tokens) return;

  const auto kv_input = static_cast<const InputFloat*>(params.kv_input);
  const auto kv_buffer = static_cast<BufferFloat*>(params.kv_buffer);

  PDLWaitPrimary<kUsePDL>();
  C4Window<InputFloat, BufferFloat> w;
  const auto write_state = c4_write_decode<Trait, BufferFloat, InputFloat>;
  if (!c4_window<Trait, kMode>(store.handle, work_id, kv_input, kv_buffer, w, write_state)) return;
  int32_t position;
  int64_t out_loc;
  if (!store_target<kMode>(store, work_id, position, out_loc)) return;

  const auto pooled =
      c4_pool<kTileElements, Trait>(w, static_cast<const BiasFloat*>(params.score_bias), threadIdx.x % kWarpThreads);
  const float input[4] = {pooled[0], pooled[1], pooled[2], pooled[3]};
  norm_rope_store_indexer<kPageBits, kUsePDL, kPreshuffleSize>(store, input, position, out_loc);
}

// One block per (plan row, 64-column split) as in flash_c128_*. Each block publishes its split to
// `scratch` and the last of the row's kNumSplit blocks runs the norm/rope/store epilogue. Split blocks
// may run on different XCDs, hence the agent-scope release/acquire around the arrival counter; tickets
// never reset, every launch adds exactly kNumSplit to each row it reaches.
template <
    typename BufferFloat,
    typename InputFloat,
    typename BiasFloat,
    ForwardMode kMode,
    int32_t kPageBits,
    bool kBf16Store,
    bool kUsePDL>
C128_KERNEL void fused_c128_norm_rope_flashmla(const __grid_constant__ CompressNormRopeParams params) {
  using namespace device;
  using Trait = C128Trait<512>;
  static_assert(Trait::kBlockSize >= kBlockSize);

  const auto& store = params.store;
  const uint32_t work_id = blockIdx.x / Trait::kNumSplit;
  const int64_t split_offset = (blockIdx.x % Trait::kNumSplit) * Trait::kTileDim;
  if (work_id >= store.num_tokens) return;

  const auto kv_input = static_cast<const InputFloat*>(params.kv_input) + split_offset;
  const auto kv_buffer = static_cast<BufferFloat*>(params.kv_buffer) + split_offset;
  const auto score_bias = static_cast<const BiasFloat*>(params.score_bias) + split_offset;

  PDLWaitPrimary<kUsePDL>();
  const InputFloat* kv_src;
  const BufferFloat* kv_buf;
  int32_t buffer_len;
  if constexpr (kMode == ForwardMode::CompressExtend) {
    const auto plan = static_cast<const PlanC*>(store.handle)[work_id];
    if (plan.is_invalid()) return;
    kv_src = kv_input + plan.ragged_id * Trait::kElementSize;
    kv_buf = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;
    buffer_len = plan.buffer_len;
  } else {
    const auto plan = static_cast<const PlanD*>(store.handle)[work_id];
    kv_src = kv_input + work_id * Trait::kElementSize;
    // the write warp must match the load warp in the following `c128_pool`
    if (threadIdx.x / kWarpThreads == Trait::kNumWarps - 1) {
      c128_write_decode<Trait, BufferFloat, InputFloat>(kv_buffer + plan.write_loc * Trait::kElementSize, kv_src);
    }
    kv_buf = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;
    buffer_len = 128;
  }
  int32_t position;
  int64_t out_loc;
  if (!store_target<kMode>(store, work_id, position, out_loc)) return;

  __shared__ float split[Trait::kTileDim];
  __shared__ bool is_last;
  c128_pool<Trait, kUsePDL>(kv_buf, kv_src, split, score_bias, buffer_len);
  __syncthreads();
  const auto row = params.scratch + work_id * Trait::kHeadDim;
  if (threadIdx.x < Trait::kTileDim) row[split_offset + threadIdx.x] = split[threadIdx.x];
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();
  if (threadIdx.x == 0) {
    const auto ticket =
        __hip_atomic_fetch_add(params.arrivals + work_id, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    is_last = ticket % Trait::kNumSplit == Trait::kNumSplit - 1;
  }
  __syncthreads();
  // Whole waves exit, so the epilogue's barrier only counts the first kBlockSize threads.
  if (!is_last || threadIdx.x >= kBlockSize) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  const float input[2] = {row[threadIdx.x * 2], row[threadIdx.x * 2 + 1]};
  norm_rope_store_flashmla<kPageBits, kBf16Store, kUsePDL>(store, input, position, out_loc);
}

template <
    int64_t kRatio,
    int64_t kHeadDim,
    typename BufferFloat,
    typename InputFloat,
    typename BiasFloat,
    uint32_t kPageSize,
    int32_t kPreshuffleSize,
    bool kBf16Store,
    bool kUsePDL>
struct FusedCompressNormRopeKernel {
  static_assert(kRatio == 4 || kRatio == 128);
  static constexpr bool kIsC4 = kRatio == 4;
  static constexpr bool kIsIndexer = kHeadDim == 128;
  static_assert(kHeadDim == 512 || (kIsIndexer && kIsC4));
  using Compress = std::conditional_t<
      kIsC4,
      FlashCompress4Kernel<kHeadDim, BufferFloat, InputFloat, float, kUsePDL>,
      FlashCompress128Kernel<kHeadDim, BufferFloat, InputFloat, float, kUsePDL>>;
  using Trait = typename Compress::Trait;
  using Store = FusedNormRopeKernel<
      float,
      kHeadDim,
      64,
      kPageSize,
      kPreshuffleSize,
      kBf16Store,
      deepseek_v4::KVLayout::V4,
      kUsePDL>;
  // C4 windows overlap two compress groups, so its ape has 2 * 4 rows.
  static constexpr int64_t kApeRows = kIsC4 ? 8 : 128;
  // The indexer keeps the plain C4 compress block: one row per warp, at its register budget.
  using IndexerCompress = FlashCompress4Kernel<128, BufferFloat, InputFloat, float, kUsePDL>;
  static constexpr uint32_t kThreads = kIsIndexer ? IndexerCompress::kBlockSize
                                       : kIsC4    ? kBlockSize
                                                  : C128Config::kBlockSize;
  static constexpr uint32_t kBlocksPerRow = kIsC4 ? 1 : Trait::kNumSplit;

  template <ForwardMode kMode>
  static constexpr auto select_kernel() {
    constexpr auto kPageBits = Store::kLogPageSize;
    if constexpr (kIsIndexer) {
      return fused_c4_norm_rope_indexer<BufferFloat, InputFloat, BiasFloat, kMode, kPageBits, kPreshuffleSize, kUsePDL>;
    } else if constexpr (kIsC4) {
      return fused_c4_norm_rope_flashmla<BufferFloat, InputFloat, BiasFloat, kMode, kPageBits, kBf16Store, kUsePDL>;
    } else {
      return fused_c128_norm_rope_flashmla<BufferFloat, InputFloat, BiasFloat, kMode, kPageBits, kBf16Store, kUsePDL>;
    }
  }

  template <ForwardMode kMode>
  static void launch(
      const tvm::ffi::TensorView kv_buffer,
      const tvm::ffi::TensorView kv_input,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView weight,
      const float eps,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView kvcache,
      const tvm::ffi::TensorView plan,
      const tvm::ffi::Optional<tvm::ffi::TensorView> scratch,
      const tvm::ffi::Optional<tvm::ffi::TensorView> arrivals,
      host::SymbolicSize& rows,
      host::SymbolicDevice& device_) {
    using namespace host;
    TensorMatcher({-1, kRatio, Trait::kElementSize}).with_dtype<BufferFloat>().with_device(device_).verify(kv_buffer);
    TensorMatcher({-1, Trait::kElementSize}).with_dtype<InputFloat>().with_device(device_).verify(kv_input);
    TensorMatcher({kApeRows, kHeadDim}).with_dtype<BiasFloat>().with_device(device_).verify(ape);
    TensorMatcher({kHeadDim}).with_dtype<float>().with_device(device_).verify(weight);
    TensorMatcher({-1, 64}).with_dtype<float>().with_device(device_).verify(freqs_cis);
    TensorMatcher({-1}).with_dtype<int64_t>().with_device(device_).verify(out_loc);
    TensorMatcher({-1, -1})
        .with_strides({Store::kPageBytes, 1})
        .template with_dtype<uint8_t>()
        .with_device(device_)
        .verify(kvcache);
    // Extend rows address out_loc by ragged q index, decode rows by batch index.
    RuntimeCheck(out_loc.size(0) >= kv_input.size(0));

    RuntimeCheck(scratch.has_value() == !kIsC4 && arrivals.has_value() == !kIsC4);
    if constexpr (!kIsC4) {
      TensorMatcher({rows, 512}).with_dtype<float>().with_device(device_).verify(scratch.value());
      RuntimeCheck(arrivals.value().size(0) >= rows.unwrap());
      TensorMatcher({-1}).with_dtype<int32_t>().with_device(device_).verify(arrivals.value());
    }

    const auto num_rows = static_cast<uint32_t>(rows.unwrap());
    if (num_rows == 0) return;
    const auto params = CompressNormRopeParams{
        .kv_buffer = kv_buffer.data_ptr(),
        .kv_input = kv_input.data_ptr(),
        .score_bias = ape.data_ptr(),
        .scratch = kIsC4 ? nullptr : static_cast<float*>(scratch.value().data_ptr()),
        .arrivals = kIsC4 ? nullptr : static_cast<uint32_t*>(arrivals.value().data_ptr()),
        .store =
            FusedNormRopeStoreParams{
                .input = nullptr,
                .handle = plan.data_ptr(),
                .weight = weight.data_ptr(),
                .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
                .out_loc = static_cast<const int64_t*>(out_loc.data_ptr()),
                .kvcache = static_cast<uint8_t*>(kvcache.data_ptr()),
                .eps = eps,
                .compress_ratio = kRatio,
                .num_tokens = num_rows,
            },
    };
    const auto num_blocks = kIsIndexer ? div_ceil(num_rows, IndexerCompress::kWarpsPerBlock) : num_rows * kBlocksPerRow;
    LaunchKernel(num_blocks, kThreads, device_.unwrap()).enable_pdl(kUsePDL)(select_kernel<kMode>(), params);
  }

  static void decode(
      const tvm::ffi::TensorView kv_buffer,
      const tvm::ffi::TensorView kv_input,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView weight,
      const float eps,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView kvcache,
      const tvm::ffi::TensorView plan_d,
      const tvm::ffi::Optional<tvm::ffi::TensorView> scratch,
      const tvm::ffi::Optional<tvm::ffi::TensorView> arrivals) {
    using namespace host;
    auto N = SymbolicSize{"batch_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();
    TensorMatcher({N, Trait::kElementSize}).with_dtype<InputFloat>().with_device(device_).verify(kv_input);
    compress::verify_plan_d(plan_d, N, device_);
    launch<ForwardMode::CompressDecode>(
        kv_buffer, kv_input, ape, weight, eps, freqs_cis, out_loc, kvcache, plan_d, scratch, arrivals, N, device_);
  }

  // The state write for the next forward stays a separate launch after the compress reads.
  static void extend(
      const tvm::ffi::TensorView kv_buffer,
      const tvm::ffi::TensorView kv_input,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView weight,
      const float eps,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView kvcache,
      const tvm::ffi::TensorView plan_c,
      const tvm::ffi::TensorView plan_w,
      const tvm::ffi::Optional<tvm::ffi::TensorView> scratch,
      const tvm::ffi::Optional<tvm::ffi::TensorView> arrivals) {
    using namespace host;
    auto C = SymbolicSize{"num_c_plans"};
    auto W = SymbolicSize{"num_w_plans"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();
    compress::verify_plan_c(plan_c, C, device_);
    const auto plan_w_ptr = compress::verify_plan_w(plan_w, W, device_);
    launch<ForwardMode::CompressExtend>(
        kv_buffer, kv_input, ape, weight, eps, freqs_cis, out_loc, kvcache, plan_c, scratch, arrivals, C, device_);

    const auto num_w = static_cast<uint32_t>(W.unwrap());
    RuntimeCheck(kv_input.size(0) >= num_w, "invalid prefill plan: num_q < num_w");
    using WriteParams = std::conditional_t<kIsC4, Compress4PrefillParams, Compress128PrefillParams>;
    constexpr uint32_t kWriteThreads = [] {
      if constexpr (kIsC4)
        return Compress::kBlockSize;
      else
        return Trait::kWriteBlockSize;
    }();
    if (const auto num_blocks = div_ceil(num_w * Trait::kNumSplit, kWriteThreads / device::kWarpThreads)) {
      LaunchKernel(num_blocks, kWriteThreads, device_.unwrap())
          .enable_pdl(kUsePDL)(
              Compress::prefill_w_kernel,
              WriteParams{
                  .kv_buffer = kv_buffer.data_ptr(),
                  .kv_input = kv_input.data_ptr(),
                  .plan_w = plan_w_ptr,
                  .num_write = num_w});
    }
  }
};

}  // namespace sglang
