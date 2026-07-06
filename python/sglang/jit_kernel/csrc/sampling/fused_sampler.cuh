#pragma once

#include <sgl_kernel/tensor.h>   // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>    // For RuntimeCheck, div_ceil

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, scalar dtype aliases

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <climits>
#include <cstdint>
#include <type_traits>

namespace {

constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kTileSize = 1024;

template <typename T>
SGL_DEVICE float load_logit_as_float(const T* ptr, int64_t offset) {
  if constexpr (std::is_same_v<T, fp32_t>) {
    return ptr[offset];
  } else if constexpr (std::is_same_v<T, fp16_t>) {
    return __half2float(ptr[offset]);
  } else {
    return __bfloat162float(ptr[offset]);
  }
}

SGL_DEVICE bool pair_greater(float lhs_score, int32_t lhs_idx, float rhs_score, int32_t rhs_idx) {
  if (lhs_idx < 0) return false;
  if (rhs_idx < 0) return true;
  return lhs_score > rhs_score || (lhs_score == rhs_score && lhs_idx < rhs_idx);
}

SGL_DEVICE bool pair_less_than_prev(float score, int32_t idx, float prev_score, int32_t prev_idx) {
  if (idx < 0) return false;
  if (prev_idx < 0) return true;
  return score < prev_score || (score == prev_score && idx > prev_idx);
}

template <typename DType>
__global__ void greedy_sample_kernel(
    int64_t* __restrict__ out,
    const DType* __restrict__ logits,
    uint32_t batch_size,
    uint32_t vocab_size) {
  const uint32_t row = blockIdx.x;
  if (row >= batch_size) return;

  const DType* row_logits = logits + static_cast<int64_t>(row) * vocab_size;
  float local_best = -FLT_MAX;
  int32_t local_idx = -1;

  for (uint32_t col = threadIdx.x; col < vocab_size; col += blockDim.x) {
    const float score = load_logit_as_float(row_logits, col);
    if (pair_greater(score, static_cast<int32_t>(col), local_best, local_idx)) {
      local_best = score;
      local_idx = static_cast<int32_t>(col);
    }
  }

  __shared__ float reduce_scores[kThreadsPerBlock];
  __shared__ int32_t reduce_indices[kThreadsPerBlock];
  reduce_scores[threadIdx.x] = local_best;
  reduce_indices[threadIdx.x] = local_idx;
  __syncthreads();

  for (uint32_t stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride &&
        pair_greater(reduce_scores[threadIdx.x + stride], reduce_indices[threadIdx.x + stride],
                     reduce_scores[threadIdx.x], reduce_indices[threadIdx.x])) {
      reduce_scores[threadIdx.x] = reduce_scores[threadIdx.x + stride];
      reduce_indices[threadIdx.x] = reduce_indices[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    out[row] = static_cast<int64_t>(reduce_indices[0]);
  }
}

template <typename DType, uint32_t kTopK, bool kNeedsTopP>
__global__ void tile_topk_iterative_kernel(
    float* __restrict__ tile_scores,
    int32_t* __restrict__ tile_indices,
    float* __restrict__ tile_exp_sums,
    const DType* __restrict__ logits,
    const float* __restrict__ temperatures,
    uint32_t batch_size,
    uint32_t vocab_size,
    uint32_t num_tiles) {
  const uint32_t tile_id = blockIdx.x;
  const uint32_t row = blockIdx.y;
  if (row >= batch_size || tile_id >= num_tiles) return;

  const uint32_t tile_start = tile_id * kTileSize;
  const DType* row_logits = logits + static_cast<int64_t>(row) * vocab_size;

  __shared__ float values[kTileSize];
  __shared__ float reduce_scores[kThreadsPerBlock];
  __shared__ int32_t reduce_indices[kThreadsPerBlock];
  __shared__ float selected_score;
  __shared__ int32_t selected_idx;

  for (uint32_t offset = threadIdx.x; offset < kTileSize; offset += blockDim.x) {
    const uint32_t col = tile_start + offset;
    values[offset] = col < vocab_size ? load_logit_as_float(row_logits, col) : -FLT_MAX;
  }
  __syncthreads();

  float prev_score = FLT_MAX;
  int32_t prev_idx = -1;
  const int64_t out_base =
      (static_cast<int64_t>(row) * num_tiles + tile_id) * static_cast<int64_t>(kTopK);

#pragma unroll
  for (uint32_t rank = 0; rank < kTopK; ++rank) {
    float local_best = -FLT_MAX;
    int32_t local_idx = -1;

    for (uint32_t offset = threadIdx.x; offset < kTileSize; offset += blockDim.x) {
      const uint32_t col = tile_start + offset;
      if (col >= vocab_size) continue;
      const float score = values[offset];
      const int32_t idx = static_cast<int32_t>(col);
      if (pair_less_than_prev(score, idx, prev_score, prev_idx) &&
          pair_greater(score, idx, local_best, local_idx)) {
        local_best = score;
        local_idx = idx;
      }
    }

    reduce_scores[threadIdx.x] = local_best;
    reduce_indices[threadIdx.x] = local_idx;
    __syncthreads();

    for (uint32_t stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride &&
          pair_greater(reduce_scores[threadIdx.x + stride], reduce_indices[threadIdx.x + stride],
                       reduce_scores[threadIdx.x], reduce_indices[threadIdx.x])) {
        reduce_scores[threadIdx.x] = reduce_scores[threadIdx.x + stride];
        reduce_indices[threadIdx.x] = reduce_indices[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      selected_score = reduce_scores[0];
      selected_idx = reduce_indices[0];
      tile_scores[out_base + rank] = selected_score;
      tile_indices[out_base + rank] = selected_idx;
    }
    __syncthreads();

    prev_score = selected_score;
    prev_idx = selected_idx;
  }

  if constexpr (kNeedsTopP) {
    const float tile_max_score = tile_scores[out_base];
    const float inv_temperature = 1.0f / fmaxf(temperatures[row], 1.0e-6f);
    const float tile_max_scaled = tile_max_score * inv_temperature;
    float local_sum = 0.0f;

    for (uint32_t offset = threadIdx.x; offset < kTileSize; offset += blockDim.x) {
      const uint32_t col = tile_start + offset;
      if (col < vocab_size) {
        local_sum += expf(values[offset] * inv_temperature - tile_max_scaled);
      }
    }

    reduce_scores[threadIdx.x] = local_sum;
    __syncthreads();

    for (uint32_t stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reduce_scores[threadIdx.x] += reduce_scores[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      tile_exp_sums[static_cast<int64_t>(row) * num_tiles + tile_id] = reduce_scores[0];
    }
  }
}

template <uint32_t kTopK, bool kNeedsTopP>
__global__ void merge_topk_sample_kernel(
    int64_t* __restrict__ out,
    const float* __restrict__ tile_scores,
    const int32_t* __restrict__ tile_indices,
    const float* __restrict__ tile_exp_sums,
    const float* __restrict__ temperatures,
    const float* __restrict__ top_ps,
    const float* __restrict__ uniforms,
    uint32_t batch_size,
    uint32_t num_tiles) {
  const uint32_t row = blockIdx.x;
  if (row >= batch_size) return;

  __shared__ float reduce_scores[kThreadsPerBlock];
  __shared__ int32_t reduce_indices[kThreadsPerBlock];
  __shared__ float selected_scores[kTopK];
  __shared__ int32_t selected_indices[kTopK];
  __shared__ float selected_score;
  __shared__ int32_t selected_idx;
  __shared__ float full_total_exp;

  const uint32_t candidate_count = num_tiles * kTopK;
  const int64_t row_base = static_cast<int64_t>(row) * candidate_count;
  float prev_score = FLT_MAX;
  int32_t prev_idx = -1;

#pragma unroll
  for (uint32_t rank = 0; rank < kTopK; ++rank) {
    float local_best = -FLT_MAX;
    int32_t local_idx = -1;

    for (uint32_t i = threadIdx.x; i < candidate_count; i += blockDim.x) {
      const int64_t offset = row_base + i;
      const int32_t idx = tile_indices[offset];
      if (idx < 0) continue;
      const float score = tile_scores[offset];
      if (pair_less_than_prev(score, idx, prev_score, prev_idx) &&
          pair_greater(score, idx, local_best, local_idx)) {
        local_best = score;
        local_idx = idx;
      }
    }

    reduce_scores[threadIdx.x] = local_best;
    reduce_indices[threadIdx.x] = local_idx;
    __syncthreads();

    for (uint32_t stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride &&
          pair_greater(reduce_scores[threadIdx.x + stride], reduce_indices[threadIdx.x + stride],
                       reduce_scores[threadIdx.x], reduce_indices[threadIdx.x])) {
        reduce_scores[threadIdx.x] = reduce_scores[threadIdx.x + stride];
        reduce_indices[threadIdx.x] = reduce_indices[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      selected_score = reduce_scores[0];
      selected_idx = reduce_indices[0];
      selected_scores[rank] = selected_score;
      selected_indices[rank] = selected_idx;
    }
    __syncthreads();

    prev_score = selected_score;
    prev_idx = selected_idx;
  }

  const float inv_temperature = 1.0f / fmaxf(temperatures[row], 1.0e-6f);
  const float max_scaled = selected_scores[0] * inv_temperature;
  const float top_p = fminf(fmaxf(top_ps[row], 0.0f), 1.0f);

  if constexpr (kNeedsTopP) {
    float local_total = 0.0f;
    for (uint32_t tile = threadIdx.x; tile < num_tiles; tile += blockDim.x) {
      const int64_t offset = row_base + static_cast<int64_t>(tile) * kTopK;
      if (tile_indices[offset] >= 0) {
        local_total +=
            expf(tile_scores[offset] * inv_temperature - max_scaled) *
            tile_exp_sums[static_cast<int64_t>(row) * num_tiles + tile];
      }
    }

    reduce_scores[threadIdx.x] = local_total;
    __syncthreads();

    for (uint32_t stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reduce_scores[threadIdx.x] += reduce_scores[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      full_total_exp = reduce_scores[0];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float weights[kTopK];
    uint32_t valid_count = 0;

#pragma unroll
    for (uint32_t i = 0; i < kTopK; ++i) {
      if (selected_indices[i] >= 0) {
        const float w = expf(selected_scores[i] * inv_temperature - max_scaled);
        weights[i] = w;
        valid_count = i + 1;
      } else {
        weights[i] = 0.0f;
      }
    }

    float accepted_sum = 0.0f;
    uint32_t accepted_count = 0;
    const float cutoff = kNeedsTopP ? top_p * full_total_exp : FLT_MAX;
#pragma unroll
    for (uint32_t i = 0; i < kTopK; ++i) {
      if (i >= valid_count) break;
      if (accepted_sum > cutoff) break;
      accepted_sum += weights[i];
      accepted_count = i + 1;
    }
    if (accepted_count == 0 && valid_count > 0) {
      accepted_count = 1;
      accepted_sum = weights[0];
    }

    const float target = fminf(fmaxf(uniforms[row], 0.0f), 0.99999994f) * accepted_sum;
    float cumsum = 0.0f;
    int32_t sampled = selected_indices[0];
#pragma unroll
    for (uint32_t i = 0; i < kTopK; ++i) {
      if (i >= accepted_count) break;
      cumsum += weights[i];
      sampled = selected_indices[i];
      if (cumsum >= target) break;
    }
    out[row] = static_cast<int64_t>(sampled);
  }
}

template <typename DType, uint32_t kTopK, bool kNeedsTopP>
struct FusedTopKSampleKernel {
  static_assert(
      std::is_same_v<DType, fp32_t> || std::is_same_v<DType, fp16_t> || std::is_same_v<DType, bf16_t>,
      "fused sampler supports fp32/fp16/bf16 logits");
  static_assert(kTopK >= 1 && kTopK <= 8, "top_k must be in [1, 8]");

  static void run(
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView temperatures,
      const tvm::ffi::TensorView top_ps,
      const tvm::ffi::TensorView uniforms,
      const tvm::ffi::TensorView scratch_scores,
      const tvm::ffi::TensorView scratch_indices,
      const tvm::ffi::TensorView scratch_sums) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto V = SymbolicSize{"vocab_size"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, V}).with_dtype<DType>().with_device(device).verify(logits);
    TensorMatcher({B}).with_dtype<fp32_t>().with_device(device).verify(temperatures).verify(top_ps).verify(uniforms);
    TensorMatcher({B}).with_dtype<int64_t>().with_device(device).verify(out);

    const uint32_t batch_size = static_cast<uint32_t>(B.unwrap());
    const uint32_t vocab_size = static_cast<uint32_t>(V.unwrap());
    const auto dl_device = device.unwrap();
    RuntimeCheck(batch_size > 0, "fused_topk_sample: batch_size must be > 0");
    RuntimeCheck(vocab_size > 0, "fused_topk_sample: vocab_size must be > 0");
    RuntimeCheck(kTopK <= vocab_size, "fused_topk_sample: top_k exceeds vocab_size");

    if constexpr (kTopK == 1) {
      LaunchKernel(batch_size, kThreadsPerBlock, dl_device)(
          greedy_sample_kernel<DType>,
          static_cast<int64_t*>(out.data_ptr()),
          static_cast<const DType*>(logits.data_ptr()),
          batch_size,
          vocab_size);
      return;
    }

    const uint32_t num_tiles = static_cast<uint32_t>(div_ceil(vocab_size, kTileSize));
    auto T = SymbolicSize{"num_tiles"};
    auto K = SymbolicSize{"top_k"};
    T.set_value(num_tiles);
    K.set_value(kTopK);
    TensorMatcher({B, T, K}).with_dtype<fp32_t>().with_device(device).verify(scratch_scores);
    TensorMatcher({B, T, K}).with_dtype<int32_t>().with_device(device).verify(scratch_indices);
    float* scratch_sums_ptr = nullptr;
    if constexpr (kNeedsTopP) {
      TensorMatcher({B, T}).with_dtype<fp32_t>().with_device(device).verify(scratch_sums);
      scratch_sums_ptr = static_cast<float*>(scratch_sums.data_ptr());
    }

    LaunchKernel(dim3(num_tiles, batch_size), kThreadsPerBlock, dl_device)(
        tile_topk_iterative_kernel<DType, kTopK, kNeedsTopP>,
        static_cast<float*>(scratch_scores.data_ptr()),
        static_cast<int32_t*>(scratch_indices.data_ptr()),
        scratch_sums_ptr,
        static_cast<const DType*>(logits.data_ptr()),
        static_cast<const float*>(temperatures.data_ptr()),
        batch_size,
        vocab_size,
        num_tiles);
    LaunchKernel(batch_size, kThreadsPerBlock, dl_device)(
        merge_topk_sample_kernel<kTopK, kNeedsTopP>,
        static_cast<int64_t*>(out.data_ptr()),
        static_cast<const float*>(scratch_scores.data_ptr()),
        static_cast<const int32_t*>(scratch_indices.data_ptr()),
        static_cast<const float*>(scratch_sums_ptr),
        static_cast<const float*>(temperatures.data_ptr()),
        static_cast<const float*>(top_ps.data_ptr()),
        static_cast<const float*>(uniforms.data_ptr()),
        batch_size,
        num_tiles);
  }
};

}  // namespace
