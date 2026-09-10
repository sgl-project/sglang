#include <ATen/cuda/CUDAContext.h>

#include <cmath>
#include <flashinfer/vec_dtypes.cuh>

#include "utils.h"

static constexpr int kWarpSize = 32;
static constexpr int DEFAULT_SHARED_MEM_THRESHOLD_KB = 48;  // Default shared memory quota in KB

__device__ __forceinline__ int
find_expert_for_token(int64_t token, const int32_t* __restrict__ expert_offsets, int64_t num_experts) {
  // upper_bound(expert_offsets, token) - 1. Repeated offsets from empty
  // experts are handled by moving to the rightmost matching interval.
  int lo = 0;
  int hi = static_cast<int>(num_experts);
  while (lo + 1 < hi) {
    const int mid = (lo + hi) / 2;
    if (token >= expert_offsets[mid]) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// ---------------------------------------------------------------------------
// 1. Warp‑local with configurable shared memory
//    • One warp handles one token.
//    • Eight tokens per 256‑thread CTA.
//    • Shared memory usage is configurable via template parameter.
// ---------------------------------------------------------------------------
template <
    typename T,
    typename DST_DTYPE,
    int kTokensPerCTA = 8,
    int kVecSize = 16,
    bool USE_SMEM = true,
    bool APPLY_EXPERT_RESIDUAL = false>
__global__ void per_token_quant_fp8_kernel(
    const T* __restrict__ input,
    DST_DTYPE* __restrict__ output_q,
    float* __restrict__ output_s,
    const float* __restrict__ residual,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ permutation,
    const int64_t hidden_dim,
    const int64_t num_tokens,
    const int64_t num_experts) {
  const int warp_id = threadIdx.x / kWarpSize;        // 0‑7  (8 warps)
  const int lane_id = threadIdx.x & (kWarpSize - 1);  // 0‑31
  const int token_id = blockIdx.x * kTokensPerCTA + warp_id;
  if (token_id >= num_tokens) return;

  // Global tensors for this token
  const int64_t source_token_id = permutation == nullptr ? token_id : permutation[token_id];
  const T* token_input = input + source_token_id * hidden_dim;
  DST_DTYPE* token_output = output_q + token_id * hidden_dim;
  float* token_scale = output_s + token_id;

  extern __shared__ char smem_buffer[];
  const int smem_padding = 32;  // Pad to bank boundary (32 banks * 4 bytes = 128 bytes)
  const int warp_smem_stride = (hidden_dim * sizeof(T) + smem_padding - 1) / smem_padding * smem_padding;
  const int warp_smem_offset = warp_id * warp_smem_stride;
  T* shared_input = reinterpret_cast<T*>(smem_buffer + warp_smem_offset);

  //
  // Pass-1: Load data and compute max_value
  //
  float max_value = 0.f;
  using vec_t = flashinfer::vec_t<T, kVecSize>;
  const int32_t num_vec_elems = hidden_dim / kVecSize;

  for (int32_t i = lane_id; i < num_vec_elems; i += kWarpSize) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * kVecSize);

    // Store to shared memory if USE_SMEM=true
    if constexpr (USE_SMEM) {
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        shared_input[i * kVecSize + j] = input_vec[j];
      }
    }

    // Compute max value in parallel
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      max_value = fmaxf(max_value, fabsf(static_cast<float>(input_vec[j])));
    }
  }

  // Ensure all threads in the warp have finished writing to shared memory
  if constexpr (USE_SMEM) {
    __syncwarp();
  }

  float warp_max = warpReduceMax(max_value);

  // NOTE: one CTA has multiple warps (each warp handles one token), so `scale`
  // must be per-warp/per-thread (register) instead of a single shared variable.
  const float scale = warp_max / FP8_E4M3_MAX;
  // Broadcast scale
  if (lane_id == 0) {
    if constexpr (APPLY_EXPERT_RESIDUAL) {
      const int expert = find_expert_for_token(token_id, expert_offsets, num_experts);
      token_scale[0] = scale * residual[expert];
    } else {
      token_scale[0] = scale;
    }
  }
  const float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

  //
  // Pass-2: Quantize and write back
  //
  for (int i = lane_id; i < num_vec_elems; i += kWarpSize) {
    vec_t input_vec;

    if constexpr (USE_SMEM) {
      // Load from shared memory
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        input_vec[j] = shared_input[i * kVecSize + j];
      }
    } else {
      // Reload from global memory
      input_vec.cast_load(token_input + i * kVecSize);
    }

    DST_DTYPE output_arr[kVecSize];
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = static_cast<float>(input_vec[j]) * scale_inv;
      val = fmaxf(fminf(val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
      output_arr[j] = static_cast<DST_DTYPE>(val);
#else
      output_arr[j] = c10::Float8_e4m3fnuz(
          __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
          c10::Float8_e4m3fnuz::from_bits());
#endif
    }
    if constexpr (kVecSize == 16) {
      *(uint4*)(token_output + i * kVecSize) = *(uint4*)output_arr;
    } else {
      // Use element-wise copy for vector size 8 to ensure correctness
      for (int k = 0; k < kVecSize; ++k) {
        token_output[i * kVecSize + k] = output_arr[k];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 2.  Baseline kernel (1 token / CTA, CUB block reduce)
// ---------------------------------------------------------------------------
template <typename T, typename DST_DTYPE, int kVecSize = 16, bool APPLY_EXPERT_RESIDUAL = false>
__global__ void per_token_quant_fp8_small_batch_kernel(
    const T* __restrict__ input,
    DST_DTYPE* __restrict__ output_q,
    float* __restrict__ output_s,
    const float* __restrict__ residual,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ permutation,
    const int64_t hidden_dim,
    const int64_t num_tokens,
    const int64_t num_experts) {
  const int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;

  const int tid = threadIdx.x;
  const int block_dim = blockDim.x;

  const int64_t source_token_idx = permutation == nullptr ? token_idx : permutation[token_idx];
  const T* token_input = input + source_token_idx * hidden_dim;
  DST_DTYPE* token_output = output_q + token_idx * hidden_dim;

  float max_value = 0.0f;

  // Use template parameter for vector size
  using vec_t = flashinfer::vec_t<T, kVecSize>;
  const int32_t num_vec_elems = hidden_dim / kVecSize;

  // Find max using vectorized loads
  for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * kVecSize);

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = static_cast<float>(input_vec[j]);
      max_value = fmaxf(max_value, fabsf(val));
    }
  }

  max_value = blockReduceMax(max_value);

  __shared__ float scale;
  if (tid == 0) {
    scale = max_value / FP8_E4M3_MAX;
    if constexpr (APPLY_EXPERT_RESIDUAL) {
      const int expert = find_expert_for_token(token_idx, expert_offsets, num_experts);
      output_s[token_idx] = scale * residual[expert];
    } else {
      output_s[token_idx] = scale;
    }
  }
  __syncthreads();

  const float scale_inv = scale == 0.0f ? 0.0f : 1.0f / scale;

  // Quantize using vectorized loads
  for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * kVecSize);

    DST_DTYPE output_arr[kVecSize];
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = fmaxf(fminf(static_cast<float>(input_vec[j]) * scale_inv, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
      output_arr[j] = static_cast<DST_DTYPE>(val);
#else
      output_arr[j] = c10::Float8_e4m3fnuz(
          __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
          c10::Float8_e4m3fnuz::from_bits());
#endif
    }

    if constexpr (kVecSize == 16) {
      *(uint4*)(token_output + i * kVecSize) = *(uint4*)output_arr;
    } else {
      // Use element-wise copy for vector size 8 to ensure correctness
      for (int k = 0; k < kVecSize; ++k) {
        token_output[i * kVecSize + k] = output_arr[k];
      }
    }
  }
}

// Tiny fused-MoE path: keep one quantization CTA per routed row, but let those
// CTAs derive the stable expert-major destination directly from topk_ids. CTA 0
// also publishes the metadata needed by the following grouped GEMMs. This
// removes the separate tiny metadata launch without serializing quantization.
template <typename T, typename DST_DTYPE, int kVecSize = 16>
__global__ void prepare_moe_input_and_quant_fp8_shuffled_kernel(
    const T* __restrict__ input,
    const int32_t* __restrict__ topk_ids,
    DST_DTYPE* __restrict__ output_q,
    float* __restrict__ output_s,
    const float* __restrict__ residual,
    int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ problem_sizes1,
    int32_t* __restrict__ problem_sizes2,
    int32_t* __restrict__ input_permutation,
    int32_t* __restrict__ output_permutation,
    const int hidden_dim,
    const int num_routes,
    const int topk,
    const int num_experts,
    const int intermediate_size) {
  const int route = blockIdx.x;
  const int tid = threadIdx.x;
  if (route >= num_routes) return;

  __shared__ int expert_counts[256];
  if (blockIdx.x == 0) {
    if (tid < num_experts) {
      int count = 0;
      for (int i = 0; i < num_routes; ++i) {
        count += topk_ids[i] == tid;
      }
      expert_counts[tid] = count;
    }
    __syncthreads();

    if (tid < num_experts) {
      int offset = 0;
      for (int expert = 0; expert < tid; ++expert) {
        offset += expert_counts[expert];
      }
      const int count = expert_counts[tid];
      expert_offsets[tid] = offset;
      problem_sizes1[tid * 3] = 2 * intermediate_size;
      problem_sizes1[tid * 3 + 1] = count;
      problem_sizes1[tid * 3 + 2] = hidden_dim;
      problem_sizes2[tid * 3] = hidden_dim;
      problem_sizes2[tid * 3 + 1] = count;
      problem_sizes2[tid * 3 + 2] = intermediate_size;
      if (tid + 1 == num_experts) {
        expert_offsets[num_experts] = offset + count;
      }
    }
    __syncthreads();
  }

  const int expert = topk_ids[route];
  __shared__ int route_destination;
  if (tid == 0) {
    route_destination = 0;
  }
  __syncthreads();

  int local_destination = 0;
  for (int i = tid; i < num_routes; i += blockDim.x) {
    const int other_expert = topk_ids[i];
    local_destination += other_expert < expert || (other_expert == expert && i < route);
  }
  if (local_destination != 0) {
    atomicAdd(&route_destination, local_destination);
  }
  __syncthreads();

  if (tid == 0) {
    input_permutation[route_destination] = route / topk;
    output_permutation[route] = route_destination;
  }
  __syncthreads();

  const T* token_input = input + (route / topk) * hidden_dim;
  DST_DTYPE* token_output = output_q + route_destination * hidden_dim;
  float max_value = 0.0f;
  using vec_t = flashinfer::vec_t<T, kVecSize>;
  const int32_t num_vec_elems = hidden_dim / kVecSize;

  for (int32_t i = tid; i < num_vec_elems; i += blockDim.x) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * kVecSize);
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      max_value = fmaxf(max_value, fabsf(static_cast<float>(input_vec[j])));
    }
  }

  max_value = blockReduceMax(max_value);
  __shared__ float scale;
  if (tid == 0) {
    scale = max_value / FP8_E4M3_MAX;
    output_s[route_destination] = scale * residual[expert];
  }
  __syncthreads();

  const float scale_inv = scale == 0.0f ? 0.0f : 1.0f / scale;
  for (int32_t i = tid; i < num_vec_elems; i += blockDim.x) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * kVecSize);
    DST_DTYPE output_arr[kVecSize];
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = fmaxf(fminf(static_cast<float>(input_vec[j]) * scale_inv, FP8_E4M3_MAX), -FP8_E4M3_MAX);
      output_arr[j] = static_cast<DST_DTYPE>(val);
    }
    if constexpr (kVecSize == 16) {
      *(uint4*)(token_output + i * kVecSize) = *(uint4*)output_arr;
    } else {
#pragma unroll
      for (int j = 0; j < kVecSize; ++j) {
        token_output[i * kVecSize + j] = output_arr[j];
      }
    }
  }
}

bool fused_prepare_moe_input_and_quant_fp8_shuffled(
    const torch::Tensor& input,
    const torch::Tensor& topk_ids,
    torch::Tensor& output_q,
    torch::Tensor& output_s,
    const torch::Tensor& residual,
    torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1,
    torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation,
    torch::Tensor& output_permutation,
    int64_t num_experts,
    int64_t intermediate_size) {
  constexpr int kMaxExperts = 256;
  constexpr int kMaxRoutes = 384;
  const int64_t num_routes = topk_ids.numel();
  const int64_t hidden_dim = input.size(1);
  if (num_experts < 1 || num_experts > kMaxExperts || num_routes < 1 || num_routes > kMaxRoutes ||
      topk_ids.dim() != 2 || topk_ids.scalar_type() != at::kInt || hidden_dim % 4 != 0) {
    return false;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int kThreads = 256;
  const bool use_vec16 = hidden_dim % 16 == 0;
  const bool use_vec8 = hidden_dim % 8 == 0;
#define LAUNCH_TINY_FUSED_QUANT(VEC_SIZE)                                            \
  prepare_moe_input_and_quant_fp8_shuffled_kernel<scalar_t, __nv_fp8_e4m3, VEC_SIZE> \
      <<<num_routes, kThreads, 0, stream>>>(                                         \
          static_cast<const scalar_t*>(input.data_ptr()),                            \
          static_cast<const int32_t*>(topk_ids.data_ptr()),                          \
          static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),                          \
          static_cast<float*>(output_s.data_ptr()),                                  \
          static_cast<const float*>(residual.data_ptr()),                            \
          static_cast<int32_t*>(expert_offsets.data_ptr()),                          \
          static_cast<int32_t*>(problem_sizes1.data_ptr()),                          \
          static_cast<int32_t*>(problem_sizes2.data_ptr()),                          \
          static_cast<int32_t*>(input_permutation.data_ptr()),                       \
          static_cast<int32_t*>(output_permutation.data_ptr()),                      \
          hidden_dim,                                                                \
          num_routes,                                                                \
          topk_ids.size(1),                                                          \
          num_experts,                                                               \
          intermediate_size)
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    if (use_vec16) {
      LAUNCH_TINY_FUSED_QUANT(16);
    } else if (use_vec8) {
      LAUNCH_TINY_FUSED_QUANT(8);
    } else {
      LAUNCH_TINY_FUSED_QUANT(4);
    }
    return true;
  });
#undef LAUNCH_TINY_FUSED_QUANT
  return true;
}

template <bool USE_SMEM, bool APPLY_EXPERT_RESIDUAL, typename scalar_t, int TOKENS_PER_CTA>
static inline void launch_per_token_quant_fp8_warp_kernel(
    const dim3& grid,
    const dim3& block,
    size_t dynamicSmemSz,
    cudaStream_t stream,
    bool use_vec16,
    bool use_vec8,
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    const float* residual,
    const int32_t* expert_offsets,
    const int32_t* permutation,
    const int64_t hidden_dim,
    const int64_t num_tokens,
    const int64_t num_experts) {
  const size_t smem_size = USE_SMEM ? dynamicSmemSz : 0;

  if (use_vec16) {
    per_token_quant_fp8_kernel<scalar_t, __nv_fp8_e4m3, TOKENS_PER_CTA, 16, USE_SMEM, APPLY_EXPERT_RESIDUAL>
        <<<grid, block, smem_size, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
            static_cast<float*>(output_s.data_ptr()),
            residual,
            expert_offsets,
            permutation,
            hidden_dim,
            num_tokens,
            num_experts);
  } else if (use_vec8) {
    per_token_quant_fp8_kernel<scalar_t, __nv_fp8_e4m3, TOKENS_PER_CTA, 8, USE_SMEM, APPLY_EXPERT_RESIDUAL>
        <<<grid, block, smem_size, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
            static_cast<float*>(output_s.data_ptr()),
            residual,
            expert_offsets,
            permutation,
            hidden_dim,
            num_tokens,
            num_experts);
  } else {
    per_token_quant_fp8_kernel<scalar_t, __nv_fp8_e4m3, TOKENS_PER_CTA, 4, USE_SMEM, APPLY_EXPERT_RESIDUAL>
        <<<grid, block, smem_size, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
            static_cast<float*>(output_s.data_ptr()),
            residual,
            expert_offsets,
            permutation,
            hidden_dim,
            num_tokens,
            num_experts);
  }
}

template <bool APPLY_EXPERT_RESIDUAL>
void per_token_quant_fp8_impl(
    const torch::Tensor& input,
    torch::Tensor& output_q,
    torch::Tensor& output_s,
    const float* residual,
    const int32_t* expert_offsets,
    const int32_t* permutation,
    int64_t num_experts) {
  const auto input_sizes = input.sizes();
  const int64_t num_tokens = permutation == nullptr ? input_sizes[0] : output_q.size(0);
  const int64_t hidden_dim = input_sizes[1];
  TORCH_CHECK(hidden_dim % 4 == 0, "Hidden dimension must be divisible by 4, but got ", hidden_dim);
  if (num_tokens == 0) return;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int TOKENS_PER_CTA = 8;
  const bool use_warp_kernel = (num_tokens >= sm_count * 2 * TOKENS_PER_CTA);
  const bool use_vec16 = (hidden_dim % 16 == 0);
  const bool use_vec8 = (hidden_dim % 8 == 0);

  const int sizeof_T = input.scalar_type() == torch::kFloat16 ? 2 : (input.scalar_type() == torch::kBFloat16 ? 2 : 4);
  const int smem_padding = 32;  // Pad to bank boundary to avoid conflicts
  const int warp_smem_stride = (hidden_dim * sizeof_T + smem_padding - 1) / smem_padding * smem_padding;
  const size_t dynamicSmemSz = warp_smem_stride * TOKENS_PER_CTA;

  bool use_smem = (hidden_dim < 2048);

  if (dynamicSmemSz >= DEFAULT_SHARED_MEM_THRESHOLD_KB) {
    use_smem = false;  // Disable shared memory if >= 48KB to avoid allocation failures
  }

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    if (use_warp_kernel) {
      // -------- warp‑local ---------------------------------------------------
      constexpr int THREADS = TOKENS_PER_CTA * kWarpSize;
      dim3 grid((num_tokens + TOKENS_PER_CTA - 1) / TOKENS_PER_CTA);
      dim3 block(THREADS);

      if (use_smem) {
        launch_per_token_quant_fp8_warp_kernel<
            /*USE_SMEM=*/true,
            APPLY_EXPERT_RESIDUAL,
            scalar_t,
            TOKENS_PER_CTA>(
            grid,
            block,
            dynamicSmemSz,
            stream,
            use_vec16,
            use_vec8,
            input,
            output_q,
            output_s,
            residual,
            expert_offsets,
            permutation,
            hidden_dim,
            num_tokens,
            num_experts);
      } else {
        launch_per_token_quant_fp8_warp_kernel<
            /*USE_SMEM=*/false,
            APPLY_EXPERT_RESIDUAL,
            scalar_t,
            TOKENS_PER_CTA>(
            grid,
            block,
            dynamicSmemSz,
            stream,
            use_vec16,
            use_vec8,
            input,
            output_q,
            output_s,
            residual,
            expert_offsets,
            permutation,
            hidden_dim,
            num_tokens,
            num_experts);
      }
    } else {
      // -------- baseline -----------------------------------------------------
      constexpr int THREADS = 256;
      dim3 grid(num_tokens);
      dim3 block(THREADS);

      if (use_vec16) {
        per_token_quant_fp8_small_batch_kernel<scalar_t, __nv_fp8_e4m3, 16, APPLY_EXPERT_RESIDUAL>
            <<<grid, block, 0, stream>>>(
                static_cast<const scalar_t*>(input.data_ptr()),
                static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                static_cast<float*>(output_s.data_ptr()),
                residual,
                expert_offsets,
                permutation,
                hidden_dim,
                num_tokens,
                num_experts);
      } else if (use_vec8) {
        per_token_quant_fp8_small_batch_kernel<scalar_t, __nv_fp8_e4m3, 8, APPLY_EXPERT_RESIDUAL>
            <<<grid, block, 0, stream>>>(
                static_cast<const scalar_t*>(input.data_ptr()),
                static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                static_cast<float*>(output_s.data_ptr()),
                residual,
                expert_offsets,
                permutation,
                hidden_dim,
                num_tokens,
                num_experts);
      } else {
        per_token_quant_fp8_small_batch_kernel<scalar_t, __nv_fp8_e4m3, 4, APPLY_EXPERT_RESIDUAL>
            <<<grid, block, 0, stream>>>(
                static_cast<const scalar_t*>(input.data_ptr()),
                static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                static_cast<float*>(output_s.data_ptr()),
                residual,
                expert_offsets,
                permutation,
                hidden_dim,
                num_tokens,
                num_experts);
      }
    }
    return true;
  });
}

void sgl_per_token_quant_fp8(torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);
  TORCH_CHECK(input.dim() == 2, "input must have shape [M, K]");
  per_token_quant_fp8_impl</*APPLY_EXPERT_RESIDUAL=*/false>(input, output_q, output_s, nullptr, nullptr, nullptr, 0);
}

#ifndef USE_ROCM
void fused_per_token_quant_fp8(
    const torch::Tensor& input,
    torch::Tensor& output_q,
    torch::Tensor& output_s,
    const torch::Tensor& residual,
    const torch::Tensor& expert_offsets,
    int64_t num_experts) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);
  CHECK_INPUT(residual);
  CHECK_INPUT(expert_offsets);
  TORCH_CHECK(input.dim() == 2, "input must have shape [M, K]");
  TORCH_CHECK(output_q.sizes() == input.sizes(), "output_q must have the same shape as input");
  TORCH_CHECK(output_q.scalar_type() == at::kFloat8_e4m3fn, "output_q must be float8_e4m3fn");
  TORCH_CHECK(
      output_s.numel() == input.size(0) && output_s.scalar_type() == at::kFloat,
      "output_s must be float32 with M elements");
  TORCH_CHECK(residual.numel() == num_experts && residual.scalar_type() == at::kFloat, "residual must be float32 [E]");
  TORCH_CHECK(
      expert_offsets.numel() == num_experts + 1 && expert_offsets.scalar_type() == at::kInt,
      "expert_offsets must be int32 [E + 1]");
  TORCH_CHECK(num_experts >= 1 && num_experts <= 256, "num_experts must be in [1, 256]");
  TORCH_CHECK(
      input.device() == output_q.device() && input.device() == output_s.device() &&
          input.device() == residual.device() && input.device() == expert_offsets.device(),
      "all tensors must be on the same device");

  per_token_quant_fp8_impl</*APPLY_EXPERT_RESIDUAL=*/true>(
      input,
      output_q,
      output_s,
      static_cast<const float*>(residual.data_ptr()),
      static_cast<const int32_t*>(expert_offsets.data_ptr()),
      nullptr,
      num_experts);
}

void fused_per_token_quant_fp8_shuffled(
    const torch::Tensor& input,
    const torch::Tensor& permutation,
    torch::Tensor& output_q,
    torch::Tensor& output_s,
    const torch::Tensor& residual,
    const torch::Tensor& expert_offsets,
    int64_t num_experts) {
  CHECK_INPUT(input);
  CHECK_INPUT(permutation);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);
  CHECK_INPUT(residual);
  CHECK_INPUT(expert_offsets);
  TORCH_CHECK(input.dim() == 2, "input must have shape [M, K]");
  TORCH_CHECK(permutation.dim() == 1 && permutation.scalar_type() == at::kInt, "permutation must be int32 [N]");
  TORCH_CHECK(
      output_q.dim() == 2 && output_q.size(0) == permutation.numel() && output_q.size(1) == input.size(1),
      "output_q must have shape [N, K]");
  TORCH_CHECK(output_q.scalar_type() == at::kFloat8_e4m3fn, "output_q must be float8_e4m3fn");
  TORCH_CHECK(
      output_s.numel() == permutation.numel() && output_s.scalar_type() == at::kFloat,
      "output_s must be float32 with N elements");
  TORCH_CHECK(residual.numel() == num_experts && residual.scalar_type() == at::kFloat, "residual must be float32 [E]");
  TORCH_CHECK(
      expert_offsets.numel() == num_experts + 1 && expert_offsets.scalar_type() == at::kInt,
      "expert_offsets must be int32 [E + 1]");
  TORCH_CHECK(num_experts >= 1 && num_experts <= 256, "num_experts must be in [1, 256]");
  TORCH_CHECK(
      input.device() == permutation.device() && input.device() == output_q.device() &&
          input.device() == output_s.device() && input.device() == residual.device() &&
          input.device() == expert_offsets.device(),
      "all tensors must be on the same device");

  per_token_quant_fp8_impl</*APPLY_EXPERT_RESIDUAL=*/true>(
      input,
      output_q,
      output_s,
      static_cast<const float*>(residual.data_ptr()),
      static_cast<const int32_t*>(expert_offsets.data_ptr()),
      static_cast<const int32_t*>(permutation.data_ptr()),
      num_experts);
}
#endif
