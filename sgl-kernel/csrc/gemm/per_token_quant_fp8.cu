#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>

#include <cmath>
#include <flashinfer/vec_dtypes.cuh>
#include <type_traits>

#ifdef USE_ROCM
#if defined(HIP_FP8_TYPE_FNUZ)
#include <c10/util/Float8_e4m3fnuz.h>
#endif
#endif

#include "utils.h"

static constexpr int kWarpSize = 32;

template <typename DstFp8T>
__host__ __device__ __forceinline__ float fp8_max() {
  if constexpr (std::is_same_v<DstFp8T, __nv_fp8_e4m3>) {
    return static_cast<float>(std::numeric_limits<c10::Float8_e4m3fn>::max());
  } else if constexpr (std::is_same_v<DstFp8T, __nv_fp8_e5m2>) {
    return static_cast<float>(std::numeric_limits<c10::Float8_e5m2>::max());
  } else {
    return 0.0f;
  }
}

template <typename T>
__device__ __forceinline__ float to_float(T x) {
  if constexpr (std::is_same_v<T, __half>) {
    return __half2float(x);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(x);
  } else {
    return static_cast<float>(x);
  }
}

template <typename T>
__device__ __forceinline__ T from_float(float x) {
  if constexpr (std::is_same_v<T, __half>) {
    return __float2half(x);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __float2bfloat16(x);
  } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
    return static_cast<T>(x);
  } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
    return static_cast<T>(x);
  } else {
    return static_cast<T>(x);
  }
}

// ---------------------------------------------------------------------------
// 1. Warp‑local, no shared memory
//    • One warp handles one token.
//    • Eight tokens per 256‑thread CTA.
// ---------------------------------------------------------------------------
template <typename T, typename DST_Q_DTYPE, typename DST_S_DTYPE = float, int kTokensPerCTA = 8, int kVecSize = 16>
__global__ void per_token_quant_fp8_kernel(
    const T* __restrict__ input,
    DST_Q_DTYPE* __restrict__ output_q,
    DST_S_DTYPE* __restrict__ output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens) {
  const int warp_id = threadIdx.x / kWarpSize;        // 0‑7  (8 warps)
  const int lane_id = threadIdx.x & (kWarpSize - 1);  // 0‑31
  const int token_id = blockIdx.x * kTokensPerCTA + warp_id;
  if (token_id >= num_tokens) return;

  // Global tensors for this token
  const T* token_input = input + token_id * hidden_dim;
  DST_Q_DTYPE* token_output = output_q + token_id * hidden_dim;
  DST_S_DTYPE* token_scale = output_s + token_id;

  //
  // Pass-1: Perform a warp reduce to find the max_value of a token's hidden_dim
  //
  float max_value = 0.f;
  using vec_t = flashinfer::vec_t<T, kVecSize>;
  const int32_t num_vec_elems = hidden_dim / kVecSize;

  for (int32_t i = lane_id; i < num_vec_elems; i += kWarpSize) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * kVecSize);

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      max_value = fmaxf(max_value, fabsf(to_float(input_vec[j])));
    }
  }

  float warp_max = warpReduceMax(max_value);

  __shared__ float scale;
  const float FP8_MAX = fp8_max<DST_Q_DTYPE>();
  scale = warp_max / FP8_MAX;
  // Broadcast scale
  if (lane_id == 0) {
    token_scale[0] = from_float<DST_S_DTYPE>(scale);
  }
  float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

  //
  // Pass-2: quantize and write back
  //
  for (int i = lane_id; i < num_vec_elems; i += kWarpSize) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * kVecSize);
    DST_Q_DTYPE output_arr[kVecSize];
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = to_float(input_vec[j]) * scale_inv;
      val = fmaxf(fminf(val, FP8_MAX), -FP8_MAX);
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
      output_arr[j] = static_cast<DST_Q_DTYPE>(val);
#else
      // ROCM with FNUZ type
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
template <typename T, typename DST_Q_DTYPE, typename DST_S_DTYPE = float, int kVecSize = 16>
__global__ void per_token_quant_fp8_small_batch_kernel(
    const T* __restrict__ input,
    DST_Q_DTYPE* __restrict__ output_q,
    DST_S_DTYPE* __restrict__ output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens) {
  const int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;

  const int tid = threadIdx.x;
  const int block_dim = blockDim.x;

  const T* token_input = input + token_idx * hidden_dim;
  DST_Q_DTYPE* token_output = output_q + token_idx * hidden_dim;

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
      float val = to_float(input_vec[j]);
      max_value = fmaxf(max_value, fabsf(val));
    }
  }

  max_value = blockReduceMax(max_value);

  __shared__ float scale;
  if (tid == 0) {
    const float FP8_MAX = fp8_max<DST_Q_DTYPE>();
    scale = max_value / FP8_MAX;
    output_s[token_idx] = from_float<DST_S_DTYPE>(scale);
  }
  __syncthreads();

  const float scale_inv = 1.0f / scale;

  // Quantize using vectorized loads
  for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * kVecSize);

    DST_Q_DTYPE output_arr[kVecSize];
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      const float FP8_MAX = fp8_max<DST_Q_DTYPE>();
      float val = fmaxf(fminf(to_float(input_vec[j]) * scale_inv, FP8_MAX), -FP8_MAX);
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
      output_arr[j] = static_cast<DST_Q_DTYPE>(val);
#else
      // ROCM with FNUZ type
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

void sgl_per_token_quant_fp8(torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);
  TORCH_CHECK(
      input.scalar_type() == torch::kFloat || input.scalar_type() == torch::kHalf ||
          input.scalar_type() == torch::kBFloat16,
      "Input must be a Float, Half, or BFloat16 tensor, but got ",
      input.scalar_type());
#ifdef USE_ROCM
  // ROCM platform only supports e4m3, not e5m2
  TORCH_CHECK(
      output_q.scalar_type() == torch::kFloat8_e4m3fn,
      "Output_q must be a Float8_e4m3fn tensor on ROCM platform, but got ",
      output_q.scalar_type());
#else
  TORCH_CHECK(
      output_q.scalar_type() == torch::kFloat8_e4m3fn || output_q.scalar_type() == torch::kFloat8_e5m2,
      "Output_q must be a Float8_e4m3fn or Float8_e5m2 tensor, but got ",
      output_q.scalar_type());
#endif
  TORCH_CHECK(
      output_s.scalar_type() == torch::kFloat || output_s.scalar_type() == torch::kHalf ||
          output_s.scalar_type() == torch::kBFloat16,
      "Output_s must be a Float, Half, or BFloat16 tensor, but got ",
      output_s.scalar_type());
  const auto input_sizes = input.sizes();
  const int64_t num_tokens = input_sizes[0];
  const int64_t hidden_dim = input_sizes[1];
  TORCH_CHECK(hidden_dim % 8 == 0, "Hidden dimension must be divisible by 8, but got ", hidden_dim);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int TOKENS_PER_CTA = 8;
  const bool use_warp_kernel = (num_tokens >= sm_count * 2 * TOKENS_PER_CTA);
  const bool use_vec16 = (hidden_dim % 16 == 0);
  const bool use_vec8 = (hidden_dim % 8 == 0);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), input_scalar_t, [&] {
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(output_q.scalar_type(), output_q_scalar_t, [&] {
      DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(output_s.scalar_type(), output_s_scalar_t, [&] {
        if (use_warp_kernel) {
          // -------- warp‑local ---------------------------------------------------
          constexpr int THREADS = TOKENS_PER_CTA * kWarpSize;  // 256
          dim3 grid((num_tokens + TOKENS_PER_CTA - 1) / TOKENS_PER_CTA);
          dim3 block(THREADS);

          if (use_vec16) {
            per_token_quant_fp8_kernel<input_scalar_t, output_q_scalar_t, output_s_scalar_t, TOKENS_PER_CTA, 16>
                <<<grid, block, 0, stream>>>(
                    static_cast<const input_scalar_t*>(input.data_ptr()),
                    static_cast<output_q_scalar_t*>(output_q.data_ptr()),
                    static_cast<output_s_scalar_t*>(output_s.data_ptr()),
                    hidden_dim,
                    num_tokens);
          } else if (use_vec8) {
            per_token_quant_fp8_kernel<input_scalar_t, output_q_scalar_t, output_s_scalar_t, TOKENS_PER_CTA, 8>
                <<<grid, block, 0, stream>>>(
                    static_cast<const input_scalar_t*>(input.data_ptr()),
                    static_cast<output_q_scalar_t*>(output_q.data_ptr()),
                    static_cast<output_s_scalar_t*>(output_s.data_ptr()),
                    hidden_dim,
                    num_tokens);
          } else {
            per_token_quant_fp8_kernel<input_scalar_t, output_q_scalar_t, output_s_scalar_t, TOKENS_PER_CTA, 4>
                <<<grid, block, 0, stream>>>(
                    static_cast<const input_scalar_t*>(input.data_ptr()),
                    static_cast<output_q_scalar_t*>(output_q.data_ptr()),
                    static_cast<output_s_scalar_t*>(output_s.data_ptr()),
                    hidden_dim,
                    num_tokens);
          }
        } else {
          // -------- baseline -----------------------------------------------------
          constexpr int THREADS = 256;
          dim3 grid(num_tokens);
          dim3 block(THREADS);

          if (use_vec16) {
            per_token_quant_fp8_small_batch_kernel<input_scalar_t, output_q_scalar_t, output_s_scalar_t, 16>
                <<<grid, block, 0, stream>>>(
                    static_cast<const input_scalar_t*>(input.data_ptr()),
                    static_cast<output_q_scalar_t*>(output_q.data_ptr()),
                    static_cast<output_s_scalar_t*>(output_s.data_ptr()),
                    hidden_dim,
                    num_tokens);
          } else if (use_vec8) {
            per_token_quant_fp8_small_batch_kernel<input_scalar_t, output_q_scalar_t, output_s_scalar_t, 8>
                <<<grid, block, 0, stream>>>(
                    static_cast<const input_scalar_t*>(input.data_ptr()),
                    static_cast<output_q_scalar_t*>(output_q.data_ptr()),
                    static_cast<output_s_scalar_t*>(output_s.data_ptr()),
                    hidden_dim,
                    num_tokens);
          } else {
            per_token_quant_fp8_small_batch_kernel<input_scalar_t, output_q_scalar_t, output_s_scalar_t, 4>
                <<<grid, block, 0, stream>>>(
                    static_cast<const input_scalar_t*>(input.data_ptr()),
                    static_cast<output_q_scalar_t*>(output_q.data_ptr()),
                    static_cast<output_s_scalar_t*>(output_s.data_ptr()),
                    hidden_dim,
                    num_tokens);
          }
        }
        return true;
      });
      return true;
    });
    return true;
  });
}
