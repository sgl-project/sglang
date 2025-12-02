/* Copyright 2025 SGLang Team. */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include "cutlass/cutlass.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/tensor_ref.h"

#include <cuda_fp16.h>
#include <cassert>
#include <type_traits>

#include "sgl_kernel_ops.h"
#include "utils.h"

// Minimal warp/block reduction helpers (sum) for small arrays
template <typename T, int NumVals>
__device__ __forceinline__ void warpReduceSum(T (&vals)[NumVals]) {
  unsigned mask = 0xffffffffu;
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    #pragma unroll
    for (int i = 0; i < NumVals; ++i) {
      vals[i] += __shfl_down_sync(mask, vals[i], offset);
    }
  }
}

template <typename T, int NumVals>
__device__ __forceinline__ void blockReduceSum(T (&vals)[NumVals]) {
  __shared__ T shared[32][NumVals]; // up to 32 warps (1024 threads)
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  warpReduceSum<T, NumVals>(vals);
  if (lane == 0) {
    #pragma unroll
    for (int i = 0; i < NumVals; ++i) {
      shared[wid][i] = vals[i];
    }
  }
  __syncthreads();
  if (wid == 0) {
    T acc[NumVals];
    #pragma unroll
    for (int i = 0; i < NumVals; ++i) acc[i] = T(0);
    int num_warps = (blockDim.x + 31) / 32;
    #pragma unroll
    for (int w = 0; w < 32; ++w) {
      if (w < num_warps) {
        #pragma unroll
        for (int i = 0; i < NumVals; ++i) {
          acc[i] += shared[w][i];
        }
      }
    }
    #pragma unroll
    for (int i = 0; i < NumVals; ++i) vals[i] = acc[i];
  }
  __syncthreads();
}

// Vector-of-4 type for bfloat16
struct alignas(8) bf16_4 {
  cutlass::bfloat16_t x, y, z, w;
};

struct alignas(8) half4 {
  __half x, y, z, w;
};

template<typename T4, typename T, int ITEM_PER_THREAD>
__global__ void layernorm_twoPassAlgo_stored_locally_e4(T4* output,
                                                        const T4* input,
                                                        const T4* gamma,
                                                        const T4* beta,
                                                        const int m,
                                                        const int n)
{
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};
  T4 local_val[ITEM_PER_THREAD];
  const int n_4 = n / 4;
  int offset = m_idx * n_4;
  input += offset;
  output += offset;

  const T4 zero = {T(0.0f), T(0.0f), T(0.0f), T(0.0f)};
  #pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i * bdimx + tid;
    local_val[i] = index < n_4 ? input[index] : zero;
    local_sums[0] += static_cast<float>(local_val[i].x) + static_cast<float>(local_val[i].y) +
                     static_cast<float>(local_val[i].z) + static_cast<float>(local_val[i].w);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  #pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i * bdimx + tid;
    if (index < n_4){
      const float4 tmp = {static_cast<float>(local_val[i].x) - s_mean,
                          static_cast<float>(local_val[i].y) - s_mean,
                          static_cast<float>(local_val[i].z) - s_mean,
                          static_cast<float>(local_val[i].w) - s_mean};
      local_sums[0] += tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z + tmp.w * tmp.w;
    }
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + 1e-5f);
  }
  __syncthreads();

  #pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i * bdimx + tid;
    if (index < n_4){
      const T4 gamma_val = gamma[index];
      const T4 beta_val = beta[index];
      T4 tmp;
      tmp.x = T((static_cast<float>(local_val[i].x) - s_mean)*s_variance*static_cast<float>(gamma_val.x) + static_cast<float>(beta_val.x));
      tmp.y = T((static_cast<float>(local_val[i].y) - s_mean)*s_variance*static_cast<float>(gamma_val.y) + static_cast<float>(beta_val.y));
      tmp.z = T((static_cast<float>(local_val[i].z) - s_mean)*s_variance*static_cast<float>(gamma_val.z) + static_cast<float>(beta_val.z));
      tmp.w = T((static_cast<float>(local_val[i].w) - s_mean)*s_variance*static_cast<float>(gamma_val.w) + static_cast<float>(beta_val.w));
      output[index] = tmp;
    }
  }
}

template<typename T4, typename T, int ITEM_PER_THREAD>
__global__ void layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift(
                                                       T4* output,
                                                       const T4* input,
                                                       const T4* gamma,
                                                       const T4* beta,
                                                       const T4* scale,
                                                       const T4* shift,
                                                       const int m,
                                                       const int n)
{
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};
  T4 local_val[ITEM_PER_THREAD];
  const int n_4 = n / 4;
  int offset = m_idx * n_4;
  input  += offset;
  output += offset;
  scale  += offset;
  shift  += offset;

  const T4 zero = {T(0.0f), T(0.0f), T(0.0f), T(0.0f)};
  #pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i * bdimx + tid;
    local_val[i] = index < n_4 ? input[index] : zero;
    local_sums[0] += static_cast<float>(local_val[i].x) + static_cast<float>(local_val[i].y) +
                     static_cast<float>(local_val[i].z) + static_cast<float>(local_val[i].w);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  #pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i * bdimx + tid;
    if (index < n_4){
      const float4 tmp = {static_cast<float>(local_val[i].x) - s_mean,
                          static_cast<float>(local_val[i].y) - s_mean,
                          static_cast<float>(local_val[i].z) - s_mean,
                          static_cast<float>(local_val[i].w) - s_mean};
      local_sums[0] += tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z + tmp.w * tmp.w;
    }
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + 1e-5f);
  }
  __syncthreads();

  #pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i * bdimx + tid;
    if (index < n_4){
      const T4 gamma_val = gamma[index];
      const T4 beta_val  = beta[index];
      const T4 scale_val = scale[index];
      const T4 shift_val = shift[index];
      T4 tmp;
      tmp.x = T(((static_cast<float>(local_val[i].x) - s_mean) * s_variance * static_cast<float>(gamma_val.x) + static_cast<float>(beta_val.x)) * (1.0f + static_cast<float>(scale_val.x)) + static_cast<float>(shift_val.x));
      tmp.y = T(((static_cast<float>(local_val[i].y) - s_mean) * s_variance * static_cast<float>(gamma_val.y) + static_cast<float>(beta_val.y)) * (1.0f + static_cast<float>(scale_val.y)) + static_cast<float>(shift_val.y));
      tmp.z = T(((static_cast<float>(local_val[i].z) - s_mean) * s_variance * static_cast<float>(gamma_val.z) + static_cast<float>(beta_val.z)) * (1.0f + static_cast<float>(scale_val.z)) + static_cast<float>(shift_val.z));
      tmp.w = T(((static_cast<float>(local_val[i].w) - s_mean) * s_variance * static_cast<float>(gamma_val.w) + static_cast<float>(beta_val.w)) * (1.0f + static_cast<float>(scale_val.w)) + static_cast<float>(shift_val.w));
      output[index] = tmp;
    }
  }
}

template <typename T>
static void layernorm_launch_cutlass(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& gamma_opt,
    const c10::optional<torch::Tensor>& beta_opt,
    torch::Tensor& y) {
  const int64_t M = x.size(0);
  const int64_t N = x.size(1);

  const bool has_gamma = gamma_opt.has_value() && gamma_opt->defined();
  const bool has_beta = beta_opt.has_value() && beta_opt->defined();

  const T* gamma_ptr = nullptr;
  const T* beta_ptr = nullptr;

  torch::Tensor gamma_fallback, beta_fallback;
  if (!has_gamma) {
    gamma_fallback = torch::ones({N}, x.options());
    gamma_ptr = reinterpret_cast<const T*>(gamma_fallback.data_ptr());
  } else {
    const auto& g = *gamma_opt;
    gamma_ptr = reinterpret_cast<const T*>(g.data_ptr());
  }
  if (!has_beta) {
    beta_fallback = torch::zeros({N}, x.options());
    beta_ptr = reinterpret_cast<const T*>(beta_fallback.data_ptr());
  } else {
    const auto& b = *beta_opt;
    beta_ptr = reinterpret_cast<const T*>(b.data_ptr());
  }

  dim3 grid((unsigned)M);
  // n must be divisible by 4 for vectorized path
  TORCH_CHECK((N % 4) == 0, "N must be divisible by 4");
  dim3 block(0);

  if (N <= 4096) {
    block.x = (int)((N/4 + 31)/32*32);
    if (block.x > 1024) block.x = 1024;
    if (std::is_same<T, float>::value) {
      layernorm_twoPassAlgo_stored_locally_e4<float4, float, 1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma_ptr, (const float4*)beta_ptr, (int)M, (int)N);
    } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
      layernorm_twoPassAlgo_stored_locally_e4<bf16_4, cutlass::bfloat16_t, 1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma_ptr, (const bf16_4*)beta_ptr, (int)M, (int)N);
    } else {
      layernorm_twoPassAlgo_stored_locally_e4<half4, half, 1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma_ptr, (const half4*)beta_ptr, (int)M, (int)N);
    }
  } else if (N <= 8192) {
    block.x = (int)(((N + 7)/8 + 31)/32*32);
    if (block.x > 1024) block.x = 1024;
    if (std::is_same<T, float>::value) {
      layernorm_twoPassAlgo_stored_locally_e4<float4, float, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma_ptr, (const float4*)beta_ptr, (int)M, (int)N);
    } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
      layernorm_twoPassAlgo_stored_locally_e4<bf16_4, cutlass::bfloat16_t, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma_ptr, (const bf16_4*)beta_ptr, (int)M, (int)N);
    } else {
      layernorm_twoPassAlgo_stored_locally_e4<half4, half, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma_ptr, (const half4*)beta_ptr, (int)M, (int)N);
    }
  } else if (N <= 16384) {
    block.x = (int)(((N + 15)/16 + 31)/32*32);
    if (block.x > 1024) block.x = 1024;
    if (std::is_same<T, float>::value) {
      layernorm_twoPassAlgo_stored_locally_e4<float4, float, 4><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma_ptr, (const float4*)beta_ptr, (int)M, (int)N);
    } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
      layernorm_twoPassAlgo_stored_locally_e4<bf16_4, cutlass::bfloat16_t, 4><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma_ptr, (const bf16_4*)beta_ptr, (int)M, (int)N);
    } else {
      layernorm_twoPassAlgo_stored_locally_e4<half4, half, 4><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma_ptr, (const half4*)beta_ptr, (int)M, (int)N);
    }
  } else if (N <= 32768) {
    block.x = (int)(((N + 31)/32 + 31)/32*32);
    if (block.x > 1024) block.x = 1024;
    if (std::is_same<T, float>::value) {
      layernorm_twoPassAlgo_stored_locally_e4<float4, float, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma_ptr, (const float4*)beta_ptr, (int)M, (int)N);
    } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
      layernorm_twoPassAlgo_stored_locally_e4<bf16_4, cutlass::bfloat16_t, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma_ptr, (const bf16_4*)beta_ptr, (int)M, (int)N);
    } else {
      layernorm_twoPassAlgo_stored_locally_e4<half4, half, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma_ptr, (const half4*)beta_ptr, (int)M, (int)N);
    }
  } else {
    block.x = (int)(((N + 63)/64 + 31)/32*32);
    if (block.x > 1024) block.x = 1024;
    if (std::is_same<T, float>::value) {
      layernorm_twoPassAlgo_stored_locally_e4<float4, float, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma_ptr, (const float4*)beta_ptr, (int)M, (int)N);
    } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
      layernorm_twoPassAlgo_stored_locally_e4<bf16_4, cutlass::bfloat16_t, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma_ptr, (const bf16_4*)beta_ptr, (int)M, (int)N);
    } else {
      layernorm_twoPassAlgo_stored_locally_e4<half4, half, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma_ptr, (const half4*)beta_ptr, (int)M, (int)N);
    }
  }
}

template <typename T>
static void layernorm_fused_scale_shift_launch(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    const torch::Tensor& scale,
    const torch::Tensor& shift,
    torch::Tensor& y) {
  const int64_t M = x.size(0);
  const int64_t N = x.size(1);
  dim3 grid((unsigned)M);
  TORCH_CHECK((N % 4) == 0, "N must be divisible by 4");
  dim3 block(0);
  if (N <= 4096) {
    block.x = (int)((N/4 + 31)/32*32);
    if (block.x > 1024) block.x = 1024;
    if (std::is_same<T, float>::value) {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<float4, float, 1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(),
        (const float4*)scale.data_ptr(), (const float4*)shift.data_ptr(), (int)M, (int)N);
    } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<bf16_4, cutlass::bfloat16_t, 1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(),
        (const bf16_4*)scale.data_ptr(), (const bf16_4*)shift.data_ptr(), (int)M, (int)N);
    } else {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<half4, half, 1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(),
        (const half4*)scale.data_ptr(), (const half4*)shift.data_ptr(), (int)M, (int)N);
    }
  } else if (N <= 8192) {
    block.x = (int)(((N + 7)/8 + 31)/32*32);
    if (block.x > 1024) block.x = 1024;
    if (std::is_same<T, float>::value) {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<float4, float, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(),
        (const float4*)scale.data_ptr(), (const float4*)shift.data_ptr(), (int)M, (int)N);
    } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<bf16_4, cutlass::bfloat16_t, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(),
        (const bf16_4*)scale.data_ptr(), (const bf16_4*)shift.data_ptr(), (int)M, (int)N);
    } else {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<half4, half, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(),
        (const half4*)scale.data_ptr(), (const half4*)shift.data_ptr(), (int)M, (int)N);
    }
  } else if (N <= 16384) {
    block.x = (int)(((N + 15)/16 + 31)/32*32);
    if (block.x > 1024) block.x = 1024;
    if (std::is_same<T, float>::value) {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<float4, float, 4><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(),
        (const float4*)scale.data_ptr(), (const float4*)shift.data_ptr(), (int)M, (int)N);
    } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<bf16_4, cutlass::bfloat16_t, 4><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(),
        (const bf16_4*)scale.data_ptr(), (const bf16_4*)shift.data_ptr(), (int)M, (int)N);
    } else {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<half4, half, 4><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(),
        (const half4*)scale.data_ptr(), (const half4*)shift.data_ptr(), (int)M, (int)N);
    }
  } else if (N <= 32768) {
    block.x = (int)(((N + 31)/32 + 31)/32*32);
    if (block.x > 1024) block.x = 1024;
    if (std::is_same<T, float>::value) {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<float4, float, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(),
        (const float4*)scale.data_ptr(), (const float4*)shift.data_ptr(), (int)M, (int)N);
    } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<bf16_4, cutlass::bfloat16_t, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(),
        (const bf16_4*)scale.data_ptr(), (const bf16_4*)shift.data_ptr(), (int)M, (int)N);
    } else {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<half4, half, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(),
        (const half4*)scale.data_ptr(), (const half4*)shift.data_ptr(), (int)M, (int)N);
    }
  } else {
    block.x = (int)(((N + 63)/64 + 31)/32*32);
    if (block.x > 1024) block.x = 1024;
    if (std::is_same<T, float>::value) {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<float4, float, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(),
        (const float4*)scale.data_ptr(), (const float4*)shift.data_ptr(), (int)M, (int)N);
    } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<bf16_4, cutlass::bfloat16_t, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(),
        (const bf16_4*)scale.data_ptr(), (const bf16_4*)shift.data_ptr(), (int)M, (int)N);
    } else {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<half4, half, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(),
        (const half4*)scale.data_ptr(), (const half4*)shift.data_ptr(), (int)M, (int)N);
    }
  }
}

// Public interfaces (registered in common_extension.cc)
torch::Tensor device_layernorm(torch::Tensor x,
                               const c10::optional<torch::Tensor>& gamma,
                               const c10::optional<torch::Tensor>& beta) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, N]");
  TORCH_CHECK(x.stride(-1) == 1, "last dim of x must be contiguous (stride 1)");
  const int64_t N = x.size(1);
  if (gamma.has_value() && gamma->defined()) {
    TORCH_CHECK(gamma->is_cuda(), "gamma must be CUDA");
    TORCH_CHECK(gamma->dtype() == x.dtype(), "gamma must have same dtype as x");
    TORCH_CHECK(gamma->dim() == 1 && gamma->numel() == N, "gamma must be shape [N]");
    TORCH_CHECK(gamma->stride(0) == 1, "gamma must be contiguous");
  }
  if (beta.has_value() && beta->defined()) {
    TORCH_CHECK(beta->is_cuda(), "beta must be CUDA");
    TORCH_CHECK(beta->dtype() == x.dtype(), "beta must have same dtype as x");
    TORCH_CHECK(beta->dim() == 1 && beta->numel() == N, "beta must be shape [N]");
    TORCH_CHECK(beta->stride(0) == 1, "beta must be contiguous");
  }
  auto y = torch::empty_like(x);
  if (x.dtype() == torch::kFloat32) {
    layernorm_launch_cutlass<float>(x, gamma, beta, y);
  } else if (x.dtype() == torch::kFloat16) {
    layernorm_launch_cutlass<cutlass::half_t>(x, gamma, beta, y);
  } else if (x.dtype() == torch::kBFloat16) {
    layernorm_launch_cutlass<cutlass::bfloat16_t>(x, gamma, beta, y);
  } else {
    TORCH_CHECK(false, "Unsupported dtype. Use float32, float16, or bfloat16.");
  }
  return y;
}

torch::Tensor device_layernorm_fuse_scale_shift(torch::Tensor x,
                                                torch::Tensor gamma,
                                                torch::Tensor beta,
                                                torch::Tensor scale,
                                                torch::Tensor shift) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(scale.is_cuda() && shift.is_cuda(), "scale/shift must be CUDA");
  TORCH_CHECK(x.dim() == 2 && scale.dim() == 2 && shift.dim() == 2, "x/scale/shift must be 2D [M, N]");
  TORCH_CHECK(x.stride(-1) == 1 && scale.stride(-1) == 1 && shift.stride(-1) == 1,
              "last dim of x/scale/shift must be contiguous (stride 1)");
  TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma/beta must be 1D [N]");
  TORCH_CHECK(x.dtype() == scale.dtype() && x.dtype() == shift.dtype(), "x, scale, shift must have same dtype");
  TORCH_CHECK(x.dtype() == gamma.dtype() && x.dtype() == beta.dtype(), "x, gamma, beta must have same dtype");
  const int64_t M = x.size(0);
  const int64_t N = x.size(1);
  TORCH_CHECK(scale.size(0) == M && scale.size(1) == N, "scale must be shape [M, N]");
  TORCH_CHECK(shift.size(0) == M && shift.size(1) == N, "shift must be shape [M, N]");
  TORCH_CHECK(gamma.numel() == N && beta.numel() == N, "gamma/beta must be length N");
  auto y = torch::empty_like(x);
  if (x.dtype() == torch::kFloat32) {
    layernorm_fused_scale_shift_launch<float>(x, gamma, beta, scale, shift, y);
  } else if (x.dtype() == torch::kFloat16) {
    layernorm_fused_scale_shift_launch<cutlass::half_t>(x, gamma, beta, scale, shift, y);
  } else if (x.dtype() == torch::kBFloat16) {
    layernorm_fused_scale_shift_launch<cutlass::bfloat16_t>(x, gamma, beta, scale, shift, y);
  } else {
    TORCH_CHECK(false, "Unsupported dtype. Use float32, float16, or bfloat16.");
  }
  return y;
}


