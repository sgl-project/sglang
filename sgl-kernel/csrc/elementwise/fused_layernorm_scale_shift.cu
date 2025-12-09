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

// -----------------------------
// Launch config helpers (ILP↑)
// -----------------------------
static __forceinline__ int round_up32(int x) {
  return ((x + 31) / 32) * 32;
}
static __forceinline__ int clamp_item_per_thread(int ip_req) {
  if (ip_req <= 1) return 1;
  if (ip_req <= 2) return 2;
  if (ip_req <= 4) return 4;
  if (ip_req <= 8) return 8;
  return 16;
}
static __forceinline__ void compute_block_and_ipt(int N, int& block_x, int& ipt, int M = 1000) {
  // Adaptive strategy based on M (grid size):
  // - Small M (<200): Prioritize block occupancy, use ipt=1 for larger blocks
  // - Large M (>=200): Prioritize ILP, use dynamic ipt for better per-thread work
  const int n4 = N / 4;
  
  if (M < 200) {
    // Small grid: maximize block size for better occupancy per block
    ipt = 1;
    int bx = n4;
    bx = round_up32(bx);
    if (bx < 32) bx = 32;
    if (bx > 1024) bx = 1024;
    block_x = bx;
  } else {
    // Large grid: balance block size and ILP
    const int target_block = 256; // base target; will be rounded to warp-aligned
    int ip_req = (n4 + target_block - 1) / target_block; // how many T4 groups each thread should process
    ipt = clamp_item_per_thread(ip_req);
    int bx = (n4 + ipt - 1) / ipt;
    bx = round_up32(bx);
    if (bx < 32) bx = 32;
    if (bx > 1024) bx = 1024;
    block_x = bx;
  }
}

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

// 4D scale/shift variant: scale/shift shape [B, F, 1, N]
template<typename T4, typename T, int ITEM_PER_THREAD>
__global__ void layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift_4d(
                                                       T4* output,
                                                       const T4* input,
                                                       const T4* gamma,
                                                       const T4* beta,
                                                       const T4* scale4d,
                                                       const T4* shift4d,
                                                       const int m,
                                                       const int n,
                                                       const int B,
                                                       const int F,
                                                       const int frame_seqlen)
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

  // Compute (b, f) indices for this row
  const int rows_per_b = F * frame_seqlen;
  const int b = m_idx / rows_per_b;
  const int s_in_b = m_idx - b * rows_per_b;
  const int f = s_in_b / frame_seqlen;
  const int base4d = (b * F + f) * n_4;

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
      const T4 scale_val = scale4d[base4d + index];
      const T4 shift_val = shift4d[base4d + index];
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
  int ipt = 1;
  int bx = 0;
  compute_block_and_ipt((int)N, bx, ipt, (int)M);  // Pass M for adaptive strategy
  dim3 block(bx);

  if (std::is_same<T, float>::value) {
    switch (ipt) {
      case 1:  layernorm_twoPassAlgo_stored_locally_e4<float4, float, 1 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma_ptr, (const float4*)beta_ptr, (int)M, (int)N); break;
      case 2:  layernorm_twoPassAlgo_stored_locally_e4<float4, float, 2 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma_ptr, (const float4*)beta_ptr, (int)M, (int)N); break;
      case 4:  layernorm_twoPassAlgo_stored_locally_e4<float4, float, 4 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma_ptr, (const float4*)beta_ptr, (int)M, (int)N); break;
      case 8:  layernorm_twoPassAlgo_stored_locally_e4<float4, float, 8 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma_ptr, (const float4*)beta_ptr, (int)M, (int)N); break;
      default: layernorm_twoPassAlgo_stored_locally_e4<float4, float, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma_ptr, (const float4*)beta_ptr, (int)M, (int)N); break;
    }
  } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
    switch (ipt) {
      case 1:  layernorm_twoPassAlgo_stored_locally_e4<bf16_4, cutlass::bfloat16_t, 1 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma_ptr, (const bf16_4*)beta_ptr, (int)M, (int)N); break;
      case 2:  layernorm_twoPassAlgo_stored_locally_e4<bf16_4, cutlass::bfloat16_t, 2 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma_ptr, (const bf16_4*)beta_ptr, (int)M, (int)N); break;
      case 4:  layernorm_twoPassAlgo_stored_locally_e4<bf16_4, cutlass::bfloat16_t, 4 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma_ptr, (const bf16_4*)beta_ptr, (int)M, (int)N); break;
      case 8:  layernorm_twoPassAlgo_stored_locally_e4<bf16_4, cutlass::bfloat16_t, 8 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma_ptr, (const bf16_4*)beta_ptr, (int)M, (int)N); break;
      default: layernorm_twoPassAlgo_stored_locally_e4<bf16_4, cutlass::bfloat16_t, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma_ptr, (const bf16_4*)beta_ptr, (int)M, (int)N); break;
    }
  } else {
    switch (ipt) {
      case 1:  layernorm_twoPassAlgo_stored_locally_e4<half4, half, 1 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma_ptr, (const half4*)beta_ptr, (int)M, (int)N); break;
      case 2:  layernorm_twoPassAlgo_stored_locally_e4<half4, half, 2 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma_ptr, (const half4*)beta_ptr, (int)M, (int)N); break;
      case 4:  layernorm_twoPassAlgo_stored_locally_e4<half4, half, 4 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma_ptr, (const half4*)beta_ptr, (int)M, (int)N); break;
      case 8:  layernorm_twoPassAlgo_stored_locally_e4<half4, half, 8 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma_ptr, (const half4*)beta_ptr, (int)M, (int)N); break;
      default: layernorm_twoPassAlgo_stored_locally_e4<half4, half, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma_ptr, (const half4*)beta_ptr, (int)M, (int)N); break;
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
  const bool use_2d = (scale.dim() == 2 && shift.dim() == 2);
  const bool use_4d = (scale.dim() == 4 && shift.dim() == 4);
  const bool scalar_both = (scale.dim() == 1 && scale.numel() == 1 && shift.dim() == 1 && shift.numel() == 1);
  bool skip = false;
  if (scalar_both) {
    const float s0 = scale.item<float>();
    const float sh0 = shift.item<float>();
    skip = (s0 == 0.0f && sh0 == 0.0f);
  }
  TORCH_CHECK(use_2d || use_4d || skip, "scale/shift must be 2D [M, N], 4D [B, F, 1, N], or scalar zeros to skip");

  // If skipping scale/shift, launch the non-fused LN kernel (no temporary tensors).
  if (skip) {
    int ipt = 1;
    int bx = 0;
    compute_block_and_ipt((int)N, bx, ipt, (int)M);  // Use adaptive strategy based on M
    block.x = bx;
    
    if (std::is_same<T, float>::value) {
      switch (ipt) {
        case 1:  layernorm_twoPassAlgo_stored_locally_e4<float4, float, 1 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(), (int)M, (int)N); break;
        case 2:  layernorm_twoPassAlgo_stored_locally_e4<float4, float, 2 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(), (int)M, (int)N); break;
        case 4:  layernorm_twoPassAlgo_stored_locally_e4<float4, float, 4 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(), (int)M, (int)N); break;
        case 8:  layernorm_twoPassAlgo_stored_locally_e4<float4, float, 8 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(), (int)M, (int)N); break;
        default: layernorm_twoPassAlgo_stored_locally_e4<float4, float, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(), (int)M, (int)N); break;
      }
    } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
      switch (ipt) {
        case 1:  layernorm_twoPassAlgo_stored_locally_e4<bf16_4, cutlass::bfloat16_t, 1 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(), (int)M, (int)N); break;
        case 2:  layernorm_twoPassAlgo_stored_locally_e4<bf16_4, cutlass::bfloat16_t, 2 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(), (int)M, (int)N); break;
        case 4:  layernorm_twoPassAlgo_stored_locally_e4<bf16_4, cutlass::bfloat16_t, 4 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(), (int)M, (int)N); break;
        case 8:  layernorm_twoPassAlgo_stored_locally_e4<bf16_4, cutlass::bfloat16_t, 8 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(), (int)M, (int)N); break;
        default: layernorm_twoPassAlgo_stored_locally_e4<bf16_4, cutlass::bfloat16_t, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(), (int)M, (int)N); break;
      }
    } else {
      switch (ipt) {
        case 1:  layernorm_twoPassAlgo_stored_locally_e4<half4, half, 1 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(), (int)M, (int)N); break;
        case 2:  layernorm_twoPassAlgo_stored_locally_e4<half4, half, 2 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(), (int)M, (int)N); break;
        case 4:  layernorm_twoPassAlgo_stored_locally_e4<half4, half, 4 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(), (int)M, (int)N); break;
        case 8:  layernorm_twoPassAlgo_stored_locally_e4<half4, half, 8 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(), (int)M, (int)N); break;
        default: layernorm_twoPassAlgo_stored_locally_e4<half4, half, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(), (int)M, (int)N); break;
      }
    }
    return;
  }

  if (use_2d) {
    int ipt = 1;
    int bx = 0;
    compute_block_and_ipt((int)N, bx, ipt, (int)M);  // Use adaptive strategy based on M
    block.x = bx;
    
    if (std::is_same<T, float>::value) {
      switch (ipt) {
        case 1:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<float4, float, 1 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(), (const float4*)scale.data_ptr(), (const float4*)shift.data_ptr(), (int)M, (int)N); break;
        case 2:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<float4, float, 2 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(), (const float4*)scale.data_ptr(), (const float4*)shift.data_ptr(), (int)M, (int)N); break;
        case 4:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<float4, float, 4 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(), (const float4*)scale.data_ptr(), (const float4*)shift.data_ptr(), (int)M, (int)N); break;
        case 8:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<float4, float, 8 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(), (const float4*)scale.data_ptr(), (const float4*)shift.data_ptr(), (int)M, (int)N); break;
        default: layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<float4, float, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(), (const float4*)scale.data_ptr(), (const float4*)shift.data_ptr(), (int)M, (int)N); break;
      }
    } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
      switch (ipt) {
        case 1:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<bf16_4, cutlass::bfloat16_t, 1 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(), (const bf16_4*)scale.data_ptr(), (const bf16_4*)shift.data_ptr(), (int)M, (int)N); break;
        case 2:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<bf16_4, cutlass::bfloat16_t, 2 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(), (const bf16_4*)scale.data_ptr(), (const bf16_4*)shift.data_ptr(), (int)M, (int)N); break;
        case 4:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<bf16_4, cutlass::bfloat16_t, 4 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(), (const bf16_4*)scale.data_ptr(), (const bf16_4*)shift.data_ptr(), (int)M, (int)N); break;
        case 8:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<bf16_4, cutlass::bfloat16_t, 8 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(), (const bf16_4*)scale.data_ptr(), (const bf16_4*)shift.data_ptr(), (int)M, (int)N); break;
        default: layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<bf16_4, cutlass::bfloat16_t, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(), (const bf16_4*)scale.data_ptr(), (const bf16_4*)shift.data_ptr(), (int)M, (int)N); break;
      }
    } else {
      switch (ipt) {
        case 1:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<half4, half, 1 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(), (const half4*)scale.data_ptr(), (const half4*)shift.data_ptr(), (int)M, (int)N); break;
        case 2:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<half4, half, 2 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(), (const half4*)scale.data_ptr(), (const half4*)shift.data_ptr(), (int)M, (int)N); break;
        case 4:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<half4, half, 4 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(), (const half4*)scale.data_ptr(), (const half4*)shift.data_ptr(), (int)M, (int)N); break;
        case 8:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<half4, half, 8 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(), (const half4*)scale.data_ptr(), (const half4*)shift.data_ptr(), (int)M, (int)N); break;
        default: layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<half4, half, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(), (const half4*)scale.data_ptr(), (const half4*)shift.data_ptr(), (int)M, (int)N); break;
      }
    }
    return;
  }

  // 4D launcher path
  TORCH_CHECK(scale.size(2) == 1 && shift.size(2) == 1, "scale/shift 4D must have size 1 at dim=2");
  TORCH_CHECK(scale.size(3) == N && shift.size(3) == N, "scale/shift last dim must be N");
  const int64_t B = scale.size(0);
  const int64_t F = scale.size(1);
  TORCH_CHECK((M % (B * F)) == 0, "M must be divisible by B*F for 4D scale/shift");
  const int frame_seqlen = (int)(M / (B * F));

  int ipt = 1;
  int bx = 0;
  compute_block_and_ipt((int)N, bx, ipt, (int)M);  // Use adaptive strategy based on M
  block.x = bx;

  if (std::is_same<T, float>::value) {
    switch (ipt) {
      case 1:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift_4d<float4, float, 1 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(), (const float4*)scale.data_ptr(), (const float4*)shift.data_ptr(), (int)M, (int)N, (int)B, (int)F, (int)frame_seqlen); break;
      case 2:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift_4d<float4, float, 2 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(), (const float4*)scale.data_ptr(), (const float4*)shift.data_ptr(), (int)M, (int)N, (int)B, (int)F, (int)frame_seqlen); break;
      case 4:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift_4d<float4, float, 4 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(), (const float4*)scale.data_ptr(), (const float4*)shift.data_ptr(), (int)M, (int)N, (int)B, (int)F, (int)frame_seqlen); break;
      case 8:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift_4d<float4, float, 8 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(), (const float4*)scale.data_ptr(), (const float4*)shift.data_ptr(), (int)M, (int)N, (int)B, (int)F, (int)frame_seqlen); break;
      default: layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift_4d<float4, float, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((float4*)y.data_ptr(), (const float4*)x.data_ptr(), (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(), (const float4*)scale.data_ptr(), (const float4*)shift.data_ptr(), (int)M, (int)N, (int)B, (int)F, (int)frame_seqlen); break;
    }
  } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
    switch (ipt) {
      case 1:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift_4d<bf16_4, cutlass::bfloat16_t, 1 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(), (const bf16_4*)scale.data_ptr(), (const bf16_4*)shift.data_ptr(), (int)M, (int)N, (int)B, (int)F, (int)frame_seqlen); break;
      case 2:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift_4d<bf16_4, cutlass::bfloat16_t, 2 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(), (const bf16_4*)scale.data_ptr(), (const bf16_4*)shift.data_ptr(), (int)M, (int)N, (int)B, (int)F, (int)frame_seqlen); break;
      case 4:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift_4d<bf16_4, cutlass::bfloat16_t, 4 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(), (const bf16_4*)scale.data_ptr(), (const bf16_4*)shift.data_ptr(), (int)M, (int)N, (int)B, (int)F, (int)frame_seqlen); break;
      case 8:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift_4d<bf16_4, cutlass::bfloat16_t, 8 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(), (const bf16_4*)scale.data_ptr(), (const bf16_4*)shift.data_ptr(), (int)M, (int)N, (int)B, (int)F, (int)frame_seqlen); break;
      default: layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift_4d<bf16_4, cutlass::bfloat16_t, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((bf16_4*)y.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(), (const bf16_4*)scale.data_ptr(), (const bf16_4*)shift.data_ptr(), (int)M, (int)N, (int)B, (int)F, (int)frame_seqlen); break;
    }
  } else {
    switch (ipt) {
      case 1:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift_4d<half4, half, 1 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(), (const half4*)scale.data_ptr(), (const half4*)shift.data_ptr(), (int)M, (int)N, (int)B, (int)F, (int)frame_seqlen); break;
      case 2:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift_4d<half4, half, 2 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(), (const half4*)scale.data_ptr(), (const half4*)shift.data_ptr(), (int)M, (int)N, (int)B, (int)F, (int)frame_seqlen); break;
      case 4:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift_4d<half4, half, 4 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(), (const half4*)scale.data_ptr(), (const half4*)shift.data_ptr(), (int)M, (int)N, (int)B, (int)F, (int)frame_seqlen); break;
      case 8:  layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift_4d<half4, half, 8 ><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(), (const half4*)scale.data_ptr(), (const half4*)shift.data_ptr(), (int)M, (int)N, (int)B, (int)F, (int)frame_seqlen); break;
      default: layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift_4d<half4, half, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>((half4*)y.data_ptr(), (const half4*)x.data_ptr(), (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(), (const half4*)scale.data_ptr(), (const half4*)shift.data_ptr(), (int)M, (int)N, (int)B, (int)F, (int)frame_seqlen); break;
    }
  }
}

// Public interfaces (registered in common_extension.cc)
torch::Tensor device_layernorm(torch::Tensor x,
                               const c10::optional<torch::Tensor>& gamma_opt,
                               const c10::optional<torch::Tensor>& beta_opt) {
  CHECK_CUDA(x);
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, N]");
  TORCH_CHECK(x.stride(-1) == 1, "last dim of x must be contiguous (stride 1)");
  const int64_t N = x.size(1);
  if (gamma_opt.has_value() && gamma_opt->defined()) {
    const auto& gamma = gamma_opt.value();
    TORCH_CHECK(gamma.is_cuda(), "gamma must be on CUDA");
    TORCH_CHECK(gamma_opt->dtype() == x.dtype(), "gamma must have same dtype as x");
    TORCH_CHECK(gamma_opt->dim() == 1 && gamma_opt->numel() == N, "gamma must be shape [N]");
    TORCH_CHECK(gamma_opt->stride(0) == 1, "gamma must be contiguous");
  }
  if (beta_opt.has_value() && beta_opt->defined()) {
    const auto& beta = beta_opt.value();
    TORCH_CHECK(beta.is_cuda(), "beta must be on CUDA");
    TORCH_CHECK(beta_opt->dtype() == x.dtype(), "beta must have same dtype as x");
    TORCH_CHECK(beta_opt->dim() == 1 && beta_opt->numel() == N, "beta must be shape [N]");
    TORCH_CHECK(beta_opt->stride(0) == 1, "beta must be contiguous");
  }
  auto y = torch::empty_like(x);
  if (x.dtype() == torch::kFloat32) {
    layernorm_launch_cutlass<float>(x, gamma_opt, beta_opt, y);
  } else if (x.dtype() == torch::kFloat16) {
    layernorm_launch_cutlass<cutlass::half_t>(x, gamma_opt, beta_opt, y);
  } else if (x.dtype() == torch::kBFloat16) {
    layernorm_launch_cutlass<cutlass::bfloat16_t>(x, gamma_opt, beta_opt, y);
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
  CHECK_CUDA(x);
  CHECK_CUDA(scale);
  CHECK_CUDA(shift);
  CHECK_CUDA(gamma);
  CHECK_CUDA(beta);
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, N]");
  TORCH_CHECK(x.stride(-1) == 1, "last dim of x must be contiguous (stride 1)");
  TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma/beta must be 1D [N]");
  TORCH_CHECK(x.dtype() == scale.dtype() && x.dtype() == shift.dtype(), "x, scale, shift must have same dtype");
  TORCH_CHECK(x.dtype() == gamma.dtype() && x.dtype() == beta.dtype(), "x, gamma, beta must have same dtype");
  const int64_t M = x.size(0);
  const int64_t N = x.size(1);
  if (scale.dim() == 2 && shift.dim() == 2) {
    TORCH_CHECK(scale.size(0) == M && scale.size(1) == N, "scale must be shape [M, N]");
    TORCH_CHECK(shift.size(0) == M && shift.size(1) == N, "shift must be shape [M, N]");
    TORCH_CHECK(scale.stride(-1) == 1 && shift.stride(-1) == 1, "last dim of scale/shift must be contiguous (stride 1)");
  } else if (scale.dim() == 4 && shift.dim() == 4) {
    TORCH_CHECK(scale.size(3) == N && shift.size(3) == N, "scale/shift last dim must be N");
    TORCH_CHECK(scale.size(2) == 1 && shift.size(2) == 1, "scale/shift 4D must have size 1 at dim=2");
    TORCH_CHECK(scale.stride(3) == 1 && shift.stride(3) == 1, "last dim of scale/shift must be contiguous (stride 1)");
    const int64_t B = scale.size(0);
    const int64_t F = scale.size(1);
    TORCH_CHECK((M % (B * F)) == 0, "M must be divisible by B*F for 4D scale/shift");
  } else if (scale.dim() == 1 && scale.numel() == 1 && shift.dim() == 1 && shift.numel() == 1) {
    const float s0 = scale.item<float>();
    const float sh0 = shift.item<float>();
    TORCH_CHECK(s0 == 0.0f && sh0 == 0.0f, "When scale/shift are scalar, both must be 0 to skip");
  } else {
    TORCH_CHECK(false, "scale/shift must be 2D [M, N] or 4D [B, F, 1, N]");
  }
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

// =========================
// Fused Residual + Gate + LayerNorm + Scale/Shift
// =========================

// gate_mode:
// 0: no gate (scalar 1.0), residual_output = residual + x
// 1: 2D gate [M, N]
// 2: Bx1xN gate [B, 1, N]
// 3: BxFx1xN gate [B, F, 1, N]
template<typename T4, typename T, int ITEM_PER_THREAD>
__global__ void layernorm_e4_fused_res_gate_scale_shift_2d(
    T4* __restrict__ output,
    T4* __restrict__ residual_out,
    const T4* __restrict__ x,
    const T4* __restrict__ residual,
    const T4* __restrict__ gamma,
    const T4* __restrict__ beta,
    const T4* __restrict__ scale,
    const T4* __restrict__ shift,
    const T4* __restrict__ gate_mn,   // used when gate_mode == 1
    const T4* __restrict__ gate_b1,   // used when gate_mode == 2 (flattened [B,1,N] -> [B,N])
    const int m,
    const int n,
    const int gate_mode,
    const int rows_per_b // valid when gate_mode == 2
){
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};
  T4 local_val[ITEM_PER_THREAD];
  const int n_4 = n / 4;
  const int offset = m_idx * n_4;

  const T4 zero = {T(0.0f), T(0.0f), T(0.0f), T(0.0f)};

  const int b = (gate_mode == 2) ? (m_idx / rows_per_b) : 0;
  const int gate_b_base = (gate_mode == 2) ? (b * n_4) : 0;

  #pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i * bdimx + tid;
    if (index < n_4){
      const T4 x_v  = x[offset + index];
      const T4 r_v  = residual[offset + index];
      T4 g_v;
      if (gate_mode == 0) {
        g_v = {T(1.0f), T(1.0f), T(1.0f), T(1.0f)};
      } else if (gate_mode == 1) {
        g_v = gate_mn[offset + index];
      } else { // gate_mode == 2
        g_v = gate_b1[gate_b_base + index];
      }
      T4 sum_v;
      sum_v.x = T(static_cast<float>(r_v.x) + static_cast<float>(x_v.x) * static_cast<float>(g_v.x));
      sum_v.y = T(static_cast<float>(r_v.y) + static_cast<float>(x_v.y) * static_cast<float>(g_v.y));
      sum_v.z = T(static_cast<float>(r_v.z) + static_cast<float>(x_v.z) * static_cast<float>(g_v.z));
      sum_v.w = T(static_cast<float>(r_v.w) + static_cast<float>(x_v.w) * static_cast<float>(g_v.w));
      local_val[i] = sum_v;
      if (residual_out != nullptr) {
        residual_out[offset + index] = sum_v;
      }
      local_sums[0] += static_cast<float>(sum_v.x) + static_cast<float>(sum_v.y) +
                       static_cast<float>(sum_v.z) + static_cast<float>(sum_v.w);
    } else {
      local_val[i] = zero;
    }
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
      const T4 scale_val = scale[offset + index];
      const T4 shift_val = shift[offset + index];
      T4 tmp;
      tmp.x = T(((static_cast<float>(local_val[i].x) - s_mean) * s_variance * static_cast<float>(gamma_val.x) + static_cast<float>(beta_val.x)) * (1.0f + static_cast<float>(scale_val.x)) + static_cast<float>(shift_val.x));
      tmp.y = T(((static_cast<float>(local_val[i].y) - s_mean) * s_variance * static_cast<float>(gamma_val.y) + static_cast<float>(beta_val.y)) * (1.0f + static_cast<float>(scale_val.y)) + static_cast<float>(shift_val.y));
      tmp.z = T(((static_cast<float>(local_val[i].z) - s_mean) * s_variance * static_cast<float>(gamma_val.z) + static_cast<float>(beta_val.z)) * (1.0f + static_cast<float>(scale_val.z)) + static_cast<float>(shift_val.z));
      tmp.w = T(((static_cast<float>(local_val[i].w) - s_mean) * s_variance * static_cast<float>(gamma_val.w) + static_cast<float>(beta_val.w)) * (1.0f + static_cast<float>(scale_val.w)) + static_cast<float>(shift_val.w));
      output[offset + index] = tmp;
    }
  }
}

template<typename T4, typename T, int ITEM_PER_THREAD>
__global__ void layernorm_e4_fused_res_gate_scale_shift_4d(
    T4* __restrict__ output,
    T4* __restrict__ residual_out,
    const T4* __restrict__ x,
    const T4* __restrict__ residual,
    const T4* __restrict__ gamma,
    const T4* __restrict__ beta,
    const T4* __restrict__ scale4d,
    const T4* __restrict__ shift4d,
    const T4* __restrict__ gate_mn,  // unused for 4d
    const T4* __restrict__ gate_b1,  // unused for 4d
    const T4* __restrict__ gate4d,   // used when gate_mode == 3
    const int m,
    const int n,
    const int gate_mode, // 0 or 3 expected here
    const int B,
    const int F,
    const int frame_seqlen)
{
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};
  T4 local_val[ITEM_PER_THREAD];
  const int n_4 = n / 4;
  const int offset = m_idx * n_4;

  // Compute (b, f) for this row to index 4D tensors
  const int rows_per_b = F * frame_seqlen;
  const int b = m_idx / rows_per_b;
  const int s_in_b = m_idx - b * rows_per_b;
  const int f = s_in_b / frame_seqlen;
  const int base4d = (b * F + f) * n_4;

  const T4 zero = {T(0.0f), T(0.0f), T(0.0f), T(0.0f)};

  #pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i * bdimx + tid;
    if (index < n_4){
      const T4 x_v  = x[offset + index];
      const T4 r_v  = residual[offset + index];
      T4 g_v;
      if (gate_mode == 0) {
        g_v = {T(1.0f), T(1.0f), T(1.0f), T(1.0f)};
      } else { // gate_mode == 3
        g_v = gate4d[base4d + index];
      }
      T4 sum_v;
      sum_v.x = T(static_cast<float>(r_v.x) + static_cast<float>(x_v.x) * static_cast<float>(g_v.x));
      sum_v.y = T(static_cast<float>(r_v.y) + static_cast<float>(x_v.y) * static_cast<float>(g_v.y));
      sum_v.z = T(static_cast<float>(r_v.z) + static_cast<float>(x_v.z) * static_cast<float>(g_v.z));
      sum_v.w = T(static_cast<float>(r_v.w) + static_cast<float>(x_v.w) * static_cast<float>(g_v.w));
      local_val[i] = sum_v;
      if (residual_out != nullptr) {
        residual_out[offset + index] = sum_v;
      }
      local_sums[0] += static_cast<float>(sum_v.x) + static_cast<float>(sum_v.y) +
                       static_cast<float>(sum_v.z) + static_cast<float>(sum_v.w);
    } else {
      local_val[i] = zero;
    }
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
      const T4 scale_val = scale4d[base4d + index];
      const T4 shift_val = shift4d[base4d + index];
      T4 tmp;
      tmp.x = T(((static_cast<float>(local_val[i].x) - s_mean) * s_variance * static_cast<float>(gamma_val.x) + static_cast<float>(beta_val.x)) * (1.0f + static_cast<float>(scale_val.x)) + static_cast<float>(shift_val.x));
      tmp.y = T(((static_cast<float>(local_val[i].y) - s_mean) * s_variance * static_cast<float>(gamma_val.y) + static_cast<float>(beta_val.y)) * (1.0f + static_cast<float>(scale_val.y)) + static_cast<float>(shift_val.y));
      tmp.z = T(((static_cast<float>(local_val[i].z) - s_mean) * s_variance * static_cast<float>(gamma_val.z) + static_cast<float>(beta_val.z)) * (1.0f + static_cast<float>(scale_val.z)) + static_cast<float>(shift_val.z));
      tmp.w = T(((static_cast<float>(local_val[i].w) - s_mean) * s_variance * static_cast<float>(gamma_val.w) + static_cast<float>(beta_val.w)) * (1.0f + static_cast<float>(scale_val.w)) + static_cast<float>(shift_val.w));
      output[offset + index] = tmp;
    }
  }
}

template <typename T>
static void layernorm_fused_res_gate_scale_shift_launch_with_residual(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    const torch::Tensor& scale,
    const torch::Tensor& shift,
    const c10::optional<torch::Tensor>& gate_opt,
    torch::Tensor& y,
    torch::Tensor& residual_out) {
  const int64_t M = x.size(0);
  const int64_t N = x.size(1);
  dim3 grid((unsigned)M);
  TORCH_CHECK((N % 4) == 0, "N must be divisible by 4");
  
  // Use adaptive strategy based on M
  int ipt = 1;
  int bx = 0;
  compute_block_and_ipt((int)N, bx, ipt, (int)M);  // Pass M for adaptive strategy
  dim3 block(bx);

  const bool use_2d = (scale.dim() == 2 && shift.dim() == 2);
  const bool use_4d = (scale.dim() == 4 && shift.dim() == 4);
  const bool scalar_both = (scale.dim() == 1 && scale.numel() == 1 && shift.dim() == 1 && shift.numel() == 1);
  bool skip = false;
  if (scalar_both) {
    const float s0 = scale.item<float>();
    const float sh0 = shift.item<float>();
    skip = (s0 == 0.0f && sh0 == 0.0f);
  }
  TORCH_CHECK(use_2d || use_4d || skip, "scale/shift must be 2D [M, N], 4D [B, F, 1, N], or scalar zeros to skip");

  // Determine gate mode
  int gate_mode = 0;
  torch::Tensor gate;
  if (gate_opt.has_value() && gate_opt->defined()) {
    gate = *gate_opt;
    TORCH_CHECK(gate.dtype() == x.dtype(), "gate must have same dtype as x");
    if (gate.dim() == 2) {
      TORCH_CHECK(gate.size(0) == M && gate.size(1) == N, "2D gate must be [M, N]");
      gate_mode = 1;
    } else if (gate.dim() == 3) {
      TORCH_CHECK(gate.size(1) == 1 && gate.size(2) == N, "3D gate must be [B, 1, N]");
      const int64_t B = gate.size(0);
      TORCH_CHECK((M % B) == 0, "M must be divisible by B for 3D gate [B,1,N]");
      gate_mode = 2;
    } else if (gate.dim() == 4) {
      TORCH_CHECK(gate.size(2) == 1 && gate.size(3) == N, "4D gate must be [B, F, 1, N]");
      gate_mode = 3;
    } else {
      TORCH_CHECK(false, "Unsupported gate shape. Use [M,N], [B,1,N], or [B,F,1,N]");
    }
  }

  if (use_2d || skip) {
    const int rows_per_b = (gate_mode == 2) ? (int)(M / gate.size(0)) : 0;
    torch::Tensor scale2d = scale;
    torch::Tensor shift2d = shift;
    if (skip) {
      TORCH_CHECK(gate_mode != 3, "When skipping with scalar scale/shift, 4D gate is not supported. Provide 2D/3D gate or 4D scale/shift.");
      scale2d = torch::zeros({M, N}, x.options());
      shift2d = torch::zeros({M, N}, x.options());
    }
    if (std::is_same<T, float>::value) {
      if (N <= 4096) {
        layernorm_e4_fused_res_gate_scale_shift_2d<float4, float, 1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          (float4*)y.data_ptr(), (float4*)residual_out.data_ptr(), (const float4*)x.data_ptr(), (const float4*)residual.data_ptr(),
          (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(),
          (const float4*)scale2d.data_ptr(), (const float4*)shift2d.data_ptr(),
          (gate_mode == 1) ? (const float4*)gate.data_ptr() : nullptr,
          (gate_mode == 2) ? (const float4*)gate.data_ptr() : nullptr,
          (int)M, (int)N, gate_mode, rows_per_b);
      } else if (N <= 8192) {
        layernorm_e4_fused_res_gate_scale_shift_2d<float4, float, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          (float4*)y.data_ptr(), (float4*)residual_out.data_ptr(), (const float4*)x.data_ptr(), (const float4*)residual.data_ptr(),
          (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(),
          (const float4*)scale2d.data_ptr(), (const float4*)shift2d.data_ptr(),
          (gate_mode == 1) ? (const float4*)gate.data_ptr() : nullptr,
          (gate_mode == 2) ? (const float4*)gate.data_ptr() : nullptr,
          (int)M, (int)N, gate_mode, rows_per_b);
      } else if (N <= 16384) {
        layernorm_e4_fused_res_gate_scale_shift_2d<float4, float, 4><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          (float4*)y.data_ptr(), (float4*)residual_out.data_ptr(), (const float4*)x.data_ptr(), (const float4*)residual.data_ptr(),
          (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(),
          (const float4*)scale2d.data_ptr(), (const float4*)shift2d.data_ptr(),
          (gate_mode == 1) ? (const float4*)gate.data_ptr() : nullptr,
          (gate_mode == 2) ? (const float4*)gate.data_ptr() : nullptr,
          (int)M, (int)N, gate_mode, rows_per_b);
      } else if (N <= 32768) {
        layernorm_e4_fused_res_gate_scale_shift_2d<float4, float, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          (float4*)y.data_ptr(), (float4*)residual_out.data_ptr(), (const float4*)x.data_ptr(), (const float4*)residual.data_ptr(),
          (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(),
          (const float4*)scale2d.data_ptr(), (const float4*)shift2d.data_ptr(),
          (gate_mode == 1) ? (const float4*)gate.data_ptr() : nullptr,
          (gate_mode == 2) ? (const float4*)gate.data_ptr() : nullptr,
          (int)M, (int)N, gate_mode, rows_per_b);
      } else {
        layernorm_e4_fused_res_gate_scale_shift_2d<float4, float, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          (float4*)y.data_ptr(), (float4*)residual_out.data_ptr(), (const float4*)x.data_ptr(), (const float4*)residual.data_ptr(),
          (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(),
          (const float4*)scale2d.data_ptr(), (const float4*)shift2d.data_ptr(),
          (gate_mode == 1) ? (const float4*)gate.data_ptr() : nullptr,
          (gate_mode == 2) ? (const float4*)gate.data_ptr() : nullptr,
          (int)M, (int)N, gate_mode, rows_per_b);
      }
    } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
      if (N <= 4096) {
        layernorm_e4_fused_res_gate_scale_shift_2d<bf16_4, cutlass::bfloat16_t, 1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          (bf16_4*)y.data_ptr(), (bf16_4*)residual_out.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)residual.data_ptr(),
          (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(),
          (const bf16_4*)scale2d.data_ptr(), (const bf16_4*)shift2d.data_ptr(),
          (gate_mode == 1) ? (const bf16_4*)gate.data_ptr() : nullptr,
          (gate_mode == 2) ? (const bf16_4*)gate.data_ptr() : nullptr,
          (int)M, (int)N, gate_mode, rows_per_b);
      } else if (N <= 8192) {
        layernorm_e4_fused_res_gate_scale_shift_2d<bf16_4, cutlass::bfloat16_t, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          (bf16_4*)y.data_ptr(), (bf16_4*)residual_out.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)residual.data_ptr(),
          (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(),
          (const bf16_4*)scale2d.data_ptr(), (const bf16_4*)shift2d.data_ptr(),
          (gate_mode == 1) ? (const bf16_4*)gate.data_ptr() : nullptr,
          (gate_mode == 2) ? (const bf16_4*)gate.data_ptr() : nullptr,
          (int)M, (int)N, gate_mode, rows_per_b);
      } else if (N <= 16384) {
        layernorm_e4_fused_res_gate_scale_shift_2d<bf16_4, cutlass::bfloat16_t, 4><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          (bf16_4*)y.data_ptr(), (bf16_4*)residual_out.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)residual.data_ptr(),
          (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(),
          (const bf16_4*)scale2d.data_ptr(), (const bf16_4*)shift2d.data_ptr(),
          (gate_mode == 1) ? (const bf16_4*)gate.data_ptr() : nullptr,
          (gate_mode == 2) ? (const bf16_4*)gate.data_ptr() : nullptr,
          (int)M, (int)N, gate_mode, rows_per_b);
      } else if (N <= 32768) {
        layernorm_e4_fused_res_gate_scale_shift_2d<bf16_4, cutlass::bfloat16_t, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          (bf16_4*)y.data_ptr(), (bf16_4*)residual_out.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)residual.data_ptr(),
          (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(),
          (const bf16_4*)scale2d.data_ptr(), (const bf16_4*)shift2d.data_ptr(),
          (gate_mode == 1) ? (const bf16_4*)gate.data_ptr() : nullptr,
          (gate_mode == 2) ? (const bf16_4*)gate.data_ptr() : nullptr,
          (int)M, (int)N, gate_mode, rows_per_b);
      } else {
        layernorm_e4_fused_res_gate_scale_shift_2d<bf16_4, cutlass::bfloat16_t, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          (bf16_4*)y.data_ptr(), (bf16_4*)residual_out.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)residual.data_ptr(),
          (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(),
          (const bf16_4*)scale2d.data_ptr(), (const bf16_4*)shift2d.data_ptr(),
          (gate_mode == 1) ? (const bf16_4*)gate.data_ptr() : nullptr,
          (gate_mode == 2) ? (const bf16_4*)gate.data_ptr() : nullptr,
          (int)M, (int)N, gate_mode, rows_per_b);
      }
    } else {
      if (N <= 4096) {
        layernorm_e4_fused_res_gate_scale_shift_2d<half4, half, 1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          (half4*)y.data_ptr(), (half4*)residual_out.data_ptr(), (const half4*)x.data_ptr(), (const half4*)residual.data_ptr(),
          (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(),
          (const half4*)scale2d.data_ptr(), (const half4*)shift2d.data_ptr(),
          (gate_mode == 1) ? (const half4*)gate.data_ptr() : nullptr,
          (gate_mode == 2) ? (const half4*)gate.data_ptr() : nullptr,
          (int)M, (int)N, gate_mode, rows_per_b);
      } else if (N <= 8192) {
        layernorm_e4_fused_res_gate_scale_shift_2d<half4, half, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          (half4*)y.data_ptr(), (half4*)residual_out.data_ptr(), (const half4*)x.data_ptr(), (const half4*)residual.data_ptr(),
          (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(),
          (const half4*)scale2d.data_ptr(), (const half4*)shift2d.data_ptr(),
          (gate_mode == 1) ? (const half4*)gate.data_ptr() : nullptr,
          (gate_mode == 2) ? (const half4*)gate.data_ptr() : nullptr,
          (int)M, (int)N, gate_mode, rows_per_b);
      } else if (N <= 16384) {
        layernorm_e4_fused_res_gate_scale_shift_2d<half4, half, 4><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          (half4*)y.data_ptr(), (half4*)residual_out.data_ptr(), (const half4*)x.data_ptr(), (const half4*)residual.data_ptr(),
          (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(),
          (const half4*)scale2d.data_ptr(), (const half4*)shift2d.data_ptr(),
          (gate_mode == 1) ? (const half4*)gate.data_ptr() : nullptr,
          (gate_mode == 2) ? (const half4*)gate.data_ptr() : nullptr,
          (int)M, (int)N, gate_mode, rows_per_b);
      } else if (N <= 32768) {
        layernorm_e4_fused_res_gate_scale_shift_2d<half4, half, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          (half4*)y.data_ptr(), (half4*)residual_out.data_ptr(), (const half4*)x.data_ptr(), (const half4*)residual.data_ptr(),
          (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(),
          (const half4*)scale2d.data_ptr(), (const half4*)shift2d.data_ptr(),
          (gate_mode == 1) ? (const half4*)gate.data_ptr() : nullptr,
          (gate_mode == 2) ? (const half4*)gate.data_ptr() : nullptr,
          (int)M, (int)N, gate_mode, rows_per_b);
      } else {
        layernorm_e4_fused_res_gate_scale_shift_2d<half4, half, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          (half4*)y.data_ptr(), (half4*)residual_out.data_ptr(), (const half4*)x.data_ptr(), (const half4*)residual.data_ptr(),
          (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(),
          (const half4*)scale2d.data_ptr(), (const half4*)shift2d.data_ptr(),
          (gate_mode == 1) ? (const half4*)gate.data_ptr() : nullptr,
          (gate_mode == 2) ? (const half4*)gate.data_ptr() : nullptr,
          (int)M, (int)N, gate_mode, rows_per_b);
      }
    }
    return;
  }

  // 4D path with residual_out
  const int64_t B = scale.size(0);
  const int64_t F = scale.size(1);
  const int frame_seqlen = (int)(M / (B * F));
  TORCH_CHECK(gate_mode == 0 || gate_mode == 3, "When scale/shift are 4D, gate must be none or 4D [B,F,1,N]");

  if (std::is_same<T, float>::value) {
    if (N <= 4096) {
      layernorm_e4_fused_res_gate_scale_shift_4d<float4, float, 1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (float4*)y.data_ptr(), (float4*)residual_out.data_ptr(), (const float4*)x.data_ptr(), (const float4*)residual.data_ptr(),
        (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(),
        (const float4*)scale.data_ptr(), (const float4*)shift.data_ptr(),
        nullptr, nullptr, (gate_mode == 3) ? (const float4*)gate.data_ptr() : nullptr,
        (int)M, (int)N, gate_mode, (int)B, (int)F, frame_seqlen);
    } else if (N <= 8192) {
      layernorm_e4_fused_res_gate_scale_shift_4d<float4, float, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (float4*)y.data_ptr(), (float4*)residual_out.data_ptr(), (const float4*)x.data_ptr(), (const float4*)residual.data_ptr(),
        (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(),
        (const float4*)scale.data_ptr(), (const float4*)shift.data_ptr(),
        nullptr, nullptr, (gate_mode == 3) ? (const float4*)gate.data_ptr() : nullptr,
        (int)M, (int)N, gate_mode, (int)B, (int)F, frame_seqlen);
    } else if (N <= 16384) {
      layernorm_e4_fused_res_gate_scale_shift_4d<float4, float, 4><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (float4*)y.data_ptr(), (float4*)residual_out.data_ptr(), (const float4*)x.data_ptr(), (const float4*)residual.data_ptr(),
        (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(),
        (const float4*)scale.data_ptr(), (const float4*)shift.data_ptr(),
        nullptr, nullptr, (gate_mode == 3) ? (const float4*)gate.data_ptr() : nullptr,
        (int)M, (int)N, gate_mode, (int)B, (int)F, frame_seqlen);
    } else if (N <= 32768) {
      layernorm_e4_fused_res_gate_scale_shift_4d<float4, float, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (float4*)y.data_ptr(), (float4*)residual_out.data_ptr(), (const float4*)x.data_ptr(), (const float4*)residual.data_ptr(),
        (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(),
        (const float4*)scale.data_ptr(), (const float4*)shift.data_ptr(),
        nullptr, nullptr, (gate_mode == 3) ? (const float4*)gate.data_ptr() : nullptr,
        (int)M, (int)N, gate_mode, (int)B, (int)F, frame_seqlen);
    } else {
      layernorm_e4_fused_res_gate_scale_shift_4d<float4, float, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (float4*)y.data_ptr(), (float4*)residual_out.data_ptr(), (const float4*)x.data_ptr(), (const float4*)residual.data_ptr(),
        (const float4*)gamma.data_ptr(), (const float4*)beta.data_ptr(),
        (const float4*)scale.data_ptr(), (const float4*)shift.data_ptr(),
        nullptr, nullptr, (gate_mode == 3) ? (const float4*)gate.data_ptr() : nullptr,
        (int)M, (int)N, gate_mode, (int)B, (int)F, frame_seqlen);
    }
  } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
    if (N <= 4096) {
      layernorm_e4_fused_res_gate_scale_shift_4d<bf16_4, cutlass::bfloat16_t, 1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (bf16_4*)y.data_ptr(), (bf16_4*)residual_out.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)residual.data_ptr(),
        (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(),
        (const bf16_4*)scale.data_ptr(), (const bf16_4*)shift.data_ptr(),
        nullptr, nullptr, (gate_mode == 3) ? (const bf16_4*)gate.data_ptr() : nullptr,
        (int)M, (int)N, gate_mode, (int)B, (int)F, frame_seqlen);
    } else if (N <= 8192) {
      layernorm_e4_fused_res_gate_scale_shift_4d<bf16_4, cutlass::bfloat16_t, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (bf16_4*)y.data_ptr(), (bf16_4*)residual_out.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)residual.data_ptr(),
        (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(),
        (const bf16_4*)scale.data_ptr(), (const bf16_4*)shift.data_ptr(),
        nullptr, nullptr, (gate_mode == 3) ? (const bf16_4*)gate.data_ptr() : nullptr,
        (int)M, (int)N, gate_mode, (int)B, (int)F, frame_seqlen);
    } else if (N <= 16384) {
      layernorm_e4_fused_res_gate_scale_shift_4d<bf16_4, cutlass::bfloat16_t, 4><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (bf16_4*)y.data_ptr(), (bf16_4*)residual_out.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)residual.data_ptr(),
        (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(),
        (const bf16_4*)scale.data_ptr(), (const bf16_4*)shift.data_ptr(),
        nullptr, nullptr, (gate_mode == 3) ? (const bf16_4*)gate.data_ptr() : nullptr,
        (int)M, (int)N, gate_mode, (int)B, (int)F, frame_seqlen);
    } else if (N <= 32768) {
      layernorm_e4_fused_res_gate_scale_shift_4d<bf16_4, cutlass::bfloat16_t, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (bf16_4*)y.data_ptr(), (bf16_4*)residual_out.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)residual.data_ptr(),
        (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(),
        (const bf16_4*)scale.data_ptr(), (const bf16_4*)shift.data_ptr(),
        nullptr, nullptr, (gate_mode == 3) ? (const bf16_4*)gate.data_ptr() : nullptr,
        (int)M, (int)N, gate_mode, (int)B, (int)F, frame_seqlen);
    } else {
      layernorm_e4_fused_res_gate_scale_shift_4d<bf16_4, cutlass::bfloat16_t, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (bf16_4*)y.data_ptr(), (bf16_4*)residual_out.data_ptr(), (const bf16_4*)x.data_ptr(), (const bf16_4*)residual.data_ptr(),
        (const bf16_4*)gamma.data_ptr(), (const bf16_4*)beta.data_ptr(),
        (const bf16_4*)scale.data_ptr(), (const bf16_4*)shift.data_ptr(),
        nullptr, nullptr, (gate_mode == 3) ? (const bf16_4*)gate.data_ptr() : nullptr,
        (int)M, (int)N, gate_mode, (int)B, (int)F, frame_seqlen);
    }
  } else {
    if (N <= 4096) {
      layernorm_e4_fused_res_gate_scale_shift_4d<half4, half, 1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (half4*)y.data_ptr(), (half4*)residual_out.data_ptr(), (const half4*)x.data_ptr(), (const half4*)residual.data_ptr(),
        (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(),
        (const half4*)scale.data_ptr(), (const half4*)shift.data_ptr(),
        nullptr, nullptr, (gate_mode == 3) ? (const half4*)gate.data_ptr() : nullptr,
        (int)M, (int)N, gate_mode, (int)B, (int)F, frame_seqlen);
    } else if (N <= 8192) {
      layernorm_e4_fused_res_gate_scale_shift_4d<half4, half, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (half4*)y.data_ptr(), (half4*)residual_out.data_ptr(), (const half4*)x.data_ptr(), (const half4*)residual.data_ptr(),
        (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(),
        (const half4*)scale.data_ptr(), (const half4*)shift.data_ptr(),
        nullptr, nullptr, (gate_mode == 3) ? (const half4*)gate.data_ptr() : nullptr,
        (int)M, (int)N, gate_mode, (int)B, (int)F, frame_seqlen);
    } else if (N <= 16384) {
      layernorm_e4_fused_res_gate_scale_shift_4d<half4, half, 4><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (half4*)y.data_ptr(), (half4*)residual_out.data_ptr(), (const half4*)x.data_ptr(), (const half4*)residual.data_ptr(),
        (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(),
        (const half4*)scale.data_ptr(), (const half4*)shift.data_ptr(),
        nullptr, nullptr, (gate_mode == 3) ? (const half4*)gate.data_ptr() : nullptr,
        (int)M, (int)N, gate_mode, (int)B, (int)F, frame_seqlen);
    } else if (N <= 32768) {
      layernorm_e4_fused_res_gate_scale_shift_4d<half4, half, 8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (half4*)y.data_ptr(), (half4*)residual_out.data_ptr(), (const half4*)x.data_ptr(), (const half4*)residual.data_ptr(),
        (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(),
        (const half4*)scale.data_ptr(), (const half4*)shift.data_ptr(),
        nullptr, nullptr, (gate_mode == 3) ? (const half4*)gate.data_ptr() : nullptr,
        (int)M, (int)N, gate_mode, (int)B, (int)F, frame_seqlen);
    } else {
      layernorm_e4_fused_res_gate_scale_shift_4d<half4, half, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (half4*)y.data_ptr(), (half4*)residual_out.data_ptr(), (const half4*)x.data_ptr(), (const half4*)residual.data_ptr(),
        (const half4*)gamma.data_ptr(), (const half4*)beta.data_ptr(),
        (const half4*)scale.data_ptr(), (const half4*)shift.data_ptr(),
        nullptr, nullptr, (gate_mode == 3) ? (const half4*)gate.data_ptr() : nullptr,
        (int)M, (int)N, gate_mode, (int)B, (int)F, frame_seqlen);
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor> device_scale_residual_layernorm_fuse_scale_shift(torch::Tensor residual,
                                                                                                        torch::Tensor x,
                                                                                                        torch::Tensor gamma,
                                                                                                        torch::Tensor beta,
                                                                                                        torch::Tensor scale,
                                                                                                        torch::Tensor shift,
                                                                                                        const c10::optional<torch::Tensor>& gate_opt) {
  CHECK_CUDA(x);
  CHECK_CUDA(residual);
  CHECK_CUDA(scale);
  CHECK_CUDA(shift);
  CHECK_CUDA(gamma);
  CHECK_CUDA(beta);
  TORCH_CHECK(x.dim() == 2 && residual.dim() == 2, "x and residual must be 2D [M, N]");
  TORCH_CHECK(x.stride(-1) == 1 && residual.stride(-1) == 1, "last dim of x/residual must be contiguous (stride 1)");
  TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma/beta must be 1D [N]");
  TORCH_CHECK(x.dtype() == residual.dtype(), "x and residual must have same dtype");
  TORCH_CHECK(x.size(0) == residual.size(0) && x.size(1) == residual.size(1), "x and residual shapes must match");
  TORCH_CHECK(x.dtype() == scale.dtype() && x.dtype() == shift.dtype(), "x, scale, shift must have same dtype");
  TORCH_CHECK(x.dtype() == gamma.dtype() && x.dtype() == beta.dtype(), "x, gamma, beta must have same dtype");
  const int64_t M = x.size(0);
  const int64_t N = x.size(1);
  if (scale.dim() == 2 && shift.dim() == 2) {
    TORCH_CHECK(scale.size(0) == M && scale.size(1) == N, "scale must be shape [M, N]");
    TORCH_CHECK(shift.size(0) == M && shift.size(1) == N, "shift must be shape [M, N]");
    TORCH_CHECK(scale.stride(-1) == 1 && shift.stride(-1) == 1, "last dim of scale/shift must be contiguous (stride 1)");
  } else if (scale.dim() == 4 && shift.dim() == 4) {
    TORCH_CHECK(scale.size(3) == N && shift.size(3) == N, "scale/shift last dim must be N");
    TORCH_CHECK(scale.size(2) == 1 && shift.size(2) == 1, "scale/shift 4D must have size 1 at dim=2");
    TORCH_CHECK(scale.stride(3) == 1 && shift.stride(3) == 1, "last dim of scale/shift must be contiguous (stride 1)");
    const int64_t B = scale.size(0);
    const int64_t F = scale.size(1);
    TORCH_CHECK((M % (B * F)) == 0, "M must be divisible by B*F for 4D scale/shift");
  } else if (scale.dim() == 1 && scale.numel() == 1 && shift.dim() == 1 && shift.numel() == 1) {
    const float s0 = scale.item<float>();
    const float sh0 = shift.item<float>();
    TORCH_CHECK(s0 == 0.0f && sh0 == 0.0f, "When scale/shift are scalar, both must be 0 to skip");
  } else {
    TORCH_CHECK(false, "scale/shift must be 2D [M, N] or 4D [B, F, 1, N]");
  }
  if (gate_opt.has_value() && gate_opt->defined()) {
    const auto& gate = *gate_opt;
    CHECK_CUDA(gate);
    TORCH_CHECK(gate.dtype() == x.dtype(), "gate must have same dtype as x");
    if (gate.dim() == 2) {
      TORCH_CHECK(gate.size(0) == M && gate.size(1) == N, "2D gate must be [M, N]");
      TORCH_CHECK(gate.stride(-1) == 1, "last dim of gate must be contiguous (stride 1)");
    } else if (gate.dim() == 3) {
      TORCH_CHECK(gate.size(1) == 1 && gate.size(2) == N, "3D gate must be [B, 1, N]");
      TORCH_CHECK((M % gate.size(0)) == 0, "M must be divisible by B for 3D gate [B,1,N]");
      TORCH_CHECK(gate.stride(2) == 1, "last dim of 3D gate must be contiguous (stride 1)");
    } else if (gate.dim() == 4) {
      TORCH_CHECK(gate.size(2) == 1 && gate.size(3) == N, "4D gate must be [B, F, 1, N]");
      TORCH_CHECK(gate.stride(3) == 1, "last dim of 4D gate must be contiguous (stride 1)");
      const int64_t B = gate.size(0);
      const int64_t F = gate.size(1);
      TORCH_CHECK((M % (B * F)) == 0, "M must be divisible by B*F for 4D gate");
      if (scale.dim() == 4) {
        TORCH_CHECK(scale.size(0) == B && scale.size(1) == F, "gate [B,F,1,N] must match scale/shift [B,F,1,N]");
      }
    } else {
      TORCH_CHECK(false, "Unsupported gate shape. Use [M,N], [B,1,N], or [B,F,1,N]");
    }
  }
  TORCH_CHECK(gamma.numel() == N && beta.numel() == N, "gamma/beta must be length N");

  auto y = torch::empty_like(x);
  auto residual_output = torch::empty_like(x);
  if (x.dtype() == torch::kFloat32) {
    layernorm_fused_res_gate_scale_shift_launch_with_residual<float>(x, residual, gamma, beta, scale, shift, gate_opt, y, residual_output);
  } else if (x.dtype() == torch::kFloat16) {
    layernorm_fused_res_gate_scale_shift_launch_with_residual<cutlass::half_t>(x, residual, gamma, beta, scale, shift, gate_opt, y, residual_output);
  } else if (x.dtype() == torch::kBFloat16) {
    layernorm_fused_res_gate_scale_shift_launch_with_residual<cutlass::bfloat16_t>(x, residual, gamma, beta, scale, shift, gate_opt, y, residual_output);
  } else {
    TORCH_CHECK(false, "Unsupported dtype. Use float32, float16, or bfloat16.");
  }
  return std::make_tuple(y, residual_output);
}
