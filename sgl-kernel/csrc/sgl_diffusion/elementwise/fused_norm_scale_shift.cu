/* Copyright 2025 SGLang Team. */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

#include <cassert>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/tensor_ref.h"
#include "utils.h"

enum NormType : int {
  LayerNorm = 0,
  RMSNorm = 1,
};

template <NormType NT>
struct NormTag {
  static constexpr NormType value = NT;
};

template <int V>
struct ItemPerThreadTag {
  static constexpr int value = V;
};

template <typename T4_, typename T_>
struct DTypeTag {
  using T4 = T4_;
  using T = T_;
};

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
  __shared__ T shared[32][NumVals];  // up to 32 warps (1024 threads)
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
    for (int i = 0; i < NumVals; ++i)
      acc[i] = T(0);
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
    for (int i = 0; i < NumVals; ++i)
      vals[i] = acc[i];
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

template <typename T4, typename T, int ITEM_PER_THREAD, NormType norm_type>
__global__ void norm_twoPassAlgo_stored_locally_e4(
    T4* output, const T4* input, const T4* gamma, const T4* beta, const int m, const int n, bool affine, float eps) {
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
    if constexpr (norm_type == NormType::LayerNorm) {
      local_sums[0] += static_cast<float>(local_val[i].x) + static_cast<float>(local_val[i].y) +
                       static_cast<float>(local_val[i].z) + static_cast<float>(local_val[i].w);
    } else {
      local_sums[0] += static_cast<float>(local_val[i].x) * static_cast<float>(local_val[i].x) +
                       static_cast<float>(local_val[i].y) * static_cast<float>(local_val[i].y) +
                       static_cast<float>(local_val[i].z) * static_cast<float>(local_val[i].z) +
                       static_cast<float>(local_val[i].w) * static_cast<float>(local_val[i].w);
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

  if constexpr (norm_type == NormType::LayerNorm) {
    local_sums[0] = 0.0f;
#pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
      const int index = i * bdimx + tid;
      if (index < n_4) {
        const float4 tmp = {
            static_cast<float>(local_val[i].x) - s_mean,
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
  }
  if (threadIdx.x == 0) {
    // In rms norm, s_variance represents rsqrtf(x^2/n+eps).
    s_variance = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i * bdimx + tid;
    if (index < n_4) {
      if constexpr (norm_type == NormType::LayerNorm) {
        const T4 gamma_val = affine ? gamma[index] : T4{T(1.0f), T(1.0f), T(1.0f), T(1.0f)};
        const T4 beta_val = affine ? beta[index] : T4{T(0.0f), T(0.0f), T(0.0f), T(0.0f)};
        T4 tmp;
        tmp.x =
            T((static_cast<float>(local_val[i].x) - s_mean) * s_variance * static_cast<float>(gamma_val.x) +
              static_cast<float>(beta_val.x));
        tmp.y =
            T((static_cast<float>(local_val[i].y) - s_mean) * s_variance * static_cast<float>(gamma_val.y) +
              static_cast<float>(beta_val.y));
        tmp.z =
            T((static_cast<float>(local_val[i].z) - s_mean) * s_variance * static_cast<float>(gamma_val.z) +
              static_cast<float>(beta_val.z));
        tmp.w =
            T((static_cast<float>(local_val[i].w) - s_mean) * s_variance * static_cast<float>(gamma_val.w) +
              static_cast<float>(beta_val.w));
        output[index] = tmp;
      } else {
        const T4 gamma_val = affine ? gamma[index] : T4{T(1.0f), T(1.0f), T(1.0f), T(1.0f)};
        T4 tmp;
        tmp.x = T(static_cast<float>(local_val[i].x) * s_variance * static_cast<float>(gamma_val.x));
        tmp.y = T(static_cast<float>(local_val[i].y) * s_variance * static_cast<float>(gamma_val.y));
        tmp.z = T(static_cast<float>(local_val[i].z) * s_variance * static_cast<float>(gamma_val.z));
        tmp.w = T(static_cast<float>(local_val[i].w) * s_variance * static_cast<float>(gamma_val.w));
        output[index] = tmp;
      }
    }
  }
}

template <typename T4, typename T, int ITEM_PER_THREAD, NormType norm_type>
__global__ void norm_twoPassAlgo_stored_locally_e4_fused_scale_shift(
    T4* output,
    const T4* input,
    const T4* gamma,
    const T4* beta,
    const T4* scale,
    const T4* shift,
    const int m,
    const int n,
    bool affine,
    bool is_scale_c_1,
    bool is_shift_c_1,
    float eps) {
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
  if (!is_scale_c_1) scale += offset;
  if (!is_shift_c_1) shift += offset;

  const T4 zero = {T(0.0f), T(0.0f), T(0.0f), T(0.0f)};
#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i * bdimx + tid;
    local_val[i] = index < n_4 ? input[index] : zero;
    if constexpr (norm_type == NormType::LayerNorm) {
      local_sums[0] += static_cast<float>(local_val[i].x) + static_cast<float>(local_val[i].y) +
                       static_cast<float>(local_val[i].z) + static_cast<float>(local_val[i].w);
    } else {
      local_sums[0] += static_cast<float>(local_val[i].x) * static_cast<float>(local_val[i].x) +
                       static_cast<float>(local_val[i].y) * static_cast<float>(local_val[i].y) +
                       static_cast<float>(local_val[i].z) * static_cast<float>(local_val[i].z) +
                       static_cast<float>(local_val[i].w) * static_cast<float>(local_val[i].w);
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

  if constexpr (norm_type == NormType::LayerNorm) {
    local_sums[0] = 0.0f;
#pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
      const int index = i * bdimx + tid;
      if (index < n_4) {
        const float4 tmp = {
            static_cast<float>(local_val[i].x) - s_mean,
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
  }
  if (threadIdx.x == 0) {
    // In rms norm, s_variance represents rsqrtf(x^2/n+eps).
    s_variance = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i * bdimx + tid;
    if (index < n_4) {
      if constexpr (norm_type == NormType::LayerNorm) {
        const T4 gamma_val = affine ? gamma[index] : T4{T(1.0f), T(1.0f), T(1.0f), T(1.0f)};
        const T4 beta_val = affine ? beta[index] : T4{T(0.0f), T(0.0f), T(0.0f), T(0.0f)};
        const T4 scale_val = scale[index];
        const T4 shift_val = shift[index];
        T4 tmp;
        tmp.x =
            T(((static_cast<float>(local_val[i].x) - s_mean) * s_variance * static_cast<float>(gamma_val.x) +
               static_cast<float>(beta_val.x)) *
                  (1.0f + static_cast<float>(scale_val.x)) +
              static_cast<float>(shift_val.x));
        tmp.y =
            T(((static_cast<float>(local_val[i].y) - s_mean) * s_variance * static_cast<float>(gamma_val.y) +
               static_cast<float>(beta_val.y)) *
                  (1.0f + static_cast<float>(scale_val.y)) +
              static_cast<float>(shift_val.y));
        tmp.z =
            T(((static_cast<float>(local_val[i].z) - s_mean) * s_variance * static_cast<float>(gamma_val.z) +
               static_cast<float>(beta_val.z)) *
                  (1.0f + static_cast<float>(scale_val.z)) +
              static_cast<float>(shift_val.z));
        tmp.w =
            T(((static_cast<float>(local_val[i].w) - s_mean) * s_variance * static_cast<float>(gamma_val.w) +
               static_cast<float>(beta_val.w)) *
                  (1.0f + static_cast<float>(scale_val.w)) +
              static_cast<float>(shift_val.w));
        output[index] = tmp;
      } else {
        const T4 gamma_val = affine ? gamma[index] : T4{T(1.0f), T(1.0f), T(1.0f), T(1.0f)};
        const T4 scale_val = scale[index];
        const T4 shift_val = shift[index];
        T4 tmp;
        tmp.x =
            T((static_cast<float>(local_val[i].x) * s_variance * static_cast<float>(gamma_val.x)) *
                  (1.0f + static_cast<float>(scale_val.x)) +
              static_cast<float>(shift_val.x));
        tmp.y =
            T((static_cast<float>(local_val[i].y) * s_variance * static_cast<float>(gamma_val.y)) *
                  (1.0f + static_cast<float>(scale_val.y)) +
              static_cast<float>(shift_val.y));
        tmp.z =
            T((static_cast<float>(local_val[i].z) * s_variance * static_cast<float>(gamma_val.z)) *
                  (1.0f + static_cast<float>(scale_val.z)) +
              static_cast<float>(shift_val.z));
        tmp.w =
            T((static_cast<float>(local_val[i].w) * s_variance * static_cast<float>(gamma_val.w)) *
                  (1.0f + static_cast<float>(scale_val.w)) +
              static_cast<float>(shift_val.w));
        output[index] = tmp;
      }
    }
  }
}

// 4D scale/shift variant: scale/shift shape [B, F, 1, N]
template <typename T4, typename T, int ITEM_PER_THREAD, NormType norm_type>
__global__ void norm_twoPassAlgo_stored_locally_e4_fused_scale_shift_4d(
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
    const int frame_seqlen,
    bool affine,
    float eps) {
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
    if constexpr (norm_type == NormType::LayerNorm) {
      local_sums[0] += static_cast<float>(local_val[i].x) + static_cast<float>(local_val[i].y) +
                       static_cast<float>(local_val[i].z) + static_cast<float>(local_val[i].w);
    } else {
      local_sums[0] += static_cast<float>(local_val[i].x) * static_cast<float>(local_val[i].x) +
                       static_cast<float>(local_val[i].y) * static_cast<float>(local_val[i].y) +
                       static_cast<float>(local_val[i].z) * static_cast<float>(local_val[i].z) +
                       static_cast<float>(local_val[i].w) * static_cast<float>(local_val[i].w);
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

  if constexpr (norm_type == NormType::LayerNorm) {
    local_sums[0] = 0.0f;
#pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
      const int index = i * bdimx + tid;
      if (index < n_4) {
        const float4 tmp = {
            static_cast<float>(local_val[i].x) - s_mean,
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
  }
  if (threadIdx.x == 0) {
    // In rms norm, s_variance represents rsqrtf(x^2/n+eps).
    s_variance = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i * bdimx + tid;
    if (index < n_4) {
      if constexpr (norm_type == NormType::LayerNorm) {
        const T4 gamma_val = affine ? gamma[index] : T4{T(1.0f), T(1.0f), T(1.0f), T(1.0f)};
        const T4 beta_val = affine ? beta[index] : T4{T(0.0f), T(0.0f), T(0.0f), T(0.0f)};
        const T4 scale_val = scale4d[base4d + index];
        const T4 shift_val = shift4d[base4d + index];
        T4 tmp;
        tmp.x =
            T(((static_cast<float>(local_val[i].x) - s_mean) * s_variance * static_cast<float>(gamma_val.x) +
               static_cast<float>(beta_val.x)) *
                  (1.0f + static_cast<float>(scale_val.x)) +
              static_cast<float>(shift_val.x));
        tmp.y =
            T(((static_cast<float>(local_val[i].y) - s_mean) * s_variance * static_cast<float>(gamma_val.y) +
               static_cast<float>(beta_val.y)) *
                  (1.0f + static_cast<float>(scale_val.y)) +
              static_cast<float>(shift_val.y));
        tmp.z =
            T(((static_cast<float>(local_val[i].z) - s_mean) * s_variance * static_cast<float>(gamma_val.z) +
               static_cast<float>(beta_val.z)) *
                  (1.0f + static_cast<float>(scale_val.z)) +
              static_cast<float>(shift_val.z));
        tmp.w =
            T(((static_cast<float>(local_val[i].w) - s_mean) * s_variance * static_cast<float>(gamma_val.w) +
               static_cast<float>(beta_val.w)) *
                  (1.0f + static_cast<float>(scale_val.w)) +
              static_cast<float>(shift_val.w));
        output[index] = tmp;
      } else {
        const T4 gamma_val = affine ? gamma[index] : T4{T(1.0f), T(1.0f), T(1.0f), T(1.0f)};
        const T4 scale_val = scale4d[base4d + index];
        const T4 shift_val = shift4d[base4d + index];
        T4 tmp;
        tmp.x =
            T((static_cast<float>(local_val[i].x) * s_variance * static_cast<float>(gamma_val.x)) *
                  (1.0f + static_cast<float>(scale_val.x)) +
              static_cast<float>(shift_val.x));
        tmp.y =
            T((static_cast<float>(local_val[i].y) * s_variance * static_cast<float>(gamma_val.y)) *
                  (1.0f + static_cast<float>(scale_val.y)) +
              static_cast<float>(shift_val.y));
        tmp.z =
            T((static_cast<float>(local_val[i].z) * s_variance * static_cast<float>(gamma_val.z)) *
                  (1.0f + static_cast<float>(scale_val.z)) +
              static_cast<float>(shift_val.z));
        tmp.w =
            T((static_cast<float>(local_val[i].w) * s_variance * static_cast<float>(gamma_val.w)) *
                  (1.0f + static_cast<float>(scale_val.w)) +
              static_cast<float>(shift_val.w));
        output[index] = tmp;
      }
    }
  }
}

static void norm_fused_scale_shift_launch(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& gamma_opt,
    const c10::optional<torch::Tensor>& beta_opt,
    const torch::Tensor& scale,
    const torch::Tensor& shift,
    torch::Tensor& y,
    NormType norm_type,
    float eps) {
  bool has_gamma = gamma_opt.has_value() && gamma_opt->defined();
  bool has_beta = beta_opt.has_value() && beta_opt->defined();
  // layermorm requires gamma and beta to be either both defined or both undefined.
  bool affine = has_gamma;
  auto gamma_ptr = has_gamma ? gamma_opt.value().data_ptr() : nullptr;
  auto beta_ptr = has_beta ? beta_opt.value().data_ptr() : nullptr;

  const int64_t M = x.size(0);
  const int64_t N = x.size(1);
  dim3 grid((unsigned)M);
  TORCH_CHECK((N % 4) == 0, "N must be divisible by 4");
  dim3 block(0);

  auto is_broadcast_2d = [&](const torch::Tensor& t) {
    if (t.dim() == 2) return (t.size(0) == M || t.size(0) == 1) && t.size(1) == N;
    if (t.dim() == 3) return t.size(0) == 1 && t.size(1) == 1 && t.size(2) == N;
    return false;
  };

  bool use_2d = is_broadcast_2d(scale) && is_broadcast_2d(shift);
  bool is_scale_c_1 = false;
  bool is_shift_c_1 = false;
  if (use_2d) {
    is_scale_c_1 = (scale.dim() == 3) || (scale.size(0) == 1);
    is_shift_c_1 = (shift.dim() == 3) || (shift.size(0) == 1);
  }

  const bool use_4d = (scale.dim() == 4 && shift.dim() == 4);
  const bool scalar_both = (scale.dim() == 1 && scale.numel() == 1 && shift.dim() == 1 && shift.numel() == 1);
  bool skip = false;
  if (scalar_both) {
    const float s0 = scale.item<float>();
    const float sh0 = shift.item<float>();
    skip = (s0 == 0.0f && sh0 == 0.0f);
  }
  TORCH_CHECK(use_2d || use_4d || skip, "scale/shift must be 2D [M, N], 4D [B, F, 1, N], or scalar zeros to skip");

  auto dispatch = [&](auto launch_kernel) {
    auto dispatch_dtype = [&](auto dtype) {
      auto dispatch_item_per_thread = [&](auto item_per_thread_tag) {
        auto dispatch_norm_type = [&](auto norm_tag) { launch_kernel(dtype, item_per_thread_tag, norm_tag); };

        if (norm_type == 0) {
          dispatch_norm_type(NormTag<NormType::LayerNorm>{});
        } else {
          dispatch_norm_type(NormTag<NormType::RMSNorm>{});
        }
      };

      if (N <= 4096) {
        block.x = (int)((N / 4 + 31) / 32 * 32);
        if (block.x > 1024) block.x = 1024;
        dispatch_item_per_thread(ItemPerThreadTag<1>{});
      } else {
        // For all N > 4096, use the configuration previously used for 4096 < N <= 8192.
        block.x = (int)(((N + 7) / 8 + 31) / 32 * 32);
        if (block.x > 1024) block.x = 1024;
        dispatch_item_per_thread(ItemPerThreadTag<8>{});
      }
    };

    if (x.dtype() == torch::kFloat32) {
      dispatch_dtype(DTypeTag<float4, float>{});
    } else if (x.dtype() == torch::kFloat16) {
      dispatch_dtype(DTypeTag<half4, half>{});
    } else if (x.dtype() == torch::kBFloat16) {
      dispatch_dtype(DTypeTag<bf16_4, cutlass::bfloat16_t>{});
    } else {
      TORCH_CHECK(false, "Unsupported dtype. Use float32, float16, or bfloat16.");
    }
  };

  // If skipping scale/shift, launch the non-fused LN kernel (no temporary tensors).
  if (skip) {
    auto launch_kernel = [&](auto dtype_tag, auto ipt_tag, auto norm_tag) {
      using T4 = typename decltype(dtype_tag)::T4;
      using T = typename decltype(dtype_tag)::T;
      using IPT = decltype(ipt_tag);
      using NT = decltype(norm_tag);
      norm_twoPassAlgo_stored_locally_e4<T4, T, IPT::value, NT::value>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
              (T4*)y.data_ptr(),
              (const T4*)x.data_ptr(),
              (const T4*)gamma_ptr,
              (const T4*)beta_ptr,
              (int)M,
              (int)N,
              affine,
              eps);
    };

    dispatch(launch_kernel);
    return;
  }

  if (use_2d) {
    auto launch_kernel = [&](auto dtype_tag, auto ipt_tag, auto norm_tag) {
      using T4 = typename decltype(dtype_tag)::T4;
      using T = typename decltype(dtype_tag)::T;
      using IPT = decltype(ipt_tag);
      using NT = decltype(norm_tag);
      norm_twoPassAlgo_stored_locally_e4_fused_scale_shift<T4, T, IPT::value, NT::value>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
              (T4*)y.data_ptr(),
              (const T4*)x.data_ptr(),
              (const T4*)gamma_ptr,
              (const T4*)beta_ptr,
              (const T4*)scale.data_ptr(),
              (const T4*)shift.data_ptr(),
              (int)M,
              (int)N,
              affine,
              is_scale_c_1,
              is_shift_c_1,
              eps);
    };

    dispatch(launch_kernel);
    return;
  }

  // 4D launcher path
  TORCH_CHECK(scale.size(2) == 1 && shift.size(2) == 1, "scale/shift 4D must have size 1 at dim=2");
  TORCH_CHECK(scale.size(3) == N && shift.size(3) == N, "scale/shift last dim must be N");
  const int64_t B = scale.size(0);
  const int64_t F = scale.size(1);
  TORCH_CHECK((M % (B * F)) == 0, "M must be divisible by B*F for 4D scale/shift");
  const int frame_seqlen = (int)(M / (B * F));

  auto launch_kernel = [&](auto dtype_tag, auto ipt_tag, auto norm_tag) {
    using T4 = typename decltype(dtype_tag)::T4;
    using T = typename decltype(dtype_tag)::T;
    using IPT = decltype(ipt_tag);
    using NT = decltype(norm_tag);
    norm_twoPassAlgo_stored_locally_e4_fused_scale_shift_4d<T4, T, IPT::value, NT::value>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            (T4*)y.data_ptr(),
            (const T4*)x.data_ptr(),
            (const T4*)gamma_ptr,
            (const T4*)beta_ptr,
            (const T4*)scale.data_ptr(),
            (const T4*)shift.data_ptr(),
            (int)M,
            (int)N,
            (int)B,
            (int)F,
            (int)frame_seqlen,
            affine,
            eps);
  };

  dispatch(launch_kernel);
}

torch::Tensor fused_norm_scale_shift(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& gamma_opt,
    const c10::optional<torch::Tensor>& beta_opt,
    const torch::Tensor& scale,
    const torch::Tensor& shift,
    int64_t norm_type,
    double eps) {
  bool has_gamma = gamma_opt.has_value() && gamma_opt->defined();
  bool has_beta = beta_opt.has_value() && beta_opt->defined();

  CHECK_CUDA(x);
  CHECK_CUDA(scale);
  CHECK_CUDA(shift);
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, N]");
  TORCH_CHECK(x.stride(-1) == 1, "last dim of x must be contiguous (stride 1)");
  TORCH_CHECK(x.dtype() == scale.dtype() && x.dtype() == shift.dtype(), "x, scale, shift must have same dtype");
  const int64_t M = x.size(0);
  const int64_t N = x.size(1);
  if ((scale.dim() == 2 || scale.dim() == 3) && (shift.dim() == 2 || shift.dim() == 3)) {
    TORCH_CHECK(scale.size(-1) == N && shift.size(-1) == N, "scale/shift last dim must be N");
    TORCH_CHECK(
        scale.stride(-1) == 1 && shift.stride(-1) == 1, "last dim of scale/shift must be contiguous (stride 1)");
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
  if (has_gamma) {
    const auto& gamma = gamma_opt.value();
    CHECK_CUDA(gamma);
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D [N]");
    TORCH_CHECK(gamma.numel() == N, "gamma must be length N");
    if (has_beta) {
      const auto& beta = beta_opt.value();
      CHECK_CUDA(beta);
      TORCH_CHECK(beta.dim() == 1, "beta must be 1D [N]");
      TORCH_CHECK(beta.numel() == N, "beta must be length N");
      TORCH_CHECK(x.dtype() == gamma.dtype() && x.dtype() == beta.dtype(), "x, gamma, beta must have same dtype");
    }
  }
  TORCH_CHECK(norm_type == 0 || norm_type == 1, "norm_type must be 0 (layer) or 1 (rms).");

  auto y = torch::empty_like(x);
  norm_fused_scale_shift_launch(
      x, gamma_opt, beta_opt, scale, shift, y, NormType((int)norm_type), static_cast<float>(eps));
  return y;
}

// =========================
// Fused Residual + Gate + LayerNorm/RMSNorm + Scale/Shift
// =========================

// gate_mode:
// 0: no gate (scalar 1.0), residual_output = residual + x
// 1: 2D gate [M, N]
// 2: Bx1xN gate [B, 1, N]
// 3: BxFx1xN gate [B, F, 1, N]
template <typename T4, typename T, int ITEM_PER_THREAD, NormType norm_type>
__global__ void norm_e4_fused_res_gate_scale_shift_2d(
    T4* __restrict__ output,
    T4* __restrict__ residual_out,
    const T4* __restrict__ x,
    const T4* __restrict__ residual,
    const T4* __restrict__ gamma,
    const T4* __restrict__ beta,
    const T4* __restrict__ scale,
    const T4* __restrict__ shift,
    const T4* __restrict__ gate_mn,  // used when gate_mode == 1
    const T4* __restrict__ gate_b1,  // used when gate_mode == 2 (flattened [B,1,N] -> [B,N])
    const int m,
    const int n,
    const int gate_mode,
    const int rows_per_b,  // valid when gate_mode == 2
    bool affine,
    bool is_scale_c_1,
    bool is_shift_c_1,
    bool is_gate_c_1,
    float eps) {
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
    if (index < n_4) {
      const T4 x_v = x[offset + index];
      const T4 r_v = residual[offset + index];
      T4 g_v;
      if (gate_mode == 0) {
        g_v = {T(1.0f), T(1.0f), T(1.0f), T(1.0f)};
      } else if (gate_mode == 1) {
        g_v = gate_mn[is_gate_c_1 ? index : (offset + index)];
      } else {  // gate_mode == 2
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
      if constexpr (norm_type == NormType::LayerNorm) {
        local_sums[0] += static_cast<float>(sum_v.x) + static_cast<float>(sum_v.y) + static_cast<float>(sum_v.z) +
                         static_cast<float>(sum_v.w);
      } else {
        local_sums[0] += static_cast<float>(sum_v.x) * static_cast<float>(sum_v.x) +
                         static_cast<float>(sum_v.y) * static_cast<float>(sum_v.y) +
                         static_cast<float>(sum_v.z) * static_cast<float>(sum_v.z) +
                         static_cast<float>(sum_v.w) * static_cast<float>(sum_v.w);
      }
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

  if constexpr (norm_type == NormType::LayerNorm) {
    local_sums[0] = 0.0f;
#pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
      const int index = i * bdimx + tid;
      if (index < n_4) {
        const float4 tmp = {
            static_cast<float>(local_val[i].x) - s_mean,
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
  }
  if (threadIdx.x == 0) {
    // In rms norm, s_variance represents rsqrtf(x^2/n+eps).
    s_variance = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i * bdimx + tid;
    if (index < n_4) {
      if constexpr (norm_type == NormType::LayerNorm) {
        const T4 gamma_val = affine ? gamma[index] : T4{T(1.0f), T(1.0f), T(1.0f), T(1.0f)};
        const T4 beta_val = affine ? beta[index] : T4{T(0.0f), T(0.0f), T(0.0f), T(0.0f)};
        const T4 scale_val = scale[is_scale_c_1 ? index : (offset + index)];
        const T4 shift_val = shift[is_shift_c_1 ? index : (offset + index)];
        T4 tmp;
        tmp.x =
            T(((static_cast<float>(local_val[i].x) - s_mean) * s_variance * static_cast<float>(gamma_val.x) +
               static_cast<float>(beta_val.x)) *
                  (1.0f + static_cast<float>(scale_val.x)) +
              static_cast<float>(shift_val.x));
        tmp.y =
            T(((static_cast<float>(local_val[i].y) - s_mean) * s_variance * static_cast<float>(gamma_val.y) +
               static_cast<float>(beta_val.y)) *
                  (1.0f + static_cast<float>(scale_val.y)) +
              static_cast<float>(shift_val.y));
        tmp.z =
            T(((static_cast<float>(local_val[i].z) - s_mean) * s_variance * static_cast<float>(gamma_val.z) +
               static_cast<float>(beta_val.z)) *
                  (1.0f + static_cast<float>(scale_val.z)) +
              static_cast<float>(shift_val.z));
        tmp.w =
            T(((static_cast<float>(local_val[i].w) - s_mean) * s_variance * static_cast<float>(gamma_val.w) +
               static_cast<float>(beta_val.w)) *
                  (1.0f + static_cast<float>(scale_val.w)) +
              static_cast<float>(shift_val.w));
        output[offset + index] = tmp;
      } else {
        const T4 gamma_val = affine ? gamma[index] : T4{T(1.0f), T(1.0f), T(1.0f), T(1.0f)};
        const T4 scale_val = scale[is_scale_c_1 ? index : (offset + index)];
        const T4 shift_val = shift[is_shift_c_1 ? index : (offset + index)];
        T4 tmp;
        tmp.x =
            T((static_cast<float>(local_val[i].x) * s_variance * static_cast<float>(gamma_val.x)) *
                  (1.0f + static_cast<float>(scale_val.x)) +
              static_cast<float>(shift_val.x));
        tmp.y =
            T((static_cast<float>(local_val[i].y) * s_variance * static_cast<float>(gamma_val.y)) *
                  (1.0f + static_cast<float>(scale_val.y)) +
              static_cast<float>(shift_val.y));
        tmp.z =
            T((static_cast<float>(local_val[i].z) * s_variance * static_cast<float>(gamma_val.z)) *
                  (1.0f + static_cast<float>(scale_val.z)) +
              static_cast<float>(shift_val.z));
        tmp.w =
            T((static_cast<float>(local_val[i].w) * s_variance * static_cast<float>(gamma_val.w)) *
                  (1.0f + static_cast<float>(scale_val.w)) +
              static_cast<float>(shift_val.w));
        output[offset + index] = tmp;
      }
    }
  }
}

template <typename T4, typename T, int ITEM_PER_THREAD, NormType norm_type>
__global__ void norm_e4_fused_res_gate_scale_shift_4d(
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
    const int gate_mode,  // 0 or 3 expected here
    const int B,
    const int F,
    const int frame_seqlen,
    bool affine,
    float eps) {
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
    if (index < n_4) {
      const T4 x_v = x[offset + index];
      const T4 r_v = residual[offset + index];
      T4 g_v;
      if (gate_mode == 0) {
        g_v = {T(1.0f), T(1.0f), T(1.0f), T(1.0f)};
      } else {  // gate_mode == 3
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
      if constexpr (norm_type == NormType::LayerNorm) {
        local_sums[0] += static_cast<float>(sum_v.x) + static_cast<float>(sum_v.y) + static_cast<float>(sum_v.z) +
                         static_cast<float>(sum_v.w);
      } else {
        local_sums[0] += static_cast<float>(sum_v.x) * static_cast<float>(sum_v.x) +
                         static_cast<float>(sum_v.y) * static_cast<float>(sum_v.y) +
                         static_cast<float>(sum_v.z) * static_cast<float>(sum_v.z) +
                         static_cast<float>(sum_v.w) * static_cast<float>(sum_v.w);
      }
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

  if constexpr (norm_type == NormType::LayerNorm) {
    local_sums[0] = 0.0f;
#pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
      const int index = i * bdimx + tid;
      if (index < n_4) {
        const float4 tmp = {
            static_cast<float>(local_val[i].x) - s_mean,
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
  }
  if (threadIdx.x == 0) {
    // In rms norm, s_variance represents rsqrtf(x^2/n+eps).
    s_variance = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i * bdimx + tid;
    if (index < n_4) {
      if constexpr (norm_type == NormType::LayerNorm) {
        const T4 gamma_val = affine ? gamma[index] : T4{T(1.0f), T(1.0f), T(1.0f), T(1.0f)};
        const T4 beta_val = affine ? beta[index] : T4{T(0.0f), T(0.0f), T(0.0f), T(0.0f)};
        const T4 scale_val = scale4d[base4d + index];
        const T4 shift_val = shift4d[base4d + index];
        T4 tmp;
        tmp.x =
            T(((static_cast<float>(local_val[i].x) - s_mean) * s_variance * static_cast<float>(gamma_val.x) +
               static_cast<float>(beta_val.x)) *
                  (1.0f + static_cast<float>(scale_val.x)) +
              static_cast<float>(shift_val.x));
        tmp.y =
            T(((static_cast<float>(local_val[i].y) - s_mean) * s_variance * static_cast<float>(gamma_val.y) +
               static_cast<float>(beta_val.y)) *
                  (1.0f + static_cast<float>(scale_val.y)) +
              static_cast<float>(shift_val.y));
        tmp.z =
            T(((static_cast<float>(local_val[i].z) - s_mean) * s_variance * static_cast<float>(gamma_val.z) +
               static_cast<float>(beta_val.z)) *
                  (1.0f + static_cast<float>(scale_val.z)) +
              static_cast<float>(shift_val.z));
        tmp.w =
            T(((static_cast<float>(local_val[i].w) - s_mean) * s_variance * static_cast<float>(gamma_val.w) +
               static_cast<float>(beta_val.w)) *
                  (1.0f + static_cast<float>(scale_val.w)) +
              static_cast<float>(shift_val.w));
        output[offset + index] = tmp;
      } else {
        const T4 gamma_val = affine ? gamma[index] : T4{T(1.0f), T(1.0f), T(1.0f), T(1.0f)};
        const T4 scale_val = scale4d[base4d + index];
        const T4 shift_val = shift4d[base4d + index];
        T4 tmp;
        tmp.x =
            T((static_cast<float>(local_val[i].x) * s_variance * static_cast<float>(gamma_val.x)) *
                  (1.0f + static_cast<float>(scale_val.x)) +
              static_cast<float>(shift_val.x));
        tmp.y =
            T((static_cast<float>(local_val[i].y) * s_variance * static_cast<float>(gamma_val.y)) *
                  (1.0f + static_cast<float>(scale_val.y)) +
              static_cast<float>(shift_val.y));
        tmp.z =
            T((static_cast<float>(local_val[i].z) * s_variance * static_cast<float>(gamma_val.z)) *
                  (1.0f + static_cast<float>(scale_val.z)) +
              static_cast<float>(shift_val.z));
        tmp.w =
            T((static_cast<float>(local_val[i].w) * s_variance * static_cast<float>(gamma_val.w)) *
                  (1.0f + static_cast<float>(scale_val.w)) +
              static_cast<float>(shift_val.w));
        output[offset + index] = tmp;
      }
    }
  }
}

static void norm_fused_res_gate_scale_shift_launch_with_residual(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const c10::optional<torch::Tensor>& gate_opt,
    const c10::optional<torch::Tensor>& gamma_opt,
    const c10::optional<torch::Tensor>& beta_opt,
    const torch::Tensor& scale,
    const torch::Tensor& shift,
    torch::Tensor& y,
    torch::Tensor& residual_out,
    NormType norm_type,
    float eps) {
  bool has_gamma = gamma_opt.has_value() && gamma_opt->defined();
  bool has_beta = beta_opt.has_value() && beta_opt->defined();
  auto gamma_ptr = has_gamma ? gamma_opt.value().data_ptr() : nullptr;
  auto beta_ptr = has_beta ? beta_opt.value().data_ptr() : nullptr;
  bool affine = has_gamma;

  const int64_t M = x.size(0);
  const int64_t N = x.size(1);
  dim3 grid((unsigned)M);
  TORCH_CHECK((N % 4) == 0, "N must be divisible by 4");
  dim3 block(0);

  // Configure thread block
  if (N <= 4096) {
    block.x = (int)((N / 4 + 31) / 32 * 32);
  } else {
    // For all N > 4096, use the configuration previously used for 4096 < N <= 8192.
    block.x = (int)(((N + 7) / 8 + 31) / 32 * 32);
  }
  if (block.x > 1024) block.x = 1024;

  auto is_broadcast_2d = [&](const torch::Tensor& t) {
    if (t.dim() == 2) return (t.size(0) == M || t.size(0) == 1) && t.size(1) == N;
    if (t.dim() == 3) return t.size(0) == 1 && t.size(1) == 1 && t.size(2) == N;
    return false;
  };

  bool use_2d = is_broadcast_2d(scale) && is_broadcast_2d(shift);
  bool is_scale_c_1 = false;
  bool is_shift_c_1 = false;
  if (use_2d) {
    is_scale_c_1 = (scale.dim() == 3) || (scale.size(0) == 1);
    is_shift_c_1 = (shift.dim() == 3) || (shift.size(0) == 1);
  }

  const bool use_4d = (scale.dim() == 4 && shift.dim() == 4);
  const bool scalar_both = (scale.dim() == 1 && scale.numel() == 1 && shift.dim() == 1 && shift.numel() == 1);
  bool skip = false;
  if (scalar_both) {
    const float s0 = scale.item<float>();
    const float sh0 = shift.item<float>();
    skip = (s0 == 0.0f && sh0 == 0.0f);
  }
  TORCH_CHECK(
      use_2d || use_4d || skip,
      "scale/shift must be 2D [M, N] , 2D [1, N], 3D [1, 1, N], 4D [B, F, 1, N], or scalar zeros to skip");

  // Determine gate mode
  int gate_mode = 0;
  bool is_gate_c_1 = false;
  torch::Tensor gate;
  if (gate_opt.has_value() && gate_opt->defined()) {
    gate = *gate_opt;
    TORCH_CHECK(gate.dtype() == x.dtype(), "gate must have same dtype as x");
    if (gate.dim() == 2) {
      if (gate.size(0) == M && gate.size(1) == N) {
        gate_mode = 1;
      } else if (gate.size(0) == 1 && gate.size(1) == N) {
        gate_mode = 1;
        is_gate_c_1 = true;
      } else {
        TORCH_CHECK(false, "2D gate must be [M, N] or [1, N]");
      }
    } else if (gate.dim() == 3) {
      if (gate.size(0) == 1 && gate.size(1) == 1 && gate.size(2) == N) {
        gate_mode = 1;
        is_gate_c_1 = true;
      } else {
        TORCH_CHECK(gate.size(1) == 1 && gate.size(2) == N, "3D gate must be [B, 1, N]");
        const int64_t B = gate.size(0);
        TORCH_CHECK((M % B) == 0, "M must be divisible by B for 3D gate [B,1,N]");
        gate_mode = 2;
      }
    } else if (gate.dim() == 4) {
      TORCH_CHECK(gate.size(2) == 1 && gate.size(3) == N, "4D gate must be [B, F, 1, N]");
      gate_mode = 3;
    } else {
      TORCH_CHECK(false, "Unsupported gate shape. Use [M,N], [B,1,N], [B,F,1,N] , 2D [1, N], 3D [1, 1, N]");
    }
  }

  auto dispatch = [&](auto launch_kernel) {
    auto dispatch_dtype = [&](auto dtype) {
      auto dispatch_item_per_thread = [&](auto item_per_thread_tag) {
        auto dispatch_norm_type = [&](auto norm_tag) { launch_kernel(dtype, item_per_thread_tag, norm_tag); };

        if (norm_type == 0) {
          dispatch_norm_type(NormTag<NormType::LayerNorm>{});
        } else {
          dispatch_norm_type(NormTag<NormType::RMSNorm>{});
        }
      };

      if (N <= 4096) {
        dispatch_item_per_thread(ItemPerThreadTag<1>{});
      } else {
        dispatch_item_per_thread(ItemPerThreadTag<8>{});
      }
    };

    if (x.dtype() == torch::kFloat32) {
      dispatch_dtype(DTypeTag<float4, float>{});
    } else if (x.dtype() == torch::kFloat16) {
      dispatch_dtype(DTypeTag<half4, half>{});
    } else if (x.dtype() == torch::kBFloat16) {
      dispatch_dtype(DTypeTag<bf16_4, cutlass::bfloat16_t>{});
    } else {
      TORCH_CHECK(false, "Unsupported dtype. Use float32, float16, or bfloat16.");
    }
  };

  if (use_2d || skip) {
    const int rows_per_b = (gate_mode == 2) ? (int)(M / gate.size(0)) : 0;
    torch::Tensor scale2d = scale;
    torch::Tensor shift2d = shift;
    if (skip) {
      TORCH_CHECK(
          gate_mode != 3,
          "When skipping with scalar scale/shift, 4D gate is not supported. Provide 2D/3D gate or 4D scale/shift.");
      scale2d = torch::zeros({M, N}, x.options());
      shift2d = torch::zeros({M, N}, x.options());
    }

    auto launch_kernel = [&](auto dtype_tag, auto ipt_tag, auto norm_tag) {
      using T4 = typename decltype(dtype_tag)::T4;
      using T = typename decltype(dtype_tag)::T;
      using IPT = decltype(ipt_tag);
      using NT = decltype(norm_tag);
      norm_e4_fused_res_gate_scale_shift_2d<T4, T, IPT::value, NT::value>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
              (T4*)y.data_ptr(),
              (T4*)residual_out.data_ptr(),
              (const T4*)x.data_ptr(),
              (const T4*)residual.data_ptr(),
              (const T4*)gamma_ptr,
              (const T4*)beta_ptr,
              (const T4*)scale2d.data_ptr(),
              (const T4*)shift2d.data_ptr(),
              (gate_mode == 1) ? (const T4*)gate.data_ptr() : nullptr,
              (gate_mode == 2) ? (const T4*)gate.data_ptr() : nullptr,
              (int)M,
              (int)N,
              gate_mode,
              rows_per_b,
              affine,
              is_scale_c_1,
              is_shift_c_1,
              is_gate_c_1,
              eps);
    };

    dispatch(launch_kernel);
    return;
  }

  // 4D path with residual_out
  const int64_t B = scale.size(0);
  const int64_t F = scale.size(1);
  const int frame_seqlen = (int)(M / (B * F));
  TORCH_CHECK(gate_mode == 0 || gate_mode == 3, "When scale/shift are 4D, gate must be none or 4D [B,F,1,N]");

  auto launch_kernel = [&](auto dtype_tag, auto ipt_tag, auto norm_tag) {
    using T4 = typename decltype(dtype_tag)::T4;
    using T = typename decltype(dtype_tag)::T;
    using IPT = decltype(ipt_tag);
    using NT = decltype(norm_tag);
    norm_e4_fused_res_gate_scale_shift_4d<T4, T, IPT::value, NT::value>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            (T4*)y.data_ptr(),
            (T4*)residual_out.data_ptr(),
            (const T4*)x.data_ptr(),
            (const T4*)residual.data_ptr(),
            (const T4*)gamma_ptr,
            (const T4*)beta_ptr,
            (const T4*)scale.data_ptr(),
            (const T4*)shift.data_ptr(),
            nullptr,
            nullptr,
            (gate_mode == 3) ? (const T4*)gate.data_ptr() : nullptr,
            (int)M,
            (int)N,
            gate_mode,
            (int)B,
            (int)F,
            frame_seqlen,
            affine,
            eps);
  };

  dispatch(launch_kernel);
}

std::tuple<torch::Tensor, torch::Tensor> fused_scale_residual_norm_scale_shift(
    const torch::Tensor& residual,
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& gate_opt,
    const c10::optional<torch::Tensor>& gamma_opt,
    const c10::optional<torch::Tensor>& beta_opt,
    const torch::Tensor& scale,
    const torch::Tensor& shift,
    int64_t norm_type,
    double eps) {
  CHECK_CUDA(x);
  CHECK_CUDA(residual);
  CHECK_CUDA(scale);
  CHECK_CUDA(shift);
  TORCH_CHECK(x.dim() == 2 && residual.dim() == 2, "x and residual must be 2D [M, N]");
  TORCH_CHECK(x.stride(-1) == 1 && residual.stride(-1) == 1, "last dim of x/residual must be contiguous (stride 1)");
  TORCH_CHECK(x.dtype() == residual.dtype(), "x and residual must have same dtype");
  TORCH_CHECK(x.size(0) == residual.size(0) && x.size(1) == residual.size(1), "x and residual shapes must match");
  TORCH_CHECK(x.dtype() == scale.dtype() && x.dtype() == shift.dtype(), "x, scale, shift must have same dtype");
  const int64_t M = x.size(0);
  const int64_t N = x.size(1);
  if ((scale.dim() == 2 || scale.dim() == 3) && (shift.dim() == 2 || shift.dim() == 3)) {
    TORCH_CHECK(scale.size(-1) == N && shift.size(-1) == N, "scale/shift last dim must be N");
    TORCH_CHECK(
        scale.stride(-1) == 1 && shift.stride(-1) == 1, "last dim of scale/shift must be contiguous (stride 1)");
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
      TORCH_CHECK((gate.size(0) == M || gate.size(0) == 1) && gate.size(1) == N, "2D gate must be [M, N] or [1, N]");
      TORCH_CHECK(gate.stride(-1) == 1, "last dim of gate must be contiguous (stride 1)");
    } else if (gate.dim() == 3) {
      TORCH_CHECK(gate.size(1) == 1 && gate.size(2) == N, "3D gate must be [B, 1, N]");
      if (gate.size(0) != 1) {
        TORCH_CHECK((M % gate.size(0)) == 0, "M must be divisible by B for 3D gate [B,1,N]");
      }
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

  if (gamma_opt.has_value() && gamma_opt->defined()) {
    const auto& gamma = gamma_opt.value();
    CHECK_CUDA(gamma);
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D [N]");
    TORCH_CHECK(x.dtype() == gamma.dtype(), "x, gamma must have same dtype");
    TORCH_CHECK(gamma.numel() == N, "gamma must be length N");
    if (beta_opt.has_value() && beta_opt->defined()) {
      const auto& beta = beta_opt.value();
      CHECK_CUDA(beta);
      TORCH_CHECK(beta.dim() == 1, "beta must be 1D [N]");
      TORCH_CHECK(x.dtype() == beta.dtype(), "x, beta must have same dtype");
      TORCH_CHECK(beta.numel() == N, "beta must be length N");
    }
  }
  TORCH_CHECK(norm_type == 0 || norm_type == 1, "norm_type must be 0 (layer) or 1 (rms).");

  auto y = torch::empty_like(x);
  auto residual_output = torch::empty_like(x);

  norm_fused_res_gate_scale_shift_launch_with_residual(
      x,
      residual,
      gate_opt,
      gamma_opt,
      beta_opt,
      scale,
      shift,
      y,
      residual_output,
      NormType((int)norm_type),
      static_cast<float>(eps));
  return std::make_tuple(y, residual_output);
}
