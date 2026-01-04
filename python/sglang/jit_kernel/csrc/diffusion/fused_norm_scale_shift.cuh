/* Copyright 2025 SGLang Team. */
#include <sgl_kernel/tensor.h>   // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.cuh>  // For LaunchKernel
#include <sgl_kernel/utils.h>    // For div_ceil, RuntimeCheck

#include <cuda_fp16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include "cutlass/numeric_types.h"

namespace {

namespace ffi = tvm::ffi;

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

// both scale and shift are scalar
template <typename T4, typename T, int ITEM_PER_THREAD, NormType norm_type>
__global__ void norm_twoPassAlgo_stored_locally_e4(
    T4* output,
    const T4* input,
    const T4* gamma,
    const T4* beta,
    const T* scale,
    const T* shift,
    const int m,
    const int n,
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
        const T scale_v = scale[0], shift_v = shift[0];
        const T4 scale_val = {scale_v, scale_v, scale_v, scale_v};
        const T4 shift_val = {shift_v, shift_v, shift_v, shift_v};
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
        const T scale_v = scale[0], shift_v = shift[0];
        const T4 scale_val = {scale_v, scale_v, scale_v, scale_v};
        const T4 shift_val = {shift_v, shift_v, shift_v, shift_v};
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
    ffi::TensorView& out,
    const ffi::TensorView& x,
    const ffi::Optional<ffi::TensorView>& gamma_opt,
    const ffi::Optional<ffi::TensorView>& beta_opt,
    const ffi::TensorView& scale,
    const ffi::TensorView& shift,
    NormType norm_type,
    float eps) {
  using namespace host;

  bool has_gamma = gamma_opt.has_value();
  bool has_beta = beta_opt.has_value();
  // layermorm requires gamma and beta to be either both defined or both undefined.
  bool affine = has_gamma;
  auto gamma_ptr = has_gamma ? gamma_opt.value().data_ptr() : nullptr;
  auto beta_ptr = has_beta ? beta_opt.value().data_ptr() : nullptr;

  const int64_t M = x.size(0);
  const int64_t N = x.size(1);
  dim3 grid((unsigned)M);
  RuntimeCheck((N % 4) == 0, "N must be divisible by 4");
  dim3 block(0);

  auto is_broadcast_2d = [&](const ffi::TensorView& t) {
    if (t.ndim() == 2) return (t.size(0) == M || t.size(0) == 1) && t.size(1) == N;
    if (t.ndim() == 3) return t.size(0) == 1 && t.size(1) == 1 && t.size(2) == N;
    return false;
  };

  bool use_2d = is_broadcast_2d(scale) && is_broadcast_2d(shift);
  bool is_scale_c_1 = false;
  bool is_shift_c_1 = false;
  if (use_2d) {
    is_scale_c_1 = (scale.ndim() == 3) || (scale.size(0) == 1);
    is_shift_c_1 = (shift.ndim() == 3) || (shift.size(0) == 1);
  }

  const bool use_4d = (scale.ndim() == 4 && shift.ndim() == 4);
  const bool scalar_both = (scale.ndim() == 1 && scale.numel() == 1 && shift.ndim() == 1 && shift.numel() == 1);
  RuntimeCheck(use_2d || use_4d || scalar_both, "scale/shift must be 2D [M, N], 4D [B, F, 1, N], or 1D [1]");

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

    const auto& dtype = x.dtype();
    if (dtype.code == kDLFloat && dtype.bits == 32) {
      dispatch_dtype(DTypeTag<float4, float>{});
    } else if (dtype.code == kDLFloat && dtype.bits == 16) {
      dispatch_dtype(DTypeTag<half4, half>{});
    } else if (dtype.code == kDLBfloat && dtype.bits == 16) {
      dispatch_dtype(DTypeTag<bf16_4, cutlass::bfloat16_t>{});
    } else {
      RuntimeCheck(false, "Unsupported dtype. Use float32, float16, or bfloat16.");
    }
  };

  // If both scale and shift are scalar, launch the below kernel.
  if (scalar_both) {
    auto launch_kernel = [&](auto dtype_tag, auto ipt_tag, auto norm_tag) {
      using T4 = typename decltype(dtype_tag)::T4;
      using T = typename decltype(dtype_tag)::T;
      using IPT = decltype(ipt_tag);
      using NT = decltype(norm_tag);
      LaunchKernel(grid, block, x.device())(
          norm_twoPassAlgo_stored_locally_e4<T4, T, IPT::value, NT::value>,
          (T4*)out.data_ptr(),
          (const T4*)x.data_ptr(),
          (const T4*)gamma_ptr,
          (const T4*)beta_ptr,
          (const T*)scale.data_ptr(),
          (const T*)shift.data_ptr(),
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
      LaunchKernel(grid, block, x.device())(
          norm_twoPassAlgo_stored_locally_e4_fused_scale_shift<T4, T, IPT::value, NT::value>,
          (T4*)out.data_ptr(),
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
  RuntimeCheck(scale.size(2) == 1 && shift.size(2) == 1, "scale/shift 4D must have size 1 at dim=2");
  RuntimeCheck(scale.size(3) == N && shift.size(3) == N, "scale/shift last dim must be N");
  const int64_t B = scale.size(0);
  const int64_t F = scale.size(1);
  const int frame_seqlen = (int)(M / (B * F));

  auto launch_kernel = [&](auto dtype_tag, auto ipt_tag, auto norm_tag) {
    using T4 = typename decltype(dtype_tag)::T4;
    using T = typename decltype(dtype_tag)::T;
    using IPT = decltype(ipt_tag);
    using NT = decltype(norm_tag);
    LaunchKernel(grid, block, x.device())(
        norm_twoPassAlgo_stored_locally_e4_fused_scale_shift_4d<T4, T, IPT::value, NT::value>,
        (T4*)out.data_ptr(),
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

template <int norm_type>
void fused_norm_scale_shift(
    ffi::TensorView& out,
    const ffi::TensorView& x,
    const ffi::Optional<ffi::TensorView>& gamma_opt,
    const ffi::Optional<ffi::TensorView>& beta_opt,
    const ffi::TensorView& scale,
    const ffi::TensorView& shift,
    double eps) {
  using namespace host;

  SymbolicSize M_ = {"M"};
  SymbolicSize N_ = {"N"};
  SymbolicDevice device_;
  TensorMatcher({M_, N_})  // 2D tensor, must be contiguous
      .with_dtype<float, half, nv_bfloat16>()
      .with_strides({N_, 1})
      .with_device<kDLCUDA>(device_)
      .verify(x)
      .verify(out);

  RuntimeCheck(x.dtype() == scale.dtype() && x.dtype() == shift.dtype(), "x, scale, shift must have same dtype");
  const auto M = M_.unwrap();
  const auto N = N_.unwrap();
  RuntimeCheck((N % 4) == 0, "N must be divisible by 4");

  if ((scale.ndim() == 2 || scale.ndim() == 3) && (shift.ndim() == 2 || shift.ndim() == 3)) {
    RuntimeCheck(scale.size(-1) == N && shift.size(-1) == N, "scale/shift last dim must be N");
    RuntimeCheck(
        scale.stride(-1) == 1 && shift.stride(-1) == 1, "last dim of scale/shift must be contiguous (stride 1)");
  } else if (scale.ndim() == 4 && shift.ndim() == 4) {
    RuntimeCheck(scale.size(3) == N && shift.size(3) == N, "scale/shift last dim must be N");
    RuntimeCheck(scale.size(2) == 1 && shift.size(2) == 1, "scale/shift 4D must have size 1 at dim=2");
    RuntimeCheck(scale.stride(3) == 1 && shift.stride(3) == 1, "last dim of scale/shift must be contiguous (stride 1)");
    const int64_t B = scale.size(0);
    const int64_t F = scale.size(1);
    RuntimeCheck((M % (B * F)) == 0, "M must be divisible by B*F for 4D scale/shift");
  } else if (scale.ndim() == 1 && scale.numel() == 1 && shift.ndim() == 1 && shift.numel() == 1) {
    // Do nothing
  } else {
    RuntimeCheck(false, "scale/shift must be 2D [M, N] or 4D [B, F, 1, N]");
  }
  if (gamma_opt.has_value()) {
    const auto& gamma = gamma_opt.value();
    TensorMatcher({N_})  // 1D tensor, must be contiguous
        .with_dtype<float, half, nv_bfloat16>()
        .with_device<kDLCUDA>(device_)
        .verify(gamma);
    RuntimeCheck(x.dtype() == gamma.dtype(), "x, gamma must have same dtype");
    if (beta_opt.has_value()) {
      const auto& beta = beta_opt.value();
      TensorMatcher({N_})  // 1D tensor, must be contiguous
          .with_dtype<float, half, nv_bfloat16>()
          .with_device<kDLCUDA>(device_)
          .verify(beta);
      RuntimeCheck(x.dtype() == beta.dtype(), "x, beta must have same dtype");
    }
  }
  RuntimeCheck(norm_type == 0 || norm_type == 1, "norm_type must be 0 (layer) or 1 (rms).");

  norm_fused_scale_shift_launch(
      out, x, gamma_opt, beta_opt, scale, shift, NormType((int)norm_type), static_cast<float>(eps));
}

void fused_layernorm_scale_shift(
    ffi::TensorView out,
    const ffi::TensorView& x,
    const ffi::Optional<ffi::TensorView>& gamma_opt,
    const ffi::Optional<ffi::TensorView>& beta_opt,
    const ffi::TensorView& scale,
    const ffi::TensorView& shift,
    double eps) {
  fused_norm_scale_shift<0>(out, x, gamma_opt, beta_opt, scale, shift, eps);
}

void fused_rmsnorm_scale_shift(
    ffi::TensorView out,
    const ffi::TensorView x,
    const ffi::Optional<ffi::TensorView> gamma_opt,
    const ffi::Optional<ffi::TensorView> beta_opt,
    const ffi::TensorView scale,
    const ffi::TensorView shift,
    double eps) {
  fused_norm_scale_shift<1>(out, x, gamma_opt, beta_opt, scale, shift, eps);
}

// =========================
// Fused Residual + Gate + LayerNorm/RMSNorm + Scale/Shift
// =========================

// gate_mode:
// 0: no gate (scalar 1.0), residual_output = residual + x
// 1: 2D gate [M, N]
// 2: Bx1xN gate [B, 1, N]
// 3: BxFx1xN gate [B, F, 1, N]
template <typename T4, typename T, int ITEM_PER_THREAD, NormType norm_type, bool scalar_both>
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
        T4 scale_val, shift_val;
        if constexpr (scalar_both) {
          T scale_v = ((const T*)scale)[0];
          T shift_v = ((const T*)shift)[0];
          scale_val = {scale_v, scale_v, scale_v, scale_v};
          shift_val = {shift_v, shift_v, shift_v, shift_v};
        } else {
          scale_val = scale[is_scale_c_1 ? index : (offset + index)];
          shift_val = shift[is_shift_c_1 ? index : (offset + index)];
        }
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
        T4 scale_val, shift_val;
        if constexpr (scalar_both) {
          T scale_v = ((const T*)scale)[0];
          T shift_v = ((const T*)shift)[0];
          scale_val = {scale_v, scale_v, scale_v, scale_v};
          shift_val = {shift_v, shift_v, shift_v, shift_v};
        } else {
          scale_val = scale[is_scale_c_1 ? index : (offset + index)];
          shift_val = shift[is_shift_c_1 ? index : (offset + index)];
        }
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
    ffi::TensorView& y,
    ffi::TensorView& residual_out,
    const ffi::TensorView& x,
    const ffi::TensorView& residual,
    const ffi::Optional<ffi::TensorView>& gate_opt,
    const ffi::Optional<ffi::TensorView>& gamma_opt,
    const ffi::Optional<ffi::TensorView>& beta_opt,
    const ffi::TensorView& scale,
    const ffi::TensorView& shift,
    NormType norm_type,
    float eps) {
  using namespace host;

  auto gamma_ptr = gamma_opt.has_value() ? gamma_opt.value().data_ptr() : nullptr;
  auto beta_ptr = beta_opt.has_value() ? beta_opt.value().data_ptr() : nullptr;
  bool affine = gamma_opt.has_value();

  const int64_t M = x.size(0);
  const int64_t N = x.size(1);
  dim3 grid((unsigned)M);
  dim3 block(0);

  // Configure thread block
  if (N <= 4096) {
    block.x = (int)((N / 4 + 31) / 32 * 32);
  } else {
    // For all N > 4096, use the configuration previously used for 4096 < N <= 8192.
    block.x = (int)(((N + 7) / 8 + 31) / 32 * 32);
  }
  if (block.x > 1024) block.x = 1024;

  auto is_broadcast_2d = [&](const ffi::TensorView& t) {
    if (t.ndim() == 2) return (t.size(0) == M || t.size(0) == 1) && t.size(1) == N;
    if (t.ndim() == 3) return t.size(0) == 1 && t.size(1) == 1 && t.size(2) == N;
    return false;
  };

  bool use_2d = is_broadcast_2d(scale) && is_broadcast_2d(shift);
  bool is_scale_c_1 = false;
  bool is_shift_c_1 = false;
  if (use_2d) {
    is_scale_c_1 = (scale.ndim() == 3) || (scale.size(0) == 1);
    is_shift_c_1 = (shift.ndim() == 3) || (shift.size(0) == 1);
  }

  const bool use_4d = (scale.ndim() == 4 && shift.ndim() == 4);
  const bool scalar_both = (scale.ndim() == 1 && scale.numel() == 1 && shift.ndim() == 1 && shift.numel() == 1);
  RuntimeCheck(
      use_2d || use_4d || scalar_both,
      "scale/shift must be 2D [M, N] , 2D [1, N], 3D [1, 1, N], 4D [B, F, 1, N], or scalar");

  // Determine gate mode
  int gate_mode = 0;
  bool is_gate_c_1 = false;
  if (gate_opt.has_value()) {
    const auto& gate = gate_opt.value();
    RuntimeCheck(gate.dtype() == x.dtype(), "gate must have same dtype as x");
    if (gate.ndim() == 2) {
      if (gate.size(0) == M && gate.size(1) == N) {
        gate_mode = 1;
      } else if (gate.size(0) == 1 && gate.size(1) == N) {
        gate_mode = 1;
        is_gate_c_1 = true;
      } else {
        RuntimeCheck(false, "2D gate must be [M, N] or [1, N]");
      }
    } else if (gate.ndim() == 3) {
      if (gate.size(0) == 1 && gate.size(1) == 1 && gate.size(2) == N) {
        gate_mode = 1;
        is_gate_c_1 = true;
      } else {
        RuntimeCheck(gate.size(1) == 1 && gate.size(2) == N, "3D gate must be [B, 1, N]");
        const int64_t B = gate.size(0);
        RuntimeCheck((M % B) == 0, "M must be divisible by B for 3D gate [B,1,N]");
        gate_mode = 2;
      }
    } else if (gate.ndim() == 4) {
      RuntimeCheck(gate.size(2) == 1 && gate.size(3) == N, "4D gate must be [B, F, 1, N]");
      gate_mode = 3;
    } else {
      RuntimeCheck(false, "Unsupported gate shape. Use [M,N], [B,1,N], [B,F,1,N] , 2D [1, N], 3D [1, 1, N]");
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

    const auto& dtype = x.dtype();
    if (dtype.code == kDLFloat && dtype.bits == 32) {
      dispatch_dtype(DTypeTag<float4, float>{});
    } else if (dtype.code == kDLFloat && dtype.bits == 16) {
      dispatch_dtype(DTypeTag<half4, half>{});
    } else if (dtype.code == kDLBfloat && dtype.bits == 16) {
      dispatch_dtype(DTypeTag<bf16_4, cutlass::bfloat16_t>{});
    } else {
      RuntimeCheck(false, "Unsupported dtype. Use float32, float16, or bfloat16.");
    }
  };

  if (use_2d || scalar_both) {
    const int rows_per_b = (gate_mode == 2) ? (int)(M / gate_opt.value().size(0)) : 0;
    const auto& scale2d = scale;
    const auto& shift2d = shift;
    if (scalar_both) {
      RuntimeCheck(
          gate_mode != 3,
          "When skipping with scalar scale/shift, 4D gate is not supported. Provide 2D/3D gate or 4D scale/shift.");
    }

    auto launch_kernel = [&](auto dtype_tag, auto ipt_tag, auto norm_tag) {
      using T4 = typename decltype(dtype_tag)::T4;
      using T = typename decltype(dtype_tag)::T;
      using IPT = decltype(ipt_tag);
      using NT = decltype(norm_tag);
      if (scalar_both) {
        LaunchKernel(grid, block, x.device())(
            norm_e4_fused_res_gate_scale_shift_2d<T4, T, IPT::value, NT::value, true>,
            (T4*)y.data_ptr(),
            (T4*)residual_out.data_ptr(),
            (const T4*)x.data_ptr(),
            (const T4*)residual.data_ptr(),
            (const T4*)gamma_ptr,
            (const T4*)beta_ptr,
            (const T4*)scale2d.data_ptr(),
            (const T4*)shift2d.data_ptr(),
            (gate_mode == 1) ? (const T4*)gate_opt.value().data_ptr() : nullptr,
            (gate_mode == 2) ? (const T4*)gate_opt.value().data_ptr() : nullptr,
            (int)M,
            (int)N,
            gate_mode,
            rows_per_b,
            affine,
            is_scale_c_1,
            is_shift_c_1,
            is_gate_c_1,
            eps);
      } else {
        LaunchKernel(grid, block, x.device())(
            norm_e4_fused_res_gate_scale_shift_2d<T4, T, IPT::value, NT::value, false>,
            (T4*)y.data_ptr(),
            (T4*)residual_out.data_ptr(),
            (const T4*)x.data_ptr(),
            (const T4*)residual.data_ptr(),
            (const T4*)gamma_ptr,
            (const T4*)beta_ptr,
            (const T4*)scale2d.data_ptr(),
            (const T4*)shift2d.data_ptr(),
            (gate_mode == 1) ? (const T4*)gate_opt.value().data_ptr() : nullptr,
            (gate_mode == 2) ? (const T4*)gate_opt.value().data_ptr() : nullptr,
            (int)M,
            (int)N,
            gate_mode,
            rows_per_b,
            affine,
            is_scale_c_1,
            is_shift_c_1,
            is_gate_c_1,
            eps);
      }
    };

    dispatch(launch_kernel);
    return;
  }

  // 4D path with residual_out
  const int64_t B = scale.size(0);
  const int64_t F = scale.size(1);
  const int frame_seqlen = (int)(M / (B * F));
  RuntimeCheck(gate_mode == 0 || gate_mode == 3, "When scale/shift are 4D, gate must be none or 4D [B,F,1,N]");

  auto launch_kernel = [&](auto dtype_tag, auto ipt_tag, auto norm_tag) {
    using T4 = typename decltype(dtype_tag)::T4;
    using T = typename decltype(dtype_tag)::T;
    using IPT = decltype(ipt_tag);
    using NT = decltype(norm_tag);
    LaunchKernel(grid, block, x.device())(
        norm_e4_fused_res_gate_scale_shift_4d<T4, T, IPT::value, NT::value>,
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
        (gate_mode == 3) ? (const T4*)gate_opt.value().data_ptr() : nullptr,
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

template <int norm_type>
void fused_scale_residual_norm_scale_shift(
    ffi::TensorView& y,
    ffi::TensorView& residual_output,
    const ffi::TensorView& residual,
    const ffi::TensorView& x,
    const ffi::Optional<ffi::TensorView>& gate_opt,
    const ffi::Optional<ffi::TensorView>& gamma_opt,
    const ffi::Optional<ffi::TensorView>& beta_opt,
    const ffi::TensorView& scale,
    const ffi::TensorView& shift,
    double eps) {
  using namespace host;

  SymbolicSize M_ = {"M"};
  SymbolicSize N_ = {"N"};
  SymbolicDevice device_;
  TensorMatcher({M_, N_})  // 2D tensor, must be contiguous
      .with_dtype<float, half, nv_bfloat16>()
      .with_strides({N_, 1})
      .with_device<kDLCUDA>(device_)
      .verify(residual)
      .verify(x)
      .verify(y)
      .verify(residual_output);

  RuntimeCheck(x.dtype() == residual.dtype(), "x and residual must have same dtype");
  RuntimeCheck(x.dtype() == scale.dtype() && x.dtype() == shift.dtype(), "x, scale, shift must have same dtype");
  const int64_t M = x.size(0);
  const int64_t N = x.size(1);
  if ((scale.ndim() == 2 || scale.ndim() == 3) && (shift.ndim() == 2 || shift.ndim() == 3)) {
    RuntimeCheck(scale.size(-1) == N && shift.size(-1) == N, "scale/shift last dim must be N");
    RuntimeCheck(
        scale.stride(-1) == 1 && shift.stride(-1) == 1, "last dim of scale/shift must be contiguous (stride 1)");
  } else if (scale.ndim() == 4 && shift.ndim() == 4) {
    RuntimeCheck(scale.size(3) == N && shift.size(3) == N, "scale/shift last dim must be N");
    RuntimeCheck(scale.size(2) == 1 && shift.size(2) == 1, "scale/shift 4D must have size 1 at dim=2");
    RuntimeCheck(scale.stride(3) == 1 && shift.stride(3) == 1, "last dim of scale/shift must be contiguous (stride 1)");
    const int64_t B = scale.size(0);
    const int64_t F = scale.size(1);
    RuntimeCheck((M % (B * F)) == 0, "M must be divisible by B*F for 4D scale/shift");
  } else if (scale.ndim() == 1 && scale.numel() == 1 && shift.ndim() == 1 && shift.numel() == 1) {
    // Do nothing
  } else {
    RuntimeCheck(false, "scale/shift must be 2D [M, N] or 4D [B, F, 1, N]");
  }
  if (gate_opt.has_value()) {
    const auto& gate = gate_opt.value();
    RuntimeCheck(gate.dtype() == x.dtype(), "gate must have same dtype as x");
    if (gate.ndim() == 2) {
      RuntimeCheck((gate.size(0) == M || gate.size(0) == 1) && gate.size(1) == N, "2D gate must be [M, N] or [1, N]");
      RuntimeCheck(gate.stride(-1) == 1, "last dim of gate must be contiguous (stride 1)");
    } else if (gate.ndim() == 3) {
      RuntimeCheck(gate.size(1) == 1 && gate.size(2) == N, "3D gate must be [B, 1, N]");
      if (gate.size(0) != 1) {
        RuntimeCheck((M % gate.size(0)) == 0, "M must be divisible by B for 3D gate [B,1,N]");
      }
      RuntimeCheck(gate.stride(2) == 1, "last dim of 3D gate must be contiguous (stride 1)");
    } else if (gate.ndim() == 4) {
      RuntimeCheck(gate.size(2) == 1 && gate.size(3) == N, "4D gate must be [B, F, 1, N]");
      RuntimeCheck(gate.stride(3) == 1, "last dim of 4D gate must be contiguous (stride 1)");
      const int64_t B = gate.size(0);
      const int64_t F = gate.size(1);
      RuntimeCheck((M % (B * F)) == 0, "M must be divisible by B*F for 4D gate");
      if (scale.ndim() == 4) {
        RuntimeCheck(scale.size(0) == B && scale.size(1) == F, "gate [B,F,1,N] must match scale/shift [B,F,1,N]");
      }
    } else {
      RuntimeCheck(false, "Unsupported gate shape. Use [M,N], [B,1,N], or [B,F,1,N]");
    }
  }
  if (gamma_opt.has_value()) {
    const auto& gamma = gamma_opt.value();
    TensorMatcher({N_})  // 1D tensor, must be contiguous
        .with_dtype<float, half, nv_bfloat16>()
        .with_device<kDLCUDA>(device_)
        .verify(gamma);
    RuntimeCheck(x.dtype() == gamma.dtype(), "x, gamma must have same dtype");
    if (beta_opt.has_value()) {
      const auto& beta = beta_opt.value();
      TensorMatcher({N_})  // 1D tensor, must be contiguous
          .with_dtype<float, half, nv_bfloat16>()
          .with_device<kDLCUDA>(device_)
          .verify(beta);
      RuntimeCheck(x.dtype() == beta.dtype(), "x, beta must have same dtype");
    }
  }
  RuntimeCheck(norm_type == 0 || norm_type == 1, "norm_type must be 0 (layer) or 1 (rms).");

  norm_fused_res_gate_scale_shift_launch_with_residual(
      y,
      residual_output,
      x,
      residual,
      gate_opt,
      gamma_opt,
      beta_opt,
      scale,
      shift,
      NormType((int)norm_type),
      static_cast<float>(eps));
}

void fused_scale_residual_layernorm_scale_shift(
    ffi::TensorView y,
    ffi::TensorView residual_output,
    const ffi::TensorView residual,
    const ffi::TensorView x,
    const ffi::Optional<ffi::TensorView> gate_opt,
    const ffi::Optional<ffi::TensorView> gamma_opt,
    const ffi::Optional<ffi::TensorView> beta_opt,
    const ffi::TensorView scale,
    const ffi::TensorView shift,
    double eps) {
  fused_scale_residual_norm_scale_shift<0>(
      y, residual_output, residual, x, gate_opt, gamma_opt, beta_opt, scale, shift, eps);
}

void fused_scale_residual_rmsnorm_scale_shift(
    ffi::TensorView y,
    ffi::TensorView residual_output,
    const ffi::TensorView residual,
    const ffi::TensorView x,
    const ffi::Optional<ffi::TensorView> gate_opt,
    const ffi::Optional<ffi::TensorView> gamma_opt,
    const ffi::Optional<ffi::TensorView> beta_opt,
    const ffi::TensorView scale,
    const ffi::TensorView shift,
    double eps) {
  fused_scale_residual_norm_scale_shift<1>(
      y, residual_output, residual, x, gate_opt, gamma_opt, beta_opt, scale, shift, eps);
}

// ========== New Fused Kernels for ZImage ==========

// Kernel: RMSNorm(x) + Add -> output1, then RMSNorm(output1) -> output2
// This fuses: x = x + self.attention_norm2(attn_out); self.ffn_norm1(x)
template <typename T4, typename T, int ITEM_PER_THREAD>
__global__ void fused_rmsnorm_add_rmsnorm_kernel(
    T4* __restrict__ output1,      // intermediate result after add
    T4* __restrict__ output2,      // final normalized result
    const T4* __restrict__ x,      // input x
    const T4* __restrict__ residual,  // residual to add
    const T4* __restrict__ gamma1,    // weight for first RMSNorm
    const T4* __restrict__ gamma2,    // weight for second RMSNorm
    const int m,
    const int n,
    bool affine1,
    bool affine2,
    float eps) {
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_variance1, s_variance2;
  float local_sums[1] = {0.0f};
  T4 local_val[ITEM_PER_THREAD];
  T4 intermediate[ITEM_PER_THREAD];
  const int n_4 = n / 4;
  const int offset = m_idx * n_4;
  const T4 zero = {T(0.0f), T(0.0f), T(0.0f), T(0.0f)};

  // Step 1: Apply first RMSNorm to residual
#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i * bdimx + tid;
    if (index < n_4) {
      const T4 r_v = residual[offset + index];
      local_val[i] = r_v;
      local_sums[0] += static_cast<float>(r_v.x) * static_cast<float>(r_v.x) +
                       static_cast<float>(r_v.y) * static_cast<float>(r_v.y) +
                       static_cast<float>(r_v.z) * static_cast<float>(r_v.z) +
                       static_cast<float>(r_v.w) * static_cast<float>(r_v.w);
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
    s_variance1 = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

  // Apply first RMSNorm and add x
#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i * bdimx + tid;
    if (index < n_4) {
      const T4 gamma_val = affine1 ? gamma1[index] : T4{T(1.0f), T(1.0f), T(1.0f), T(1.0f)};
      const T4 x_v = x[offset + index];
      T4 tmp;
      tmp.x = T(static_cast<float>(x_v.x) +
                static_cast<float>(local_val[i].x) * s_variance1 * static_cast<float>(gamma_val.x));
      tmp.y = T(static_cast<float>(x_v.y) +
                static_cast<float>(local_val[i].y) * s_variance1 * static_cast<float>(gamma_val.y));
      tmp.z = T(static_cast<float>(x_v.z) +
                static_cast<float>(local_val[i].z) * s_variance1 * static_cast<float>(gamma_val.z));
      tmp.w = T(static_cast<float>(x_v.w) +
                static_cast<float>(local_val[i].w) * s_variance1 * static_cast<float>(gamma_val.w));
      intermediate[i] = tmp;
      output1[offset + index] = tmp;
    }
  }
  __syncthreads();

  // Step 2: Apply second RMSNorm to intermediate result
  local_sums[0] = 0.0f;
#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i * bdimx + tid;
    if (index < n_4) {
      local_sums[0] += static_cast<float>(intermediate[i].x) * static_cast<float>(intermediate[i].x) +
                       static_cast<float>(intermediate[i].y) * static_cast<float>(intermediate[i].y) +
                       static_cast<float>(intermediate[i].z) * static_cast<float>(intermediate[i].z) +
                       static_cast<float>(intermediate[i].w) * static_cast<float>(intermediate[i].w);
    }
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance2 = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

  // Apply second RMSNorm
#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i * bdimx + tid;
    if (index < n_4) {
      const T4 gamma_val = affine2 ? gamma2[index] : T4{T(1.0f), T(1.0f), T(1.0f), T(1.0f)};
      T4 tmp;
      tmp.x = T(static_cast<float>(intermediate[i].x) * s_variance2 * static_cast<float>(gamma_val.x));
      tmp.y = T(static_cast<float>(intermediate[i].y) * s_variance2 * static_cast<float>(gamma_val.y));
      tmp.z = T(static_cast<float>(intermediate[i].z) * s_variance2 * static_cast<float>(gamma_val.z));
      tmp.w = T(static_cast<float>(intermediate[i].w) * s_variance2 * static_cast<float>(gamma_val.w));
      output2[offset + index] = tmp;
    }
  }
}

// Kernel: RMSNorm(x) + Add -> output
// This fuses: x = x + self.ffn_norm2(ffn_out)
template <typename T4, typename T, int ITEM_PER_THREAD>
__global__ void fused_rmsnorm_add_kernel(
    T4* __restrict__ output,
    const T4* __restrict__ x,
    const T4* __restrict__ residual,
    const T4* __restrict__ gamma,
    const int m,
    const int n,
    bool affine,
    float eps) {
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_variance;
  float local_sums[1] = {0.0f};
  T4 local_val[ITEM_PER_THREAD];
  const int n_4 = n / 4;
  const int offset = m_idx * n_4;
  const T4 zero = {T(0.0f), T(0.0f), T(0.0f), T(0.0f)};

  // Compute RMS of residual
#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i * bdimx + tid;
    if (index < n_4) {
      const T4 r_v = residual[offset + index];
      local_val[i] = r_v;
      local_sums[0] += static_cast<float>(r_v.x) * static_cast<float>(r_v.x) +
                       static_cast<float>(r_v.y) * static_cast<float>(r_v.y) +
                       static_cast<float>(r_v.z) * static_cast<float>(r_v.z) +
                       static_cast<float>(r_v.w) * static_cast<float>(r_v.w);
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
    s_variance = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

  // Apply RMSNorm and add x
#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i * bdimx + tid;
    if (index < n_4) {
      const T4 gamma_val = affine ? gamma[index] : T4{T(1.0f), T(1.0f), T(1.0f), T(1.0f)};
      const T4 x_v = x[offset + index];
      T4 tmp;
      tmp.x = T(static_cast<float>(x_v.x) +
                static_cast<float>(local_val[i].x) * s_variance * static_cast<float>(gamma_val.x));
      tmp.y = T(static_cast<float>(x_v.y) +
                static_cast<float>(local_val[i].y) * s_variance * static_cast<float>(gamma_val.y));
      tmp.z = T(static_cast<float>(x_v.z) +
                static_cast<float>(local_val[i].z) * s_variance * static_cast<float>(gamma_val.z));
      tmp.w = T(static_cast<float>(x_v.w) +
                static_cast<float>(local_val[i].w) * s_variance * static_cast<float>(gamma_val.w));
      output[offset + index] = tmp;
    }
  }
}

// Host function for fused_rmsnorm_add_rmsnorm
void fused_rmsnorm_add_rmsnorm(
    ffi::TensorView output1,
    ffi::TensorView output2,
    const ffi::TensorView& x,
    const ffi::TensorView& residual,
    const ffi::Optional<ffi::TensorView>& gamma1_opt,
    const ffi::Optional<ffi::TensorView>& gamma2_opt,
    double eps) {
  using namespace host;

  SymbolicSize M_ = {"M"};
  SymbolicSize N_ = {"N"};
  SymbolicDevice device_;
  TensorMatcher({M_, N_})
      .with_dtype<float, half, nv_bfloat16>()
      .with_strides({N_, 1})
      .with_device<kDLCUDA>(device_)
      .verify(x)
      .verify(residual)
      .verify(output1)
      .verify(output2);

  RuntimeCheck(x.dtype() == residual.dtype(), "x and residual must have same dtype");
  const int64_t M = x.size(0);
  const int64_t N = x.size(1);
  RuntimeCheck(N % 4 == 0, "N must be divisible by 4");

  bool affine1 = gamma1_opt.has_value();
  bool affine2 = gamma2_opt.has_value();
  const void* gamma1_ptr = affine1 ? gamma1_opt.value().data_ptr() : nullptr;
  const void* gamma2_ptr = affine2 ? gamma2_opt.value().data_ptr() : nullptr;

  if (affine1) {
    TensorMatcher({N_})
        .with_dtype<float, half, nv_bfloat16>()
        .with_device<kDLCUDA>(device_)
        .verify(gamma1_opt.value());
  }
  if (affine2) {
    TensorMatcher({N_})
        .with_dtype<float, half, nv_bfloat16>()
        .with_device<kDLCUDA>(device_)
        .verify(gamma2_opt.value());
  }

  dim3 grid(M);
  dim3 block(std::min((int)(N / 4), 1024));

  auto dispatch = [&](auto dtype_tag, auto ipt_tag) {
    using T4 = typename decltype(dtype_tag)::T4;
    using T = typename decltype(dtype_tag)::T;
    using IPT = decltype(ipt_tag);
    LaunchKernel(grid, block, x.device())(
        fused_rmsnorm_add_rmsnorm_kernel<T4, T, IPT::value>,
        (T4*)output1.data_ptr(),
        (T4*)output2.data_ptr(),
        (const T4*)x.data_ptr(),
        (const T4*)residual.data_ptr(),
        (const T4*)gamma1_ptr,
        (const T4*)gamma2_ptr,
        (int)M,
        (int)N,
        affine1,
        affine2,
        (float)eps);
  };

  auto dispatch_ipt = [&](auto dtype_tag) {
    if (N <= 4096) {
      dispatch(dtype_tag, ItemPerThreadTag<1>{});
    } else {
      dispatch(dtype_tag, ItemPerThreadTag<8>{});
    }
  };

  const auto& dtype = x.dtype();
  if (dtype.code == kDLFloat && dtype.bits == 32) {
    dispatch_ipt(DTypeTag<float4, float>{});
  } else if (dtype.code == kDLFloat && dtype.bits == 16) {
    dispatch_ipt(DTypeTag<half4, half>{});
  } else if (dtype.code == kDLBfloat && dtype.bits == 16) {
    dispatch_ipt(DTypeTag<bf16_4, cutlass::bfloat16_t>{});
  } else {
    RuntimeCheck(false, "Unsupported dtype. Use float32, float16, or bfloat16.");
  }
}

// Host function for fused_rmsnorm_add
void fused_rmsnorm_add(
    ffi::TensorView output,
    const ffi::TensorView& x,
    const ffi::TensorView& residual,
    const ffi::Optional<ffi::TensorView>& gamma_opt,
    double eps) {
  using namespace host;

  SymbolicSize M_ = {"M"};
  SymbolicSize N_ = {"N"};
  SymbolicDevice device_;
  TensorMatcher({M_, N_})
      .with_dtype<float, half, nv_bfloat16>()
      .with_strides({N_, 1})
      .with_device<kDLCUDA>(device_)
      .verify(x)
      .verify(residual)
      .verify(output);

  RuntimeCheck(x.dtype() == residual.dtype(), "x and residual must have same dtype");
  const int64_t M = x.size(0);
  const int64_t N = x.size(1);
  RuntimeCheck(N % 4 == 0, "N must be divisible by 4");

  bool affine = gamma_opt.has_value();
  const void* gamma_ptr = affine ? gamma_opt.value().data_ptr() : nullptr;

  if (affine) {
    TensorMatcher({N_})
        .with_dtype<float, half, nv_bfloat16>()
        .with_device<kDLCUDA>(device_)
        .verify(gamma_opt.value());
  }

  dim3 grid(M);
  dim3 block(std::min((int)(N / 4), 1024));

  auto dispatch = [&](auto dtype_tag, auto ipt_tag) {
    using T4 = typename decltype(dtype_tag)::T4;
    using T = typename decltype(dtype_tag)::T;
    using IPT = decltype(ipt_tag);
    LaunchKernel(grid, block, x.device())(
        fused_rmsnorm_add_kernel<T4, T, IPT::value>,
        (T4*)output.data_ptr(),
        (const T4*)x.data_ptr(),
        (const T4*)residual.data_ptr(),
        (const T4*)gamma_ptr,
        (int)M,
        (int)N,
        affine,
        (float)eps);
  };

  auto dispatch_ipt = [&](auto dtype_tag) {
    if (N <= 4096) {
      dispatch(dtype_tag, ItemPerThreadTag<1>{});
    } else {
      dispatch(dtype_tag, ItemPerThreadTag<8>{});
    }
  };

  const auto& dtype = x.dtype();
  if (dtype.code == kDLFloat && dtype.bits == 32) {
    dispatch_ipt(DTypeTag<float4, float>{});
  } else if (dtype.code == kDLFloat && dtype.bits == 16) {
    dispatch_ipt(DTypeTag<half4, half>{});
  } else if (dtype.code == kDLBfloat && dtype.bits == 16) {
    dispatch_ipt(DTypeTag<bf16_4, cutlass::bfloat16_t>{});
  } else {
    RuntimeCheck(false, "Unsupported dtype. Use float32, float16, or bfloat16.");
  }
}

}  // namespace
