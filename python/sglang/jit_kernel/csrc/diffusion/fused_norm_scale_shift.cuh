/* Copyright 2025 SGLang Team. */

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For div_ceil, RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <sgl_kernel/impl/norm.cuh>
#include <sgl_kernel/impl/norm_fusion.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <cuda_fp16.h>

namespace {

using host::norm::NormEnum;
using host::norm_fusion::IndexEnum;

template <typename T>
using Vec4 = device::AlignedVector<T, 4>;

template <int V>
struct ItemPerThreadTag {
  static constexpr int value = V;
};

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

// compute sum of elements in a Vec4
template <typename T>
__device__ __forceinline__ float vec4_sum(const Vec4<T>& v) {
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    sum += static_cast<float>(v[j]);
  }
  return sum;
}

// compute sum of squares of elements in a Vec4
template <typename T>
__device__ __forceinline__ float vec4_sum_sq(const Vec4<T>& v) {
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float val = static_cast<float>(v[j]);
    sum += val * val;
  }
  return sum;
}

// compute sum of squared differences from mean
template <typename T>
__device__ __forceinline__ float vec4_variance_sum(const Vec4<T>& v, float mean) {
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float diff = static_cast<float>(v[j]) - mean;
    sum += diff * diff;
  }
  return sum;
}

template <typename T, int ITEM_PER_THREAD, NormEnum norm_enum, IndexEnum scale_index_enum, IndexEnum shift_index_enum>
__global__ void norm_fused_scale_shift_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const T* __restrict__ scale,
    const T* __restrict__ shift,
    const int B,
    const int S,
    const int F,
    const int D,
    bool affine,
    float eps) {
  using namespace device;

  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int bdimx = blockDim.x;
  const int D4 = D >> 2;
  const int b_id = bidx / S;
  const int s_id = bidx % S;
  const int offset = bidx * D4;
  const int scale_offset = norm_fusion::get_offset<scale_index_enum>(S, F, b_id, s_id) * D4;
  const int shift_offset = norm_fusion::get_offset<shift_index_enum>(S, F, b_id, s_id) * D4;

  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};
  Vec4<T> local_val[ITEM_PER_THREAD];

  // Pass 1: Load input and compute sum (LayerNorm) or sum_sq (RMSNorm)
#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; ++i) {
    const int index = i * bdimx + tidx;
    if (index < D4) {
      local_val[i].load(input, offset + index);
    } else {
      local_val[i].fill(T(0.0f));
    }
    if constexpr (norm_enum == NormEnum::LayerNorm) {
      local_sums[0] += vec4_sum(local_val[i]);
    } else {
      local_sums[0] += vec4_sum_sq(local_val[i]);
    }
  }

  // Reduce and compute mean
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / D;
  }
  __syncthreads();

  // Pass 2 (LayerNorm only): Compute variance
  if constexpr (norm_enum == NormEnum::LayerNorm) {
    local_sums[0] = 0.0f;
#pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; ++i) {
      const int index = i * bdimx + tidx;
      if (index < D4) {
        local_sums[0] += vec4_variance_sum(local_val[i], s_mean);
      }
    }
    if (blockDim.x <= 32) {
      warpReduceSum<float, 1>(local_sums);
    } else {
      blockReduceSum<float, 1>(local_sums);
    }
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / D + eps);
  }
  __syncthreads();

  // Pre-load scalar scale/shift if needed
  float scalar_scale = 0.0f, scalar_shift = 0.0f;
  if constexpr (scale_index_enum == IndexEnum::Scalar) scalar_scale = static_cast<float>(scale[0]);
  if constexpr (shift_index_enum == IndexEnum::Scalar) scalar_shift = static_cast<float>(shift[0]);

    // Pass 3: Apply normalization, affine, scale/shift
#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; ++i) {
    const int index = i * bdimx + tidx;
    if (index < D4) {
      Vec4<T> gamma_val, beta_val;
      if (affine) {
        gamma_val.load(gamma, index);
        beta_val.load(beta, index);
      } else {
        gamma_val.fill(T(1.0f));
        beta_val.fill(T(0.0f));
      }

      // Load scale/shift based on IndexEnum
      Vec4<T> scale_val, shift_val;
      if constexpr (scale_index_enum == IndexEnum::Scalar) {
        scale_val.fill(T(scale[0]));
      } else {
        scale_val.load(scale, scale_offset + index);
      }
      if constexpr (shift_index_enum == IndexEnum::Scalar) {
        shift_val.fill(T(shift[0]));
      } else {
        shift_val.load(shift, shift_offset + index);
      }

      Vec4<T> tmp;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float normalized;
        if constexpr (norm_enum == NormEnum::LayerNorm) {
          normalized = (static_cast<float>(local_val[i][j]) - s_mean) * s_variance;
        } else {
          normalized = static_cast<float>(local_val[i][j]) * s_variance;
        }
        float affine_out = normalized * static_cast<float>(gamma_val[j]);
        if constexpr (norm_enum == NormEnum::LayerNorm) {
          affine_out += static_cast<float>(beta_val[j]);
        }
        tmp[j] = T(affine_out * (1.0f + static_cast<float>(scale_val[j])) + static_cast<float>(shift_val[j]));
      }
      tmp.store(output, offset + index);
    }
  }
}

template <NormEnum norm_enum, typename T, IndexEnum scale_index_enum, IndexEnum shift_index_enum>
void fused_norm_scale_shift(
    tvm::ffi::TensorView out,
    const tvm::ffi::TensorView x,
    const tvm::ffi::Optional<tvm::ffi::TensorView> gamma_opt,
    const tvm::ffi::Optional<tvm::ffi::TensorView> beta_opt,
    const tvm::ffi::TensorView scale,
    const tvm::ffi::TensorView shift,
    double eps) {
  using namespace host;

  static_assert(
      std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>,
      "Only support float, fp16, bf16");
  static_assert(
      norm_enum == NormEnum::LayerNorm || norm_enum == NormEnum::RMSNorm, "norm_enum must be layernorm or rmsnorm.");

  host::norm_fusion::Matcher<T> checker;
  checker.template match<IndexEnum::NoBroadcast>(out);
  checker.template match<IndexEnum::NoBroadcast>(x);
  checker.template match<scale_index_enum>(scale);
  checker.template match<shift_index_enum>(shift);
  bool affine = gamma_opt.has_value();
  if (affine) {
    checker.template match<IndexEnum::BroadcastBS>(gamma_opt.value());
    if (beta_opt.has_value()) {
      checker.template match<IndexEnum::BroadcastBS>(beta_opt.value());
    }
  }

  const auto B = checker.B_.unwrap();
  const auto S = checker.S_.unwrap();
  const auto F = checker.has_value_F ? checker.F_.unwrap() : 0;
  const auto D = checker.D_.unwrap();
  RuntimeCheck((D % 4) == 0, "D must be divisible by 4");
  dim3 grid(B * S);
  dim3 block(0);
  // Configure thread block
  if (D <= 4096) {
    block.x = (int)((D / 4 + 31) / 32 * 32);
  } else {
    block.x = (int)(((D + 7) / 8 + 31) / 32 * 32);
  }
  if (block.x > 1024) block.x = 1024;

  auto gamma_ptr = gamma_opt.has_value() ? gamma_opt.value().data_ptr() : nullptr;
  auto beta_ptr = beta_opt.has_value() ? beta_opt.value().data_ptr() : nullptr;

  // Dispatch
  auto dispatch = [&]() {
    auto dispatch_ipt = [&](auto ipt_tag) {
      LaunchKernel(grid, block, x.device())(
          norm_fused_scale_shift_kernel<T, decltype(ipt_tag)::value, norm_enum, scale_index_enum, shift_index_enum>,
          (T*)out.data_ptr(),
          (const T*)x.data_ptr(),
          (const T*)gamma_ptr,
          (const T*)beta_ptr,
          (const T*)scale.data_ptr(),
          (const T*)shift.data_ptr(),
          B,
          S,
          F,
          D,
          affine,
          eps);
    };

    if (D <= 4096) {
      dispatch_ipt(ItemPerThreadTag<1>{});
    } else {
      dispatch_ipt(ItemPerThreadTag<8>{});
    }
  };

  dispatch();
}

// Unified kernel for residual + gate + norm + scale/shift
template <
    typename T,
    int ITEM_PER_THREAD,
    NormEnum norm_enum,
    IndexEnum scale_index_enum,
    IndexEnum shift_index_enum,
    IndexEnum gate_index_enum>
__global__ void norm_fused_res_gate_scale_shift_kernel(
    T* __restrict__ output,
    T* __restrict__ residual_out,
    const T* __restrict__ x,
    const T* __restrict__ residual,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const T* __restrict__ scale,
    const T* __restrict__ shift,
    const T* __restrict__ gate,  // nullptr means NoGate (use 1.0)
    const int B,
    const int S,
    const int F,
    const int D,
    bool affine,
    float eps) {
  using namespace device;

  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int bdimx = blockDim.x;
  const int D4 = D >> 2;
  const int b_id = bidx / S;
  const int s_id = bidx % S;
  const int offset = bidx * D4;
  const int scale_offset = norm_fusion::get_offset<scale_index_enum>(S, F, b_id, s_id) * D4;
  const int shift_offset = norm_fusion::get_offset<shift_index_enum>(S, F, b_id, s_id) * D4;
  const int gate_offset = norm_fusion::get_offset<gate_index_enum>(S, F, b_id, s_id) * D4;

  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};
  Vec4<T> local_val[ITEM_PER_THREAD];

  // Pass 1: Load x, residual, gate and compute residual + x * gate
#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; ++i) {
    const int index = i * bdimx + tidx;
    if (index < D4) {
      Vec4<T> x_v, r_v, g_v;
      x_v.load(x, offset + index);
      r_v.load(residual, offset + index);
      // Load gate based on IndexEnum (nullptr means NoGate)
      if constexpr (gate_index_enum == IndexEnum::NotATensor) {
        g_v.fill(T(1.0f));
      } else if constexpr (gate_index_enum == IndexEnum::Scalar) {
        g_v.fill(T(gate[0]));
      } else {
        g_v.load(gate, gate_offset + index);
      }

      Vec4<T> sum_v;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        sum_v[j] = T(static_cast<float>(r_v[j]) + static_cast<float>(x_v[j]) * static_cast<float>(g_v[j]));
      }
      local_val[i] = sum_v;

      if (residual_out != nullptr) {
        sum_v.store(residual_out, offset + index);
      }

      if constexpr (norm_enum == NormEnum::LayerNorm) {
        local_sums[0] += vec4_sum(sum_v);
      } else {
        local_sums[0] += vec4_sum_sq(sum_v);
      }
    } else {
      local_val[i].fill(T(0.0f));
    }
  }

  // Reduce and compute mean
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  } else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / D;
  }
  __syncthreads();

  // Pass 2 (LayerNorm only): Compute variance
  if constexpr (norm_enum == NormEnum::LayerNorm) {
    local_sums[0] = 0.0f;
#pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; ++i) {
      const int index = i * bdimx + tidx;
      if (index < D4) {
        local_sums[0] += vec4_variance_sum(local_val[i], s_mean);
      }
    }
    if (blockDim.x <= 32) {
      warpReduceSum<float, 1>(local_sums);
    } else {
      blockReduceSum<float, 1>(local_sums);
    }
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / D + eps);
  }
  __syncthreads();

  // Pre-load scalar scale/shift if needed
  float scalar_scale = 0.0f, scalar_shift = 0.0f;
  if constexpr (scale_index_enum == IndexEnum::Scalar) scalar_scale = static_cast<float>(scale[0]);
  if constexpr (shift_index_enum == IndexEnum::Scalar) scalar_shift = static_cast<float>(shift[0]);

    // Pass 3: Apply normalization, affine, scale/shift
#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; ++i) {
    const int index = i * bdimx + tidx;
    if (index < D4) {
      Vec4<T> gamma_val, beta_val;
      if (affine) {
        gamma_val.load(gamma, index);
        beta_val.load(beta, index);
      } else {
        gamma_val.fill(T(1.0f));
        beta_val.fill(T(0.0f));
      }

      // Load scale/shift based on IndexEnum
      Vec4<T> scale_val, shift_val;
      if constexpr (scale_index_enum == IndexEnum::Scalar) {
        scale_val.fill(T(scale[0]));
      } else {
        scale_val.load(scale, scale_offset + index);
      }
      if constexpr (shift_index_enum == IndexEnum::Scalar) {
        shift_val.fill(T(shift[0]));
      } else {
        shift_val.load(shift, shift_offset + index);
      }
      Vec4<T> tmp;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float normalized;
        if constexpr (norm_enum == NormEnum::LayerNorm) {
          normalized = (static_cast<float>(local_val[i][j]) - s_mean) * s_variance;
        } else {
          normalized = static_cast<float>(local_val[i][j]) * s_variance;
        }
        float affine_out = normalized * static_cast<float>(gamma_val[j]);
        if constexpr (norm_enum == NormEnum::LayerNorm) {
          affine_out += static_cast<float>(beta_val[j]);
        }
        tmp[j] = T(affine_out * (1.0f + static_cast<float>(scale_val[j])) + static_cast<float>(shift_val[j]));
      }
      tmp.store(output, offset + index);
    }
  }
}

template <
    NormEnum norm_enum,
    typename T,
    IndexEnum scale_index_enum,
    IndexEnum shift_index_enum,
    IndexEnum gate_index_enum>
void fused_scale_residual_norm_scale_shift(
    tvm::ffi::TensorView y,
    tvm::ffi::TensorView residual_out,
    const tvm::ffi::TensorView residual,
    const tvm::ffi::TensorView x,
    const tvm::ffi::Optional<tvm::ffi::TensorView> gate_opt,
    const tvm::ffi::Optional<tvm::ffi::TensorView> gamma_opt,
    const tvm::ffi::Optional<tvm::ffi::TensorView> beta_opt,
    const tvm::ffi::TensorView scale,
    const tvm::ffi::TensorView shift,
    double eps) {
  using namespace host;

  static_assert(
      std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>,
      "Only support float, fp16, bf16");
  static_assert(
      norm_enum == NormEnum::LayerNorm || norm_enum == NormEnum::RMSNorm, "norm_enum must be layernorm or rmsnorm.");

  norm_fusion::Matcher<T> checker;
  checker.template match<IndexEnum::NoBroadcast>(y);
  checker.template match<IndexEnum::NoBroadcast>(residual_out);
  checker.template match<IndexEnum::NoBroadcast>(x);
  checker.template match<IndexEnum::NoBroadcast>(residual);
  checker.template match<scale_index_enum>(scale);
  checker.template match<shift_index_enum>(shift);
  if (gate_opt.has_value()) {
    checker.template match<gate_index_enum>(gate_opt.value());
  }
  bool affine = gamma_opt.has_value();
  if (affine) {
    checker.template match<IndexEnum::BroadcastBS>(gamma_opt.value());
    if (beta_opt.has_value()) {
      checker.template match<IndexEnum::BroadcastBS>(beta_opt.value());
    }
  }

  const auto B = checker.B_.unwrap();
  const auto S = checker.S_.unwrap();
  const auto F = checker.has_value_F ? checker.F_.unwrap() : 0;
  const auto D = checker.D_.unwrap();
  RuntimeCheck((D % 4) == 0, "D must be divisible by 4");
  dim3 grid(B * S);
  dim3 block(0);
  // Configure thread block
  if (D <= 4096) {
    block.x = (int)((D / 4 + 31) / 32 * 32);
  } else {
    block.x = (int)(((D + 7) / 8 + 31) / 32 * 32);
  }
  if (block.x > 1024) block.x = 1024;

  auto gamma_ptr = gamma_opt.has_value() ? gamma_opt.value().data_ptr() : nullptr;
  auto beta_ptr = beta_opt.has_value() ? beta_opt.value().data_ptr() : nullptr;
  auto gate_ptr = gate_opt.has_value() ? gate_opt.value().data_ptr() : nullptr;

  // Dispatch
  auto launch = [&]() {
    auto dispatch_ipt = [&](auto ipt_tag) {
      LaunchKernel(grid, block, x.device())(
          norm_fused_res_gate_scale_shift_kernel<
              T,
              decltype(ipt_tag)::value,
              norm_enum,
              scale_index_enum,
              shift_index_enum,
              gate_index_enum>,
          (T*)y.data_ptr(),
          (T*)residual_out.data_ptr(),
          (const T*)x.data_ptr(),
          (const T*)residual.data_ptr(),
          (const T*)gamma_ptr,
          (const T*)beta_ptr,
          (const T*)scale.data_ptr(),
          (const T*)shift.data_ptr(),
          (const T*)gate_ptr,
          B,
          S,
          F,
          D,
          affine,
          eps);
    };

    if (D <= 4096) {
      dispatch_ipt(ItemPerThreadTag<1>{});
    } else {
      dispatch_ipt(ItemPerThreadTag<8>{});
    }
  };
  launch();
}
}  // namespace
