/* Copyright 2025 SGLang Team. */
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include "cutlass/numeric_types.h"
#include <cuda_fp16.h>

namespace {

namespace ffi = tvm::ffi;

template <typename T>
using Vec4 = device::aligned_vector<T, 4>;

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

template <typename T_>
struct DTypeTag {
  using T = T_;
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

// IndexMode for scale/shift/gate tensor indexing
// Scalar: [1] - single value for all
// Broadcast1N: [1, N] or [1, 1, N] - broadcast across M dimension
// FullMN: [M, N] - per-row values
// BroadcastB1N: [B, 1, N] - broadcast within each batch (for gate only)
// FullBF1N: [B, F, 1, N] - per-(batch, frame) values
enum class IndexMode : int {
  Scalar = 0,
  Broadcast1N = 1,
  FullMN = 2,
  BroadcastB1N = 3,  // [B, 1, N] gate - per-batch broadcast
  FullBF1N = 4,
};

template <IndexMode IM>
struct IndexModeTag {
  static constexpr IndexMode value = IM;
};

template <typename T, int ITEM_PER_THREAD, NormType norm_type, IndexMode scale_mode, IndexMode shift_mode>
__global__ void norm_fused_scale_shift_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const T* __restrict__ scale,
    const T* __restrict__ shift,
    const int m,
    const int n,
    const int F,
    const int L,
    bool affine,
    float eps) {
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};
  Vec4<T> local_val[ITEM_PER_THREAD];
  const int n_4 = n / 4;
  const int offset = m_idx * n_4;

  // Compute (b, f) indices only when needed for 4D indexing
  int scale_base = 0, shift_base = 0;
  if constexpr (scale_mode == IndexMode::FullBF1N || shift_mode == IndexMode::FullBF1N) {
    const int rows_per_b = F * L;
    const int b = m_idx / rows_per_b;
    const int f = (m_idx % rows_per_b) / L;
    const int bf_offset = (b * F + f) * n_4;
    if constexpr (scale_mode == IndexMode::FullBF1N) scale_base = bf_offset;
    if constexpr (shift_mode == IndexMode::FullBF1N) shift_base = bf_offset;
  }

  // Pass 1: Load input and compute sum (LayerNorm) or sum_sq (RMSNorm)
#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; ++i) {
    const int index = i * bdimx + tid;
    if (index < n_4) {
      local_val[i].load(input, offset + index);
    } else {
      local_val[i].fill(T(0.0f));
    }
    if constexpr (norm_type == NormType::LayerNorm) {
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
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  // Pass 2 (LayerNorm only): Compute variance
  if constexpr (norm_type == NormType::LayerNorm) {
    local_sums[0] = 0.0f;
#pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; ++i) {
      const int index = i * bdimx + tid;
      if (index < n_4) {
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
    s_variance = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

  // Pre-load scalar scale/shift if needed
  float scalar_scale = 0.0f, scalar_shift = 0.0f;
  if constexpr (scale_mode == IndexMode::Scalar) scalar_scale = static_cast<float>(scale[0]);
  if constexpr (shift_mode == IndexMode::Scalar) scalar_shift = static_cast<float>(shift[0]);

    // Pass 3: Apply normalization, affine, scale/shift
#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; ++i) {
    const int index = i * bdimx + tid;
    if (index < n_4) {
      Vec4<T> gamma_val, beta_val;
      if (affine) {
        gamma_val.load(gamma, index);
        beta_val.load(beta, index);
      } else {
        gamma_val.fill(T(1.0f));
        beta_val.fill(T(0.0f));
      }

      // Load scale/shift based on IndexMode
      Vec4<T> scale_val, shift_val;
      if constexpr (scale_mode == IndexMode::Scalar) {
        scale_val.fill(T(scalar_scale));
      } else if constexpr (scale_mode == IndexMode::Broadcast1N) {
        scale_val.load(scale, index);
      } else if constexpr (scale_mode == IndexMode::FullMN) {
        scale_val.load(scale, offset + index);
      } else {  // FullBF1N
        scale_val.load(scale, scale_base + index);
      }

      if constexpr (shift_mode == IndexMode::Scalar) {
        shift_val.fill(T(scalar_shift));
      } else if constexpr (shift_mode == IndexMode::Broadcast1N) {
        shift_val.load(shift, index);
      } else if constexpr (shift_mode == IndexMode::FullMN) {
        shift_val.load(shift, offset + index);
      } else {  // FullBF1N
        shift_val.load(shift, shift_base + index);
      }

      Vec4<T> tmp;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float normalized;
        if constexpr (norm_type == NormType::LayerNorm) {
          normalized = (static_cast<float>(local_val[i][j]) - s_mean) * s_variance;
        } else {
          normalized = static_cast<float>(local_val[i][j]) * s_variance;
        }
        float affine_out = normalized * static_cast<float>(gamma_val[j]);
        if constexpr (norm_type == NormType::LayerNorm) {
          affine_out += static_cast<float>(beta_val[j]);
        }
        tmp[j] = T(affine_out * (1.0f + static_cast<float>(scale_val[j])) + static_cast<float>(shift_val[j]));
      }
      tmp.store(output, offset + index);
    }
  }
}

// Helper to determine IndexMode from tensor shape
inline IndexMode get_index_mode(const ffi::TensorView& t, int64_t M, int64_t N) {
  if (t.ndim() == 1 && t.numel() == 1) return IndexMode::Scalar;
  if (t.ndim() == 3 && t.size(0) == 1 && t.size(1) == 1) return IndexMode::Broadcast1N;
  if (t.ndim() == 2 && t.size(0) == 1) return IndexMode::Broadcast1N;
  if (t.ndim() == 2 && t.size(0) == M) return IndexMode::FullMN;
  if (t.ndim() == 4) return IndexMode::FullBF1N;
  return IndexMode::FullMN;  // fallback
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

  bool affine = gamma_opt.has_value();
  auto gamma_ptr = gamma_opt.has_value() ? gamma_opt.value().data_ptr() : nullptr;
  auto beta_ptr = beta_opt.has_value() ? beta_opt.value().data_ptr() : nullptr;

  const int64_t M = x.size(0);
  const int64_t N = x.size(1);
  dim3 grid((unsigned)M);
  dim3 block(0);

  // Determine IndexMode for scale and shift
  IndexMode scale_mode = get_index_mode(scale, M, N);
  IndexMode shift_mode = get_index_mode(shift, M, N);

  // Compute F and L for 4D indexing (needed even if not used, passed as dummy)
  int F = 1, L = (int)M;
  if (scale_mode == IndexMode::FullBF1N || shift_mode == IndexMode::FullBF1N) {
    const auto& t4d = (scale.ndim() == 4) ? scale : shift;
    const int64_t B = t4d.size(0);
    F = (int)t4d.size(1);
    L = (int)(M / (B * F));
  }

  // Dispatch helper with IndexMode
  auto dispatch = [&](auto scale_mode_tag, auto shift_mode_tag) {
    auto dispatch_dtype = [&](auto dtype_tag) {
      auto dispatch_ipt = [&](auto ipt_tag) {
        auto dispatch_norm = [&](auto norm_tag) {
          using T = typename decltype(dtype_tag)::T;
          LaunchKernel(grid, block, x.device())(
              norm_fused_scale_shift_kernel<
                  T,
                  decltype(ipt_tag)::value,
                  decltype(norm_tag)::value,
                  decltype(scale_mode_tag)::value,
                  decltype(shift_mode_tag)::value>,
              (T*)out.data_ptr(),
              (const T*)x.data_ptr(),
              (const T*)gamma_ptr,
              (const T*)beta_ptr,
              (const T*)scale.data_ptr(),
              (const T*)shift.data_ptr(),
              (int)M,
              (int)N,
              F,
              L,
              affine,
              eps);
        };

        if (norm_type == NormType::LayerNorm) {
          dispatch_norm(NormTag<NormType::LayerNorm>{});
        } else {
          dispatch_norm(NormTag<NormType::RMSNorm>{});
        }
      };

      if (N <= 4096) {
        block.x = (int)((N / 4 + 31) / 32 * 32);
        if (block.x > 1024) block.x = 1024;
        dispatch_ipt(ItemPerThreadTag<1>{});
      } else {
        block.x = (int)(((N + 7) / 8 + 31) / 32 * 32);
        if (block.x > 1024) block.x = 1024;
        dispatch_ipt(ItemPerThreadTag<8>{});
      }
    };

    const auto& dtype = x.dtype();
    if (is_type<float>(dtype)) {
      dispatch_dtype(DTypeTag<float>{});
    } else if (is_type<half>(dtype)) {
      dispatch_dtype(DTypeTag<half>{});
    } else if (is_type<nv_bfloat16>(dtype)) {
      dispatch_dtype(DTypeTag<cutlass::bfloat16_t>{});
    } else {
      RuntimeCheck(false, "Unsupported dtype. Use float32, float16, or bfloat16.");
    }
  };

  // Dispatch based on IndexMode combinations
  // We only instantiate commonly used combinations to limit template explosion
  auto dispatch_shift = [&](auto scale_mode_tag) {
    switch (shift_mode) {
      case IndexMode::Scalar:
        dispatch(scale_mode_tag, IndexModeTag<IndexMode::Scalar>{});
        break;
      case IndexMode::Broadcast1N:
        dispatch(scale_mode_tag, IndexModeTag<IndexMode::Broadcast1N>{});
        break;
      case IndexMode::FullMN:
        dispatch(scale_mode_tag, IndexModeTag<IndexMode::FullMN>{});
        break;
      case IndexMode::FullBF1N:
        dispatch(scale_mode_tag, IndexModeTag<IndexMode::FullBF1N>{});
        break;
    }
  };

  switch (scale_mode) {
    case IndexMode::Scalar:
      dispatch_shift(IndexModeTag<IndexMode::Scalar>{});
      break;
    case IndexMode::Broadcast1N:
      dispatch_shift(IndexModeTag<IndexMode::Broadcast1N>{});
      break;
    case IndexMode::FullMN:
      dispatch_shift(IndexModeTag<IndexMode::FullMN>{});
      break;
    case IndexMode::FullBF1N:
      dispatch_shift(IndexModeTag<IndexMode::FullBF1N>{});
      break;
  }
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
  TensorMatcher({M_, N_})  // 2D tensor, must be contiguous
      .with_dtype<float, half, nv_bfloat16>()
      .with_strides({N_, 1})
      .with_device<kDLCUDA>()
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
        .with_device<kDLCUDA>()
        .verify(gamma);
    RuntimeCheck(x.dtype() == gamma.dtype(), "x, gamma must have same dtype");
    if (beta_opt.has_value()) {
      const auto& beta = beta_opt.value();
      TensorMatcher({N_})  // 1D tensor, must be contiguous
          .with_dtype<float, half, nv_bfloat16>()
          .with_device<kDLCUDA>()
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

// Unified kernel for residual + gate + norm + scale/shift
template <
    typename T,
    int ITEM_PER_THREAD,
    NormType norm_type,
    IndexMode scale_mode,
    IndexMode shift_mode,
    IndexMode gate_mode>
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
    const int m,
    const int n,
    const int F,
    const int L,
    bool affine,
    float eps) {
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};
  Vec4<T> local_val[ITEM_PER_THREAD];
  const int n_4 = n / 4;
  const int offset = m_idx * n_4;

  // Compute (b, f) indices for 4D indexing (FullBF1N mode) and BroadcastB1N gate
  int scale_base = 0, shift_base = 0, gate_base = 0;
  constexpr bool need_bf =
      (scale_mode == IndexMode::FullBF1N || shift_mode == IndexMode::FullBF1N || gate_mode == IndexMode::FullBF1N);
  constexpr bool need_b1n = (gate_mode == IndexMode::BroadcastB1N);
  if constexpr (need_bf) {
    const int rows_per_b = F * L;
    const int b = m_idx / rows_per_b;
    const int f = (m_idx % rows_per_b) / L;
    const int bf_offset = (b * F + f) * n_4;
    if constexpr (scale_mode == IndexMode::FullBF1N) scale_base = bf_offset;
    if constexpr (shift_mode == IndexMode::FullBF1N) shift_base = bf_offset;
    if constexpr (gate_mode == IndexMode::FullBF1N) gate_base = bf_offset;
  }

  // For BroadcastB1N mode with gate ([B, 1, N] - per-batch broadcast)
  // L here represents rows_per_b (number of rows per batch)
  if constexpr (need_b1n) {
    gate_base = (m_idx / L) * n_4;
  }

  // Pass 1: Load x, residual, gate and compute residual + x * gate
#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; ++i) {
    const int index = i * bdimx + tid;
    if (index < n_4) {
      Vec4<T> x_v, r_v, g_v;
      x_v.load(x, offset + index);
      r_v.load(residual, offset + index);

      // Load gate based on IndexMode (nullptr means NoGate)
      if (gate == nullptr) {
        g_v.fill(T(1.0f));
      } else if constexpr (gate_mode == IndexMode::Scalar) {
        g_v.load(gate, index);
      } else if constexpr (gate_mode == IndexMode::Broadcast1N) {
        // [1, N] or [1, 1, N] - same value for all rows
        g_v.load(gate, index);
      } else if constexpr (gate_mode == IndexMode::FullMN) {
        // [M, N] - per-row gate, use offset like scale/shift
        g_v.load(gate, offset + index);
      } else if constexpr (gate_mode == IndexMode::BroadcastB1N) {
        // [B, 1, N] - per-batch gate
        g_v.load(gate, gate_base + index);
      } else {  // FullBF1N
        g_v.load(gate, gate_base + index);
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

      if constexpr (norm_type == NormType::LayerNorm) {
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
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  // Pass 2 (LayerNorm only): Compute variance
  if constexpr (norm_type == NormType::LayerNorm) {
    local_sums[0] = 0.0f;
#pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; ++i) {
      const int index = i * bdimx + tid;
      if (index < n_4) {
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
    s_variance = rsqrtf(local_sums[0] / n + eps);
  }
  __syncthreads();

  // Pre-load scalar scale/shift if needed
  float scalar_scale = 0.0f, scalar_shift = 0.0f;
  if constexpr (scale_mode == IndexMode::Scalar) scalar_scale = static_cast<float>(scale[0]);
  if constexpr (shift_mode == IndexMode::Scalar) scalar_shift = static_cast<float>(shift[0]);

    // Pass 3: Apply normalization, affine, scale/shift
#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; ++i) {
    const int index = i * bdimx + tid;
    if (index < n_4) {
      Vec4<T> gamma_val, beta_val;
      if (affine) {
        gamma_val.load(gamma, index);
        beta_val.load(beta, index);
      } else {
        gamma_val.fill(T(1.0f));
        beta_val.fill(T(0.0f));
      }

      // Load scale/shift based on IndexMode
      Vec4<T> scale_val, shift_val;
      if constexpr (scale_mode == IndexMode::Scalar) {
        scale_val.fill(T(scalar_scale));
      } else if constexpr (scale_mode == IndexMode::Broadcast1N) {
        scale_val.load(scale, index);
      } else if constexpr (scale_mode == IndexMode::FullMN) {
        scale_val.load(scale, offset + index);
      } else {  // FullBF1N
        scale_val.load(scale, scale_base + index);
      }

      if constexpr (shift_mode == IndexMode::Scalar) {
        shift_val.fill(T(scalar_shift));
      } else if constexpr (shift_mode == IndexMode::Broadcast1N) {
        shift_val.load(shift, index);
      } else if constexpr (shift_mode == IndexMode::FullMN) {
        shift_val.load(shift, offset + index);
      } else {  // FullBF1N
        shift_val.load(shift, shift_base + index);
      }

      Vec4<T> tmp;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float normalized;
        if constexpr (norm_type == NormType::LayerNorm) {
          normalized = (static_cast<float>(local_val[i][j]) - s_mean) * s_variance;
        } else {
          normalized = static_cast<float>(local_val[i][j]) * s_variance;
        }
        float affine_out = normalized * static_cast<float>(gamma_val[j]);
        if constexpr (norm_type == NormType::LayerNorm) {
          affine_out += static_cast<float>(beta_val[j]);
        }
        tmp[j] = T(affine_out * (1.0f + static_cast<float>(scale_val[j])) + static_cast<float>(shift_val[j]));
      }
      tmp.store(output, offset + index);
    }
  }
}

// Helper to determine IndexMode for gate tensor
// For gate: NoGate -> nullptr, [1,N]/[1,1,N] -> Broadcast1N, [M,N] -> FullMN, [B,1,N] -> BroadcastB1N, [B,F,1,N] ->
// FullBF1N
inline IndexMode get_gate_index_mode(const ffi::Optional<ffi::TensorView>& gate_opt, int64_t M, int64_t N) {
  if (!gate_opt.has_value()) return IndexMode::Scalar;  // NoGate case, will use nullptr
  const auto& g = gate_opt.value();
  if (g.ndim() == 2 && g.size(0) == 1) return IndexMode::Broadcast1N;                    // [1, N]
  if (g.ndim() == 2 && g.size(0) == M) return IndexMode::FullMN;                         // [M, N] - per-row gate
  if (g.ndim() == 3 && g.size(0) == 1 && g.size(1) == 1) return IndexMode::Broadcast1N;  // [1, 1, N]
  if (g.ndim() == 3 && g.size(1) == 1) return IndexMode::BroadcastB1N;                   // [B, 1, N] - per-batch gate
  if (g.ndim() == 4) return IndexMode::FullBF1N;
  return IndexMode::FullMN;  // fallback for any 2D gate
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

  bool affine = gamma_opt.has_value();
  auto gamma_ptr = gamma_opt.has_value() ? gamma_opt.value().data_ptr() : nullptr;
  auto beta_ptr = beta_opt.has_value() ? beta_opt.value().data_ptr() : nullptr;
  auto gate_ptr = gate_opt.has_value() ? gate_opt.value().data_ptr() : nullptr;

  const int64_t M = x.size(0);
  const int64_t N = x.size(1);
  dim3 grid((unsigned)M);
  dim3 block(0);

  // Configure thread block
  if (N <= 4096) {
    block.x = (int)((N / 4 + 31) / 32 * 32);
  } else {
    block.x = (int)(((N + 7) / 8 + 31) / 32 * 32);
  }
  if (block.x > 1024) block.x = 1024;

  // Determine IndexMode for scale, shift, and gate
  IndexMode scale_mode = get_index_mode(scale, M, N);
  IndexMode shift_mode = get_index_mode(shift, M, N);
  IndexMode gate_mode_idx = get_gate_index_mode(gate_opt, M, N);

  // Compute F and L for 4D indexing
  int F = 1, L = (int)M;
  if (scale_mode == IndexMode::FullBF1N || shift_mode == IndexMode::FullBF1N || gate_mode_idx == IndexMode::FullBF1N) {
    // Find a 4D tensor to get B and F
    const ffi::TensorView* t4d = nullptr;
    if (scale.ndim() == 4)
      t4d = &scale;
    else if (shift.ndim() == 4)
      t4d = &shift;
    else if (gate_opt.has_value() && gate_opt.value().ndim() == 4)
      t4d = &gate_opt.value();
    if (t4d) {
      const int64_t B = t4d->size(0);
      F = (int)t4d->size(1);
      L = (int)(M / (B * F));
    }
  }
  // For gate [B, 1, N] case (BroadcastB1N mode), L represents rows_per_b
  if (gate_mode_idx == IndexMode::BroadcastB1N && gate_opt.has_value()) {
    L = (int)(M / gate_opt.value().size(0));
  }

  // Dispatch with IndexMode for scale, shift, and gate
  auto dispatch = [&](auto scale_mode_tag, auto shift_mode_tag, auto gate_mode_tag) {
    auto dispatch_dtype = [&](auto dtype_tag) {
      auto dispatch_ipt = [&](auto ipt_tag) {
        auto dispatch_norm = [&](auto norm_tag) {
          using T = typename decltype(dtype_tag)::T;
          LaunchKernel(grid, block, x.device())(
              norm_fused_res_gate_scale_shift_kernel<
                  T,
                  decltype(ipt_tag)::value,
                  decltype(norm_tag)::value,
                  decltype(scale_mode_tag)::value,
                  decltype(shift_mode_tag)::value,
                  decltype(gate_mode_tag)::value>,
              (T*)y.data_ptr(),
              (T*)residual_out.data_ptr(),
              (const T*)x.data_ptr(),
              (const T*)residual.data_ptr(),
              (const T*)gamma_ptr,
              (const T*)beta_ptr,
              (const T*)scale.data_ptr(),
              (const T*)shift.data_ptr(),
              (const T*)gate_ptr,
              (int)M,
              (int)N,
              F,
              L,
              affine,
              eps);
        };

        if (norm_type == NormType::LayerNorm) {
          dispatch_norm(NormTag<NormType::LayerNorm>{});
        } else {
          dispatch_norm(NormTag<NormType::RMSNorm>{});
        }
      };

      if (N <= 4096) {
        dispatch_ipt(ItemPerThreadTag<1>{});
      } else {
        dispatch_ipt(ItemPerThreadTag<8>{});
      }
    };

    const auto& dtype = x.dtype();
    if (is_type<float>(dtype)) {
      dispatch_dtype(DTypeTag<float>{});
    } else if (is_type<half>(dtype)) {
      dispatch_dtype(DTypeTag<half>{});
    } else if (is_type<nv_bfloat16>(dtype)) {
      dispatch_dtype(DTypeTag<cutlass::bfloat16_t>{});
    } else {
      RuntimeCheck(false, "Unsupported dtype. Use float32, float16, or bfloat16.");
    }
  };

  // Dispatch based on IndexMode combinations
  auto dispatch_gate = [&](auto scale_mode_tag, auto shift_mode_tag) {
    switch (gate_mode_idx) {
      case IndexMode::Scalar:
        dispatch(scale_mode_tag, shift_mode_tag, IndexModeTag<IndexMode::Scalar>{});
        break;
      case IndexMode::Broadcast1N:
        dispatch(scale_mode_tag, shift_mode_tag, IndexModeTag<IndexMode::Broadcast1N>{});
        break;
      case IndexMode::FullMN:
        dispatch(scale_mode_tag, shift_mode_tag, IndexModeTag<IndexMode::FullMN>{});
        break;
      case IndexMode::BroadcastB1N:
        dispatch(scale_mode_tag, shift_mode_tag, IndexModeTag<IndexMode::BroadcastB1N>{});
        break;
      case IndexMode::FullBF1N:
        dispatch(scale_mode_tag, shift_mode_tag, IndexModeTag<IndexMode::FullBF1N>{});
        break;
    }
  };

  auto dispatch_shift = [&](auto scale_mode_tag) {
    switch (shift_mode) {
      case IndexMode::Scalar:
        dispatch_gate(scale_mode_tag, IndexModeTag<IndexMode::Scalar>{});
        break;
      case IndexMode::Broadcast1N:
        dispatch_gate(scale_mode_tag, IndexModeTag<IndexMode::Broadcast1N>{});
        break;
      case IndexMode::FullMN:
        dispatch_gate(scale_mode_tag, IndexModeTag<IndexMode::FullMN>{});
        break;
      case IndexMode::FullBF1N:
        dispatch_gate(scale_mode_tag, IndexModeTag<IndexMode::FullBF1N>{});
        break;
    }
  };

  switch (scale_mode) {
    case IndexMode::Scalar:
      dispatch_shift(IndexModeTag<IndexMode::Scalar>{});
      break;
    case IndexMode::Broadcast1N:
      dispatch_shift(IndexModeTag<IndexMode::Broadcast1N>{});
      break;
    case IndexMode::FullMN:
      dispatch_shift(IndexModeTag<IndexMode::FullMN>{});
      break;
    case IndexMode::FullBF1N:
      dispatch_shift(IndexModeTag<IndexMode::FullBF1N>{});
      break;
  }
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

  SymbolicSize N_ = {"N"};
  TensorMatcher({-1, N_})  // 2D tensor, must be contiguous
      .with_dtype<float, half, nv_bfloat16>()
      .with_strides({N_, 1})
      .with_device<kDLCUDA>()
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
        .with_device<kDLCUDA>()
        .verify(gamma);
    RuntimeCheck(x.dtype() == gamma.dtype(), "x, gamma must have same dtype");
    if (beta_opt.has_value()) {
      const auto& beta = beta_opt.value();
      TensorMatcher({N_})  // 1D tensor, must be contiguous
          .with_dtype<float, half, nv_bfloat16>()
          .with_device<kDLCUDA>()
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
}  // namespace
