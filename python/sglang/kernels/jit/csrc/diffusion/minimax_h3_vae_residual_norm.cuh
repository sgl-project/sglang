// MiniMax-H3 VAE decoder row producers: fp32 residual update + RMSNorm /
// LayerNorm, with a dynamic per-token FP8 E4M3 epilogue for the GEMM inputs.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace minimax_h3_vae_residual_norm {

/// \brief Which epilogue `residual_norm_kernel` computes on each row.
enum class NormMode : int32_t {
  kRmsNormFp8 = 0,          ///< q, s = fp8(RMSNorm(x) * weight)
  kResidualRmsNormFp8 = 1,  ///< r = fma(projected, layer_scale, x); q, s = fp8(RMSNorm(r) * weight)
  kResidualLayerNorm = 2,   ///< out = LayerNorm(fma(projected, layer_scale, x)) * weight + bias
};

/// One CTA per row; the fp32 row stays in registers between the two passes.
constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
/// Elements per thread per step. Every stream but `x` / `projected` is fp32
/// (weights, residual, LayerNorm output), so 16-byte vectors on those fix the
/// chunk at four elements; a 16-bit `x` / `projected` then loads 8 bytes.
constexpr uint32_t kVecSize = 16 / sizeof(fp32_t);
constexpr int64_t kAlignment = 16;

/// \brief Row pointers for one launch; pointers a mode does not read are null.
template <typename T, typename U>
struct Params {
  const T* __restrict__ x;                ///< [rows, kWidth]
  const U* __restrict__ projected;        ///< [rows, kWidth]; residual modes only
  const float* __restrict__ layer_scale;  ///< [kWidth]; residual modes only
  const float* __restrict__ weight;       ///< [kWidth]
  const float* __restrict__ bias;         ///< [kWidth]; kResidualLayerNorm only
  float* __restrict__ residual_out;       ///< [rows, kWidth]; kResidualRmsNormFp8 only
  fp8_e4m3_t* __restrict__ out_q;         ///< [rows, kWidth]; FP8 modes only
  float* __restrict__ out_s;              ///< [rows]; FP8 modes only
  float* __restrict__ out;                ///< [rows, kWidth]; kResidualLayerNorm only
  float eps;
};

/**
 * \brief Residual update + RMSNorm / LayerNorm (+ dynamic per-token FP8), one row per CTA.
 *
 * Statistics accumulate in fp32 over the fp32 row; LayerNorm uses the two-pass
 * centered variance; both norms scale by `rsqrt(var + eps)`. The FP8 epilogue
 * is `per_token_quant_fp8`'s: scale = amax / FP8_E4M3_MAX, codes clamped to
 * +-FP8_E4M3_MAX, and an all-zero row writes scale 0 with zero codes.
 *
 * \tparam T      Element type of `x`: fp32_t | fp16_t
 * \tparam U      Element type of `projected`: fp32_t | fp16_t (unused for kRmsNormFp8)
 * \tparam kWidth Row width; a multiple of kBlockSize * kVecSize
 * \tparam kMode  Epilogue selection
 * \param params  Row pointers and eps
 */
template <typename T, typename U, uint32_t kWidth, NormMode kMode>
__global__ void __launch_bounds__(kBlockSize) residual_norm_kernel(const Params<T, U> __grid_constant__ params) {
  using namespace device;
  static_assert(kWidth % (kBlockSize * kVecSize) == 0, "row width must be a multiple of kBlockSize * kVecSize");
  constexpr uint32_t kSteps = kWidth / (kBlockSize * kVecSize);
  constexpr uint32_t kPerThread = kSteps * kVecSize;
  constexpr bool kResidual = kMode != NormMode::kRmsNormFp8;
  constexpr bool kLayerNorm = kMode == NormMode::kResidualLayerNorm;
  using XVec = AlignedVector<T, kVecSize>;
  using PVec = AlignedVector<U, kVecSize>;
  using FVec = AlignedVector<float, kVecSize>;
  using QVec = AlignedVector<fp8_e4m3_t, kVecSize>;

  // One buffer per reduction, so no reduction waits on the readers of another.
  __shared__ float smem_sum[kNumWarps];
  __shared__ float smem_sum_sq[kNumWarps];
  __shared__ float smem_amax[kNumWarps];

  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * kWidth;

  float values[kPerThread];
  float sum = 0.0f;
  float sum_sq = 0.0f;
#pragma unroll
  for (uint32_t step = 0; step < kSteps; ++step) {
    const uint32_t chunk = step * kBlockSize + threadIdx.x;
    XVec x_vec;
    x_vec.load(params.x + row_offset, chunk);
    [[maybe_unused]] PVec projected_vec;
    [[maybe_unused]] FVec layer_scale_vec;
    if constexpr (kResidual) {
      projected_vec.load(params.projected + row_offset, chunk);
      layer_scale_vec.load(params.layer_scale, chunk);
    }
    [[maybe_unused]] FVec residual_vec;
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      float value = cast<fp32_t>(x_vec[i]);
      if constexpr (kResidual) {
        value = fmaf(cast<fp32_t>(projected_vec[i]), layer_scale_vec[i], value);
      }
      values[step * kVecSize + i] = value;
      sum += value;
      sum_sq = fmaf(value, value, sum_sq);
      if constexpr (kMode == NormMode::kResidualRmsNormFp8) {
        residual_vec[i] = value;
      }
    }
    if constexpr (kMode == NormMode::kResidualRmsNormFp8) {
      residual_vec.store(params.residual_out + row_offset, chunk);
    }
  }

  float mean = 0.0f;
  if constexpr (kLayerNorm) {
    mean = cta::reduce_sum<static_cast<int>(kBlockSize)>(sum, smem_sum) / static_cast<float>(kWidth);
    sum_sq = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kPerThread; ++j) {
      const float centered = values[j] - mean;
      sum_sq = fmaf(centered, centered, sum_sq);
    }
  }
  const float variance =
      cta::reduce_sum<static_cast<int>(kBlockSize)>(sum_sq, smem_sum_sq) / static_cast<float>(kWidth);
  const float inv_std = math::rsqrt(variance + params.eps);

  float amax = 0.0f;
#pragma unroll
  for (uint32_t step = 0; step < kSteps; ++step) {
    const uint32_t chunk = step * kBlockSize + threadIdx.x;
    FVec weight_vec;
    weight_vec.load(params.weight, chunk);
    [[maybe_unused]] FVec bias_vec;
    [[maybe_unused]] FVec out_vec;
    if constexpr (kLayerNorm) {
      bias_vec.load(params.bias, chunk);
    }
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      const uint32_t j = step * kVecSize + i;
      if constexpr (kLayerNorm) {
        out_vec[i] = (values[j] - mean) * inv_std * weight_vec[i] + bias_vec[i];
      } else {
        values[j] = values[j] * inv_std * weight_vec[i];
        amax = math::max(amax, math::abs(values[j]));
      }
    }
    if constexpr (kLayerNorm) {
      out_vec.store(params.out + row_offset, chunk);
    }
  }

  if constexpr (!kLayerNorm) {
    cta::reduce_max(amax, smem_amax);
    __syncthreads();
    const float scale = smem_amax[0] / math::FP8_E4M3_MAX;
    if (threadIdx.x == 0) {
      params.out_s[blockIdx.x] = scale;
    }
    const float scale_inv = scale == 0.0f ? 0.0f : 1.0f / scale;
#pragma unroll
    for (uint32_t step = 0; step < kSteps; ++step) {
      QVec q_vec;
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) {
        const float value = values[step * kVecSize + i] * scale_inv;
        q_vec[i] = static_cast<fp8_e4m3_t>(math::max(math::min(value, math::FP8_E4M3_MAX), -math::FP8_E4M3_MAX));
      }
      q_vec.store(params.out_q + row_offset, step * kBlockSize + threadIdx.x);
    }
  }
}

namespace details {

template <typename E, uint32_t kWidth>
void verify_rows(tvm::ffi::TensorView view, host::SymbolicSize& rows, host::SymbolicDevice& device) {
  host::TensorMatcher({rows, kWidth}).with_dtype<E>().with_device(device).ensure_alignment(kAlignment).verify(view);
}

template <uint32_t kWidth>
void verify_vector(tvm::ffi::TensorView view, host::SymbolicDevice& device) {
  host::TensorMatcher({kWidth}).with_dtype<fp32_t>().with_device(device).ensure_alignment(kAlignment).verify(view);
}

inline void verify_scales(tvm::ffi::TensorView view, host::SymbolicSize& rows, host::SymbolicDevice& device) {
  host::TensorMatcher({rows, 1}).with_dtype<fp32_t>().with_device(device).verify(view);
}

template <typename T, typename U, uint32_t kWidth, NormMode kMode>
void launch(const Params<T, U>& params, const host::SymbolicSize& rows, const host::SymbolicDevice& device) {
  const int64_t num_rows = rows.unwrap();
  CHECK_HOST(num_rows > 0 && num_rows <= UINT32_MAX)
      << "minimax_h3_vae_residual_norm: rows must be in (0, UINT32_MAX], got " << num_rows;
  host::LaunchKernel(static_cast<uint32_t>(num_rows), kBlockSize, device.unwrap())(
      residual_norm_kernel<T, U, kWidth, kMode>, params);
}

}  // namespace details

/**
 * \brief `q, s = fp8(RMSNorm(x) * weight)` per row.
 *
 * \tparam T      Element type of `x`: fp32_t | fp16_t
 * \tparam kWidth Row width; a multiple of kBlockSize * kVecSize
 */
template <typename T, uint32_t kWidth>
struct RmsNormFp8Kernel {
  /**
   * \param x       [rows, kWidth] T, contiguous, 16-byte aligned
   * \param weight  [kWidth] fp32
   * \param out_q   [rows, kWidth] fp8 e4m3, written
   * \param out_s   [rows, 1] fp32 per-row scales, written
   * \param eps     RMSNorm epsilon
   */
  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView out_q,
      tvm::ffi::TensorView out_s,
      double eps) {
    using namespace host;
    auto rows = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    details::verify_rows<T, kWidth>(x, rows, device);
    details::verify_vector<kWidth>(weight, device);
    details::verify_rows<fp8_e4m3_t, kWidth>(out_q, rows, device);
    details::verify_scales(out_s, rows, device);

    const Params<T, T> params{
        .x = static_cast<const T*>(x.data_ptr()),
        .projected = nullptr,
        .layer_scale = nullptr,
        .weight = static_cast<const float*>(weight.data_ptr()),
        .bias = nullptr,
        .residual_out = nullptr,
        .out_q = static_cast<fp8_e4m3_t*>(out_q.data_ptr()),
        .out_s = static_cast<float*>(out_s.data_ptr()),
        .out = nullptr,
        .eps = static_cast<float>(eps),
    };
    details::launch<T, T, kWidth, NormMode::kRmsNormFp8>(params, rows, device);
  }
};

/**
 * \brief `residual = fma(projected, layer_scale, x)` in fp32, then `q, s = fp8(RMSNorm(residual) * weight)`.
 *
 * \tparam T      Element type of `x`: fp32_t | fp16_t
 * \tparam U      Element type of `projected`: fp32_t | fp16_t
 * \tparam kWidth Row width; a multiple of kBlockSize * kVecSize
 */
template <typename T, typename U, uint32_t kWidth>
struct ResidualRmsNormFp8Kernel {
  /**
   * \param x             [rows, kWidth] T, contiguous, 16-byte aligned
   * \param projected     [rows, kWidth] U, contiguous, 16-byte aligned
   * \param layer_scale   [kWidth] fp32
   * \param weight        [kWidth] fp32
   * \param residual_out  [rows, kWidth] fp32, written
   * \param out_q         [rows, kWidth] fp8 e4m3, written
   * \param out_s         [rows, 1] fp32 per-row scales, written
   * \param eps           RMSNorm epsilon
   */
  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView projected,
      tvm::ffi::TensorView layer_scale,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView residual_out,
      tvm::ffi::TensorView out_q,
      tvm::ffi::TensorView out_s,
      double eps) {
    using namespace host;
    auto rows = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    details::verify_rows<T, kWidth>(x, rows, device);
    details::verify_rows<U, kWidth>(projected, rows, device);
    details::verify_vector<kWidth>(layer_scale, device);
    details::verify_vector<kWidth>(weight, device);
    details::verify_rows<fp32_t, kWidth>(residual_out, rows, device);
    details::verify_rows<fp8_e4m3_t, kWidth>(out_q, rows, device);
    details::verify_scales(out_s, rows, device);

    const Params<T, U> params{
        .x = static_cast<const T*>(x.data_ptr()),
        .projected = static_cast<const U*>(projected.data_ptr()),
        .layer_scale = static_cast<const float*>(layer_scale.data_ptr()),
        .weight = static_cast<const float*>(weight.data_ptr()),
        .bias = nullptr,
        .residual_out = static_cast<float*>(residual_out.data_ptr()),
        .out_q = static_cast<fp8_e4m3_t*>(out_q.data_ptr()),
        .out_s = static_cast<float*>(out_s.data_ptr()),
        .out = nullptr,
        .eps = static_cast<float>(eps),
    };
    details::launch<T, U, kWidth, NormMode::kResidualRmsNormFp8>(params, rows, device);
  }
};

/**
 * \brief `out = LayerNorm(fma(projected, layer_scale, x)) * weight + bias` in fp32.
 *
 * \tparam T      Element type of `x`: fp32_t | fp16_t
 * \tparam U      Element type of `projected`: fp32_t | fp16_t
 * \tparam kWidth Row width; a multiple of kBlockSize * kVecSize
 */
template <typename T, typename U, uint32_t kWidth>
struct ResidualLayerNormKernel {
  /**
   * \param x            [rows, kWidth] T, contiguous, 16-byte aligned
   * \param projected    [rows, kWidth] U, contiguous, 16-byte aligned
   * \param layer_scale  [kWidth] fp32
   * \param weight       [kWidth] fp32 LayerNorm gamma
   * \param bias         [kWidth] fp32 LayerNorm beta
   * \param out          [rows, kWidth] fp32, written
   * \param eps          LayerNorm epsilon
   */
  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView projected,
      tvm::ffi::TensorView layer_scale,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView bias,
      tvm::ffi::TensorView out,
      double eps) {
    using namespace host;
    auto rows = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    details::verify_rows<T, kWidth>(x, rows, device);
    details::verify_rows<U, kWidth>(projected, rows, device);
    details::verify_vector<kWidth>(layer_scale, device);
    details::verify_vector<kWidth>(weight, device);
    details::verify_vector<kWidth>(bias, device);
    details::verify_rows<fp32_t, kWidth>(out, rows, device);

    const Params<T, U> params{
        .x = static_cast<const T*>(x.data_ptr()),
        .projected = static_cast<const U*>(projected.data_ptr()),
        .layer_scale = static_cast<const float*>(layer_scale.data_ptr()),
        .weight = static_cast<const float*>(weight.data_ptr()),
        .bias = static_cast<const float*>(bias.data_ptr()),
        .residual_out = nullptr,
        .out_q = nullptr,
        .out_s = nullptr,
        .out = static_cast<float*>(out.data_ptr()),
        .eps = static_cast<float>(eps),
    };
    details::launch<T, U, kWidth, NormMode::kResidualLayerNorm>(params, rows, device);
  }
};

}  // namespace minimax_h3_vae_residual_norm

}  // namespace sglang
