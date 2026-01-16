#pragma once
#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cstdint>
#include <type_traits>

namespace host::norm {

/**
 * \brief Check if the given configuration is supported.
 * \tparam T Element type (only fp16_t/bf16_t is supported)
 * \tparam kDim Dimension size (usually hidden size)
 */
template <typename T, int64_t kDim>
inline constexpr bool is_config_supported() {
  if (!std::is_same_v<T, fp16_t> && !std::is_same_v<T, bf16_t>) return false;
  if (kDim <= 256) {
    return (kDim == 64 || kDim == 128 || kDim == 256);
  } else {
    return (kDim % 256 == 0 && kDim <= 8192);
  }
}

/**
 * \brief Determine whether to use cta norm based on dimension size.
 * TL;DR: use warp norm for dim <= 256, cta norm otherwise.
 * \tparam T Element type (fp16_t or bf16_t)
 * \tparam kDim Dimension size (usually hidden size)
 * \note This function assumes that the configuration is supported.
 * \see `is_config_supported`
 */
template <typename T, int64_t kDim>
inline constexpr bool should_use_cta() {
  static_assert(is_config_supported<T, kDim>(), "Unsupported norm configuration");
  return kDim > 256;
}

/**
 * \brief Get the number of threads per CTA for cta norm.
 * \tparam T Element type (fp16_t or bf16_t)
 * \tparam kDim Dimension size (usually hidden size)
 * \return Number of threads per CTA
 */
template <typename T, int64_t kDim>
inline constexpr uint32_t get_cta_threads() {
  static_assert(should_use_cta<T, kDim>());
  return (kDim / 256) * device::kWarpThreads;
}

/**
 * \brief Norm type selector for fused norm kernels.
 */
enum NormEnum : int {
  LayerNorm = 0,
  RMSNorm = 1,
};

}  // namespace host::norm

namespace device::norm {

using host::norm::NormEnum;

namespace details {

template <int64_t kDim, bool kUseCTA, typename PackedFloat, std::size_t N>
SGL_DEVICE AlignedVector<PackedFloat, N> apply_rmsnorm_impl(
    const AlignedVector<PackedFloat, N> input,
    const AlignedVector<PackedFloat, N> weight,
    const float eps,
    [[maybe_unused]] float* smem_buffer) {
  float sum_of_squares = 0.0f;

#pragma unroll
  for (auto i = 0u; i < N; ++i) {
    const auto fp32_input = cast<fp32x2_t>(input[i]);
    sum_of_squares += fp32_input.x * fp32_input.x;
    sum_of_squares += fp32_input.y * fp32_input.y;
  }

  if constexpr (kUseCTA) {
    cta::reduce_sum(sum_of_squares, smem_buffer);
    __syncthreads();
    sum_of_squares = smem_buffer[0];
  } else {
    sum_of_squares = warp::reduce_sum(sum_of_squares);
  }
  float norm_factor = math::rsqrt(sum_of_squares / kDim + eps);

  AlignedVector<PackedFloat, N> output;

#pragma unroll
  for (auto i = 0u; i < N; ++i) {
    const auto fp32_input = cast<fp32x2_t>(input[i]);
    const auto fp32_weight = cast<fp32x2_t>(weight[i]);
    output[i] = cast<PackedFloat, fp32x2_t>({
        fp32_input.x * norm_factor * fp32_weight.x,
        fp32_input.y * norm_factor * fp32_weight.y,
    });
  }

  return output;
}

template <int64_t kDim, bool kUseCTA, typename PackedFloat, std::size_t N>
SGL_DEVICE AlignedVector<PackedFloat, N> apply_layernorm_impl(
    const AlignedVector<PackedFloat, N> input,
    const AlignedVector<PackedFloat, N> weight,
    const AlignedVector<PackedFloat, N> bias,
    const float eps,
    [[maybe_unused]] float* smem_buffer) {
  float local_sum = 0.0f;

  // ========== Pass 1: Compute mean ==========
#pragma unroll
  for (auto i = 0u; i < N; ++i) {
    const auto fp32_input = cast<fp32x2_t>(input[i]);
    local_sum += fp32_input.x + fp32_input.y;
  }

  // Reduce
  if constexpr (kUseCTA) {
    cta::reduce_sum(local_sum, smem_buffer);
    __syncthreads();
    local_sum = smem_buffer[0];
  } else {
    local_sum = warp::reduce_sum(local_sum);
  }
  float mean = local_sum / kDim;

  // ========== Pass 2: Compute variance ==========
  float variance_sum = 0.0f;
#pragma unroll
  for (auto i = 0u; i < N; ++i) {
    const auto fp32_input = cast<fp32x2_t>(input[i]);
    float diff_x = fp32_input.x - mean;
    float diff_y = fp32_input.y - mean;
    variance_sum += diff_x * diff_x + diff_y * diff_y;
  }

  // Reduce variance
  if constexpr (kUseCTA) {
    cta::reduce_sum(variance_sum, smem_buffer);
    __syncthreads();
    variance_sum = smem_buffer[0];
  } else {
    variance_sum = warp::reduce_sum(variance_sum);
  }
  float norm_factor = math::rsqrt(variance_sum / kDim + eps);

  // ========== Pass 3: Apply normalization ==========
  AlignedVector<PackedFloat, N> output;

#pragma unroll
  for (auto i = 0u; i < N; ++i) {
    const auto fp32_input = cast<fp32x2_t>(input[i]);
    const auto fp32_weight = cast<fp32x2_t>(weight[i]);
    const auto fp32_bias = cast<fp32x2_t>(bias[i]);

    float norm_x = (fp32_input.x - mean) * norm_factor;
    float norm_y = (fp32_input.y - mean) * norm_factor;

    float out_x = norm_x * fp32_weight.x + fp32_bias.x;
    float out_y = norm_y * fp32_weight.y + fp32_bias.y;

    output[i] = cast<PackedFloat, fp32x2_t>({out_x, out_y});
  }

  return output;
}

}  // namespace details

/**
 * \brief Apply RMSNorm using warp-level implementation.
 * \tparam kDim Dimension size
 * \tparam T Element type (fp16_t or bf16_t)
 * \param input Input vector
 * \param weight Weight vector
 * \param eps Epsilon value for numerical stability
 * \return Normalized output vector
 */
template <int64_t kDim, typename T>
SGL_DEVICE T apply_rmsnorm_warp(const T& input, const T& weight, float eps) {
  static_assert(kDim <= 256, "Warp norm only supports dim <= 256");
  return details::apply_rmsnorm_impl<kDim, false>(input, weight, eps, nullptr);
}

/**
 * \brief Apply RMSNorm using CTA-level implementation.
 * \tparam kDim Dimension size
 * \tparam T Element type (fp16_t or bf16_t)
 * \param input Input vector
 * \param weight Weight vector
 * \param eps Epsilon value for numerical stability
 * \param smem Shared memory buffer
 * \return Normalized output vector
 */
template <int64_t kDim, typename T>
SGL_DEVICE T apply_rmsnorm_cta(const T& input, const T& weight, float eps, float* smem) {
  static_assert(kDim > 256, "CTA norm only supports dim > 256");
  return details::apply_rmsnorm_impl<kDim, true>(input, weight, eps, smem);
}

/**
 * \brief Apply LayerNorm using warp-level implementation.
 * \tparam kDim Dimension size
 * \tparam T Element type (fp16_t or bf16_t)
 * \param input Input vector
 * \param weight Weight vector (gamma)
 * \param bias Bias vector (beta)
 * \param eps Epsilon value for numerical stability
 * \return Normalized output vector
 */
template <int64_t kDim, typename T>
SGL_DEVICE T apply_layernorm_warp(const T& input, const T& weight, const T& bias, float eps) {
  static_assert(kDim <= 256, "Warp norm only supports dim <= 256");
  return details::apply_layernorm_impl<kDim, false>(input, weight, bias, eps, nullptr);
}

/**
 * \brief Apply LayerNorm using CTA-level implementation.
 * \tparam kDim Dimension size
 * \tparam T Element type (fp16_t or bf16_t)
 * \param input Input vector
 * \param weight Weight vector (gamma)
 * \param bias Bias vector (beta)
 * \param eps Epsilon value for numerical stability
 * \param smem Shared memory buffer
 * \return Normalized output vector
 */
template <int64_t kDim, typename T>
SGL_DEVICE T apply_layernorm_cta(const T& input, const T& weight, const T& bias, float eps, float* smem) {
  static_assert(kDim > 256, "CTA norm only supports dim > 256");
  return details::apply_layernorm_impl<kDim, true>(input, weight, bias, eps, smem);
}

/**
 * \brief Apply norm (RMSNorm or LayerNorm) using warp-level implementation.
 * \tparam norm_enum Norm type (RMSNorm or LayerNorm)
 * \tparam kDim Dimension size
 * \tparam T Element type (fp16_t or bf16_t)
 * \param input Input vector
 * \param weight Weight vector (gamma)
 * \param bias Bias vector (beta), only used for LayerNorm (ignored for RMSNorm)
 * \param eps Epsilon value for numerical stability
 * \return Normalized output vector
 */
template <NormEnum norm_enum, int64_t kDim, typename T>
SGL_DEVICE T apply_norm_warp(const T& input, const T& weight, [[maybe_unused]] const T& bias, float eps) {
  static_assert(kDim <= 256, "Warp norm only supports dim <= 256");
  if constexpr (norm_enum == NormEnum::LayerNorm) {
    return apply_layernorm_warp<kDim>(input, weight, bias, eps);
  } else {
    return apply_rmsnorm_warp<kDim>(input, weight, eps);
  }
}

/**
 * \brief Apply norm (RMSNorm or LayerNorm) using CTA-level implementation.
 * \tparam norm_enum Norm type (RMSNorm or LayerNorm)
 * \tparam kDim Dimension size
 * \tparam T Element type (fp16_t or bf16_t)
 * \param input Input vector
 * \param weight Weight vector (gamma)
 * \param bias Bias vector (beta), only used for LayerNorm (ignored for RMSNorm)
 * \param eps Epsilon value for numerical stability
 * \param smem Shared memory buffer
 * \return Normalized output vector
 */
template <NormEnum norm_enum, int64_t kDim, typename T>
SGL_DEVICE T apply_norm_cta(const T& input, const T& weight, [[maybe_unused]] const T& bias, float eps, float* smem) {
  static_assert(kDim > 256, "CTA norm only supports dim > 256");
  if constexpr (norm_enum == NormEnum::LayerNorm) {
    return apply_layernorm_cta<kDim>(input, weight, bias, eps, smem);
  } else {
    return apply_rmsnorm_cta<kDim>(input, weight, eps, smem);
  }
}

/**
 * \brief Storage type for norm operation.
 * For warp norm, the storage size depends on kDim.
 * For cta norm, the storage size is fixed to 16B.
 * We will also pack the input 16-bit floats into 32-bit types
 * for faster CUDA core operations.
 *
 * \tparam T Element type (fp16_t or bf16_t)
 * \tparam kDim Dimension size
 */
template <typename T, int64_t kDim>
using StorageType = std::conditional_t<                    // storage type
    (kDim > 256),                                          // whether to use cta norm
    AlignedVector<packed_t<T>, 4>,                         // cta norm storage, fixed to 16B
    AlignedVector<packed_t<T>, kDim / (2 * kWarpThreads)>  // warp norm storage
    >;

/**
 * \brief Minimum shared memory size (in bytes) required for cta norm.
 */
inline constexpr uint32_t kSmemBufferSize = 33;

}  // namespace device::norm
