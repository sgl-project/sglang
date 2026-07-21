#pragma once
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cstddef>
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

}  // namespace host::norm

namespace device::norm {

namespace details {

template <typename PackedFloat, std::size_t N>
SGL_DEVICE float compute_sum_of_squares(const AlignedVector<PackedFloat, N>& input) {
  float sum_of_squares = 0.0f;

#pragma unroll
  for (auto i = 0u; i < N; ++i) {
    const auto fp32_input = cast<fp32x2_t>(input[i]);
    sum_of_squares += fp32_input.x * fp32_input.x;
    sum_of_squares += fp32_input.y * fp32_input.y;
  }
  return sum_of_squares;
}

template <typename PackedFloat, std::size_t N>
SGL_DEVICE AlignedVector<PackedFloat, N> apply_norm_factor(
    const AlignedVector<PackedFloat, N>& input, const AlignedVector<PackedFloat, N>& weight, const float norm_factor) {
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

template <int64_t kDim, uint32_t kNumWarps>
SGL_DEVICE float compute_norm_factor_cta_static(float sum_of_squares, const float eps, float* smem_buffer) {
  static_assert(kNumWarps >= 1 && kNumWarps <= kWarpThreads);

  sum_of_squares = warp::reduce_sum(sum_of_squares);
  if constexpr (kNumWarps == 1) {
    return math::rsqrt(sum_of_squares / kDim + eps);
  } else {
    const auto warp_id = threadIdx.x / kWarpThreads;
    const auto lane_id = threadIdx.x % kWarpThreads;
    if (lane_id == 0) smem_buffer[warp_id] = sum_of_squares;
    __syncthreads();

    if constexpr (kNumWarps == 4) {
      // Four warps can combine their partials directly. This avoids a second
      // CTA barrier on the common 2048-wide, 32-byte-per-thread path.
      const auto total_sum = ((smem_buffer[0] + smem_buffer[1]) + smem_buffer[2]) + smem_buffer[3];
      return math::rsqrt(total_sum / kDim + eps);
    } else {
      if (warp_id == 0) {
        const auto local_sum = lane_id < kNumWarps ? smem_buffer[lane_id] : 0.0f;
        const auto total_sum = warp::reduce_sum(local_sum);
        if (lane_id == 0) smem_buffer[0] = total_sum;
      }
      __syncthreads();
      return math::rsqrt(smem_buffer[0] / kDim + eps);
    }
  }
}

template <int64_t kDim, bool kUseCTA, typename PackedFloat, std::size_t N>
SGL_DEVICE AlignedVector<PackedFloat, N> apply_norm_impl(
    const AlignedVector<PackedFloat, N> input,
    const AlignedVector<PackedFloat, N> weight,
    const float eps,
    [[maybe_unused]] float* smem_buffer,
    [[maybe_unused]] uint32_t num_warps) {
  // Keep the legacy dynamic-topology implementation textually stable. Some
  // callers compare BF16 results at a tight boundary; the static fused path
  // below owns the factored implementation.
  float sum_of_squares = 0.0f;

#pragma unroll
  for (auto i = 0u; i < N; ++i) {
    const auto fp32_input = cast<fp32x2_t>(input[i]);
    sum_of_squares += fp32_input.x * fp32_input.x;
    sum_of_squares += fp32_input.y * fp32_input.y;
  }

  sum_of_squares = warp::reduce_sum(sum_of_squares);
  float norm_factor;
  if constexpr (kUseCTA) {
    // need to synchronize across the cta
    const auto warp_id = threadIdx.x / kWarpThreads;
    smem_buffer[warp_id] = sum_of_squares;
    __syncthreads();
    // use the first warp to reduce
    if (warp_id == 0) {
      const auto tx = threadIdx.x;
      const auto local_sum = tx < num_warps ? smem_buffer[tx] : 0.0f;
      sum_of_squares = warp::reduce_sum(local_sum);
      smem_buffer[32] = math::rsqrt(sum_of_squares / kDim + eps);
    }
    __syncthreads();
    norm_factor = smem_buffer[32];
  } else {
    norm_factor = math::rsqrt(sum_of_squares / kDim + eps);
  }

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

}  // namespace details

/**
 * \brief Apply norm using warp-level implementation.
 * \tparam kDim Dimension size
 * \tparam T Element type (fp16_t or bf16_t)
 * \param input Input vector
 * \param weight Weight vector
 * \param eps Epsilon value for numerical stability
 * \return Normalized output vector
 */
template <int64_t kDim, typename T>
SGL_DEVICE T apply_norm_warp(const T& input, const T& weight, float eps) {
  static_assert(kDim <= 256, "Warp norm only supports dim <= 256");
  return details::apply_norm_impl<kDim, false>(input, weight, eps, nullptr, 0);
}

/**
 * \brief Apply norm using CTA-level implementation.
 * \tparam kDim Dimension size
 * \tparam T Element type (fp16_t or bf16_t)
 * \param input Input vector
 * \param weight Weight vector
 * \param eps Epsilon value for numerical stability
 * \param smem Shared memory buffer
 * \param num_warps Number of warps in the CTA
 * \return Normalized output vector
 */
template <int64_t kDim, typename T>
SGL_DEVICE T apply_norm_cta(
    const T& input, const T& weight, float eps, float* smem, uint32_t num_warps = blockDim.x / kWarpThreads) {
  static_assert(kDim > 256, "CTA norm only supports dim > 256");
  return details::apply_norm_impl<kDim, true>(input, weight, eps, smem, num_warps);
}

/**
 * \brief Apply norm with a compile-time CTA reduction topology.
 *
 * This variant is intended for static kernels that retain one input vector in
 * registers across the reduction. A one-warp launch needs no CTA barrier, and
 * a four-warp launch uses a single-barrier direct reduction.
 *
 * \tparam kDim Dimension size
 * \tparam kNumWarps Number of warps in the CTA
 * \tparam T Packed aligned-vector type
 * \param input Input vector retained by this thread
 * \param weight Weight vector retained by this thread
 * \param eps Epsilon value for numerical stability
 * \param smem Shared buffer with at least kNumWarps floats
 * \return Normalized output vector
 */
template <int64_t kDim, uint32_t kNumWarps, typename T>
SGL_DEVICE T apply_norm_cta_static(const T& input, const T& weight, float eps, float* smem) {
  const auto sum_of_squares = details::compute_sum_of_squares(input);
  const auto norm_factor = details::compute_norm_factor_cta_static<kDim, kNumWarps>(sum_of_squares, eps, smem);
  return details::apply_norm_factor(input, weight, norm_factor);
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
