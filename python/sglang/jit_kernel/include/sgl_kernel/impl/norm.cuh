#pragma once
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

}  // namespace host::norm

namespace device::norm {

namespace details {

/**
 * \brief Core RMSNorm implementation parameterized by a compile-time epilogue.
 *
 * Computes the rsqrt-normalized value for every packed lane, then hands each
 * `(lane_index, fp32x2_t normed)` pair to `epilogue` for type conversion,
 * scaling, quantization, and output storage. The epilogue is a stateless or
 * POD functor whose `operator()` is fully inlined by the compiler, so the
 * compiled kernel is identical to a hand-written specialization.
 *
 * \tparam Epilogue Functor with signature
 *         `SGL_DEVICE void operator()(uint32_t i, fp32x2_t normed) const`.
 */
template <int64_t kDim, bool kUseCTA, typename PackedFloat, std::size_t N, typename Epilogue>
SGL_DEVICE void apply_norm_impl(
    const AlignedVector<PackedFloat, N> input,
    const AlignedVector<PackedFloat, N> weight,
    const float eps,
    [[maybe_unused]] float* smem_buffer,
    [[maybe_unused]] uint32_t num_warps,
    Epilogue epilogue) {
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

#pragma unroll
  for (auto i = 0u; i < N; ++i) {
    const auto fp32_input = cast<fp32x2_t>(input[i]);
    const auto fp32_weight = cast<fp32x2_t>(weight[i]);
    const fp32x2_t normed{
        fp32_input.x * norm_factor * fp32_weight.x,
        fp32_input.y * norm_factor * fp32_weight.y,
    };
    epilogue(i, normed);
  }
}

// ---------------------------------------------------------------------------
// Stock epilogues.
//
// Each epilogue is a small POD struct holding the per-call state (output
// reference, scale, etc.). Because they are passed by value into a templated
// function, the compiler inlines `operator()` and folds member accesses, so
// there is no runtime overhead vs. a hand-written specialization.
// ---------------------------------------------------------------------------

/**
 * \brief Default epilogue: cast the normed pair back to `PackedFloat` and
 *        store into `out[i]`. Reproduces the original `apply_norm_impl`.
 */
template <typename PackedFloat, std::size_t N>
struct PassthroughEpilogue {
  AlignedVector<PackedFloat, N>& out;

  SGL_DEVICE void operator()(uint32_t i, fp32x2_t v) const {
    out[i] = cast<PackedFloat, fp32x2_t>(v);
  }
};

/**
 * \brief Per-tensor fp8 quant epilogue.
 *
 * Multiplies each normed value by a scalar `inv_scale`, clamps to the fp8
 * dynamic range, and writes two fp8 elements per lane into `out`.
 */
template <std::size_t N>
struct PerTensorQuantFp8Epilogue {
  AlignedVector<fp8_e4m3_t, 2 * N>& out;
  float inv_scale;

  SGL_DEVICE void operator()(uint32_t i, fp32x2_t v) const {
    const float a = v.x * inv_scale;
    const float b = v.y * inv_scale;
    out[2 * i + 0] = static_cast<fp8_e4m3_t>(math::max(-math::FP8_E4M3_MAX, math::min(a, math::FP8_E4M3_MAX)));
    out[2 * i + 1] = static_cast<fp8_e4m3_t>(math::max(-math::FP8_E4M3_MAX, math::min(b, math::FP8_E4M3_MAX)));
  }
};

/**
 * \brief Per-channel fp8 quant epilogue.
 *
 * Uses a per-lane fp32x2_t inv-scale vector (loaded once, e.g. via the same
 * `tile::Memory<...>::cta` pattern as `input` / `weight`), then quantizes.
 */
template <std::size_t N>
struct PerChannelQuantFp8Epilogue {
  AlignedVector<fp8_e4m3_t, 2 * N>& out;
  const AlignedVector<fp32x2_t, N>& inv_scales;

  SGL_DEVICE void operator()(uint32_t i, fp32x2_t v) const {
    const auto s = inv_scales[i];
    const float a = v.x * s.x;
    const float b = v.y * s.y;
    out[2 * i + 0] = static_cast<fp8_e4m3_t>(math::max(-math::FP8_E4M3_MAX, math::min(a, math::FP8_E4M3_MAX)));
    out[2 * i + 1] = static_cast<fp8_e4m3_t>(math::max(-math::FP8_E4M3_MAX, math::min(b, math::FP8_E4M3_MAX)));
  }
};

}  // namespace details

/**
 * \brief Apply norm (warp-level) with a user-provided epilogue.
 *
 * Lower-level entry point for kernels that need to fuse quantization,
 * scaling, or alternative output layouts with the normalization. The
 * `epilogue` is a POD functor; see the `details::*Epilogue` helpers.
 */
template <int64_t kDim, typename PackedFloat, std::size_t N, typename Epilogue>
SGL_DEVICE void apply_norm_warp_with_epilogue(
    const AlignedVector<PackedFloat, N>& input,
    const AlignedVector<PackedFloat, N>& weight,
    float eps,
    Epilogue epilogue) {
  static_assert(kDim <= 256, "Warp norm only supports dim <= 256");
  details::apply_norm_impl<kDim, false>(input, weight, eps, nullptr, 0, epilogue);
}

/**
 * \brief Apply norm (CTA-level) with a user-provided epilogue.
 *
 * Lower-level entry point for kernels that need to fuse quantization,
 * scaling, or alternative output layouts with the normalization. The
 * `epilogue` is a POD functor; see the `details::*Epilogue` helpers.
 */
template <int64_t kDim, typename PackedFloat, std::size_t N, typename Epilogue>
SGL_DEVICE void apply_norm_cta_with_epilogue(
    const AlignedVector<PackedFloat, N>& input,
    const AlignedVector<PackedFloat, N>& weight,
    float eps,
    float* smem,
    Epilogue epilogue,
    uint32_t num_warps = blockDim.x / kWarpThreads) {
  static_assert(kDim > 256, "CTA norm only supports dim > 256");
  details::apply_norm_impl<kDim, true>(input, weight, eps, smem, num_warps, epilogue);
}

/**
 * \brief Apply norm using warp-level implementation.
 * \tparam kDim Dimension size
 * \param input Input vector
 * \param weight Weight vector
 * \param eps Epsilon value for numerical stability
 * \return Normalized output vector
 */
template <int64_t kDim, typename PackedFloat, std::size_t N>
SGL_DEVICE AlignedVector<PackedFloat, N>
apply_norm_warp(const AlignedVector<PackedFloat, N>& input, const AlignedVector<PackedFloat, N>& weight, float eps) {
  static_assert(kDim <= 256, "Warp norm only supports dim <= 256");
  AlignedVector<PackedFloat, N> output;
  details::apply_norm_impl<kDim, false>(
      input, weight, eps, nullptr, 0, details::PassthroughEpilogue<PackedFloat, N>{output});
  return output;
}

/**
 * \brief Apply norm using CTA-level implementation.
 * \tparam kDim Dimension size
 * \param input Input vector
 * \param weight Weight vector
 * \param eps Epsilon value for numerical stability
 * \param smem Shared memory buffer
 * \param num_warps Number of warps in the CTA
 * \return Normalized output vector
 */
template <int64_t kDim, typename PackedFloat, std::size_t N>
SGL_DEVICE AlignedVector<PackedFloat, N> apply_norm_cta(
    const AlignedVector<PackedFloat, N>& input,
    const AlignedVector<PackedFloat, N>& weight,
    float eps,
    float* smem,
    uint32_t num_warps = blockDim.x / kWarpThreads) {
  static_assert(kDim > 256, "CTA norm only supports dim > 256");
  AlignedVector<PackedFloat, N> output;
  details::apply_norm_impl<kDim, true>(
      input, weight, eps, smem, num_warps, details::PassthroughEpilogue<PackedFloat, N>{output});
  return output;
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
