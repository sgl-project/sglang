/// \file utils.cuh
/// \brief bf16 x bf16 -> fp32 dot product of two packed vectors, shared by the
/// small-GEMM kernels (`gemm/tiny_gemm.cuh`, `gemm/small/n128k512.cuh`) so that
/// they accumulate in the same order and agree bitwise where both apply.
#pragma once

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cstddef>
#include <cstdint>

namespace sglang {

namespace device {

/// Accumulate `sum_i a[i] * b[i]` into `acc`, in element order, one fp32 fma per
/// product. On Blackwell the mixed-precision fma consumes bf16 directly; the
/// fallback converts first, which is exact, so both round once per product.
template <std::size_t N>
SGL_DEVICE void dot_product_vec(AlignedVector<bf16x2_t, N> a, AlignedVector<bf16x2_t, N> b, float& acc) {
#pragma unroll
  for (uint32_t i = 0; i < N; ++i) {
#if SGL_ARCH_BLACKWELL_OR_GREATER
    acc = math::fma_f32_bf16(a[i].x, b[i].x, acc);
    acc = math::fma_f32_bf16(a[i].y, b[i].y, acc);
#else
    const auto [a0, a1] = cast<fp32x2_t>(a[i]);
    const auto [b0, b1] = cast<fp32x2_t>(b[i]);
    acc += a0 * b0;
    acc += a1 * b1;
#endif
  }
}

}  // namespace device

}  // namespace sglang
