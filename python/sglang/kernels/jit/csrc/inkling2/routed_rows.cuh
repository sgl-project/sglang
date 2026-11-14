// SPDX-License-Identifier: Apache-2.0
// This rank's routed expert rows, equal to the Triton _routed_rows in moe.py:
// rows in [offsets[0], offsets[E]) of x, scaled like the NVFP4 GEMM output when
// scales are given, rounded to bfloat16; written into a routed exchange slot,
// or into a dense output whose other rows are zero.

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>

namespace sglang::inkling2_routed_rows {

constexpr int kMaxExperts = 256;

template <typename T>
struct In;
template <>
struct In<float> {
  static constexpr int kVec = 8;
  struct Raw {
    float4 a, b;
  };
  __device__ __forceinline__ static float get(Raw const& r, int i) {
    return i < 4 ? (&r.a.x)[i] : (&r.b.x)[i - 4];
  }
};
template <>
struct In<__nv_bfloat16> {
  static constexpr int kVec = 8;
  using Raw = uint4;
  __device__ __forceinline__ static float get(Raw const& r, int i) {
    return __bfloat162float(reinterpret_cast<__nv_bfloat16 const*>(&r)[i]);
  }
};

// OWNED: write only rows in [offsets[0], offsets[E]) into out + slot (the
// exchange's current slot); otherwise write every row, zero outside. Each row
// is split over kChunks CTAs.
constexpr int kChunks = 4;

template <int H, typename T, bool SCALED, bool OWNED>
__global__ void __launch_bounds__(H / In<T>::kVec / kChunks) routed_rows_kernel(
    T const* __restrict__ x,
    float const* __restrict__ x_amax,
    float const* __restrict__ w_amax,
    float const* __restrict__ w_denominator,
    int const* __restrict__ offsets,
    int experts,
    __nv_bfloat16* out,
    int64_t const* calls,
    int64_t slot_size,
    int rows) {
  constexpr int kVec = In<T>::kVec;
  using Raw = typename In<T>::Raw;
  __shared__ int bounds[kMaxExperts + 1];
  // Offsets, amax, and the weight scales come from kernels before the
  // preceding GEMM: read them while it finishes.
  for (int e = threadIdx.x; e <= experts; e += blockDim.x)
    bounds[e] = offsets[e];
  float rcp = 0.0f;
  if constexpr (SCALED) rcp = __frcp_rn(__fmul_rn(1536.0f, *w_denominator));
  __syncthreads();
  int const lo = bounds[0], hi = bounds[experts];
  int const col = ((blockIdx.x % kChunks) * blockDim.x + threadIdx.x) * kVec;
  int const first = (OWNED ? lo : 0) + blockIdx.x / kChunks;
  int const last = OWNED ? hi : rows;
  int const stride = gridDim.x / kChunks;
  auto scale_of = [&](int row) {
    int e = 0;
    for (int step = kMaxExperts; step > 0; step /= 2)
      if (e + step < experts && bounds[e + step] <= row) e += step;
    return __fmul_rn(__fmul_rn(x_amax[row], w_amax[e]), rcp);
  };
  float scale = 0.0f;
  if (SCALED && first < last && first >= lo && first < hi) scale = scale_of(first);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (OWNED && calls != nullptr) out += (calls[0] & 1) * slot_size;
  for (int row = first; row < last; row += stride) {
    alignas(16) __nv_bfloat16 values[kVec];
    if (row >= lo && row < hi) {
      Raw raw = *reinterpret_cast<Raw const*>(x + int64_t(row) * H + col);
      if constexpr (SCALED) {
        if (row != first) scale = scale_of(row);
#pragma unroll
        for (int i = 0; i < kVec; ++i)
          values[i] = __float2bfloat16_rn(__fmul_rn(In<T>::get(raw, i), scale));
      } else {
#pragma unroll
        for (int i = 0; i < kVec; ++i)
          values[i] = __float2bfloat16_rn(In<T>::get(raw, i));
      }
    } else {
#pragma unroll
      for (int i = 0; i < kVec; ++i)
        values[i] = __float2bfloat16_rn(0.0f);
    }
    *reinterpret_cast<uint4*>(out + int64_t(row) * H + col) = *reinterpret_cast<uint4*>(values);
  }
  asm volatile("griddepcontrol.launch_dependents;");
}

template <int H, typename T, bool SCALED, bool OWNED>
void run(
    tvm::ffi::TensorView x,
    int64_t x_amax,
    int64_t w_amax,
    int64_t w_denominator,
    tvm::ffi::TensorView offsets,
    int64_t out,
    int64_t calls,
    int64_t slot_size,
    int64_t blocks) {
  int const experts = offsets.size(0) - 1;
  host::RuntimeCheck(x.size(1) == H && experts <= kMaxExperts, "Unsupported routed rows shape");
  static_assert(H % (In<T>::kVec * kChunks) == 0);
  host::LaunchKernel(blocks * kChunks, H / In<T>::kVec / kChunks, x.device())
      .enable_pdl(true)(
          routed_rows_kernel<H, T, SCALED, OWNED>,
          static_cast<T const*>(x.data_ptr()),
          reinterpret_cast<float const*>(x_amax),
          reinterpret_cast<float const*>(w_amax),
          reinterpret_cast<float const*>(w_denominator),
          static_cast<int const*>(offsets.data_ptr()),
          experts,
          reinterpret_cast<__nv_bfloat16*>(out),
          reinterpret_cast<int64_t const*>(calls),
          slot_size,
          int(x.size(0)));
}

}  // namespace sglang::inkling2_routed_rows
