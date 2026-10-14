// CUDA fast path for diffusion adaLN modulation.
//
// Reproduces the eager storage-dtype rounding boundaries:
//   out = round(round(x * round(1 + scale)) + shift)

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace sglang {

namespace modulate_scale_shift {

namespace {

constexpr uint32_t kRowsPerBlock = 4;
constexpr uint32_t kColsVecPerBlock = 256;
constexpr uint32_t kMaxGridY = 65535;
constexpr uintptr_t kAlignment = 16;

template <typename T>
SGL_DEVICE T modulate_value(T x, T scale, T shift) {
  const T one_plus_scale = device::cast<T>(1.0f + device::cast<fp32_t>(scale));
  const T product = device::cast<T>(device::cast<fp32_t>(x) * device::cast<fp32_t>(one_plus_scale));
  return device::cast<T>(device::cast<fp32_t>(product) + device::cast<fp32_t>(shift));
}

template <typename T, int kVec>
__global__ void modulate_scale_shift_kernel(
    T* __restrict__ out,
    const T* __restrict__ x,
    const T* __restrict__ scale,
    const T* __restrict__ shift,
    int64_t rows,
    int64_t rows_per_batch,
    int64_t row_vec) {
  using Vec = device::AlignedVector<T, kVec>;
  const int64_t col_vec = static_cast<int64_t>(blockIdx.x) * kColsVecPerBlock + threadIdx.x;
  if (col_vec >= row_vec) {
    return;
  }

  const int64_t row_stride = static_cast<int64_t>(gridDim.y) * kRowsPerBlock;
  for (int64_t row_base = static_cast<int64_t>(blockIdx.y) * kRowsPerBlock; row_base < rows; row_base += row_stride) {
#pragma unroll
    for (uint32_t row_offset = 0; row_offset < kRowsPerBlock; ++row_offset) {
      const int64_t row = row_base + row_offset;
      if (row >= rows) {
        continue;
      }

      const int64_t batch = row / rows_per_batch;
      const int64_t modulation_offset = batch * row_vec + col_vec;
      const int64_t activation_offset = row * row_vec + col_vec;
      Vec x_vec, scale_vec, shift_vec, out_vec;
      x_vec.load(x, activation_offset);
      scale_vec.load(scale, modulation_offset);
      shift_vec.load(shift, modulation_offset);
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        out_vec[i] = modulate_value(x_vec[i], scale_vec[i], shift_vec[i]);
      }
      out_vec.store(out, activation_offset);
    }
  }
}

}  // namespace

/**
 * \brief Validate and launch bit-exact diffusion adaLN modulation.
 *
 * \tparam T Activation type: fp16_t or bf16_t.
 */
template <typename T>
struct ModulateScaleShiftKernel {
  static_assert(std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>);

  /**
   * \param out Output tensor with shape [B, L, D].
   * \param x Input tensor with shape [B, L, D].
   * \param scale Scale tensor with shape [B, D].
   * \param shift Shift tensor with shape [B, D].
   */
  static void
  run(tvm::ffi::TensorView out, tvm::ffi::TensorView x, tvm::ffi::TensorView scale, tvm::ffi::TensorView shift) {
    using namespace host;

    auto B = SymbolicSize{"batch"};
    auto L = SymbolicSize{"sequence_length"};
    auto D = SymbolicSize{"hidden_size"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, L, D}).with_dtype<T>().with_device(device).verify(out).verify(x);
    TensorMatcher({B, D}).with_dtype<T>().with_device(device).verify(scale).verify(shift);

    const int64_t batch = B.unwrap();
    const int64_t sequence_length = L.unwrap();
    const int64_t hidden_size = D.unwrap();
    const int64_t rows = batch * sequence_length;
    if (rows == 0 || hidden_size == 0) {
      return;
    }

    constexpr int kVec = kAlignment / sizeof(T);
    CHECK_HOST(hidden_size % kVec == 0) << "hidden size must be a multiple of " << kVec;

    auto* out_ptr = static_cast<T*>(out.data_ptr());
    const auto* x_ptr = static_cast<const T*>(x.data_ptr());
    const auto* scale_ptr = static_cast<const T*>(scale.data_ptr());
    const auto* shift_ptr = static_cast<const T*>(shift.data_ptr());
    CHECK_HOST(
        reinterpret_cast<uintptr_t>(out_ptr) % kAlignment == 0 &&
        reinterpret_cast<uintptr_t>(x_ptr) % kAlignment == 0 &&
        reinterpret_cast<uintptr_t>(scale_ptr) % kAlignment == 0 &&
        reinterpret_cast<uintptr_t>(shift_ptr) % kAlignment == 0)
        << "modulate_scale_shift requires 16-byte aligned tensors";
    CHECK_HOST(out_ptr != x_ptr && out_ptr != scale_ptr && out_ptr != shift_ptr) << "output must not alias an input";

    const int64_t row_vec = hidden_size / kVec;
    const auto col_blocks = static_cast<uint32_t>(div_ceil(row_vec, static_cast<int64_t>(kColsVecPerBlock)));
    const int64_t row_tiles = div_ceil(rows, static_cast<int64_t>(kRowsPerBlock));
    const auto row_blocks = static_cast<uint32_t>(std::min<int64_t>(row_tiles, kMaxGridY));
    LaunchKernel(dim3(col_blocks, row_blocks), kColsVecPerBlock, device.unwrap())(
        modulate_scale_shift_kernel<T, kVec>, out_ptr, x_ptr, scale_ptr, shift_ptr, rows, sequence_length, row_vec);
  }
};

}  // namespace modulate_scale_shift

}  // namespace sglang
