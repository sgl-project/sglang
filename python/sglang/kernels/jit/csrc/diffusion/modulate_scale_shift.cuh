// CUDA fast path for diffusion adaLN modulate chains.
//
// Implements, with each intermediate computed in fp32 and rounded to the
// storage dtype (the per-op kernel boundaries of the eager aten chain):
//
//   out = (x * (1 + scale)) + shift
//       = round(round(x * round(1 + scale)) + shift)
//
// so the fused kernel is bit-exact vs eager for fp16/bf16.  x is a
// contiguous [B, L, D] activation; scale/shift are contiguous [B, D]
// modulation rows.
//
// Intentionally narrow: 16-byte aligned tensors, D % kVec == 0 (the Python
// guard enforces this).

#pragma once

#include <sgl_kernel/tensor.h>  // For host dtype helpers and TensorView metadata
#include <sgl_kernel/utils.h>   // For RuntimeCheck and div_ceil

#include <sgl_kernel/type.cuh>   // For DTypeTrait conversions
#include <sgl_kernel/utils.cuh>  // For LaunchKernel and CUDA dtype aliases
#include <sgl_kernel/vec.cuh>    // For device::AlignedVector

#include <cstdint>

namespace sglang {

namespace sglang_modulate_scale_shift {

namespace {

constexpr int kRowsPerBlock = 4;
constexpr int kColsVecPerBlock = 256;
constexpr int64_t kMaxGrid = 65535;

inline const char* data_ptr(const tvm::ffi::TensorView& t) {
  return static_cast<const char*>(t.data_ptr()) + t.byte_offset();
}

inline char* mutable_data_ptr(const tvm::ffi::TensorView& t) {
  return static_cast<char*>(t.data_ptr()) + t.byte_offset();
}

inline bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

inline int64_t numel(const tvm::ffi::TensorView& t) {
  int64_t n = 1;
  for (int i = 0; i < t.ndim(); ++i) {
    n *= t.size(i);
  }
  return n;
}

inline bool is_dense_contiguous(const tvm::ffi::TensorView& t) {
  int64_t expected = 1;
  for (int i = t.ndim() - 1; i >= 0; --i) {
    if (t.size(i) == 1) {
      continue;
    }
    if (t.stride(i) != expected) {
      return false;
    }
    expected *= t.size(i);
  }
  return true;
}

template <typename T>
inline void check_dtype(const tvm::ffi::TensorView& t) {
  host::RuntimeCheck(host::is_type<T>(t.dtype()), "unexpected dtype for modulate_scale_shift");
}

template <typename T>
__device__ __forceinline__ float to_float(T v) {
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<fp16_t>(fp16_t v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float to_float<bf16_t>(bf16_t v) {
  return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T modulate_value(T x, T scale, T shift) {
  // Round each intermediate back to T (the eager chain's kernel boundaries;
  // also blocks fmul+fadd FMA contraction).
  const T one_plus_scale = DTypeTrait<T>::from(1.0f + to_float(scale));
  const T product = DTypeTrait<T>::from(to_float(x) * to_float(one_plus_scale));
  return DTypeTrait<T>::from(to_float(product) + to_float(shift));
}

template <typename T, int kVec>
__global__ void modulate_scale_shift_vec_kernel(
    const T* __restrict__ x,
    const T* __restrict__ scale,
    const T* __restrict__ shift,
    T* __restrict__ out,
    int64_t rows,
    int64_t rows_per_batch,
    int64_t row_vec) {
  using Vec = device::AlignedVector<T, kVec>;
  const int64_t col_vec = static_cast<int64_t>(blockIdx.x) * kColsVecPerBlock + threadIdx.x;
  if (col_vec >= row_vec) {
    return;
  }

  // Grid-stride: the row-tile count can exceed the gridDim.y hardware limit.
  const int64_t row_tile_stride = static_cast<int64_t>(gridDim.y) * kRowsPerBlock;
  for (int64_t row_base = static_cast<int64_t>(blockIdx.y) * kRowsPerBlock; row_base < rows;
       row_base += row_tile_stride) {
#pragma unroll
    for (int row_offset = 0; row_offset < kRowsPerBlock; ++row_offset) {
      const int64_t row = row_base + row_offset;
      if (row < rows) {
        const int64_t batch = row / rows_per_batch;
        const int64_t mod_v = batch * row_vec + col_vec;
        const int64_t v = row * row_vec + col_vec;
        Vec xv, s, b, o;
        s.load(scale, mod_v);
        b.load(shift, mod_v);
        xv.load(x, v);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          o[i] = modulate_value(xv[i], s[i], b[i]);
        }
        o.store(out, v);
      }
    }
  }
}

template <typename T>
inline void launch_modulate_scale_shift(
    const tvm::ffi::TensorView& out,
    const tvm::ffi::TensorView& x,
    const tvm::ffi::TensorView& scale,
    const tvm::ffi::TensorView& shift) {
  const int64_t total = numel(x);
  if (total == 0) {
    return;
  }

  const int64_t D = x.size(x.ndim() - 1);
  const int64_t rows = total / D;
  const int64_t batches = scale.size(0);
  const int64_t rows_per_batch = rows / batches;
  const T* x_ptr = reinterpret_cast<const T*>(data_ptr(x));
  const T* scale_ptr = reinterpret_cast<const T*>(data_ptr(scale));
  const T* shift_ptr = reinterpret_cast<const T*>(data_ptr(shift));
  T* out_ptr = reinterpret_cast<T*>(mutable_data_ptr(out));
  constexpr int kVec = 16 / sizeof(T);

  host::RuntimeCheck(
      aligned16(x_ptr) && aligned16(scale_ptr) && aligned16(shift_ptr) && aligned16(out_ptr),
      "modulate_scale_shift requires 16-byte aligned tensors");
  host::RuntimeCheck(D % kVec == 0, "modulate_scale_shift requires D to be a multiple of the vector width");

  const int64_t row_vec = D / kVec;
  const int64_t col_blocks = host::div_ceil(row_vec, static_cast<int64_t>(kColsVecPerBlock));
  const int64_t row_tiles = host::div_ceil(rows, static_cast<int64_t>(kRowsPerBlock));
  const int64_t row_blocks = row_tiles > kMaxGrid ? kMaxGrid : row_tiles;
  host::LaunchKernel(
      dim3(static_cast<uint32_t>(col_blocks), static_cast<uint32_t>(row_blocks)), dim3(kColsVecPerBlock), out.device())(
      modulate_scale_shift_vec_kernel<T, kVec>, x_ptr, scale_ptr, shift_ptr, out_ptr, rows, rows_per_batch, row_vec);
}

template <typename T>
inline void validate_modulate_scale_shift(
    const tvm::ffi::TensorView& out,
    const tvm::ffi::TensorView& x,
    const tvm::ffi::TensorView& scale,
    const tvm::ffi::TensorView& shift) {
  check_dtype<T>(out);
  check_dtype<T>(x);
  check_dtype<T>(scale);
  check_dtype<T>(shift);
  host::RuntimeCheck(x.device().device_type == kDLCUDA, "x must be CUDA");
  host::RuntimeCheck(scale.device().device_type == kDLCUDA, "scale must be CUDA");
  host::RuntimeCheck(shift.device().device_type == kDLCUDA, "shift must be CUDA");
  host::RuntimeCheck(out.device().device_type == kDLCUDA, "out must be CUDA");
  host::RuntimeCheck(
      x.device().device_id == scale.device().device_id && x.device().device_id == shift.device().device_id &&
          x.device().device_id == out.device().device_id,
      "x/scale/shift/out must be on the same CUDA device");
  host::RuntimeCheck(x.ndim() == 3, "x must be [B, L, D]");
  host::RuntimeCheck(scale.ndim() == 2, "scale must be [B, D]");
  host::RuntimeCheck(shift.ndim() == 2, "shift must be [B, D]");
  host::RuntimeCheck(out.ndim() == x.ndim(), "out rank must match x");
  for (int i = 0; i < x.ndim(); ++i) {
    host::RuntimeCheck(out.size(i) == x.size(i), "out shape must match x");
  }
  host::RuntimeCheck(scale.size(0) == x.size(0), "scale batch dim must match x");
  host::RuntimeCheck(scale.size(1) == x.size(2), "scale last dim must match x");
  host::RuntimeCheck(shift.size(0) == scale.size(0) && shift.size(1) == scale.size(1), "shift shape must match scale");
  host::RuntimeCheck(is_dense_contiguous(x), "x must be contiguous");
  host::RuntimeCheck(is_dense_contiguous(scale), "scale must be contiguous");
  host::RuntimeCheck(is_dense_contiguous(shift), "shift must be contiguous");
  host::RuntimeCheck(is_dense_contiguous(out), "out must be contiguous");
  host::RuntimeCheck(data_ptr(out) != data_ptr(x), "out must not alias x");
  host::RuntimeCheck(data_ptr(out) != data_ptr(scale), "out must not alias scale");
  host::RuntimeCheck(data_ptr(out) != data_ptr(shift), "out must not alias shift");
}

}  // namespace

template <typename T>
struct ModulateScaleShiftKernel {
  static void
  run(tvm::ffi::TensorView out, tvm::ffi::TensorView x, tvm::ffi::TensorView scale, tvm::ffi::TensorView shift) {
    validate_modulate_scale_shift<T>(out, x, scale, shift);
    launch_modulate_scale_shift<T>(out, x, scale, shift);
  }
};

}  // namespace sglang_modulate_scale_shift

}  // namespace sglang
