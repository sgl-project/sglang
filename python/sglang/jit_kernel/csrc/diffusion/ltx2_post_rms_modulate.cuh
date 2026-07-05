// CUDA fast path for LTX2 post-RMSNorm BF16 modulation.
//
// Implements the elementwise eager contract:
//   out = x * (1 + scale) + shift
//
// This intentionally leaves RMSNorm itself on the PyTorch path so the norm
// reduction order stays identical to main.  The kernel only fuses the BF16
// modulation chain after the normalized tensor has already been materialized.

#pragma once

#include <sgl_kernel/tensor.h>  // For TensorView metadata and dtype helpers
#include <sgl_kernel/utils.h>   // For RuntimeCheck and div_ceil

#include <sgl_kernel/utils.cuh>  // For LaunchKernel and CUDA dtype aliases
#include <sgl_kernel/vec.cuh>    // For device::AlignedVector

#include <cstdint>
#include <cuda_bf16.h>

namespace sglang_ltx2_post_rms_modulate {

namespace {

constexpr int kBlockSize = 256;
constexpr int kVec = 8;
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

inline int64_t grid_for(int64_t total) {
  int64_t grid = host::div_ceil(total, static_cast<int64_t>(kBlockSize));
  if (grid < 1) {
    grid = 1;
  }
  if (grid > kMaxGrid) {
    grid = kMaxGrid;
  }
  return grid;
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

inline void check_bf16_cuda(const tvm::ffi::TensorView& t, const char* name) {
  host::RuntimeCheck(host::is_type<bf16_t>(t.dtype()), name, " must be BF16");
  host::RuntimeCheck(t.device().device_type == kDLCUDA, name, " must be CUDA");
}

inline void check_same_device(
    const tvm::ffi::TensorView& a,
    const tvm::ffi::TensorView& b,
    const char* a_name,
    const char* b_name) {
  host::RuntimeCheck(
      a.device().device_id == b.device().device_id,
      a_name,
      " and ",
      b_name,
      " must be on the same CUDA device");
}

inline void check_param_shape(
    const tvm::ffi::TensorView& param,
    const tvm::ffi::TensorView& x,
    const char* name) {
  host::RuntimeCheck(param.ndim() == 3, name, " must be 3D");
  host::RuntimeCheck(param.size(0) == 1 || param.size(0) == x.size(0), name, " batch dim must broadcast to x");
  host::RuntimeCheck(param.size(1) == 1 || param.size(1) == x.size(1), name, " seq dim must broadcast to x");
  host::RuntimeCheck(param.size(2) == x.size(2), name, " hidden dim must match x");
  host::RuntimeCheck(param.stride(2) == 1, name, " last dim must be contiguous");
}

SGL_DEVICE bf16_t bf16_from_float(float v) {
  return __float2bfloat16_rn(v);
}

SGL_DEVICE float bf16_to_float(bf16_t v) {
  return __bfloat162float(v);
}

SGL_DEVICE bf16_t modulate_value(bf16_t x, bf16_t scale, bf16_t shift) {
  const bf16_t one_plus_scale = bf16_from_float(1.0f + bf16_to_float(scale));
  const bf16_t product = bf16_from_float(bf16_to_float(x) * bf16_to_float(one_plus_scale));
  return bf16_from_float(bf16_to_float(product) + bf16_to_float(shift));
}

struct ParamStrides {
  int64_t b;
  int64_t s;
  int64_t d;
  int64_t batch;
  int64_t seq;
};

SGL_DEVICE int64_t param_offset(const ParamStrides strides, int64_t batch, int64_t token, int64_t col) {
  const int64_t param_batch = strides.batch == 1 ? 0 : batch;
  const int64_t param_token = strides.seq == 1 ? 0 : token;
  return param_batch * strides.b + param_token * strides.s + col * strides.d;
}

__global__ void ltx2_post_rms_modulate_vec_kernel(
    const bf16_t* __restrict__ x,
    const bf16_t* __restrict__ scale,
    const bf16_t* __restrict__ shift,
    bf16_t* __restrict__ out,
    int64_t total_vec,
    int64_t seq,
    int64_t hidden,
    ParamStrides scale_strides,
    ParamStrides shift_strides) {
  using Vec = device::AlignedVector<bf16_t, kVec>;
  const int64_t row_vec = hidden / kVec;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t vi = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; vi < total_vec; vi += stride) {
    const int64_t row = vi / row_vec;
    const int64_t col_vec = vi - row * row_vec;
    const int64_t batch = row / seq;
    const int64_t token = row - batch * seq;
    const int64_t col = col_vec * kVec;

    Vec x_vec;
    Vec scale_vec;
    Vec shift_vec;
    Vec out_vec;
    x_vec.load(x, vi);
    scale_vec.load(scale + param_offset(scale_strides, batch, token, col), 0);
    shift_vec.load(shift + param_offset(shift_strides, batch, token, col), 0);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      out_vec[i] = modulate_value(x_vec[i], scale_vec[i], shift_vec[i]);
    }
    out_vec.store(out, vi);
  }
}

__global__ void ltx2_post_rms_dual_modulate_vec_kernel(
    const bf16_t* __restrict__ x,
    const bf16_t* __restrict__ scale0,
    const bf16_t* __restrict__ shift0,
    const bf16_t* __restrict__ scale1,
    const bf16_t* __restrict__ shift1,
    bf16_t* __restrict__ out0,
    bf16_t* __restrict__ out1,
    int64_t total_vec,
    int64_t seq,
    int64_t hidden,
    ParamStrides scale0_strides,
    ParamStrides shift0_strides,
    ParamStrides scale1_strides,
    ParamStrides shift1_strides) {
  using Vec = device::AlignedVector<bf16_t, kVec>;
  const int64_t row_vec = hidden / kVec;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t vi = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; vi < total_vec; vi += stride) {
    const int64_t row = vi / row_vec;
    const int64_t col_vec = vi - row * row_vec;
    const int64_t batch = row / seq;
    const int64_t token = row - batch * seq;
    const int64_t col = col_vec * kVec;

    Vec x_vec;
    Vec scale0_vec;
    Vec shift0_vec;
    Vec scale1_vec;
    Vec shift1_vec;
    Vec out0_vec;
    Vec out1_vec;
    x_vec.load(x, vi);
    scale0_vec.load(scale0 + param_offset(scale0_strides, batch, token, col), 0);
    shift0_vec.load(shift0 + param_offset(shift0_strides, batch, token, col), 0);
    scale1_vec.load(scale1 + param_offset(scale1_strides, batch, token, col), 0);
    shift1_vec.load(shift1 + param_offset(shift1_strides, batch, token, col), 0);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      out0_vec[i] = modulate_value(x_vec[i], scale0_vec[i], shift0_vec[i]);
      out1_vec[i] = modulate_value(x_vec[i], scale1_vec[i], shift1_vec[i]);
    }
    out0_vec.store(out0, vi);
    out1_vec.store(out1, vi);
  }
}

inline ParamStrides strides_for(const tvm::ffi::TensorView& t) {
  return ParamStrides{t.stride(0), t.stride(1), t.stride(2), t.size(0), t.size(1)};
}

inline void check_launchable(
    const tvm::ffi::TensorView& out,
    const tvm::ffi::TensorView& x,
    const tvm::ffi::TensorView& scale,
    const tvm::ffi::TensorView& shift) {
  check_bf16_cuda(out, "out");
  check_bf16_cuda(x, "x");
  check_bf16_cuda(scale, "scale");
  check_bf16_cuda(shift, "shift");
  check_same_device(out, x, "out", "x");
  check_same_device(out, scale, "out", "scale");
  check_same_device(out, shift, "out", "shift");
  host::RuntimeCheck(x.ndim() == 3, "x must be 3D");
  host::RuntimeCheck(out.ndim() == 3, "out must be 3D");
  for (int i = 0; i < 3; ++i) {
    host::RuntimeCheck(out.size(i) == x.size(i), "out shape must match x");
  }
  host::RuntimeCheck(is_dense_contiguous(x), "x must be contiguous");
  host::RuntimeCheck(is_dense_contiguous(out), "out must be contiguous");
  host::RuntimeCheck(x.size(2) % kVec == 0, "hidden size must be divisible by vector width");
  host::RuntimeCheck(aligned16(data_ptr(x)), "x must be 16-byte aligned");
  host::RuntimeCheck(aligned16(data_ptr(out)), "out must be 16-byte aligned");
  host::RuntimeCheck(aligned16(data_ptr(scale)), "scale must be 16-byte aligned");
  host::RuntimeCheck(aligned16(data_ptr(shift)), "shift must be 16-byte aligned");
  check_param_shape(scale, x, "scale");
  check_param_shape(shift, x, "shift");
}

}  // namespace

struct LTX2PostRMSModulateKernel {
  static void run(
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView scale,
      tvm::ffi::TensorView shift) {
    check_launchable(out, x, scale, shift);
    const int64_t total_vec = x.size(0) * x.size(1) * x.size(2) / kVec;
    if (total_vec == 0) {
      return;
    }
    host::LaunchKernel(static_cast<uint32_t>(grid_for(total_vec)), kBlockSize, out.device())(
        ltx2_post_rms_modulate_vec_kernel,
        reinterpret_cast<const bf16_t*>(data_ptr(x)),
        reinterpret_cast<const bf16_t*>(data_ptr(scale)),
        reinterpret_cast<const bf16_t*>(data_ptr(shift)),
        reinterpret_cast<bf16_t*>(mutable_data_ptr(out)),
        total_vec,
        x.size(1),
        x.size(2),
        strides_for(scale),
        strides_for(shift));
  }
};

struct LTX2PostRMSDualModulateKernel {
  static void run(
      tvm::ffi::TensorView out0,
      tvm::ffi::TensorView out1,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView scale0,
      tvm::ffi::TensorView shift0,
      tvm::ffi::TensorView scale1,
      tvm::ffi::TensorView shift1) {
    check_launchable(out0, x, scale0, shift0);
    check_launchable(out1, x, scale1, shift1);
    check_same_device(out0, out1, "out0", "out1");
    const int64_t total_vec = x.size(0) * x.size(1) * x.size(2) / kVec;
    if (total_vec == 0) {
      return;
    }
    host::LaunchKernel(static_cast<uint32_t>(grid_for(total_vec)), kBlockSize, out0.device())(
        ltx2_post_rms_dual_modulate_vec_kernel,
        reinterpret_cast<const bf16_t*>(data_ptr(x)),
        reinterpret_cast<const bf16_t*>(data_ptr(scale0)),
        reinterpret_cast<const bf16_t*>(data_ptr(shift0)),
        reinterpret_cast<const bf16_t*>(data_ptr(scale1)),
        reinterpret_cast<const bf16_t*>(data_ptr(shift1)),
        reinterpret_cast<bf16_t*>(mutable_data_ptr(out0)),
        reinterpret_cast<bf16_t*>(mutable_data_ptr(out1)),
        total_vec,
        x.size(1),
        x.size(2),
        strides_for(scale0),
        strides_for(shift0),
        strides_for(scale1),
        strides_for(shift1));
  }
};

}  // namespace sglang_ltx2_post_rms_modulate
