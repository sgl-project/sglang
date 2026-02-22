#pragma once

// Optimized cast kernel: fixed 256 threads, scaled out via 2D grid.
// Each thread handles exactly one float4 (kVecSize fp16/bf16 elements).
// No per-thread loop — pure grid scaling for any head*dim.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>   // For ConvertToFP8, ConvertFromFloat
#include <sgl_kernel/utils.cuh>  // For LaunchKernel
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr int kBlockSize = 256;

template <typename T>
__global__ void fused_downcast_kernel(
    const T* cache_k,
    const T* cache_v,
    const float* k_scale,
    const float* v_scale,
    __nv_fp8_storage_t* output_k,
    __nv_fp8_storage_t* output_v,
    const int input_sl,
    const int head,
    const int dim,
    const T max_fp8,
    const T min_fp8,
    const int64_t mult,
    const int64_t offset,
    const int64_t* loc) {
  using namespace device;

  constexpr int kVecSize = 16 / sizeof(T);
  using vec_t = AlignedVector<T, kVecSize>;
  using out_vec_t = AlignedVector<uint8_t, kVecSize>;

  const int token_idx = blockIdx.x;
  const int vec_idx = blockIdx.y * kBlockSize + threadIdx.x;
  const int num_vecs = head * dim / kVecSize;

  if (token_idx >= input_sl || vec_idx >= num_vecs) return;

  T k_scale_inv = static_cast<T>(1.f) / ConvertFromFloat<T>::convert_from_float(k_scale[0]);
  T v_scale_inv = static_cast<T>(1.f) / ConvertFromFloat<T>::convert_from_float(v_scale[0]);

  auto clamp = [&](T val) { return val > max_fp8 ? max_fp8 : (min_fp8 > val ? min_fp8 : val); };

  const int out_seq_idx = loc[token_idx];
  const T* in_k_base = cache_k + token_idx * head * dim;
  const T* in_v_base = cache_v + token_idx * head * dim;
  __nv_fp8_storage_t* out_k_base = output_k + (out_seq_idx * mult + offset) * head * dim;
  __nv_fp8_storage_t* out_v_base = output_v + (out_seq_idx * mult + offset) * head * dim;

  vec_t k_vec, v_vec;
  k_vec.load(in_k_base, vec_idx);
  v_vec.load(in_v_base, vec_idx);

  out_vec_t out_k, out_v;
#pragma unroll
  for (int j = 0; j < kVecSize; j++) {
    out_k[j] = ConvertToFP8<T>::convert_to_fp8(clamp(k_vec[j] * k_scale_inv));
    out_v[j] = ConvertToFP8<T>::convert_to_fp8(clamp(v_vec[j] * v_scale_inv));
  }

  out_k.store(out_k_base, vec_idx);
  out_v.store(out_v_base, vec_idx);
}

template <typename T>
void downcast_fp8(
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    tvm::ffi::TensorView k_out,
    tvm::ffi::TensorView v_out,
    tvm::ffi::TensorView k_scale,
    tvm::ffi::TensorView v_scale,
    tvm::ffi::TensorView loc,
    int64_t mult,
    int64_t offset) {
  using namespace host;

  auto input_sl = SymbolicSize{"input_sl"};
  auto head = SymbolicSize{"head"};
  auto dim = SymbolicSize{"dim"};
  auto out_sl = SymbolicSize{"out_sl"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({input_sl, head, dim}).with_dtype<T>().with_device(device).verify(k);
  TensorMatcher({input_sl, head, dim}).with_dtype<T>().with_device(device).verify(v);
  TensorMatcher({out_sl, head, dim}).with_dtype<uint8_t>().with_device(device).verify(k_out);
  TensorMatcher({out_sl, head, dim}).with_dtype<uint8_t>().with_device(device).verify(v_out);
  TensorMatcher({1}).with_dtype<float>().with_device(device).verify(k_scale);
  TensorMatcher({1}).with_dtype<float>().with_device(device).verify(v_scale);
  TensorMatcher({input_sl}).with_dtype<int64_t>().with_device(device).verify(loc);

  const int isl = static_cast<int>(input_sl.unwrap());
  const int h = static_cast<int>(head.unwrap());
  const int d = static_cast<int>(dim.unwrap());

  constexpr int kVecSize = 16 / sizeof(T);
  const int num_vecs = h * d / kVecSize;
  const int grid_y = (num_vecs + kBlockSize - 1) / kBlockSize;

  dim3 grid(isl, grid_y);
  dim3 block(kBlockSize);

  const T max_fp8 = static_cast<T>(448.0f);
  const T min_fp8 = static_cast<T>(-448.0f);

  LaunchKernel(grid, block, device.unwrap())(
      fused_downcast_kernel<T>,
      static_cast<const T*>(k.data_ptr()),
      static_cast<const T*>(v.data_ptr()),
      static_cast<const float*>(k_scale.data_ptr()),
      static_cast<const float*>(v_scale.data_ptr()),
      static_cast<__nv_fp8_storage_t*>(k_out.data_ptr()),
      static_cast<__nv_fp8_storage_t*>(v_out.data_ptr()),
      isl,
      h,
      d,
      max_fp8,
      min_fp8,
      mult,
      offset,
      static_cast<const int64_t*>(loc.data_ptr()));
}

}  // namespace
