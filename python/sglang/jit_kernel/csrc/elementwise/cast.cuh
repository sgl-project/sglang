#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicDevice, SymbolicSize
#include <sgl_kernel/type.cuh>  // For ConvertToFP8, ConvertFromFloat
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

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
  int token_idx = blockIdx.x;
  int thread_idx = threadIdx.x;
  int total_threads = blockDim.x;

  T k_scale_val = ConvertFromFloat<T>::convert_from_float(k_scale[0]);
  T v_scale_val = ConvertFromFloat<T>::convert_from_float(v_scale[0]);

  T k_scale_inv = static_cast<T>(1.f) / k_scale_val;
  T v_scale_inv = static_cast<T>(1.f) / v_scale_val;

  auto clamp = [&](T val) { return val > max_fp8 ? max_fp8 : (min_fp8 > val ? min_fp8 : val); };

  if (token_idx < input_sl) {
    int out_seq_idx = loc[token_idx];

#pragma unroll
    for (int i = thread_idx; i < head * dim; i += total_threads) {
      int in_idx = token_idx * head * dim + i;
      int out_idx = (out_seq_idx * mult + offset) * head * dim + i;

      T k_val = cache_k[in_idx] * k_scale_inv;
      k_val = clamp(k_val);
      output_k[out_idx] = ConvertToFP8<T>::convert_to_fp8(k_val);

      T v_val = cache_v[in_idx] * v_scale_inv;
      v_val = clamp(v_val);
      output_v[out_idx] = ConvertToFP8<T>::convert_to_fp8(v_val);
    }
  }
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

  const int vec_size = 8;
  dim3 grid(isl);
  dim3 block(std::min(d / vec_size, 1024));

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
