#pragma once

// Optimized cast kernel: fixed 256 threads, scaled out via 2D grid.
// Each thread handles exactly one float4 (kVecSize fp16/bf16 elements).
// No per-thread loop — pure grid scaling for any head*dim.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>   // For dtype_trait fp8 specialization
#include <sgl_kernel/utils.cuh>  // For LaunchKernel
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

constexpr int kBlockSize = 256;

template <typename T>
__global__ void fused_downcast_kernel(
    const T* __restrict__ cache_k,
    const T* __restrict__ cache_v,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    fp8_e4m3_t* __restrict__ output_k,
    fp8_e4m3_t* __restrict__ output_v,
    const int input_num_tokens,
    const int head,
    const int dim,
    const T max_fp8,
    const T min_fp8,
    const int64_t mult,
    const int64_t offset,
    const int64_t* __restrict__ loc) {
  using namespace device;

  constexpr int kVecSize = 16 / sizeof(T);
  using vec_t = AlignedVector<T, kVecSize>;
  using out_vec_t = AlignedVector<fp8_e4m3_t, kVecSize>;

  const int token_idx = blockIdx.x;
  const int vec_idx = blockIdx.y * kBlockSize + threadIdx.x;
  const int num_vecs = head * dim / kVecSize;

  if (token_idx >= input_num_tokens || vec_idx >= num_vecs) return;

  T k_scale_inv = static_cast<T>(1.f) / cast<T>(k_scale[0]);
  T v_scale_inv = static_cast<T>(1.f) / cast<T>(v_scale[0]);

  auto clamp = [&](T val) { return val > max_fp8 ? max_fp8 : (min_fp8 > val ? min_fp8 : val); };

  const int out_seq_idx = loc[token_idx];
  const T* in_k_base = cache_k + token_idx * head * dim;
  const T* in_v_base = cache_v + token_idx * head * dim;
  fp8_e4m3_t* out_k_base = output_k + (out_seq_idx * mult + offset) * head * dim;
  fp8_e4m3_t* out_v_base = output_v + (out_seq_idx * mult + offset) * head * dim;

  vec_t k_vec, v_vec;
  k_vec.load(in_k_base, vec_idx);
  v_vec.load(in_v_base, vec_idx);

  out_vec_t out_k, out_v;
#pragma unroll
  for (int j = 0; j < kVecSize; j++) {
    out_k[j] = cast<fp8_e4m3_t>(clamp(k_vec[j] * k_scale_inv));
    out_v[j] = cast<fp8_e4m3_t>(clamp(v_vec[j] * v_scale_inv));
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

  auto input_num_tokens = SymbolicSize{"input_num_tokens"};
  auto head = SymbolicSize{"head"};
  auto dim = SymbolicSize{"dim"};
  auto output_num_tokens = SymbolicSize{"out_sl"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({input_num_tokens, head, dim}).with_dtype<T>().with_device(device).verify(k);
  TensorMatcher({input_num_tokens, head, dim}).with_dtype<T>().with_device(device).verify(v);
  TensorMatcher({output_num_tokens, head, dim}).with_dtype<uint8_t>().with_device(device).verify(k_out);
  TensorMatcher({output_num_tokens, head, dim}).with_dtype<uint8_t>().with_device(device).verify(v_out);
  TensorMatcher({1}).with_dtype<float>().with_device(device).verify(k_scale);
  TensorMatcher({1}).with_dtype<float>().with_device(device).verify(v_scale);
  TensorMatcher({input_num_tokens}).with_dtype<int64_t>().with_device(device).verify(loc);

  const int num_tokens = static_cast<int>(input_num_tokens.unwrap());
  const int h = static_cast<int>(head.unwrap());
  const int d = static_cast<int>(dim.unwrap());

  constexpr int kVecSize = 16 / sizeof(T);
  const int num_vecs = h * d / kVecSize;
  const int grid_y = (num_vecs + kBlockSize - 1) / kBlockSize;

  dim3 grid(num_tokens, grid_y);
  dim3 block(kBlockSize);

  const T max_fp8 = static_cast<T>(kFP8E4M3Max);
  const T min_fp8 = static_cast<T>(-kFP8E4M3Max);

  LaunchKernel(grid, block, device.unwrap())(
      fused_downcast_kernel<T>,
      static_cast<const T*>(k.data_ptr()),
      static_cast<const T*>(v.data_ptr()),
      static_cast<const float*>(k_scale.data_ptr()),
      static_cast<const float*>(v_scale.data_ptr()),
      static_cast<fp8_e4m3_t*>(k_out.data_ptr()),
      static_cast<fp8_e4m3_t*>(v_out.data_ptr()),
      num_tokens,
      h,
      d,
      max_fp8,
      min_fp8,
      mult,
      offset,
      static_cast<const int64_t*>(loc.data_ptr()));
}

}  // namespace
