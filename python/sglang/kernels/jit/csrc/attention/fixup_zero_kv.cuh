#pragma once

// Fixup kernel for TRT-LLM ragged attention zero-KV rows.
// For sequences with kv_len == 0, forces out=0 and lse=-inf.
// 2D grid: (blocks_per_seq, batch_size). Y-dim early-exits for non-zero KV.
//
// Store modes are compiled independently for out and lse. The host launcher
// selects the right specialization at dispatch time:
//   vector -- float4 stores (16-byte aligned row bases, maximum bandwidth)
//   scalar -- element-wise stores (any alignment, e.g. TP configs where
//             lse row stride = nh*4 bytes is not a 16-byte multiple)
// The alignment check is done once on the host; no branch lives in the kernel.

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace {

constexpr int kFixupBlockSize = 256;

// -- fill helpers: vectorised (16-byte aligned rows) -----------------------

template <typename T>
__device__ __forceinline__ void vec_zero_fill(T* ptr, int n) {
  constexpr int kVec = 16 / sizeof(T);
  const int n_vec = n / kVec;
  float4* dst4 = reinterpret_cast<float4*>(ptr);
  const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int i = threadIdx.x; i < n_vec; i += blockDim.x)
    dst4[i] = z4;
  const int tail_start = n_vec * kVec;
  for (int i = tail_start + threadIdx.x; i < n; i += blockDim.x)
    ptr[i] = static_cast<T>(0);
}

__device__ __forceinline__ void vec_neginf_fill(float* ptr, int n) {
  constexpr int kVec = 4;
  const int n_vec = n / kVec;
  float4* dst4 = reinterpret_cast<float4*>(ptr);
  const float ninf = -INFINITY;
  const float4 inf4 = make_float4(ninf, ninf, ninf, ninf);
  for (int i = threadIdx.x; i < n_vec; i += blockDim.x)
    dst4[i] = inf4;
  const int tail_start = n_vec * kVec;
  for (int i = tail_start + threadIdx.x; i < n; i += blockDim.x)
    ptr[i] = ninf;
}

// -- fill helpers: scalar (arbitrary alignment) ----------------------------

template <typename T>
__device__ __forceinline__ void scalar_zero_fill(T* ptr, int n) {
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    ptr[i] = static_cast<T>(0);
}

__device__ __forceinline__ void scalar_neginf_fill(float* ptr, int n) {
  const float ninf = -INFINITY;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    ptr[i] = ninf;
}

// -- kernels ---------------------------------------------------------------

template <typename OutT, bool VecOut, bool VecLse>
__global__ void fixup_zero_kv_rows_kernel(
    OutT* __restrict__ out,
    float* __restrict__ lse,
    const int32_t* __restrict__ kv_lens,
    const int32_t* __restrict__ cum_seq_lens,
    const int out_stride,
    const int lse_stride) {
  const int seq_idx = blockIdx.y;
  if (kv_lens[seq_idx] > 0) return;
  const int tok_start = cum_seq_lens[seq_idx];
  const int tok_end = cum_seq_lens[seq_idx + 1];
  if (tok_start >= tok_end) return;
  const int tok = tok_start + blockIdx.x;
  if (tok >= tok_end) return;
  if constexpr (VecOut) {
    vec_zero_fill(out + tok * out_stride, out_stride);
  } else {
    scalar_zero_fill(out + tok * out_stride, out_stride);
  }
  if constexpr (VecLse) {
    vec_neginf_fill(lse + tok * lse_stride, lse_stride);
  } else {
    scalar_neginf_fill(lse + tok * lse_stride, lse_stride);
  }
}

// -- host launcher ---------------------------------------------------------

template <typename OutT>
void fixup_zero_kv_rows(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView lse,
    tvm::ffi::TensorView kv_lens,
    tvm::ffi::TensorView cum_seq_lens,
    int64_t max_seq_len) {
  using namespace host;

  auto batch_size = SymbolicSize{"batch_size"};
  auto total_tokens = SymbolicSize{"total_tokens"};
  auto num_heads = SymbolicSize{"num_heads"};
  auto v_head_dim = SymbolicSize{"v_head_dim"};
  auto batch_size_plus_1 = SymbolicSize{"batch_size_plus_1"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({total_tokens, num_heads, v_head_dim}).with_dtype<OutT>().with_device(device).verify(out);
  TensorMatcher({total_tokens, num_heads}).with_dtype<float>().with_device(device).verify(lse);
  TensorMatcher({batch_size}).with_dtype<int32_t>().with_device(device).verify(kv_lens);
  TensorMatcher({batch_size_plus_1}).with_dtype<int32_t>().with_device(device).verify(cum_seq_lens);

  const int bs = static_cast<int>(batch_size.unwrap());
  const int nh = static_cast<int>(num_heads.unwrap());
  const int vd = static_cast<int>(v_head_dim.unwrap());

  const int blocks_x = static_cast<int>(max_seq_len);
  dim3 grid(blocks_x, bs);
  dim3 block(kFixupBlockSize);

  auto* out_ptr = static_cast<OutT*>(out.data_ptr());
  auto* lse_ptr = static_cast<float*>(lse.data_ptr());

  // Every row base is 16-byte aligned iff the current data pointer and row
  // stride are both 16-byte aligned. The data pointer may include a non-zero
  // PyTorch storage offset even for a contiguous view.
  const bool lse_aligned = (reinterpret_cast<uintptr_t>(lse_ptr) % 16 == 0) && ((nh * sizeof(float)) % 16 == 0);
  const bool out_aligned = (reinterpret_cast<uintptr_t>(out_ptr) % 16 == 0) && ((nh * vd * sizeof(OutT)) % 16 == 0);

  if (lse_aligned && out_aligned) {
    LaunchKernel(grid, block, device.unwrap())(
        fixup_zero_kv_rows_kernel<OutT, true, true>,
        out_ptr,
        lse_ptr,
        static_cast<const int32_t*>(kv_lens.data_ptr()),
        static_cast<const int32_t*>(cum_seq_lens.data_ptr()),
        nh * vd,
        nh);
  } else if (out_aligned) {
    LaunchKernel(grid, block, device.unwrap())(
        fixup_zero_kv_rows_kernel<OutT, true, false>,
        out_ptr,
        lse_ptr,
        static_cast<const int32_t*>(kv_lens.data_ptr()),
        static_cast<const int32_t*>(cum_seq_lens.data_ptr()),
        nh * vd,
        nh);
  } else if (lse_aligned) {
    LaunchKernel(grid, block, device.unwrap())(
        fixup_zero_kv_rows_kernel<OutT, false, true>,
        out_ptr,
        lse_ptr,
        static_cast<const int32_t*>(kv_lens.data_ptr()),
        static_cast<const int32_t*>(cum_seq_lens.data_ptr()),
        nh * vd,
        nh);
  } else {
    LaunchKernel(grid, block, device.unwrap())(
        fixup_zero_kv_rows_kernel<OutT, false, false>,
        out_ptr,
        lse_ptr,
        static_cast<const int32_t*>(kv_lens.data_ptr()),
        static_cast<const int32_t*>(cum_seq_lens.data_ptr()),
        nh * vd,
        nh);
  }
}

}  // namespace
