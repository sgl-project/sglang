#pragma once

// Fixup kernel for TRT-LLM ragged attention zero-KV rows.
// For sequences with kv_len == 0, forces out=0 and lse=-inf.
// 2D grid: (blocks_per_seq, batch_size). Y-dim early-exits for non-zero KV.
// Uses vectorised float4 stores for bandwidth efficiency.

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace {

constexpr int kFixupBlockSize = 256;

// -- vectorised zero-fill helpers ------------------------------------------

// Zero-fill `n` elements of type T starting at `ptr`, using float4 stores.
// `ptr` must be 16-byte aligned (guaranteed by PyTorch allocator).
template <typename T>
__device__ __forceinline__ void vec_zero_fill(T* ptr, int n) {
  constexpr int kVec = 16 / sizeof(T);  // elements per float4
  const int n_vec = n / kVec;           // full vectors
  float4* dst4 = reinterpret_cast<float4*>(ptr);
  const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int i = threadIdx.x; i < n_vec; i += blockDim.x) {
    dst4[i] = z4;
  }
  // tail elements
  const int tail_start = n_vec * kVec;
  for (int i = tail_start + threadIdx.x; i < n; i += blockDim.x) {
    ptr[i] = static_cast<T>(0);
  }
}

// Fill `n` float elements with -inf using float4 stores.
__device__ __forceinline__ void vec_neginf_fill(float* ptr, int n) {
  constexpr int kVec = 4;  // float4 = 4 floats
  const int n_vec = n / kVec;
  float4* dst4 = reinterpret_cast<float4*>(ptr);
  const float ninf = -INFINITY;
  const float4 inf4 = make_float4(ninf, ninf, ninf, ninf);
  for (int i = threadIdx.x; i < n_vec; i += blockDim.x) {
    dst4[i] = inf4;
  }
  const int tail_start = n_vec * kVec;
  for (int i = tail_start + threadIdx.x; i < n; i += blockDim.x) {
    ptr[i] = ninf;
  }
}

// -- main kernel -----------------------------------------------------------

template <typename OutT>
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
  const int num_tokens = tok_end - tok_start;
  if (num_tokens <= 0) return;

  // blockIdx.x selects a token within this sequence.
  const int tok = tok_start + blockIdx.x;
  if (tok >= tok_end) return;

  // Each block handles one token: zero out[tok] and set lse[tok] = -inf.
  vec_zero_fill(out + tok * out_stride, out_stride);
  vec_neginf_fill(lse + tok * lse_stride, lse_stride);
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

  // Grid: one block per (token, sequence). X = max tokens in any seq.
  const int blocks_x = static_cast<int>(max_seq_len);
  dim3 grid(blocks_x, bs);
  dim3 block(kFixupBlockSize);

  LaunchKernel(grid, block, device.unwrap())(
      fixup_zero_kv_rows_kernel<OutT>,
      static_cast<OutT*>(out.data_ptr()),
      static_cast<float*>(lse.data_ptr()),
      static_cast<const int32_t*>(kv_lens.data_ptr()),
      static_cast<const int32_t*>(cum_seq_lens.data_ptr()),
      nh * vd,
      nh);
}

}  // namespace
