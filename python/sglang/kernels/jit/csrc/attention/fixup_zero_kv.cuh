#pragma once

// Fixup kernel for TRT-LLM ragged attention zero-KV rows.
// For sequences with kv_len == 0, forces out=0 and lse=-inf.
// One CTA per (sequence, query offset), with a flat launch grid.
// Uses vectorised float4 stores for bandwidth efficiency.

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace sglang {

constexpr int kFixupBlockSize = 256;

// -- vectorised zero-fill helpers ------------------------------------------

// Zero-fill `n` elements of type T starting at `ptr`.
template <typename T>
__device__ __forceinline__ void vec_zero_fill(T* ptr, int n) {
  if ((reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0) {
    // 16-byte aligned -> vectorised float4 stores
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
  } else {
    // misaligned row base -> scalar stores
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      ptr[i] = static_cast<T>(0);
    }
  }
}

// Fill `n` float elements with -inf.
__device__ __forceinline__ void vec_neginf_fill(float* ptr, int n) {
  const float ninf = -INFINITY;
  if ((reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0) {
    // 16-byte aligned -> vectorised float4 stores
    constexpr int kVec = 4;  // float4 = 4 floats
    const int n_vec = n / kVec;
    float4* dst4 = reinterpret_cast<float4*>(ptr);
    const float4 inf4 = make_float4(ninf, ninf, ninf, ninf);
    for (int i = threadIdx.x; i < n_vec; i += blockDim.x) {
      dst4[i] = inf4;
    }
    const int tail_start = n_vec * kVec;
    for (int i = tail_start + threadIdx.x; i < n; i += blockDim.x) {
      ptr[i] = ninf;
    }
  } else {
    // misaligned row base -> scalar stores
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      ptr[i] = ninf;
    }
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
    const int lse_stride,
    const int blocks_per_seq) {
  const int seq_idx = blockIdx.x / blocks_per_seq;
  if (kv_lens[seq_idx] > 0) return;

  const int tok_start = cum_seq_lens[seq_idx];
  const int tok_end = cum_seq_lens[seq_idx + 1];
  const int num_tokens = tok_end - tok_start;
  if (num_tokens <= 0) return;

  const int query_offset = static_cast<int>(blockIdx.x % blocks_per_seq);
  if (query_offset >= num_tokens) return;

  const int64_t tok = static_cast<int64_t>(tok_start) + query_offset;
  const int64_t out_offset = tok * static_cast<int64_t>(out_stride);
  const int64_t lse_offset = tok * static_cast<int64_t>(lse_stride);

  // Each block handles one token: zero out[tok] and set lse[tok] = -inf.
  vec_zero_fill(out + out_offset, out_stride);
  vec_neginf_fill(lse + lse_offset, lse_stride);
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
  device.set_options<kDLGPU>();

  TensorMatcher({total_tokens, num_heads, v_head_dim}).with_dtype<OutT>().with_device(device).verify(out);
  TensorMatcher({total_tokens, num_heads}).with_dtype<float>().with_device(device).verify(lse);
  TensorMatcher({batch_size}).with_dtype<int32_t>().with_device(device).verify(kv_lens);
  TensorMatcher({batch_size_plus_1}).with_dtype<int32_t>().with_device(device).verify(cum_seq_lens);

  const int bs = static_cast<int>(batch_size.unwrap());
  const int nh = static_cast<int>(num_heads.unwrap());
  const int vd = static_cast<int>(v_head_dim.unwrap());
  const int64_t out_stride = static_cast<int64_t>(nh) * vd;
  RuntimeCheck(out_stride >= 0 && out_stride <= 2147483647, "invalid fixup output stride");

  if (bs == 0 || max_seq_len == 0) return;
  const int64_t total_blocks = bs * max_seq_len;
  RuntimeCheck(total_blocks > 0 && total_blocks <= 2147483647, "invalid fixup grid size");
  dim3 grid(static_cast<unsigned int>(total_blocks));
  dim3 block(kFixupBlockSize);

  LaunchKernel(grid, block, device.unwrap())(
      fixup_zero_kv_rows_kernel<OutT>,
      static_cast<OutT*>(out.data_ptr()),
      static_cast<float*>(lse.data_ptr()),
      static_cast<const int32_t*>(kv_lens.data_ptr()),
      static_cast<const int32_t*>(cum_seq_lens.data_ptr()),
      static_cast<int>(out_stride),
      nh,
      static_cast<int>(max_seq_len));
}

}  // namespace sglang
