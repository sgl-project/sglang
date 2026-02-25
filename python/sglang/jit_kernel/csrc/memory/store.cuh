// Adapted from https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/memory/store.cu
#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <cstddef>
#include <cstdint>

namespace {

using std::size_t;
using std::uint64_t;

// Each warp will process 256 bytes per loop iteration
template <typename T>
__global__ void store_kv_cache_256x1(
    uint64_t* __restrict__ k_cache,
    uint64_t* __restrict__ v_cache,
    const T* __restrict__ out_loc,
    const size_t length,
    const uint64_t* __restrict__ k,
    const uint64_t* __restrict__ v,
    const size_t kv_cache_stride,
    const size_t kv_input_stride,
    const size_t num_items) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto warp_id = idx / 32;
  const auto lane_id = idx % 32;
  if (warp_id >= length) return;
  const auto offset = out_loc[warp_id];
  const auto k_dst = k_cache + offset * kv_cache_stride;
  const auto v_dst = v_cache + offset * kv_cache_stride;
  const auto k_src = k + warp_id * kv_input_stride;
  const auto v_src = v + warp_id * kv_input_stride;
  for (size_t i = 0; i < num_items; ++i) {
    k_dst[lane_id + i * 32] = k_src[lane_id + i * 32];
    v_dst[lane_id + i * 32] = v_src[lane_id + i * 32];
  }
}

// Each warp will process 128 bytes per loop iteration
template <typename T>
__global__ void store_kv_cache_128x2(
    uint64_t* __restrict__ k_cache,
    uint64_t* __restrict__ v_cache,
    const T* __restrict__ out_loc,
    const size_t length,
    const uint64_t* __restrict__ k,
    const uint64_t* __restrict__ v,
    const size_t kv_cache_stride,
    const size_t kv_input_stride,
    const size_t num_items) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto warp_id = idx / 32;
  const auto lane_id = idx % 32;
  if (warp_id >= length) return;
  const auto offset = out_loc[warp_id];
  const auto copy_k = lane_id < 16;
  const auto copy_id = lane_id % 16;
  const auto cache = copy_k ? k_cache : v_cache;
  const auto input = copy_k ? k : v;
  const auto dst = cache + offset * kv_cache_stride;
  const auto src = input + warp_id * kv_input_stride;
  for (size_t i = 0; i < num_items; ++i) {
    dst[copy_id + i * 16] = src[copy_id + i * 16];
  }
}

template <typename T>
void dispatch_store_kv_cache(
    uint64_t* k_cache_ptr,
    uint64_t* v_cache_ptr,
    const T* out_loc_ptr,
    const size_t length,
    const uint64_t* k_ptr,
    const uint64_t* v_ptr,
    const size_t kv_cache_stride,
    const size_t kv_input_stride,
    const int64_t size_bytes,
    const int num_blocks,
    const int num_threads,
    cudaStream_t stream) {
  if (size_bytes % 256 == 0) {
    const size_t items_per_warp = static_cast<size_t>(size_bytes / 256);
    store_kv_cache_256x1<<<num_blocks, num_threads, 0, stream>>>(
        k_cache_ptr,
        v_cache_ptr,
        out_loc_ptr,
        length,
        k_ptr,
        v_ptr,
        kv_cache_stride,
        kv_input_stride,
        items_per_warp);
  } else if (size_bytes % 128 == 0) {
    const size_t items_per_warp = static_cast<size_t>(size_bytes / 128);
    store_kv_cache_128x2<<<num_blocks, num_threads, 0, stream>>>(
        k_cache_ptr,
        v_cache_ptr,
        out_loc_ptr,
        length,
        k_ptr,
        v_ptr,
        kv_cache_stride,
        kv_input_stride,
        items_per_warp);
  } else {
    host::Panic(
        "Last dim size bytes of k/v must be divisible by 128, got: {}", size_bytes);
  }
}

// Expects 2D inputs: k_cache/v_cache shape (max_tokens, head_dim),
// k/v shape (num_tokens, head_dim), out_loc shape (num_tokens,).
void store_kv_cache(
    tvm::ffi::TensorView k_cache,
    tvm::ffi::TensorView v_cache,
    tvm::ffi::TensorView out_loc,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v) {
  using namespace host;

  RuntimeCheck(k_cache.dim() == 2, "k_cache must be 2D");
  RuntimeCheck(v_cache.dim() == 2, "v_cache must be 2D");
  RuntimeCheck(k.dim() == 2, "k must be 2D");
  RuntimeCheck(v.dim() == 2, "v must be 2D");
  RuntimeCheck(out_loc.dim() == 1 && out_loc.is_contiguous(), "out_loc must be 1D contiguous");
  RuntimeCheck(
      k_cache.size(1) == v_cache.size(1),
      "k_cache and v_cache must have the same head dim");
  RuntimeCheck(k.size(1) == v.size(1), "k and v must have the same head dim");
  RuntimeCheck(k.size(1) == k_cache.size(1), "k and k_cache must have the same head dim");
  RuntimeCheck(k.stride(1) == 1 && k_cache.stride(1) == 1, "k and k_cache must be contiguous in head dim");
  static_assert(sizeof(uint64_t) == 8, "uint64_t must be 8 bytes");

  const size_t length = static_cast<size_t>(out_loc.size(0));
  const int64_t elem_size = k.dtype().bits / 8;
  const int64_t size_bytes = elem_size * k.size(1);
  const size_t kv_cache_stride = static_cast<size_t>(elem_size * k_cache.stride(0) / 8);
  const size_t kv_input_stride = static_cast<size_t>(elem_size * k.stride(0) / 8);

  const auto k_cache_ptr = static_cast<uint64_t*>(k_cache.data_ptr());
  const auto v_cache_ptr = static_cast<uint64_t*>(v_cache.data_ptr());
  const auto k_ptr = static_cast<const uint64_t*>(k.data_ptr());
  const auto v_ptr = static_cast<const uint64_t*>(v.data_ptr());

  constexpr int num_threads = 256;
  constexpr int num_warps = num_threads / 32;
  const int num_blocks = static_cast<int>((length + num_warps - 1) / num_warps);

  const auto device = k_cache.device();
  const auto stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));

  if (host::is_type<int32_t>(out_loc.dtype())) {
    dispatch_store_kv_cache<int32_t>(
        k_cache_ptr,
        v_cache_ptr,
        static_cast<const int32_t*>(out_loc.data_ptr()),
        length,
        k_ptr,
        v_ptr,
        kv_cache_stride,
        kv_input_stride,
        size_bytes,
        num_blocks,
        num_threads,
        stream);
  } else if (host::is_type<int64_t>(out_loc.dtype())) {
    dispatch_store_kv_cache<int64_t>(
        k_cache_ptr,
        v_cache_ptr,
        static_cast<const int64_t*>(out_loc.data_ptr()),
        length,
        k_ptr,
        v_ptr,
        kv_cache_stride,
        kv_input_stride,
        size_bytes,
        num_blocks,
        num_threads,
        stream);
  } else {
    RuntimeCheck(false, "out_loc must be int32 or int64");
  }
}

}  // namespace
