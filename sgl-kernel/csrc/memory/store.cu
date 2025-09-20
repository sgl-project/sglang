#include <ATen/Dispatch.h>
#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

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

}  // namespace

auto store_kv_cache(at::Tensor k_cache, at::Tensor v_cache, at::Tensor out_loc, at::Tensor k, at::Tensor v) -> void {
  const auto max_tokens = k_cache.size(0);
  const auto num_tokens = out_loc.size(0);
  k_cache = k_cache.view({max_tokens, -1});
  v_cache = v_cache.view({max_tokens, -1});
  k = k.view({num_tokens, -1});
  v = v.view({num_tokens, -1});

  TORCH_CHECK(
      k_cache.is_cuda() && v_cache.is_cuda() && out_loc.is_cuda() && k.is_cuda() && v.is_cuda(),
      "All tensors must be CUDA tensors");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "k_cache and v_cache must have the same size");
  TORCH_CHECK(k_cache.strides() == v_cache.strides(), "k_cache and v_cache must have the same strides");
  TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have the same size");
  TORCH_CHECK(k.strides() == v.strides(), "k and v must have the same strides");
  TORCH_CHECK(k.stride(-1) == 1 && k_cache.stride(-1) == 1, "k and k_cache must be contiguous in head.");
  TORCH_CHECK(k.size(-1) == k_cache.size(-1), "k and k_cache must have the same head size");
  TORCH_CHECK(out_loc.dim() == 1 && out_loc.is_contiguous(), "out_loc must be a 1D contiguous tensor");
  static_assert(sizeof(uint64_t) == 8, "uint64_t must be 8 bytes, our code assumes that");

  const auto length = out_loc.size(0);
  const auto elem_size = k.element_size();
  const auto size_bytes = elem_size * k.size(-1);
  const auto kv_cache_stride_bytes = elem_size * k_cache.stride(-2);
  const auto kv_input_stride_bytes = elem_size * k.stride(-2);
  const auto kv_cache_stride = kv_cache_stride_bytes / 8;
  const auto kv_input_stride = kv_input_stride_bytes / 8;

  const auto k_cache_ptr = static_cast<uint64_t*>(k_cache.data_ptr());
  const auto v_cache_ptr = static_cast<uint64_t*>(v_cache.data_ptr());
  const auto k_ptr = static_cast<const uint64_t*>(k.data_ptr());
  const auto v_ptr = static_cast<const uint64_t*>(v.data_ptr());
  const auto num_threads = 256;
  const auto num_warps = num_threads / 32;
  const auto num_blocks = (length + num_warps - 1) / num_warps;
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_INTEGRAL_TYPES(out_loc.scalar_type(), "store_kv_cache", [&] {
    if constexpr (!std::is_same_v<scalar_t, int32_t> && !std::is_same_v<scalar_t, int64_t>) {
      // do not instantiate the kernel if out_loc is not int32 or int64
      TORCH_CHECK(false, "out_loc must be of type int32 or int64, got: ", out_loc.scalar_type());
    } else {
      if (size_bytes % 256 == 0) {
        const auto items_per_warp = size_bytes / 256;
        store_kv_cache_256x1<<<num_blocks, num_threads, 0, stream>>>(
            k_cache_ptr,
            v_cache_ptr,
            out_loc.data_ptr<scalar_t>(),
            length,
            k_ptr,
            v_ptr,
            kv_cache_stride,
            kv_input_stride,
            items_per_warp);
      } else if (size_bytes % 128 == 0) {
        const auto items_per_warp = size_bytes / 128;
        store_kv_cache_128x2<<<num_blocks, num_threads, 0, stream>>>(
            k_cache_ptr,
            v_cache_ptr,
            out_loc.data_ptr<scalar_t>(),
            length,
            k_ptr,
            v_ptr,
            kv_cache_stride,
            kv_input_stride,
            items_per_warp);
      } else {
        TORCH_CHECK(
            false,
            "The last dimension size bytes of k and v must be"
            " divisible by 128 at least, got: ",
            size_bytes);
      }
    }
  });
}
