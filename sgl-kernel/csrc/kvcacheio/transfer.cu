#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/irange.h>

#include <cstdint>

#include "pytorch_extension_utils.h"

__device__ __forceinline__ void
transfer_item_warp(int32_t lane_id, const void* src_addr, void* dst_addr, int64_t item_size_bytes) {
  // todo, different chunk size
  int total_chunks = item_size_bytes / 8;
  const int64_t* src_8 = reinterpret_cast<const int64_t*>(src_addr);
  int64_t* dst_8 = reinterpret_cast<int64_t*>(dst_addr);
#pragma unroll
  for (int j = lane_id; j < total_chunks; j += 32) {
    const int64_t* src_addr_lane = &src_8[j];
    int64_t* dst_addr_lane = &dst_8[j];
    int64_t temp_val;
    asm volatile("ld.global.nc.b64 %0, [%1];" : "=l"(temp_val) : "l"(src_addr_lane) : "memory");
    asm volatile("st.global.cg.b64 [%0], %1;" ::"l"(dst_addr_lane), "l"(temp_val) : "memory");
  }
}

// todo, structs for different memory layout
__device__ __forceinline__ int64_t
get_global_offset_lf(int64_t layer_id, int64_t layer_dim, int64_t page_id, int64_t item_size_bytes) {
  // layer first
  return layer_id * layer_dim + page_id * item_size_bytes;
}

__device__ __forceinline__ int64_t
get_global_offset_pf(int64_t layer_id, int64_t page_dim, int64_t page_id, int64_t item_size_bytes) {
  // page first
  return page_id * page_dim + layer_id * item_size_bytes;
}

template <auto SrcOffsetFn, auto DstOffsetFn, bool IsMLA>
__global__ void transfer_kernel_impl(
    const void* __restrict__ src_k,
    void* __restrict__ dst_k,
    const void* __restrict__ src_v,
    void* __restrict__ dst_v,
    const int64_t* __restrict__ src_indices,
    const int64_t* __restrict__ dst_indices,
    int64_t start_layer_id,
    int64_t num_layers_to_process,
    int64_t num_items,
    int64_t items_per_warp,
    int64_t item_size_bytes,
    int64_t src_layout_dim,
    int64_t dst_layout_dim) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t lane_id = tid % 32;
  int32_t warp_id = tid / 32;

  for (int i = 0; i < items_per_warp; ++i) {
    int32_t item_id = warp_id * items_per_warp + i;
    if (item_id >= num_items) {
      return;
    }
    const int64_t src_page_id = src_indices[item_id];
    const int64_t dst_page_id = dst_indices[item_id];

    // Loop over layers if necessary
    for (int64_t layer_id = start_layer_id; layer_id < start_layer_id + num_layers_to_process; ++layer_id) {
      // Calculate offsets using the provided function pointers
      const int64_t src_offset = SrcOffsetFn(layer_id, src_layout_dim, src_page_id, item_size_bytes);
      const int64_t dst_offset = DstOffsetFn(layer_id, dst_layout_dim, dst_page_id, item_size_bytes);

      if constexpr (IsMLA) {
        transfer_item_warp(
            lane_id,
            static_cast<const char*>(src_k) + src_offset,
            static_cast<char*>(dst_k) + dst_offset,
            item_size_bytes);
      } else {
        transfer_item_warp(
            lane_id,
            static_cast<const char*>(src_k) + src_offset,
            static_cast<char*>(dst_k) + dst_offset,
            item_size_bytes);
        transfer_item_warp(
            lane_id,
            static_cast<const char*>(src_v) + src_offset,
            static_cast<char*>(dst_v) + dst_offset,
            item_size_bytes);
      }
    }
  }
}

template <auto SrcOffsetFn, auto DstOffsetFn, bool IsMLA>
void transfer_kv_launcher(
    const at::Tensor& src_k,
    at::Tensor& dst_k,
    const at::Tensor& src_v,
    at::Tensor& dst_v,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t start_layer_id,
    int64_t num_layers_to_process,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t dst_layout_dim,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  TORCH_CHECK(src_k.scalar_type() == dst_k.scalar_type(), "Source and destination keys must have the same type");
  TORCH_CHECK(src_indices.is_cuda(), "Source indices must be a CUDA tensor");
  TORCH_CHECK(dst_indices.is_cuda(), "Destination indices must be a CUDA tensor");
  TORCH_CHECK(src_indices.scalar_type() == at::kLong, "Source indices must be of type long");
  TORCH_CHECK(dst_indices.scalar_type() == at::kLong, "Destination indices must be of type long");
  TORCH_CHECK(src_indices.numel() == dst_indices.numel(), "Source and destination indices must have the same length");

  if (!IsMLA) {
    TORCH_CHECK(src_v.scalar_type() == dst_v.scalar_type(), "Source and destination values must have the same type");
  }

  int dtype_size = src_k.element_size();
  TORCH_CHECK((item_size * dtype_size) % 8 == 0, "Item byte size must be divisible by 8");

  auto div_up = [](int32_t x, int32_t y) { return (x + y - 1) / y; };
  const int64_t num_items = src_indices.numel();
  const int64_t items_per_warp = div_up(num_items, block_quota * num_warps_per_block);
  const int32_t num_blocks = div_up(num_items, items_per_warp * num_warps_per_block);
  dim3 grid_dim(num_blocks, 1, 1);
  const int32_t threads_per_block = num_warps_per_block * 32;

  cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
  transfer_kernel_impl<SrcOffsetFn, DstOffsetFn, IsMLA><<<grid_dim, threads_per_block, 0, torch_current_stream>>>(
      src_k.data_ptr(),
      dst_k.data_ptr(),
      (IsMLA ? nullptr : src_v.data_ptr()),
      (IsMLA ? nullptr : dst_v.data_ptr()),
      src_indices.data_ptr<int64_t>(),
      dst_indices.data_ptr<int64_t>(),
      start_layer_id,
      num_layers_to_process,
      num_items,
      items_per_warp,
      item_size * dtype_size,
      src_layout_dim * dtype_size,
      dst_layout_dim * dtype_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void transfer_kv_per_layer(
    const at::Tensor src_k,
    at::Tensor dst_k,
    const at::Tensor src_v,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  transfer_kv_launcher<get_global_offset_lf, get_global_offset_lf, false>(
      src_k, dst_k, src_v, dst_v, src_indices, dst_indices, 0, 1, item_size, 0, 0, block_quota, num_warps_per_block);
}

void transfer_kv_all_layer(
    const at::Tensor src_k,
    at::Tensor dst_k,
    const at::Tensor src_v,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t num_layers,
    int64_t src_layer_offset,
    int64_t dst_layer_offset,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  transfer_kv_launcher<get_global_offset_lf, get_global_offset_lf, false>(
      src_k,
      dst_k,
      src_v,
      dst_v,
      src_indices,
      dst_indices,
      0,
      num_layers,
      item_size,
      src_layer_offset,
      dst_layer_offset,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_per_layer_mla(
    const at::Tensor src,
    at::Tensor dst,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  at::Tensor empty_tensor = at::Tensor();
  transfer_kv_launcher<get_global_offset_lf, get_global_offset_lf, true>(
      src,
      dst,
      empty_tensor,
      empty_tensor,
      src_indices,
      dst_indices,
      0,
      1,
      item_size,
      0,
      0,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_all_layer_mla(
    const at::Tensor src,
    at::Tensor dst,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t num_layers,
    int64_t src_layer_offset,
    int64_t dst_layer_offset,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  at::Tensor empty_tensor = at::Tensor();
  transfer_kv_launcher<get_global_offset_lf, get_global_offset_lf, true>(
      src,
      dst,
      empty_tensor,
      empty_tensor,
      src_indices,
      dst_indices,
      0,
      num_layers,
      item_size,
      src_layer_offset,
      dst_layer_offset,
      block_quota,
      num_warps_per_block);
}

inline void transfer_page_direct(
    const at::Tensor src_buffer,
    at::Tensor dst_buffer,
    int64_t src_page_index,
    int64_t dst_page_index,
    int64_t page_size) {
  dst_buffer.slice(0, dst_page_index, dst_page_index + page_size)
      .copy_(
          src_buffer.slice(0, src_page_index, src_page_index + page_size),
          /* non_blocking= */ true);
}

template <bool IsMLA, bool AllLayers>
inline void transfer_kv_direct_impl(
    const at::Tensor& src_k,
    at::Tensor& dst_k,
    const at::Tensor& src_v_opt,  // Only used when IsMLA is false (for src_v)
    at::Tensor& dst_v_opt,        // Only used when IsMLA is false (for dst_v)
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t page_size,
    int64_t num_layers = 1) {
  TORCH_CHECK(src_indices.numel() == dst_indices.numel(), "Source and destination indices must have the same length");
  TORCH_CHECK(page_size > 0, "Page size must be positive");
  TORCH_CHECK(src_indices.numel() % page_size == 0, "Source indices size must be divisible by page size");

  auto src_indices_cpu = src_indices.cpu();
  auto dst_indices_cpu = dst_indices.cpu();

  const int64_t num_pages = src_indices_cpu.size(0) / page_size;

  for (const auto i : c10::irange(num_pages)) {
    auto s_index = src_indices_cpu[i * page_size].item<int64_t>();
    auto d_index = dst_indices_cpu[i * page_size].item<int64_t>();

    if constexpr (AllLayers) {
      for (const auto j : c10::irange(num_layers)) {
        if constexpr (IsMLA) {
          transfer_page_direct(src_k.select(0, j), dst_k.select(0, j), s_index, d_index, page_size);
        } else {
          transfer_page_direct(src_k.select(0, j), dst_k.select(0, j), s_index, d_index, page_size);
          transfer_page_direct(src_v_opt.select(0, j), dst_v_opt.select(0, j), s_index, d_index, page_size);
        }
      }
    } else {  // Per-layer
      if constexpr (IsMLA) {
        transfer_page_direct(src_k, dst_k, s_index, d_index, page_size);
      } else {
        transfer_page_direct(src_k, dst_k, s_index, d_index, page_size);
        transfer_page_direct(src_v_opt, dst_v_opt, s_index, d_index, page_size);
      }
    }
  }
}

void transfer_kv_per_layer_direct(
    const at::Tensor src_k,
    at::Tensor dst_k,
    const at::Tensor src_v,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t page_size) {
  transfer_kv_direct_impl<false, false>(src_k, dst_k, src_v, dst_v, src_indices, dst_indices, page_size);
}

void transfer_kv_all_layer_direct(
    const at::Tensor src_k,
    at::Tensor dst_k,
    const at::Tensor src_v,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t page_size,
    int64_t num_layers) {
  transfer_kv_direct_impl<false, true>(src_k, dst_k, src_v, dst_v, src_indices, dst_indices, page_size, num_layers);
}

void transfer_kv_per_layer_mla_direct(
    const at::Tensor src,
    at::Tensor dst,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t page_size) {
  at::Tensor empty_tensor = at::Tensor();

  transfer_kv_direct_impl<true, false>(src, dst, empty_tensor, empty_tensor, src_indices, dst_indices, page_size);
}

void transfer_kv_all_layer_mla_direct(
    const at::Tensor src,
    at::Tensor dst,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t page_size,
    int64_t num_layers) {
  at::Tensor empty_tensor = at::Tensor();
  transfer_kv_direct_impl<true, true>(
      src, dst, empty_tensor, empty_tensor, src_indices, dst_indices, page_size, num_layers);
}
