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

template <typename T>
__device__ __forceinline__ T* get_global_offset_lf(
    T* base,
    const uintptr_t* __restrict__ /*unused*/,
    int64_t layer_id,
    int64_t layer_dim,
    int64_t page_id,
    int64_t item_size_bytes) {
  // layer first
  return base + layer_id * layer_dim + page_id * item_size_bytes;
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_pf(
    T* base,
    const uintptr_t* __restrict__ /*unused*/,
    int64_t layer_id,
    int64_t page_dim,
    int64_t page_id,
    int64_t item_size_bytes) {
  // page first
  return base + page_id * page_dim + layer_id * item_size_bytes;
}

// get offset from layer base table when layers are not contiguous
template <typename T>
__device__ __forceinline__ T* get_global_offset_lf_tbl(
    T* /*unused*/,
    const uintptr_t* __restrict__ layer_base_tbl,
    int64_t layer_id,
    int64_t /*unused*/,
    int64_t page_id,
    int64_t item_size_bytes) {
  return reinterpret_cast<T*>(layer_base_tbl[layer_id]) + page_id * item_size_bytes;
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
    int64_t dst_layout_dim,
    const uintptr_t* __restrict__ src_k_layer_tbl,
    const uintptr_t* __restrict__ dst_k_layer_tbl,
    const uintptr_t* __restrict__ src_v_layer_tbl,
    const uintptr_t* __restrict__ dst_v_layer_tbl) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t lane_id = tid % 32;
  int32_t warp_id = tid / 32;

  for (int i = 0; i < items_per_warp; ++i) {
    int64_t item_id = warp_id * items_per_warp + i;
    if (item_id >= num_items) {
      break;
    }
    const int64_t src_page_id = src_indices[item_id];
    const int64_t dst_page_id = dst_indices[item_id];

    // Loop over layers if necessary
    for (int64_t layer_id = start_layer_id; layer_id < start_layer_id + num_layers_to_process; ++layer_id) {
      const char* src_ptr = SrcOffsetFn(
          static_cast<const char*>(src_k), src_k_layer_tbl, layer_id, src_layout_dim, src_page_id, item_size_bytes);
      char* dst_ptr = DstOffsetFn(
          static_cast<char*>(dst_k), dst_k_layer_tbl, layer_id, dst_layout_dim, dst_page_id, item_size_bytes);
      transfer_item_warp(lane_id, src_ptr, dst_ptr, item_size_bytes);

      if constexpr (!IsMLA) {
        const char* src_v_ptr = SrcOffsetFn(
            static_cast<const char*>(src_v), src_v_layer_tbl, layer_id, src_layout_dim, src_page_id, item_size_bytes);
        char* dst_v_ptr = DstOffsetFn(
            static_cast<char*>(dst_v), dst_v_layer_tbl, layer_id, dst_layout_dim, dst_page_id, item_size_bytes);
        transfer_item_warp(lane_id, src_v_ptr, dst_v_ptr, item_size_bytes);
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
    const at::Tensor& src_k_layers,
    const at::Tensor& dst_k_layers,
    const at::Tensor& src_v_layers,
    const at::Tensor& dst_v_layers,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  TORCH_CHECK(src_indices.is_cuda(), "Source indices must be a CUDA tensor");
  TORCH_CHECK(dst_indices.is_cuda(), "Destination indices must be a CUDA tensor");
  TORCH_CHECK(src_indices.scalar_type() == at::kLong, "Source indices must be of type long");
  TORCH_CHECK(dst_indices.scalar_type() == at::kLong, "Destination indices must be of type long");
  TORCH_CHECK(src_indices.numel() == dst_indices.numel(), "Source and destination indices must have the same length");
  TORCH_CHECK(item_size % 8 == 0, "Item byte size must be divisible by 8");

  auto div_up = [](int64_t x, int64_t y) { return (x + y - 1) / y; };
  const int64_t num_items = src_indices.numel();
  const int64_t items_per_warp = div_up(num_items, block_quota * num_warps_per_block);
  const int32_t num_blocks = div_up(num_items, items_per_warp * num_warps_per_block);
  dim3 grid_dim(num_blocks, 1, 1);
  const int32_t threads_per_block = num_warps_per_block * 32;

  const void* src_k_ptr = src_k.defined() ? src_k.data_ptr() : nullptr;
  void* dst_k_ptr = dst_k.defined() ? dst_k.data_ptr() : nullptr;
  const void* src_v_ptr = IsMLA || !src_v.defined() ? nullptr : src_v.data_ptr();
  void* dst_v_ptr = IsMLA || !dst_v.defined() ? nullptr : dst_v.data_ptr();
  const uintptr_t* src_k_tbl_ptr = src_k_layers.defined() ? src_k_layers.data_ptr<uintptr_t>() : nullptr;
  const uintptr_t* dst_k_tbl_ptr = dst_k_layers.defined() ? dst_k_layers.data_ptr<uintptr_t>() : nullptr;
  const uintptr_t* src_v_tbl_ptr = IsMLA || !src_v_layers.defined() ? nullptr : src_v_layers.data_ptr<uintptr_t>();
  const uintptr_t* dst_v_tbl_ptr = IsMLA || !dst_v_layers.defined() ? nullptr : dst_v_layers.data_ptr<uintptr_t>();

  cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
  transfer_kernel_impl<SrcOffsetFn, DstOffsetFn, IsMLA><<<grid_dim, threads_per_block, 0, torch_current_stream>>>(
      src_k_ptr,
      dst_k_ptr,
      src_v_ptr,
      dst_v_ptr,
      src_indices.data_ptr<int64_t>(),
      dst_indices.data_ptr<int64_t>(),
      start_layer_id,
      num_layers_to_process,
      num_items,
      items_per_warp,
      item_size,
      src_layout_dim,
      dst_layout_dim,
      src_k_tbl_ptr,
      dst_k_tbl_ptr,
      src_v_tbl_ptr,
      dst_v_tbl_ptr);
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
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_lf<const char>, get_global_offset_lf<char>, false>(
      src_k,
      dst_k,
      src_v,
      dst_v,
      src_indices,
      dst_indices,
      0,
      1,
      item_size,
      0,
      0,
      empty,
      empty,
      empty,
      empty,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_per_layer_pf_lf(
    const at::Tensor src_k,
    at::Tensor dst_k,
    const at::Tensor src_v,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_pf<const char>, get_global_offset_lf<char>, false>(
      src_k,
      dst_k,
      src_v,
      dst_v,
      src_indices,
      dst_indices,
      0,
      1,
      item_size,
      src_layout_dim,
      0,
      empty,
      empty,
      empty,
      empty,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_all_layer(
    const at::Tensor src_k_layers,
    const at::Tensor dst_k_layers,
    const at::Tensor src_v_layers,
    const at::Tensor dst_v_layers,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  TORCH_CHECK(num_layers == src_k_layers.size(0), "Number of layers in source k tensor does not match num_layers");
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_lf_tbl<const char>, get_global_offset_lf_tbl<char>, false>(
      empty,
      empty,
      empty,
      empty,
      src_indices,
      dst_indices,
      0,
      num_layers,
      item_size,
      0,
      0,
      src_k_layers,
      dst_k_layers,
      src_v_layers,
      dst_v_layers,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_all_layer_lf_pf(
    const at::Tensor src_k_layers,
    at::Tensor dst_k,
    const at::Tensor src_v_layers,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  TORCH_CHECK(num_layers == src_k_layers.size(0), "Number of layers in source k tensor does not match num_layers");
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_lf_tbl<const char>, get_global_offset_pf<char>, false>(
      empty,
      dst_k,
      empty,
      dst_v,
      src_indices,
      dst_indices,
      0,
      num_layers,
      item_size,
      0,
      dst_layout_dim,
      src_k_layers,
      empty,
      src_v_layers,
      empty,
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
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_lf<const char>, get_global_offset_lf<char>, true>(
      src,
      dst,
      empty,
      empty,
      src_indices,
      dst_indices,
      0,
      1,
      item_size,
      0,
      0,
      empty,
      empty,
      empty,
      empty,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_per_layer_mla_pf_lf(
    const at::Tensor src,
    at::Tensor dst,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_pf<const char>, get_global_offset_lf<char>, true>(
      src,
      dst,
      empty,
      empty,
      src_indices,
      dst_indices,
      0,
      1,
      item_size,
      src_layout_dim,
      0,
      empty,
      empty,
      empty,
      empty,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_all_layer_mla(
    const at::Tensor src_layers,
    const at::Tensor dst_layers,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  TORCH_CHECK(num_layers == src_layers.size(0), "Number of layers in source tensor does not match num_layers");
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_lf_tbl<const char>, get_global_offset_lf_tbl<char>, true>(
      empty,
      empty,
      empty,
      empty,
      src_indices,
      dst_indices,
      0,
      num_layers,
      item_size,
      0,
      0,
      src_layers,
      dst_layers,
      empty,
      empty,
      block_quota,
      num_warps_per_block);
}

void transfer_kv_all_layer_mla_lf_pf(
    const at::Tensor src_layers,
    at::Tensor dst,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  TORCH_CHECK(num_layers == src_layers.size(0), "Number of layers in source tensor does not match num_layers");
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_lf_tbl<const char>, get_global_offset_pf<char>, true>(
      empty,
      dst,
      empty,
      empty,
      src_indices,
      dst_indices,
      0,
      num_layers,
      item_size,
      0,
      dst_layout_dim,
      src_layers,
      empty,
      empty,
      empty,
      block_quota,
      num_warps_per_block);
}

inline void transfer_page_direct(
    const at::Tensor& src_buffer,
    at::Tensor& dst_buffer,
    int64_t src_page_index,
    int64_t dst_page_index,
    int64_t page_size) {
  dst_buffer.slice(0, dst_page_index, dst_page_index + page_size)
      .copy_(
          src_buffer.slice(0, src_page_index, src_page_index + page_size),
          /* non_blocking= */ true);
}

void transfer_kv_direct(
    const std::vector<at::Tensor>& src_layers,
    std::vector<at::Tensor> dst_layers,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t page_size) {
  TORCH_CHECK(
      src_layers.size() == dst_layers.size(), "Source and destination layers must have the same number of layers");
  TORCH_CHECK(src_indices.numel() == dst_indices.numel(), "Source and destination indices must have the same length");
  TORCH_CHECK(page_size > 0, "Page size must be positive");
  TORCH_CHECK(src_indices.numel() % page_size == 0, "Source indices size must be divisible by page size");

  auto src_indices_cpu = src_indices.cpu();
  auto dst_indices_cpu = dst_indices.cpu();

  const int64_t num_pages = src_indices_cpu.size(0) / page_size;
  const int64_t num_layers = src_layers.size();

  for (int64_t i = 0; i < num_pages; ++i) {
    auto src_index = src_indices_cpu[i * page_size].item<int64_t>();
    auto dst_index = dst_indices_cpu[i * page_size].item<int64_t>();

    for (int64_t j = 0; j < num_layers; ++j) {
      transfer_page_direct(src_layers[j], dst_layers[j], src_index, dst_index, page_size);
    }
  }
}
