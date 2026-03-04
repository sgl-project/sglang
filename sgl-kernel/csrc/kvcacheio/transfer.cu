#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/irange.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <vector>

#ifndef USE_ROCM
#include <dlfcn.h>
#define WARP_SIZE 32
#include "pytorch_extension_utils.h"
#else
#include "pytorch_extension_utils_rocm.h"
#include "utils.h"  // WARP_SIZE
#endif

__device__ __forceinline__ void
transfer_item_warp(int32_t lane_id, const void* src_addr, void* dst_addr, int64_t item_size_bytes) {
  const uint64_t* __restrict__ src = static_cast<const uint64_t*>(src_addr);
  uint64_t* __restrict__ dst = static_cast<uint64_t*>(dst_addr);
  const int total_chunks = item_size_bytes / sizeof(uint64_t);

#pragma unroll
  for (int j = lane_id; j < total_chunks; j += WARP_SIZE) {
#ifndef USE_ROCM
    uint64_t tmp;
    asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src + j) : "memory");
    asm volatile("st.global.cg.b64 [%0],%1;" ::"l"(dst + j), "l"(tmp) : "memory");

#else
    uint64_t tmp = __builtin_nontemporal_load(src + j);
    __builtin_nontemporal_store(tmp, dst + j);
#endif
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

template <typename T>
__device__ __forceinline__ T* get_global_offset_per_head_lf(
    T* base,
    const uintptr_t* __restrict__ /*unused*/,
    int64_t layer_id,
    int64_t layer_dim,
    int64_t page_id,
    int64_t item_size_bytes,
    int64_t head_id,
    int64_t head_num,
    int64_t /*unused*/) {
  // layer first offset func per head
  return base + layer_id * layer_dim + page_id * item_size_bytes + item_size_bytes / head_num * head_id;
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_per_head_lf_tbl(
    T* /*unused*/,
    const uintptr_t* __restrict__ layer_base_tbl,
    int64_t layer_id,
    int64_t /*unused*/,
    int64_t page_id,
    int64_t item_size_bytes,
    int64_t head_id,
    int64_t head_num,
    int64_t /*unused*/) {
  return reinterpret_cast<T*>(layer_base_tbl[layer_id]) + page_id * item_size_bytes +
         item_size_bytes / head_num * head_id;
}

template <typename T>
__device__ __forceinline__ T* get_global_offset_ph(
    T* base,
    const uintptr_t* __restrict__ /*unused*/,
    int64_t layer_id,
    int64_t page_dim,
    int64_t page_id,
    int64_t item_size_bytes,
    int64_t head_id,
    int64_t head_num,
    int64_t page_size) {
  // page head layout: [page_num, head_num, page_size, layer_num, head_dim]
  return base + page_id / page_size * page_size * page_dim +  // page_num dimension offset
         page_dim / head_num * head_id * page_size +          // head_num dimension offset
         page_id % page_size * page_dim / head_num +          // page_size dimension offset
         layer_id * item_size_bytes / head_num;               // layer_num dimension offset
}

template <auto SrcOffsetFn, auto DstOffsetFn>
__global__ void transfer_page_head_kernel_impl(
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
    const uintptr_t* __restrict__ dst_v_layer_tbl,
    const int64_t page_size,
    const int64_t head_num) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t lane_id = tid % WARP_SIZE;
  int32_t warp_id = tid / WARP_SIZE;
  const int64_t head_size_bytes = item_size_bytes / head_num;

  for (int i = 0; i < items_per_warp; ++i) {
    int64_t item_id = warp_id * items_per_warp + i;
    if (item_id >= num_items) {
      break;
    }
    const int64_t src_page_id = src_indices[item_id];
    const int64_t dst_page_id = dst_indices[item_id];

    // Loop over layers if necessary
    for (int64_t layer_id = start_layer_id; layer_id < start_layer_id + num_layers_to_process; ++layer_id) {
      // For page head layout, the cache of each head in the token is discontinuous, need to loop
      for (int64_t head_id = 0; head_id < head_num; ++head_id) {
        const char* src_k_ptr = SrcOffsetFn(
            static_cast<const char*>(src_k),
            src_k_layer_tbl,
            layer_id,
            src_layout_dim,
            src_page_id,
            item_size_bytes,
            head_id,
            head_num,
            page_size);
        char* dst_k_ptr = DstOffsetFn(
            static_cast<char*>(dst_k),
            dst_k_layer_tbl,
            layer_id,
            dst_layout_dim,
            dst_page_id,
            item_size_bytes,
            head_id,
            head_num,
            page_size);
        transfer_item_warp(lane_id, src_k_ptr, dst_k_ptr, head_size_bytes);

        const char* src_v_ptr = SrcOffsetFn(
            static_cast<const char*>(src_v),
            src_v_layer_tbl,
            layer_id,
            src_layout_dim,
            src_page_id,
            item_size_bytes,
            head_id,
            head_num,
            page_size);
        char* dst_v_ptr = DstOffsetFn(
            static_cast<char*>(dst_v),
            dst_v_layer_tbl,
            layer_id,
            dst_layout_dim,
            dst_page_id,
            item_size_bytes,
            head_id,
            head_num,
            page_size);
        transfer_item_warp(lane_id, src_v_ptr, dst_v_ptr, head_size_bytes);
      }
    }
  }
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
  int32_t lane_id = tid % WARP_SIZE;
  int32_t warp_id = tid / WARP_SIZE;

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

template <auto SrcOffsetFn, auto DstOffsetFn, bool IsMLA, bool PageHeadLayout = false>
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
    int64_t num_warps_per_block,
    const int64_t page_size = 16,
    const int64_t head_num = 1) {
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
  const int32_t threads_per_block = num_warps_per_block * WARP_SIZE;

  const void* src_k_ptr = src_k.defined() ? src_k.data_ptr() : nullptr;
  void* dst_k_ptr = dst_k.defined() ? dst_k.data_ptr() : nullptr;
  const void* src_v_ptr = IsMLA || !src_v.defined() ? nullptr : src_v.data_ptr();
  void* dst_v_ptr = IsMLA || !dst_v.defined() ? nullptr : dst_v.data_ptr();
  const uintptr_t* src_k_tbl_ptr = src_k_layers.defined() ? src_k_layers.data_ptr<uintptr_t>() : nullptr;
  const uintptr_t* dst_k_tbl_ptr = dst_k_layers.defined() ? dst_k_layers.data_ptr<uintptr_t>() : nullptr;
  const uintptr_t* src_v_tbl_ptr = IsMLA || !src_v_layers.defined() ? nullptr : src_v_layers.data_ptr<uintptr_t>();
  const uintptr_t* dst_v_tbl_ptr = IsMLA || !dst_v_layers.defined() ? nullptr : dst_v_layers.data_ptr<uintptr_t>();

  cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
  if constexpr (PageHeadLayout) {
    transfer_page_head_kernel_impl<SrcOffsetFn, DstOffsetFn><<<grid_dim, threads_per_block, 0, torch_current_stream>>>(
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
        dst_v_tbl_ptr,
        page_size,
        head_num);
  } else {
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
  }
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
    int64_t layer_id,
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
      layer_id,
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

void transfer_kv_per_layer_ph_lf(
    const at::Tensor src_k,
    at::Tensor dst_k,
    const at::Tensor src_v,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t layer_id,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t page_size,
    int64_t head_num,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_ph<const char>, get_global_offset_per_head_lf<char>, false, true>(
      src_k,
      dst_k,
      src_v,
      dst_v,
      src_indices,
      dst_indices,
      layer_id,
      1,
      item_size,
      src_layout_dim,
      0,
      empty,
      empty,
      empty,
      empty,
      block_quota,
      num_warps_per_block,
      page_size,
      head_num);
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

void transfer_kv_all_layer_lf_ph(
    const at::Tensor src_k_layers,
    at::Tensor dst_k,
    const at::Tensor src_v_layers,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers,
    int64_t page_size,
    int64_t head_num,
    int64_t block_quota,
    int64_t num_warps_per_block) {
  TORCH_CHECK(num_layers == src_k_layers.size(0), "Number of layers in source k tensor does not match num_layers");
  at::Tensor empty;
  transfer_kv_launcher<get_global_offset_per_head_lf_tbl<const char>, get_global_offset_ph<char>, false, true>(
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
      num_warps_per_block,
      page_size,
      head_num);
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
    int64_t layer_id,
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
      layer_id,
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

  const auto num_indices = src_indices_cpu.numel();
  const int64_t num_layers = src_layers.size();
  int64_t* src_indices_ptr = src_indices_cpu.data_ptr<int64_t>();
  int64_t* dst_indices_ptr = dst_indices_cpu.data_ptr<int64_t>();

  int64_t start_index = 0;
  int64_t end_index = 0;

  for (int64_t i = 0; i < num_indices; ++i) {
    if (i < num_indices - 1) {
      auto src_diff = src_indices_ptr[i + 1] - src_indices_ptr[i];
      auto dst_diff = dst_indices_ptr[i + 1] - dst_indices_ptr[i];

      if (src_diff == 1 && dst_diff == 1) {
        continue;
      }
      end_index = i + 1;
    } else {  // last batch
      end_index = num_indices;
    }
    auto src_index = src_indices_ptr[start_index];
    auto dst_index = dst_indices_ptr[start_index];
    auto num_tokens = end_index - start_index;

    for (int64_t j = 0; j < num_layers; ++j) {
      transfer_page_direct(src_layers[j], dst_layers[j], src_index, dst_index, num_tokens);
    }
    start_index = end_index;
  }
}

template <bool IsLf2Pf>
inline void transfer_kv_page_first_direct_impl(
    const std::vector<at::Tensor>& src_ptrs,
    std::vector<at::Tensor> dst_ptrs,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t start_layer_id,
    int64_t page_size) {
  TORCH_CHECK(src_indices.numel() == dst_indices.numel(), "Source and destination indices must have the same length");
  TORCH_CHECK(page_size > 0, "Page size must be positive");
  TORCH_CHECK(src_indices.numel() % page_size == 0, "Source indices size must be divisible by page size");

  auto src_indices_cpu = src_indices.cpu();
  auto dst_indices_cpu = dst_indices.cpu();
  const int64_t num_pages = src_indices_cpu.size(0) / page_size;
  int64_t* src_indices_ptr = src_indices_cpu.data_ptr<int64_t>();
  int64_t* dst_indices_ptr = dst_indices_cpu.data_ptr<int64_t>();

  auto fallback_to_page_copy = [&]() {
    if constexpr (IsLf2Pf) {
      const bool is_mla = dst_ptrs.size() == 1;
      const int64_t num_layers = is_mla ? src_ptrs.size() : src_ptrs.size() / 2;
      for (const auto i : c10::irange(num_pages)) {
        const int64_t s_index = src_indices_ptr[i * page_size];
        const int64_t d_index = dst_indices_ptr[i * page_size] / page_size;
        for (int64_t j = 0; j < num_layers; ++j) {
          transfer_page_direct(
              src_ptrs[j], dst_ptrs[0].select(0, d_index).select(0, start_layer_id + j), s_index, 0, page_size);
          if (!is_mla) {
            transfer_page_direct(
                src_ptrs[j + num_layers],
                dst_ptrs[1].select(0, d_index).select(0, start_layer_id + j),
                s_index,
                0,
                page_size);
          }
        }
      }
    } else {
      const bool is_mla = src_ptrs.size() == 1;
      const int64_t num_layers = is_mla ? dst_ptrs.size() : dst_ptrs.size() / 2;
      for (const auto i : c10::irange(num_pages)) {
        const int64_t s_index = src_indices_ptr[i * page_size] / page_size;
        const int64_t d_index = dst_indices_ptr[i * page_size];
        for (int64_t j = 0; j < num_layers; ++j) {
          transfer_page_direct(
              src_ptrs[0].select(0, s_index).select(0, start_layer_id + j), dst_ptrs[j], 0, d_index, page_size);
          if (!is_mla) {
            transfer_page_direct(
                src_ptrs[1].select(0, s_index).select(0, start_layer_id + j),
                dst_ptrs[j + num_layers],
                0,
                d_index,
                page_size);
          }
        }
      }
    }
  };

#if defined(USE_ROCM) || !defined(CUDA_VERSION) || CUDA_VERSION < 12080
  fallback_to_page_copy();
  return;

#else
  // Driver capability gate: only use cudaMemcpyBatchAsync on CUDA 12.8+ drivers.
  int driver_version = 0;
  cudaError_t driver_version_err = cudaDriverGetVersion(&driver_version);
  if (driver_version_err != cudaSuccess || driver_version < 12080) {
    fallback_to_page_copy();
    return;
  }

  // Symbol gate: runtime may not expose cudaMemcpyBatchAsync in some environments.
  using CudaMemcpyBatchAsyncFn =
      cudaError_t (*)(void**, void**, size_t*, size_t, cudaMemcpyAttributes*, size_t*, size_t, size_t*, cudaStream_t);
  static CudaMemcpyBatchAsyncFn cuda_memcpy_batch_async = []() {
    void* symbol = dlsym(RTLD_DEFAULT, "cudaMemcpyBatchAsync");
    return reinterpret_cast<CudaMemcpyBatchAsyncFn>(symbol);
  }();
  if (cuda_memcpy_batch_async == nullptr) {
    fallback_to_page_copy();
    return;
  }

  size_t num_copies = 0;
  std::vector<void*> batch_srcs;
  std::vector<void*> batch_dsts;
  std::vector<size_t> batch_sizes;
  std::vector<size_t> attrs_idxs(1, 0);
  cudaMemcpyAttributes attrs{};
  const int device_id = at::cuda::current_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto append_copy = [&](void* src, void* dst, size_t size_bytes) {
    batch_srcs.push_back(src);
    batch_dsts.push_back(dst);
    batch_sizes.push_back(size_bytes);
  };

  if constexpr (IsLf2Pf) {
    const bool is_mla = dst_ptrs.size() == 1;
    const int64_t num_layers = is_mla ? src_ptrs.size() : src_ptrs.size() / 2;

    const int64_t dst_stride0 = dst_ptrs[0].stride(0);
    const int64_t dst_stride1 = dst_ptrs[0].stride(1);
    const int64_t src_stride0 = src_ptrs[0].stride(0);
    const int64_t elem_size = dst_ptrs[0].element_size();
    const int64_t copy_size_bytes = page_size * src_stride0 * elem_size;
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attrs.srcLocHint.type = cudaMemLocationTypeDevice;
    attrs.srcLocHint.id = device_id;
    attrs.dstLocHint.type = cudaMemLocationTypeHost;
    attrs.dstLocHint.id = 0;
    attrs.flags = 0;

    num_copies = static_cast<size_t>(num_pages) * static_cast<size_t>(num_layers) * static_cast<size_t>(is_mla ? 1 : 2);
    batch_srcs.reserve(num_copies);
    batch_dsts.reserve(num_copies);
    batch_sizes.reserve(num_copies);

    for (const auto i : c10::irange(num_pages)) {
      auto s_index = src_indices_ptr[i * page_size];
      auto d_index = dst_indices_ptr[i * page_size] / page_size;

      for (int64_t j = 0; j < num_layers; ++j) {
        const char* src_k_ptr = static_cast<const char*>(src_ptrs[j].data_ptr()) + s_index * src_stride0 * elem_size;
        char* dst_k_ptr = static_cast<char*>(dst_ptrs[0].data_ptr()) + d_index * dst_stride0 * elem_size +
                          (start_layer_id + j) * dst_stride1 * elem_size;
        append_copy(const_cast<char*>(src_k_ptr), dst_k_ptr, copy_size_bytes);

        if (!is_mla) {
          const char* src_v_ptr =
              static_cast<const char*>(src_ptrs[j + num_layers].data_ptr()) + s_index * src_stride0 * elem_size;
          char* dst_v_ptr = static_cast<char*>(dst_ptrs[1].data_ptr()) + d_index * dst_stride0 * elem_size +
                            (start_layer_id + j) * dst_stride1 * elem_size;
          append_copy(const_cast<char*>(src_v_ptr), dst_v_ptr, copy_size_bytes);
        }
      }
    }

  } else {
    const bool is_mla = src_ptrs.size() == 1;
    const int64_t num_layers = is_mla ? dst_ptrs.size() : dst_ptrs.size() / 2;

    const int64_t src_stride0 = src_ptrs[0].stride(0);
    const int64_t src_stride1 = src_ptrs[0].stride(1);
    const int64_t dst_stride0 = dst_ptrs[0].stride(0);
    const int64_t elem_size = src_ptrs[0].element_size();
    const int64_t copy_size_bytes = page_size * dst_stride0 * elem_size;
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attrs.srcLocHint.type = cudaMemLocationTypeHost;
    attrs.srcLocHint.id = 0;
    attrs.dstLocHint.type = cudaMemLocationTypeDevice;
    attrs.dstLocHint.id = device_id;
    attrs.flags = 0;

    num_copies = static_cast<size_t>(num_pages) * static_cast<size_t>(num_layers) * static_cast<size_t>(is_mla ? 1 : 2);
    batch_srcs.reserve(num_copies);
    batch_dsts.reserve(num_copies);
    batch_sizes.reserve(num_copies);

    for (const auto i : c10::irange(num_pages)) {
      auto s_index = src_indices_ptr[i * page_size] / page_size;
      auto d_index = dst_indices_ptr[i * page_size];

      for (int64_t j = 0; j < num_layers; ++j) {
        const char* src_k_ptr = static_cast<const char*>(src_ptrs[0].data_ptr()) + s_index * src_stride0 * elem_size +
                                (start_layer_id + j) * src_stride1 * elem_size;
        char* dst_k_ptr = static_cast<char*>(dst_ptrs[j].data_ptr()) + d_index * dst_stride0 * elem_size;
        append_copy(const_cast<char*>(src_k_ptr), dst_k_ptr, copy_size_bytes);

        if (!is_mla) {
          const char* src_v_ptr = static_cast<const char*>(src_ptrs[1].data_ptr()) + s_index * src_stride0 * elem_size +
                                  (start_layer_id + j) * src_stride1 * elem_size;
          char* dst_v_ptr = static_cast<char*>(dst_ptrs[j + num_layers].data_ptr()) + d_index * dst_stride0 * elem_size;
          append_copy(const_cast<char*>(src_v_ptr), dst_v_ptr, copy_size_bytes);
        }
      }
    }
  }

  TORCH_CHECK(batch_srcs.size() == num_copies, "Batch memcpy count mismatch");
  if (num_copies > 0) {
    size_t fail_idx = std::numeric_limits<size_t>::max();
    cudaError_t err = cuda_memcpy_batch_async(
        batch_dsts.data(),
        batch_srcs.data(),
        batch_sizes.data(),
        num_copies,
        &attrs,
        attrs_idxs.data(),
        1,
        &fail_idx,
        stream);
    if (err == cudaErrorNotSupported || err == cudaErrorCallRequiresNewerDriver) {
      fallback_to_page_copy();
      return;
    }
    if (err != cudaSuccess) {
      TORCH_CHECK(false, "cudaMemcpyBatchAsync failed. failIdx=", fail_idx, " error=", cudaGetErrorString(err));
    }
  }
#endif
}

void transfer_kv_per_layer_direct_pf_lf(
    const std::vector<at::Tensor>& src_ptrs,
    std::vector<at::Tensor> dst_ptrs,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t layer_id,
    int64_t page_size) {
  transfer_kv_page_first_direct_impl<false>(src_ptrs, dst_ptrs, src_indices, dst_indices, layer_id, page_size);
}

void transfer_kv_all_layer_direct_lf_pf(
    const std::vector<at::Tensor>& src_ptrs,
    std::vector<at::Tensor> dst_ptrs,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t page_size) {
  transfer_kv_page_first_direct_impl<true>(src_ptrs, dst_ptrs, src_indices, dst_indices, 0, page_size);
}
