#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/irange.h>

#include <cstdint>

#include "pytorch_extension_utils.h"

template <typename T>
__global__ void transfer_kv_kernel(
    const T* __restrict__ src_k,
    T* __restrict__ dst_k,
    const T* __restrict__ src_v,
    T* __restrict__ dst_v,
    const int64_t* __restrict__ src_indices,
    const int64_t* __restrict__ dst_indices,
    int64_t num_items,
    int64_t item_size,
    int64_t src_layer_offset,
    int64_t dst_layer_offset,
    int64_t items_per_warp) {
  // Thread identifiers
  const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t lane = tid % 32;
  const int32_t warp_id = tid / 32;

  const int32_t layer_id = blockIdx.y;
  const int64_t src_global_offset = src_layer_offset * layer_id;
  const int64_t dst_global_offset = dst_layer_offset * layer_id;

  for (int i = 0; i < items_per_warp; ++i) {
    const int32_t item_id = warp_id * items_per_warp + i;
    if (item_id >= num_items) {
      return;
    }

    // Compute the per-item offsets
    const int64_t src_offset = src_indices[item_id] * item_size;
    const int64_t dst_offset = dst_indices[item_id] * item_size;

    // pad each chunk per thread to be 8 bytes
    const int total_chunks = item_size * sizeof(T) / 8;

    const int64_t* src_k_4 = reinterpret_cast<const int64_t*>(src_k + src_global_offset + src_offset);
    int64_t* dst_k_4 = reinterpret_cast<int64_t*>(dst_k + dst_global_offset + dst_offset);

#pragma unroll
    for (int j = lane; j < total_chunks; j += 32) {
      dst_k_4[j] = src_k_4[j];
    }

    const int64_t* src_v_4 = reinterpret_cast<const int64_t*>(src_v + src_global_offset + src_offset);
    int64_t* dst_v_4 = reinterpret_cast<int64_t*>(dst_v + dst_global_offset + dst_offset);

#pragma unroll
    for (int j = lane; j < total_chunks; j += 32) {
      dst_v_4[j] = src_v_4[j];
    }
  }
}
template <typename T>
void transfer_kv_launcher_T(
    const at::Tensor& src_k,
    at::Tensor& dst_k,
    const at::Tensor& src_v,
    at::Tensor& dst_v,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t item_size,
    int64_t num_layers,
    int64_t src_layer_offset,
    int64_t dst_layer_offset,
    int64_t block_quota) {
  if (src_indices.size(0) != dst_indices.size(0)) {
    throw std::invalid_argument("Source and destination indices must have the same length");
  }
  const int64_t num_items = src_indices.size(0);

  const int32_t num_warps_per_block = 32;
  const int32_t threads_per_block = num_warps_per_block * 32;

  // Could be adjusted based on the GPU architecture
  // int32_t BLOCK_QUOTA = 32;
  auto div_up = [](int32_t x, int32_t y) { return (x + y - 1) / y; };
  block_quota = div_up(block_quota, num_layers);
  const int64_t items_per_warp = div_up(num_items, block_quota * num_warps_per_block);
  const int32_t num_blocks = div_up(num_items, items_per_warp * num_warps_per_block);

  cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
  dim3 grid_dim(num_blocks, num_layers, 1);

  // Shared kernel from your deduped version:
  transfer_kv_kernel<T><<<grid_dim, threads_per_block, 0, torch_current_stream>>>(
      src_k.data_ptr<T>(),
      dst_k.data_ptr<T>(),
      src_v.data_ptr<T>(),
      dst_v.data_ptr<T>(),
      src_indices.data_ptr<int64_t>(),
      dst_indices.data_ptr<int64_t>(),
      num_items,
      item_size,
      src_layer_offset,
      dst_layer_offset,
      items_per_warp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void transfer_kv_launcher(
    const at::Tensor& src_k,
    at::Tensor& dst_k,
    const at::Tensor& src_v,
    at::Tensor& dst_v,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t item_size,
    int64_t num_layers,
    int64_t src_layer_offset,
    int64_t dst_layer_offset,
    int64_t block_quota) {
  // Check that all src/dst tensor types match
  if (src_k.scalar_type() != dst_k.scalar_type() || src_k.scalar_type() != src_v.scalar_type() ||
      src_k.scalar_type() != dst_v.scalar_type()) {
    throw std::invalid_argument("All input tensors must have the same type");
  }

  at::ScalarType dtype = src_k.scalar_type();
  switch (dtype) {
    case at::ScalarType::Half:
      transfer_kv_launcher_T<at::Half>(
          src_k,
          dst_k,
          src_v,
          dst_v,
          src_indices,
          dst_indices,
          item_size,
          num_layers,
          src_layer_offset,
          dst_layer_offset,
          block_quota);
      break;
    case at::ScalarType::BFloat16:
      transfer_kv_launcher_T<at::BFloat16>(
          src_k,
          dst_k,
          src_v,
          dst_v,
          src_indices,
          dst_indices,
          item_size,
          num_layers,
          src_layer_offset,
          dst_layer_offset,
          block_quota);
      break;
    case at::ScalarType::Float:
      transfer_kv_launcher_T<float>(
          src_k,
          dst_k,
          src_v,
          dst_v,
          src_indices,
          dst_indices,
          item_size,
          num_layers,
          src_layer_offset,
          dst_layer_offset,
          block_quota);
      break;
    case at::ScalarType::Double:
      transfer_kv_launcher_T<double>(
          src_k,
          dst_k,
          src_v,
          dst_v,
          src_indices,
          dst_indices,
          item_size,
          num_layers,
          src_layer_offset,
          dst_layer_offset,
          block_quota);
      break;
    default:
      throw std::invalid_argument("Unsupported scalar type");
  }
}

void transfer_kv_per_layer(
    const at::Tensor src_k,
    at::Tensor dst_k,
    const at::Tensor src_v,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t block_quota) {
  transfer_kv_launcher(src_k, dst_k, src_v, dst_v, src_indices, dst_indices, item_size, 1, 0, 0, block_quota);
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
    int64_t block_quota) {
  transfer_kv_launcher(
      src_k,
      dst_k,
      src_v,
      dst_v,
      src_indices,
      dst_indices,
      item_size,
      num_layers,
      src_layer_offset,
      dst_layer_offset,
      block_quota);
}

void transfer_kv_to_cpu_all_layer_naive(
    at::Tensor host_indices,
    at::Tensor host_k_buffer,
    at::Tensor host_v_buffer,
    at::Tensor device_indices,
    at::Tensor device_k_buffer,
    at::Tensor device_v_buffer,
    int64_t page_size,
    int64_t layer_num) {
  if (device_indices.size(0) != host_indices.size(0)) {
    throw std::invalid_argument("Source and destination indices must have the same length");
  }
  if (device_indices.size(0) % page_size != 0) {
    throw std::invalid_argument("Source indice size must be divisible by page size");
  }
  device_indices = device_indices.cpu();
  for (const auto i : c10::irange(device_indices.size(0) / page_size)) {
    auto h_index = host_indices[i * page_size].item<int64_t>();
    auto d_index = device_indices[i * page_size].item<int64_t>();
    for (const auto j : c10::irange(layer_num)) {
      host_k_buffer[j]
          .slice(0, h_index, h_index + page_size)
          .copy_(
              device_k_buffer[j].slice(0, d_index, d_index + page_size),
              /* non_blocking= */ true);
      host_v_buffer[j]
          .slice(0, h_index, h_index + page_size)
          .copy_(
              device_v_buffer[j].slice(0, d_index, d_index + page_size),
              /* non_blocking= */ true);
    }
  }
}

void transfer_kv_to_gpu_per_layer_naive(
    at::Tensor host_indices,
    at::Tensor host_k_buffer,
    at::Tensor host_v_buffer,
    at::Tensor device_indices,
    at::Tensor device_k_buffer,
    at::Tensor device_v_buffer,
    int64_t page_size,
    int64_t layer_id) {
  if (device_indices.size(0) != host_indices.size(0)) {
    throw std::invalid_argument("Source and destination indices must have the same length");
  }
  if (device_indices.size(0) % page_size != 0) {
    throw std::invalid_argument("Source indice size must be divisible by page size");
  }
  device_indices = device_indices.cpu();
  for (const auto i : c10::irange(device_indices.size(0) / page_size)) {
    auto h_index = host_indices[i * page_size].item<int64_t>();
    auto d_index = device_indices[i * page_size].item<int64_t>();
    device_k_buffer[layer_id]
        .slice(0, d_index, d_index + page_size)
        .copy_(
            host_k_buffer[layer_id].slice(0, h_index, h_index + page_size),
            /* non_blocking= */ true);
    device_v_buffer[layer_id]
        .slice(0, d_index, d_index + page_size)
        .copy_(
            host_v_buffer[layer_id].slice(0, h_index, h_index + page_size),
            /* non_blocking= */ true);
  }
}
