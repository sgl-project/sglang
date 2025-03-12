#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "pytorch_extension_utils.h"

template <typename T>
__global__ void transfer_kv_kernel(
    const T* __restrict__ src_k,
    T* __restrict__ dst_k,
    const T* __restrict__ src_v,
    T* __restrict__ dst_v,
    const int32_t* __restrict__ src_indices,
    const int32_t* __restrict__ dst_indices,
    int64_t num_items,
    int64_t item_size,
    int64_t layer_offset) {
  // Thread identifiers
  const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t lane = tid % 32;
  const int32_t warp_id = tid / 32;

  const int32_t layer_id = blockIdx.y;
  const int64_t global_offset = layer_offset * layer_id;

  // Each warp processes exactly one "item" (one row of size `item_size`)
  if (warp_id >= num_items) {
    return;
  }

  // Compute the per-item offsets
  const int32_t src_offset = src_indices[warp_id] * item_size;
  const int32_t dst_offset = dst_indices[warp_id] * item_size;

  // pad each chunk per thread to be 8 bytes
  const int total_chunks = item_size * sizeof(T) / 8;

  const int64_t* src_k_4 = reinterpret_cast<const int64_t*>(src_k + global_offset + src_offset);
  int64_t* dst_k_4 = reinterpret_cast<int64_t*>(dst_k + global_offset + dst_offset);

#pragma unroll
  for (int i = lane; i < total_chunks; i += 32) {
    dst_k_4[i] = src_k_4[i];
  }

  const int64_t* src_v_4 = reinterpret_cast<const int64_t*>(src_v + global_offset + src_offset);
  int64_t* dst_v_4 = reinterpret_cast<int64_t*>(dst_v + global_offset + dst_offset);

#pragma unroll
  for (int i = lane; i < total_chunks; i += 32) {
    dst_v_4[i] = src_v_4[i];
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
    int64_t layer_offset) {
  if (src_indices.size(0) != dst_indices.size(0)) {
    throw std::invalid_argument("Source and destination indices must have the same length");
  }
  const int64_t num_items = src_indices.size(0);

  const int32_t num_warps_per_block = 4;
  const int32_t threads_per_block = num_warps_per_block * 32;
  const int32_t blocks_per_grid = (num_items + num_warps_per_block - 1) / num_warps_per_block;

  cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
  dim3 grid_dim(blocks_per_grid, num_layers, 1);

  // Shared kernel from your deduped version:
  transfer_kv_kernel<T><<<grid_dim, threads_per_block, 0, torch_current_stream>>>(
      src_k.data_ptr<T>(),
      dst_k.data_ptr<T>(),
      src_v.data_ptr<T>(),
      dst_v.data_ptr<T>(),
      src_indices.data_ptr<int32_t>(),
      dst_indices.data_ptr<int32_t>(),
      num_items,
      item_size,
      layer_offset);
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
    int64_t layer_offset) {
  // Check that all src/dst tensor types match
  if (src_k.scalar_type() != dst_k.scalar_type() || src_k.scalar_type() != src_v.scalar_type() ||
      src_k.scalar_type() != dst_v.scalar_type()) {
    throw std::invalid_argument("All input tensors must have the same type");
  }

  at::ScalarType dtype = src_k.scalar_type();
  switch (dtype) {
    case at::ScalarType::Half:
      transfer_kv_launcher_T<at::Half>(
          src_k, dst_k, src_v, dst_v, src_indices, dst_indices, item_size, num_layers, layer_offset);
      break;
    case at::ScalarType::BFloat16:
      transfer_kv_launcher_T<at::BFloat16>(
          src_k, dst_k, src_v, dst_v, src_indices, dst_indices, item_size, num_layers, layer_offset);
      break;
    case at::ScalarType::Float:
      transfer_kv_launcher_T<float>(
          src_k, dst_k, src_v, dst_v, src_indices, dst_indices, item_size, num_layers, layer_offset);
      break;
    case at::ScalarType::Double:
      transfer_kv_launcher_T<double>(
          src_k, dst_k, src_v, dst_v, src_indices, dst_indices, item_size, num_layers, layer_offset);
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
    int64_t item_size) {
  transfer_kv_launcher(src_k, dst_k, src_v, dst_v, src_indices, dst_indices, item_size, 1, 0);
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
    int64_t layer_offset) {
  transfer_kv_launcher(src_k, dst_k, src_v, dst_v, src_indices, dst_indices, item_size, num_layers, layer_offset);
}
