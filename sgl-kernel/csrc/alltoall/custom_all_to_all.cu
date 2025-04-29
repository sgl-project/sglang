/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// reference:
// https://github.com/NVIDIA/TensorRT-LLM/blob/release/0.14/cpp/tensorrt_llm/kernels/customAllReduceKernels.cu
/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tuple>

#include "allreduce/custom_all_reduce.cuh"
#include "utils.h"

using namespace sglang;

constexpr int MAX_RANKS_PER_NODE = 8;
constexpr int MAX_ALL_TO_ALL_BLOCKS = kMaxAll2AllBlocks;
constexpr int MAX_ALL_TO_ALL_WARPS = 16;
// Fake pointer type, must match fptr_t type in ops.h.
// We use this type alias to indicate when pointers are passed in as int64_t.
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));
int g_sm_count = 0;

namespace {
inline int divUp(int a, int b) {
  return (a + b - 1) / b;
}

struct All2AllParams {
  int64_t elts_size;
  int64_t rank;
  int64_t ranks_per_node;
  RankData* peer_comm_buffer_ptrs;
  void* local_input_buffer_ptr;
  int64_t* plan_meta_ptr;

  int64_t output_elts_total;
  int64_t input_elts_total;

  int64_t input_stride0;
  int64_t input_stride1;
  int64_t input_dim1;

  RankSignals sg;
  Signal* self_sg;
};

struct All2AllPlanParams {
  int64_t elts_size;
  int64_t rank;
  int64_t ranks_per_node;
  int64_t* output_split_sizes;
  int64_t* input_split_sizes;
  int64_t* output_split_offsets;
  int64_t* input_split_offsets;
  int64_t* plan_meta_ptr;

  int64_t output_elts_total;
  int64_t input_elts_total;

  int64_t chunk_size;
  int64_t output_stride0;
  int64_t output_stride1;
  int64_t output_dim1;

  int64_t blocks_per_grid;
  int64_t threads_per_block;

  RankSignals sg;
  Signal* self_sg;
};

struct __align__(16) AllToAllCommMeta {
  int32_t output_elts_offset[MAX_RANKS_PER_NODE];
  int32_t output_elts_length[MAX_RANKS_PER_NODE];
  int32_t output_dim1;
  int32_t output_stride0;
  int32_t output_stride1;
};

struct __align__(16) AllToAllPeerMeta {
  int32_t output_elts_offset;
  int32_t output_dim1;
  int32_t output_stride0;
  int32_t output_stride1;
};

struct __align__(16) AllToAllPlanMeta {
  int32_t local_input_elts_offset[MAX_RANKS_PER_NODE];
  int32_t local_input_elts_length[MAX_RANKS_PER_NODE];

  AllToAllPeerMeta peer_meta[MAX_RANKS_PER_NODE];
  int32_t chunk_size;
  int32_t input_split_elts_total;
  int32_t total_opt_warps_count;
  int32_t warps_beg[MAX_RANKS_PER_NODE];
  int8_t warp_peer_gpu[8];  // real size total_opt_warps_count
};

inline int getSmCount() {
  if (g_sm_count <= 0) {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    g_sm_count = prop.multiProcessorCount;
  }
  return g_sm_count;
}

template <int ngpus>
static __global__ void __launch_bounds__(32, 1) all2AllPlanKernel(All2AllPlanParams params) {
  int const bidx = blockIdx.x;
  int const tidx = threadIdx.x;
  const int chunk_size = params.chunk_size;
  auto input_split_sizes = params.input_split_sizes;
  auto output_split_sizes = params.output_split_sizes;

  int64_t* split_offset_size[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    split_offset_size[i] = get_tmp_buf<int64_t>(params.sg.signals[i]);
  }
  auto& cur_meta = *reinterpret_cast<struct AllToAllCommMeta*>(split_offset_size[params.rank]);

  auto& plan_meta = *reinterpret_cast<struct AllToAllPlanMeta*>(params.plan_meta_ptr);
  if (bidx == 0 && tidx == 0) {
    plan_meta.chunk_size = params.chunk_size;

    int32_t cur_offset = 0;
    int32_t cur_total_len = 0;
    int32_t offset;
    int32_t len;
    // Set the length and offset of each device's output copied from current device
    for (int i = 0; i < ngpus; i++) {
      len = output_split_sizes[i];
      if (params.output_split_offsets != nullptr) {
        offset = params.output_split_offsets[i];
        cur_offset = std::max(cur_offset, offset + len);
      } else {
        offset = cur_offset;
        cur_offset += output_split_sizes[i];
      }
      if (cur_offset * chunk_size > params.output_elts_total) {
        assert(false & "invalid output_split_sizes");
        return;
      }
      cur_meta.output_elts_offset[i] = offset * chunk_size;
      cur_meta.output_elts_length[i] = len * chunk_size;
      cur_total_len += len;
    }
    cur_meta.output_dim1 = params.output_dim1;
    cur_meta.output_stride0 = params.output_stride0;
    cur_meta.output_stride1 = params.output_stride1;

    cur_offset = 0;
    cur_total_len = 0;
    // Calculate the length and offset of the current device's input copied to each device's output
    for (int i = 0; i < ngpus; i++) {
      len = input_split_sizes[i];
      if (params.input_split_offsets != nullptr) {
        offset = params.input_split_offsets[i];
        cur_offset = std::max(cur_offset, offset + len);
      } else {
        offset = cur_offset;
        cur_offset += input_split_sizes[i];
      }
      if (cur_offset * chunk_size > params.input_elts_total) {
        assert(false & "invalid input_split_sizes");
        return;
      }
      plan_meta.local_input_elts_offset[i] = offset * chunk_size;
      plan_meta.local_input_elts_length[i] = len * chunk_size;
      cur_total_len += len;
    }
    plan_meta.input_split_elts_total = cur_total_len * chunk_size;
  }
  multi_gpu_barrier<ngpus, true>(params.sg, params.self_sg, params.rank);

  // Get the length and offset of each target device's output
  if (bidx == 0 && tidx < ngpus) {
    int peer_rank = tidx;
    // get peer output offset and length from current rank
    auto& peer_meta = *reinterpret_cast<struct AllToAllCommMeta*>(split_offset_size[peer_rank]);
    // offset in peer output buffer
    int64_t peer_elts_offset = peer_meta.output_elts_offset[params.rank];
    // element count copy to peer output buffer
    int64_t peer_elts_cnt = peer_meta.output_elts_length[params.rank];
    if (peer_elts_cnt != input_split_sizes[peer_rank] * chunk_size) {
      assert(false & "input_split_sizes mismatch peer output_split_sizes");
      multi_gpu_barrier<ngpus, false>(params.sg, params.self_sg, params.rank);
      return;
    }
    plan_meta.peer_meta[peer_rank].output_elts_offset = peer_elts_offset;
    plan_meta.peer_meta[peer_rank].output_dim1 = peer_meta.output_dim1;
    plan_meta.peer_meta[peer_rank].output_stride0 = peer_meta.output_stride0;
    plan_meta.peer_meta[peer_rank].output_stride1 = peer_meta.output_stride1;
  }
  // Calculate the number of warps used to copy to each target device
  int total_opt_warps_count = 0;
  const int elts_once = sizeof(int4) / params.elts_size * WARP_SIZE;
  for (int i = 0; i < ngpus; i++) {
    const int peer_rank = (i + params.rank) % ngpus;
    const int opt_warps_count = (input_split_sizes[peer_rank] * chunk_size + elts_once - 1) / elts_once;
    for (int k = bidx * blockDim.x + tidx; k < opt_warps_count; k += gridDim.x * blockDim.x) {
      plan_meta.warp_peer_gpu[k + total_opt_warps_count] = peer_rank;
    }
    if (bidx == 0 && tidx == 0) {
      plan_meta.warps_beg[peer_rank] = total_opt_warps_count;
    }
    total_opt_warps_count += opt_warps_count;
  }
  if (bidx == 0 && tidx == 0) {
    plan_meta.total_opt_warps_count = total_opt_warps_count;
  }
  multi_gpu_barrier<ngpus, false>(params.sg, params.self_sg, params.rank);
}

__device__ __forceinline__ int32_t
real_offset(int32_t offset, int32_t dim1, int32_t chunk_size, int32_t stride0, int32_t stride1) {
  auto dim01 = offset / chunk_size;
  return (dim01 / dim1) * stride0 + (dim01 % dim1) * stride1 + offset % chunk_size;
}

template <typename T, int ngpus>
static __global__ void __launch_bounds__(MAX_ALL_TO_ALL_WARPS* WARP_SIZE, 1) all2AllKernel(All2AllParams params) {
  int const bidx = blockIdx.x;
  int const tidx = threadIdx.x;
  int const lane_id = tidx % WARP_SIZE;
  // The source pointers. Distributed round-robin for the different warps.
  auto peer_comm_buffer_ptrs = params.peer_comm_buffer_ptrs->ptrs;
  // Start and end offsets of the thread
  const auto& plan_meta = *reinterpret_cast<struct AllToAllPlanMeta*>(params.plan_meta_ptr);
  int32_t input_split_elts_total = plan_meta.input_split_elts_total;
  if (input_split_elts_total > params.input_elts_total) {
    assert(false & "invalid input_split_sizes");
    return;
  }

  constexpr int VEC_SIZE = 16 / sizeof(T);  // every copy element cnt

  const int32_t chunk_size = plan_meta.chunk_size;
  const int32_t cur_dim1 = params.input_dim1;
  const int32_t cur_stride0 = params.input_stride0;
  const int32_t cur_stride1 = params.input_stride1;
  const T* __restrict__ const local_input = reinterpret_cast<const T*>(params.local_input_buffer_ptr);

  int warp_idx = (bidx * blockDim.x + tidx) / WARP_SIZE;
  const int warp_stride = gridDim.x * blockDim.x / WARP_SIZE;
  const int total_opt_warps_count = plan_meta.total_opt_warps_count;
  for (; warp_idx < total_opt_warps_count; warp_idx += warp_stride) {
    int peer_rank = plan_meta.warp_peer_gpu[warp_idx];
    auto warp_beg = plan_meta.warps_beg[peer_rank];
    const int32_t local_offset = plan_meta.local_input_elts_offset[peer_rank];

    auto& peer_meta = plan_meta.peer_meta[peer_rank];
    const int32_t peer_offset = peer_meta.output_elts_offset;
    const int32_t peer_dim1 = peer_meta.output_dim1;
    const int32_t peer_stride0 = peer_meta.output_stride0;
    const int32_t peer_stride1 = peer_meta.output_stride1;

    T* __restrict__ const peer_output = reinterpret_cast<T*>(peer_comm_buffer_ptrs[peer_rank]);

    int32_t chunk_start = ((warp_idx - warp_beg) * WARP_SIZE + lane_id) * VEC_SIZE;

    auto output_offset = real_offset(chunk_start + peer_offset, peer_dim1, chunk_size, peer_stride0, peer_stride1);
    auto input_offset = real_offset(chunk_start + local_offset, cur_dim1, chunk_size, cur_stride0, cur_stride1);
    *reinterpret_cast<int4*>(peer_output + output_offset) = *reinterpret_cast<int4 const*>(local_input + input_offset);
  }
  multi_gpu_barrier<ngpus, false>(params.sg, params.self_sg, params.rank);
}

std::tuple<int, int> all2AllkernelLaunchConfig(int64_t output_elts_total, int64_t input_elts_total, int64_t elts_size) {
  int chunk_size = sizeof(int4) * WARP_SIZE / elts_size;
  int elts_total = input_elts_total;
  int total_splits = std::max(1, divUp(elts_total, chunk_size));
  int sm_count = getSmCount();
  int blocks_per_grid = std::min(sm_count, MAX_ALL_TO_ALL_BLOCKS);
  int splits_per_block = divUp(total_splits, blocks_per_grid);

  int num_warps = std::min(MAX_ALL_TO_ALL_WARPS, splits_per_block);
  int threads_per_block = num_warps * WARP_SIZE;
  return std::make_tuple(blocks_per_grid, threads_per_block);
}

template <typename T>
void invokeAll2AllKernel(All2AllParams& params, cudaStream_t stream) {
  auto [blocks_per_grid, threads_per_block] =
      all2AllkernelLaunchConfig(params.output_elts_total, params.input_elts_total, params.elts_size);

#define A2A_KERNEL(ngpus) all2AllKernel<T, ngpus><<<blocks_per_grid, threads_per_block, 0, stream>>>(params)
  switch (params.ranks_per_node) {
    case 2:
      A2A_KERNEL(2);
      break;
    case 4:
      A2A_KERNEL(4);
      break;
    case 6:
      A2A_KERNEL(6);
      break;
    case 8:
      A2A_KERNEL(8);
      break;
    default:
      throw std::runtime_error("invalid world size " + std::to_string(params.ranks_per_node));
  }
#undef A2A_KERNEL

  CHECK_CUDA_SUCCESS(cudaGetLastError());
}

void trtCustomAll2All(All2AllParams& params, cudaStream_t stream) {
  switch (params.elts_size) {
    case 2:
      invokeAll2AllKernel<half>(params, stream);
      break;
    case 4:
      invokeAll2AllKernel<float>(params, stream);
      break;
    default:
      throw std::runtime_error("Unsupported data element size " + std::to_string(params.elts_size));
  }
}

void invokeAll2AllPlanKernel(All2AllPlanParams& params, cudaStream_t stream) {
  switch (params.ranks_per_node) {
    case 2:
      all2AllPlanKernel<2><<<32, 32, 0, stream>>>(params);
      break;
    case 4:
      all2AllPlanKernel<4><<<32, 32, 0, stream>>>(params);
      break;
    case 6:
      all2AllPlanKernel<6><<<32, 32, 0, stream>>>(params);
      break;
    case 8:
      all2AllPlanKernel<8><<<32, 32, 0, stream>>>(params);
      break;
    default:
      throw std::runtime_error("invalid world size " + std::to_string(params.ranks_per_node));
  }
  CHECK_CUDA_SUCCESS(cudaGetLastError());
}
}  // namespace

void all_to_all(fptr_t _fa, torch::Tensor& out, torch::Tensor& inp, torch::Tensor& plan_meta, fptr_t _reg_buffer) {
  auto fa = reinterpret_cast<CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  if (inp.dim() != 3 || out.dim() != 3) {
    throw std::runtime_error(
        "custom all_to_all currently requires input or output dim count to be 3, but got input dim count " +
        std::to_string(inp.dim()) + " and output dim count of " + std::to_string(out.dim()) + ".");
    return;
  }

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  All2AllParams params;
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  auto it = fa->buffers_.find(reg_buffer);
  if (it == fa->buffers_.end()) throw std::runtime_error("output buffer address is not registered!");
  params.peer_comm_buffer_ptrs = it->second;

  params.ranks_per_node = fa->world_size_;
  params.rank = fa->rank_;
  params.sg = fa->sg_;
  params.self_sg = fa->self_sg_;

  params.local_input_buffer_ptr = inp.data_ptr();
  params.plan_meta_ptr = reinterpret_cast<int64_t*>(plan_meta.data_ptr());

  params.input_elts_total = inp.numel();
  params.output_elts_total = out.numel();
  params.elts_size = inp.element_size();
  params.input_stride0 = inp.stride(0);
  params.input_stride1 = inp.stride(1);
  params.input_dim1 = inp.size(1);

  trtCustomAll2All(params, stream);
  if (out.numel() != 0 && out.data_ptr() != reg_buffer) {
    auto output_size = out.numel() * out.element_size();
    AT_CUDA_CHECK(cudaMemcpyAsync(out.data_ptr(), reg_buffer, output_size, cudaMemcpyDeviceToDevice, stream));
  }
}

void all_to_all_plan(
    fptr_t _fa,
    torch::Tensor& out,
    torch::Tensor& inp,
    torch::Tensor& output_split_sizes,
    torch::Tensor& input_split_sizes,
    int64_t chunk_size,
    torch::Tensor& output_split_offsets,
    torch::Tensor& input_split_offsets,
    torch::Tensor& plan_meta) {
  auto fa = reinterpret_cast<CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  if (plan_meta.numel() * plan_meta.element_size() < sizeof(AllToAllPlanMeta)) {
    assert(false && "Invalid plan meta size");
    throw std::runtime_error(
        "custom all_to_all: invalid plan meta size, requires >= " + std::to_string(sizeof(AllToAllPlanMeta)) +
        ", but got " + std::to_string(plan_meta.numel() * plan_meta.element_size()) + ".");
    return;
  }
  if (inp.dim() != 3 || out.dim() != 3) {
    throw std::runtime_error(
        "custom all_to_all currently requires input or output dim count to be 3, but got input dim count " +
        std::to_string(inp.dim()) + " and output dim count of " + std::to_string(out.dim()) + ".");
    return;
  }

  All2AllPlanParams params;
  params.ranks_per_node = fa->world_size_;
  params.rank = fa->rank_;
  params.sg = fa->sg_;
  params.self_sg = fa->self_sg_;

  params.input_elts_total = inp.numel();
  params.output_elts_total = out.numel();

  params.output_split_sizes = reinterpret_cast<int64_t*>(output_split_sizes.data_ptr());
  params.input_split_sizes = reinterpret_cast<int64_t*>(input_split_sizes.data_ptr());
  if (output_split_offsets.numel() == 0) {
    params.output_split_offsets = nullptr;
  } else {
    params.output_split_offsets = reinterpret_cast<int64_t*>(output_split_offsets.data_ptr());
  }
  if (input_split_offsets.numel() == 0) {
    params.input_split_offsets = nullptr;
  } else {
    params.input_split_offsets = reinterpret_cast<int64_t*>(input_split_offsets.data_ptr());
  }

  params.plan_meta_ptr = reinterpret_cast<int64_t*>(plan_meta.data_ptr());

  params.chunk_size = chunk_size;
  params.output_stride0 = out.stride(0);
  params.output_stride1 = out.stride(1);
  params.output_dim1 = out.size(1);

  params.elts_size = inp.element_size();
  int d = sizeof(int4) * WARP_SIZE / params.elts_size;
  if (params.input_elts_total % d != 0) {
    throw std::runtime_error(
        "custom all_to_all currently requires input length to be multiple "
        "of " +
        std::to_string(d));
    return;
  }
  if (params.input_elts_total % d != 0) {
    throw std::runtime_error(
        "custom all_to_all currently requires output length to be multiple "
        "of " +
        std::to_string(d));
    return;
  }
  auto [blocks_per_grid, threads_per_block] =
      all2AllkernelLaunchConfig(params.output_elts_total, params.input_elts_total, params.elts_size);
  params.blocks_per_grid = blocks_per_grid;
  params.threads_per_block = threads_per_block;
  invokeAll2AllPlanKernel(params, stream);
}
