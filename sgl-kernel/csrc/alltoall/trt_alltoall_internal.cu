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

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tuple>

#include "trt_alltoall_internal.cuh"
#include "trt_common_internal.cuh"
#include "utils.h"

#define A2A_DEFAULT_NUM_WARPS 16

namespace trt_llm {
struct AllToAllPlanMeta {
  int64_t local_output_elts_offset[MAX_RANKS_PER_NODE];
  int64_t local_output_elts_length[MAX_RANKS_PER_NODE];
  int64_t peer_input_elts_offset[MAX_RANKS_PER_NODE];  // send to current rank
  int64_t output_split_elts_total;
  int64_t input_split_elts_total;
  int64_t output_split_pad_elts_total;
  int64_t input_split_pad_elts_total;
};

static __global__ void __launch_bounds__(32, 1) all2AllPlanKernel(All2AllPlanParams params) {
  int const bidx = blockIdx.x;
  int const tidx = threadIdx.x;
  int const grid_size = gridDim.x;
  const int block_size = params.block_size;
  auto input_split_sizes = params.input_split_sizes;
  auto output_split_sizes = params.output_split_sizes;
  auto split_offset_size = params.buffer_meta_ptrs[params.local_rank];
  // 0~2*RANKS_PER_NODE: local output offset&length
  // 2*RANKS_PER_NODE~3*RANKS_PER_NODE: peer input offset
  // 3*RANKS_PER_NODE: local output total size
  // 3*RANKS_PER_NODE+1: local input total size
  auto& plan_meta = *reinterpret_cast<struct AllToAllPlanMeta*>(params.plan_meta_ptr);
  const int RANKS_PER_NODE = params.ranks_per_node;
  if (bidx == 0 && tidx == 0) {
    int64_t cur_offset = 0;
    int64_t cur_total_len = 0;
    int64_t offset;
    int64_t len;
    for (int i = 0; i < RANKS_PER_NODE; i++) {
      len = input_split_sizes[i];
      if (params.input_split_offsets != nullptr) {
        offset = params.input_split_offsets[i];
        cur_offset = std::max(cur_offset, offset + len);
      } else {
        offset = cur_offset;
        cur_offset += input_split_sizes[i];
      }
      if (cur_offset * params.block_size > params.input_elts_total) {
        assert(false & "invalid input_split_sizes");
        return;
      }
      split_offset_size[2 * i] = offset;
      split_offset_size[2 * i + 1] = len;
      cur_total_len += len;
    }
    plan_meta.input_split_pad_elts_total = cur_offset * block_size;
    plan_meta.input_split_elts_total = cur_total_len * block_size;
    cur_offset = 0;
    cur_total_len = 0;
    for (int i = 0; i < RANKS_PER_NODE; i++) {
      len = output_split_sizes[i];
      if (params.output_split_offsets != nullptr) {
        offset = params.output_split_offsets[i];
        cur_offset = std::max(cur_offset, offset + len);
      } else {
        offset = cur_offset;
        cur_offset += output_split_sizes[i];
      }
      if (cur_offset * params.block_size > params.output_elts_total) {
        assert(false & "invalid output_split_sizes");
        return;
      }
      // current output offset, length
      plan_meta.local_output_elts_offset[i] = offset * block_size;
      plan_meta.local_output_elts_length[i] = len * block_size;
      cur_total_len += len;
    }
    plan_meta.output_split_pad_elts_total = cur_offset * block_size;
    plan_meta.output_split_elts_total = cur_total_len * block_size;
  }
  // wait for equivalent blocks of other GPUs to have copied data to their shareable buffer
  block_barrier<true>(
      params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx, grid_size);
  // peer input offset, length
  // local output offset, length
  if (bidx == 0 && tidx < RANKS_PER_NODE) {
    int peer_rank = tidx;
    // get peer output offset and length for current rank
    auto split_offset_size = params.buffer_meta_ptrs[peer_rank];
    // offset in peer output buffer
    int64_t peer_blk_offset = split_offset_size[params.local_rank * 2];
    // element count copy from peer output buffer
    int64_t peer_blk_cnt = split_offset_size[params.local_rank * 2 + 1];
    if (peer_blk_cnt != output_split_sizes[peer_rank]) {
      assert(false & "output_split_sizes mismatch peer input_split_sizes");
      block_barrier<false>(
          params.peer_barrier_ptrs_out, params.barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx, grid_size);
      return;
    }
    plan_meta.peer_input_elts_offset[peer_rank] = peer_blk_offset * block_size;
  }
  block_barrier<false>(
      params.peer_barrier_ptrs_out, params.barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx, grid_size);
}

template <typename T, int RANKS_PER_NODE, bool COPY_INPUT = true>
static __global__ void __launch_bounds__(A2A_DEFAULT_NUM_WARPS* WARP_SIZE, 1)
    all2AllKernelSyncInput(All2AllParams params) {
  int const bidx = blockIdx.x;
  int const tidx = threadIdx.x;
  // The source pointers. Distributed round-robin for the different warps.
  auto peer_comm_buffer_ptrs = params.peer_comm_buffer_ptrs->ptrs;
  auto& plan_meta = *reinterpret_cast<struct AllToAllPlanMeta*>(params.plan_meta_ptr);
  int64_t input_split_pad_elts_total = plan_meta.input_split_pad_elts_total;
  if (input_split_pad_elts_total > params.input_elts_total) {
    assert(false & "invalid input_split_sizes");
    return;
  }
  static constexpr int NUM_ELTS = 16 / sizeof(T);  // every copy element cnt

  // Start and end offsets of the thread
  int64_t chunk_start = bidx * params.elts_per_block + tidx * NUM_ELTS;
  int64_t chunk_end = std::min((bidx + 1) * params.elts_per_block, input_split_pad_elts_total);

  T* local_shared_buffer = reinterpret_cast<T*>(peer_comm_buffer_ptrs[params.local_rank]);
  T const* local_input_buffer = reinterpret_cast<T const*>(params.local_input_buffer_ptr);
  // Copy from local buffer to shareable buffer
  for (int64_t iter_offset = chunk_start; iter_offset < chunk_end; iter_offset += blockDim.x * NUM_ELTS) {
    *reinterpret_cast<int4*>(local_shared_buffer + iter_offset) =
        *reinterpret_cast<int4 const*>(local_input_buffer + iter_offset);
  }
}

template <typename T, int RANKS_PER_NODE, bool COPY_INPUT = true>
static __global__ void __launch_bounds__(A2A_DEFAULT_NUM_WARPS* WARP_SIZE, 1)
    all2AllKernelOutput(All2AllParams params) {
  int const bidx = blockIdx.x;
  int const tidx = threadIdx.x;
  int const grid_size = gridDim.x;
  const int block_size = params.block_size;
  // The source pointers. Distributed round-robin for the different warps.
  auto peer_comm_buffer_ptrs = params.peer_comm_buffer_ptrs->ptrs;
  // Start and end offsets of the thread
  auto& plan_meta = *reinterpret_cast<struct AllToAllPlanMeta*>(params.plan_meta_ptr);
  int64_t output_split_elts_total = plan_meta.output_split_elts_total;
  if (output_split_elts_total > params.output_elts_total) {
    assert(false & "invalid output_split_sizes");
    return;
  }

  static constexpr int NUM_ELTS = 16 / sizeof(T);  // every copy element cnt
  // wait for equivalent blocks of other GPUs to have copied data to their shareable buffer
  block_barrier<true>(
      params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx, grid_size);

  constexpr int NUM_ELTS_WARP_STEP = NUM_ELTS * WARP_SIZE;
  const int64_t warps_start = bidx * params.elts_per_block_per_round + (tidx / WARP_SIZE) * block_size;
  int64_t chunk_start = warps_start + (tidx % WARP_SIZE) * NUM_ELTS;
  int64_t chunk_end = std::min(warps_start + block_size, output_split_elts_total);

  auto output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);
  for (int i = 0; i < params.round_count; i++) {  // every round, one warp one block_size
    if (chunk_start >= chunk_end) {
      break;
    }

    int cur_offset = 0;
    int peer_rank = 0;
    for (; peer_rank < RANKS_PER_NODE; peer_rank++) {
      int end_cnt = cur_offset + plan_meta.local_output_elts_length[peer_rank];
      if (chunk_start >= end_cnt) {
        cur_offset = end_cnt;
        continue;
      }
      break;
    }
    int64_t offset_in_chunk = chunk_start - cur_offset;
    int64_t end_in_chunk = chunk_end - cur_offset;

    auto comm_data = reinterpret_cast<T const*>(peer_comm_buffer_ptrs[peer_rank]);
    comm_data = comm_data + plan_meta.peer_input_elts_offset[peer_rank];
    auto pad_output = output_buffer + plan_meta.local_output_elts_offset[peer_rank];

    // Each block accumulates the values from the different GPUs on the same node.
    for (int64_t iter_offset = offset_in_chunk; iter_offset < end_in_chunk; iter_offset += NUM_ELTS_WARP_STEP) {
      // Store to the destination buffer.
      *reinterpret_cast<int4*>(pad_output + iter_offset) = *reinterpret_cast<int4 const*>(comm_data + iter_offset);
    }

    chunk_start += params.elts_per_round;
    chunk_end = std::min(chunk_end + params.elts_per_round, output_split_elts_total);
  }
  block_barrier<false>(
      params.peer_barrier_ptrs_out, params.barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx, grid_size);
}

template <typename T, int RANKS_PER_NODE, bool COPY_INPUT = true>
void all2AllKernelStep(All2AllParams& params, int blocks_per_grid, int threads_per_block, cudaStream_t stream) {
  // Two kernel calls ensure that all input is copied to shared memory, not just one block.
  if (COPY_INPUT) {
    all2AllKernelSyncInput<T, RANKS_PER_NODE, COPY_INPUT><<<blocks_per_grid, threads_per_block, 0, stream>>>(params);
  }
  all2AllKernelOutput<T, RANKS_PER_NODE, COPY_INPUT><<<blocks_per_grid, threads_per_block, 0, stream>>>(params);
}

std::tuple<int, int> all2AllkernelLaunchConfig(All2AllParams& params) {
  if (params.output_elts_total % params.block_size != 0) {  // seq*16*512 or seq*16*576
    assert(false & "invalid output element count");
    return std::make_tuple(0, 0);
  }
  if (params.input_elts_total % params.block_size != 0) {
    assert(false & "invalid input element count");
    return std::make_tuple(0, 0);
  }
  int block_size = params.block_size;
  int elts_total = std::max(params.output_elts_total, params.input_elts_total);
  int total_splits = std::max(1, elts_total / block_size);
  int blocks_per_grid = MAX_ALL_TO_ALL_BLOCKS;
  int splits_per_block = divUp(total_splits, blocks_per_grid);
  // one warps one round one block_size
  int num_warps = std::min(A2A_DEFAULT_NUM_WARPS, splits_per_block);
  params.elts_per_block = splits_per_block * block_size;
  params.elts_per_block_per_round = num_warps * block_size;
  params.round_count = divUp(splits_per_block, num_warps);
  // last round may only some warps work
  params.elts_per_round = blocks_per_grid * num_warps * block_size;
  int threads_per_block = num_warps * WARP_SIZE;
  return std::make_tuple(blocks_per_grid, threads_per_block);
}

template <typename T, bool COPY_INPUT>
void dispatchA2AKernelsCopyInput(All2AllParams& params, cudaStream_t stream) {
  auto [blocks_per_grid, threads_per_block] = all2AllkernelLaunchConfig(params);
  switch (params.ranks_per_node) {
    case 2:
      all2AllKernelStep<T, 2, COPY_INPUT>(params, blocks_per_grid, threads_per_block, stream);
      break;
    case 4:
      all2AllKernelStep<T, 4, COPY_INPUT>(params, blocks_per_grid, threads_per_block, stream);
      break;
    case 6:
      all2AllKernelStep<T, 6, COPY_INPUT>(params, blocks_per_grid, threads_per_block, stream);
      break;
    case 8:
      all2AllKernelStep<T, 8, COPY_INPUT>(params, blocks_per_grid, threads_per_block, stream);
      break;
    default:
      break;
  }
}

template <typename T>
void invokeAll2AllKernel(All2AllParams& params, cudaStream_t stream) {
  if (params.is_capturing) {
    dispatchA2AKernelsCopyInput<T, false>(params, stream);
  } else {
    dispatchA2AKernelsCopyInput<T, true>(params, stream);
  }
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
      assert(false && "Unsupported data type");
  }
}

void trtCustomAll2AllPlan(All2AllPlanParams& params, cudaStream_t stream) {
  all2AllPlanKernel<<<1, 8, 0, stream>>>(params);
  CHECK_CUDA_SUCCESS(cudaGetLastError());
}
}  // namespace trt_llm

void all_to_all(
    trt_llm::fptr_t _fa, torch::Tensor& out, torch::Tensor& inp, torch::Tensor& plan_meta, int64_t block_size) {
  trt_llm::AllReduceMeta* m = reinterpret_cast<trt_llm::AllReduceMeta*>(_fa);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  auto inp_num_elements = inp.numel();
  auto outp_num_elements = out.numel();

  int world_size = m->world_size;

  trt_llm::All2AllParams params;
  params.ranks_per_node = world_size;
  params.rank = m->rank_id;
  params.local_rank = m->rank_id;
  params.local_input_buffer_ptr = inp.data_ptr();
  params.local_output_buffer_ptr = out.data_ptr();
  params.input_elts_total = inp_num_elements;
  params.output_elts_total = outp_num_elements;

  params.plan_meta_ptr = reinterpret_cast<int64_t*>(plan_meta.data_ptr());

  params.block_size = block_size;
  params.elts_size = inp.element_size();
  params.barrier_flag = ++(m->barrier_flag);

  if (block_size == 0 || block_size * params.elts_size % (sizeof(int4)) != 0) {
    assert(false && "Invalid block size, expect to be multiple of 16/element_size");
    return;
  }

  cudaStreamCaptureStatus status;
  CHECK_CUDA_SUCCESS(cudaStreamIsCapturing(stream, &status));
  params.is_capturing = (status == cudaStreamCaptureStatusActive);
  if (params.is_capturing) {
    params.peer_comm_buffer_ptrs = m->rank_data_base + m->graph_unreg_buffers.size();
    m->graph_unreg_buffers.push_back(params.local_input_buffer_ptr);
  } else {
    params.peer_comm_buffer_ptrs = m->buffers;
  }
  for (int i = 0; i < world_size; ++i) {
    params.buffer_meta_ptrs[i] = reinterpret_cast<int64_t*>(m->tmp_result_buffers[i]);
  }
  for (int i = 0; i < world_size; ++i) {
    params.peer_barrier_ptrs_in[i] = reinterpret_cast<uint32_t*>(m->barrier_in[i]);
  }
  for (int i = 0; i < world_size; ++i) {
    params.peer_barrier_ptrs_out[i] = reinterpret_cast<uint32_t*>(m->barrier_out[i]);
  }
  trt_llm::trtCustomAll2All(params, stream);
}

void all_to_all_plan(
    trt_llm::fptr_t _fa,
    torch::Tensor& out,
    torch::Tensor& inp,
    torch::Tensor& output_split_sizes,
    torch::Tensor& input_split_sizes,
    torch::Tensor& output_split_offsets,
    torch::Tensor& input_split_offsets,
    torch::Tensor& plan_meta,
    int64_t block_size) {
  trt_llm::AllReduceMeta* m = reinterpret_cast<trt_llm::AllReduceMeta*>(_fa);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  auto inp_num_elements = inp.numel();
  auto outp_num_elements = out.numel();

  if (plan_meta.numel() * plan_meta.element_size() < sizeof(trt_llm::AllToAllPlanMeta)) {
    assert(false && "Invalid plan meta size");
    return;
  }

  int world_size = m->world_size;

  trt_llm::All2AllPlanParams params;
  params.ranks_per_node = world_size;
  params.rank = m->rank_id;
  params.local_rank = m->rank_id;
  params.input_elts_total = inp_num_elements;
  params.output_elts_total = outp_num_elements;

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

  params.block_size = block_size;
  params.elts_size = inp.element_size();
  params.barrier_flag = ++(m->barrier_flag);

  if (block_size == 0 || block_size * params.elts_size % (sizeof(int4)) != 0) {
    assert(false && "Invalid block size, expect to be multiple of 16/element_size");
    return;
  }

  for (int i = 0; i < world_size; ++i) {
    params.buffer_meta_ptrs[i] = reinterpret_cast<int64_t*>(m->tmp_result_buffers[i]);
  }
  for (int i = 0; i < world_size; ++i) {
    params.peer_barrier_ptrs_in[i] = reinterpret_cast<uint32_t*>(m->barrier_in[i]);
  }
  for (int i = 0; i < world_size; ++i) {
    params.peer_barrier_ptrs_out[i] = reinterpret_cast<uint32_t*>(m->barrier_out[i]);
  }
  trt_llm::trtCustomAll2AllPlan(params, stream);
}
