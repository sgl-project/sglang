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

#include "trt_reduce_internal.cuh"
#include "utils.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void st_flag_release(uint32_t const& flag, uint32_t* flag_addr) {
  asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t ld_flag_acquire(uint32_t* flag_addr) {
  uint32_t flag;
  asm volatile("ld.global.acquire.sys.b32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

static inline __device__ void st_flag_volatile(uint32_t const& flag, uint32_t* flag_addr) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static inline __device__ uint32_t ld_flag_volatile(uint32_t* flag_addr) {
  uint32_t flag;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

namespace trt_llm {
////////////////////////////////////////////////////////////////////////////////////////////////////

// Type Converter that packs data format to 128 bits data type
//
using PackedFloat = union {
  int4 packed;
  float unpacked[4];
};

using PackedHalf = union {
  int4 packed;
  half2 unpacked[4];
};

template <typename T>
struct PackedOn16Bytes {};

template <>
struct PackedOn16Bytes<float> {
  using Type = PackedFloat;
};

template <>
struct PackedOn16Bytes<half> {
  using Type = PackedHalf;
};

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
using PackedBFloat16 = union {
  int4 packed;
  __nv_bfloat162 unpacked[4];
};

template <>
struct PackedOn16Bytes<__nv_bfloat16> {
  using Type = PackedBFloat16;
};
#endif

// add two 128b data
template <typename T>
inline __device__ int4 add128b(T& a, T& b) {
  T c;
  c.unpacked[0] = a.unpacked[0] + b.unpacked[0];
  c.unpacked[1] = a.unpacked[1] + b.unpacked[1];
  c.unpacked[2] = a.unpacked[2] + b.unpacked[2];
  c.unpacked[3] = a.unpacked[3] + b.unpacked[3];
  return c.packed;
}

__inline__ __device__ void multi_gpu_barrier(
    uint32_t** signals,
    uint32_t const flag,
    size_t const local_rank,
    size_t const world_size,
    int const tidx,
    int const bidx) {
  // After this function, at least one block in each GPU has reached the barrier
  if (tidx < world_size) {
    // we can think of signals having the shape [world_size, world_size]
    // Dimension 0 is the "listening" dimension, dimension 1 is "emitting" dimension

    // Block 0 broadcasts its flag (local_rank on emitting dimension) to all receivers
    size_t offset = (flag % 2) ? world_size : 0;

    if (bidx == 0) {
      st_flag_release(flag, signals[tidx] + offset + local_rank);
    }

    // All blocks check that corresponding block 0 on other GPUs have set the flag
    // No deadlock because block #0 is always the first block started
    uint32_t* peer_barrier_d = signals[local_rank] + offset + tidx;
    while (ld_flag_acquire(peer_barrier_d) != flag) {
    }
  }

  __syncthreads();
}

template <bool start, bool need_fence = false>
__inline__ __device__ void block_barrier(
    uint32_t** signals,
    uint32_t const flag,
    size_t const local_rank,
    size_t const world_size,
    int const tidx,
    int const bidx,
    int const grid_size) {
  if constexpr (!start) {
    __syncthreads();
  }
  // After this function, the block of id == bidx of each GPU has reached the barrier
  if (tidx < world_size) {
    // we can think of signals having the shape [world_size, 2, num_blocks, world_size]
    // (+ an offset on dim 2 to account for flags used in multi_gpu_barrier)
    // Dimension 0 is the "listening" dimension, dimension 3 is "emitting" dimension

    // Block broadcast its flag (local_rank on emitting dimension) to all receivers
    uint32_t flag_block_offset = world_size + bidx * world_size;

    flag_block_offset += (grid_size + 1) * world_size * (flag % 2);

    uint32_t* peer_barrier_d = signals[local_rank] + flag_block_offset + tidx;
    // Blocks check that corresponding blocks on other GPUs have also set the flag
    if constexpr (need_fence) {
      st_flag_release(flag, signals[tidx] + flag_block_offset + local_rank);
      while (ld_flag_acquire(peer_barrier_d) != flag) {
      }
    } else {
      st_flag_volatile(flag, signals[tidx] + flag_block_offset + local_rank);
      while (ld_flag_volatile(peer_barrier_d) != flag) {
      }
    }
  }
  if constexpr (start || need_fence) {
    __syncthreads();
  }
}

template <typename T, int RANKS_PER_NODE, bool COPY_INPUT = true>
static __global__ void __launch_bounds__(512, 1) oneShotAllReduceKernel(AllReduceParams params) {
  // Suppose that two GPUs participate in the AR exchange, and we start four blocks.
  // The message is partitioned into chunks as detailed below:
  //               message
  //       |-------------------|
  // GPU 0 | B0 | B1 | B2 | B3 |
  // GPU 1 | B0 | B1 | B2 | B3 |
  //
  // Here the step-by-step behavior of one block:
  // 1. B0 copies the chunk it  is responsible for, from local_input to shareable buffer
  // 2. B0 on GPU 0 and B0 on GPU 1 wait for each other (block_barrier)
  // 3. B0 on GPU 0 pull and sum the chunk from GPU 1, writes the result to local_output
  //
  // With COPY_INPUT == false, skip step 1. and use gpu_barrier instead of block barrier during step 2.
  // We only to know if the other GPU as arrived at the AR kernel, that would mean that data is ready
  //
  // With PUSH_MODE, we consider that the shared buffer is of size:
  // params.peer_comm_buffer_ptrs: [world_size, world_size, message_size]
  //
  // Here the step-by-step behavior of one block:
  // 1. B0 push the chunk is it responsible for into all other GPUs:
  //    params.peer_comm_buffer_ptrs[:, local_gpu, B0 slice]
  // 2. block sync so the block is shared by other GPUs
  // 3. Reduce along second dimension params.peer_comm_buffer_ptrs[local_gpu, :, B0 slice]

  int const bidx = blockIdx.x;
  int const tidx = threadIdx.x;
  int const grid_size = gridDim.x;

  // The number of elements packed into one for comms
  static constexpr int NUM_ELTS = 16 / sizeof(T);

  // Packed data type for comms
  using PackedStruct = typename PackedOn16Bytes<T>::Type;

  // The source pointers. Distributed round-robin for the different warps.
  auto peer_comm_buffer_ptrs = params.peer_comm_buffer_ptrs->ptrs;
  T* local_shared_buffer = reinterpret_cast<T*>(peer_comm_buffer_ptrs[params.local_rank]);
  // Start and end offsets of the thread
  size_t chunk_start = bidx * params.elts_per_block + tidx * NUM_ELTS;
  size_t chunk_end = std::min((bidx + 1) * params.elts_per_block, params.elts_per_rank);

  if constexpr (COPY_INPUT) {
    T const* local_input_buffer = reinterpret_cast<T const*>(params.local_input_buffer_ptr);
    // Copy from local buffer to shareable buffer
    for (size_t iter_offset = chunk_start; iter_offset < chunk_end; iter_offset += blockDim.x * NUM_ELTS) {
      *reinterpret_cast<int4*>(&local_shared_buffer[iter_offset]) =
          *reinterpret_cast<int4 const*>(&local_input_buffer[iter_offset]);
    }
  }
  // wait for equivalent blocks of other GPUs to have copied data to their shareable buffer
  block_barrier<true>(
      params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx, grid_size);

  // Each block accumulates the values from the different GPUs on the same node.
  for (size_t iter_offset = chunk_start; iter_offset < chunk_end; iter_offset += blockDim.x * NUM_ELTS) {
    // Iterate over the different ranks/devices on the node to load the values.
    PackedStruct vals[RANKS_PER_NODE];
#pragma unroll
    for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
      vals[ii].packed = *reinterpret_cast<int4 const*>(&((T*)peer_comm_buffer_ptrs[ii])[iter_offset]);
    }

    // Sum the values from the different ranks.
    PackedStruct sums;
    sums.packed = {0, 0, 0, 0};
#pragma unroll
    for (int rank = 0; rank < RANKS_PER_NODE; ++rank) {
      // Always reduce from rank 0 to ensure stable reduce order.
      sums.packed = add128b(sums, vals[rank]);
    }

    // Store to the destination buffer.
    *reinterpret_cast<int4*>(&reinterpret_cast<T*>(params.local_output_buffer_ptr)[iter_offset]) = sums.packed;
  }
  block_barrier<false>(
      params.peer_barrier_ptrs_out, params.barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx, grid_size);
}

template <typename T, int RANKS_PER_NODE, bool COPY_INPUT = true>
static __global__ void __launch_bounds__(512, 1) twoShotAllReduceKernel(AllReduceParams params) {
  // Suppose that two GPUs participate in the AR exchange, and we start two blocks.
  // The message is partitioned into chunks as detailed below:
  //               message
  //       |-------------------|
  //       |--GPU 0--|--GPU 1--| (GPU responsibility parts)
  // GPU 0 | B0 | B1 | B0 | B1 |
  // GPU 1 | B0 | B1 | B0 | B1 |
  //
  // Here the step-by-step behavior of one block:
  // 1. B0 copies all chunks is it responsible for, from local_input to shareable buffer
  // 2. B0 on GPU 0 and B0 on GPU 1 wait for each other (block_barrier #0)
  // 3. B0 on GPU 0 gather and sum the B0 chunks from GPU 1, that are in the GPU 0 responsibility
  //    part (the first half of the message, see GPU responsibility row above)
  // 3bis. Likewise, B0 on GPU 1 copies and sum the chunks for GPU 0,
  //       where GPU 1 is responsible: the second half of the message.
  // 4. B0 on GPU 0 and B0 on GPU 1 wait for each other (block_barrier #1)
  // 5. B0 writes result to local_output. It gathers each chunk from its responsible GPU.
  //    For example, here it reads the first chunk from GPU 0 and second chunk from GPU 1.
  //
  // With COPY_INPUT == false, skip step 1. and use gpu_barrier instead of block barrier during step 2.
  // We only to know if the other GPU as arrived at the AR kernel, that would mean that data is ready
  // to be read.
  //
  // Note that compared to one-shot, one block (CTA) writes multiple input chunks and write multiple output chunks.
  // However, it's only responsible for the summation of a single chunk.
  //
  // With PUSH_MODE, we consider that the shared buffer is of size:
  // params.peer_comm_buffer_ptrs: [world_size, world_size, message_size / world_size]
  //
  // Here the step-by-step behavior of one block:
  // 1. B0 push the chunks is it responsible for into the corresponding GPUs:
  //    params.peer_comm_buffer_ptrs[target_gpu, local_gpu, current B0 slice]
  // 2. block sync so the blocks have been shared by other GPUs
  // 3. Reduce along second dimension params.peer_comm_buffer_ptrs[local_gpu, :, B0 slice]
  // 4. block barrier (corresponding blocks have finished reduction)
  // 5. pull and write on local buffer, by reading params.peer_comm_buffer_ptrs[:, 0, B0 slice] (reduction result is
  //    written at index 0 of 2nd dim)

  int const bidx = blockIdx.x;
  int const tidx = threadIdx.x;
  int const grid_size = gridDim.x;

  // The number of elements packed into one for comms
  static constexpr int PACKED_ELTS = 16 / sizeof(T);
  using PackedType = typename PackedOn16Bytes<T>::Type;

  T const* local_input_buffer = reinterpret_cast<T const*>(params.local_input_buffer_ptr);
  auto peer_comm_buffer_ptrs = params.peer_comm_buffer_ptrs->ptrs;
  T* local_shared_buffer = reinterpret_cast<T*>(peer_comm_buffer_ptrs[params.local_rank]);
  T* local_output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);

  size_t const chunk_start = bidx * params.elts_per_block + tidx * PACKED_ELTS;
  size_t const chunk_end = min(chunk_start + params.elts_per_block, params.elts_per_rank);

  T* buffers[RANKS_PER_NODE];
  T* buffers_unorder[RANKS_PER_NODE];
  int ranks[RANKS_PER_NODE];
#pragma unroll
  for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
    // A mapping of the ranks to scatter reads as much as possible
    int rank = (params.local_rank + ii) % RANKS_PER_NODE;
    ranks[ii] = rank;
    buffers[ii] = reinterpret_cast<T*>(peer_comm_buffer_ptrs[rank]);
    buffers_unorder[ii] = reinterpret_cast<T*>(peer_comm_buffer_ptrs[ii]);
  }

#if (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12))
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
#endif

  if constexpr (COPY_INPUT) {
    // Copy all blocks from local buffer to shareable buffer
    for (size_t local_offset = chunk_start; local_offset < chunk_end; local_offset += blockDim.x * PACKED_ELTS) {
#pragma unroll
      for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
        size_t offset_rank = ranks[ii] * params.elts_per_rank + local_offset;
        if (offset_rank >= params.elts_total) {
          continue;
        }
        *reinterpret_cast<int4*>(&local_shared_buffer[offset_rank]) =
            *reinterpret_cast<int4 const*>(&local_input_buffer[offset_rank]);
      }
    }
  }
  block_barrier<true>(
      params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx, grid_size);

  // Each block accumulates the values from the different GPUs on the same node.
  for (size_t local_offset = chunk_start; local_offset < chunk_end; local_offset += blockDim.x * PACKED_ELTS) {
    size_t const responsible_block_offset = local_offset + params.rank_offset;

    // Iterate over the different ranks/devices on the node to load the values.
    PackedType vals[RANKS_PER_NODE];
#pragma unroll
    for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
      vals[ii].packed = *reinterpret_cast<int4 const*>(&buffers_unorder[ii][responsible_block_offset]);
    }

    // Sum the values from the different ranks.
    PackedType sums;
    sums.packed = {0, 0, 0, 0};
#pragma unroll
    for (int rank = 0; rank < RANKS_PER_NODE; ++rank) {
      // Always reduce from rank 0 to ensure stable reduce order.
      sums.packed = add128b(sums, vals[rank]);
    }

    // Store to the local buffer or tmp buffer
    if constexpr (COPY_INPUT) {
      *reinterpret_cast<int4*>(&local_shared_buffer[responsible_block_offset]) = sums.packed;
    } else {
      *reinterpret_cast<int4*>(&params.tmp_result_buffers[params.local_rank][responsible_block_offset]) = sums.packed;
    }
  }

  block_barrier<false, true>(
      params.peer_barrier_ptrs_out, params.barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx, grid_size);

  // Gather all needed elts from other intra-node ranks
  for (size_t local_offset = chunk_start; local_offset < chunk_end; local_offset += blockDim.x * PACKED_ELTS) {
#pragma unroll
    for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
      // use round-robin gathering from other ranks
      size_t offset_rank = ranks[ii] * params.elts_per_rank + local_offset;
      if (offset_rank >= params.elts_total) {
        continue;
      }
      if constexpr (COPY_INPUT) {
        *reinterpret_cast<int4*>(&local_output_buffer[offset_rank]) =
            *reinterpret_cast<int4*>(&buffers[ii][offset_rank]);
      } else {
        *reinterpret_cast<int4*>(&local_output_buffer[offset_rank]) =
            *reinterpret_cast<int4*>(&params.tmp_result_buffers[ranks[ii]][offset_rank]);
      }
    }
  }
#if (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12))
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int divUp(int a, int b) {
  return (a + b - 1) / b;
}

inline int roundUp(int a, int n) {
  return divUp(a, n) * n;
}

std::tuple<int, int> kernelLaunchConfig(AllReduceStrategyType algo, AllReduceParams& params, size_t elts_per_thread) {
  int blocks_per_grid = 1, threads_per_block = DEFAULT_BLOCK_SIZE;
  switch (algo) {
    case AllReduceStrategyType::ONESHOT: {
      assert(params.elts_total % elts_per_thread == 0);
      size_t const total_threads = roundUp(params.elts_total / elts_per_thread, WARP_SIZE);
      threads_per_block = std::min(DEFAULT_BLOCK_SIZE, total_threads);
      blocks_per_grid = std::min(static_cast<int>(MAX_ALL_REDUCE_BLOCKS), divUp(total_threads, threads_per_block));
      params.elts_per_block = roundUp(divUp(params.elts_total, blocks_per_grid), elts_per_thread);
      params.elts_per_rank = params.elts_total;
      break;
    }
    case AllReduceStrategyType::TWOSHOT: {
      assert(params.elts_total % (elts_per_thread * params.ranks_per_node) == 0);
      size_t const total_threads = roundUp(params.elts_total / (elts_per_thread * params.ranks_per_node), WARP_SIZE);

      threads_per_block = std::min(DEFAULT_BLOCK_SIZE, total_threads);
      blocks_per_grid = std::min(static_cast<int>(MAX_ALL_REDUCE_BLOCKS), divUp(total_threads, threads_per_block));
      params.elts_per_rank = params.elts_total / params.ranks_per_node;
      params.rank_offset = params.local_rank * params.elts_per_rank;
      params.elts_per_block = roundUp(divUp(params.elts_per_rank, blocks_per_grid), elts_per_thread);
      break;
    }
    default:
      assert(false && "Algorithm not supported here.");
  }

  return std::make_tuple(blocks_per_grid, threads_per_block);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int RANKS_PER_NODE, bool COPY_INPUT>
void dispatchARKernels(
    AllReduceStrategyType algo,
    AllReduceParams& param,
    int blocks_per_grid,
    int threads_per_block,
    cudaStream_t stream) {
  switch (algo) {
    case AllReduceStrategyType::ONESHOT: {
      oneShotAllReduceKernel<T, RANKS_PER_NODE, COPY_INPUT><<<blocks_per_grid, threads_per_block, 0, stream>>>(param);
      break;
    }
    case AllReduceStrategyType::TWOSHOT: {
      twoShotAllReduceKernel<T, RANKS_PER_NODE, COPY_INPUT><<<blocks_per_grid, threads_per_block, 0, stream>>>(param);
      break;
    }
  }
}

template <typename T, bool COPY_INPUT>
void dispatchARKernelsCopyInput(AllReduceStrategyType strat, AllReduceParams& param, cudaStream_t stream) {
  size_t elts_per_thread = 16 / sizeof(T);
  auto [blocks_per_grid, threads_per_block] = kernelLaunchConfig(strat, param, elts_per_thread);
  switch (param.ranks_per_node) {
    case 2:
      dispatchARKernels<T, 2, COPY_INPUT>(strat, param, blocks_per_grid, threads_per_block, stream);
      break;
    case 4:
      dispatchARKernels<T, 4, COPY_INPUT>(strat, param, blocks_per_grid, threads_per_block, stream);
      break;
    case 6:
      dispatchARKernels<T, 6, COPY_INPUT>(strat, param, blocks_per_grid, threads_per_block, stream);
      break;
    case 8:
      dispatchARKernels<T, 8, COPY_INPUT>(strat, param, blocks_per_grid, threads_per_block, stream);
      break;
    default:
      break;
  }
}

template <typename T>
void invokeOneOrTwoShotAllReduceKernel(AllReduceParams& param, AllReduceStrategyType strat, cudaStream_t stream) {
  if (param.is_capturing) {
    dispatchARKernelsCopyInput<T, false>(strat, param, stream);
  } else {
    dispatchARKernelsCopyInput<T, true>(strat, param, stream);
  }
  CHECK_CUDA_SUCCESS(cudaGetLastError());
}

void trtCustomAllReduce(
    AllReduceParams& params, at::ScalarType data_type, AllReduceStrategyType strat, cudaStream_t stream) {
  if (params.elts_total == 0) {
    return;
  }

  switch (data_type) {
    case at::ScalarType::Float:
      invokeOneOrTwoShotAllReduceKernel<float>(params, strat, stream);
      break;
    case at::ScalarType::Half:
      invokeOneOrTwoShotAllReduceKernel<half>(params, strat, stream);
      break;
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16:
      invokeOneOrTwoShotAllReduceKernel<__nv_bfloat16>(params, strat, stream);
      break;
#endif
    default:
      assert(false && "Unsupported data type");
  }
}
}  // namespace trt_llm
