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

__inline__ __device__ void multi_gpu_barrier(uint32_t** signals, uint32_t const flag, size_t const local_rank,
                                             size_t const world_size, int const tidx, int const bidx) {
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

template <typename T, int RANKS_PER_NODE> /* COPY_INPUT = false, PUSH_MODE = false */
static __global__ void oneShotAllReduceKernel(AllReduceParams params) {
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

  // The number of elements packed into one for comms
  static constexpr int NUM_ELTS = 16 / sizeof(T);

  // Packed data type for comms
  using PackedStruct = typename PackedOn16Bytes<T>::Type;

  // The source pointers. Distributed round-robin for the different warps.
  T const* buffers[RANKS_PER_NODE];

  // Start and end offsets of the thread
  size_t chunk_start = bidx * params.elts_per_block + tidx * NUM_ELTS;
  size_t chunk_end = std::min((bidx + 1) * params.elts_per_block, params.elts_per_rank);
#pragma unroll
  for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
    int rank = (params.local_rank + ii) % RANKS_PER_NODE;
    buffers[ii] = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[rank]);
  }

  multi_gpu_barrier(params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx);

  // Each block accumulates the values from the different GPUs on the same node.
  for (size_t iter_offset = chunk_start; iter_offset < chunk_end; iter_offset += blockDim.x * NUM_ELTS) {
    // Iterate over the different ranks/devices on the node to load the values.
    PackedStruct vals[RANKS_PER_NODE];
#pragma unroll
    for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
      vals[ii].packed = *reinterpret_cast<int4 const*>(&buffers[ii][iter_offset]);
    }

    // Sum the values from the different ranks.
    PackedStruct sums;
    sums.packed = {0, 0, 0, 0};
#pragma unroll
    for (int rank = 0; rank < RANKS_PER_NODE; ++rank) {
      // Always reduce from rank 0 to ensure stable reduce order.
      int ii = (rank + RANKS_PER_NODE - params.local_rank) % RANKS_PER_NODE;
      sums.packed = add128b(sums, vals[ii]);
    }

    // Store to the destination buffer.
    *reinterpret_cast<int4*>(&reinterpret_cast<T*>(params.local_output_buffer_ptr)[iter_offset]) = sums.packed;
  }
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
    default:
      assert(false && "Algorithm not supported here.");
  }

  return std::make_tuple(blocks_per_grid, threads_per_block);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int RANKS_PER_NODE>
void dispatchARKernels(AllReduceStrategyType algo, AllReduceParams& param, int blocks_per_grid, int threads_per_block,
                       cudaStream_t stream) {
  oneShotAllReduceKernel<T, RANKS_PER_NODE><<<blocks_per_grid, threads_per_block, 0, stream>>>(param);
}

template <typename T>
void invokeOneOrTwoShotAllReduceKernel(AllReduceParams& param, AllReduceStrategyType strat, cudaStream_t stream) {
  void* buffer = reinterpret_cast<void*>(param.peer_comm_buffer_ptrs[param.rank]);
  void* local_inp_buffer = param.local_input_buffer_ptr;
  CHECK_CUDA_SUCCESS(
      cudaMemcpyAsync(buffer, local_inp_buffer, param.elts_total * param.elts_size, cudaMemcpyDeviceToDevice, stream));

  assert(strat == AllReduceStrategyType::ONESHOT && "Custom allreduce only support oneshot");
  CHECK_CUDA_SUCCESS(cudaGetLastError());

  size_t elts_per_thread = 16 / sizeof(T);
  auto [blocks_per_grid, threads_per_block] = kernelLaunchConfig(strat, param, elts_per_thread);
  switch (param.ranks_per_node) {
    case 2:
      dispatchARKernels<T, 2>(strat, param, blocks_per_grid, threads_per_block, stream);
      break;
    case 4:
      dispatchARKernels<T, 4>(strat, param, blocks_per_grid, threads_per_block, stream);
      break;
    case 6:
      dispatchARKernels<T, 6>(strat, param, blocks_per_grid, threads_per_block, stream);
      break;
    case 8:
      dispatchARKernels<T, 8>(strat, param, blocks_per_grid, threads_per_block, stream);
      break;
    default:
      break;
  }
  CHECK_CUDA_SUCCESS(cudaGetLastError());
}

void trtCustomAllReduce(AllReduceParams& params, at::ScalarType data_type, AllReduceStrategyType strat,
                        cudaStream_t stream) {
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
