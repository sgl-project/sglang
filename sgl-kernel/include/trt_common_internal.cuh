#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tuple>

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
}  // namespace trt_llm
