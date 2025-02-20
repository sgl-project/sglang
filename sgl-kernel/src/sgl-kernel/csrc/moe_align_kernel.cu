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

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <torch/extension.h>

#include <THC/THCAtomics.cuh>

#include "utils.h"

#define MAX_NUM_EXPERTS 256
#define EXPERTS_PER_WARP ((MAX_NUM_EXPERTS) / (WARP_SIZE))

#define FRAGS_PER_BLOCK 4

#define FRAG_SIZE_M 16
#define FRAG_SIZE_N 16

#ifndef USE_ROCM
#define kWarpsToLoad 2
#else
#define kWarpsToLoad 1
#endif

#define kElementsPerAccess 4
#define kElementsPerThr 16

#define SGLANG_FORCE_INLINE_DEVICE_FUNC static __forceinline__ __attribute__((always_inline)) __device__

namespace cg = cooperative_groups;

SGLANG_FORCE_INLINE_DEVICE_FUNC void store_global_cumsum(int* cumsum /*dest*/, int* total_tokens_post_pad /*dest*/,
                                                         const int32_t* local_offsets, const int& tid,
                                                         const int& num_experts, cg::grid_group& grid) {
  int active_threads = CEILDIV(num_experts + 1, kElementsPerThr);
  if (tid < active_threads - 1) {
    for (int i = tid * kElementsPerThr; i < (tid + 1) * kElementsPerThr; i += kElementsPerAccess) {
      *(int4*)(cumsum + i) = *(int4*)(local_offsets + i);
    }
  }

  if (tid == active_threads - 1) {
#pragma unroll
    for (int i = tid * kElementsPerThr; i < num_experts + 1; i++) {
      *(cumsum + i) = *(local_offsets + i);
    }
  }
  if (tid == active_threads) {
    *total_tokens_post_pad = local_offsets[num_experts];
  }
  __threadfence_system();
  grid.sync();
}

SGLANG_FORCE_INLINE_DEVICE_FUNC void align_global_cumsum(int32_t* local_offsets /*src_and_dest*/, const int tid,
                                                         const int32_t& block_size, const int32_t& num_experts) {
  if (tid == 0) {
    for (int i = num_experts; i > 0; i--) {
      local_offsets[i] = local_offsets[i] - local_offsets[i - 1];
    }
    for (int i = 1; i < num_experts + 1; i++) {
      local_offsets[i] = local_offsets[i - 1] + CEILDIV(local_offsets[i], block_size) * block_size;
    }
  }
  __syncthreads();
}

SGLANG_FORCE_INLINE_DEVICE_FUNC void reduce_unaligned_cumsum(int* tokens_cnts_ptr /*src_and_dest*/, int* smem_ptr, int32_t* local_offsets,
                                                             const int& tid, const int& lane_id, const int& warp_id,
                                                             const int32_t& num_experts, cg::grid_group& grid) {
  int total_fragments = CEILDIV(num_experts, FRAG_SIZE_N);
  int fragments_per_block = CEILDIV(total_fragments, gridDim.x);
  int fragments_per_warp = CEILDIV(fragments_per_block, FRAGS_PER_BLOCK);

  for (int i = 0; i < gridDim.x; i += FRAG_SIZE_M) {
    for (int j = 0; j < fragments_per_warp; j++) {
      if (warp_id * fragments_per_warp < kWarpsToLoad * fragments_per_block) {
        const int kNumThrPerRow = WARP_SIZE / FRAG_SIZE_N;
        int sRow = lane_id / kNumThrPerRow;

        int sWarpColStride = kNumThrPerRow * kElementsPerAccess;
        int sWarpColOff = warp_id * sWarpColStride;
        int sThrColOff = lane_id % kNumThrPerRow * kElementsPerAccess;

        int sCol = sThrColOff + sWarpColOff;

        int gRow = i + sRow;

        int gBlockColOff = blockIdx.x * fragments_per_block * FRAG_SIZE_N;
        int gWarpColOff_0 = (warp_id / kWarpsToLoad * fragments_per_warp + j) * FRAG_SIZE_N;
        int gWarpColOff_1 = warp_id % kWarpsToLoad * sWarpColStride;

        int gCol = gBlockColOff + gWarpColOff_0 + gWarpColOff_1 + sThrColOff;

        if (gRow < num_experts && gCol < num_experts) {
          int4* tokens_cnts_4i_ptr = (int4*)(tokens_cnts_ptr + (gRow + 1) * num_experts + gCol);
          int4* smem_4i_ptr = (int4*)(smem_ptr + sRow * FRAGS_PER_BLOCK * FRAG_SIZE_N + sCol);

          *smem_4i_ptr = *tokens_cnts_4i_ptr;
        }
      }
      __syncthreads();

      if (warp_id * fragments_per_warp < kWarpsToLoad * fragments_per_block) {
        if (warp_id % kWarpsToLoad == 0) {
          for (int k = 0; k < FRAG_SIZE_M; k += (WARP_SIZE / FRAG_SIZE_N)) {
            int sRow = lane_id / FRAG_SIZE_N + k;
            int sThrColOff = lane_id % FRAG_SIZE_N;
            int sCol = sThrColOff + (warp_id / kWarpsToLoad) * FRAG_SIZE_N;

            int gBlockColOff = blockIdx.x * fragments_per_block * FRAG_SIZE_N;
            int gWarpColOff_0 = (warp_id / kWarpsToLoad * fragments_per_warp + j) * FRAG_SIZE_N;
            int gCol = gBlockColOff + gWarpColOff_0 + sThrColOff;
            if (gCol < num_experts) {
              atomicAdd(tokens_cnts_ptr + gCol, *(smem_ptr + sRow * FRAGS_PER_BLOCK * FRAG_SIZE_N + sCol));
            }
          }
        }
      }
      __syncthreads();

    }  // end of j
  }    // end of i
  __threadfence_system();
  grid.sync();

  if (tid < num_experts) {
    *(local_offsets + tid + 1) = *(tokens_cnts_ptr + tid);
  }
  __syncthreads();
}

SGLANG_FORCE_INLINE_DEVICE_FUNC void parallel_unaligned_local_cumsum(
    const int& tid, int* tokens_cnts_ptr /*dest*/, int32_t* local_offsets /*dest*/, int32_t* local_offsets_buf,
    const int32_t (*shared_counts)[EXPERTS_PER_WARP] /*src*/, const int& experts_per_warp, const int32_t& num_experts,
    cg::grid_group& grid) {
  int active_threads = CEILDIV(num_experts, kElementsPerThr);

  if (threadIdx.x == 0) {
    local_offsets[0] = 0;
  }
  if (threadIdx.x < active_threads - 1) {
    for (int i = threadIdx.x * kElementsPerThr + 1; i < (threadIdx.x + 1) * kElementsPerThr + 1; ++i) {
      int warp_idx = (i - 1) / experts_per_warp;
      int expert_offset = (i - 1) % experts_per_warp;

      int expert_count = shared_counts[warp_idx][expert_offset];

      int last_val = (i - 1) % kElementsPerThr == 0 ? 0 : local_offsets[i - 1];
      local_offsets[i] = last_val + expert_count;
    }

    local_offsets_buf[threadIdx.x] = local_offsets[(threadIdx.x + 1) * kElementsPerThr];
  }

  if (threadIdx.x == active_threads - 1) {
#pragma unroll
    for (int i = threadIdx.x * kElementsPerThr + 1; i < num_experts + 1; ++i) {
      int warp_idx = (i - 1) / experts_per_warp;
      int expert_offset = (i - 1) % experts_per_warp;

      int expert_count = shared_counts[warp_idx][expert_offset];

      int last_val = (i - 1) % kElementsPerThr == 0 ? 0 : local_offsets[i - 1];
      local_offsets[i] = last_val + expert_count;
    }

    local_offsets_buf[threadIdx.x] = local_offsets[num_experts];
  }
  __syncthreads();

  if (threadIdx.x < active_threads - 1 && threadIdx.x > 0) {
    int offset = 0;
    for (int j = 0; j < threadIdx.x; ++j) {
      offset += local_offsets_buf[j];
    }

    for (int i = threadIdx.x * kElementsPerThr + 1; i < (threadIdx.x + 1) * kElementsPerThr + 1; ++i) {
      local_offsets[i] += offset;
    }
  }
  if (threadIdx.x == active_threads - 1) {
    int offset = 0;
    for (int j = 0; j < threadIdx.x; ++j) {
      offset += local_offsets_buf[j];
    }
    for (int i = threadIdx.x * kElementsPerThr + 1; i < num_experts + 1; ++i) {
      local_offsets[i] += offset;
    }
  }
  __syncthreads();

  if (tid < num_experts) {
    *(tokens_cnts_ptr + tid) = 0;
  }
  if (threadIdx.x < num_experts) {
    *(tokens_cnts_ptr + (blockIdx.x + 1) * num_experts + threadIdx.x) = *(local_offsets + threadIdx.x + 1);
    *(local_offsets + threadIdx.x + 1) = 0;
  } else if (threadIdx.x < MAX_NUM_EXPERTS) {
    *(local_offsets + threadIdx.x + 1) = 0;
  }
  __threadfence_system();
  grid.sync();
}

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(const scalar_t* __restrict__ topk_ids,
                                            int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
                                            int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts,
                                            int32_t block_size, size_t numel, int32_t* __restrict__ tokens_cnts,
                                            int32_t* __restrict__ cumsum, const int tokens_per_block,
                                            const int tokens_per_thread, const int K) {
  __shared__ int32_t smem[FRAG_SIZE_M * FRAG_SIZE_N * FRAGS_PER_BLOCK];
  int32_t(*shared_counts)[EXPERTS_PER_WARP] = (int32_t(*)[EXPERTS_PER_WARP]) & smem[0];

  __shared__ int32_t local_offsets[MAX_NUM_EXPERTS + 1];
  __shared__ int32_t local_offsets_buf[CEILDIV(MAX_NUM_EXPERTS, kElementsPerThr)];

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int experts_per_warp = EXPERTS_PER_WARP;

  int* tokens_cnts_ptr = &(tokens_cnts[0]);
  int* smem_ptr = &(smem[0]);

  cg::grid_group grid = cg::this_grid();

  if (threadIdx.x < FRAG_SIZE_M * FRAG_SIZE_N) {
    for (int i = 0; i < FRAG_SIZE_M * FRAG_SIZE_N * FRAGS_PER_BLOCK; i += FRAG_SIZE_M * FRAG_SIZE_N) {
      smem[threadIdx.x + i] = 0;
    }
  }
  __syncthreads();

  const size_t start_idx = tokens_per_block * blockIdx.x + tokens_per_thread * threadIdx.x;
  const size_t end_idx = start_idx + tokens_per_thread;

  if (threadIdx.x * tokens_per_thread < tokens_per_block) {
    for (int i = start_idx; i < MIN(numel, end_idx); ++i) {
      int expert_id = topk_ids[i];
      int warp_idx = expert_id / experts_per_warp;
      int expert_offset = expert_id % experts_per_warp;
      atomicAdd(&shared_counts[warp_idx][expert_offset], 1);
    }
  }
  __syncthreads();

  parallel_unaligned_local_cumsum(tid, tokens_cnts_ptr /*dest*/, local_offsets, local_offsets_buf, shared_counts,
                                  experts_per_warp, num_experts, grid);

  reduce_unaligned_cumsum(tokens_cnts_ptr /*src_and_dest*/, smem_ptr, local_offsets, tid, lane_id, warp_id, num_experts, grid);

  align_global_cumsum(local_offsets /*src_and_dest*/, tid, block_size, num_experts);

  store_global_cumsum(cumsum /*dest*/, total_tokens_post_pad /*dest*/, local_offsets /*src*/, tid, num_experts, grid);

  if (tid < num_experts) {
    for (int i = local_offsets[tid]; i < local_offsets[tid + 1]; i += block_size) {
      expert_ids[i / block_size] = tid;
    }
  }
  __syncthreads();

  if (threadIdx.x * tokens_per_thread < tokens_per_block) {
    for (int i = start_idx; i < MIN(numel, end_idx); ++i) {
      int32_t expert_id = topk_ids[i];
      int32_t rank_post_pad = atomicAdd(&cumsum[expert_id], 1);
      sorted_token_ids[rank_post_pad] = i;
    }
  }
}

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
                          torch::Tensor sorted_token_ids, torch::Tensor experts_ids, torch::Tensor num_tokens_post_pad,
                          torch::Tensor token_cnts_buffer, torch::Tensor cumsum_buffer) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
    auto kernel = moe_align_block_size_kernel<scalar_t>;

    const int block_threads = 256;

    const int num_blocks = MIN(CEILDIV(topk_ids.sizes()[0], block_threads), num_experts);

    scalar_t* topk_ids_ptr = topk_ids.data_ptr<scalar_t>();
    int32_t* sorted_token_ids_ptr = sorted_token_ids.data_ptr<int32_t>();
    int32_t* experts_ids_ptr = experts_ids.data_ptr<int32_t>();
    int32_t* num_tokens_post_pad_ptr = num_tokens_post_pad.data_ptr<int32_t>();
    size_t num_tokens = topk_ids.numel();
    int32_t* token_cnts_buffer_ptr = token_cnts_buffer.data_ptr<int32_t>();
    int32_t* cumsum_buffer_ptr = cumsum_buffer.data_ptr<int32_t>();
    int tokens_per_block = CEILDIV(topk_ids.sizes()[0], num_blocks) * topk_ids.sizes()[1];
    int tokens_per_thread = CEILDIV(tokens_per_block, block_threads);
    int K = topk_ids.sizes()[1];

    void* kernelArgs[] = {&topk_ids_ptr,      &sorted_token_ids_ptr, &experts_ids_ptr,   &num_tokens_post_pad_ptr,
                          &num_experts,       &block_size,           &num_tokens,        &token_cnts_buffer_ptr,
                          &cumsum_buffer_ptr, &tokens_per_block,     &tokens_per_thread, &K};

    cudaLaunchCooperativeKernel((void*)kernel, num_blocks, block_threads, kernelArgs);
  });
}
