// Adapted from https://github.com/vllm-project/vllm/blob/v0.6.5/csrc/moe/moe_align_sum_kernels.cu

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <THC/THCAtomics.cuh>

#include "utils.hpp"

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#endif // USE_ROCM
#include <iostream> // TODO (yiakwy) : remove
#ifndef USE_ROCM
#define WARP_SIZE 32
#else
#define WARP_SIZE warpSize
#endif

#ifndef USE_ROCM
#define DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
  cudaFuncSetAttribute(FUNC, cudaFuncAttributeMaxDynamicSharedMemorySize, VAL)
#else
#define DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
  hipFuncSetAttribute(FUNC, hipFuncAttributeMaxDynamicSharedMemorySize, VAL)

  // NOTE(yiakwy) : func alias
  template <typename... Args>
  static __inline__ __host__ __device__
  auto cudaLaunchCooperativeKernel(Args&&... args) -> decltype(cudaLaunchCooperativeKernel(std::forward<Args>(args)...)) {
    return cudaLaunchCooperativeKernel(std::forward<Args>(args)...);
  }
#endif

#define CEILDIV(x, y) (((x) + (y)-1) / (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#define DISPATCH_CASE_INTEGRAL_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))

__device__ __forceinline__ int32_t index(int32_t total_col, int32_t row, int32_t col) {
  // don't worry about overflow because num_experts is relatively small
  return row * total_col + col;
}

#define OFFSETS_PAD 1

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(scalar_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
                                            int32_t* expert_ids, int32_t* total_tokens_post_pad, int32_t num_experts,
                                            int32_t block_size, size_t numel, int32_t* cumsum) {
  __shared__ int32_t shared_counts[32][8];
  // NOTE (yiakwy) : this assumes num_experts <= 256
  __shared__ int32_t local_offsets[256+OFFSETS_PAD];
  __shared__ int32_t local_offsets_buf[16];

  const int tid = threadIdx.x;
  const int experts_per_warp = 8;

  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  int *shared_counts_base = &(shared_counts[0][0]);
  if (threadIdx.x < 256) {
    *(shared_counts_base + threadIdx.x) = 0;
  }

  // NOTE (yiakwy) : this warp of threads may access other warp of threads based on the value of expert id fetched
  __syncthreads();

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int expert_id = topk_ids[i];
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    atomicAdd(&shared_counts[warp_idx][expert_offset], 1);
  }

  __syncthreads();

#define kElementsPerThr    16

  {

    int active_threads = CEILDIV(num_experts, kElementsPerThr);
    if (tid == 0) {
      local_offsets[0] = 0;
    }
    if (tid < active_threads - 1) { // NOTE(yaikwy) : algo here assumes single block execution

      // NOTE (yiakwy) : loop body, a simple reduction prototype, useful for workload with the number of experts upto 256
      // NOTE (yiakwy) : each thread process 16 expert, then only 2 steps needed

      // NOTE (yiakwy) : step 1, loop body
      for (int i=tid*kElementsPerThr+1; i < (tid + 1)*kElementsPerThr+1; ++i) {
        int warp_idx = (i-1) / experts_per_warp;
        int expert_offset = (i-1) % experts_per_warp;

        int expert_count = shared_counts[warp_idx][expert_offset];

        int last_val = (i-1) % kElementsPerThr == 0 ? 0 : local_offsets[i-1];
        local_offsets[i] = last_val + CEILDIV(expert_count, block_size) * block_size;
      }

      local_offsets_buf[tid] = local_offsets[(tid + 1)*kElementsPerThr];

    }

    // NOTE (yiakwy) : step 1, unroll loop tail
    if (tid == active_threads - 1) {
      #pragma unroll
      for (int i=tid * kElementsPerThr+1; i < num_experts+1; ++i) {
        int warp_idx = (i-1) / experts_per_warp;
        int expert_offset = (i-1) % experts_per_warp;

        int expert_count = shared_counts[warp_idx][expert_offset];

        int last_val = (i-1) % kElementsPerThr == 0 ? 0 : local_offsets[i-1];
        local_offsets[i] = last_val + CEILDIV(expert_count, block_size) * block_size;
      }

      local_offsets_buf[tid] = local_offsets[num_experts];

    }

    __syncthreads();

    // NOTE (yiakwy) : step 2, loop body
    if (tid < active_threads - 1 && tid > 0) {
      int offset = 0;
      for (int j=0; j < tid; ++j) {
        offset += local_offsets_buf[j];
      }

      for (int i=tid*kElementsPerThr+1; i < (tid + 1)*kElementsPerThr+1; ++i) {
        local_offsets[i] += offset;
      }
    }

    // NOTE (yiakwy) : step 2, loop tail
    if (tid == active_threads - 1) {
      int offset = 0;
      for (int j=0; j < tid; ++j) {
        offset += local_offsets_buf[j];
      }
      for (int i=tid*kElementsPerThr+1; i < num_experts+1; ++i) {
        local_offsets[i] += offset;
      }
    }

  } // code block of computing cumsum

  __syncthreads();

#define kElementsPerThr    16
#define kElementsPerAccess 4

  {

    int active_threads = CEILDIV(num_experts+1, kElementsPerThr);
    if (tid < active_threads - 1) {

      // NOTE(yiakwy) : loop body useful for workload with the number of experts upto 256
      for (int i=tid * kElementsPerThr ; i < (tid + 1) * kElementsPerThr; i += kElementsPerAccess) {
        *(int4 *)(cumsum + i) = *(int4 *)(local_offsets + i);
      }
    }

    if (tid == active_threads - 1) {
      // NOTE(yiakwy) : unroll loop tail
      #pragma unroll
      for (int i=tid * kElementsPerThr; i < num_experts+1; i++) {
        *(cumsum + i) = *(local_offsets + i);
      }
    }

    if (tid == active_threads) {
      *total_tokens_post_pad = local_offsets[num_experts];
    }

  } // code block of storing to cumsum

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = local_offsets[threadIdx.x]; i < local_offsets[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }

  __syncthreads();

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int32_t expert_id = topk_ids[i];
    int32_t rank_post_pad = atomicAdd(&local_offsets[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}


#define checkCudaErrors(val) check ( (val), #val, __FILE__, __LINE__ )
template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}


template <typename scalar_t>
__global__ void moe_align_block_size_multiblocks_kernel(scalar_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
                                            int32_t* expert_ids, int32_t* total_tokens_post_pad, int32_t num_experts,
                                            int32_t block_size, size_t numel, int32_t* tokens_cnts, int32_t* cumsum, const int tokens_per_block, const int tokens_per_thread, const int K) {
  __shared__ int32_t shared_counts[32][8];
  // NOTE (yiakwy) : this assumes num_experts <= 256
  __shared__ int32_t local_offsets[256+OFFSETS_PAD];
  __shared__ int32_t local_offsets_buf[16];

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  // NOTE (yiakwy) : we use local warp_id, lane_id for warp aggregation of shared_counts
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int experts_per_warp = 8;

  // NOTE (yiakwy) : used to synchronize blocks,
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  // NOTE (yiakwy) : not all threads participate in
  const size_t start_idx = tokens_per_block * blockIdx.x + tokens_per_thread * threadIdx.x;
  const size_t end_idx = start_idx + tokens_per_thread;

  int *shared_counts_base = &(shared_counts[0][0]);
  if (threadIdx.x < 256) {
    *(shared_counts_base + threadIdx.x) = 0;
  }

  // NOTE (yiakwy) : this warp of threads may access other warp of threads based on the value of expert id fetched
  __syncthreads();

  // NOTE (yiakwy) : since each block processes less tokens, less possibility for threads acces these localtions ([0][0], [4][0], [8][0], ...) simutaneously
  if (threadIdx.x < tokens_per_block) {
    for (int i = start_idx; i < MIN(numel, end_idx); ++i) {
      int expert_id = topk_ids[i];
      int warp_idx = expert_id / experts_per_warp;
      int expert_offset = expert_id % experts_per_warp;
      atomicAdd(&shared_counts[warp_idx][expert_offset], 1);
    }
  }

  __syncthreads();

#define kElementsPerThr    16

  {

    int active_threads = CEILDIV(num_experts, kElementsPerThr);
    if (threadIdx.x == 0) {
      local_offsets[0] = 0;
    }
    if (threadIdx.x < active_threads - 1) { // NOTE(yaikwy) : algo here assumes single block execution

      // NOTE (yiakwy) : loop body, a simple reduction prototype, useful for workload with the number of experts upto 256
      // NOTE (yiakwy) : each thread process 16 expert, then only 2 steps needed

      // NOTE (yiakwy) : step 1, loop body
      for (int i=threadIdx.x*kElementsPerThr+1; i < (threadIdx.x + 1)*kElementsPerThr+1; ++i) {
        int warp_idx = (i-1) / experts_per_warp;
        int expert_offset = (i-1) % experts_per_warp;

        int expert_count = shared_counts[warp_idx][expert_offset];

        int last_val = (i-1) % kElementsPerThr == 0 ? 0 : local_offsets[i-1];
        // local_offsets[i] = last_val + CEILDIV(expert_count, block_size) * block_size;
        local_offsets[i] = last_val + expert_count;
      }

      local_offsets_buf[threadIdx.x] = local_offsets[(threadIdx.x + 1)*kElementsPerThr];

    }

    // NOTE (yiakwy) : step 1, unroll loop tail
    if (threadIdx.x == active_threads - 1) {
      #pragma unroll
      for (int i=threadIdx.x * kElementsPerThr+1; i < num_experts+1; ++i) {
        int warp_idx = (i-1) / experts_per_warp;
        int expert_offset = (i-1) % experts_per_warp;

        int expert_count = shared_counts[warp_idx][expert_offset];

        int last_val = (i-1) % kElementsPerThr == 0 ? 0 : local_offsets[i-1];
        // local_offsets[i] = last_val + CEILDIV(expert_count, block_size) * block_size;
        local_offsets[i] = last_val + expert_count;
      }

      local_offsets_buf[threadIdx.x] = local_offsets[num_experts];

    }

    __syncthreads();

    // NOTE (yiakwy) : step 2, loop body
    if (threadIdx.x < active_threads - 1 && threadIdx.x > 0) {
      int offset = 0;
      for (int j=0; j < threadIdx.x; ++j) {
        offset += local_offsets_buf[j];
      }

      for (int i=threadIdx.x*kElementsPerThr+1; i < (threadIdx.x + 1)*kElementsPerThr+1; ++i) {
        local_offsets[i] += offset;
      }
    }

    // NOTE (yiakwy) : step 2, loop tail
    if (threadIdx.x == active_threads - 1) {
      int offset = 0;
      for (int j=0; j < threadIdx.x; ++j) {
        offset += local_offsets_buf[j];
      }
      for (int i=threadIdx.x*kElementsPerThr+1; i < num_experts+1; ++i) {
        local_offsets[i] += offset;
      }
    }

  } // code block of computing local unaligned cumsum

  __syncthreads();

#ifdef DEBUG
  if (threadIdx.x == 0) {
    printf("[Block#%d] local_offsets[1:num_experts+1] = [%d, %d, ..., %d, %d]\n", blockIdx.x, local_offsets[1], local_offsets[2], local_offsets[num_experts-1], local_offsets[num_experts]);
  }
  __syncthreads();
#endif

  {
    if (tid < num_experts) {
      *(tokens_cnts + tid) = 0;
    }
    if (threadIdx.x < num_experts) {
      *(tokens_cnts + (blockIdx.x + 1) * num_experts + threadIdx.x) = *(local_offsets + threadIdx.x + 1);
      *(local_offsets + threadIdx.x + 1) = 0;
    } else if (threadIdx.x < 256) {
      *(local_offsets + threadIdx.x + 1) = 0;
    }
    __threadfence_system();
    grid.sync();

#define BLOCK_SIZE_M      16
#define BLOCK_SIZE_N      16

#define kElementsPerAccess 4
#define kWarpsToLoad       2

    int total_fragments = CEILDIV(num_experts, BLOCK_SIZE_N);
    int fragments_per_block = CEILDIV(total_fragments, gridDim.x);

    if (blockIdx.x * fragments_per_block < total_fragments) {
      int *tokens_cnts_ptr = &(tokens_cnts[0]);

      for (int i=0; i < gridDim.x; i += BLOCK_SIZE_M) {
        if (warp_id < kWarpsToLoad * fragments_per_block) { // NOTE (yiakwy) : kWarpsToLoad warps (CUDA) for loading 16x16 fragment
          const int kNumThrPerRow = WARP_SIZE / BLOCK_SIZE_N;
          int sRow = lane_id / kNumThrPerRow ; // sRow=7, lane_id=14
          int sCol = lane_id % kNumThrPerRow * kElementsPerAccess + warp_id * (kNumThrPerRow * kElementsPerAccess);

          int gRow = i * BLOCK_SIZE_M + sRow;
          int gCol = blockIdx.x * fragments_per_block * BLOCK_SIZE_N + sCol;

          if (gRow < num_experts && gCol < num_experts) { // NOTE (yiakwy) : defensive guard
            // NOTE (yiakwy) : useful to coalesce memory transaction when loading a column of data
            int4 *tokens_cnts_4i_ptr = (int4 *)(tokens_cnts_ptr + (gRow+1) * num_experts + gCol);
            int4 *shared_counts_4i_ptr = (int4 *)(shared_counts_base + sRow * BLOCK_SIZE_N + sCol);

            *shared_counts_4i_ptr = *tokens_cnts_4i_ptr;
          }
        }
        __syncthreads();

        if (warp_id < kWarpsToLoad * fragments_per_block && warp_id % kWarpsToLoad == 0) { // NOTE (yiakwy) : 1 warp (CUDA) for processing 16x16 fragment
          for (int k=0; k < BLOCK_SIZE_N; k+=2) { // NOTE (yiakwy) : this simple arangement enables thread 0 accessing addresses limited to bank 0, thread 16 accesses limited to bank 16, etc in CUDA.
            int sRow = lane_id / BLOCK_SIZE_N + k;
            int sCol = lane_id % BLOCK_SIZE_N;

            int gCol = blockIdx.x * fragments_per_block * BLOCK_SIZE_N + sCol;
            if (gCol < num_experts) { // NOTE (yiakwy) : defensive guard
              atomicAdd(tokens_cnts_ptr + gCol, *(shared_counts_base + sRow * BLOCK_SIZE_N + sCol));
            }
          }
        }
        __syncthreads();

      }
    }
    __threadfence_system();
    grid.sync();

    // NOTE (yiakwy) : sync unaigned offsets in block#0
    if (tid < num_experts) {
      *(local_offsets + tid + 1) = *(tokens_cnts + tid);
    }
     __syncthreads();

#ifdef DEBUG
    if (tid == 0) {
      printf("[Block#%d] unaligned global offsets[1:num_experts+1] = [%d, %d, ..., %d, %d]\n", blockIdx.x, local_offsets[1], local_offsets[2], local_offsets[num_experts-1], local_offsets[num_experts]);
    }
    __syncthreads();
#endif

  } // code block of computing global cumsum

  __syncthreads();

  // NOTE (yiakwy) : convert unaligned cumsum to aligned cumsum
  // TODO (yiakwy) : distribute work to other threads
  if (tid == 0) {
    for (int i=num_experts; i > 0; i--) {
      local_offsets[i] = local_offsets[i] - local_offsets[i-1];
    }
    for (int i=1; i < num_experts+1; i++) {
      local_offsets[i] = local_offsets[i-1] + CEILDIV(local_offsets[i], block_size) * block_size;
    }
  }
  __syncthreads();

#ifdef DEBUG
  if (tid == 0) {
    printf("[Block#%d] aligned global offsets[1:num_experts+1] = [%d, %d, ..., %d, %d]\n", blockIdx.x, local_offsets[1], local_offsets[2], local_offsets[num_experts-1], local_offsets[num_experts]);
  }
  __syncthreads();
#endif

#define kElementsPerThr    16
#define kElementsPerAccess 4

  {

    int active_threads = CEILDIV(num_experts+1, kElementsPerThr);
    if (tid < active_threads - 1) {

      // NOTE(yiakwy) : loop body useful for workload with the number of experts upto 256
      for (int i=tid * kElementsPerThr ; i < (tid + 1) * kElementsPerThr; i += kElementsPerAccess) {
        *(int4 *)(cumsum + i) = *(int4 *)(local_offsets + i);
      }
    }

    if (tid == active_threads - 1) {
      // NOTE(yiakwy) : unroll loop tail
      #pragma unroll
      for (int i=tid * kElementsPerThr; i < num_experts+1; i++) {
        *(cumsum + i) = *(local_offsets + i);
      }
    }

    if (tid == active_threads) {
      *total_tokens_post_pad = local_offsets[num_experts];
    }

  } // code block of storing to cumsum

  __syncthreads();

  if (tid < num_experts) {
    for (int i = local_offsets[tid]; i < local_offsets[tid + 1]; i += block_size) {
      expert_ids[i / block_size] = tid;
    }
  }

  __syncthreads();

  // NOTE (yiakwy) : sync cumsum to each block
  if (blockIdx.x > 0) {
    int active_threads = CEILDIV(num_experts+1, kElementsPerThr);

    if (threadIdx.x < active_threads - 1) {
      for (int i=threadIdx.x * kElementsPerThr ; i < (threadIdx.x + 1) * kElementsPerThr; i += kElementsPerAccess) {
        *(int4 *)(local_offsets + i) = *(int4 *)(cumsum + i);
      }
    }

    if (threadIdx.x == active_threads - 1) {
      #pragma unroll
      for (int i=threadIdx.x * kElementsPerThr; i < num_experts+1; i++) {
        *(local_offsets + i) = *(cumsum + i);
      }
    }
  }

  __syncthreads();

  if (threadIdx.x < tokens_per_block) {
    for (int i = start_idx; i < MIN(numel, end_idx); ++i) {
      int32_t expert_id = topk_ids[i];
      int32_t rank_post_pad = atomicAdd(&local_offsets[expert_id], 1);
      sorted_token_ids[rank_post_pad] = i;
    }
  }
}


void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
                          torch::Tensor sorted_token_ids, torch::Tensor experts_ids, torch::Tensor num_tokens_post_pad,
                          torch::Tensor token_cnts_buffer, torch::Tensor cumsum_buffer) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
    if (false/*topk_ids.sizes()[0] < 16384 / 2*/) {
      auto kernel = moe_align_block_size_kernel<scalar_t>;
      // NOTE(yiakwy) : this assumes a single block execution, will be slow if too many tokens (>1024) feeded in
      kernel<<<1, 1024, 0, stream>>>(topk_ids.data_ptr<scalar_t>(), sorted_token_ids.data_ptr<int32_t>(),
                                    experts_ids.data_ptr<int32_t>(), num_tokens_post_pad.data_ptr<int32_t>(),
                                    num_experts, block_size, topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>());
      // printf("%d blocks used.\n", 1);
    } else {
      auto kernel = moe_align_block_size_multiblocks_kernel<scalar_t>;
// NOTE (yiakwy) : reduce registers consumed
#define BLOCK_SIZE 512
      auto BLOCKS = MIN( CEILDIV(topk_ids.sizes()[0], BLOCK_SIZE), num_experts );

      int32_t tokens_per_block = CEILDIV(topk_ids.sizes()[0], BLOCKS) * topk_ids.sizes()[1];
      int32_t tokens_per_thread = CEILDIV(tokens_per_block, BLOCK_SIZE);

      // printf("%d BLOCKS used. %d tokens per block. %d tokens per thread\n", BLOCKS, tokens_per_block, tokens_per_thread);
      /*
      kernel<<<BLOCKS, BLOCK_SIZE, 0, stream>>>(topk_ids.data_ptr<scalar_t>(), sorted_token_ids.data_ptr<int32_t>(),
                                    experts_ids.data_ptr<int32_t>(), num_tokens_post_pad.data_ptr<int32_t>(),
                                    num_experts, block_size, topk_ids.numel(), token_cnts_buffer.data_ptr<int32_t>(), cumsum_buffer.data_ptr<int32_t>(), tokens_per_block, tokens_per_thread, topk_ids.sizes()[1]);
       */
      // NOTE (yiakwy) : remove const decorator for kernel args
      scalar_t* topk_ids_ptr = topk_ids.data_ptr<scalar_t>();
      int32_t* sorted_token_ids_ptr = sorted_token_ids.data_ptr<int32_t>();
      int32_t* experts_ids_ptr = experts_ids.data_ptr<int32_t>();
      int32_t* num_tokens_post_pad_ptr = num_tokens_post_pad.data_ptr<int32_t>();
      size_t num_tokens = topk_ids.numel();
      int32_t* token_cnts_buffer_ptr = token_cnts_buffer.data_ptr<int32_t>();
      int32_t* cumsum_buffer_ptr = cumsum_buffer.data_ptr<int32_t>();
      int K = topk_ids.sizes()[1];

      void *kernelArgs[] = { &topk_ids_ptr, &sorted_token_ids_ptr,
        &experts_ids_ptr, &num_tokens_post_pad_ptr,
        &num_experts, &block_size, &num_tokens, &token_cnts_buffer_ptr, &cumsum_buffer_ptr, &tokens_per_block, &tokens_per_thread, &K
      };
      cudaLaunchCooperativeKernel((void*)kernel, BLOCKS, BLOCK_SIZE, kernelArgs);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  });
}
