// Adapted from https://github.com/vllm-project/vllm/blob/v0.6.5/csrc/moe/moe_align_sum_kernels.cu

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/extension.h>

#include <THC/THCAtomics.cuh>

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#endif

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
#endif

#define CEILDIV(x, y) (((x) + (y)-1) / (y))

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

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(scalar_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
                                            int32_t* expert_ids, int32_t* total_tokens_post_pad, int32_t num_experts,
                                            int32_t block_size, size_t numel, int32_t* tokens_cnts, int32_t* cumsum) {
  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[index(num_experts, threadIdx.x + 1, i)] = 0;
  }

  /**
   * In the first step we compute token_cnts[thread_index + 1][expert_index],
   * which counts how many tokens in the token shard of thread_index are
   * assigned to expert expert_index.
   */
  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    ++tokens_cnts[index(num_experts, threadIdx.x + 1, topk_ids[i])];
  }

  __syncthreads();

  // For each expert we accumulate the token counts from the different threads.
  if (threadIdx.x < num_experts) {
    tokens_cnts[index(num_experts, 0, threadIdx.x)] = 0;
    for (int i = 1; i <= blockDim.x; ++i) {
      tokens_cnts[index(num_experts, i, threadIdx.x)] += tokens_cnts[index(num_experts, i - 1, threadIdx.x)];
    }
  }

  __syncthreads();

  // We accumulate the token counts of all experts in thread 0.
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] + CEILDIV(tokens_cnts[index(num_experts, blockDim.x, i - 1)], block_size) * block_size;
    }
    *total_tokens_post_pad = cumsum[num_experts];
  }

  __syncthreads();

  /**
   * For each expert, each thread processes the tokens of the corresponding
   * blocks and stores the corresponding expert_id for each block.
   */
  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }

  /**
   * Each thread processes a token shard, calculating the index of each token
   * after sorting by expert number. Given the example topk_ids =
   * [0,1,2,1,2,3,0,3,4] and block_size = 4, then the output would be [0, 6, *,
   * *, 1, 3, *, *, 2, 4, *, *, 5, 7, *, *, 8, *, *, *], where * represents a
   * padding value(preset in python).
   */
  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int32_t expert_id = topk_ids[i];
    /** The cumsum[expert_id] stores the starting index of the tokens that the
     * expert with expert_id needs to process, and
     * tokens_cnts[threadIdx.x][expert_id] stores the indices of the tokens
     * processed by the expert with expert_id within the current thread's token
     * shard.
     */
    int32_t rank_post_pad = tokens_cnts[index(num_experts, threadIdx.x, expert_id)] + cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[index(num_experts, threadIdx.x, expert_id)];
  }
}

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
                          torch::Tensor sorted_token_ids, torch::Tensor experts_ids, torch::Tensor num_tokens_post_pad,
                          torch::Tensor token_cnts_buffer, torch::Tensor cumsum_buffer) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
    // calc needed amount of shared mem for `tokens_cnts` and `cumsum`
    // tensors
    const int32_t num_thread = max((int32_t)num_experts, WARP_SIZE);

    auto kernel = moe_align_block_size_kernel<scalar_t>;
    kernel<<<1, num_thread, 0, stream>>>(topk_ids.data_ptr<scalar_t>(), sorted_token_ids.data_ptr<int32_t>(),
                                         experts_ids.data_ptr<int32_t>(), num_tokens_post_pad.data_ptr<int32_t>(),
                                         num_experts, block_size, topk_ids.numel(),
                                         token_cnts_buffer.data_ptr<int32_t>(), cumsum_buffer.data_ptr<int32_t>());
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_align_block_size", &moe_align_block_size, "MOE Align Block Size (CUDA)");
}
