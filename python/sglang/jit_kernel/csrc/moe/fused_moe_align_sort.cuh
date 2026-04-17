#include <sgl_kernel/tensor.h>   // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>    // For RuntimeCheck, div_ceil
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

// ----------------------------------------------------------------
// Fused MoE Align Block Size + Count-and-Sort kernel.
//
// Replaces two separate kernels:
//   1. moe_align_block_size_kernel (prefix sum + expert_ids)
//   2. count_and_sort_expert_tokens_kernel (atomic scatter)
//
// This single-kernel version uses per-thread local counting to avoid
// atomics, matching the moe_align_block_size_small_batch_expert_kernel
// pattern from sgl-kernel but as a JIT kernel (no sgl-kernel build needed).
//
// Supports up to 256 experts. Shared memory ~65KB for 128 experts.
// ----------------------------------------------------------------

constexpr int kFillThreads = 256;

template <typename IdType>
__global__ void fused_moe_align_sort_kernel(
    const IdType* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    int32_t numel,
    int32_t max_num_tokens_padded) {

  // First kFillThreads threads: fill sorted_token_ids with sentinel
  if (threadIdx.x < kFillThreads) {
    for (int32_t it = threadIdx.x; it < max_num_tokens_padded; it += kFillThreads) {
      sorted_token_ids[it] = numel;
    }
    __syncthreads();
    __syncthreads();
    __syncthreads();
    return;
  }

  const int32_t tid = threadIdx.x - kFillThreads;
  const int32_t stride = blockDim.x - kFillThreads;

  // Shared memory layout:
  //   cumsum:     [num_experts + 1]
  //   tokens_cnts: [(stride + 1) * num_experts]
  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;
  int32_t* tokens_cnts = shared_mem + num_experts + 1;

  // Initialize per-thread counts to 0
  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(tid + 1) * num_experts + i] = 0;
  }

  // Phase 1: Count tokens per expert per thread
  for (int32_t i = tid; i < numel; i += stride) {
    int32_t expert_id = static_cast<int32_t>(topk_ids[i]) + 1;
    ++tokens_cnts[(tid + 1) * num_experts + expert_id];
  }

  __syncthreads();

  // Phase 2: Prefix sum per expert across threads (loop for workers < experts)
  for (int32_t e = tid; e < num_experts; e += stride) {
    tokens_cnts[e] = 0;
    for (int i = 1; i <= stride; ++i) {
      tokens_cnts[i * num_experts + e] += tokens_cnts[(i - 1) * num_experts + e];
    }
  }

  __syncthreads();

  // Phase 3: Compute cumulative padded offsets and total
  if (tid == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      int32_t count = tokens_cnts[stride * num_experts + i - 1];
      int32_t padded = ((count + block_size - 1) / block_size) * block_size;
      cumsum[i] = cumsum[i - 1] + padded;
    }
    *total_tokens_post_pad = cumsum[num_experts];
  }

  __syncthreads();

  // Phase 4: Fill expert_ids (loop for workers < experts)
  for (int32_t e = tid; e < num_experts; e += stride) {
    for (int i = cumsum[e]; i < cumsum[e + 1]; i += block_size) {
      expert_ids[i / block_size] = e - 1;
    }
  }

  // Phase 5: Scatter sorted token IDs using per-thread prefix counts
  for (int32_t i = tid; i < numel; i += stride) {
    int32_t expert_id = static_cast<int32_t>(topk_ids[i]) + 1;
    int32_t rank_post_pad = tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[tid * num_experts + expert_id];
  }
}

template <typename IdType>
void fused_moe_align_sort(
    tvm::ffi::TensorView topk_ids,
    tvm::ffi::TensorView sorted_token_ids,
    tvm::ffi::TensorView expert_ids,
    tvm::ffi::TensorView num_tokens_post_pad,
    int64_t num_experts,
    int64_t block_size) {
  using namespace host;

  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  auto N = SymbolicSize{"numel"};
  TensorMatcher({N}).with_dtype<IdType>().with_device(device).verify(topk_ids);

  const int32_t numel = static_cast<int32_t>(N.unwrap());
  const int32_t max_num_tokens_padded = sorted_token_ids.shape()[0];
  const int32_t ne = static_cast<int32_t>(num_experts);
  const int32_t bs = static_cast<int32_t>(block_size);

  // Worker threads (non-fill): must be >= num_experts for phase 2/4
  // Cap at 64 to keep shared memory under 48KB for CUDA graph compatibility
  // smem = ((workers+1)*ne + ne+1) * 4 bytes; at workers=64, ne=129: ~34KB
  const int32_t worker_threads = std::min(std::max(ne, 32), 64);

  // Shared memory: cumsum [ne+1] + tokens_cnts [(worker_threads+1) * ne]
  const int32_t smem_size = ((worker_threads + 1) * ne + (ne + 1)) * sizeof(int32_t);
  RuntimeCheck(smem_size <= 100 * 1024,
      "Shared memory ", smem_size, " exceeds 100KB limit for ", ne, " experts");

  const int32_t total_threads = kFillThreads + worker_threads;

  // Configure dynamic shared memory once so graph capture does not hit
  // a per-launch host-side cudaFuncSetAttribute call.
  auto kernel = fused_moe_align_sort_kernel<IdType>;
  static bool attr_initialized = false;
  if (!attr_initialized) {
    cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        100 * 1024);
    attr_initialized = true;
  }

  LaunchKernel(1, total_threads, topk_ids.device(), smem_size)(
      kernel,
      static_cast<const IdType*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(sorted_token_ids.data_ptr()),
      static_cast<int32_t*>(expert_ids.data_ptr()),
      static_cast<int32_t*>(num_tokens_post_pad.data_ptr()),
      ne, bs, numel, max_num_tokens_padded);
}

}  // namespace
