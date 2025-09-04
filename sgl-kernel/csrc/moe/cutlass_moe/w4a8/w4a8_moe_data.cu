#include <c10/cuda/CUDAGuard.h>
#include <cudaTypedefs.h>
#include <torch/all.h>

#include <iostream>

constexpr uint64_t THREADS_PER_EXPERT = 512;

__global__ void compute_problem_sizes_w4a8(
    const int32_t* __restrict__ topk_ids,
    int32_t* problem_sizes1,
    int32_t* problem_sizes2,
    int32_t* atomic_buffer,
    const int topk_length,
    const int n,
    const int k) {
  int expert_id = blockIdx.x;
  if (threadIdx.x == 0) {
    atomic_buffer[expert_id] = 0;
  }

  int occurrences = 0;
  for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
    occurrences += (topk_ids[i] == expert_id);
  }
  atomicAdd(&atomic_buffer[expert_id], occurrences);
  __syncthreads();

  if (threadIdx.x == 0) {
    int final_occurrences = atomic_buffer[expert_id];
    problem_sizes1[expert_id * 3] = 2 * n;
    problem_sizes1[expert_id * 3 + 1] = final_occurrences;
    problem_sizes1[expert_id * 3 + 2] = k;
    problem_sizes2[expert_id * 3] = k;
    problem_sizes2[expert_id * 3 + 1] = final_occurrences;
    problem_sizes2[expert_id * 3 + 2] = n;
  }
}

__global__ void compute_expert_offsets_w4a8_default(
    const int32_t* __restrict__ count_buffer, int32_t* expert_offsets, const int num_experts) {
  int32_t tot_offset = 0;
  expert_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    tot_offset += count_buffer[i];
    expert_offsets[i + 1] = tot_offset;
  }
}

constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCK_SIZE = 1024;

__inline__ __device__ int32_t warp_prefix_sum(int32_t val, int lane, int n = WARP_SIZE, unsigned mask = 0xffffffff) {
  for (int offset = 1; offset < n; offset <<= 1) {
    int32_t x = __shfl_up_sync(mask, val, offset, WARP_SIZE);
    if (lane >= offset) {
      val += x;
    }
  }
  return val;
}

__device__ void cumsum(const int32_t* __restrict__ input, int32_t* __restrict__ output, int n, int stride) {
  __shared__ int32_t warp_sums[MAX_BLOCK_SIZE / WARP_SIZE];

  int tid = threadIdx.x;
  int lane = threadIdx.x % WARP_SIZE;     // warp 内的 lane id
  int warp_id = threadIdx.x / WARP_SIZE;  // 当前 warp id

  // prefix sum within a warp
  int32_t val = tid < n ? input[tid * stride] : 0;
  val = warp_prefix_sum(val, lane);

  // write warp sum to shared memory
  if (lane == WARP_SIZE - 1) {
    warp_sums[warp_id] = val;
  }
  __syncthreads();

  // prefix sum of warp sums
  if (warp_id == 0) {
    int num_warps = blockDim.x / WARP_SIZE;
    // it does not matter if lane >= num_warps, since we do not use the result for these lanes
    int warp_total = warp_sums[lane];
    warp_total = warp_prefix_sum(warp_total, lane, num_warps);
    warp_sums[lane] = warp_total;
  }
  __syncthreads();

  // finalize the result
  if (warp_id > 0) {
    val += warp_sums[warp_id - 1];
  }

  if (tid < n) {
    output[tid] = val;
  }
}

__global__ void compute_expert_offsets_w4a8_parallel_kernel(
    const int32_t* __restrict__ input, int32_t* __restrict__ output, int n, int stride) {
  if (threadIdx.x == 0) {
    output[0] = 0;
  }
  cumsum(input, output + 1, n, stride);
}

void compute_expert_offsets_w4a8_parallel(
    cudaStream_t stream, const int32_t* input, int32_t* output, int n, int stride = 1, int off = 0) {
  int block_size = max((n + WARP_SIZE - 1) / WARP_SIZE, 1) * WARP_SIZE;
  compute_expert_offsets_w4a8_parallel_kernel<<<1, block_size, 0, stream>>>(input + off, output, n, stride);
}

void get_cutlass_w4a8_moe_mm_data_caller(
    const torch::Tensor& topk_ids,
    torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1,
    torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation,
    torch::Tensor& output_permutation,
    const int64_t num_experts,
    const int64_t n,
    const int64_t k) {
  auto stream = at::cuda::getCurrentCUDAStream(topk_ids.device().index());
  auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
  torch::Tensor atomic_buffer = torch::empty(num_experts, options_int32);

  int num_threads = min(THREADS_PER_EXPERT, topk_ids.numel());
  compute_problem_sizes_w4a8<<<num_experts, num_threads, 0, stream>>>(
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(problem_sizes1.data_ptr()),
      static_cast<int32_t*>(problem_sizes2.data_ptr()),
      static_cast<int32_t*>(atomic_buffer.data_ptr()),
      topk_ids.numel(),
      n,
      k);

  if (num_experts <= 1024) {
    compute_expert_offsets_w4a8_parallel(
        stream,
        static_cast<const int32_t*>(atomic_buffer.data_ptr()),
        static_cast<int32_t*>(expert_offsets.data_ptr()),
        num_experts);
  } else {
    compute_expert_offsets_w4a8_default<<<1, 1, 0, stream>>>(
        static_cast<const int32_t*>(atomic_buffer.data_ptr()),
        static_cast<int32_t*>(expert_offsets.data_ptr()),
        num_experts);
  }
}
