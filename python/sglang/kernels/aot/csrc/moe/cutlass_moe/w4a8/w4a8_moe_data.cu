#include <c10/cuda/CUDAGuard.h>
#include <cudaTypedefs.h>
#include <torch/all.h>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

template <int BLOCK_SIZE>
__global__ void compute_problem_sizes_w4a8(
    const int32_t* __restrict__ topk_ids,
    int32_t* problem_sizes1,
    int32_t* problem_sizes2,
    const int topk_length,
    const int n,
    const int k) {
  int expert_id = blockIdx.x;

  int occurrences = 0;
  // Optimized: vectorized memory access using int4 for better memory bandwidth
  // Process vectorized chunks first
  bool aligned = (reinterpret_cast<uintptr_t>(topk_ids) % 16 == 0);

  if (aligned) {
    const int4* vec_ptr = reinterpret_cast<const int4*>(topk_ids);
    int vec_length = topk_length / 4;

    for (int i = threadIdx.x; i < vec_length; i += BLOCK_SIZE) {
      int4 vec_data = vec_ptr[i];
      occurrences +=
          (vec_data.x == expert_id) + (vec_data.y == expert_id) + (vec_data.z == expert_id) + (vec_data.w == expert_id);
    }

    for (int i = vec_length * 4 + threadIdx.x; i < topk_length; i += BLOCK_SIZE) {
      occurrences += (topk_ids[i] == expert_id);
    }
  } else {
    for (int i = threadIdx.x; i < topk_length; i += BLOCK_SIZE) {
      occurrences += (topk_ids[i] == expert_id);
    }
  }

  using BlockReduce = cub::BlockReduce<int, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int final_occurrences = BlockReduce(temp_storage).Sum(occurrences);

  if (threadIdx.x == 0) {
    problem_sizes1[expert_id * 3] = 2 * n;
    problem_sizes1[expert_id * 3 + 1] = final_occurrences;
    problem_sizes1[expert_id * 3 + 2] = k;
    problem_sizes2[expert_id * 3] = k;
    problem_sizes2[expert_id * 3 + 1] = final_occurrences;
    problem_sizes2[expert_id * 3 + 2] = n;
  }
}

template <int BLOCK_SIZE>
__device__ void
cumsum_block_scan(const int32_t* __restrict__ input, int32_t* __restrict__ output, int n, int input_stride) {
  using BlockScan = cub::BlockScan<int32_t, BLOCK_SIZE>;
  __shared__ typename BlockScan::TempStorage temp_scan_storage;
  __shared__ int32_t s_broadcast_val;

  int tid = threadIdx.x;
  int32_t base_prefix_sum = 0;
  const int num_chunks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (int chunk = 0; chunk < num_chunks; chunk++) {
    const int base_idx = chunk * BLOCK_SIZE;
    const int index = base_idx + tid;

    const int32_t val = (index < n) ? input[index * input_stride] : 0;
    int32_t local_prefix_sum;
    BlockScan(temp_scan_storage).InclusiveSum(val, local_prefix_sum);
    const int32_t prefix_sum = local_prefix_sum + base_prefix_sum;
    if (index < n) {
      output[index] = prefix_sum;
    }
    if (chunk < num_chunks - 1) {
      if (tid == BLOCK_SIZE - 1) {
        s_broadcast_val = prefix_sum;
      }
      __syncthreads();
      base_prefix_sum = s_broadcast_val;
    }
  }
}

template <int BLOCK_SIZE>
__global__ void compute_expert_offsets_w4a8_kernel(
    const int32_t* __restrict__ problem_sizes1, int32_t* __restrict__ expert_offsets, int n, int stride) {
  if (threadIdx.x == 0) {
    expert_offsets[0] = 0;
  }
  cumsum_block_scan<BLOCK_SIZE>(problem_sizes1, expert_offsets + 1, n, stride);
}

void compute_expert_offsets_w4a8(
    cudaStream_t stream, const int32_t* problem_sizes1, int32_t* expert_offsets, int n, int stride = 1, int off = 0) {
#define compute_expert_offsets_w4a8_call(BLOCK_SIZE) \
  compute_expert_offsets_w4a8_kernel<BLOCK_SIZE>     \
      <<<1, BLOCK_SIZE, 0, stream>>>(problem_sizes1 + off, expert_offsets, n, stride);

  if (n <= 32) {
    compute_expert_offsets_w4a8_call(32);
  } else if (n <= 64) {
    compute_expert_offsets_w4a8_call(64);
  } else if (n <= 128) {
    compute_expert_offsets_w4a8_call(128);
  } else if (n <= 256) {
    compute_expert_offsets_w4a8_call(256);
  } else if (n <= 512) {
    compute_expert_offsets_w4a8_call(512);
  } else {
    compute_expert_offsets_w4a8_call(1024);
  }
#undef compute_expert_offsets_w4a8_call
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

  constexpr uint64_t BLOCK_SIZE = 512;
  compute_problem_sizes_w4a8<BLOCK_SIZE><<<num_experts, BLOCK_SIZE, 0, stream>>>(
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(problem_sizes1.data_ptr()),
      static_cast<int32_t*>(problem_sizes2.data_ptr()),
      topk_ids.numel(),
      n,
      k);

  compute_expert_offsets_w4a8(
      stream,
      static_cast<const int32_t*>(problem_sizes1.data_ptr()),
      static_cast<int32_t*>(expert_offsets.data_ptr()),
      num_experts,
      3,
      1);
}
