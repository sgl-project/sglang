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

__global__ void compute_expert_offsets_and_starts_w4a8_kernel(
    const int32_t* __restrict__ problem_sizes1,
    int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ atomic_buffer,
    const int64_t num_experts) {
  int32_t tot_offset = 0;
  expert_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    atomic_buffer[i] = tot_offset;
    tot_offset += problem_sizes1[i * 3 + 1];
    expert_offsets[i + 1] = tot_offset;
  }
}

__global__ void compute_arg_sorts_w4a8(
    const int32_t* __restrict__ topk_ids,
    int32_t* input_permutation,
    int32_t* output_permutation,
    int32_t* atomic_buffer,
    const int64_t topk_length,
    const int64_t topk) {
  int expert_id = blockIdx.x;

  for (int i = threadIdx.x; i < topk_length; i += blockDim.x) {
    if (topk_ids[i] == expert_id) {
      int start = atomicAdd(&atomic_buffer[expert_id], 1);
      input_permutation[start] = i / topk;
      output_permutation[i] = start;
    }
  }
}

template <int BLOCK_SIZE, int MAX_EXPERTS>
__global__ void compute_tiny_moe_data_w4a8(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ problem_sizes1,
    int32_t* __restrict__ problem_sizes2,
    int32_t* __restrict__ input_permutation,
    int32_t* __restrict__ output_permutation,
    const int topk_length,
    const int topk,
    const int num_experts,
    const int n,
    const int k) {
  using BlockScan = cub::BlockScan<int32_t, BLOCK_SIZE>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ int32_t routed_experts[384];
  __shared__ int32_t write_offsets[MAX_EXPERTS];

  if (threadIdx.x < MAX_EXPERTS) {
    write_offsets[threadIdx.x] = 0;
  }
  __syncthreads();

  for (int index = threadIdx.x; index < topk_length; index += BLOCK_SIZE) {
    int const expert = topk_ids[index];
    routed_experts[index] = expert;
    atomicAdd(write_offsets + expert, 1);
  }
  __syncthreads();

  int32_t const count = threadIdx.x < num_experts ? write_offsets[threadIdx.x] : 0;
  int32_t offset = 0;
  BlockScan(scan_storage).ExclusiveSum(count, offset);

  if (threadIdx.x < num_experts) {
    int const expert = threadIdx.x;
    expert_offsets[expert] = offset;
    problem_sizes1[expert * 3] = 2 * n;
    problem_sizes1[expert * 3 + 1] = count;
    problem_sizes1[expert * 3 + 2] = k;
    problem_sizes2[expert * 3] = k;
    problem_sizes2[expert * 3 + 1] = count;
    problem_sizes2[expert * 3 + 2] = n;
    write_offsets[expert] = offset;
    if (expert + 1 == num_experts) {
      expert_offsets[num_experts] = offset + count;
    }
  }
  __syncthreads();

  for (int index = threadIdx.x; index < topk_length; index += BLOCK_SIZE) {
    int const expert = routed_experts[index];
    int const position = atomicAdd(write_offsets + expert, 1);
    input_permutation[position] = index / topk;
    output_permutation[index] = position;
  }
}

void get_cutlass_w4a8_moe_mm_data_with_permutation(
    const torch::Tensor& topk_ids,
    torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1,
    torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation,
    torch::Tensor& output_permutation,
    const int64_t num_experts,
    const int64_t n,
    const int64_t k) {
  TORCH_CHECK(topk_ids.dtype() == torch::kInt32);
  auto stream = at::cuda::getCurrentCUDAStream(topk_ids.device().index());

  constexpr uint64_t BLOCK_SIZE = 512;
  constexpr int TINY_BLOCK_SIZE = 256;
  constexpr int TINY_MAX_EXPERTS = 256;
  constexpr int TINY_MAX_ROUTES = 384;
  if (num_experts <= TINY_MAX_EXPERTS && topk_ids.numel() <= TINY_MAX_ROUTES) {
    compute_tiny_moe_data_w4a8<TINY_BLOCK_SIZE, TINY_MAX_EXPERTS><<<1, TINY_BLOCK_SIZE, 0, stream>>>(
        static_cast<const int32_t*>(topk_ids.data_ptr()),
        static_cast<int32_t*>(expert_offsets.data_ptr()),
        static_cast<int32_t*>(problem_sizes1.data_ptr()),
        static_cast<int32_t*>(problem_sizes2.data_ptr()),
        static_cast<int32_t*>(input_permutation.data_ptr()),
        static_cast<int32_t*>(output_permutation.data_ptr()),
        topk_ids.numel(),
        topk_ids.size(1),
        num_experts,
        n,
        k);
    return;
  }

  auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
  torch::Tensor atomic_buffer = torch::empty(num_experts, options_int32);
  compute_problem_sizes_w4a8<BLOCK_SIZE><<<num_experts, BLOCK_SIZE, 0, stream>>>(
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(problem_sizes1.data_ptr()),
      static_cast<int32_t*>(problem_sizes2.data_ptr()),
      topk_ids.numel(),
      n,
      k);
  compute_expert_offsets_and_starts_w4a8_kernel<<<1, 1, 0, stream>>>(
      static_cast<const int32_t*>(problem_sizes1.data_ptr()),
      static_cast<int32_t*>(expert_offsets.data_ptr()),
      static_cast<int32_t*>(atomic_buffer.data_ptr()),
      num_experts);
  compute_arg_sorts_w4a8<<<num_experts, BLOCK_SIZE, 0, stream>>>(
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(input_permutation.data_ptr()),
      static_cast<int32_t*>(output_permutation.data_ptr()),
      static_cast<int32_t*>(atomic_buffer.data_ptr()),
      topk_ids.numel(),
      topk_ids.size(1));
}

__global__ void compact_cutlass_w4a8_moe_mm_data_kernel(
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ problem_sizes1,
    const int32_t* __restrict__ problem_sizes2,
    int32_t* __restrict__ compact_expert_offsets,
    int32_t* __restrict__ compact_problem_sizes1,
    int32_t* __restrict__ compact_problem_sizes2,
    int32_t* __restrict__ compact_expert_ids,
    const int64_t num_experts,
    const int64_t max_groups) {
  int32_t out = 0;
  int32_t total = expert_offsets[num_experts];

  for (int32_t e = 0; e < num_experts; ++e) {
    if (out >= max_groups) {
      break;
    }
    int32_t m = problem_sizes1[e * 3 + 1];
    if (m <= 0) {
      continue;
    }

    compact_expert_offsets[out] = expert_offsets[e];
    compact_problem_sizes1[out * 3] = problem_sizes1[e * 3];
    compact_problem_sizes1[out * 3 + 1] = problem_sizes1[e * 3 + 1];
    compact_problem_sizes1[out * 3 + 2] = problem_sizes1[e * 3 + 2];
    compact_problem_sizes2[out * 3] = problem_sizes2[e * 3];
    compact_problem_sizes2[out * 3 + 1] = problem_sizes2[e * 3 + 1];
    compact_problem_sizes2[out * 3 + 2] = problem_sizes2[e * 3 + 2];
    compact_expert_ids[out] = e;
    ++out;
  }

  for (int32_t i = out; i < max_groups; ++i) {
    compact_expert_offsets[i] = total;
    compact_problem_sizes1[i * 3] = 0;
    compact_problem_sizes1[i * 3 + 1] = 0;
    compact_problem_sizes1[i * 3 + 2] = 0;
    compact_problem_sizes2[i * 3] = 0;
    compact_problem_sizes2[i * 3 + 1] = 0;
    compact_problem_sizes2[i * 3 + 2] = 0;
    compact_expert_ids[i] = 0;
  }
}

void compact_cutlass_w4a8_moe_mm_data(
    const torch::Tensor& expert_offsets,
    const torch::Tensor& problem_sizes1,
    const torch::Tensor& problem_sizes2,
    torch::Tensor& compact_expert_offsets,
    torch::Tensor& compact_problem_sizes1,
    torch::Tensor& compact_problem_sizes2,
    torch::Tensor& compact_expert_ids,
    const int64_t num_experts,
    const int64_t max_groups) {
  TORCH_CHECK(expert_offsets.dtype() == torch::kInt32, "expert_offsets must be int32");
  TORCH_CHECK(problem_sizes1.dtype() == torch::kInt32, "problem_sizes1 must be int32");
  TORCH_CHECK(problem_sizes2.dtype() == torch::kInt32, "problem_sizes2 must be int32");
  TORCH_CHECK(compact_expert_offsets.dtype() == torch::kInt32, "compact_expert_offsets must be int32");
  TORCH_CHECK(compact_problem_sizes1.dtype() == torch::kInt32, "compact_problem_sizes1 must be int32");
  TORCH_CHECK(compact_problem_sizes2.dtype() == torch::kInt32, "compact_problem_sizes2 must be int32");
  TORCH_CHECK(compact_expert_ids.dtype() == torch::kInt32, "compact_expert_ids must be int32");
  TORCH_CHECK(expert_offsets.numel() >= num_experts + 1, "expert_offsets must have num_experts + 1 entries");
  TORCH_CHECK(problem_sizes1.numel() >= num_experts * 3, "problem_sizes1 must have num_experts rows");
  TORCH_CHECK(problem_sizes2.numel() >= num_experts * 3, "problem_sizes2 must have num_experts rows");
  TORCH_CHECK(max_groups > 0 && max_groups <= num_experts, "max_groups must be in (0, num_experts]");
  TORCH_CHECK(compact_expert_offsets.numel() >= max_groups, "compact_expert_offsets must have max_groups entries");
  TORCH_CHECK(compact_problem_sizes1.numel() >= max_groups * 3, "compact_problem_sizes1 must have max_groups rows");
  TORCH_CHECK(compact_problem_sizes2.numel() >= max_groups * 3, "compact_problem_sizes2 must have max_groups rows");
  TORCH_CHECK(compact_expert_ids.numel() >= max_groups, "compact_expert_ids must have max_groups entries");

  auto stream = at::cuda::getCurrentCUDAStream(expert_offsets.device().index());
  compact_cutlass_w4a8_moe_mm_data_kernel<<<1, 1, 0, stream>>>(
      static_cast<const int32_t*>(expert_offsets.data_ptr()),
      static_cast<const int32_t*>(problem_sizes1.data_ptr()),
      static_cast<const int32_t*>(problem_sizes2.data_ptr()),
      static_cast<int32_t*>(compact_expert_offsets.data_ptr()),
      static_cast<int32_t*>(compact_problem_sizes1.data_ptr()),
      static_cast<int32_t*>(compact_problem_sizes2.data_ptr()),
      static_cast<int32_t*>(compact_expert_ids.data_ptr()),
      num_experts,
      max_groups);
}
