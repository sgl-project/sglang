#include <assert.h>
#include <c10/cuda/CUDAGuard.h>
#include <cudaTypedefs.h>
#include <inttypes.h>
#include <torch/all.h>

#include <iostream>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

constexpr int32_t MAX_NUM_GPUS = 64;
constexpr int32_t MAX_EXPERT_COPIES_MULTIPLE = 8;
constexpr int32_t GROUP_SIZE = 1024;

__global__ void gpu_workloads_count_kernel(
    const int32_t* __restrict__ topk_ids,
    const int32_t num_logical_experts,
    const int32_t num_gpus,
    const int32_t num_topk_ids,
    int32_t* __restrict__ gpu_workloads,                // [num_gpus]
    int32_t* __restrict__ grouped_gpu_workloads,        // [ceil_div(num_topk_ids/GROUP_SIZE), num_gpus]
    int32_t* __restrict__ cumsum_grouped_gpu_workloads  // [ceil_div(num_topk_ids/GROUP_SIZE), num_gpus]
) {
  __shared__ int32_t smem_grouped_gpu_workloads[MAX_NUM_GPUS];

  const int32_t num_groups = (num_topk_ids + GROUP_SIZE - 1) / GROUP_SIZE;
  const int32_t num_logical_experts_per_gpu = num_logical_experts / num_gpus;
  int32_t previous_cumsum_grouped_gpu_workload = 0;

  for (int32_t group_idx = 0; group_idx < num_groups; ++group_idx) {
    if (threadIdx.x < num_gpus) {
      smem_grouped_gpu_workloads[threadIdx.x] = 0;
    }
    __syncthreads();

    const int32_t global_idx = group_idx * GROUP_SIZE + threadIdx.x;

    if (global_idx < num_topk_ids) {
      const int32_t topk_id = topk_ids[global_idx];
      const int32_t gpu_id = topk_id / num_logical_experts_per_gpu;
      atomicAdd(&smem_grouped_gpu_workloads[gpu_id], 1);
    }

    __syncthreads();
    if (threadIdx.x < num_gpus) {
      int32_t workload = smem_grouped_gpu_workloads[threadIdx.x];
      grouped_gpu_workloads[group_idx * num_gpus + threadIdx.x] = workload;
      previous_cumsum_grouped_gpu_workload += workload;
      cumsum_grouped_gpu_workloads[group_idx * num_gpus + threadIdx.x] = previous_cumsum_grouped_gpu_workload;
    }
    __syncthreads();
  }
  if (threadIdx.x < num_gpus) {
    gpu_workloads[threadIdx.x] = previous_cumsum_grouped_gpu_workload;
  }
}

__global__ void search_balance_mapping_kernel(
    const int32_t* __restrict__ gpu_workloads_wo_balance,
    const int32_t num_gpus,
    const int32_t num_topk_ids,
    const int32_t global_search_interval_left,
    const int32_t global_search_interval_right,
    const int32_t num_search_iter,
    const int32_t search_stride,
    const int32_t expert_copies_multiple,
    int32_t* __restrict__ max_workload_after_balance,
    int32_t* __restrict__ gpu_workloads_balance_mapping) {
  __shared__ int32_t smem_gpu_workloads_wo_balance[MAX_NUM_GPUS];
  __shared__ int32_t smem_gpu_workloads_balance_mapping[MAX_EXPERT_COPIES_MULTIPLE * MAX_NUM_GPUS];
  __shared__ int32_t smem_max_workload_after_balance;

  int32_t tid = threadIdx.x;

  if (tid == 0) {
    smem_max_workload_after_balance = num_topk_ids;
  }

  if (tid < num_gpus) {
    smem_gpu_workloads_wo_balance[tid] = gpu_workloads_wo_balance[tid];
  }
  __syncthreads();

  // find the gpu_id with the most workload
  int32_t gpu_id_with_max_workload_wo_balance = 0;
  int32_t max_workload = 0;
  // num_gpus is usually very small(<=16)
  for (int32_t i = 0; i < num_gpus; ++i) {
    if (smem_gpu_workloads_wo_balance[i] > max_workload) {
      max_workload = smem_gpu_workloads_wo_balance[i];
      gpu_id_with_max_workload_wo_balance = i;
    }
  }

  const int32_t local_search_interval_left = global_search_interval_left + tid * num_search_iter * search_stride;
  const int32_t local_search_interval_right =
      MIN(local_search_interval_left + (num_search_iter - 1) * search_stride, global_search_interval_right);

  int32_t headroom_on_gpus_after_balance[MAX_NUM_GPUS];
  int32_t i32_max_workload_after_balance = -1;

  for (int32_t capacity = local_search_interval_left; capacity <= local_search_interval_right;
       capacity += search_stride) {
    // We first assume that after workload balance, the workload on ANY GPU will not exceed `capacity`,
    // and then verify whether this assumption is true.
    for (int32_t gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      headroom_on_gpus_after_balance[gpu_id] = capacity;
    }
    int32_t num_balanced_topk_ids = 0;

    for (int32_t i = 0; i < num_gpus; ++i) {
      // We start with the GPU with most workload and transfer its workload to other GPUs which loaded redundant
      // experts.
      int32_t gpu_id = (gpu_id_with_max_workload_wo_balance + i) % num_gpus;
      // how much workload on current GPU
      int32_t workload_waiting_for_balance = smem_gpu_workloads_wo_balance[gpu_id];
      for (int32_t multiple_idx = expert_copies_multiple - 1; multiple_idx >= 0; --multiple_idx) {
        // the gpu id which we want to transfer workload to
        const int32_t balance_gpu_id = (gpu_id - multiple_idx + num_gpus) % num_gpus;

        // calculate how many workload balance_gpu_id can actually accept
        const int32_t headroom = headroom_on_gpus_after_balance[balance_gpu_id];
        const int32_t num_tokens_actual_accept = MIN(workload_waiting_for_balance, headroom);

        // update the remaining workload and capacity
        workload_waiting_for_balance -= num_tokens_actual_accept;
        headroom_on_gpus_after_balance[balance_gpu_id] -= num_tokens_actual_accept;
        num_balanced_topk_ids += num_tokens_actual_accept;
      }
    }

    assert(num_balanced_topk_ids <= num_topk_ids);
    if (num_balanced_topk_ids == num_topk_ids) {
      // The previous assumption is valid and current thread has obtained an alternative answer.
      i32_max_workload_after_balance = capacity;
      break;
    }
  }

  if (i32_max_workload_after_balance != -1) {
    // write to shared memory and calculate the optimal answer
    atomicMin(&smem_max_workload_after_balance, i32_max_workload_after_balance);
  }
  __syncthreads();

  if (smem_max_workload_after_balance == i32_max_workload_after_balance) {
    // The current thread has obtained the optimal answer and needs to output the solution to shared memory.
    for (int32_t gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      headroom_on_gpus_after_balance[gpu_id] = i32_max_workload_after_balance;
    }

    for (int32_t i = 0; i < num_gpus; ++i) {
      const int32_t gpu_id = (gpu_id_with_max_workload_wo_balance + i) % num_gpus;
      int32_t workload_waiting_for_balance = smem_gpu_workloads_wo_balance[gpu_id];
      for (int32_t multiple_idx = expert_copies_multiple - 1; multiple_idx >= 0; --multiple_idx) {
        const int32_t balance_gpu_id = (gpu_id - multiple_idx + num_gpus) % num_gpus;

        const int32_t headroom = headroom_on_gpus_after_balance[balance_gpu_id];
        const int32_t num_tokens_actual_accept = MIN(workload_waiting_for_balance, headroom);

        workload_waiting_for_balance -= num_tokens_actual_accept;
        headroom_on_gpus_after_balance[balance_gpu_id] -= num_tokens_actual_accept;
        smem_gpu_workloads_balance_mapping[multiple_idx * MAX_NUM_GPUS + gpu_id] = num_tokens_actual_accept;
      }
    }
  }

  __syncthreads();

  // Write the solution to global memory.
  if (tid < num_gpus) {
    for (int32_t i = 0; i < expert_copies_multiple; ++i) {
      gpu_workloads_balance_mapping[i * num_gpus + tid] = smem_gpu_workloads_balance_mapping[i * MAX_NUM_GPUS + tid];
    }
  }

  if (tid == 0) {
    max_workload_after_balance[0] = smem_max_workload_after_balance;
  }
}

void search_balance_mapping(
    const torch::Tensor& gpu_workloads_wo_balance,  // [num_gpus]
    const int32_t num_topk_ids,
    const int32_t expert_copies_multiple,
    const int32_t num_gpus,
    torch::Tensor& max_workload_after_balance,    // [1]
    torch::Tensor& gpu_workloads_balance_mapping  // [expert_copies_multiple, num_gpus]
) {
  auto stream = at::cuda::getCurrentCUDAStream(gpu_workloads_wo_balance.device().index());
  const int32_t search_interval_left = (num_topk_ids + num_gpus - 1) / num_gpus;
  const int32_t search_interval_right = (num_topk_ids + expert_copies_multiple - 1) / expert_copies_multiple;

  // search_interval_len should <= num_threads * num_search_iter * search_stride
  const int32_t search_interval_len = search_interval_right - search_interval_left + 1;

  // We set the search precision to 1% of the search_interval_len, and the maximum value is no more than 16.
  const int32_t search_stride =
      std::min<int32_t>(std::max<int32_t>(static_cast<int32_t>(search_interval_len / 100), 1), 16);

  unsigned int num_threads = static_cast<unsigned int>(std::max<int32_t>(search_interval_len, num_gpus));
  num_threads = std::min<unsigned int>(num_threads, 1024);
  int32_t num_search_iter = (search_interval_len + num_threads * search_stride - 1) / (num_threads * search_stride);

  search_balance_mapping_kernel<<<1, num_threads, 0, stream>>>(
      static_cast<const int32_t*>(gpu_workloads_wo_balance.data_ptr()),
      static_cast<int32_t>(num_gpus),
      num_topk_ids,
      search_interval_left,
      search_interval_right,
      num_search_iter,
      search_stride,
      expert_copies_multiple,
      static_cast<int32_t*>(max_workload_after_balance.data_ptr()),
      static_cast<int32_t*>(gpu_workloads_balance_mapping.data_ptr()));
}

__global__ void rewrite_topk_ids_by_balance_mapping_kernel(
    const int32_t* __restrict__ topk_ids,
    const int32_t* __restrict__ gpu_workloads_balance_mapping,
    const int32_t* __restrict__ grouped_gpu_workloads_wo_balance,
    const int32_t* __restrict__ cumsum_grouped_gpu_workloads_wo_balance,
    const int32_t num_gpus,
    const int32_t num_topk_ids,
    const int32_t num_logical_experts,
    const int32_t num_physical_experts,
    int32_t* __restrict__ new_topk_ids) {
  __shared__ int32_t smem_new_topk_ids[GROUP_SIZE];
  __shared__ int32_t smem_gpu_workloads_balance_mapping[MAX_EXPERT_COPIES_MULTIPLE * MAX_NUM_GPUS];

  __shared__ int32_t smem_target_multiple_indices[MAX_NUM_GPUS];
  __shared__ int32_t smem_num_cases_balance_to_multi_gpus;
  __shared__ int32_t smem_group_gpu_workloads_counter[MAX_NUM_GPUS];

  const int32_t expert_copies_multiple = num_physical_experts / num_logical_experts;
  const int32_t num_logical_experts_per_gpu = num_logical_experts / num_gpus;
  const int32_t num_physical_experts_per_gpu = num_physical_experts / num_gpus;

  const int32_t tid = threadIdx.x;
  const int32_t global_tid = blockIdx.x * GROUP_SIZE + threadIdx.x;
  const int32_t group_idx = blockIdx.x;

  if (tid == 0) {
    smem_num_cases_balance_to_multi_gpus = 0;
  }
  __syncthreads();

  if (tid < num_gpus) {
    int32_t workload_balanced_by_previous_groups = 0;
    if (group_idx != 0) {
      workload_balanced_by_previous_groups = cumsum_grouped_gpu_workloads_wo_balance[(group_idx - 1) * num_gpus + tid];
    }

    int32_t group_task = grouped_gpu_workloads_wo_balance[group_idx * num_gpus + tid];
    bool determined = false;

    // Determine whether the following situation A exists: In this group of workloads,
    // there is a GPU that wants to distribute its own workload to at least two other GPUs(including itself).
    // If the situation A exists, we will get smem_num_cases_balance_to_multi_gpus > 0.
    int32_t num_cases_balance_to_multi_gpus = 0;
    for (int32_t multiple_idx = expert_copies_multiple - 1; multiple_idx >= 0; --multiple_idx) {
      int32_t headroom = gpu_workloads_balance_mapping[multiple_idx * num_gpus + tid];
      const int32_t tmp = MIN(headroom, workload_balanced_by_previous_groups);
      headroom -= tmp;
      workload_balanced_by_previous_groups -= tmp;
      smem_gpu_workloads_balance_mapping[multiple_idx * MAX_NUM_GPUS + tid] = headroom;
      if (headroom == 0 || determined) {
        continue;
      }

      const bool balance_to_exactly_single_gpu = group_task <= headroom;
      if (balance_to_exactly_single_gpu) {
        smem_target_multiple_indices[tid] = multiple_idx;
      } else {
        num_cases_balance_to_multi_gpus += 1;
      }
      determined = true;
    }
    if (num_cases_balance_to_multi_gpus > 0) {
      atomicAdd(&smem_num_cases_balance_to_multi_gpus, 1);
    }
  }
  __syncthreads();

  int32_t logical_expert_id = 0;
  int32_t gpu_id = 0;
  int32_t physical_expert_id = 0;

  if (global_tid < num_topk_ids) {
    logical_expert_id = topk_ids[global_tid];
    gpu_id = logical_expert_id / num_logical_experts_per_gpu;
    physical_expert_id = logical_expert_id + gpu_id * (num_physical_experts_per_gpu - num_logical_experts_per_gpu);
  }

  if (smem_num_cases_balance_to_multi_gpus == 0) {
    // Happy path:
    // Situation A does not exist. In the current grouped workload,
    // all GPUs will send their workloads to the corresponding only GPU.
    // We can achieve maximum concurrency.
    if (global_tid < num_topk_ids) {
      // The token originally dispatched to GPU i will be dispatched to GPU (i-target_multiple_idx)
      const int32_t target_multiple_idx = smem_target_multiple_indices[gpu_id];
      int32_t new_topk_id =
          physical_expert_id - target_multiple_idx * (expert_copies_multiple - 1) * num_logical_experts_per_gpu;
      new_topk_id = (new_topk_id + num_physical_experts) % num_physical_experts;
      new_topk_ids[global_tid] = new_topk_id;
    }
    return;
  }

  if (tid < num_gpus) {
    smem_group_gpu_workloads_counter[tid] = 0;
  }
  __syncthreads();

  if (global_tid < num_topk_ids) {
    for (int32_t query_gpu_id = 0; query_gpu_id < num_gpus; ++query_gpu_id) {
      if (query_gpu_id != gpu_id) {
        continue;
      }
      // All threads with topk_id belonging to the same gpu will execute to this point
      // Get a sequence number by atomicAdd, which determines which gpu the token will transfer to
      const int32_t idx = static_cast<int32_t>(atomicAdd(&smem_group_gpu_workloads_counter[gpu_id], 1));
      int32_t acc = 0;
      int32_t multiple_idx = expert_copies_multiple - 1;
      for (; multiple_idx >= 0; --multiple_idx) {
        int32_t headroom = smem_gpu_workloads_balance_mapping[multiple_idx * MAX_NUM_GPUS + gpu_id];
        int32_t next_acc = acc + headroom;
        if (acc <= idx && idx < next_acc) {
          break;
        }
        acc = next_acc;
      }
      // The token originally dispatched to GPU i will be dispatched to GPU (i-target_multiple_idx)
      int32_t new_topk_id =
          physical_expert_id - multiple_idx * (expert_copies_multiple - 1) * num_logical_experts_per_gpu;
      new_topk_id = (new_topk_id + num_physical_experts) % num_physical_experts;
      smem_new_topk_ids[tid] = new_topk_id;
    }
  }

  __syncthreads();
  if (global_tid < num_topk_ids) {
    new_topk_ids[global_tid] = smem_new_topk_ids[tid];
  }
}

void balance_topk_ids(
    const torch::Tensor& topk_ids,
    const int64_t num_gpus,
    const int64_t num_logical_experts,
    const int64_t num_physical_experts,
    torch::Tensor& max_workload_after_balance,     // [1], int32
    torch::Tensor& gpu_workloads_balance_mapping,  // [num_physical_experts/num_logical_experts, num_gpus], int32
    torch::Tensor& new_topk_ids) {
  TORCH_CHECK(num_physical_experts % num_logical_experts == 0);
  TORCH_CHECK(num_logical_experts % num_gpus == 0);
  TORCH_CHECK(num_gpus <= MAX_NUM_GPUS);
  TORCH_CHECK(topk_ids.dtype() == torch::kInt32);
  TORCH_CHECK(max_workload_after_balance.dtype() == torch::kInt32);
  TORCH_CHECK(gpu_workloads_balance_mapping.dtype() == torch::kInt32);
  TORCH_CHECK(new_topk_ids.dtype() == torch::kInt32);

  const int32_t expert_copies_multiple = static_cast<int32_t>(num_physical_experts / num_logical_experts);
  TORCH_CHECK(expert_copies_multiple <= MAX_EXPERT_COPIES_MULTIPLE);

  auto stream = at::cuda::getCurrentCUDAStream(topk_ids.device().index());
  const int32_t num_topk_ids = static_cast<int32_t>(topk_ids.numel());

  auto i32_options = torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
  torch::Tensor gpu_workloads_wo_balance = torch::zeros({num_gpus}, i32_options);
  int32_t num_groups = (num_topk_ids + GROUP_SIZE - 1) / GROUP_SIZE;
  torch::Tensor grouped_gpu_workloads_wo_balance =
      torch::empty({static_cast<int64_t>(num_groups), num_gpus}, i32_options);
  torch::Tensor cumsum_grouped_gpu_workloads_wo_balance =
      torch::empty({static_cast<int64_t>(num_groups), num_gpus}, i32_options);

  // gpu_workloads_wo_balance[i] indicates the number of tokens received on each gpu after
  // dispatching the token without workload balance.
  // grouped_gpu_workloads_wo_balance: similar to gpu_workloads_wo_balance, but group by GROUP_SIZE.
  gpu_workloads_count_kernel<<<1, GROUP_SIZE, 0, stream>>>(
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<int32_t>(num_logical_experts),
      static_cast<int32_t>(num_gpus),
      num_topk_ids,
      static_cast<int32_t*>(gpu_workloads_wo_balance.data_ptr()),
      static_cast<int32_t*>(grouped_gpu_workloads_wo_balance.data_ptr()),
      static_cast<int32_t*>(cumsum_grouped_gpu_workloads_wo_balance.data_ptr()));

  // We traverse the entire solution space and search for how many tokens the GPU
  // with the most loadwork will receive after workload balance (saved in max_workload_after_balance).
  // Besides, we figure out how to shift some of the workload from the GPUs with more workload
  // to the GPUs with less workload (saved in gpu_workloads_balance_mapping).
  // gpu_workloads_balance_mapping[i][j] indicates how much workload will be transferred from gpu j to gpu j-i.
  search_balance_mapping(
      gpu_workloads_wo_balance,
      num_topk_ids,
      expert_copies_multiple,
      static_cast<int32_t>(num_gpus),
      max_workload_after_balance,
      gpu_workloads_balance_mapping);

  // do workload balance according to gpu_workloads_balance_mapping
  rewrite_topk_ids_by_balance_mapping_kernel<<<num_groups, GROUP_SIZE, 0, stream>>>(
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<const int32_t*>(gpu_workloads_balance_mapping.data_ptr()),
      static_cast<const int32_t*>(grouped_gpu_workloads_wo_balance.data_ptr()),
      static_cast<const int32_t*>(cumsum_grouped_gpu_workloads_wo_balance.data_ptr()),
      static_cast<int32_t>(num_gpus),
      num_topk_ids,
      static_cast<int32_t>(num_logical_experts),
      static_cast<int32_t>(num_physical_experts),
      static_cast<int32_t*>(new_topk_ids.data_ptr()));
}
