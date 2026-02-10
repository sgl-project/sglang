#include <ATen/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cmath>

__global__ void prefix_sum(int* g_odata, const int* g_idata, const int n) {
  extern __shared__ int temp[];
  const int tid = threadIdx.x;
  int offset = 1;

  temp[2 * tid] = tid * 2 < n ? g_idata[2 * tid] : 0;
  temp[2 * tid + 1] = tid * 2 + 1 < n ? g_idata[2 * tid + 1] : 0;

  __syncthreads();

  for (int d = n >> 1; d > 0; d >>= 1) {
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
    __syncthreads();
  }

  if (tid == 0) {
    temp[n - 1] = 0;
  }
  __syncthreads();

  for (int d = 1; d < n; d <<= 1) {
    offset >>= 1;
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;

      int v = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += v;
    }
    __syncthreads();
  }

  if (tid * 2 < n) {
    g_odata[tid * 2] = temp[tid * 2];
  }
  if (tid * 2 + 1 < n) {
    g_odata[tid * 2 + 1] = temp[tid * 2 + 1];
  }
}

__global__ void
_fwd_kernel_ep_scatter_1_kernel(const int* num_recv_tokens_per_expert, int* expert_start_loc, int* m_indices) {
  const int expert_id = blockIdx.x;
  const int tid = threadIdx.x;

  int local_start_id = expert_start_loc[expert_id];
  int local_expert_token_num = num_recv_tokens_per_expert[expert_id];
  int* m_indices_ptr = m_indices + local_start_id;

  for (uint32_t i = tid; i < local_expert_token_num; i += blockDim.x) {
    m_indices_ptr[i] = expert_id;
  }
}

template <typename RECV_X_TYPE>
__global__ void _fwd_kernel_ep_scatter_2_kernel(
    int* expert_start_loc,
    const RECV_X_TYPE* recv_x,
    const int recv_x_stride0,
    const int recv_x_stride1,
    const float* recv_x_scale,
    const int recv_x_scale_stride0,
    const int recv_x_scale_stride1,
    const int* recv_topk,
    const int recv_topk_stride0,
    const int recv_topk_stride1,
    RECV_X_TYPE* output_tensor,
    const int output_tensor_stride0,
    const int output_tensor_stride1,
    float* output_tensor_scale,
    const int output_tensor_scale_stride0,
    const int output_tensor_scale_stride1,
    int* output_index,
    const int output_index_stride0,
    const int output_index_stride1,
    const int HIDDEN_SIZE,
    const int SCALE_HIDDEN_SIZE) {
  const int thread_per_group = 32;

  const int token_id = blockIdx.x;
  const int topk_id = threadIdx.x / thread_per_group;
  const int lane_id = threadIdx.x % thread_per_group;

  const int expert_id = recv_topk[token_id * recv_topk_stride0 + topk_id];
  if (expert_id < 0) return;

  extern __shared__ int dest_token_index[];

  if (lane_id == 0) {
    dest_token_index[topk_id] = atomicAdd(&expert_start_loc[expert_id], 1);
    output_index[token_id * output_index_stride0 + topk_id] = dest_token_index[topk_id];
  }

  __syncthreads();

  const int local_dest_token_index = dest_token_index[topk_id];

  const RECV_X_TYPE* recv_x_ptr = recv_x + token_id * recv_x_stride0;
  const float* recv_x_scale_ptr = recv_x_scale + token_id * recv_x_scale_stride0;

  RECV_X_TYPE* output_tensor_ptr = output_tensor + local_dest_token_index * output_tensor_stride0;
  float* output_tensor_scale_ptr = output_tensor_scale + local_dest_token_index * output_tensor_scale_stride0;

  for (uint32_t i = lane_id; i < HIDDEN_SIZE; i += thread_per_group) {
    output_tensor_ptr[i] = recv_x_ptr[i];
  }

  for (uint32_t i = lane_id; i < SCALE_HIDDEN_SIZE; i += thread_per_group) {
    output_tensor_scale_ptr[i] = recv_x_scale_ptr[i];
  }
}

template <typename TENSOR_TYPE>
__global__ void _fwd_kernel_ep_gather(
    const TENSOR_TYPE* input_tensor,
    const int input_tensor_stride0,
    const int input_tensor_stride1,
    const int* recv_topk_ids,
    const int recv_topk_ids_stride0,
    const int recv_topk_ids_stride1,
    const float* recv_topk_weight,
    const int recv_topk_weight_stride0,
    const int recv_topk_weight_stride1,
    const int* input_index,
    const int input_index_stride0,
    const int input_index_stride1,
    TENSOR_TYPE* output_tensor,
    const int output_tensor_stride0,
    const int output_tensor_stride1,
    const int hidden_size,
    const int num_topk) {
  const int thread_per_group = 32;
  const int token_id = blockIdx.x;
  const int tid = threadIdx.x;

  const int* start_idx = input_index + token_id * input_index_stride0;
  const float* start_weight = recv_topk_weight + token_id * recv_topk_weight_stride0;

  extern __shared__ float accumulator[];
  for (int m = tid; m < hidden_size; m += thread_per_group) {
    accumulator[m] = 0;
  }
  __syncthreads();

  for (int i = 0; i < num_topk; ++i) {
    const int expert_id = recv_topk_ids[token_id * recv_topk_ids_stride0 + i];
    if (expert_id >= 0) {
      const float acc_weight = start_weight[i];
      const int source_token_index = start_idx[i];
      for (int m = tid; m < hidden_size; m += thread_per_group) {
        accumulator[m] += static_cast<float>(input_tensor[source_token_index * input_tensor_stride0 + m]) * acc_weight;
      }
    }
  }

  __syncthreads();

  for (int m = tid; m < hidden_size; m += thread_per_group) {
    output_tensor[token_id * output_tensor_stride0 + m] = accumulator[m];
  }
}

void fwd_ep_scatter(
    torch::Tensor& recv_x,
    torch::Tensor& recv_x_scale,
    torch::Tensor& recv_topk,
    torch::Tensor& num_recv_tokens_per_expert,
    torch::Tensor& expert_start_loc,
    torch::Tensor& output_tensor,
    torch::Tensor& output_tensor_scale,
    torch::Tensor& m_indices,
    torch::Tensor& output_index) {
  const int num_experts = num_recv_tokens_per_expert.size(0);
  const int num_tokens = recv_x.size(0);
  const int hidden_size = recv_x.size(1);
  const int scale_hidden_size = recv_x_scale.size(1);
  const int num_topk = recv_topk.size(1);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(num_experts % 2 == 0, "Number of experts must be even for this implementation.");

  const int threadsPerBlock = num_experts / 2;
  prefix_sum<<<1, threadsPerBlock, 2 * threadsPerBlock * sizeof(int), stream>>>(
      static_cast<int*>(expert_start_loc.data_ptr()),
      static_cast<int*>(num_recv_tokens_per_expert.data_ptr()),
      num_experts);

  const int num_blocks1 = num_experts;
  const int num_threads1 = 32;

  dim3 grid1(num_blocks1);
  dim3 block1(num_threads1);

  _fwd_kernel_ep_scatter_1_kernel<<<grid1, block1, 0, stream>>>(
      static_cast<int*>(num_recv_tokens_per_expert.data_ptr()),
      static_cast<int*>(expert_start_loc.data_ptr()),
      static_cast<int*>(m_indices.data_ptr()));

  auto kernel_2 = _fwd_kernel_ep_scatter_2_kernel<c10::Float8_e4m3fn>;

  const int thread_per_group = 32;
  const int num_blocks2 = num_tokens;
  const int num_threads2 = num_topk * thread_per_group;

  dim3 grid2(num_blocks2);
  dim3 block2(num_threads2);

  kernel_2<<<grid2, block2, num_topk * sizeof(int), stream>>>(
      static_cast<int*>(expert_start_loc.data_ptr()),
      static_cast<c10::Float8_e4m3fn*>(recv_x.data_ptr()),
      recv_x.stride(0),
      recv_x.stride(1),
      static_cast<float*>(recv_x_scale.data_ptr()),
      recv_x_scale.stride(0),
      recv_x_scale.stride(1),
      static_cast<int*>(recv_topk.data_ptr()),
      recv_topk.stride(0),
      recv_topk.stride(1),
      static_cast<c10::Float8_e4m3fn*>(output_tensor.data_ptr()),
      output_tensor.stride(0),
      output_tensor.stride(1),
      static_cast<float*>(output_tensor_scale.data_ptr()),
      output_tensor_scale.stride(0),
      output_tensor_scale.stride(1),
      static_cast<int*>(output_index.data_ptr()),
      output_index.stride(0),
      output_index.stride(1),
      hidden_size,
      scale_hidden_size);
}

void fwd_ep_gather(
    torch::Tensor& input_tensor,
    torch::Tensor& recv_topk_ids,
    torch::Tensor& recv_topk_weight,
    torch::Tensor& input_index,
    torch::Tensor& output_tensor) {
  const int num_tokens = recv_topk_ids.size(0);
  const int num_topk = recv_topk_ids.size(1);
  const int hidden_size = input_tensor.size(1);

  const int thread_per_group = 32;

  const int num_blocks = num_tokens;
  const int num_threads = thread_per_group;

  auto kernel = _fwd_kernel_ep_gather<c10::BFloat16>;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(num_blocks);
  dim3 block(num_threads);

  kernel<<<grid, block, hidden_size * sizeof(float), stream>>>(
      static_cast<c10::BFloat16*>(input_tensor.data_ptr()),
      input_tensor.stride(0),
      input_tensor.stride(1),
      static_cast<int*>(recv_topk_ids.data_ptr()),
      recv_topk_ids.stride(0),
      recv_topk_ids.stride(1),
      static_cast<float*>(recv_topk_weight.data_ptr()),
      recv_topk_weight.stride(0),
      recv_topk_weight.stride(1),
      static_cast<int*>(input_index.data_ptr()),
      input_index.stride(0),
      input_index.stride(1),
      static_cast<c10::BFloat16*>(output_tensor.data_ptr()),
      output_tensor.stride(0),
      output_tensor.stride(1),
      hidden_size,
      num_topk);
}
