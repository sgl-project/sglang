#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/all.h>

#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "allreduce/custom_all_reduce.cuh"
#include "utils.h"

namespace {

constexpr uint64_t kSpinLimit = 100000000;

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t value) {
  if constexpr (std::is_same_v<scalar_t, half>) {
    return __half2float(value);
  } else {
    return __bfloat162float(value);
  }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float value) {
  if constexpr (std::is_same_v<scalar_t, half>) {
    return __float2half_rn(value);
  } else {
    return __float2bfloat16_rn(value);
  }
}

__global__ void increment_epoch_kernel(uint64_t* epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    epoch[0] += 1;
  }
}

__global__ void dispatch_output_lse_kernel(
    const uint4* __restrict__ partial_output,
    const float* __restrict__ partial_lse,
    const int64_t* __restrict__ peer_output_ptrs,
    const int64_t* __restrict__ peer_lse_ptrs,
    const uint64_t* __restrict__ epoch_ptr,
    int64_t world_size,
    int64_t rank,
    int64_t num_tokens,
    int64_t max_num_tokens,
    int64_t heads_per_rank,
    int64_t head_dim,
    int64_t output_token_stride,
    int64_t lse_token_stride) {
  const int64_t item = static_cast<int64_t>(blockIdx.x);
  const int64_t destination_rank = item / num_tokens;
  const int64_t token_idx = item - destination_rank * num_tokens;

  const uint64_t epoch = epoch_ptr[0];
  const int64_t parity = static_cast<int64_t>(epoch & 1ULL);
  const int64_t destination_item =
      ((parity * world_size + rank) * max_num_tokens + token_idx) * heads_per_rank;
  const int64_t source_head = destination_rank * heads_per_rank;

  auto* peer_output = reinterpret_cast<uint4*>(static_cast<uintptr_t>(peer_output_ptrs[destination_rank]));
  const int64_t vectors_per_item = heads_per_rank * head_dim / 8;
  const int64_t source_vector = (token_idx * output_token_stride + source_head * head_dim) / 8;
  const int64_t destination_vector = destination_item * head_dim / 8;
  for (int64_t vector_idx = threadIdx.x; vector_idx < vectors_per_item; vector_idx += blockDim.x) {
    peer_output[destination_vector + vector_idx] = partial_output[source_vector + vector_idx];
  }

  auto* peer_lse = reinterpret_cast<float*>(static_cast<uintptr_t>(peer_lse_ptrs[destination_rank]));
  const int64_t source_lse = token_idx * lse_token_stride + source_head;
  for (int64_t head_idx = threadIdx.x; head_idx < heads_per_rank; head_idx += blockDim.x) {
    peer_lse[destination_item + head_idx] = partial_lse[source_lse + head_idx];
  }
}

__global__ void signal_kernel(
    const int64_t* __restrict__ peer_signal_ptrs,
    const uint64_t* __restrict__ epoch_ptr,
    int64_t world_size,
    int64_t rank) {
  const int64_t destination_rank = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (destination_rank >= world_size) {
    return;
  }

  const uint64_t epoch = epoch_ptr[0];
  const int64_t parity = static_cast<int64_t>(epoch & 1ULL);
  auto* peer_signal =
      reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(peer_signal_ptrs[destination_rank]));
  const int64_t signal_item = parity * world_size + rank;
  sglang::st_flag_release_u64(peer_signal + signal_item, epoch);
}

template <typename scalar_t>
__global__ void wait_lse_combine_kernel(
    const scalar_t* __restrict__ received_output,
    const float* __restrict__ received_lse,
    const uint64_t* __restrict__ received_signal,
    const uint64_t* __restrict__ epoch_ptr,
    scalar_t* __restrict__ combined_output,
    int64_t world_size,
    int64_t num_tokens,
    int64_t max_num_tokens,
    int64_t heads_per_rank,
    int64_t head_dim,
    bool is_lse_base_on_e) {
  extern __shared__ float weights[];

  const int64_t item = static_cast<int64_t>(blockIdx.x);
  const int64_t token_idx = item / heads_per_rank;
  const int64_t head_idx = item - token_idx * heads_per_rank;
  const uint64_t epoch = epoch_ptr[0];
  const int64_t parity = static_cast<int64_t>(epoch & 1ULL);

  if (threadIdx.x == 0) {
    float lse_max = -CUDART_INF_F;
    for (int64_t source_rank = 0; source_rank < world_size; ++source_rank) {
      const int64_t source_item =
          ((parity * world_size + source_rank) * max_num_tokens + token_idx) * heads_per_rank + head_idx;
      const int64_t signal_item = parity * world_size + source_rank;
      uint64_t spins = 0;
      while (sglang::ld_flag_acquire_u64(received_signal + signal_item) != epoch) {
        if (++spins >= kSpinLimit) {
          printf(
              "direct DCP A2A timeout source=%lld token=%lld head=%lld epoch=%llu\n",
              static_cast<long long>(source_rank),
              static_cast<long long>(token_idx),
              static_cast<long long>(head_idx),
              static_cast<unsigned long long>(epoch));
          asm volatile("trap;");
        }
      }

      float value = received_lse[source_item];
      if (isnan(value) || value == CUDART_INF_F) {
        value = -CUDART_INF_F;
      }
      if (is_lse_base_on_e) {
        value *= CUDART_L2E_F;
      }
      weights[source_rank] = value;
      lse_max = fmaxf(lse_max, value);
    }

    if (lse_max == -CUDART_INF_F) {
      lse_max = 0.0f;
    }
    float lse_sum = 0.0f;
    for (int64_t source_rank = 0; source_rank < world_size; ++source_rank) {
      lse_sum += exp2f(weights[source_rank] - lse_max);
    }
    const float inverse_lse_sum = lse_sum > 0.0f ? 1.0f / lse_sum : 0.0f;
    for (int64_t source_rank = 0; source_rank < world_size; ++source_rank) {
      weights[source_rank] = exp2f(weights[source_rank] - lse_max) * inverse_lse_sum;
    }
  }
  __syncthreads();

  for (int64_t dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
    float accumulator = 0.0f;
    for (int64_t source_rank = 0; source_rank < world_size; ++source_rank) {
      const float weight = weights[source_rank];
      if (weight == 0.0f) {
        continue;
      }
      const int64_t source_item =
          ((parity * world_size + source_rank) * max_num_tokens + token_idx) * heads_per_rank + head_idx;
      accumulator += to_float(received_output[source_item * head_dim + dim_idx]) * weight;
    }
    combined_output[item * head_dim + dim_idx] = from_float<scalar_t>(accumulator);
  }
}

void check_launch(const char* kernel_name) {
  const cudaError_t status = cudaGetLastError();
  TORCH_CHECK(
      status == cudaSuccess, kernel_name, " launch failed: ", cudaGetErrorString(status));
}

void check_cuda_tensor(const at::Tensor& tensor, const at::Device& device, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.device() == device, name, " must be on the same CUDA device as partial_output");
}

}  // namespace

void direct_dcp_a2a_lse_reduce(
    const at::Tensor& partial_output,
    const at::Tensor& partial_lse,
    const at::Tensor& peer_output_ptrs,
    const at::Tensor& peer_lse_ptrs,
    const at::Tensor& peer_signal_ptrs,
    at::Tensor& received_output,
    at::Tensor& received_lse,
    at::Tensor& received_signal,
    at::Tensor& epoch,
    at::Tensor& combined_output,
    int64_t world_size,
    int64_t rank,
    int64_t max_num_tokens,
    bool is_lse_base_on_e) {
  TORCH_CHECK(partial_output.is_cuda(), "partial output and LSE must be CUDA tensors");
  TORCH_CHECK(partial_lse.is_cuda(), "partial output and LSE must be CUDA tensors");
  const auto device = partial_output.device();
  check_cuda_tensor(partial_lse, device, "partial_lse");
  check_cuda_tensor(peer_output_ptrs, device, "peer_output_ptrs");
  check_cuda_tensor(peer_lse_ptrs, device, "peer_lse_ptrs");
  check_cuda_tensor(peer_signal_ptrs, device, "peer_signal_ptrs");
  check_cuda_tensor(received_output, device, "received_output");
  check_cuda_tensor(received_lse, device, "received_lse");
  check_cuda_tensor(received_signal, device, "received_signal");
  check_cuda_tensor(epoch, device, "epoch");
  check_cuda_tensor(combined_output, device, "combined_output");

  const auto output_dtype = partial_output.scalar_type();
  TORCH_CHECK(
      output_dtype == at::ScalarType::Half || output_dtype == at::ScalarType::BFloat16,
      "symm_a2a only supports FP16 and BF16 attention output; use "
      "--dcp-comm-backend a2a when the attention backend emits another dtype");
  TORCH_CHECK(partial_lse.scalar_type() == at::ScalarType::Float, "partial LSE must be FP32");
  TORCH_CHECK(partial_output.dim() == 3 && partial_lse.dim() == 2, "expected output [T,H,D] and LSE [T,H]");
  TORCH_CHECK(world_size > 1, "world_size must be greater than 1");
  TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank");
  TORCH_CHECK(max_num_tokens > 0, "max_num_tokens must be positive");

  const int64_t num_tokens = partial_output.size(0);
  const int64_t total_heads = partial_output.size(1);
  const int64_t head_dim = partial_output.size(2);
  const int64_t output_token_stride = partial_output.stride(0);
  const int64_t lse_token_stride = partial_lse.stride(0);
  TORCH_CHECK(
      partial_output.stride(2) == 1 && partial_output.stride(1) == head_dim &&
          output_token_stride >= total_heads * head_dim && output_token_stride % 8 == 0,
      "partial output must have packed heads and an aligned token stride");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(partial_output.data_ptr()) % alignof(uint4) == 0,
      "partial output base pointer must be 16-byte aligned");
  TORCH_CHECK(
      partial_lse.stride(1) == 1 && lse_token_stride >= total_heads,
      "partial LSE must have packed heads");
  TORCH_CHECK(
      num_tokens > 0 && num_tokens <= max_num_tokens,
      "token count exceeds symmetric buffer capacity");
  TORCH_CHECK(total_heads > 0 && total_heads % world_size == 0, "attention heads must divide evenly across DCP ranks");
  TORCH_CHECK(
      partial_lse.size(0) == num_tokens && partial_lse.size(1) == total_heads,
      "LSE shape must match attention output");
  TORCH_CHECK(head_dim > 0 && head_dim % 8 == 0, "head_dim must be divisible by 8 for 16-byte stores");
  const int64_t heads_per_rank = total_heads / world_size;

  TORCH_CHECK(combined_output.scalar_type() == output_dtype, "combined output dtype must match partial output");
  TORCH_CHECK(combined_output.is_contiguous(), "combined output must be contiguous");
  TORCH_CHECK(
      combined_output.dim() == 3 && combined_output.size(0) == num_tokens &&
          combined_output.size(1) == heads_per_rank && combined_output.size(2) == head_dim,
      "combined output has the wrong shape");

  TORCH_CHECK(received_output.scalar_type() == output_dtype, "received output dtype must match partial output");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(received_output.data_ptr()) % alignof(uint4) == 0,
      "received output base pointer must be 16-byte aligned");
  TORCH_CHECK(received_lse.scalar_type() == at::ScalarType::Float, "received LSE must be FP32");
  TORCH_CHECK(received_signal.scalar_type() == at::ScalarType::Long, "received signal must be int64");
  TORCH_CHECK(epoch.scalar_type() == at::ScalarType::Long, "epoch must be int64");
  TORCH_CHECK(
      received_output.is_contiguous() && received_lse.is_contiguous() && received_signal.is_contiguous() &&
          epoch.is_contiguous(),
      "symmetric staging tensors and epoch must be contiguous");
  TORCH_CHECK(
      received_output.dim() == 5 && received_output.size(0) == 2 && received_output.size(1) == world_size &&
          received_output.size(2) == max_num_tokens && received_output.size(3) == heads_per_rank &&
          received_output.size(4) == head_dim,
      "received output has the wrong shape or capacity");
  TORCH_CHECK(
      received_lse.dim() == 4 && received_lse.size(0) == 2 && received_lse.size(1) == world_size &&
          received_lse.size(2) == max_num_tokens && received_lse.size(3) == heads_per_rank,
      "received LSE has the wrong shape or capacity");
  TORCH_CHECK(
      received_signal.dim() == 2 && received_signal.size(0) == 2 && received_signal.size(1) == world_size,
      "received signal has the wrong shape");
  TORCH_CHECK(epoch.numel() == 1, "epoch must contain exactly one counter");

  for (const auto* pointer_table : {&peer_output_ptrs, &peer_lse_ptrs, &peer_signal_ptrs}) {
    TORCH_CHECK(pointer_table->scalar_type() == at::ScalarType::Long, "peer pointer tables must be int64");
    TORCH_CHECK(pointer_table->is_contiguous(), "peer pointer tables must be contiguous");
    TORCH_CHECK(pointer_table->dim() == 1 && pointer_table->numel() == world_size, "peer pointer tables must have world_size entries");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(partial_output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int64_t kExchangeThreads = 256;
  constexpr int64_t kCombineThreads = 128;

  increment_epoch_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<uint64_t*>(epoch.data_ptr<int64_t>()));
  check_launch("increment_epoch_kernel");

  const int64_t dispatch_blocks = world_size * num_tokens;
  dispatch_output_lse_kernel<<<dispatch_blocks, kExchangeThreads, 0, stream>>>(
      reinterpret_cast<const uint4*>(partial_output.data_ptr()),
      partial_lse.data_ptr<float>(),
      peer_output_ptrs.data_ptr<int64_t>(),
      peer_lse_ptrs.data_ptr<int64_t>(),
      reinterpret_cast<const uint64_t*>(epoch.data_ptr<int64_t>()),
      world_size,
      rank,
      num_tokens,
      max_num_tokens,
      heads_per_rank,
      head_dim,
      output_token_stride,
      lse_token_stride);
  check_launch("dispatch_output_lse_kernel");

  const int64_t signal_blocks = (world_size + kExchangeThreads - 1) / kExchangeThreads;
  signal_kernel<<<signal_blocks, kExchangeThreads, 0, stream>>>(
      peer_signal_ptrs.data_ptr<int64_t>(),
      reinterpret_cast<const uint64_t*>(epoch.data_ptr<int64_t>()),
      world_size,
      rank);
  check_launch("signal_kernel");

  const int64_t combine_blocks = num_tokens * heads_per_rank;
  const size_t shared_memory_bytes = static_cast<size_t>(world_size) * sizeof(float);
  if (output_dtype == at::ScalarType::BFloat16) {
    wait_lse_combine_kernel<nv_bfloat16><<<combine_blocks, kCombineThreads, shared_memory_bytes, stream>>>(
        reinterpret_cast<const nv_bfloat16*>(received_output.data_ptr()),
        received_lse.data_ptr<float>(),
        reinterpret_cast<const uint64_t*>(received_signal.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(epoch.data_ptr<int64_t>()),
        reinterpret_cast<nv_bfloat16*>(combined_output.data_ptr()),
        world_size,
        num_tokens,
        max_num_tokens,
        heads_per_rank,
        head_dim,
        is_lse_base_on_e);
  } else {
    wait_lse_combine_kernel<half><<<combine_blocks, kCombineThreads, shared_memory_bytes, stream>>>(
        reinterpret_cast<const half*>(received_output.data_ptr()),
        received_lse.data_ptr<float>(),
        reinterpret_cast<const uint64_t*>(received_signal.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(epoch.data_ptr<int64_t>()),
        reinterpret_cast<half*>(combined_output.data_ptr()),
        world_size,
        num_tokens,
        max_num_tokens,
        heads_per_rank,
        head_dim,
        is_lse_base_on_e);
  }
  check_launch("wait_lse_combine_kernel");
}
