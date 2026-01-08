// ********************************************************************************
// Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri
// Dao
// ********************************************************************************

// modified
// https://github.com/open-lm-engine/accelerated-model-architectures/blob/main/xma/functional/continuous_count/cuda_implementation/forward.cu
// to fuse cumsum

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cub/cub.cuh>

#include "include/utils.h"

#define MAX_ALLOWED_E 50000
#define BLOCK_SIZE 1024
#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

namespace cg = cooperative_groups;

using BlockScan = cub::BlockScan<uint32_t, BLOCK_SIZE>;

template <typename scalar_t>
inline __device__ void
_update_local_count(const scalar_t *x, int32_t *shared_memory, const int64_t &N,
                    const uint32_t &global_thread_id,
                    const uint32_t &grid_size) {
  constexpr uint32_t N_per_thread = 16 / sizeof(scalar_t);
  const uint32_t N_vec = N / N_per_thread;

  for (uint32_t i = global_thread_id; i < N_vec; i += grid_size) {
    const scalar_t *x_vec = load_128_bits<scalar_t>(x, i);

    for (uint32_t j = 0; j < N_per_thread; j++) {
      atomicAdd(&shared_memory[x_vec[j]], 1);
    }
  }

  const uint32_t i = (N_vec * N_per_thread) + global_thread_id;
  if (i < N) {
    atomicAdd(&shared_memory[x[i]], 1);
  }
}

struct BlockPrefixCallbackOp {
  uint32_t running_total;

  // Constructor
  __device__ BlockPrefixCallbackOp(uint32_t running_total)
      : running_total(running_total) {}

  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide
  // scan.
  __device__ int operator()(uint32_t block_aggregate) {
    uint32_t old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

inline __device__ void
_compute_cumsum(typename BlockScan::TempStorage &temp_storage,
                int32_t *shared_memory, const uint32_t &E) {
  const uint32_t num_loops = (E + blockDim.x - 1) / blockDim.x;
  uint32_t i = threadIdx.x;

  BlockPrefixCallbackOp prefix_op(0);

  for (uint32_t j = 0; j < num_loops; j++) {
    const bool is_valid_i = i < E;
    const uint32_t count = is_valid_i ? shared_memory[i] : 0;

    __syncwarp();

    uint32_t scan_value;
    BlockScan(temp_storage).InclusiveSum(count, scan_value, prefix_op);

    __syncthreads();

    if (is_valid_i) {
      shared_memory[i] = scan_value;
    }

    i += blockDim.x;
  }
}

template <typename scalar_t, bool do_cumsum>
__global__ void count_cumsum_cuda_kernel(const scalar_t *x,
                                         int32_t *count_output,
                                         int32_t *cumsum_output,
                                         const int64_t N, const uint32_t E) {
  const uint32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t grid_size = gridDim.x * blockDim.x;

  extern __shared__ int32_t shared_memory[];

  const uint32_t E4 = E >> 2;
  const int32_t init_value[] = {0, 0, 0, 0};

  for (uint32_t i = threadIdx.x; i < E4; i += blockDim.x) {
    store_128_bits<int32_t>(init_value, shared_memory, i);
  }

  for (uint32_t i = global_thread_id; i < E4; i += grid_size) {
    store_128_bits<int32_t>(init_value, count_output, i);
  }

  cg::this_grid().sync();

  _update_local_count<scalar_t>(x, shared_memory, N, global_thread_id,
                                grid_size);

  __syncthreads();

  // write the count_output to the global memory
  for (uint32_t i = threadIdx.x; i < E; i += blockDim.x) {
    atomicAdd(&count_output[i], shared_memory[i]);
  }

  if constexpr (do_cumsum) {
    __shared__ typename BlockScan::TempStorage temp_storage;

    cg::this_grid().sync();

    // load counts to shared memory
    for (uint32_t i = threadIdx.x; i < E4; i += blockDim.x) {
      const int32_t *output_vec = load_128_bits<int32_t>(count_output, i);
      store_128_bits<int32_t>(output_vec, shared_memory, i);
    }

    __syncthreads();

    _compute_cumsum(temp_storage, shared_memory, E);

    __syncthreads();

    // write cumsum output to global memory
    for (uint32_t i = global_thread_id; i < E4; i += grid_size) {
      const int32_t *output_vec = load_128_bits<int32_t>(shared_memory, i);
      store_128_bits<int32_t>(output_vec, cumsum_output, i);
    }
  }
}

void count_cumsum_cuda(const torch::Tensor &x, torch::Tensor &count_output,
                       std::optional<torch::Tensor> &_cumsum_output) {
  const bool do_cumsum = _cumsum_output.has_value();

  // check that all tensors are on GPU
  CHECK_CUDA_TENSOR(x);
  CHECK_CUDA_TENSOR(count_output);
  if (_cumsum_output.has_value()) {
    CHECK_CUDA_TENSOR(_cumsum_output.value());
  }

  const uint32_t E = count_output.size(0);

  TORCH_CHECK(E <= MAX_ALLOWED_E);
  TORCH_CHECK(E % 4 == 0);

  const int64_t N = x.numel();

  int device_id;
  cudaGetDevice(&device_id);

  int num_SMs;
  cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, device_id);

  const uint32_t block_reduce_smem_size =
      do_cumsum
          ? std::max(sizeof(uint32_t), sizeof(typename BlockScan::TempStorage))
          : 0;

  DISPATCH_INT_KERNEL(
      x.scalar_type(), "count_cumsum_cuda_kernel", scalar_t, ([&] {
        cudaFuncSetAttribute(
            (do_cumsum ? count_cumsum_cuda_kernel<scalar_t, true>
                       : count_cumsum_cuda_kernel<scalar_t, false>),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            MAX_ALLOWED_E * sizeof(uint32_t) + block_reduce_smem_size);

        // dynamically sized clusters need this stupid way of launching the
        // kernel
        cudaLaunchConfig_t launch_config = {0};
        launch_config.blockDim = BLOCK_SIZE;
        launch_config.gridDim = num_SMs;
        launch_config.dynamicSmemBytes =
            E * sizeof(uint32_t) + block_reduce_smem_size;

        cudaLaunchAttribute attributes[1];

        attributes[0].id = cudaLaunchAttributeCooperative;
        attributes[0].val.cooperative = 1;

        launch_config.attrs = attributes;
        launch_config.numAttrs = 1;

        cudaLaunchKernelEx(
            &launch_config,
            (do_cumsum ? count_cumsum_cuda_kernel<scalar_t, true>
                       : count_cumsum_cuda_kernel<scalar_t, false>),
            x.data_ptr<scalar_t>(), count_output.data_ptr<int32_t>(),
            _cumsum_output.has_value()
                ? _cumsum_output.value().data_ptr<int32_t>()
                : nullptr,
            N, E);
      }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("count_cumsum_cuda", &count_cumsum_cuda, "count cumsum (CUDA)");
}
