#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define FINAL_MASK 0xffffffff
#define BLOCK_SIZE 256

template <typename scalar_t>
__device__ __forceinline__ scalar_t add(scalar_t a, scalar_t b) {
  return a + b;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t warpReduceSum(scalar_t val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(FINAL_MASK, val, offset);
  }
  return val;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t blockReduceSum(scalar_t val) {
  __shared__ scalar_t shared[32];
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  val = warpReduceSum(val); // First reduce within warp

  if (lane == 0)
    shared[wid] = val; // Write reduced value to shared memory

  __syncthreads(); // Wait for all partial reductions

  // Read from shared memory only if that warp existed
  val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : 0;

  if (wid == 0)
    val = warpReduceSum(val); // Final reduce within first warp

  return val;
}

template <typename scalar_t>
__global__ void warp_reduce_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits>
        input,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> output,
    int N) {

  scalar_t sum = 0;

  // Grid-stride loop
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    sum += input[i];
  }

  // Perform block-wide reduction
  sum = blockReduceSum(sum);

  // Write result for this block to global memory
  if (threadIdx.x == 0) {
    output[blockIdx.x] = sum;
  }
}

torch::Tensor warp_reduce_cuda(torch::Tensor input) {
  // Input validation
  TORCH_CHECK(input.dim() == 1, "1D tensor expected");
  TORCH_CHECK(input.is_cuda(), "CUDA tensor expected");

  const auto N = input.size(0);

  // Handle empty tensor
  if (N == 0) {
    return torch::zeros({1}, input.options());
  }

  // Calculate grid dimensions
  const int threads = BLOCK_SIZE;
  const int blocks = (N + threads - 1) / threads;

  // Allocate output tensor for partial sums
  auto output = torch::empty({blocks}, input.options());

  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "warp_reduce_cuda", ([&] {
        warp_reduce_cuda_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            N);
      }));

  // Sum the partial results
  return output.sum();
}
