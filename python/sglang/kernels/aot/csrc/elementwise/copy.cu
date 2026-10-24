#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <vector>

template <int N>
struct InputArray {
  int values[N];
};

// MAX_N ints are passed by value in the kernel launch params (2KB), staying under the 4KB param limit.
constexpr int kCopyMaxN = 512;

template <int MAX_N>
__global__ void copy_to_gpu_no_ce_kernel(const InputArray<MAX_N> input_array, int* output, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    output[idx] = input_array.values[idx];
  }
}

void copy_to_gpu_no_ce(const at::Tensor& input, at::Tensor& output) {
  int N = static_cast<int>(input.numel());
  TORCH_CHECK(N <= kCopyMaxN, "copy_to_gpu_no_ce: numel ", N, " exceeds MAX_N ", kCopyMaxN);
  TORCH_CHECK(input.dim() == 1 && output.dim() == 1, "input/output must be 1-D");
  TORCH_CHECK(input.numel() == output.numel(), "input/output size mismatch");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "input/output must be contiguous");
  TORCH_CHECK(input.dtype() == torch::kInt32 && output.dtype() == torch::kInt32, "dtype must be int32");
  TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");
  TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");

  InputArray<kCopyMaxN> input_array;
  const int* input_ptr = input.data_ptr<int>();
  for (int i = 0; i < N; ++i)
    input_array.values[i] = input_ptr[i];

  dim3 block(256);
  dim3 grid((N + 255) / 256);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  copy_to_gpu_no_ce_kernel<<<grid, block, 0, stream>>>(input_array, output.data_ptr<int>(), N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
