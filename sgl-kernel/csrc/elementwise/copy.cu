#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <vector>

template <int N>
struct InputArray {
  int values[N];
};

template <int N>
__global__ void copy_to_gpu_no_ce_kernel(const InputArray<N> input_array, int* output) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    output[idx] = input_array.values[idx];
  }
}

template <int N>
void copy_to_gpu_no_ce_impl(const at::Tensor& input, at::Tensor& output) {
  TORCH_CHECK(input.dim() == 1, "input must be 1-D");
  TORCH_CHECK(static_cast<int>(input.numel()) == N, "input numel must equal template N");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.dtype() == torch::kInt32, "input dtype must be int32");

  TORCH_CHECK(output.dim() == 1, "output dim");
  TORCH_CHECK(static_cast<int>(output.numel()) == N, "output size");
  TORCH_CHECK(output.is_contiguous(), "output contiguous");
  TORCH_CHECK(output.dtype() == torch::kInt32, "output dtype");

  TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");
  TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");

  InputArray<N> input_array;
  const int* input_ptr = input.data_ptr<int>();
  for (int i = 0; i < N; ++i)
    input_array.values[i] = input_ptr[i];

  // may use multi thread blocks if performance bottleneck
  dim3 grid(1);
  dim3 block(static_cast<int>(input.numel()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  copy_to_gpu_no_ce_kernel<<<grid, block, 0, stream>>>(input_array, output.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_to_gpu_no_ce(const at::Tensor& input, at::Tensor& output) {
  int N = static_cast<int>(input.numel());
  // Can use macro if there are more N needed
  if (N == 72) {
    copy_to_gpu_no_ce_impl<72>(input, output);
  } else if (N == 64) {
    copy_to_gpu_no_ce_impl<64>(input, output);
  } else {
    TORCH_CHECK(false, "unexpected N");
  }
}
