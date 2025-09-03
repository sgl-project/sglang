template <int N>
struct InputArray {
    int values[N];
};

template <int N>
__global__ void copy_to_gpu_no_ce_kernel(InputArray<T> input_array, const int* output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        output[idx] = input_array.values[idx];
    }
}

template <int N>
void copy_to_gpu_no_ce_impl(const std::vector<int>& input, at::Tensor& output) {
  TORCH_CHECK(static_cast<int>(input.size()) == N, "input size");
  TORCH_CHECK(output.numel() == N, "output size");

  InputArray<N> input_array;
  for (int i = 0; i < N; ++i) input_array.values[i] = input[i];

  // TODO may use multi thread blocks?
  dim3 grid(1);
  dim3 block(input.size());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  copy_to_gpu_no_ce_kernel<<<grid, block, 0, stream>>>(input_array, output.data_ptr();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_to_gpu_no_ce(const std::vector<int>& input, at::Tensor& output) {
    const int N = input.size();
    if (N == 72) {
        copy_to_gpu_no_ce_impl<72>(input, output);
    } else {
        TORCH_CHECK(false, "unexpected N");
    }
}
