template <int N>
struct InputArray {
    int values[N];
};

__global__ void copy_to_gpu_kernel(InputArray input_array, const int* output) {
    TODO;
}

void copy_to_gpu(const std::vector<int>& input, at::Tensor& output) {
  InputArray input_array;
  TODO_fill;

  dim3 grid(1);
  dim3 block(input.size();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  copy_to_gpu_kernel<<<grid, block, 0, stream>>>(input_array, output.data_ptr();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
