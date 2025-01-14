// Refrence https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu
// The implementation of this RMS Norm is primarily for testing the proper use of CUB in the single-group kernel.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_reduce.cuh>
#include "utils.hpp"

template <typename scalar_t>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,           // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon,
    const int32_t hidden_size) {
    
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float s_variance;
    
    float thread_variance = 0.0f;

    // Compute sum of squares in FP32
    for (int32_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        const float x = static_cast<float>(input[blockIdx.x * hidden_size + idx]);
        thread_variance += x * x;
    }

    // Use BlockReduce for reduction sum
    float variance = BlockReduce(temp_storage).Sum(thread_variance);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / hidden_size + epsilon);
    }
    __syncthreads();

    // Normalize in FP32 and apply weight, then convert back to original dtype
    for (int32_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        const float x = static_cast<float>(input[blockIdx.x * hidden_size + idx]);
        const float w = static_cast<float>(weight[idx]);
        out[blockIdx.x * hidden_size + idx] = 
            static_cast<scalar_t>(x * s_variance * w);
    }
}

void rms_norm(
    torch::Tensor& out,         // [..., hidden_size]
    const torch::Tensor& input, // [..., hidden_size]
    const torch::Tensor& weight,// [hidden_size]
    const float epsilon) {
    
    const auto shape = input.sizes();
    const int32_t num_tokens = shape.size() > 1 ? 
        static_cast<int32_t>(shape[0]) : 1;
    const int32_t hidden_size = static_cast<int32_t>(shape.back());
    
    const int threads = 1024;
    const int blocks = num_tokens;
    
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rms_norm_kernel", ([&] {
            rms_norm_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                epsilon,
                hidden_size);
        }));
}
