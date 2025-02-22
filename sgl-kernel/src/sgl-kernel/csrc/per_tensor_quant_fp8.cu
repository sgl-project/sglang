#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>
#include <cub/block/block_reduce.cuh>
#include <cmath>
#include "utils.h"
#include <flashinfer/vec_dtypes.cuh>

#define WARP_SIZE 32

using FP8_TYPE = c10::Float8_e4m3fn;
C10_HOST_DEVICE constexpr auto FP8_E4M3_MAX = std::numeric_limits<FP8_TYPE>::max();

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
                       : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
    return old;
}

__device__ __forceinline__ float warpReduceMax(float max_value) {
    max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, 16)); 
    max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, 8));
    max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, 4));
    max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, 2));
    max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, 1));
    return max_value;
}

template <typename T>
__global__ void per_tensor_absmax_kernel(const T* __restrict__ input, float* __restrict__ output_s,
                                        const int64_t num_elements) {
    float max_value = 0.0f;
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_size = blockDim.x * gridDim.x;

    constexpr uint32_t vec_size = 16 / sizeof(T);
    using vec_t = flashinfer::vec_t<T, vec_size>;
    
    const int32_t num_vec_elems = num_elements / vec_size;
    
    for (int32_t i = gid; i < num_vec_elems; i += grid_size) {
        vec_t input_vec;
        input_vec.cast_load(input + i * vec_size);
        
        #pragma unroll
        for (uint32_t j = 0; j < vec_size; ++j) {
            float val = static_cast<float>(input_vec[j]);
            max_value = fmaxf(max_value, fabsf(val));
        }
    }
    
    const int32_t remaining_start = num_vec_elems * vec_size;
    for (int32_t idx = remaining_start + gid; idx < num_elements; idx += grid_size) {
        float val = static_cast<float>(input[idx]);
        max_value = fmaxf(max_value, fabsf(val));
    }
    
    static __shared__ float warpLevelMaxs[WARP_SIZE];
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;

    max_value = warpReduceMax(max_value);

    if(laneId == 0) warpLevelMaxs[warpId] = max_value;
    __syncthreads();

    max_value = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelMaxs[laneId] : 0;

    if (warpId == 0) max_value = warpReduceMax(max_value);

    if (tid == 0) {
        atomicMaxFloat(output_s, max_value / FP8_E4M3_MAX);
    }
}

template <typename T>
__global__ void per_tensor_quant_fp8_kernel(const T* __restrict__ input, FP8_TYPE* __restrict__ output,
                                           const float* __restrict__ scale, const int64_t num_elements) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_size = blockDim.x * gridDim.x;
    const float scale_val = 1.0f / (*scale);

    constexpr uint32_t vec_size = 16 / sizeof(T);
    using vec_t = flashinfer::vec_t<T, vec_size>;
    
    const int32_t num_vec_elems = num_elements / vec_size;
    
    for (int32_t i = gid; i < num_vec_elems; i += grid_size) {
        vec_t input_vec;
        input_vec.cast_load(input + i * vec_size);
        
        FP8_TYPE output_arr[vec_size];
        #pragma unroll
        for (uint32_t j = 0; j < vec_size; ++j) {
            float val = fmax(fmin(static_cast<float>(input_vec[j]) * scale_val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
            output_arr[j] = static_cast<FP8_TYPE>(val);
        }

        #pragma unroll
        for (uint32_t j = 0; j < vec_size; ++j) {
            output[i * vec_size + j] = output_arr[j];
        }
    }
    
    const int32_t remaining_start = num_vec_elems * vec_size;
    for (int32_t idx = remaining_start + gid; idx < num_elements; idx += grid_size) {
        float val = fmax(-FP8_E4M3_MAX, fmin(static_cast<float>(input[idx]) * scale_val, FP8_E4M3_MAX));
        output[idx] = static_cast<FP8_TYPE>(val);
    }
}

void sgl_per_tensor_quant_fp8(torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s, bool is_static) {
    CHECK_INPUT(input);
    CHECK_INPUT(output_q);
    CHECK_INPUT(output_s);

    const int block_size = 256;
    const int num_elements = input.numel();
    const int num_blocks = min((num_elements + block_size - 1) / block_size, 1024);
    
    dim3 grid(num_blocks);
    dim3 block(block_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
        if (is_static == false){
            per_tensor_absmax_kernel<scalar_t><<<grid, block, 0, stream>>>(
                static_cast<scalar_t*>(input.data_ptr()),
                static_cast<float*>(output_s.data_ptr()),
                num_elements);
        }

        per_tensor_quant_fp8_kernel<scalar_t><<<grid, block, 0, stream>>>(
            static_cast<scalar_t*>(input.data_ptr()),
            static_cast<FP8_TYPE*>(output_q.data_ptr()),
            static_cast<float*>(output_s.data_ptr()),
            num_elements);
        return true;
    });
}