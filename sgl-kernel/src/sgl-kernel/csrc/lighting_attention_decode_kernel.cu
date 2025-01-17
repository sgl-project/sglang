#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "utils.hpp"

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define WARPS_PER_BLOCK 4

// 统一使用float进行warp reduce sum
__device__ __forceinline__ float warpReduceSum(float sum) {
    sum += __shfl_down_sync(FULL_MASK, sum, 16);
    sum += __shfl_down_sync(FULL_MASK, sum, 8);
    sum += __shfl_down_sync(FULL_MASK, sum, 4); 
    sum += __shfl_down_sync(FULL_MASK, sum, 2);
    sum += __shfl_down_sync(FULL_MASK, sum, 1);
    return sum;
}

template<typename T>
__global__ void lightning_attention_decode_kernel(
    const T* __restrict__ q,      // [b, h, 1, d]
    const T* __restrict__ k,      // [b, h, 1, d]
    const T* __restrict__ v,      // [b, h, 1, e]
    const T* __restrict__ past_kv,// [b, h, d, e]
    const T* __restrict__ slope,  // [h, 1, 1]
    T* __restrict__ output,       // [b, h, 1, e]
    T* __restrict__ new_kv,       // [b, h, d, e]
    const int batch_size,
    const int num_heads,
    const int dim,
    const int embed_dim) {
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int lane_id = tx;
    
    const int current_head = blockDim.y * blockIdx.x + ty;
    const int b = current_head / num_heads;
    const int h = current_head % num_heads;
    
    if (b >= batch_size) return;
    
    const int64_t qk_offset = b * num_heads * dim + h * dim;
    const int64_t v_offset = b * num_heads * embed_dim + h * embed_dim;
    const int64_t kv_offset = b * num_heads * dim * embed_dim + h * dim * embed_dim;
    
    // 1. 计算新的kv
    for (int d = lane_id; d < dim; d += WARP_SIZE) {
        for (int e = 0; e < embed_dim; e++) {
            T val = exp(-1.0 * slope[h]) * past_kv[kv_offset + d * embed_dim + e];
            val += k[qk_offset + d] * v[v_offset + e];
            new_kv[kv_offset + d * embed_dim + e] = val;
        }
    }
    
    // 2. 计算qkv attention输出
    for (int e = lane_id; e < embed_dim; e += WARP_SIZE) {
        float sum = 0.0f;  // 使用float进行累加
        for (int d = 0; d < dim; d++) {
            sum += static_cast<float>(q[qk_offset + d]) * 
                  static_cast<float>(new_kv[kv_offset + d * embed_dim + e]);
        }
        
        sum = warpReduceSum(sum);
        
        if (lane_id == 0) {
            output[v_offset + e] = static_cast<T>(sum);
        }
    }
}

void lightning_attention_decode(
    torch::Tensor& q,
    torch::Tensor& k, 
    torch::Tensor& v,
    torch::Tensor& past_kv,
    torch::Tensor& slope,
    torch::Tensor& output,
    torch::Tensor& new_kv) {
    
    auto batch_size = q.size(0);
    auto num_heads = q.size(1);
    auto dim = q.size(3);
    auto embed_dim = v.size(3);
    
    dim3 block(WARP_SIZE, WARPS_PER_BLOCK);  // (32, 4)
    dim3 grid((batch_size * num_heads + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "lightning_attention_decode_kernel", [&] {
        lightning_attention_decode_kernel<scalar_t><<<grid, block, 0, stream>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            past_kv.data_ptr<scalar_t>(),
            slope.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            new_kv.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            dim,
            embed_dim
        );
    });
}