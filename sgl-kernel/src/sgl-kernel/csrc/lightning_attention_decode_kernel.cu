#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "utils.hpp"

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4

template<typename T>
__global__ void lightning_attention_decode_kernel(
    const T* __restrict__ q,      // [b, h, 1, d]
    const T* __restrict__ k,      // [b, h, 1, d]
    const T* __restrict__ v,      // [b, h, 1, e]
    const float* __restrict__ past_kv,// [b, h, d, e]
    const float* __restrict__ slope,  // [h, 1, 1]
    T* __restrict__ output,       // [b, h, 1, e]
    float* __restrict__ new_kv,   // [b, h, d, e]
    const int batch_size,
    const int num_heads,
    const int qk_dim,
    const int v_dim) {
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int lane_id = tx;
    
    const int current_head = blockDim.y * blockIdx.x + ty;
    const int b = current_head / num_heads;
    const int h = current_head % num_heads;
    
    if (b >= batch_size) return;
    
    const int64_t qk_offset = b * num_heads * qk_dim + h * qk_dim;
    const int64_t v_offset = b * num_heads * v_dim + h * v_dim;
    const int64_t kv_offset = b * num_heads * qk_dim * v_dim + h * qk_dim * v_dim;
    
    // 1. Calculate new kv: kv = ratio * kv + k * v^T
    const float ratio = exp(-1.0f * slope[h]);
    for (int d = lane_id; d < qk_dim; d += WARP_SIZE) {
        for (int e = 0; e < v_dim; e++) {
            float val = ratio * past_kv[kv_offset + d * v_dim + e];
            val = val + k[qk_offset + d] * v[v_offset + e];
            new_kv[kv_offset + d * v_dim + e] = val;
        }
    }
    
    // 2. Calculate qkv attention output: output = q * kv
    for (int e = lane_id; e < v_dim; e += WARP_SIZE) {
        float sum = 0.0f;
        for (int d = 0; d < qk_dim; d++) {
            sum += q[qk_offset + d] * 
                  new_kv[kv_offset + d * v_dim + e];
        }
        output[v_offset + e] = static_cast<T>(sum);
    }
}

void lightning_attention_decode(
    const torch::Tensor& q,
    const torch::Tensor& k, 
    const torch::Tensor& v,
    const torch::Tensor& past_kv,
    const torch::Tensor& slope,
    torch::Tensor output,
    torch::Tensor new_kv) {
    
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(past_kv.is_contiguous(), "past_kv must be contiguous");
    
    auto batch_size = q.size(0);
    auto num_heads = q.size(1);
    auto qk_dim = q.size(3);
    auto v_dim = v.size(3);
    
    dim3 block(WARP_SIZE, WARPS_PER_BLOCK);  // (32, 4)
    dim3 grid((batch_size * num_heads + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "lightning_attention_decode_kernel", ([&] {
        lightning_attention_decode_kernel<scalar_t><<<grid, block, 0, stream>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            past_kv.data_ptr<float>(),
            slope.data_ptr<float>(),
            output.data_ptr<scalar_t>(),
            new_kv.data_ptr<float>(),
            batch_size,
            num_heads,
            qk_dim,
            v_dim
        );
    }));
}