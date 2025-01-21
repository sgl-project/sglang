#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "utils.hpp"

#define THREADS_PER_BLOCK 128

template<typename T>
__global__ void lightning_attention_decode_kernel(
    const T* __restrict__ q,      // [b, h, 1, d]
    const T* __restrict__ k,      // [b, h, 1, d]
    const T* __restrict__ v,      // [b, h, 1, e]
    const float* __restrict__ past_kv, // [b, h, d, e]
    const float* __restrict__ slope,   // [h, 1, 1]
    T* __restrict__ output,       // [b, h, 1, e]
    float* __restrict__ new_kv,   // [b, h, d, e]
    const int batch_size,
    const int num_heads,
    const int dim,
    const int embed_dim) {
    
    extern __shared__ char smem[];
    T* q_shared = reinterpret_cast<T*>(smem);
    T* k_shared = reinterpret_cast<T*>(smem + dim * sizeof(T));
    T* v_shared = reinterpret_cast<T*>(smem + 2 * dim * sizeof(T));
    float* new_kv_shared = reinterpret_cast<float*>(smem + (2 * dim + embed_dim) * sizeof(T));
    T* output_shared = reinterpret_cast<T*>(smem + (2 * dim + embed_dim) * sizeof(T) + dim * (embed_dim + 1) * sizeof(float));
    
    const int32_t tid = threadIdx.x;
    const int32_t current_head = blockIdx.x;
    const int32_t b = current_head / num_heads;
    const int32_t h = current_head % num_heads;
    
    if (b >= batch_size) return;
    
    const int32_t qk_offset = b * num_heads * dim + h * dim;
    const int32_t v_offset = b * num_heads * embed_dim + h * embed_dim;
    const int32_t kv_offset = b * num_heads * dim * embed_dim + h * dim * embed_dim;
    
    for (int d = tid; d < dim; d += blockDim.x) {
        q_shared[d] = q[qk_offset + d];
        k_shared[d] = k[qk_offset + d];
    }
    for (int e = tid; e < embed_dim; e += blockDim.x) {
        v_shared[e] = v[v_offset + e];
    }
    
    __syncthreads();
    
    const float ratio = expf(-1.0f * slope[h]);
    
    for (int d = tid; d < dim; d += blockDim.x) {
        T k_val = k_shared[d];
        for (int e = 0; e < embed_dim; ++e) {
            int past_kv_idx = kv_offset + d * embed_dim + e;
            T v_val = v_shared[e];
            float new_val = ratio * past_kv[past_kv_idx] + k_val * v_val;
            int shared_idx = d * (embed_dim + 1) + e;
            new_kv_shared[shared_idx] = new_val;
        }
    }
    
    __syncthreads();
    
    for (int idx = tid; idx < dim * embed_dim; idx += blockDim.x) {
        int d = idx / embed_dim;
        int e = idx % embed_dim;
        int shared_idx = d * (embed_dim + 1) + e;
        int global_idx = kv_offset + idx;
        new_kv[global_idx] = new_kv_shared[shared_idx];
    }
    
    __syncthreads();
    
    for (int e = tid; e < embed_dim; e += blockDim.x) {
        float sum = 0.0f;
        for (int d = 0; d < dim; ++d) {
            int shared_idx = d * (embed_dim + 1) + e;
            sum += q_shared[d] * new_kv_shared[shared_idx];
        }
        output_shared[e] = static_cast<T>(sum);
    }
    
    __syncthreads();
    
    if (tid == 0) {
        for (int e = 0; e < embed_dim; ++e) {
            output[v_offset + e] = output_shared[e];
        }
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
    auto dim = q.size(3);
    auto embed_dim = v.size(3);
    
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(batch_size * num_heads);
    
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "lightning_attention_decode_kernel", ([&] {
        size_t smem_size = (2 * dim + 2 * embed_dim) * sizeof(scalar_t) + dim * (embed_dim + 1) * sizeof(float);
        lightning_attention_decode_kernel<scalar_t><<<grid, block, smem_size, stream>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            past_kv.data_ptr<float>(),
            slope.data_ptr<float>(),
            output.data_ptr<scalar_t>(),
            new_kv.data_ptr<float>(),
            batch_size,
            num_heads,
            dim,
            embed_dim
        );
    }));
}