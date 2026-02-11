#!/usr/bin/env python3
"""
MHC CUDA Operators - Unified nn.Module Implementation

This module provides a unified nn.Module class that combines all CUDA-optimized 
operators for better performance and PyTorch integration.
"""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


# =========================================================================================
# Combined CUDA Source for All Operators
# =========================================================================================
_combined_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>

#define CUBLAS_CHECK(status) \
    TORCH_CHECK((status) == CUBLAS_STATUS_SUCCESS, "cuBLAS call failed")

// ============ Aggregate Operator ============
struct alignas(16) Packet8 {
    __nv_bfloat16 val[8];
};

template <int MAX_N>
__global__ void __launch_bounds__(128) fused_small_n_kernel(
    const __nv_bfloat16* __restrict__ res,
    const float* __restrict__ h_pre,
    __nv_bfloat16* __restrict__ out,
    const int M, const int N, const int H_vec
) {
    int m_idx = blockIdx.x;
    const float* w_ptr = h_pre + m_idx * N;
    
    float w_cache[MAX_N];
    #pragma unroll
    for (int n = 0; n < MAX_N; ++n) {
        if (n < N) w_cache[n] = w_ptr[n];
    }

    const int4* res_base = reinterpret_cast<const int4*>(res) + m_idx * N * H_vec;
    int4* out_base = reinterpret_cast<int4*>(out) + m_idx * H_vec;

    for (int h_v = threadIdx.x; h_v < H_vec; h_v += blockDim.x) {
        float acc[8] = {0.0f};
        Packet8 loaded_packets[MAX_N];

        #pragma unroll
        for (int n = 0; n < MAX_N; ++n) {
            if (n < N) {
                const int4* row = res_base + n * H_vec;
                loaded_packets[n] = *reinterpret_cast<const Packet8*>(&row[h_v]);
            }
        }

        #pragma unroll
        for (int n = 0; n < MAX_N; ++n) {
            if (n < N) {
                float w = w_cache[n];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[i] += __bfloat162float(loaded_packets[n].val[i]) * w;
                }
            }
        }

        Packet8 out_pkt;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            out_pkt.val[i] = __float2bfloat16(acc[i]);
        }
        out_base[h_v] = *reinterpret_cast<int4*>(&out_pkt);
    }
}

template <int CHUNK_SIZE>
__global__ void __launch_bounds__(128) fused_chunked_kernel(
    const __nv_bfloat16* __restrict__ res,
    const float* __restrict__ h_pre,
    __nv_bfloat16* __restrict__ out,
    const int M, const int N, const int H_vec
) {
    int m_idx = blockIdx.x;
    const float* w_base = h_pre + m_idx * N;
    const int4* res_base = reinterpret_cast<const int4*>(res) + m_idx * N * H_vec;
    int4* out_base = reinterpret_cast<int4*>(out) + m_idx * H_vec;

    for (int h_v = threadIdx.x; h_v < H_vec; h_v += blockDim.x) {
        float acc[8] = {0.0f};
        int n_base = 0;
        
        for (; n_base <= N - CHUNK_SIZE; n_base += CHUNK_SIZE) {
            float w_cache[CHUNK_SIZE];
            #pragma unroll
            for (int k = 0; k < CHUNK_SIZE; ++k) {
                w_cache[k] = w_base[n_base + k];
            }
            
            #pragma unroll
            for (int k = 0; k < CHUNK_SIZE; k += 8) {
                Packet8 v[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int4 val = (res_base + (n_base + k + i) * H_vec)[h_v];
                    v[i] = *reinterpret_cast<Packet8*>(&val);
                }
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float w = w_cache[k + i];
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        acc[j] += __bfloat162float(v[i].val[j]) * w;
                    }
                }
            }
        }
        
        for (int n = n_base; n < N; ++n) {
            float w = w_base[n];
            int4 val = (res_base + n * H_vec)[h_v];
            Packet8 v = *reinterpret_cast<Packet8*>(&val);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[i] += __bfloat162float(v.val[i]) * w;
            }
        }

        Packet8 out_pkt;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            out_pkt.val[i] = __float2bfloat16(acc[i]);
        }
        out_base[h_v] = *reinterpret_cast<int4*>(&out_pkt);
    }
}

torch::Tensor aggregate_cuda(torch::Tensor res, torch::Tensor h_pre) {
    // CRITICAL: Ensure alignment for Packet8 (16-byte) access
    // Input is guaranteed contiguous from Python side, check alignment only
    if (reinterpret_cast<uintptr_t>(res.data_ptr()) % 16 != 0) {
        res = res.clone();
    }

    TORCH_CHECK(res.dim() == 3, "residuals must be [M, N, D]");
    TORCH_CHECK(h_pre.dim() == 2, "h_pre must be [M, N]");
    
    int64_t M = res.size(0);
    int64_t N = res.size(1);
    int64_t D = res.size(2);
    
    TORCH_CHECK(D % 8 == 0, "D must be divisible by 8");
    int D_vec = D / 8;
    
    auto out = torch::empty({M, D}, res.options());
    
    const int threads = 128;
    const int blocks = M;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (N <= 4) {
        fused_small_n_kernel<4><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(res.data_ptr<at::BFloat16>()),
            h_pre.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            M, N, D_vec
        );
    } else if (N <= 8) {
        fused_small_n_kernel<8><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(res.data_ptr<at::BFloat16>()),
            h_pre.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            M, N, D_vec
        );
    } else {
        fused_chunked_kernel<32><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(res.data_ptr<at::BFloat16>()),
            h_pre.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            M, N, D_vec
        );
    }
    
    return out;
}

// ============ Sinkhorn Operator ============
#define BLOCK_SIZE 128
#define MAT_STRIDE 5

__global__ void sinkhorn_ilp_kernel(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    int n_matrices, int n_iters
) {
    __shared__ float4 smem[2 * BLOCK_SIZE * MAT_STRIDE];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_start_mat = bid * (2 * BLOCK_SIZE);
    
    int global_vec_base = block_start_mat * 4;
    int total_vectors = n_matrices * 4;

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int local_vec_idx = tid + k * BLOCK_SIZE;
        int global_vec_idx = global_vec_base + local_vec_idx;

        float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (global_vec_idx < total_vectors) {
            val = input[global_vec_idx];
            // Clamp input to avoid inf/nan issues
            val.x = fmaxf(-1e4f, fminf(val.x, 1e4f));
            val.y = fmaxf(-1e4f, fminf(val.y, 1e4f));
            val.z = fmaxf(-1e4f, fminf(val.z, 1e4f));
            val.w = fmaxf(-1e4f, fminf(val.w, 1e4f));
        }

        int mat_idx = local_vec_idx >> 2; 
        int row_idx = local_vec_idx & 3; 
        int smem_idx = mat_idx * MAT_STRIDE + row_idx;
        
        smem[smem_idx] = val;
    }

    __syncthreads();

    int my_mat_idx_a = tid * 2;
    int my_mat_idx_b = tid * 2 + 1;
    
    float ma[16], mb[16];
    
    int smem_base_a = my_mat_idx_a * MAT_STRIDE;
    #pragma unroll
    for(int r=0; r<4; ++r) {
        float4 tmp = smem[smem_base_a + r];
        ma[r*4+0] = tmp.x; ma[r*4+1] = tmp.y; ma[r*4+2] = tmp.z; ma[r*4+3] = tmp.w;
    }
    
    int smem_base_b = my_mat_idx_b * MAT_STRIDE;
    #pragma unroll
    for(int r=0; r<4; ++r) {
        float4 tmp = smem[smem_base_b + r];
        mb[r*4+0] = tmp.x; mb[r*4+1] = tmp.y; mb[r*4+2] = tmp.z; mb[r*4+3] = tmp.w;
    }

    // Find max per row for numerical stability to avoid underflow
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int off = r * 4;
        float row_max_a = -1e30f;
        float row_max_b = -1e30f;
        
        // Find max in this row manually to avoid loops
        row_max_a = fmaxf(fmaxf(ma[off], ma[off+1]), fmaxf(ma[off+2], ma[off+3]));
        row_max_b = fmaxf(fmaxf(mb[off], mb[off+1]), fmaxf(mb[off+2], mb[off+3]));

        // Subtract and exp
        ma[off]   = expf(ma[off]   - row_max_a);
        ma[off+1] = expf(ma[off+1] - row_max_a);
        ma[off+2] = expf(ma[off+2] - row_max_a);
        ma[off+3] = expf(ma[off+3] - row_max_a);

        mb[off]   = expf(mb[off]   - row_max_b);
        mb[off+1] = expf(mb[off+1] - row_max_b);
        mb[off+2] = expf(mb[off+2] - row_max_b);
        mb[off+3] = expf(mb[off+3] - row_max_b);
    }

    for (int iter = 0; iter < n_iters; ++iter) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            int off = r * 4;
            float sum_a = ma[off] + ma[off+1] + ma[off+2] + ma[off+3];
            float sum_b = mb[off] + mb[off+1] + mb[off+2] + mb[off+3];
            
            float inv_a = __fdividef(1.0f, fmaxf(sum_a, 1e-9f));
            float inv_b = __fdividef(1.0f, fmaxf(sum_b, 1e-9f));
            
            ma[off] *= inv_a; ma[off+1] *= inv_a; ma[off+2] *= inv_a; ma[off+3] *= inv_a;
            mb[off] *= inv_b; mb[off+1] *= inv_b; mb[off+2] *= inv_b; mb[off+3] *= inv_b;
        }

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            float sum_a = ma[c] + ma[c+4] + ma[c+8] + ma[c+12];
            float sum_b = mb[c] + mb[c+4] + mb[c+8] + mb[c+12];
            
            float inv_a = __fdividef(1.0f, fmaxf(sum_a, 1e-9f));
            float inv_b = __fdividef(1.0f, fmaxf(sum_b, 1e-9f));
            
            ma[c] *= inv_a; ma[c+4] *= inv_a; ma[c+8] *= inv_a; ma[c+12] *= inv_a;
            mb[c] *= inv_b; mb[c+4] *= inv_b; mb[c+8] *= inv_b; mb[c+12] *= inv_b;
        }
    }

    #pragma unroll
    for(int r=0; r<4; ++r) {
        smem[smem_base_a + r] = make_float4(ma[r*4], ma[r*4+1], ma[r*4+2], ma[r*4+3]);
    }
    #pragma unroll
    for(int r=0; r<4; ++r) {
        smem[smem_base_b + r] = make_float4(mb[r*4], mb[r*4+1], mb[r*4+2], mb[r*4+3]);
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int local_vec_idx = tid + k * BLOCK_SIZE;
        int global_vec_idx = global_vec_base + local_vec_idx;

        if (global_vec_idx < total_vectors) {
            int mat_idx = local_vec_idx >> 2; 
            int row_idx = local_vec_idx & 3; 
            output[global_vec_idx] = smem[mat_idx * MAT_STRIDE + row_idx];
        }
    }
}

torch::Tensor sinkhorn_cuda(torch::Tensor logits, int n_iters) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(logits.dim() == 3 && logits.size(1) == 4 && logits.size(2) == 4, "Must be [M, 4, 4]");
    
    // Input is guaranteed contiguous from Python side, check alignment only
    if (reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 != 0) {
        logits = logits.clone();
    }
    
    auto output = torch::empty_like(logits);
    int64_t n_matrices = logits.size(0);
    const int block_size = 128;
    const int mats_per_block = block_size * 2;
    const int grid_size = (n_matrices + mats_per_block - 1) / mats_per_block;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    sinkhorn_ilp_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const float4*>(logits.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        n_matrices, n_iters
    );
    
    return output;
}

// ============ ExpandMerge Operator ============
struct alignas(16) BF16Vec8 { __nv_bfloat16 vals[8]; };
struct alignas(8) BF16Vec4 { __nv_bfloat16 vals[4]; };
struct alignas(4) BF16Vec2 { __nv_bfloat16 vals[2]; };

template <int N> struct VecSelector;
template <> struct VecSelector<8> { using type = BF16Vec8; };
template <> struct VecSelector<4> { using type = BF16Vec4; };
template <> struct VecSelector<2> { using type = BF16Vec2; };

template <int N>
__global__ void fused_expand_merge_kernel(
    const __nv_bfloat16* __restrict__ residuals,
    const __nv_bfloat16* __restrict__ l_out,
    const float* __restrict__ h_res,
    const float* __restrict__ h_post,
    __nv_bfloat16* __restrict__ output,
    int M, int D
) {
    int m_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float sm_h_res[4][4];
    __shared__ float sm_h_post[4];
    
    if (tid < 4) sm_h_post[tid] = h_post[m_idx * 4 + tid];
    if (tid < 16) {
        int r = tid / 4;
        int c = tid % 4;
        sm_h_res[r][c] = h_res[m_idx * 16 + tid];
    }
    __syncthreads();

    int d_start = (blockIdx.y * blockDim.x + tid) * N;
    
    // Boundary check for vectorized access
    if (d_start + N > D) {
        if (d_start >= D) return;
        
        // Handle boundary elements one by one
        for (int k = 0; k < N && (d_start + k) < D; ++k) {
            size_t off_l = (size_t)m_idx * D + d_start + k;
            float l_val = __bfloat162float(l_out[off_l]);
            
            float acc[4];
            #pragma unroll
            for(int i=0; i<4; ++i) {
                acc[i] = sm_h_post[i] * l_val;
            }
            
            #pragma unroll
            for(int j=0; j<4; ++j) {
                size_t off_r = (size_t)m_idx * 4 * D + (size_t)j * D + d_start + k;
                float r_val = __bfloat162float(residuals[off_r]);
                
                #pragma unroll
                for(int i=0; i<4; ++i) {
                    acc[i] += sm_h_res[i][j] * r_val;
                }
            }
            
            #pragma unroll
            for(int i=0; i<4; ++i) {
                size_t off_out = (size_t)m_idx * 4 * D + (size_t)i * D + d_start + k;
                output[off_out] = __float2bfloat16(acc[i]);
            }
        }
        return;
    }

    size_t off_l = (size_t)m_idx * D + d_start;
    size_t off_res_base = (size_t)m_idx * 4 * D;

    float l_vals[N];
    if constexpr (N == 1) {
        l_vals[0] = __bfloat162float(l_out[off_l]);
    } else {
        using VecT = typename VecSelector<N>::type;
        VecT v = *reinterpret_cast<const VecT*>(l_out + off_l);
        #pragma unroll
        for(int k=0; k<N; ++k) l_vals[k] = __bfloat162float(v.vals[k]);
    }

    float acc[4][N];
    #pragma unroll
    for(int i=0; i<4; ++i) {
        float hp = sm_h_post[i];
        #pragma unroll
        for(int k=0; k<N; ++k) acc[i][k] = hp * l_vals[k];
    }

    #pragma unroll
    for(int j=0; j<4; ++j) {
        float r_vals[N];
        size_t off_r = off_res_base + (size_t)j * D + d_start;
        
        if constexpr (N == 1) {
            r_vals[0] = __bfloat162float(residuals[off_r]);
        } else {
            using VecT = typename VecSelector<N>::type;
            VecT v = *reinterpret_cast<const VecT*>(residuals + off_r);
            #pragma unroll
            for(int k=0; k<N; ++k) r_vals[k] = __bfloat162float(v.vals[k]);
        }

        #pragma unroll
        for(int i=0; i<4; ++i) {
            float hr = sm_h_res[i][j];
            #pragma unroll
            for(int k=0; k<N; ++k) acc[i][k] += hr * r_vals[k];
        }
    }

    #pragma unroll
    for(int i=0; i<4; ++i) {
        size_t off_out = off_res_base + (size_t)i * D + d_start;
        if constexpr (N == 1) {
            output[off_out] = __float2bfloat16(acc[i][0]);
        } else {
            using VecT = typename VecSelector<N>::type;
            VecT v;
            #pragma unroll
            for(int k=0; k<N; ++k) v.vals[k] = __float2bfloat16(acc[i][k]);
            *reinterpret_cast<VecT*>(output + off_out) = v;
        }
    }
}

torch::Tensor expand_merge_cuda(
    torch::Tensor h_res, torch::Tensor residuals, 
    torch::Tensor l_out, torch::Tensor h_post
) {
    TORCH_CHECK(residuals.dim() == 3, "residuals must be [M, N, D]");
    TORCH_CHECK(residuals.size(1) == 4, "n_streams must be 4");
    int64_t M = residuals.size(0);
    int64_t D = residuals.size(2);

    auto output = torch::empty_like(residuals);

    auto p_res = (uintptr_t)residuals.data_ptr();
    auto p_l = (uintptr_t)l_out.data_ptr();
    auto p_out = (uintptr_t)output.data_ptr();

    int vec_size = 1;
    if (D % 8 == 0 && p_res % 16 == 0 && p_l % 16 == 0 && p_out % 16 == 0)
        vec_size = 8;
    else if (D % 4 == 0 && p_res % 8 == 0 && p_l % 8 == 0 && p_out % 8 == 0)
        vec_size = 4;
    else if (D % 2 == 0 && p_res % 4 == 0 && p_l % 4 == 0 && p_out % 4 == 0)
        vec_size = 2;

    int block_dim = 256;
    int elems_per_block = block_dim * vec_size;
    dim3 block(block_dim);
    dim3 grid(M, (D + elems_per_block - 1) / elems_per_block);
    
    auto r_ptr = (const __nv_bfloat16*)residuals.data_ptr();
    auto l_ptr = (const __nv_bfloat16*)l_out.data_ptr();
    auto h_res_ptr = h_res.data_ptr<float>();
    auto h_post_ptr = h_post.data_ptr<float>();
    auto out_ptr = (__nv_bfloat16*)output.data_ptr();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (vec_size == 8)
        fused_expand_merge_kernel<8><<<grid, block, 0, stream>>>(r_ptr, l_ptr, h_res_ptr, h_post_ptr, out_ptr, M, D);
    else if (vec_size == 4)
        fused_expand_merge_kernel<4><<<grid, block, 0, stream>>>(r_ptr, l_ptr, h_res_ptr, h_post_ptr, out_ptr, M, D);
    else if (vec_size == 2)
        fused_expand_merge_kernel<2><<<grid, block, 0, stream>>>(r_ptr, l_ptr, h_res_ptr, h_post_ptr, out_ptr, M, D);
    else
        fused_expand_merge_kernel<1><<<grid, block, 0, stream>>>(r_ptr, l_ptr, h_res_ptr, h_post_ptr, out_ptr, M, D);

    return output;
}

// ============ MapSigmoid Operator ============
constexpr float kEps = 1e-8f;
constexpr unsigned kMaskAll = 0xffffffffu;

__device__ __forceinline__ float sigmoid_fast(float x) {
    return __fdividef(1.0f, 1.0f + __expf(-x));
}

__launch_bounds__(64, 8)
__global__ void fused_kernel_n4_warp(
    const float* __restrict__ r,
    const float* __restrict__ proj,
    const float* __restrict__ bias,
    const float* __restrict__ alpha,
    float* __restrict__ out_pre,
    float* __restrict__ out_post,
    float* __restrict__ out_res,
    int rows, int proj_row_stride
) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= rows) return;

    float r_inv = 0.0f;
    if (lane == 0) {
        r_inv = __fdividef(1.0f, __ldg(r + warp_id) + kEps);
    }
    r_inv = __shfl_sync(kMaskAll, r_inv, 0);

    if (lane < 24) {
        int k = lane;
        float scale = r_inv * __ldg(alpha + k);
        float h_val = fmaf(__ldg(proj + (size_t)warp_id * proj_row_stride + k), scale, __ldg(bias + k));

        if (k < 4) {
            out_pre[(size_t)warp_id * 4 + k] = sigmoid_fast(h_val);
        } else if (k < 8) {
            out_post[(size_t)warp_id * 4 + (k - 4)] = sigmoid_fast(h_val) * 2.0f;
        } else {
            out_res[(size_t)warp_id * 16 + (k - 8)] = h_val;
        }
    }
}

void map_sigmoid_cuda(
    torch::Tensor r, torch::Tensor proj, torch::Tensor bias, torch::Tensor alpha,
    torch::Tensor out_pre, torch::Tensor out_post, torch::Tensor out_res,
    int n_streams
) {
    int rows = r.numel();
    int out_dim = n_streams * (n_streams + 2);

    if (out_pre.dim() != 2 || out_pre.size(0) != rows || out_pre.size(1) != n_streams) {
        out_pre.resize_({rows, n_streams});
    }
    if (out_post.dim() != 2 || out_post.size(0) != rows || out_post.size(1) != n_streams) {
        out_post.resize_({rows, n_streams});
    }
    if (out_res.dim() != 3 || out_res.size(0) != rows || out_res.size(1) != n_streams || out_res.size(2) != n_streams) {
        out_res.resize_({rows, n_streams, n_streams});
    }

    const float* r_ptr = r.data_ptr<float>();
    const float* proj_ptr = proj.data_ptr<float>();
    const float* bias_ptr = bias.data_ptr<float>();
    const float* alpha_ptr = alpha.data_ptr<float>();
    float* out_pre_ptr = out_pre.data_ptr<float>();
    float* out_post_ptr = out_post.data_ptr<float>();
    float* out_res_ptr = out_res.data_ptr<float>();

    if (n_streams == 4 && out_dim == 24 && proj.stride(0) == 24 && proj.is_contiguous()) {
        int block = 64;
        int total_threads = rows * 32;
        int grid = (total_threads + block - 1) / block;
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        fused_kernel_n4_warp<<<grid, block, 0, stream>>>(
            r_ptr, proj_ptr, bias_ptr, alpha_ptr,
            out_pre_ptr, out_post_ptr, out_res_ptr,
            rows, 24
        );
    }
}

// ============ NormLinear Operator ============
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int THREADS>
__device__ __forceinline__ float block_reduce_sum(float val) {
    static_assert(THREADS % 32 == 0, "THREADS must be multiple of warp size");
    __shared__ float shared[THREADS / 32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    float out = (threadIdx.x < (THREADS / 32)) ? shared[lane] : 0.0f;
    if (wid == 0) {
        out = warp_reduce_sum(out);
    }
    return out;
}

template <int THREADS>
__global__ void cast_and_row_l2norm_kernel(
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ x_float,
    float* __restrict__ r,
    int64_t D, float inv_sqrt_d
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const __nv_bfloat16* row_ptr = x + row * D;
    float* row_out_ptr = x_float + row * D;

    float sum = 0.0f;
    // Scalar implementation to avoid alignment issues and ensure correctness
    for (int64_t i = threadIdx.x; i < D; i += THREADS) {
        const float v = __bfloat162float(row_ptr[i]);
        row_out_ptr[i] = v;
        sum = fmaf(v, v, sum);
    }

    const float total = block_reduce_sum<THREADS>(sum);
    if (threadIdx.x == 0) {
        r[row] = sqrtf(total) * inv_sqrt_d;
    }
}

std::tuple<torch::Tensor, torch::Tensor> norm_linear_cuda(
    torch::Tensor x, torch::Tensor weight
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2, "must be 2D");

    auto x_c = x.contiguous();
    // Ensure alignment for vectorized access if we used it, but keeping it safe for general access
    if (reinterpret_cast<uintptr_t>(x_c.data_ptr()) % 4 != 0) {
        x_c = x_c.clone();
    }
    
    auto w_c = weight.contiguous();

    const int64_t M = x_c.size(0);
    const int64_t D = x_c.size(1);
    const int64_t N = w_c.size(0);
    TORCH_CHECK(w_c.size(1) == D, "dimension mismatch");

    const c10::cuda::CUDAGuard device_guard(x_c.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto r = torch::empty({M, 1}, x_c.options().dtype(at::kFloat));
    // Use empty to match original behavior, kernel overwrites all
    auto x_f = torch::empty({M, D}, x_c.options().dtype(at::kFloat));
    auto proj = torch::empty({M, N}, x_c.options().dtype(at::kFloat));


    const float inv_sqrt_d = rsqrtf(static_cast<float>(D));
    const int threads = (D >= 2048) ? 128 : 64;
    
    // Always use the kernel (dropped vectorization inside kernel for safety)
    if (threads == 128) {
        cast_and_row_l2norm_kernel<128><<<static_cast<unsigned int>(M), 128, 0, stream.stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x_c.data_ptr<at::BFloat16>()),
            x_f.data_ptr<float>(), r.data_ptr<float>(), D, inv_sqrt_d
        );
    } else {
        cast_and_row_l2norm_kernel<64><<<static_cast<unsigned int>(M), 64, 0, stream.stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x_c.data_ptr<at::BFloat16>()),
            x_f.data_ptr<float>(), r.data_ptr<float>(), D, inv_sqrt_d
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    CUBLAS_CHECK(cublasSetStream(handle, stream.stream()));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        static_cast<int>(N), static_cast<int>(M), static_cast<int>(D),
        &alpha, w_c.data_ptr<float>(), CUDA_R_32F, static_cast<int>(D),
        x_f.data_ptr<float>(), CUDA_R_32F, static_cast<int>(D),
        &beta, proj.data_ptr<float>(), CUDA_R_32F, static_cast<int>(N),
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    return std::make_tuple(r, proj);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("aggregate_cuda", &aggregate_cuda, "Aggregate Op");
    m.def("sinkhorn_cuda", &sinkhorn_cuda, "Sinkhorn Op");
    m.def("expand_merge_cuda", &expand_merge_cuda, "ExpandMerge Op");
    m.def("map_sigmoid_cuda", &map_sigmoid_cuda, "MapSigmoid Op");
    m.def("norm_linear_cuda", &norm_linear_cuda, "NormLinear Op");
}
"""


# =========================================================================================
# Unified nn.Module Class
# =========================================================================================
class MHCCudaOps(nn.Module):
    """
    Unified CUDA operators module for MHC models.
    
    This class wraps all CUDA kernels into a single nn.Module for better 
    performance and PyTorch integration.
    """
    
    def __init__(self):
        super().__init__()
        self.debug_check = False
        
        # Compile all kernels once
        # Bump version to v3 to force recompile
        self._module = load_inline(
            name="mhc_cuda_ops_unified",
            cpp_sources="",
            cuda_sources=_combined_cuda_source,
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math", "--expt-relaxed-constexpr", "-maxrregcount=64"],
        )
    
    def _ensure_contiguous(self, *tensors):
        """Helper to ensure all tensors are contiguous."""
        return [t.contiguous() if not t.is_contiguous() else t for t in tensors]
        
    def _check_tensor(self, name, tensor):
        if not self.debug_check:
            return
        if tensor is None:
            return
        
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"[MHC_DEBUG] {name} has Inf/NaN!")
            try:
                print(f"  Shape: {tensor.shape}, Min: {tensor.min()}, Max: {tensor.max()}")
            except:
                pass

    def aggregate(self, residuals, h_pre):
        """
        Perform aggregation operation.
        
        Args:
            residuals: [M, n_streams, D], bf16
            h_pre: [M, n_streams], fp32
        Returns:
            out: [M, D], bf16
        """
        residuals, h_pre = self._ensure_contiguous(residuals, h_pre)
        self._check_tensor("aggregate_input_residuals", residuals)
        self._check_tensor("aggregate_input_h_pre", h_pre)
        
        out = self._module.aggregate_cuda(residuals, h_pre)
        
        self._check_tensor("aggregate_output", out)
        return out
    
    def sinkhorn(self, logits, n_iters=20):
        """
        Perform Sinkhorn algorithm.
        
        Args:
            logits: [M, n_streams, n_streams], fp32
            n_iters: int, default 20
        Returns:
            matrix: [M, n_streams, n_streams], fp32
        """
        (logits,) = self._ensure_contiguous(logits)
        self._check_tensor("sinkhorn_input_logits", logits)

        out = self._module.sinkhorn_cuda(logits, n_iters)
        
        self._check_tensor("sinkhorn_output", out)
        return out
    
    def expand_merge(self, residuals, layer_output, h_res, h_post):
        """
        Perform fused Expand and Merge operation.
        
        Args:
            residuals: [M, n_streams, D], bf16
            layer_output: [M, D], bf16
            h_res: [M, n_streams, n_streams], fp32
            h_post: [M, n_streams], fp32
        Returns:
            mixed: [M, n_streams, D], bf16
        """
        residuals, layer_output, h_res, h_post = self._ensure_contiguous(
            residuals, layer_output, h_res, h_post
        )
        
        self._check_tensor("expand_merge_input_residuals", residuals)
        self._check_tensor("expand_merge_input_layer_output", layer_output)
        self._check_tensor("expand_merge_input_h_res", h_res)
        self._check_tensor("expand_merge_input_h_post", h_post)

        out = self._module.expand_merge_cuda(h_res, residuals, layer_output, h_post)
        
        self._check_tensor("expand_merge_output", out)
        return out
    
    def map_sigmoid(self, r, proj, bias, alpha_pre, alpha_post, alpha_res, n_streams):
        """
        Perform mapped sigmoid operation.
        
        Args:
            r: [M, 1], fp32
            proj: [M, 2 * n_streams + n_streams * n_streams], fp32
            bias: [2 * n_streams + n_streams * n_streams], fp32
            alpha_pre: [1, ], fp32
            alpha_post: [1, ], fp32
            alpha_res: [1, ], fp32
            n_streams: int
        Returns:
            h_pre: [M, n_streams], fp32
            h_post: [M, n_streams], fp32
            h_res: [M, n_streams, n_streams], fp32
        """
        # Expand and concatenate alpha components
        alpha = torch.cat([
            alpha_pre.expand(n_streams),
            alpha_post.expand(n_streams),
            alpha_res.expand(n_streams * n_streams)
        ], dim=-1).contiguous()
        
        r, proj, bias, alpha = self._ensure_contiguous(r, proj, bias, alpha)
        
        self._check_tensor("map_sigmoid_input_r", r)
        self._check_tensor("map_sigmoid_input_proj", proj)
        self._check_tensor("map_sigmoid_input_bias", bias)
        self._check_tensor("map_sigmoid_input_alpha", alpha)

        M = r.size(0)
        
        # Allocate output tensors directly to avoid cache pollution issues
        # Use empty to match original behavior, avoids kernel launch overhead
        out_pre = torch.empty((M, n_streams), device=r.device, dtype=torch.float32)
        out_post = torch.empty((M, n_streams), device=r.device, dtype=torch.float32)
        out_res = torch.empty((M, n_streams, n_streams), device=r.device, dtype=torch.float32)

        self._module.map_sigmoid_cuda(
            r, proj, bias, alpha,
            out_pre, out_post, out_res,
            n_streams
        )
        
        self._check_tensor("map_sigmoid_output_pre", out_pre)
        self._check_tensor("map_sigmoid_output_post", out_post)
        self._check_tensor("map_sigmoid_output_res", out_res)

        return out_pre, out_post, out_res
    
    def norm_linear(self, x, weight):
        """
        Perform fused Norm and Linear operation.
        
        Args:
            x: [M, n_streams, D], bf16
            weight: [2 * n_streams + n_streams * n_streams, n_streams * D], fp32
        Returns:
            r: [M, 1], fp32
            proj: [M, 2 * n_streams + n_streams * n_streams], fp32
        """
        x, weight = self._ensure_contiguous(x, weight)
        self._check_tensor("norm_linear_input_x", x)
        self._check_tensor("norm_linear_input_weight", weight)

        x_flat = x.flatten(1)
        r, proj = self._module.norm_linear_cuda(x_flat, weight)
        
        self._check_tensor("norm_linear_output_r", r)
        self._check_tensor("norm_linear_output_proj", proj)
        return r, proj
    
    def forward(self, *args, **kwargs):
        """
        Forward pass - not used directly, call specific methods instead.
        """
        raise NotImplementedError(
            "MHCCudaOps is a collection of operators. "
            "Call specific methods like .aggregate(), .sinkhorn(), etc."
        )


# =========================================================================================
# Convenience Functions (compatible with old API)
# =========================================================================================
_global_ops = None

def _get_ops():
    """Get or create the global ops instance."""
    global _global_ops
    if _global_ops is None:
        _global_ops = MHCCudaOps()
    return _global_ops


def mhc_cuda_aggregate(residuals, h_pre):
    """compatible function."""
    return _get_ops().aggregate(residuals, h_pre)


def mhc_cuda_sinkhorn(logits, n_iters=20):
    """compatible function."""
    return _get_ops().sinkhorn(logits, n_iters)


def mhc_cuda_expand_merge(residuals, layer_output, h_res, h_post):
    """compatible function."""
    return _get_ops().expand_merge(residuals, layer_output, h_res, h_post)


def mhc_cuda_map_sigmoid(r, proj, bias, alpha_pre, alpha_post, alpha_res, n_streams):
    """compatible function."""
    return _get_ops().map_sigmoid(r, proj, bias, alpha_pre, alpha_post, alpha_res, n_streams)


def mhc_cuda_norm_linear(x, weight):
    """compatible function."""
    return _get_ops().norm_linear(x, weight)
