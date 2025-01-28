#include <cfloat>
#include <stdio.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <ATen/cuda/CUDAContext.h>

#include "utils.h"


template <int VPT, int NUM_EXPERTS, int THREADS_PER_ROW, int ROWS_PER_WARP, int ROWS_PER_CTA, int WARPS_PER_CTA>
__global__ void deepseekv3_fused_gate_kernel(void* input, void* bias, void* output, int64_t* indices_ptr, int num_rows, int k, cutlass::bfloat16_t route_scale) {

    int tidx = threadIdx.x;
    // thread row offset: block offset + warp_offset + within_group_offset
    int64_t const thread_row = blockIdx.x * ROWS_PER_CTA + threadIdx.y * ROWS_PER_WARP + tidx / THREADS_PER_ROW;

    // out of bound threads
    if (thread_row >= num_rows){
        return;
    }

    cutlass::bfloat16_t* input_ptr = reinterpret_cast<cutlass::bfloat16_t*>(input);
    cutlass::bfloat16_t* bias_ptr = reinterpret_cast<cutlass::bfloat16_t*>(bias);
    cutlass::bfloat16_t* output_ptr = reinterpret_cast<cutlass::bfloat16_t*>(output);

    cutlass::bfloat16_t* thread_row_ptr = input_ptr + thread_row * NUM_EXPERTS;

    int thread_group_idx = tidx % THREADS_PER_ROW;
    int first_elt_read_by_thread = thread_group_idx * VPT;
    cutlass::bfloat16_t* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    // Determine the pointer type to use to read in the data depending on the BYTES_PER_LDG template param.
    using AccessType = cutlass::AlignedArray<cutlass::bfloat16_t, VPT>;

    // Finally, we pull in the data from global mem
    cutlass::Array<cutlass::bfloat16_t, VPT> row_chunk;
    AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk);
    AccessType const* vec_thread_read_ptr = reinterpret_cast<AccessType const*>(thread_read_ptr);

    // Step 2: Add bias
    cutlass::bfloat16_t* bias_thread_read_ptr = bias_ptr + first_elt_read_by_thread;

    cutlass::Array<cutlass::bfloat16_t, VPT> bias_chunk;
    AccessType* bias_chunk_vec_ptr = reinterpret_cast<AccessType*>(&bias_chunk);
    AccessType const* vec_bias_thread_read_ptr = reinterpret_cast<AccessType const*>(bias_thread_read_ptr);

    row_chunk_vec_ptr[0] = vec_thread_read_ptr[0];
    bias_chunk_vec_ptr[0] = vec_bias_thread_read_ptr[0];

    ////////////////////// Sigmoid //////////////////////
    #pragma unroll
    for (int ii = 0; ii < VPT; ++ii){
        // row_chunk[ii] = 1.0f / (1.0f + expf(-row_chunk[ii]));
        row_chunk[ii] = static_cast<cutlass::bfloat16_t>(1.0f / (1.0f + expf(-float(row_chunk[ii]))));
    }
    __syncthreads();


    ////////////////////// Add Bias //////////////////////
    #pragma unroll
    for (int ii = 0; ii < VPT; ++ii){
        bias_chunk[ii] = row_chunk[ii] + bias_chunk[ii];
    }


    ////////////////////// Exclude Groups //////////////////////
    #pragma unroll
    for (int k_idx = 0; k_idx < 4; ++k_idx){
        int expert = first_elt_read_by_thread;
        // local argmax
        cutlass::bfloat16_t max_val = static_cast<cutlass::bfloat16_t>(-FLT_MAX);
        cutlass::bfloat16_t max_val_second = static_cast<cutlass::bfloat16_t>(-FLT_MAX);
        #pragma unroll
        for (int ii = 0; ii < VPT; ++ii){
            cutlass::bfloat16_t val = bias_chunk[ii];

            if (val > max_val){
                max_val_second = max_val;
                max_val = val;
            } else if (val > max_val_second){
                max_val_second = val;
            }
        }

        cutlass::bfloat16_t max_sum = max_val + max_val_second;

        // argmin reduce
        #pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2){
            cutlass::bfloat16_t other_max_sum = static_cast<cutlass::bfloat16_t>(__shfl_xor_sync(0xFFFFFFFF, static_cast<float>(max_sum), mask, THREADS_PER_ROW));
            int other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, THREADS_PER_ROW);

            // higher indices win
            if (other_max_sum < max_sum || (other_max_sum == max_sum && other_expert > expert)){
                max_sum = other_max_sum;
                expert = other_expert;
            }
        }

        // clear the max value in the thread
        if (k_idx < k){
            int const thread_to_clear_in_group = expert / VPT;

            if (thread_group_idx == thread_to_clear_in_group){
                #pragma unroll
                for (int ii = 0; ii < VPT; ++ii){
                    bias_chunk[ii] = static_cast<cutlass::bfloat16_t>(FLT_MAX);
                }
            }
        }
    }

    __syncthreads();

    ////////////////////// Topk //////////////////////
    float output_sum = 0.0f;
    for (int k_idx = 0; k_idx < k; ++k_idx){
        // local argmax
        cutlass::bfloat16_t max_val = bias_chunk[0];
        int expert = first_elt_read_by_thread;

        if (max_val != static_cast<cutlass::bfloat16_t>(FLT_MAX)){
            #pragma unroll
            for (int ii = 1; ii < VPT; ++ii){
                cutlass::bfloat16_t val = bias_chunk[ii];
                if (val > max_val){
                    max_val = val;
                    expert = first_elt_read_by_thread + ii;
                }
            }
        } else {
            max_val = static_cast<cutlass::bfloat16_t>(-FLT_MAX);
        }

        // argmax reduce
        #pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2){
            cutlass::bfloat16_t other_max = static_cast<cutlass::bfloat16_t>(__shfl_xor_sync(0xFFFFFFFF, static_cast<float>(max_val), mask, THREADS_PER_ROW));
            int other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, THREADS_PER_ROW);

            // lower indices to win
            if (other_max > max_val || (other_max == max_val && other_expert < expert)){
                max_val = other_max;
                expert = other_expert;
            }
        }

        if (k_idx < k){
            int thread_to_clear_in_group = expert / VPT;
            int64_t idx = k * thread_row + k_idx;
            
            if (thread_group_idx == thread_to_clear_in_group){
                int expert_to_clear_in_thread = expert % VPT;

                // clear the max value in the thread
                bias_chunk[expert_to_clear_in_thread] = static_cast<cutlass::bfloat16_t>(-FLT_MAX);
                
                // store output
                output_ptr[idx] = row_chunk[expert_to_clear_in_thread];
                indices_ptr[idx] = static_cast<int64_t>(expert);
            }

            // accumulate sum
            if (thread_group_idx == 0){
                output_sum += static_cast<float>(output_ptr[idx]);
            }
        }

        __syncthreads();
    }

    ////////////////////// Rescale Output //////////////////////
    if (thread_group_idx == 0){
        #pragma unroll
        for (int ii = 0; ii < k; ++ii){
            int64_t const idx = k * thread_row + ii;
            output_ptr[idx] = (output_ptr[idx] / static_cast<cutlass::bfloat16_t>(output_sum)) * route_scale;
        }
    }
}


std::vector<at::Tensor>
deepseekv3_fused_gate(at::Tensor& input, at::Tensor& bias, int64_t n_rows) {

    int num_rows = static_cast<int>(n_rows);

    static constexpr int WARPS_PER_CTA = 6;
    static constexpr int NUM_EXPERTS = 256;
    static constexpr int THREADS_PER_ROW = 8;
    static constexpr int k = 8;

    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

    auto output = torch::empty({num_rows, k}, options);
    auto indices = torch::empty({num_rows, k}, options.dtype(torch::kInt64));

    static constexpr int VPT = 32; // 32 = 256 / 8 (NUM_EXPERTS / N_EXPERTS_GROUP)
    static constexpr int ROWS_PER_WARP = 4;  // 4 = 32 / 8 (WARP_SIZE / N_EXPERTS_GROUP)
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;
    const cutlass::bfloat16_t route_scale = static_cast<cutlass::bfloat16_t>(2.5f);

    int64_t const num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    int64_t const num_blocks = (num_warps + WARPS_PER_CTA - 1) / WARPS_PER_CTA;

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 block_dim(32, WARPS_PER_CTA);
    deepseekv3_fused_gate_kernel<VPT, NUM_EXPERTS, THREADS_PER_ROW, ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA><<<num_blocks, block_dim, 0, stream>>>(
        input.data_ptr(), bias.data_ptr(), output.data_ptr(), indices.data_ptr<int64_t>(), num_rows, k, route_scale
    );

    CHECK_CUDA_SUCCESS(cudaDeviceSynchronize());

    return {output, indices};
}