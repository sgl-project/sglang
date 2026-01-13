#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include "utils.h"

__global__ void manage_sparse_cache_kernel(
    const int64_t* __restrict__ top_k_indices,
    int64_t* __restrict__ hot_buffer_token_indices,
    int64_t* __restrict__ hot_buffer_device_locations,
    const int64_t* __restrict__ cache_cpu_locations,
    int64_t* __restrict__ top_k_device_locations,
    int K,
    int B,
    int* __restrict__ victim_count,
    int* __restrict__ miss_counter) {
    
    // Dynamic shared memory
    // Layout: is_hit[B] | victim_slots[K]
    extern __shared__ int shared_mem[];
    int* is_hit = shared_mem;
    int* victim_slots = is_hit + B;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Initialize is_hit to 0
    for (int i = tid; i < B; i += stride) {
        is_hit[i] = 0;
    }
    __syncthreads();

    // Phase 1: Check Hits
    for (int i = tid; i < K; i += stride) {
        int64_t token = top_k_indices[i];
        int64_t loc = -1;
        
        // Linear search
        for (int j = 0; j < B; ++j) {
            if (hot_buffer_token_indices[j] == token) {
                loc = hot_buffer_device_locations[j];
                is_hit[j] = 1; // Mark as hit
                break;
            }
        }
        top_k_device_locations[i] = loc;
    }
    __syncthreads();

    // Phase 2: Collect Victims
    // We need to find slots that are NOT hits.
    
    // Shared counters initialized by thread 0
    if (tid == 0) {
        *victim_count = 0;
        *miss_counter = 0;
    }
    __syncthreads();

    for (int j = tid; j < B; j += stride) {
        if (is_hit[j] == 0) {
            // Available for eviction
            // We only need up to K victims.
            int idx = atomicAdd(victim_count, 1);
            if (idx < K) {
                victim_slots[idx] = j;
            }
        }
    }
    __syncthreads();

    // Phase 3: Assign Victims to Misses
    for (int i = tid; i < K; i += stride) {
        if (top_k_device_locations[i] == -1) {
            // This is a miss
            int my_miss_idx = atomicAdd(miss_counter, 1);
            
            // Check if we have a victim
            // Since counters are global (passed as ptr), we need to read the value collected in phase 2.
            // Wait, I passed victim_count as int*, which points to global memory.
            // atomicAdd updates it. The value at *victim_count is the total victims found.
            
            if (my_miss_idx < *victim_count && my_miss_idx < K) {
                int v_idx = victim_slots[my_miss_idx];
                
                // Evict v_idx
                int64_t new_token = top_k_indices[i];
                
                // Update hot buffer
                hot_buffer_token_indices[v_idx] = new_token;
                // Reuse existing location
                int64_t gpu_loc = hot_buffer_device_locations[v_idx];
                
                top_k_device_locations[i] = gpu_loc;
                
                // Note: Copy initiation is implied.
            } else {
                // Not enough space or error. 
                // Should we set top_k_device_locations[i] to -2 or something?
                // For now leave as -1.
            }
        }
    }
}

void manage_sparse_cache(
    at::Tensor top_k_indices,
    at::Tensor hot_buffer_token_indices,
    at::Tensor hot_buffer_device_locations,
    at::Tensor cache_cpu_locations,
    at::Tensor top_k_device_locations) {
    
    int K = top_k_indices.numel();
    int B = hot_buffer_token_indices.numel();
    
    TORCH_CHECK(top_k_indices.is_cuda(), "top_k_indices must be CUDA");
    TORCH_CHECK(hot_buffer_token_indices.is_cuda(), "hot_buffer_token_indices must be CUDA");
    TORCH_CHECK(hot_buffer_device_locations.is_cuda(), "hot_buffer_device_locations must be CUDA");
    TORCH_CHECK(top_k_device_locations.is_cuda(), "top_k_device_locations must be CUDA");
    
    // Shared memory size: B*sizeof(int) + K*sizeof(int)
    int shared_mem_size = (B + K) * sizeof(int);
    
    // Allocate temporary tensor for counters
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(top_k_indices.device());
    auto counters = torch::zeros({2}, options);
    
    int threads = 256;
    int blocks = 1; 
    
    manage_sparse_cache_kernel<<<blocks, threads, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
        top_k_indices.data_ptr<int64_t>(),
        hot_buffer_token_indices.data_ptr<int64_t>(),
        hot_buffer_device_locations.data_ptr<int64_t>(),
        cache_cpu_locations.defined() ? cache_cpu_locations.data_ptr<int64_t>() : nullptr,
        top_k_device_locations.data_ptr<int64_t>(),
        K,
        B,
        counters.data_ptr<int>() + 0,
        counters.data_ptr<int>() + 1
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
