"""
Sparse KV Cache Manager - Single Fused CUDA Kernel with PyTorch JIT

This module provides a JIT-compiled CUDA kernel for sparse attention KV cache management.
Uses bitmap-based residence tracking for O(1) lookups.

Key features:
1. Single fused kernel for all operations (no multi-kernel overhead)
2. Bitmap-based O(1) residence checking
3. PyTorch JIT compilation for easy iteration
4. Parallel bitmap updates overlapped with copy preparation

Data Structures:
- residence_bitmap: [max_tokens] int8 - 1 if token on GPU, 0 otherwise
- token_to_gpu_loc: [max_tokens] int32 - GPU location for each token (-1 if not resident)
- gpu_loc_to_token: [hot_buffer_size] int32 - Token at each GPU location (-1 if empty)
"""

import torch
from torch.utils.cpp_extension import load_inline
from typing import Tuple, Optional
from dataclasses import dataclass
import os

# CUDA source code for the fused sparse cache manager kernel
CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Configuration constants
constexpr int WARP_SIZE = 32;
constexpr int MAX_TOP_K = 256;
constexpr int MAX_HOT_BUFFER = 4096;

/**
 * Fused Sparse Cache Manager Kernel
 * 
 * Single kernel that performs:
 * 1. O(1) bitmap lookup for cache hits
 * 2. Parallel identification of eviction candidates  
 * 3. Atomic assignment of eviction slots to misses
 * 4. In-place bitmap and mapping updates
 * 5. Copy list preparation
 *
 * Uses shared memory for coordination within a block.
 */
template <int BLOCK_SIZE>
__global__ void sparse_cache_manager_fused_kernel(
    // Inputs
    const int32_t* __restrict__ top_k_indices,      // [K] tokens we need
    const int64_t* __restrict__ cache_cpu_locs,     // [N] CPU location for each token
    // State (modified in-place)
    int8_t* __restrict__ residence_bitmap,          // [N] 1 if token on GPU
    int32_t* __restrict__ token_to_gpu_loc,         // [N] GPU loc for each token
    int32_t* __restrict__ gpu_loc_to_token,         // [H] token at each GPU loc
    // Outputs
    int32_t* __restrict__ top_k_gpu_locs,           // [K] GPU locations for top_k
    int64_t* __restrict__ copy_src_cpu_locs,        // [K] CPU locs to copy from
    int64_t* __restrict__ copy_dst_gpu_locs,        // [K] GPU locs to copy to
    int32_t* __restrict__ num_copies,               // [1] number of copies
    // Sizes
    int32_t top_k_size,
    int32_t hot_buffer_size,
    int32_t max_tokens
) {
    // Shared memory for coordination
    __shared__ int32_t s_miss_tokens[MAX_TOP_K];      // Tokens that are misses
    __shared__ int32_t s_miss_indices[MAX_TOP_K];     // Their indices in top_k
    __shared__ int32_t s_evictable_locs[MAX_HOT_BUFFER]; // GPU locs that can be evicted
    __shared__ int32_t s_num_misses;
    __shared__ int32_t s_num_evictable;
    __shared__ int32_t s_eviction_ptr;
    __shared__ int32_t s_copy_count;
    
    const int tid = threadIdx.x;
    
    // Initialize shared counters
    if (tid == 0) {
        s_num_misses = 0;
        s_num_evictable = 0;
        s_eviction_ptr = 0;
        s_copy_count = 0;
    }
    __syncthreads();
    
    // ========== Phase 1: Check hits using bitmap (O(1) per token) ==========
    // Each thread handles one or more top_k tokens
    for (int k = tid; k < top_k_size; k += BLOCK_SIZE) {
        int32_t token = top_k_indices[k];
        
        // O(1) bitmap lookup
        int8_t is_resident = residence_bitmap[token];
        
        if (is_resident) {
            // Hit - get GPU location directly
            int32_t gpu_loc = token_to_gpu_loc[token];
            top_k_gpu_locs[k] = gpu_loc;
        } else {
            // Miss - record for later processing
            int32_t miss_idx = atomicAdd(&s_num_misses, 1);
            s_miss_tokens[miss_idx] = token;
            s_miss_indices[miss_idx] = k;
            top_k_gpu_locs[k] = -1;  // Temporary marker
        }
    }
    __syncthreads();
    
    // Early exit if all hits
    if (s_num_misses == 0) {
        if (tid == 0) {
            *num_copies = 0;
        }
        return;
    }
    
    // ========== Phase 2: Find evictable GPU locations ==========
    // A location is evictable if its token is NOT in top_k
    // We need to check this efficiently
    
    // First, build a quick lookup for top_k tokens in shared memory
    // For small top_k, linear scan is fine; for larger, could use hash
    __shared__ int32_t s_top_k_tokens[MAX_TOP_K];
    for (int k = tid; k < top_k_size; k += BLOCK_SIZE) {
        s_top_k_tokens[k] = top_k_indices[k];
    }
    __syncthreads();
    
    // Each thread checks a range of GPU locations
    for (int h = tid; h < hot_buffer_size; h += BLOCK_SIZE) {
        int32_t existing_token = gpu_loc_to_token[h];
        
        // Check if this location can be evicted
        bool can_evict = true;
        
        if (existing_token >= 0) {
            // Check if token is needed (in top_k)
            for (int k = 0; k < top_k_size; k++) {
                if (s_top_k_tokens[k] == existing_token) {
                    can_evict = false;
                    break;
                }
            }
        }
        // Empty slots (existing_token == -1) can also be used
        
        if (can_evict) {
            int32_t evict_idx = atomicAdd(&s_num_evictable, 1);
            if (evict_idx < MAX_HOT_BUFFER) {
                s_evictable_locs[evict_idx] = h;
            }
        }
    }
    __syncthreads();
    
    // ========== Phase 3: Assign eviction slots and update state ==========
    // Each thread handles one miss
    for (int m = tid; m < s_num_misses; m += BLOCK_SIZE) {
        // Claim an eviction slot atomically
        int32_t evict_slot = atomicAdd(&s_eviction_ptr, 1);
        
        if (evict_slot < s_num_evictable) {
            int32_t gpu_loc = s_evictable_locs[evict_slot];
            int32_t new_token = s_miss_tokens[m];
            int32_t top_k_idx = s_miss_indices[m];
            
            // Get old token at this location (for clearing)
            int32_t old_token = gpu_loc_to_token[gpu_loc];
            
            // === Update all mappings atomically ===
            
            // 1. Clear old token from bitmap (if valid)
            if (old_token >= 0 && old_token < max_tokens) {
                residence_bitmap[old_token] = 0;
                token_to_gpu_loc[old_token] = -1;
            }
            
            // 2. Set new token in bitmap
            residence_bitmap[new_token] = 1;
            token_to_gpu_loc[new_token] = gpu_loc;
            
            // 3. Update GPU loc -> token mapping
            gpu_loc_to_token[gpu_loc] = new_token;
            
            // 4. Set output GPU location
            top_k_gpu_locs[top_k_idx] = gpu_loc;
            
            // 5. Add to copy list
            int32_t copy_idx = atomicAdd(&s_copy_count, 1);
            int64_t cpu_loc = cache_cpu_locs[new_token];
            copy_src_cpu_locs[copy_idx] = cpu_loc;
            copy_dst_gpu_locs[copy_idx] = gpu_loc;
        }
    }
    __syncthreads();
    
    // Write final copy count
    if (tid == 0) {
        *num_copies = s_copy_count;
    }
}

/**
 * Launcher function with dynamic block size selection
 */
void sparse_cache_manager_cuda(
    torch::Tensor top_k_indices,
    torch::Tensor cache_cpu_locs,
    torch::Tensor residence_bitmap,
    torch::Tensor token_to_gpu_loc,
    torch::Tensor gpu_loc_to_token,
    torch::Tensor top_k_gpu_locs,
    torch::Tensor copy_src_cpu_locs,
    torch::Tensor copy_dst_gpu_locs,
    torch::Tensor num_copies
) {
    const int32_t top_k_size = top_k_indices.size(0);
    const int32_t hot_buffer_size = gpu_loc_to_token.size(0);
    const int32_t max_tokens = residence_bitmap.size(0);
    
    // Reset num_copies
    num_copies.zero_();
    
    // Choose block size based on problem size
    const int block_size = 256;
    
    // Launch kernel (single block for coordination via shared memory)
    sparse_cache_manager_fused_kernel<256><<<1, block_size>>>(
        top_k_indices.data_ptr<int32_t>(),
        cache_cpu_locs.data_ptr<int64_t>(),
        residence_bitmap.data_ptr<int8_t>(),
        token_to_gpu_loc.data_ptr<int32_t>(),
        gpu_loc_to_token.data_ptr<int32_t>(),
        top_k_gpu_locs.data_ptr<int32_t>(),
        copy_src_cpu_locs.data_ptr<int64_t>(),
        copy_dst_gpu_locs.data_ptr<int64_t>(),
        num_copies.data_ptr<int32_t>(),
        top_k_size,
        hot_buffer_size,
        max_tokens
    );
}

/**
 * Multi-block version for large hot buffers
 * Uses two-phase approach with global memory coordination
 */
__global__ void sparse_cache_manager_large_kernel_phase1(
    const int32_t* __restrict__ top_k_indices,
    int8_t* __restrict__ residence_bitmap,
    int32_t* __restrict__ token_to_gpu_loc,
    int32_t* __restrict__ top_k_gpu_locs,
    int32_t* __restrict__ miss_tokens,
    int32_t* __restrict__ miss_indices,
    int32_t* __restrict__ num_misses,
    int32_t top_k_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < top_k_size) {
        int32_t token = top_k_indices[tid];
        int8_t is_resident = residence_bitmap[token];
        
        if (is_resident) {
            top_k_gpu_locs[tid] = token_to_gpu_loc[token];
        } else {
            int32_t miss_idx = atomicAdd(num_misses, 1);
            miss_tokens[miss_idx] = token;
            miss_indices[miss_idx] = tid;
            top_k_gpu_locs[tid] = -1;
        }
    }
}

__global__ void sparse_cache_manager_large_kernel_phase2(
    const int32_t* __restrict__ top_k_indices,
    int32_t* __restrict__ gpu_loc_to_token,
    int32_t* __restrict__ evictable_locs,
    int32_t* __restrict__ num_evictable,
    int32_t top_k_size,
    int32_t hot_buffer_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < hot_buffer_size) {
        int32_t existing_token = gpu_loc_to_token[tid];
        bool can_evict = true;
        
        if (existing_token >= 0) {
            for (int k = 0; k < top_k_size; k++) {
                if (top_k_indices[k] == existing_token) {
                    can_evict = false;
                    break;
                }
            }
        }
        
        if (can_evict) {
            int32_t idx = atomicAdd(num_evictable, 1);
            evictable_locs[idx] = tid;
        }
    }
}

__global__ void sparse_cache_manager_large_kernel_phase3(
    const int64_t* __restrict__ cache_cpu_locs,
    int8_t* __restrict__ residence_bitmap,
    int32_t* __restrict__ token_to_gpu_loc,
    int32_t* __restrict__ gpu_loc_to_token,
    int32_t* __restrict__ top_k_gpu_locs,
    const int32_t* __restrict__ miss_tokens,
    const int32_t* __restrict__ miss_indices,
    const int32_t* __restrict__ evictable_locs,
    int64_t* __restrict__ copy_src_cpu_locs,
    int64_t* __restrict__ copy_dst_gpu_locs,
    int32_t* __restrict__ num_copies,
    int32_t num_misses,
    int32_t max_tokens
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_misses) {
        int32_t gpu_loc = evictable_locs[tid];
        int32_t new_token = miss_tokens[tid];
        int32_t top_k_idx = miss_indices[tid];
        int32_t old_token = gpu_loc_to_token[gpu_loc];
        
        // Clear old token
        if (old_token >= 0 && old_token < max_tokens) {
            residence_bitmap[old_token] = 0;
            token_to_gpu_loc[old_token] = -1;
        }
        
        // Set new token
        residence_bitmap[new_token] = 1;
        token_to_gpu_loc[new_token] = gpu_loc;
        gpu_loc_to_token[gpu_loc] = new_token;
        top_k_gpu_locs[top_k_idx] = gpu_loc;
        
        // Add to copy list
        int32_t copy_idx = atomicAdd(num_copies, 1);
        copy_src_cpu_locs[copy_idx] = cache_cpu_locs[new_token];
        copy_dst_gpu_locs[copy_idx] = gpu_loc;
    }
}

void sparse_cache_manager_large_cuda(
    torch::Tensor top_k_indices,
    torch::Tensor cache_cpu_locs,
    torch::Tensor residence_bitmap,
    torch::Tensor token_to_gpu_loc,
    torch::Tensor gpu_loc_to_token,
    torch::Tensor top_k_gpu_locs,
    torch::Tensor copy_src_cpu_locs,
    torch::Tensor copy_dst_gpu_locs,
    torch::Tensor num_copies
) {
    const int32_t top_k_size = top_k_indices.size(0);
    const int32_t hot_buffer_size = gpu_loc_to_token.size(0);
    const int32_t max_tokens = residence_bitmap.size(0);
    
    const int block_size = 256;
    
    // Temporary buffers
    auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(top_k_indices.device());
    auto miss_tokens = torch::zeros({top_k_size}, options_int32);
    auto miss_indices = torch::zeros({top_k_size}, options_int32);
    auto num_misses = torch::zeros({1}, options_int32);
    auto evictable_locs = torch::zeros({hot_buffer_size}, options_int32);
    auto num_evictable = torch::zeros({1}, options_int32);
    num_copies.zero_();
    
    // Phase 1: Check hits
    int grid1 = (top_k_size + block_size - 1) / block_size;
    sparse_cache_manager_large_kernel_phase1<<<grid1, block_size>>>(
        top_k_indices.data_ptr<int32_t>(),
        residence_bitmap.data_ptr<int8_t>(),
        token_to_gpu_loc.data_ptr<int32_t>(),
        top_k_gpu_locs.data_ptr<int32_t>(),
        miss_tokens.data_ptr<int32_t>(),
        miss_indices.data_ptr<int32_t>(),
        num_misses.data_ptr<int32_t>(),
        top_k_size
    );
    
    int32_t h_num_misses = num_misses.item<int32_t>();
    if (h_num_misses == 0) return;
    
    // Phase 2: Find evictable
    int grid2 = (hot_buffer_size + block_size - 1) / block_size;
    sparse_cache_manager_large_kernel_phase2<<<grid2, block_size>>>(
        top_k_indices.data_ptr<int32_t>(),
        gpu_loc_to_token.data_ptr<int32_t>(),
        evictable_locs.data_ptr<int32_t>(),
        num_evictable.data_ptr<int32_t>(),
        top_k_size,
        hot_buffer_size
    );
    
    // Phase 3: Assign and update
    int grid3 = (h_num_misses + block_size - 1) / block_size;
    sparse_cache_manager_large_kernel_phase3<<<grid3, block_size>>>(
        cache_cpu_locs.data_ptr<int64_t>(),
        residence_bitmap.data_ptr<int8_t>(),
        token_to_gpu_loc.data_ptr<int32_t>(),
        gpu_loc_to_token.data_ptr<int32_t>(),
        top_k_gpu_locs.data_ptr<int32_t>(),
        miss_tokens.data_ptr<int32_t>(),
        miss_indices.data_ptr<int32_t>(),
        evictable_locs.data_ptr<int32_t>(),
        copy_src_cpu_locs.data_ptr<int64_t>(),
        copy_dst_gpu_locs.data_ptr<int64_t>(),
        num_copies.data_ptr<int32_t>(),
        h_num_misses,
        max_tokens
    );
}

/**
 * Copy kernel using warp-level parallelism
 */
__global__ void sparse_cache_copy_kernel(
    const char* __restrict__ cpu_cache,
    char* __restrict__ gpu_cache,
    const int64_t* __restrict__ copy_src_locs,
    const int64_t* __restrict__ copy_dst_locs,
    int32_t num_copies,
    int64_t item_size_bytes
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    if (warp_id >= num_copies) return;
    
    int64_t src_loc = copy_src_locs[warp_id];
    int64_t dst_loc = copy_dst_locs[warp_id];
    
    const uint64_t* src = reinterpret_cast<const uint64_t*>(cpu_cache + src_loc * item_size_bytes);
    uint64_t* dst = reinterpret_cast<uint64_t*>(gpu_cache + dst_loc * item_size_bytes);
    
    int num_chunks = item_size_bytes / sizeof(uint64_t);
    for (int i = lane_id; i < num_chunks; i += WARP_SIZE) {
        dst[i] = src[i];
    }
}

void sparse_cache_copy_cuda(
    torch::Tensor cpu_cache,
    torch::Tensor gpu_cache,
    torch::Tensor copy_src_locs,
    torch::Tensor copy_dst_locs,
    int32_t num_copies,
    int64_t item_size_bytes
) {
    if (num_copies == 0) return;
    
    const int block_size = 256;
    const int warps_per_block = block_size / WARP_SIZE;
    const int num_blocks = (num_copies + warps_per_block - 1) / warps_per_block;
    
    sparse_cache_copy_kernel<<<num_blocks, block_size>>>(
        reinterpret_cast<const char*>(cpu_cache.data_ptr()),
        reinterpret_cast<char*>(gpu_cache.data_ptr()),
        copy_src_locs.data_ptr<int64_t>(),
        copy_dst_locs.data_ptr<int64_t>(),
        num_copies,
        item_size_bytes
    );
}
'''

CPP_SOURCE = r'''
void sparse_cache_manager_cuda(
    torch::Tensor top_k_indices,
    torch::Tensor cache_cpu_locs,
    torch::Tensor residence_bitmap,
    torch::Tensor token_to_gpu_loc,
    torch::Tensor gpu_loc_to_token,
    torch::Tensor top_k_gpu_locs,
    torch::Tensor copy_src_cpu_locs,
    torch::Tensor copy_dst_gpu_locs,
    torch::Tensor num_copies
);

void sparse_cache_manager_large_cuda(
    torch::Tensor top_k_indices,
    torch::Tensor cache_cpu_locs,
    torch::Tensor residence_bitmap,
    torch::Tensor token_to_gpu_loc,
    torch::Tensor gpu_loc_to_token,
    torch::Tensor top_k_gpu_locs,
    torch::Tensor copy_src_cpu_locs,
    torch::Tensor copy_dst_gpu_locs,
    torch::Tensor num_copies
);

void sparse_cache_copy_cuda(
    torch::Tensor cpu_cache,
    torch::Tensor gpu_cache,
    torch::Tensor copy_src_locs,
    torch::Tensor copy_dst_locs,
    int32_t num_copies,
    int64_t item_size_bytes
);
'''

# Global variable to hold compiled module
_sparse_cache_module = None


def _get_sparse_cache_module():
    """Lazily compile and cache the CUDA module."""
    global _sparse_cache_module
    if _sparse_cache_module is None:
        _sparse_cache_module = load_inline(
            name='sparse_cache_manager_jit',
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=[
                'sparse_cache_manager_cuda',
                'sparse_cache_manager_large_cuda', 
                'sparse_cache_copy_cuda',
            ],
            verbose=False,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
        )
    return _sparse_cache_module


@dataclass
class CopyInfo:
    """Information about CPU→GPU copies needed."""
    src_cpu_locs: torch.Tensor  # [num_copies] CPU locations
    dst_gpu_locs: torch.Tensor  # [num_copies] GPU locations
    num_copies: int


class SparseCacheManagerCUDA:
    """
    Sparse KV Cache Manager with JIT-compiled CUDA kernel.
    
    Uses bitmap-based residence tracking for O(1) lookups.
    Single fused kernel minimizes launch overhead.
    
    Example:
        manager = SparseCacheManagerCUDA(max_tokens=4096, hot_buffer_size=512)
        
        # Each iteration
        gpu_locs, copy_info = manager.process_topk(top_k_indices, cache_cpu_locs)
        
        # Execute copies
        manager.execute_copies(cpu_cache, gpu_cache, copy_info, item_size_bytes)
    """
    
    def __init__(
        self,
        max_tokens: int,
        hot_buffer_size: int,
        device: str = "cuda",
    ):
        """
        Initialize the sparse cache manager.
        
        Args:
            max_tokens: Maximum number of tokens per request
            hot_buffer_size: Size of GPU hot buffer
            device: CUDA device
        """
        self.max_tokens = max_tokens
        self.hot_buffer_size = hot_buffer_size
        self.device = device
        
        # Use fused kernel for small sizes, multi-phase for large
        self.use_fused = hot_buffer_size <= 4096 and max_tokens <= 256 * 1024
        
        # Bitmap: 1 if token is on GPU
        self.residence_bitmap = torch.zeros(
            max_tokens, dtype=torch.int8, device=device
        )
        
        # Token -> GPU location (-1 if not resident)
        self.token_to_gpu_loc = torch.full(
            (max_tokens,), -1, dtype=torch.int32, device=device
        )
        
        # GPU location -> Token (-1 if empty)
        self.gpu_loc_to_token = torch.full(
            (hot_buffer_size,), -1, dtype=torch.int32, device=device
        )
        
        # Output buffers (reused across calls)
        self._copy_src = torch.zeros(hot_buffer_size, dtype=torch.int64, device=device)
        self._copy_dst = torch.zeros(hot_buffer_size, dtype=torch.int64, device=device)
        self._num_copies = torch.zeros(1, dtype=torch.int32, device=device)
        
        # Compile kernel on first use
        self._module = None
    
    def _ensure_compiled(self):
        """Ensure CUDA kernel is compiled."""
        if self._module is None:
            self._module = _get_sparse_cache_module()
    
    def reset(self):
        """Reset all state."""
        self.residence_bitmap.zero_()
        self.token_to_gpu_loc.fill_(-1)
        self.gpu_loc_to_token.fill_(-1)
    
    def process_topk(
        self,
        top_k_indices: torch.Tensor,
        cache_cpu_locations: torch.Tensor,
    ) -> Tuple[torch.Tensor, CopyInfo]:
        """
        Process top-k tokens for sparse attention.
        
        Identifies hits/misses, assigns eviction slots, updates bitmap.
        
        Args:
            top_k_indices: [K] Token indices needed for this iteration (int32)
            cache_cpu_locations: [N] CPU memory location for each token (int64)
            
        Returns:
            top_k_gpu_locs: [K] GPU locations for all top_k tokens (int32)
            copy_info: Information about copies needed
        """
        self._ensure_compiled()
        
        top_k_size = top_k_indices.numel()
        
        # Ensure correct dtype
        if top_k_indices.dtype != torch.int32:
            top_k_indices = top_k_indices.to(torch.int32)
        if cache_cpu_locations.dtype != torch.int64:
            cache_cpu_locations = cache_cpu_locations.to(torch.int64)
        
        # Output tensor
        top_k_gpu_locs = torch.zeros(top_k_size, dtype=torch.int32, device=self.device)
        
        # Reset copy counter
        self._num_copies.zero_()
        
        # Call appropriate kernel
        if self.use_fused and top_k_size <= 256:
            self._module.sparse_cache_manager_cuda(
                top_k_indices,
                cache_cpu_locations,
                self.residence_bitmap,
                self.token_to_gpu_loc,
                self.gpu_loc_to_token,
                top_k_gpu_locs,
                self._copy_src,
                self._copy_dst,
                self._num_copies,
            )
        else:
            self._module.sparse_cache_manager_large_cuda(
                top_k_indices,
                cache_cpu_locations,
                self.residence_bitmap,
                self.token_to_gpu_loc,
                self.gpu_loc_to_token,
                top_k_gpu_locs,
                self._copy_src,
                self._copy_dst,
                self._num_copies,
            )
        
        num_copies = self._num_copies.item()
        
        return top_k_gpu_locs, CopyInfo(
            src_cpu_locs=self._copy_src[:num_copies].clone(),
            dst_gpu_locs=self._copy_dst[:num_copies].clone(),
            num_copies=num_copies,
        )
    
    def execute_copies(
        self,
        cpu_cache: torch.Tensor,
        gpu_cache: torch.Tensor,
        copy_info: CopyInfo,
        item_size_bytes: int,
    ):
        """
        Execute CPU→GPU copies.
        
        Uses warp-level parallelism for efficient transfer.
        
        Args:
            cpu_cache: CPU pinned memory buffer
            gpu_cache: GPU memory buffer  
            copy_info: Copy information from process_topk
            item_size_bytes: Size of each cache entry in bytes
        """
        if copy_info.num_copies == 0:
            return
        
        self._ensure_compiled()
        
        self._module.sparse_cache_copy_cuda(
            cpu_cache,
            gpu_cache,
            copy_info.src_cpu_locs,
            copy_info.dst_gpu_locs,
            copy_info.num_copies,
            item_size_bytes,
        )
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        num_resident = self.residence_bitmap.sum().item()
        return {
            "num_resident": num_resident,
            "hot_buffer_size": self.hot_buffer_size,
            "occupancy": num_resident / self.hot_buffer_size,
            "kernel_type": "fused" if self.use_fused else "multi-phase",
        }
    
    def prefill_tokens(self, tokens: torch.Tensor, gpu_locations: torch.Tensor):
        """
        Pre-fill hot buffer with tokens (e.g., from prefetch).
        
        Args:
            tokens: [N] Token indices to add
            gpu_locations: [N] GPU locations to assign
        """
        tokens_cpu = tokens.cpu().numpy()
        locs_cpu = gpu_locations.cpu().numpy()
        
        for tok, loc in zip(tokens_cpu, locs_cpu):
            self.residence_bitmap[tok] = 1
            self.token_to_gpu_loc[tok] = loc
            self.gpu_loc_to_token[loc] = tok


# Convenience function for standalone use
def process_sparse_cache(
    top_k_indices: torch.Tensor,
    cache_cpu_locations: torch.Tensor,
    residence_bitmap: torch.Tensor,
    token_to_gpu_loc: torch.Tensor,
    gpu_loc_to_token: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Standalone function for sparse cache processing.
    
    Args:
        top_k_indices: [K] tokens needed (int32)
        cache_cpu_locations: [N] CPU locations (int64)
        residence_bitmap: [N] residence flags (int8, modified in-place)
        token_to_gpu_loc: [N] token->GPU mapping (int32, modified in-place)
        gpu_loc_to_token: [H] GPU->token mapping (int32, modified in-place)
        
    Returns:
        top_k_gpu_locs: [K] GPU locations
        copy_src: [M] CPU locations to copy from
        copy_dst: [M] GPU locations to copy to
        num_copies: Number of copies needed
    """
    module = _get_sparse_cache_module()
    
    top_k_size = top_k_indices.numel()
    hot_buffer_size = gpu_loc_to_token.numel()
    device = top_k_indices.device
    
    top_k_gpu_locs = torch.zeros(top_k_size, dtype=torch.int32, device=device)
    copy_src = torch.zeros(top_k_size, dtype=torch.int64, device=device)
    copy_dst = torch.zeros(top_k_size, dtype=torch.int64, device=device)
    num_copies = torch.zeros(1, dtype=torch.int32, device=device)
    
    if top_k_size <= 256 and hot_buffer_size <= 4096:
        module.sparse_cache_manager_cuda(
            top_k_indices.to(torch.int32),
            cache_cpu_locations.to(torch.int64),
            residence_bitmap,
            token_to_gpu_loc,
            gpu_loc_to_token,
            top_k_gpu_locs,
            copy_src,
            copy_dst,
            num_copies,
        )
    else:
        module.sparse_cache_manager_large_cuda(
            top_k_indices.to(torch.int32),
            cache_cpu_locations.to(torch.int64),
            residence_bitmap,
            token_to_gpu_loc,
            gpu_loc_to_token,
            top_k_gpu_locs,
            copy_src,
            copy_dst,
            num_copies,
        )
    
    n = num_copies.item()
    return top_k_gpu_locs, copy_src[:n], copy_dst[:n], n
