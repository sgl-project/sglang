"""
Sparse KV Cache Manager v4 - Optimized Evictable Finding

Key optimization: Only find num_misses evictable slots, not all H.

Approaches:
1. Circular pointer with early exit: O(num_misses + num_hits) amortized
2. Warp-parallel circular scan: 32x speedup on the scan
3. Epoch-based tracking: Single comparison instead of bitmap lookup

Complexity improvement:
- v3: O(H) to build full evictable list
- v4: O((num_misses + num_hits) / 32) with warp parallelism
"""

import torch
from torch.utils.cpp_extension import load_inline
from typing import Tuple
from dataclasses import dataclass

CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int WARP_SIZE = 32;
constexpr int MAX_TOP_K = 256;
constexpr int MAX_HOT_BUFFER = 4096;
constexpr int MAX_WARPS = 32;

/**
 * v4 Kernel: Warp-Parallel Circular Scan for Evictable Finding
 * 
 * Instead of scanning all H locations, use circular pointer and stop
 * when we have enough evictable slots (exactly num_misses).
 * 
 * Complexity: O((num_misses + num_hits) / 32) for evictable finding
 * vs O(H / 256) in v3
 */
template <int BLOCK_SIZE>
__global__ void sparse_cache_manager_v4_kernel(
    const int32_t* __restrict__ top_k_indices,
    const int64_t* __restrict__ cache_cpu_locs,
    int8_t* __restrict__ residence_bitmap,
    int32_t* __restrict__ token_to_gpu_loc,
    int32_t* __restrict__ gpu_loc_to_token,
    int32_t* __restrict__ top_k_gpu_locs,
    int64_t* __restrict__ copy_src_cpu_locs,
    int64_t* __restrict__ copy_dst_gpu_locs,
    int32_t* __restrict__ num_copies,
    int32_t* __restrict__ evict_ptr,  // Persistent circular pointer
    int32_t top_k_size,
    int32_t hot_buffer_size,
    int32_t max_tokens
) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Shared memory
    __shared__ uint8_t s_protected[(MAX_HOT_BUFFER + 7) / 8];
    __shared__ int32_t s_warp_miss_count[MAX_WARPS];
    __shared__ int32_t s_warp_miss_prefix[MAX_WARPS + 1];
    __shared__ int32_t s_local_offset[MAX_TOP_K];
    __shared__ int32_t s_is_miss[MAX_TOP_K];
    __shared__ int32_t s_evictable[MAX_TOP_K];  // Only need num_misses slots!
    __shared__ int32_t s_total_misses;
    __shared__ int32_t s_current_ptr;
    
    // Initialize
    if (tid == 0) {
        s_total_misses = 0;
        s_current_ptr = *evict_ptr;
    }
    for (int i = tid; i < (hot_buffer_size + 7) / 8; i += BLOCK_SIZE) {
        s_protected[i] = 0;
    }
    for (int i = tid; i < MAX_WARPS; i += BLOCK_SIZE) {
        s_warp_miss_count[i] = 0;
    }
    __syncthreads();
    
    // ===== Phase 1: Warp-Parallel Hit Detection (same as v3) =====
    int warp_token_start = warp_id * WARP_SIZE;
    int my_token_idx = warp_token_start + lane_id;
    bool has_valid_token = my_token_idx < top_k_size;
    
    int32_t my_token = 0;
    bool is_hit = false;
    int32_t my_gpu_loc = -1;
    
    if (has_valid_token) {
        my_token = top_k_indices[my_token_idx];
        is_hit = (residence_bitmap[my_token] != 0);
        
        if (is_hit) {
            my_gpu_loc = token_to_gpu_loc[my_token];
            top_k_gpu_locs[my_token_idx] = my_gpu_loc;
            atomicOr((unsigned int*)&s_protected[my_gpu_loc / 8], 1u << (my_gpu_loc % 8));
        }
    }
    
    // Intra-warp communication
    unsigned int miss_mask = __ballot_sync(0xFFFFFFFF, has_valid_token && !is_hit);
    int warp_miss_count = __popc(miss_mask);
    unsigned int lanes_before_me = (1u << lane_id) - 1;
    int local_offset = __popc(miss_mask & lanes_before_me);
    
    if (has_valid_token && !is_hit) {
        s_local_offset[my_token_idx] = local_offset;
        s_is_miss[my_token_idx] = 1;
    } else if (has_valid_token) {
        s_is_miss[my_token_idx] = 0;
    }
    
    if (lane_id == 0) {
        s_warp_miss_count[warp_id] = warp_miss_count;
    }
    __syncthreads();
    
    // ===== Phase 2: Prefix Sum (same as v3) =====
    if (tid == 0) {
        s_warp_miss_prefix[0] = 0;
        int num_active_warps = (top_k_size + WARP_SIZE - 1) / WARP_SIZE;
        for (int w = 0; w < num_active_warps; w++) {
            s_warp_miss_prefix[w + 1] = s_warp_miss_prefix[w] + s_warp_miss_count[w];
        }
        s_total_misses = s_warp_miss_prefix[num_active_warps];
    }
    __syncthreads();
    
    if (s_total_misses == 0) {
        if (tid == 0) *num_copies = 0;
        return;
    }
    
    // ===== Phase 3: OPTIMIZED - Warp-Parallel Circular Scan =====
    // Only find exactly s_total_misses evictable slots
    // Use first warp for this scan
    
    if (warp_id == 0) {
        int found = 0;
        int base_ptr = s_current_ptr;
        int iterations = 0;
        int max_iterations = (hot_buffer_size + WARP_SIZE - 1) / WARP_SIZE;
        
        while (found < s_total_misses && iterations < max_iterations) {
            // Each lane checks one location
            int ptr = (base_ptr + lane_id) % hot_buffer_size;
            bool valid_ptr = (base_ptr + lane_id) < (base_ptr + hot_buffer_size);
            
            // Check if this location is protected
            bool is_protected = false;
            if (ptr < hot_buffer_size) {
                is_protected = (s_protected[ptr / 8] >> (ptr % 8)) & 1;
            }
            
            bool is_evictable = valid_ptr && !is_protected && (ptr < hot_buffer_size);
            
            // Warp ballot to find evictable locations
            unsigned int evict_mask = __ballot_sync(0xFFFFFFFF, is_evictable);
            int num_evictable_this_round = __popc(evict_mask);
            
            // How many can we use from this round?
            int need = s_total_misses - found;
            int use_count = min(num_evictable_this_round, need);
            
            // Each evictable lane computes its position
            if (is_evictable) {
                int my_pos = __popc(evict_mask & lanes_before_me);
                if (my_pos < use_count) {
                    s_evictable[found + my_pos] = ptr;
                }
            }
            
            found += use_count;
            base_ptr += WARP_SIZE;
            iterations++;
        }
        
        // Update circular pointer
        if (lane_id == 0) {
            *evict_ptr = base_ptr % hot_buffer_size;
        }
    }
    __syncthreads();
    
    // ===== Phase 4: Direct Assignment (same as v3) =====
    if (has_valid_token && s_is_miss[my_token_idx]) {
        int global_slot = s_warp_miss_prefix[warp_id] + s_local_offset[my_token_idx];
        
        if (global_slot < s_total_misses) {
            int32_t evict_loc = s_evictable[global_slot];
            int32_t old_token = gpu_loc_to_token[evict_loc];
            
            if (old_token >= 0 && old_token < max_tokens) {
                residence_bitmap[old_token] = 0;
                token_to_gpu_loc[old_token] = -1;
            }
            
            residence_bitmap[my_token] = 1;
            token_to_gpu_loc[my_token] = evict_loc;
            gpu_loc_to_token[evict_loc] = my_token;
            top_k_gpu_locs[my_token_idx] = evict_loc;
            
            copy_src_cpu_locs[global_slot] = cache_cpu_locs[my_token];
            copy_dst_gpu_locs[global_slot] = evict_loc;
        }
    }
    __syncthreads();
    
    if (tid == 0) {
        *num_copies = s_total_misses;
    }
}

/**
 * v4b: Epoch-Based Tracking
 * 
 * Instead of bitmap, use epoch counter for O(1) "recently used" check.
 * Each GPU location stores its last-used epoch.
 * 
 * Benefits:
 * - No bitmap clearing needed (just increment epoch)
 * - Single int16 comparison vs bit manipulation
 */
template <int BLOCK_SIZE>
__global__ void sparse_cache_manager_v4b_epoch_kernel(
    const int32_t* __restrict__ top_k_indices,
    const int64_t* __restrict__ cache_cpu_locs,
    int8_t* __restrict__ residence_bitmap,
    int32_t* __restrict__ token_to_gpu_loc,
    int32_t* __restrict__ gpu_loc_to_token,
    int16_t* __restrict__ last_used_epoch,  // [H] last epoch each loc was used
    int32_t* __restrict__ top_k_gpu_locs,
    int64_t* __restrict__ copy_src_cpu_locs,
    int64_t* __restrict__ copy_dst_gpu_locs,
    int32_t* __restrict__ num_copies,
    int32_t* __restrict__ evict_ptr,
    int16_t current_epoch,  // Current iteration epoch
    int32_t top_k_size,
    int32_t hot_buffer_size,
    int32_t max_tokens
) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    __shared__ int32_t s_warp_miss_count[MAX_WARPS];
    __shared__ int32_t s_warp_miss_prefix[MAX_WARPS + 1];
    __shared__ int32_t s_local_offset[MAX_TOP_K];
    __shared__ int32_t s_is_miss[MAX_TOP_K];
    __shared__ int32_t s_evictable[MAX_TOP_K];
    __shared__ int32_t s_total_misses;
    
    if (tid == 0) s_total_misses = 0;
    for (int i = tid; i < MAX_WARPS; i += BLOCK_SIZE) s_warp_miss_count[i] = 0;
    __syncthreads();
    
    // Phase 1: Hit detection + update epoch for hits
    int warp_token_start = warp_id * WARP_SIZE;
    int my_token_idx = warp_token_start + lane_id;
    bool has_valid_token = my_token_idx < top_k_size;
    
    int32_t my_token = 0;
    bool is_hit = false;
    
    if (has_valid_token) {
        my_token = top_k_indices[my_token_idx];
        is_hit = (residence_bitmap[my_token] != 0);
        
        if (is_hit) {
            int32_t gpu_loc = token_to_gpu_loc[my_token];
            top_k_gpu_locs[my_token_idx] = gpu_loc;
            // Update epoch - marks this location as "used this round"
            last_used_epoch[gpu_loc] = current_epoch;
        }
    }
    
    unsigned int miss_mask = __ballot_sync(0xFFFFFFFF, has_valid_token && !is_hit);
    int warp_miss_count = __popc(miss_mask);
    unsigned int lanes_before_me = (1u << lane_id) - 1;
    int local_offset = __popc(miss_mask & lanes_before_me);
    
    if (has_valid_token && !is_hit) {
        s_local_offset[my_token_idx] = local_offset;
        s_is_miss[my_token_idx] = 1;
    } else if (has_valid_token) {
        s_is_miss[my_token_idx] = 0;
    }
    
    if (lane_id == 0) s_warp_miss_count[warp_id] = warp_miss_count;
    __syncthreads();
    
    // Phase 2: Prefix sum
    if (tid == 0) {
        s_warp_miss_prefix[0] = 0;
        int num_active_warps = (top_k_size + WARP_SIZE - 1) / WARP_SIZE;
        for (int w = 0; w < num_active_warps; w++) {
            s_warp_miss_prefix[w + 1] = s_warp_miss_prefix[w] + s_warp_miss_count[w];
        }
        s_total_misses = s_warp_miss_prefix[num_active_warps];
    }
    __syncthreads();
    
    if (s_total_misses == 0) {
        if (tid == 0) *num_copies = 0;
        return;
    }
    
    // Phase 3: Warp-parallel circular scan with EPOCH check
    // A location is evictable if last_used_epoch[loc] != current_epoch
    if (warp_id == 0) {
        int found = 0;
        int base_ptr = *evict_ptr;
        int iterations = 0;
        int max_iterations = (hot_buffer_size + WARP_SIZE - 1) / WARP_SIZE;
        
        while (found < s_total_misses && iterations < max_iterations) {
            int ptr = (base_ptr + lane_id) % hot_buffer_size;
            
            // O(1) epoch check - no bitmap needed!
            bool is_evictable = (last_used_epoch[ptr] != current_epoch);
            
            unsigned int evict_mask = __ballot_sync(0xFFFFFFFF, is_evictable);
            int num_evictable = __popc(evict_mask);
            int need = s_total_misses - found;
            int use_count = min(num_evictable, need);
            
            if (is_evictable) {
                int my_pos = __popc(evict_mask & lanes_before_me);
                if (my_pos < use_count) {
                    s_evictable[found + my_pos] = ptr;
                }
            }
            
            found += use_count;
            base_ptr += WARP_SIZE;
            iterations++;
        }
        
        if (lane_id == 0) *evict_ptr = base_ptr % hot_buffer_size;
    }
    __syncthreads();
    
    // Phase 4: Assignment
    if (has_valid_token && s_is_miss[my_token_idx]) {
        int global_slot = s_warp_miss_prefix[warp_id] + s_local_offset[my_token_idx];
        
        if (global_slot < s_total_misses) {
            int32_t evict_loc = s_evictable[global_slot];
            int32_t old_token = gpu_loc_to_token[evict_loc];
            
            if (old_token >= 0 && old_token < max_tokens) {
                residence_bitmap[old_token] = 0;
                token_to_gpu_loc[old_token] = -1;
            }
            
            residence_bitmap[my_token] = 1;
            token_to_gpu_loc[my_token] = evict_loc;
            gpu_loc_to_token[evict_loc] = my_token;
            last_used_epoch[evict_loc] = current_epoch;  // Mark as used
            top_k_gpu_locs[my_token_idx] = evict_loc;
            
            copy_src_cpu_locs[global_slot] = cache_cpu_locs[my_token];
            copy_dst_gpu_locs[global_slot] = evict_loc;
        }
    }
    __syncthreads();
    
    if (tid == 0) *num_copies = s_total_misses;
}

// Launchers
void sparse_cache_manager_v4_cuda(
    torch::Tensor top_k_indices,
    torch::Tensor cache_cpu_locs,
    torch::Tensor residence_bitmap,
    torch::Tensor token_to_gpu_loc,
    torch::Tensor gpu_loc_to_token,
    torch::Tensor top_k_gpu_locs,
    torch::Tensor copy_src_cpu_locs,
    torch::Tensor copy_dst_gpu_locs,
    torch::Tensor num_copies,
    torch::Tensor evict_ptr
) {
    num_copies.zero_();
    sparse_cache_manager_v4_kernel<256><<<1, 256>>>(
        top_k_indices.data_ptr<int32_t>(),
        cache_cpu_locs.data_ptr<int64_t>(),
        residence_bitmap.data_ptr<int8_t>(),
        token_to_gpu_loc.data_ptr<int32_t>(),
        gpu_loc_to_token.data_ptr<int32_t>(),
        top_k_gpu_locs.data_ptr<int32_t>(),
        copy_src_cpu_locs.data_ptr<int64_t>(),
        copy_dst_gpu_locs.data_ptr<int64_t>(),
        num_copies.data_ptr<int32_t>(),
        evict_ptr.data_ptr<int32_t>(),
        top_k_indices.size(0),
        gpu_loc_to_token.size(0),
        residence_bitmap.size(0)
    );
}

void sparse_cache_manager_v4b_epoch_cuda(
    torch::Tensor top_k_indices,
    torch::Tensor cache_cpu_locs,
    torch::Tensor residence_bitmap,
    torch::Tensor token_to_gpu_loc,
    torch::Tensor gpu_loc_to_token,
    torch::Tensor last_used_epoch,
    torch::Tensor top_k_gpu_locs,
    torch::Tensor copy_src_cpu_locs,
    torch::Tensor copy_dst_gpu_locs,
    torch::Tensor num_copies,
    torch::Tensor evict_ptr,
    int16_t current_epoch
) {
    num_copies.zero_();
    sparse_cache_manager_v4b_epoch_kernel<256><<<1, 256>>>(
        top_k_indices.data_ptr<int32_t>(),
        cache_cpu_locs.data_ptr<int64_t>(),
        residence_bitmap.data_ptr<int8_t>(),
        token_to_gpu_loc.data_ptr<int32_t>(),
        gpu_loc_to_token.data_ptr<int32_t>(),
        last_used_epoch.data_ptr<int16_t>(),
        top_k_gpu_locs.data_ptr<int32_t>(),
        copy_src_cpu_locs.data_ptr<int64_t>(),
        copy_dst_gpu_locs.data_ptr<int64_t>(),
        num_copies.data_ptr<int32_t>(),
        evict_ptr.data_ptr<int32_t>(),
        current_epoch,
        top_k_indices.size(0),
        gpu_loc_to_token.size(0),
        residence_bitmap.size(0)
    );
}
'''

CPP_SOURCE = r'''
void sparse_cache_manager_v4_cuda(
    torch::Tensor top_k_indices,
    torch::Tensor cache_cpu_locs,
    torch::Tensor residence_bitmap,
    torch::Tensor token_to_gpu_loc,
    torch::Tensor gpu_loc_to_token,
    torch::Tensor top_k_gpu_locs,
    torch::Tensor copy_src_cpu_locs,
    torch::Tensor copy_dst_gpu_locs,
    torch::Tensor num_copies,
    torch::Tensor evict_ptr
);

void sparse_cache_manager_v4b_epoch_cuda(
    torch::Tensor top_k_indices,
    torch::Tensor cache_cpu_locs,
    torch::Tensor residence_bitmap,
    torch::Tensor token_to_gpu_loc,
    torch::Tensor gpu_loc_to_token,
    torch::Tensor last_used_epoch,
    torch::Tensor top_k_gpu_locs,
    torch::Tensor copy_src_cpu_locs,
    torch::Tensor copy_dst_gpu_locs,
    torch::Tensor num_copies,
    torch::Tensor evict_ptr,
    int16_t current_epoch
);
'''

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name='sparse_cache_v4',
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=[
                'sparse_cache_manager_v4_cuda',
                'sparse_cache_manager_v4b_epoch_cuda',
            ],
            verbose=False,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
        )
    return _module


@dataclass
class CopyInfo:
    src_cpu_locs: torch.Tensor
    dst_gpu_locs: torch.Tensor
    num_copies: int


class SparseCacheManagerV4:
    """
    v4: Optimized evictable finding with circular pointer + early exit.
    
    Key improvement over v3:
    - v3: Scans all H locations to build evictable list → O(H)
    - v4: Scans only until num_misses found → O(num_misses + num_hits)
    
    Uses warp-parallel circular scan: 32 locations checked per iteration.
    """
    
    def __init__(
        self,
        max_tokens: int,
        hot_buffer_size: int,
        device: str = "cuda",
        use_epoch: bool = False,  # Use epoch-based tracking
    ):
        self.max_tokens = max_tokens
        self.hot_buffer_size = hot_buffer_size
        self.device = device
        self.use_epoch = use_epoch
        
        self.residence_bitmap = torch.zeros(max_tokens, dtype=torch.int8, device=device)
        self.token_to_gpu_loc = torch.full((max_tokens,), -1, dtype=torch.int32, device=device)
        self.gpu_loc_to_token = torch.full((hot_buffer_size,), -1, dtype=torch.int32, device=device)
        
        # Circular pointer for eviction
        self.evict_ptr = torch.zeros(1, dtype=torch.int32, device=device)
        
        # Epoch-based tracking (v4b)
        if use_epoch:
            self.last_used_epoch = torch.zeros(hot_buffer_size, dtype=torch.int16, device=device)
            self.current_epoch = 0
        
        self._copy_src = torch.zeros(hot_buffer_size, dtype=torch.int64, device=device)
        self._copy_dst = torch.zeros(hot_buffer_size, dtype=torch.int64, device=device)
        self._num_copies = torch.zeros(1, dtype=torch.int32, device=device)
        
        self._module = None
    
    def _ensure_compiled(self):
        if self._module is None:
            self._module = _get_module()
    
    def reset(self):
        self.residence_bitmap.zero_()
        self.token_to_gpu_loc.fill_(-1)
        self.gpu_loc_to_token.fill_(-1)
        self.evict_ptr.zero_()
        if self.use_epoch:
            self.last_used_epoch.zero_()
            self.current_epoch = 0
    
    def process_topk(
        self,
        top_k_indices: torch.Tensor,
        cache_cpu_locations: torch.Tensor,
    ) -> Tuple[torch.Tensor, CopyInfo]:
        self._ensure_compiled()
        
        if top_k_indices.dtype != torch.int32:
            top_k_indices = top_k_indices.to(torch.int32)
        if cache_cpu_locations.dtype != torch.int64:
            cache_cpu_locations = cache_cpu_locations.to(torch.int64)
        
        top_k_size = top_k_indices.numel()
        top_k_gpu_locs = torch.zeros(top_k_size, dtype=torch.int32, device=self.device)
        self._num_copies.zero_()
        
        if self.use_epoch:
            self.current_epoch = (self.current_epoch + 1) % 32767
            self._module.sparse_cache_manager_v4b_epoch_cuda(
                top_k_indices, cache_cpu_locations,
                self.residence_bitmap, self.token_to_gpu_loc, self.gpu_loc_to_token,
                self.last_used_epoch,
                top_k_gpu_locs, self._copy_src, self._copy_dst, self._num_copies,
                self.evict_ptr, self.current_epoch
            )
        else:
            self._module.sparse_cache_manager_v4_cuda(
                top_k_indices, cache_cpu_locations,
                self.residence_bitmap, self.token_to_gpu_loc, self.gpu_loc_to_token,
                top_k_gpu_locs, self._copy_src, self._copy_dst, self._num_copies,
                self.evict_ptr
            )
        
        n = self._num_copies.item()
        return top_k_gpu_locs, CopyInfo(
            src_cpu_locs=self._copy_src[:n].clone(),
            dst_gpu_locs=self._copy_dst[:n].clone(),
            num_copies=n,
        )
    
    def get_stats(self) -> dict:
        stats = {
            "num_resident": self.residence_bitmap.sum().item(),
            "hot_buffer_size": self.hot_buffer_size,
            "evict_ptr": self.evict_ptr.item(),
            "algorithm": "epoch" if self.use_epoch else "bitmap",
        }
        if self.use_epoch:
            stats["current_epoch"] = self.current_epoch
        return stats


def compare_complexity():
    """Print complexity comparison between versions."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Complexity Comparison: v3 vs v4                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Operation              │ v3 (Full Scan)    │ v4 (Circular + Early Exit)    ║
║  ───────────────────────┼───────────────────┼───────────────────────────────║
║  Hit detection          │ O(K)              │ O(K)                          ║
║  Mark protected         │ O(K)              │ O(K) or O(1)*                 ║
║  Prefix sum             │ O(num_warps)      │ O(num_warps)                  ║
║  Find evictable         │ O(H)              │ O((M+K)/32)**                 ║
║  Assignment             │ O(K)              │ O(K)                          ║
║  ───────────────────────┼───────────────────┼───────────────────────────────║
║  TOTAL                  │ O(H + K)          │ O(K + (M+K)/32)               ║
║                                                                              ║
║  Where:                                                                      ║
║    K = top_k_size (typically 64-256)                                        ║
║    H = hot_buffer_size (typically 512-4096)                                 ║
║    M = num_misses (≤ K)                                                     ║
║                                                                              ║
║  * With epoch-based tracking, no bitmap clearing needed                      ║
║  ** Warp-parallel scan: 32 locations per iteration                          ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Example: K=64, H=2048, M=32 (50% hit rate)                                 ║
║  ───────────────────────────────────────────                                ║
║  v3: O(2048 + 64) = O(2112)                                                 ║
║  v4: O(64 + (32+64)/32) = O(64 + 3) = O(67)                                ║
║                                                                              ║
║  Speedup: ~31x for evictable finding!                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
