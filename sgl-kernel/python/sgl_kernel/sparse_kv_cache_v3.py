"""
Sparse KV Cache Manager v3 - Warp-Level Prefix Sum for O(1) Eviction Assignment

Key insight from user:
1. After hit detection, use intra-warp communication to count hits/misses
2. Each warp leader writes miss count to shared memory
3. Prefix sum over warp counts gives each warp's starting slot
4. Each thread computes: global_slot = prefix[warp_id] + local_offset
5. Direct indexing into evictable list - no atomic contention!

Benefits:
- O(1) per-miss slot assignment (vs atomic contention)
- Warp-level parallelism with __ballot_sync, __popc
- Prefix sum is O(num_warps), typically 8-32
- No atomic race conditions for slot assignment
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
 * Sparse Cache Manager v3 - Warp-Level Prefix Sum
 * 
 * Algorithm:
 * 
 * Phase 1: Warp-parallel hit detection
 *   - Each lane checks one token (O(1) bitmap lookup)
 *   - __ballot_sync to get miss mask
 *   - __popc to count misses in warp
 *   - __popc(mask & lanes_before_me) for local offset
 *   - Warp leader writes miss count to shared memory
 * 
 * Phase 2: Prefix sum over warp counts
 *   - Single thread does prefix sum (O(num_warps))
 *   - Result: each warp knows its starting eviction slot
 * 
 * Phase 3: Build evictable list
 *   - Parallel scan with protected bitmap (O(H), O(1) per loc)
 * 
 * Phase 4: Direct assignment
 *   - global_slot = prefix[warp_id] + local_offset
 *   - evict_loc = evictable_list[global_slot]
 *   - No atomics needed for slot assignment!
 */
template <int BLOCK_SIZE>
__global__ void sparse_cache_manager_v3_kernel(
    // Inputs
    const int32_t* __restrict__ top_k_indices,
    const int64_t* __restrict__ cache_cpu_locs,
    // State
    int8_t* __restrict__ residence_bitmap,
    int32_t* __restrict__ token_to_gpu_loc,
    int32_t* __restrict__ gpu_loc_to_token,
    // Outputs
    int32_t* __restrict__ top_k_gpu_locs,
    int64_t* __restrict__ copy_src_cpu_locs,
    int64_t* __restrict__ copy_dst_gpu_locs,
    int32_t* __restrict__ num_copies,
    // Sizes
    int32_t top_k_size,
    int32_t hot_buffer_size,
    int32_t max_tokens
) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // ===== Shared Memory =====
    // Protected GPU locations bitmap
    __shared__ uint8_t s_protected[(MAX_HOT_BUFFER + 7) / 8];
    // Warp-level miss counts and prefix sums
    __shared__ int32_t s_warp_miss_count[MAX_WARPS];
    __shared__ int32_t s_warp_miss_prefix[MAX_WARPS + 1];
    // Per-thread local offset within warp (for misses)
    __shared__ int32_t s_local_offset[MAX_TOP_K];
    __shared__ int32_t s_is_miss[MAX_TOP_K];
    // Evictable locations list
    __shared__ int32_t s_evictable[MAX_HOT_BUFFER];
    __shared__ int32_t s_num_evictable;
    __shared__ int32_t s_total_misses;
    
    // Initialize
    if (tid == 0) {
        s_num_evictable = 0;
        s_total_misses = 0;
    }
    for (int i = tid; i < (hot_buffer_size + 7) / 8; i += BLOCK_SIZE) {
        s_protected[i] = 0;
    }
    for (int i = tid; i < MAX_WARPS; i += BLOCK_SIZE) {
        s_warp_miss_count[i] = 0;
    }
    __syncthreads();
    
    // ===== Phase 1: Warp-Parallel Hit Detection =====
    // Each warp processes WARP_SIZE tokens
    
    // Determine which tokens this warp processes
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
            // Mark as protected
            atomicOr((unsigned int*)&s_protected[my_gpu_loc / 8], 1u << (my_gpu_loc % 8));
        }
    }
    
    // Intra-warp communication
    unsigned int active_mask = __ballot_sync(0xFFFFFFFF, has_valid_token);
    unsigned int miss_mask = __ballot_sync(0xFFFFFFFF, has_valid_token && !is_hit);
    
    // Count misses in this warp
    int warp_miss_count = __popc(miss_mask);
    
    // Each miss lane computes its local offset within warp
    // local_offset = number of miss lanes before me
    unsigned int lanes_before_me = (1u << lane_id) - 1;
    int local_offset = __popc(miss_mask & lanes_before_me);
    
    // Store local offset for misses
    if (has_valid_token && !is_hit) {
        s_local_offset[my_token_idx] = local_offset;
        s_is_miss[my_token_idx] = 1;
    } else if (has_valid_token) {
        s_is_miss[my_token_idx] = 0;
    }
    
    // Warp leader writes miss count
    if (lane_id == 0) {
        s_warp_miss_count[warp_id] = warp_miss_count;
    }
    __syncthreads();
    
    // ===== Phase 2: Prefix Sum over Warp Counts =====
    // Single thread computes prefix sum (fast for small NUM_WARPS)
    if (tid == 0) {
        s_warp_miss_prefix[0] = 0;
        int num_active_warps = (top_k_size + WARP_SIZE - 1) / WARP_SIZE;
        for (int w = 0; w < num_active_warps; w++) {
            s_warp_miss_prefix[w + 1] = s_warp_miss_prefix[w] + s_warp_miss_count[w];
        }
        s_total_misses = s_warp_miss_prefix[num_active_warps];
    }
    __syncthreads();
    
    // Early exit if no misses
    if (s_total_misses == 0) {
        if (tid == 0) *num_copies = 0;
        return;
    }
    
    // ===== Phase 3: Build Evictable List =====
    // Parallel scan with O(1) protected check per location
    for (int h = tid; h < hot_buffer_size; h += BLOCK_SIZE) {
        bool is_protected = (s_protected[h / 8] >> (h % 8)) & 1;
        if (!is_protected) {
            int idx = atomicAdd(&s_num_evictable, 1);
            s_evictable[idx] = h;
        }
    }
    __syncthreads();
    
    // ===== Phase 4: Direct Assignment using Prefix Sum =====
    // Each miss directly computes its eviction slot - NO ATOMICS!
    
    if (has_valid_token && s_is_miss[my_token_idx]) {
        // Direct slot computation
        int global_slot = s_warp_miss_prefix[warp_id] + s_local_offset[my_token_idx];
        
        if (global_slot < s_num_evictable) {
            int32_t evict_loc = s_evictable[global_slot];
            int32_t old_token = gpu_loc_to_token[evict_loc];
            
            // Clear old token
            if (old_token >= 0 && old_token < max_tokens) {
                residence_bitmap[old_token] = 0;
                token_to_gpu_loc[old_token] = -1;
            }
            
            // Set new token
            residence_bitmap[my_token] = 1;
            token_to_gpu_loc[my_token] = evict_loc;
            gpu_loc_to_token[evict_loc] = my_token;
            
            // Output
            top_k_gpu_locs[my_token_idx] = evict_loc;
            
            // Copy list (use global_slot as index - deterministic!)
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
 * Alternative: Multi-pass for larger top_k
 * When top_k > BLOCK_SIZE, process in chunks
 */
template <int BLOCK_SIZE>
__global__ void sparse_cache_manager_v3_large_kernel(
    const int32_t* __restrict__ top_k_indices,
    const int64_t* __restrict__ cache_cpu_locs,
    int8_t* __restrict__ residence_bitmap,
    int32_t* __restrict__ token_to_gpu_loc,
    int32_t* __restrict__ gpu_loc_to_token,
    int32_t* __restrict__ top_k_gpu_locs,
    int64_t* __restrict__ copy_src_cpu_locs,
    int64_t* __restrict__ copy_dst_gpu_locs,
    int32_t* __restrict__ num_copies,
    // Temp buffers (pre-allocated)
    int32_t* __restrict__ miss_tokens,
    int32_t* __restrict__ miss_top_k_indices,
    int32_t* __restrict__ miss_count_out,
    int32_t top_k_size,
    int32_t hot_buffer_size,
    int32_t max_tokens
) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    __shared__ uint8_t s_protected[(MAX_HOT_BUFFER + 7) / 8];
    __shared__ int32_t s_miss_count;
    
    if (tid == 0) s_miss_count = 0;
    for (int i = tid; i < (hot_buffer_size + 7) / 8; i += BLOCK_SIZE) {
        s_protected[i] = 0;
    }
    __syncthreads();
    
    // Phase 1: Hit detection and collect misses
    for (int k = tid; k < top_k_size; k += BLOCK_SIZE) {
        int32_t token = top_k_indices[k];
        bool is_hit = (residence_bitmap[token] != 0);
        
        if (is_hit) {
            int32_t gpu_loc = token_to_gpu_loc[token];
            top_k_gpu_locs[k] = gpu_loc;
            atomicOr((unsigned int*)&s_protected[gpu_loc / 8], 1u << (gpu_loc % 8));
        } else {
            int idx = atomicAdd(&s_miss_count, 1);
            miss_tokens[idx] = token;
            miss_top_k_indices[idx] = k;
        }
    }
    __syncthreads();
    
    if (s_miss_count == 0) {
        if (tid == 0) *num_copies = 0;
        return;
    }
    
    // Phase 2: Build evictable list with prefix sum approach
    __shared__ int32_t s_evictable[MAX_HOT_BUFFER];
    __shared__ int32_t s_num_evictable;
    if (tid == 0) s_num_evictable = 0;
    __syncthreads();
    
    // Warp-cooperative evictable finding
    __shared__ int32_t s_warp_evict_count[MAX_WARPS];
    __shared__ int32_t s_warp_evict_prefix[MAX_WARPS + 1];
    
    int locs_per_warp = (hot_buffer_size + NUM_WARPS - 1) / NUM_WARPS;
    int my_loc_start = warp_id * locs_per_warp;
    int my_loc_end = min(my_loc_start + locs_per_warp, hot_buffer_size);
    
    // Each lane checks one location within warp's range
    int warp_evict_count = 0;
    for (int base = my_loc_start; base < my_loc_end; base += WARP_SIZE) {
        int h = base + lane_id;
        bool is_evictable = false;
        if (h < my_loc_end) {
            is_evictable = !((s_protected[h / 8] >> (h % 8)) & 1);
        }
        unsigned int evict_mask = __ballot_sync(0xFFFFFFFF, is_evictable);
        warp_evict_count += __popc(evict_mask);
    }
    
    if (lane_id == 0) {
        s_warp_evict_count[warp_id] = warp_evict_count;
    }
    __syncthreads();
    
    // Prefix sum for evictable slots
    if (tid == 0) {
        s_warp_evict_prefix[0] = 0;
        for (int w = 0; w < NUM_WARPS; w++) {
            s_warp_evict_prefix[w + 1] = s_warp_evict_prefix[w] + s_warp_evict_count[w];
        }
        s_num_evictable = s_warp_evict_prefix[NUM_WARPS];
    }
    __syncthreads();
    
    // Each warp writes its evictable locations to the correct range
    int my_write_start = s_warp_evict_prefix[warp_id];
    int write_idx = 0;
    
    for (int base = my_loc_start; base < my_loc_end; base += WARP_SIZE) {
        int h = base + lane_id;
        bool is_evictable = false;
        if (h < my_loc_end) {
            is_evictable = !((s_protected[h / 8] >> (h % 8)) & 1);
        }
        
        unsigned int evict_mask = __ballot_sync(0xFFFFFFFF, is_evictable);
        
        // Compute position for each evictable in this iteration
        if (is_evictable) {
            int pos_in_batch = __popc(evict_mask & ((1u << lane_id) - 1));
            s_evictable[my_write_start + write_idx + pos_in_batch] = h;
        }
        
        write_idx += __popc(evict_mask);
    }
    __syncthreads();
    
    // Phase 3: Assign evictions
    for (int m = tid; m < s_miss_count && m < s_num_evictable; m += BLOCK_SIZE) {
        int32_t evict_loc = s_evictable[m];
        int32_t new_token = miss_tokens[m];
        int32_t top_k_idx = miss_top_k_indices[m];
        int32_t old_token = gpu_loc_to_token[evict_loc];
        
        if (old_token >= 0 && old_token < max_tokens) {
            residence_bitmap[old_token] = 0;
            token_to_gpu_loc[old_token] = -1;
        }
        
        residence_bitmap[new_token] = 1;
        token_to_gpu_loc[new_token] = evict_loc;
        gpu_loc_to_token[evict_loc] = new_token;
        top_k_gpu_locs[top_k_idx] = evict_loc;
        
        copy_src_cpu_locs[m] = cache_cpu_locs[new_token];
        copy_dst_gpu_locs[m] = evict_loc;
    }
    __syncthreads();
    
    if (tid == 0) {
        *num_copies = min(s_miss_count, s_num_evictable);
    }
}

void sparse_cache_manager_v3_cuda(
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
    num_copies.zero_();
    
    int32_t top_k_size = top_k_indices.size(0);
    
    if (top_k_size <= 256) {
        sparse_cache_manager_v3_kernel<256><<<1, 256>>>(
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
            gpu_loc_to_token.size(0),
            residence_bitmap.size(0)
        );
    } else {
        // Allocate temp buffers for large kernel
        auto opts = torch::TensorOptions().dtype(torch::kInt32).device(top_k_indices.device());
        auto miss_tokens = torch::zeros({top_k_size}, opts);
        auto miss_indices = torch::zeros({top_k_size}, opts);
        auto miss_count = torch::zeros({1}, opts);
        
        sparse_cache_manager_v3_large_kernel<256><<<1, 256>>>(
            top_k_indices.data_ptr<int32_t>(),
            cache_cpu_locs.data_ptr<int64_t>(),
            residence_bitmap.data_ptr<int8_t>(),
            token_to_gpu_loc.data_ptr<int32_t>(),
            gpu_loc_to_token.data_ptr<int32_t>(),
            top_k_gpu_locs.data_ptr<int32_t>(),
            copy_src_cpu_locs.data_ptr<int64_t>(),
            copy_dst_gpu_locs.data_ptr<int64_t>(),
            num_copies.data_ptr<int32_t>(),
            miss_tokens.data_ptr<int32_t>(),
            miss_indices.data_ptr<int32_t>(),
            miss_count.data_ptr<int32_t>(),
            top_k_size,
            gpu_loc_to_token.size(0),
            residence_bitmap.size(0)
        );
    }
}
'''

CPP_SOURCE = r'''
void sparse_cache_manager_v3_cuda(
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
'''

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name='sparse_cache_v3',
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=['sparse_cache_manager_v3_cuda'],
            verbose=False,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
        )
    return _module


@dataclass
class CopyInfo:
    src_cpu_locs: torch.Tensor
    dst_gpu_locs: torch.Tensor
    num_copies: int


class SparseCacheManagerV3:
    """
    Sparse KV Cache Manager v3 - Warp-Level Prefix Sum
    
    Uses warp-level parallel primitives for O(1) slot assignment:
    1. __ballot_sync for miss detection
    2. __popc for counting and local offset
    3. Prefix sum over warp counts
    4. Direct slot indexing (no atomics for assignment!)
    
    Complexity:
    - Phase 1 (hits): O(K) with warp parallelism
    - Phase 2 (prefix): O(num_warps), typically 8
    - Phase 3 (evictable): O(H) with bitmap, parallel
    - Phase 4 (assign): O(K) with direct indexing, NO ATOMICS
    """
    
    def __init__(
        self,
        max_tokens: int,
        hot_buffer_size: int,
        device: str = "cuda",
    ):
        self.max_tokens = max_tokens
        self.hot_buffer_size = hot_buffer_size
        self.device = device
        
        self.residence_bitmap = torch.zeros(max_tokens, dtype=torch.int8, device=device)
        self.token_to_gpu_loc = torch.full((max_tokens,), -1, dtype=torch.int32, device=device)
        self.gpu_loc_to_token = torch.full((hot_buffer_size,), -1, dtype=torch.int32, device=device)
        
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
        
        self._module.sparse_cache_manager_v3_cuda(
            top_k_indices, cache_cpu_locations,
            self.residence_bitmap, self.token_to_gpu_loc, self.gpu_loc_to_token,
            top_k_gpu_locs, self._copy_src, self._copy_dst, self._num_copies
        )
        
        n = self._num_copies.item()
        return top_k_gpu_locs, CopyInfo(
            src_cpu_locs=self._copy_src[:n].clone(),
            dst_gpu_locs=self._copy_dst[:n].clone(),
            num_copies=n,
        )
    
    def get_stats(self) -> dict:
        return {
            "num_resident": self.residence_bitmap.sum().item(),
            "hot_buffer_size": self.hot_buffer_size,
        }


def visualize_algorithm():
    """Visualize the warp-level prefix sum algorithm."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║           Warp-Level Prefix Sum for O(1) Eviction Assignment                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Phase 1: Warp-Parallel Hit Detection                                       ║
║  ─────────────────────────────────────                                       ║
║                                                                              ║
║  Warp 0:  [H][M][H][H][M][M][H][M]...  (H=hit, M=miss)                      ║
║           ↓                                                                  ║
║           miss_mask = 0b01001101                                             ║
║           miss_count = __popc(mask) = 4                                      ║
║                                                                              ║
║  For each miss lane:                                                         ║
║           local_offset = __popc(mask & lanes_before_me)                      ║
║           Lane 1: offset = 0  (first miss)                                   ║
║           Lane 4: offset = 1  (second miss)                                  ║
║           Lane 5: offset = 2  (third miss)                                   ║
║           Lane 7: offset = 3  (fourth miss)                                  ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Phase 2: Prefix Sum over Warp Counts                                        ║
║  ─────────────────────────────────────                                       ║
║                                                                              ║
║  warp_miss_count = [4, 2, 5, 1, ...]                                        ║
║  prefix_sum      = [0, 4, 6, 11, 12, ...]                                   ║
║                     ↑  ↑  ↑   ↑                                              ║
║                     │  │  │   └─ Warp 3 starts at slot 11                   ║
║                     │  │  └───── Warp 2 starts at slot 6                    ║
║                     │  └──────── Warp 1 starts at slot 4                    ║
║                     └─────────── Warp 0 starts at slot 0                    ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Phase 4: Direct Slot Assignment (NO ATOMICS!)                               ║
║  ─────────────────────────────────────────────                               ║
║                                                                              ║
║  For miss in Warp 0, Lane 5 (local_offset = 2):                             ║
║                                                                              ║
║      global_slot = prefix[0] + local_offset                                  ║
║                  = 0 + 2                                                     ║
║                  = 2                                                         ║
║                                                                              ║
║      evict_loc = evictable_list[2]   ← Direct index, no contention!         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Complexity:                                                                 ║
║  ───────────                                                                 ║
║  • Hit detection:    O(K/32) per warp, parallel                              ║
║  • Prefix sum:       O(num_warps) ≈ O(8)                                    ║
║  • Slot assignment:  O(1) per miss, NO ATOMICS                               ║
║  • Total:            O(K + H) with high parallelism                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
