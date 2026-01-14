"""
Sparse KV Cache Manager v7 - Atomic-Free Implementation

Key improvements over v6:
1. No atomicOr operations - all writes are conflict-free
2. Use __ballot_sync result directly as miss bitmap (warp result IS the bitmap word)
3. Collect hit slots in array, build protected bitmap with single thread

Data structure (same as v6):
- token_to_slot[max_tokens]: uint16_t, 0xFFFF = not present, else = slot index
- device_buffer_tokens[hot_buffer_size]: which token at each slot
"""

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int WARP_SIZE = 32;
constexpr uint16_t NOT_PRESENT = 0xFFFF;

/**
 * Sparse Cache Manager v7 - Atomic-Free
 * 
 * Eliminates all atomicOr by:
 * 1. Miss bitmap: __ballot_sync result IS the bitmap word for each warp
 * 2. Protected bitmap: collect hits first, then single-thread build
 * 
 * Phase 1: Hit detection + collect hit info
 *   - Each thread checks token_to_slot[my_token]
 *   - Use ballot to get miss mask (this IS the miss bitmap word!)
 *   - Hit threads store their slots in s_hit_slots using prefix sums
 * 
 * Phase 2: Build protected bitmap (single thread, no atomics)
 *   - Iterate through s_hit_slots, set bits
 * 
 * Phase 3: Prefix sum for miss assignment
 * 
 * Phase 4: Find evictable slots
 * 
 * Phase 5: Assignment + copy
 */
template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE>
__global__ void sparse_cache_manager_v7_kernel(
    const int32_t* __restrict__ top_k_tokens,
    int32_t* __restrict__ device_buffer_tokens,
    uint16_t* __restrict__ token_to_slot,
    const int64_t* __restrict__ host_cache_locs,
    const int64_t* __restrict__ device_buffer_locs,
    const char* __restrict__ host_cache,
    char* __restrict__ device_buffer,
    int64_t* __restrict__ top_k_device_locs,
    int32_t max_tokens,
    int64_t item_size_bytes
) {
    static_assert(NUM_TOP_K <= 256, "NUM_TOP_K must be <= 256");
    static_assert(HOT_BUFFER_SIZE <= 65535, "HOT_BUFFER_SIZE must be <= 65535");
    
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    constexpr int NUM_WARPS_FOR_TOPK = (NUM_TOP_K + WARP_SIZE - 1) / WARP_SIZE;
    constexpr int BITMAP_WORDS_BUFFER = (HOT_BUFFER_SIZE + 31) / 32;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const unsigned int lanes_before = (1u << lane_id) - 1;
    
    // ===== Shared Memory =====
    // Miss bitmap: one word per warp (ballot result stored directly)
    __shared__ uint32_t s_miss_bitmap[NUM_WARPS_FOR_TOPK];
    // Hit info collection (for building protected bitmap)
    __shared__ int32_t s_hit_slots[NUM_TOP_K];
    __shared__ int32_t s_warp_hit_count[NUM_WARPS];
    __shared__ int32_t s_warp_hit_prefix[NUM_WARPS + 1];
    // Miss tracking
    __shared__ int32_t s_warp_miss_count[NUM_WARPS];
    __shared__ int32_t s_warp_miss_prefix[NUM_WARPS + 1];
    __shared__ int32_t s_local_miss_offset[NUM_TOP_K];
    // Protected bitmap (built by single thread)
    __shared__ uint32_t s_protected_bitmap[BITMAP_WORDS_BUFFER];
    // Evictable slots
    __shared__ int32_t s_evictable_slots[NUM_TOP_K];
    __shared__ int32_t s_total_misses;
    __shared__ int32_t s_total_hits;
    
    // Initialize counters
    if (tid == 0) {
        s_total_misses = 0;
        s_total_hits = 0;
    }
    // Initialize warp counts
    if (tid < NUM_WARPS) {
        s_warp_hit_count[tid] = 0;
        s_warp_miss_count[tid] = 0;
    }
    // Initialize miss bitmap
    if (tid < NUM_WARPS_FOR_TOPK) {
        s_miss_bitmap[tid] = 0;
    }
    __syncthreads();
    
    // ===== Phase 1: Hit Detection =====
    const int warp_token_start = warp_id * WARP_SIZE;
    const int my_token_idx = warp_token_start + lane_id;
    const bool has_valid_token = my_token_idx < NUM_TOP_K;
    
    int32_t my_token = 0;
    bool is_hit = false;
    int32_t my_device_slot = -1;
    
    if (has_valid_token) {
        my_token = top_k_tokens[my_token_idx];
        
        // O(1) lookup
        uint16_t slot = token_to_slot[my_token];
        is_hit = (slot != NOT_PRESENT);
        
        if (is_hit) {
            my_device_slot = static_cast<int32_t>(slot);
            // Output device location immediately
            top_k_device_locs[my_token_idx] = device_buffer_locs[my_device_slot];
        }
    }
    
    // Warp-level collection of hits and misses
    const unsigned int hit_mask = __ballot_sync(0xFFFFFFFF, has_valid_token && is_hit);
    const unsigned int miss_mask = __ballot_sync(0xFFFFFFFF, has_valid_token && !is_hit);
    
    const int warp_hit_count = __popc(hit_mask);
    const int warp_miss_count = __popc(miss_mask);
    const int local_hit_offset = __popc(hit_mask & lanes_before);
    const int local_miss_offset = __popc(miss_mask & lanes_before);
    
    // Store miss offset for later use
    if (has_valid_token && !is_hit) {
        s_local_miss_offset[my_token_idx] = local_miss_offset;
    }
    
    // Warp leader stores counts and miss bitmap (NO ATOMIC!)
    if (lane_id == 0 && warp_id < NUM_WARPS_FOR_TOPK) {
        s_warp_hit_count[warp_id] = warp_hit_count;
        s_warp_miss_count[warp_id] = warp_miss_count;
        // The ballot result IS the miss bitmap word for this warp!
        s_miss_bitmap[warp_id] = miss_mask;
    }
    __syncthreads();
    
    // ===== Phase 2: Prefix Sums (single thread) =====
    if (tid == 0) {
        s_warp_hit_prefix[0] = 0;
        s_warp_miss_prefix[0] = 0;
        for (int w = 0; w < NUM_WARPS_FOR_TOPK; w++) {
            s_warp_hit_prefix[w + 1] = s_warp_hit_prefix[w] + s_warp_hit_count[w];
            s_warp_miss_prefix[w + 1] = s_warp_miss_prefix[w] + s_warp_miss_count[w];
        }
        s_total_hits = s_warp_hit_prefix[NUM_WARPS_FOR_TOPK];
        s_total_misses = s_warp_miss_prefix[NUM_WARPS_FOR_TOPK];
    }
    __syncthreads();
    
    // ===== Phase 3: Collect Hit Slots (conflict-free writes) =====
    // Each hit thread writes to its unique index
    if (has_valid_token && is_hit && warp_id < NUM_WARPS_FOR_TOPK) {
        const int global_hit_idx = s_warp_hit_prefix[warp_id] + local_hit_offset;
        s_hit_slots[global_hit_idx] = my_device_slot;
    }
    __syncthreads();
    
    // Early exit if all hits
    if (s_total_misses == 0) {
        return;
    }
    
    // ===== Phase 4: Build Protected Bitmap (single thread, NO ATOMIC!) =====
    // Clear bitmap first
    for (int i = tid; i < BITMAP_WORDS_BUFFER; i += BLOCK_SIZE) {
        s_protected_bitmap[i] = 0;
    }
    __syncthreads();
    
    // Single thread builds the bitmap from hit slots
    if (tid == 0) {
        for (int i = 0; i < s_total_hits; i++) {
            const int slot = s_hit_slots[i];
            const int word = slot / 32;
            const int bit = slot % 32;
            s_protected_bitmap[word] |= (1u << bit);
        }
    }
    __syncthreads();
    
    // ===== Phase 5: Find Evictable Slots (first warp) =====
    if (warp_id == 0) {
        int found = 0;
        int base_slot = 0;
        
        while (found < s_total_misses && base_slot < HOT_BUFFER_SIZE) {
            const int my_slot = base_slot + lane_id;
            bool is_evictable = false;
            
            if (my_slot < HOT_BUFFER_SIZE) {
                const int prot_word = my_slot / 32;
                const int prot_bit = my_slot % 32;
                const uint32_t prot_bitmap = s_protected_bitmap[prot_word];
                const bool is_protected = (prot_bitmap >> prot_bit) & 1;
                is_evictable = !is_protected;
            }
            
            const unsigned int evict_mask = __ballot_sync(0xFFFFFFFF, is_evictable);
            const int num_evictable = __popc(evict_mask);
            const int need = s_total_misses - found;
            const int use_count = min(num_evictable, need);
            
            // Each evictable lane stores its slot (conflict-free!)
            if (is_evictable) {
                const int my_pos = __popc(evict_mask & lanes_before);
                if (my_pos < use_count) {
                    s_evictable_slots[found + my_pos] = my_slot;
                }
            }
            
            found += use_count;
            base_slot += WARP_SIZE;
        }
    }
    __syncthreads();
    
    // ===== Phase 6: Assignment + Data Copy =====
    if (has_valid_token && warp_id < NUM_WARPS_FOR_TOPK) {
        // Check miss using the stored bitmap word
        const uint32_t miss_word = s_miss_bitmap[warp_id];
        const bool is_miss = (miss_word >> lane_id) & 1;
        
        if (is_miss) {
            const int global_slot = s_warp_miss_prefix[warp_id] + s_local_miss_offset[my_token_idx];
            
            if (global_slot < s_total_misses) {
                const int evict_slot = s_evictable_slots[global_slot];
                const int32_t old_token = device_buffer_tokens[evict_slot];
                
                // Clear old token's mapping
                if (old_token >= 0 && old_token < max_tokens) {
                    token_to_slot[old_token] = NOT_PRESENT;
                }
                
                // Set new token's mapping
                token_to_slot[my_token] = static_cast<uint16_t>(evict_slot);
                device_buffer_tokens[evict_slot] = my_token;
                
                // Output device location
                top_k_device_locs[my_token_idx] = device_buffer_locs[evict_slot];
                
                // Direct data copy
                const int64_t host_loc = host_cache_locs[my_token];
                const int64_t device_loc = device_buffer_locs[evict_slot];
                const char* src = host_cache + host_loc * item_size_bytes;
                char* dst = device_buffer + device_loc * item_size_bytes;
                
                for (int64_t b = 0; b < item_size_bytes; b++) {
                    dst[b] = src[b];
                }
            }
        }
    }
}

// Launcher
void sparse_cache_manager_v7_cuda(
    torch::Tensor top_k_tokens,
    torch::Tensor device_buffer_tokens,
    torch::Tensor token_to_slot,
    torch::Tensor host_cache_locs,
    torch::Tensor device_buffer_locs,
    torch::Tensor host_cache,
    torch::Tensor device_buffer,
    torch::Tensor top_k_device_locs,
    int64_t item_size_bytes
) {
    const int32_t num_top_k = top_k_tokens.size(0);
    const int32_t hot_buffer_size = device_buffer_tokens.size(0);
    const int32_t max_tokens = token_to_slot.size(0);
    
    if (num_top_k <= 64 && hot_buffer_size <= 512) {
        sparse_cache_manager_v7_kernel<256, 64, 512><<<1, 256>>>(
            top_k_tokens.data_ptr<int32_t>(),
            device_buffer_tokens.data_ptr<int32_t>(),
            reinterpret_cast<uint16_t*>(token_to_slot.data_ptr<int16_t>()),
            host_cache_locs.data_ptr<int64_t>(),
            device_buffer_locs.data_ptr<int64_t>(),
            reinterpret_cast<const char*>(host_cache.data_ptr()),
            reinterpret_cast<char*>(device_buffer.data_ptr()),
            top_k_device_locs.data_ptr<int64_t>(),
            max_tokens,
            item_size_bytes
        );
    } else if (num_top_k <= 128 && hot_buffer_size <= 1024) {
        sparse_cache_manager_v7_kernel<256, 128, 1024><<<1, 256>>>(
            top_k_tokens.data_ptr<int32_t>(),
            device_buffer_tokens.data_ptr<int32_t>(),
            reinterpret_cast<uint16_t*>(token_to_slot.data_ptr<int16_t>()),
            host_cache_locs.data_ptr<int64_t>(),
            device_buffer_locs.data_ptr<int64_t>(),
            reinterpret_cast<const char*>(host_cache.data_ptr()),
            reinterpret_cast<char*>(device_buffer.data_ptr()),
            top_k_device_locs.data_ptr<int64_t>(),
            max_tokens,
            item_size_bytes
        );
    } else {
        sparse_cache_manager_v7_kernel<256, 256, 4096><<<1, 256>>>(
            top_k_tokens.data_ptr<int32_t>(),
            device_buffer_tokens.data_ptr<int32_t>(),
            reinterpret_cast<uint16_t*>(token_to_slot.data_ptr<int16_t>()),
            host_cache_locs.data_ptr<int64_t>(),
            device_buffer_locs.data_ptr<int64_t>(),
            reinterpret_cast<const char*>(host_cache.data_ptr()),
            reinterpret_cast<char*>(device_buffer.data_ptr()),
            top_k_device_locs.data_ptr<int64_t>(),
            max_tokens,
            item_size_bytes
        );
    }
}
'''

CPP_SOURCE = r'''
void sparse_cache_manager_v7_cuda(
    torch::Tensor top_k_tokens,
    torch::Tensor device_buffer_tokens,
    torch::Tensor token_to_slot,
    torch::Tensor host_cache_locs,
    torch::Tensor device_buffer_locs,
    torch::Tensor host_cache,
    torch::Tensor device_buffer,
    torch::Tensor top_k_device_locs,
    int64_t item_size_bytes
);
'''

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name='sparse_cache_v7',
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=['sparse_cache_manager_v7_cuda'],
            verbose=False,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
        )
    return _module


class SparseCacheManagerV7:
    """
    Sparse KV Cache Manager v7 - Atomic-Free Implementation
    
    Key improvements over v6:
    1. No atomicOr operations - all writes are conflict-free
    2. Use __ballot_sync result directly as miss bitmap
    3. Collect hit slots, then single-thread bitmap build
    
    Same data structures as v6:
    - token_to_slot: uint16_t[max_tokens], 0xFFFF = not present
    - device_buffer_tokens: int32_t[H], which token at each slot
    """
    
    NOT_PRESENT = 0xFFFF
    
    def __init__(
        self,
        max_tokens: int,
        hot_buffer_size: int,
        item_size_bytes: int,
        device: str = "cuda",
    ):
        assert hot_buffer_size <= 65535, "hot_buffer_size must fit in uint16"
        
        self.max_tokens = max_tokens
        self.hot_buffer_size = hot_buffer_size
        self.item_size_bytes = item_size_bytes
        self.device = device
        
        self.token_to_slot = torch.full(
            (max_tokens,), self.NOT_PRESENT, dtype=torch.int16, device=device
        )
        self.device_buffer_tokens = torch.full(
            (hot_buffer_size,), -1, dtype=torch.int32, device=device
        )
        self.device_buffer_locs = torch.arange(
            hot_buffer_size, dtype=torch.int64, device=device
        )
        self.host_cache_locs = torch.arange(
            max_tokens, dtype=torch.int64, device=device
        )
        
        self._module = None
    
    def _ensure_compiled(self):
        if self._module is None:
            self._module = _get_module()
    
    def reset(self):
        self.token_to_slot.fill_(self.NOT_PRESENT)
        self.device_buffer_tokens.fill_(-1)
    
    def set_host_cache_locs(self, host_cache_locs: torch.Tensor):
        self.host_cache_locs = host_cache_locs.to(self.device)
    
    def process_topk(
        self,
        top_k_tokens: torch.Tensor,
        host_cache: torch.Tensor,
        device_buffer: torch.Tensor,
    ) -> torch.Tensor:
        self._ensure_compiled()
        
        if top_k_tokens.dtype != torch.int32:
            top_k_tokens = top_k_tokens.to(torch.int32)
        
        top_k_size = top_k_tokens.numel()
        top_k_device_locs = torch.zeros(top_k_size, dtype=torch.int64, device=self.device)
        
        self._module.sparse_cache_manager_v7_cuda(
            top_k_tokens,
            self.device_buffer_tokens,
            self.token_to_slot,
            self.host_cache_locs,
            self.device_buffer_locs,
            host_cache,
            device_buffer,
            top_k_device_locs,
            self.item_size_bytes,
        )
        
        return top_k_device_locs
    
    def is_resident(self, token: int) -> bool:
        slot = self.token_to_slot[token].item()
        return slot != self.NOT_PRESENT and slot != -1
    
    def get_slot(self, token: int) -> int:
        slot = self.token_to_slot[token].item()
        if slot == -1 or slot == self.NOT_PRESENT:
            return -1
        return slot if slot >= 0 else slot + 65536
    
    def get_stats(self) -> dict:
        num_resident = ((self.token_to_slot != -1) & (self.token_to_slot != self.NOT_PRESENT)).sum().item()
        return {
            "num_resident": num_resident,
            "hot_buffer_size": self.hot_buffer_size,
            "item_size_bytes": self.item_size_bytes,
        }


def visualize_v7_atomic_free():
    """Visualize the atomic-free approach."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║           Sparse KV Cache Manager v7 - Atomic-Free Implementation            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  v6 Problem: atomicOr for shared memory bitmaps                              ║
║  ─────────────────────────────────────────────────                           ║
║                                                                              ║
║    // Multiple threads write to same bitmap word                             ║
║    atomicOr(&s_protected_bitmap[slot/32], 1u << (slot%32));                  ║
║    atomicOr(&s_is_miss_bitmap[idx/32], 1u << (idx%32));                      ║
║                                                                              ║
║  v7 Solution: Conflict-free writes                                           ║
║  ─────────────────────────────────                                           ║
║                                                                              ║
║  1. Miss Bitmap - Use __ballot_sync directly:                                ║
║                                                                              ║
║     Warp 0: lanes check tokens 0-31                                          ║
║             miss_mask = __ballot_sync(is_miss)                               ║
║             → miss_mask IS the bitmap word!                                  ║
║                                                                              ║
║     // Lane 0 stores (no conflict - one write per word)                      ║
║     if (lane_id == 0) s_miss_bitmap[warp_id] = miss_mask;                    ║
║                                                                              ║
║  2. Protected Bitmap - Collect then build:                                   ║
║                                                                              ║
║     Phase A: Hit threads store slots in array (conflict-free via prefix)     ║
║       global_idx = warp_hit_prefix[warp_id] + local_hit_offset;              ║
║       s_hit_slots[global_idx] = my_slot;  // Unique index!                   ║
║                                                                              ║
║     Phase B: Single thread builds bitmap (no conflicts)                      ║
║       if (tid == 0) {                                                        ║
║           for (i = 0; i < num_hits; i++) {                                   ║
║               slot = s_hit_slots[i];                                         ║
║               s_protected_bitmap[slot/32] |= (1u << (slot%32));              ║
║           }                                                                  ║
║       }                                                                      ║
║                                                                              ║
║  3. Evictable Slot Storage - Also conflict-free:                             ║
║                                                                              ║
║     my_pos = __popc(evict_mask & lanes_before);  // Position in warp         ║
║     if (is_evictable && my_pos < use_count) {                                ║
║         s_evictable_slots[found + my_pos] = my_slot;  // Unique index!       ║
║     }                                                                        ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Benefits:                                                                   ║
║  ─────────                                                                   ║
║  • No atomic contention = better performance                                 ║
║  • Deterministic behavior (no race conditions)                               ║
║  • Simpler memory model reasoning                                            ║
║                                                                              ║
║  Trade-offs:                                                                 ║
║  ───────────                                                                 ║
║  • Extra shared memory for hit slots array                                   ║
║  • Extra __syncthreads() barriers                                            ║
║  • Single-thread bitmap build (O(num_hits), but num_hits is small)           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    visualize_v7_atomic_free()
