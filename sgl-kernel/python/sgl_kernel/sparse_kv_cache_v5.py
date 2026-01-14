"""
Sparse KV Cache Manager v5 - Final Optimized Implementation

Key design changes:
1. device_buffer_locs is const: slot n → device_buffer_locs[n]
2. Direct copy: host_cache[host_cache_locs[token]] → device_buffer[device_buffer_locs[slot]]
3. uint32_t bitmaps for warp-efficient IO (one load per warp, each lane gets its bit)
4. Returns only top_k_device_locs

uint32_t Bitmap Efficiency:
- Each warp loads ONE uint32 from bitmap
- Each lane extracts its bit via (bitmap >> lane_id) & 1
- 32x fewer memory transactions than byte-level access
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

/**
 * Sparse Cache Manager v5
 * 
 * Data structures:
 *   top_k_tokens[K]: tokens we need for this iteration
 *   device_buffer_tokens[H]: which token is at each slot
 *   token_residence_bitmap[max_tokens/32]: uint32 bitmap, bit set if token is resident
 *   host_cache_locs[max_tokens]: CPU location for each token
 *   device_buffer_locs[H]: const, device location for each slot
 *   host_cache: CPU cache memory
 *   device_buffer: GPU buffer memory
 *   top_k_device_locs[K]: output, device locations for top_k
 * 
 * uint32_t bitmap layout:
 *   Token t is resident iff: (token_residence_bitmap[t/32] >> (t%32)) & 1
 *   Warp-efficient: one load per warp, each lane extracts its bit
 */
template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE>
__global__ void sparse_cache_manager_v5_kernel(
    const int32_t* __restrict__ top_k_tokens,
    int32_t* __restrict__ device_buffer_tokens,
    uint32_t* __restrict__ token_residence_bitmap,
    const int64_t* __restrict__ host_cache_locs,
    const int64_t* __restrict__ device_buffer_locs,
    const char* __restrict__ host_cache,
    char* __restrict__ device_buffer,
    int64_t* __restrict__ top_k_device_locs,
    int32_t max_tokens,
    int64_t item_size_bytes
) {
    static_assert(NUM_TOP_K <= 256, "NUM_TOP_K must be <= 256");
    static_assert(HOT_BUFFER_SIZE <= 4096, "HOT_BUFFER_SIZE must be <= 4096");
    
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    constexpr int BITMAP_WORDS_TOP_K = (NUM_TOP_K + 31) / 32;
    constexpr int BITMAP_WORDS_BUFFER = (HOT_BUFFER_SIZE + 31) / 32;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // ===== Shared Memory =====
    // Protected slots bitmap (uint32 for warp-efficient access)
    __shared__ uint32_t s_protected_bitmap[BITMAP_WORDS_BUFFER];
    // Warp-level miss tracking
    __shared__ int32_t s_warp_miss_count[NUM_WARPS];
    __shared__ int32_t s_warp_miss_prefix[NUM_WARPS + 1];
    // Per-token info
    __shared__ int32_t s_local_offset[NUM_TOP_K];
    __shared__ uint32_t s_is_miss_bitmap[BITMAP_WORDS_TOP_K];
    // Evictable slots (only need num_misses worth)
    __shared__ int32_t s_evictable_slots[NUM_TOP_K];
    __shared__ int32_t s_total_misses;
    __shared__ int32_t s_evict_ptr;
    
    // Initialize shared memory
    if (tid == 0) {
        s_total_misses = 0;
        s_evict_ptr = 0;
    }
    // Clear bitmaps (warp-efficient: each thread clears one word)
    for (int i = tid; i < BITMAP_WORDS_BUFFER; i += BLOCK_SIZE) {
        s_protected_bitmap[i] = 0;
    }
    for (int i = tid; i < BITMAP_WORDS_TOP_K; i += BLOCK_SIZE) {
        s_is_miss_bitmap[i] = 0;
    }
    for (int i = tid; i < NUM_WARPS; i += BLOCK_SIZE) {
        s_warp_miss_count[i] = 0;
    }
    __syncthreads();
    
    // ===== Phase 1: Warp-Parallel Hit Detection with uint32 Bitmap =====
    // Each warp processes WARP_SIZE tokens
    // One warp loads one uint32 from residence bitmap, each lane gets its bit
    
    const int warp_token_start = warp_id * WARP_SIZE;
    const int my_token_idx = warp_token_start + lane_id;
    const bool has_valid_token = my_token_idx < NUM_TOP_K;
    
    int32_t my_token = 0;
    bool is_hit = false;
    int32_t my_device_slot = -1;
    
    if (has_valid_token) {
        my_token = top_k_tokens[my_token_idx];
        
        // Warp-efficient bitmap load: one load per warp
        // All lanes in warp load the same word, then extract their bit
        const int bitmap_word_idx = my_token / 32;
        const int bit_idx = my_token % 32;
        
        // Load residence bitmap word (warp-coalesced if tokens are sequential)
        uint32_t residence_word = token_residence_bitmap[bitmap_word_idx];
        is_hit = (residence_word >> bit_idx) & 1;
        
        if (is_hit) {
            // Find which slot has this token
            // Linear search in device_buffer_tokens (could optimize with reverse index)
            for (int s = 0; s < HOT_BUFFER_SIZE; s++) {
                if (device_buffer_tokens[s] == my_token) {
                    my_device_slot = s;
                    break;
                }
            }
            
            if (my_device_slot >= 0) {
                // Output device location
                top_k_device_locs[my_token_idx] = device_buffer_locs[my_device_slot];
                
                // Mark slot as protected using uint32 atomic OR
                const int prot_word = my_device_slot / 32;
                const int prot_bit = my_device_slot % 32;
                atomicOr(&s_protected_bitmap[prot_word], 1u << prot_bit);
            }
        }
    }
    
    // Intra-warp communication for miss counting
    const unsigned int miss_mask = __ballot_sync(0xFFFFFFFF, has_valid_token && !is_hit);
    const int warp_miss_count = __popc(miss_mask);
    const unsigned int lanes_before = (1u << lane_id) - 1;
    const int local_offset = __popc(miss_mask & lanes_before);
    
    // Store miss info
    if (has_valid_token && !is_hit) {
        s_local_offset[my_token_idx] = local_offset;
        // Set bit in miss bitmap
        const int miss_word = my_token_idx / 32;
        const int miss_bit = my_token_idx % 32;
        atomicOr(&s_is_miss_bitmap[miss_word], 1u << miss_bit);
    }
    
    // Warp leader stores miss count
    if (lane_id == 0) {
        s_warp_miss_count[warp_id] = warp_miss_count;
    }
    __syncthreads();
    
    // ===== Phase 2: Prefix Sum =====
    if (tid == 0) {
        s_warp_miss_prefix[0] = 0;
        const int num_active_warps = (NUM_TOP_K + WARP_SIZE - 1) / WARP_SIZE;
        for (int w = 0; w < num_active_warps; w++) {
            s_warp_miss_prefix[w + 1] = s_warp_miss_prefix[w] + s_warp_miss_count[w];
        }
        s_total_misses = s_warp_miss_prefix[num_active_warps];
    }
    __syncthreads();
    
    // Early exit if all hits
    if (s_total_misses == 0) {
        return;
    }
    
    // ===== Phase 3: Find Evictable Slots (Warp-Parallel with uint32 Protected Bitmap) =====
    // Only find exactly s_total_misses slots, then stop
    
    if (warp_id == 0) {
        int found = 0;
        int base_slot = 0;
        
        while (found < s_total_misses && base_slot < HOT_BUFFER_SIZE) {
            const int my_slot = base_slot + lane_id;
            bool is_evictable = false;
            
            if (my_slot < HOT_BUFFER_SIZE) {
                // Warp-efficient: load one uint32 protected bitmap word per warp
                const int prot_word = my_slot / 32;
                const int prot_bit = my_slot % 32;
                
                // All lanes in this iteration share the same word
                uint32_t prot_bitmap = s_protected_bitmap[prot_word];
                const bool is_protected = (prot_bitmap >> prot_bit) & 1;
                
                is_evictable = !is_protected;
            }
            
            // Warp ballot to find evictable slots
            const unsigned int evict_mask = __ballot_sync(0xFFFFFFFF, is_evictable);
            const int num_evictable = __popc(evict_mask);
            const int need = s_total_misses - found;
            const int use_count = min(num_evictable, need);
            
            // Each evictable lane stores its slot
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
    
    // ===== Phase 4: Assignment + Direct Data Copy =====
    // Check if this thread handles a miss
    if (has_valid_token) {
        const int miss_word = my_token_idx / 32;
        const int miss_bit = my_token_idx % 32;
        const bool is_miss = (s_is_miss_bitmap[miss_word] >> miss_bit) & 1;
        
        if (is_miss) {
            const int global_slot = s_warp_miss_prefix[warp_id] + s_local_offset[my_token_idx];
            
            if (global_slot < s_total_misses) {
                const int evict_slot = s_evictable_slots[global_slot];
                const int32_t old_token = device_buffer_tokens[evict_slot];
                
                // Clear old token from residence bitmap
                if (old_token >= 0 && old_token < max_tokens) {
                    const int old_word = old_token / 32;
                    const int old_bit = old_token % 32;
                    atomicAnd(&token_residence_bitmap[old_word], ~(1u << old_bit));
                }
                
                // Set new token in residence bitmap
                const int new_word = my_token / 32;
                const int new_bit = my_token % 32;
                atomicOr(&token_residence_bitmap[new_word], 1u << new_bit);
                
                // Update device buffer token mapping
                device_buffer_tokens[evict_slot] = my_token;
                
                // Output device location
                top_k_device_locs[my_token_idx] = device_buffer_locs[evict_slot];
                
                // Direct data copy: host_cache[host_loc] -> device_buffer[device_loc]
                const int64_t host_loc = host_cache_locs[my_token];
                const int64_t device_loc = device_buffer_locs[evict_slot];
                
                // Copy item (each thread copies its own item)
                const char* src = host_cache + host_loc * item_size_bytes;
                char* dst = device_buffer + device_loc * item_size_bytes;
                
                // Simple byte copy (could optimize with vectorized loads)
                for (int64_t b = 0; b < item_size_bytes; b++) {
                    dst[b] = src[b];
                }
            }
        }
    }
}

/**
 * Warp-cooperative copy version: each warp handles one copy
 * Better for larger item sizes
 */
template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE>
__global__ void sparse_cache_manager_v5_warp_copy_kernel(
    const int32_t* __restrict__ top_k_tokens,
    int32_t* __restrict__ device_buffer_tokens,
    uint32_t* __restrict__ token_residence_bitmap,
    const int64_t* __restrict__ host_cache_locs,
    const int64_t* __restrict__ device_buffer_locs,
    const char* __restrict__ host_cache,
    char* __restrict__ device_buffer,
    int64_t* __restrict__ top_k_device_locs,
    // Pre-computed from phase 1-3
    const int32_t* __restrict__ miss_tokens,      // Tokens that are misses
    const int32_t* __restrict__ miss_top_k_idx,   // Their indices in top_k
    const int32_t* __restrict__ evict_slots,      // Slots to evict
    int32_t num_misses,
    int32_t max_tokens,
    int64_t item_size_bytes
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    if (warp_id >= num_misses) return;
    
    const int32_t token = miss_tokens[warp_id];
    const int32_t top_k_idx = miss_top_k_idx[warp_id];
    const int32_t evict_slot = evict_slots[warp_id];
    const int32_t old_token = device_buffer_tokens[evict_slot];
    
    // Update bitmap (lane 0 only)
    if (lane_id == 0) {
        if (old_token >= 0 && old_token < max_tokens) {
            const int old_word = old_token / 32;
            const int old_bit = old_token % 32;
            atomicAnd(&token_residence_bitmap[old_word], ~(1u << old_bit));
        }
        
        const int new_word = token / 32;
        const int new_bit = token % 32;
        atomicOr(&token_residence_bitmap[new_word], 1u << new_bit);
        
        device_buffer_tokens[evict_slot] = token;
        top_k_device_locs[top_k_idx] = device_buffer_locs[evict_slot];
    }
    
    // Warp-cooperative copy
    const int64_t host_loc = host_cache_locs[token];
    const int64_t device_loc = device_buffer_locs[evict_slot];
    
    const uint64_t* src = reinterpret_cast<const uint64_t*>(host_cache + host_loc * item_size_bytes);
    uint64_t* dst = reinterpret_cast<uint64_t*>(device_buffer + device_loc * item_size_bytes);
    
    const int num_chunks = item_size_bytes / sizeof(uint64_t);
    for (int i = lane_id; i < num_chunks; i += WARP_SIZE) {
        dst[i] = src[i];
    }
}

// Launcher with fixed template parameters
void sparse_cache_manager_v5_cuda(
    torch::Tensor top_k_tokens,
    torch::Tensor device_buffer_tokens,
    torch::Tensor token_residence_bitmap,
    torch::Tensor host_cache_locs,
    torch::Tensor device_buffer_locs,
    torch::Tensor host_cache,
    torch::Tensor device_buffer,
    torch::Tensor top_k_device_locs,
    int64_t item_size_bytes
) {
    const int32_t num_top_k = top_k_tokens.size(0);
    const int32_t hot_buffer_size = device_buffer_tokens.size(0);
    const int32_t max_tokens = token_residence_bitmap.size(0) * 32;
    
    // Select template instantiation based on sizes
    if (num_top_k <= 64 && hot_buffer_size <= 512) {
        sparse_cache_manager_v5_kernel<256, 64, 512><<<1, 256>>>(
            top_k_tokens.data_ptr<int32_t>(),
            device_buffer_tokens.data_ptr<int32_t>(),
            token_residence_bitmap.data_ptr<uint32_t>(),
            host_cache_locs.data_ptr<int64_t>(),
            device_buffer_locs.data_ptr<int64_t>(),
            reinterpret_cast<const char*>(host_cache.data_ptr()),
            reinterpret_cast<char*>(device_buffer.data_ptr()),
            top_k_device_locs.data_ptr<int64_t>(),
            max_tokens,
            item_size_bytes
        );
    } else if (num_top_k <= 128 && hot_buffer_size <= 1024) {
        sparse_cache_manager_v5_kernel<256, 128, 1024><<<1, 256>>>(
            top_k_tokens.data_ptr<int32_t>(),
            device_buffer_tokens.data_ptr<int32_t>(),
            token_residence_bitmap.data_ptr<uint32_t>(),
            host_cache_locs.data_ptr<int64_t>(),
            device_buffer_locs.data_ptr<int64_t>(),
            reinterpret_cast<const char*>(host_cache.data_ptr()),
            reinterpret_cast<char*>(device_buffer.data_ptr()),
            top_k_device_locs.data_ptr<int64_t>(),
            max_tokens,
            item_size_bytes
        );
    } else {
        sparse_cache_manager_v5_kernel<256, 256, 4096><<<1, 256>>>(
            top_k_tokens.data_ptr<int32_t>(),
            device_buffer_tokens.data_ptr<int32_t>(),
            token_residence_bitmap.data_ptr<uint32_t>(),
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
void sparse_cache_manager_v5_cuda(
    torch::Tensor top_k_tokens,
    torch::Tensor device_buffer_tokens,
    torch::Tensor token_residence_bitmap,
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
            name='sparse_cache_v5',
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=['sparse_cache_manager_v5_cuda'],
            verbose=False,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
        )
    return _module


class SparseCacheManagerV5:
    """
    Sparse KV Cache Manager v5 - Final Optimized Implementation
    
    Key features:
    1. device_buffer_locs is const: slot n → device_buffer_locs[n]
    2. Direct copy: host_cache[host_cache_locs[token]] → device_buffer[device_buffer_locs[slot]]
    3. uint32_t bitmaps for warp-efficient IO
    4. Returns only top_k_device_locs
    
    Data layout:
    - token_residence_bitmap: uint32_t[max_tokens/32], bit set if token is resident
    - device_buffer_tokens: int32_t[H], which token is at each slot
    - device_buffer_locs: int64_t[H], const, device location for each slot
    """
    
    def __init__(
        self,
        max_tokens: int,
        hot_buffer_size: int,
        item_size_bytes: int,
        device: str = "cuda",
    ):
        self.max_tokens = max_tokens
        self.hot_buffer_size = hot_buffer_size
        self.item_size_bytes = item_size_bytes
        self.device = device
        
        # Token residence bitmap (uint32 for warp efficiency)
        # Bit i of word j is set if token j*32+i is resident
        bitmap_words = (max_tokens + 31) // 32
        self.token_residence_bitmap = torch.zeros(
            bitmap_words, dtype=torch.int32, device=device
        )
        
        # Device buffer tokens: which token is at each slot (-1 = empty)
        self.device_buffer_tokens = torch.full(
            (hot_buffer_size,), -1, dtype=torch.int32, device=device
        )
        
        # Device buffer locations: const, slot n → device_buffer_locs[n]
        # These are the actual memory locations in device_buffer
        self.device_buffer_locs = torch.arange(
            hot_buffer_size, dtype=torch.int64, device=device
        )
        
        # Host cache locations: token → host memory location
        # This would be set by the user based on their CPU memory layout
        self.host_cache_locs = torch.arange(
            max_tokens, dtype=torch.int64, device=device
        )
        
        self._module = None
    
    def _ensure_compiled(self):
        if self._module is None:
            self._module = _get_module()
    
    def reset(self):
        """Reset all state."""
        self.token_residence_bitmap.zero_()
        self.device_buffer_tokens.fill_(-1)
    
    def set_host_cache_locs(self, host_cache_locs: torch.Tensor):
        """Set the host cache location mapping."""
        self.host_cache_locs = host_cache_locs.to(self.device)
    
    def process_topk(
        self,
        top_k_tokens: torch.Tensor,
        host_cache: torch.Tensor,
        device_buffer: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process top-k tokens.
        
        Args:
            top_k_tokens: [K] int32, tokens needed for this iteration
            host_cache: CPU pinned memory cache
            device_buffer: GPU buffer
            
        Returns:
            top_k_device_locs: [K] int64, device locations for each top_k token
        """
        self._ensure_compiled()
        
        if top_k_tokens.dtype != torch.int32:
            top_k_tokens = top_k_tokens.to(torch.int32)
        
        top_k_size = top_k_tokens.numel()
        top_k_device_locs = torch.zeros(top_k_size, dtype=torch.int64, device=self.device)
        
        self._module.sparse_cache_manager_v5_cuda(
            top_k_tokens,
            self.device_buffer_tokens,
            self.token_residence_bitmap.view(torch.int32).view(-1),  # Reinterpret as uint32
            self.host_cache_locs,
            self.device_buffer_locs,
            host_cache,
            device_buffer,
            top_k_device_locs,
            self.item_size_bytes,
        )
        
        return top_k_device_locs
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        # Count set bits in residence bitmap
        bitmap = self.token_residence_bitmap.view(torch.int32)
        num_resident = 0
        for word in bitmap.cpu().numpy():
            num_resident += bin(word & 0xFFFFFFFF).count('1')
        
        return {
            "num_resident": num_resident,
            "hot_buffer_size": self.hot_buffer_size,
            "item_size_bytes": self.item_size_bytes,
        }


def visualize_uint32_bitmap():
    """Visualize the uint32 bitmap layout and warp-efficient access."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     uint32_t Bitmap for Warp-Efficient IO                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Token Residence Bitmap Layout:                                              ║
║  ─────────────────────────────                                               ║
║                                                                              ║
║  token_residence_bitmap[max_tokens/32] = {                                   ║
║                                                                              ║
║    Word 0:  [bit31 ... bit1 bit0]  ← Tokens 0-31                            ║
║    Word 1:  [bit31 ... bit1 bit0]  ← Tokens 32-63                           ║
║    Word 2:  [bit31 ... bit1 bit0]  ← Tokens 64-95                           ║
║    ...                                                                       ║
║  }                                                                           ║
║                                                                              ║
║  Token t is resident iff: (bitmap[t/32] >> (t%32)) & 1                       ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Warp-Efficient Access Pattern:                                              ║
║  ─────────────────────────────                                               ║
║                                                                              ║
║  Traditional (byte bitmap):                                                  ║
║    Lane 0: load bitmap[token0]   ← 32 separate loads per warp!              ║
║    Lane 1: load bitmap[token1]                                               ║
║    ...                                                                       ║
║    Lane 31: load bitmap[token31]                                             ║
║                                                                              ║
║  Optimized (uint32 bitmap, sequential tokens):                               ║
║    All lanes: load ONE bitmap[tokens/32]  ← 1 coalesced load!               ║
║    Lane k: bit = (word >> k) & 1                                             ║
║                                                                              ║
║  Memory transactions: 32 → 1 (32x reduction!)                                ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Protected Slots Bitmap (shared memory):                                     ║
║  ───────────────────────────────────────                                     ║
║                                                                              ║
║  s_protected_bitmap[HOT_BUFFER_SIZE/32] in shared memory                     ║
║                                                                              ║
║  When checking if slot s is protected:                                       ║
║    uint32_t word = s_protected_bitmap[s/32];                                 ║
║    bool is_protected = (word >> (s%32)) & 1;                                 ║
║                                                                              ║
║  For warp checking slots [base, base+31]:                                    ║
║    All lanes load: s_protected_bitmap[base/32]  ← 1 shared mem load         ║
║    Lane k extracts: (word >> k) & 1                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
