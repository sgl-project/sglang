"""
Sparse KV Cache Manager v6 - 16-bit Token-to-Slot Index

Key optimization over v5:
- Replace 1-bit residence bitmap with 16-bit token_to_slot index
- 0xFFFF = not resident, 0 to HOT_BUFFER_SIZE-1 = valid slot index
- O(1) lookup for both residence check AND slot retrieval
- Eliminates linear search in device_buffer_tokens for hits

Memory trade-off:
- v5: 1 bit per token = max_tokens / 8 bytes
- v6: 16 bits per token = max_tokens * 2 bytes
- For max_tokens = 1M: 128KB (v5) vs 2MB (v6) - acceptable for O(1) lookup

uint16 warp efficiency:
- Warp loading 32 uint16 values = 64 bytes = one cache line (coalesced)
"""

import torch
from torch.utils.cpp_extension import load_inline
from typing import Tuple

CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int WARP_SIZE = 32;
constexpr uint16_t NOT_PRESENT = 0xFFFF;  // Sentinel for "not in buffer"

/**
 * Sparse Cache Manager v6
 * 
 * Data structures:
 *   top_k_tokens[K]: tokens we need for this iteration
 *   device_buffer_tokens[H]: which token is at each slot (-1 = empty)
 *   token_to_slot[max_tokens]: uint16_t, 0xFFFF = not present, else = slot index
 *   host_cache_locs[max_tokens]: CPU location for each token
 *   device_buffer_locs[H]: const, device location for each slot
 *   host_cache: CPU cache memory
 *   device_buffer: GPU buffer memory
 *   top_k_device_locs[K]: output, device locations for top_k
 * 
 * Key improvement over v5:
 *   Hit check: slot = token_to_slot[my_token]
 *   If slot != 0xFFFF, it's a hit AND we have the slot directly!
 *   No more linear search through device_buffer_tokens.
 */
template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE>
__global__ void sparse_cache_manager_v6_kernel(
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
    constexpr int BITMAP_WORDS_TOP_K = (NUM_TOP_K + 31) / 32;
    constexpr int BITMAP_WORDS_BUFFER = (HOT_BUFFER_SIZE + 31) / 32;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const unsigned int lanes_before = (1u << lane_id) - 1;
    
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
    
    // Initialize shared memory
    if (tid == 0) {
        s_total_misses = 0;
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
    
    // ===== Phase 1: O(1) Hit Detection via token_to_slot =====
    // Each thread checks one token
    // slot = token_to_slot[my_token]
    // if slot != 0xFFFF: HIT (and we have the slot!)
    // else: MISS
    
    const int warp_token_start = warp_id * WARP_SIZE;
    const int my_token_idx = warp_token_start + lane_id;
    const bool has_valid_token = my_token_idx < NUM_TOP_K;
    
    int32_t my_token = 0;
    bool is_hit = false;
    int32_t my_device_slot = -1;
    
    if (has_valid_token) {
        my_token = top_k_tokens[my_token_idx];
        
        // O(1) lookup: directly get slot from token_to_slot
        // No linear search needed!
        uint16_t slot = token_to_slot[my_token];
        is_hit = (slot != NOT_PRESENT);
        
        if (is_hit) {
            my_device_slot = static_cast<int32_t>(slot);
            
            // Output device location
            top_k_device_locs[my_token_idx] = device_buffer_locs[my_device_slot];
            
            // Mark slot as protected using uint32 atomic OR
            const int prot_word = my_device_slot / 32;
            const int prot_bit = my_device_slot % 32;
            atomicOr(&s_protected_bitmap[prot_word], 1u << prot_bit);
        }
    }
    
    // Intra-warp communication for miss counting
    const unsigned int miss_mask = __ballot_sync(0xFFFFFFFF, has_valid_token && !is_hit);
    const int warp_miss_count = __popc(miss_mask);
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
                
                // Clear old token from token_to_slot (set to NOT_PRESENT)
                if (old_token >= 0 && old_token < max_tokens) {
                    token_to_slot[old_token] = NOT_PRESENT;
                }
                
                // Set new token's slot in token_to_slot
                token_to_slot[my_token] = static_cast<uint16_t>(evict_slot);
                
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
 * Warp-cooperative copy version for larger item sizes
 */
template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE>
__global__ void sparse_cache_manager_v6_warp_copy_kernel(
    const int32_t* __restrict__ top_k_tokens,
    int32_t* __restrict__ device_buffer_tokens,
    uint16_t* __restrict__ token_to_slot,
    const int64_t* __restrict__ host_cache_locs,
    const int64_t* __restrict__ device_buffer_locs,
    const char* __restrict__ host_cache,
    char* __restrict__ device_buffer,
    int64_t* __restrict__ top_k_device_locs,
    // Pre-computed from phase 1-3
    const int32_t* __restrict__ miss_tokens,
    const int32_t* __restrict__ miss_top_k_idx,
    const int32_t* __restrict__ evict_slots,
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
    
    // Update token_to_slot (lane 0 only)
    if (lane_id == 0) {
        if (old_token >= 0 && old_token < max_tokens) {
            token_to_slot[old_token] = NOT_PRESENT;
        }
        token_to_slot[token] = static_cast<uint16_t>(evict_slot);
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
void sparse_cache_manager_v6_cuda(
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
    
    // Select template instantiation based on sizes
    if (num_top_k <= 64 && hot_buffer_size <= 512) {
        sparse_cache_manager_v6_kernel<256, 64, 512><<<1, 256>>>(
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
        sparse_cache_manager_v6_kernel<256, 128, 1024><<<1, 256>>>(
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
        sparse_cache_manager_v6_kernel<256, 256, 4096><<<1, 256>>>(
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
void sparse_cache_manager_v6_cuda(
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
            name='sparse_cache_v6',
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=['sparse_cache_manager_v6_cuda'],
            verbose=False,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
        )
    return _module


class SparseCacheManagerV6:
    """
    Sparse KV Cache Manager v6 - 16-bit Token-to-Slot Index
    
    Key improvement over v5:
    - Replace 1-bit residence bitmap with 16-bit token_to_slot index
    - 0xFFFF = not resident, 0 to HOT_BUFFER_SIZE-1 = valid slot index
    - O(1) lookup for BOTH residence check AND slot retrieval
    - Eliminates linear search in device_buffer_tokens
    
    Memory usage:
    - v5: 1 bit per token = max_tokens / 8 bytes
    - v6: 16 bits per token = max_tokens * 2 bytes
    - Trade-off: 16x more memory for O(1) vs O(H) lookup
    
    Data layout:
    - token_to_slot: uint16_t[max_tokens], 0xFFFF = not present, else = slot index
    - device_buffer_tokens: int32_t[H], which token is at each slot
    - device_buffer_locs: int64_t[H], const, device location for each slot
    """
    
    NOT_PRESENT = 0xFFFF
    
    def __init__(
        self,
        max_tokens: int,
        hot_buffer_size: int,
        item_size_bytes: int,
        device: str = "cuda",
    ):
        assert hot_buffer_size <= 65535, "hot_buffer_size must fit in uint16 (max 65535)"
        
        self.max_tokens = max_tokens
        self.hot_buffer_size = hot_buffer_size
        self.item_size_bytes = item_size_bytes
        self.device = device
        
        # Token to slot mapping: uint16_t, 0xFFFF = not present
        # This replaces the 1-bit residence bitmap with O(1) slot lookup
        self.token_to_slot = torch.full(
            (max_tokens,), self.NOT_PRESENT, dtype=torch.int16, device=device
        )
        
        # Device buffer tokens: which token is at each slot (-1 = empty)
        self.device_buffer_tokens = torch.full(
            (hot_buffer_size,), -1, dtype=torch.int32, device=device
        )
        
        # Device buffer locations: const, slot n → device_buffer_locs[n]
        self.device_buffer_locs = torch.arange(
            hot_buffer_size, dtype=torch.int64, device=device
        )
        
        # Host cache locations: token → host memory location
        self.host_cache_locs = torch.arange(
            max_tokens, dtype=torch.int64, device=device
        )
        
        self._module = None
    
    def _ensure_compiled(self):
        if self._module is None:
            self._module = _get_module()
    
    def reset(self):
        """Reset all state."""
        self.token_to_slot.fill_(self.NOT_PRESENT)
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
        
        self._module.sparse_cache_manager_v6_cuda(
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
        """Check if a token is resident in the device buffer."""
        slot = self.token_to_slot[token].item()
        return slot != self.NOT_PRESENT
    
    def get_slot(self, token: int) -> int:
        """Get the slot for a token. Returns -1 if not resident."""
        slot = self.token_to_slot[token].item()
        if slot == self.NOT_PRESENT:
            return -1
        # Handle signed/unsigned conversion
        return slot if slot >= 0 else slot + 65536
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        # Count resident tokens (those with slot != 0xFFFF)
        # Note: torch.int16 represents 0xFFFF as -1
        num_resident = (self.token_to_slot != -1).sum().item()
        
        return {
            "num_resident": num_resident,
            "hot_buffer_size": self.hot_buffer_size,
            "item_size_bytes": self.item_size_bytes,
            "memory_bytes": self.max_tokens * 2,  # 2 bytes per token
        }


def visualize_v6_design():
    """Visualize the v6 design with 16-bit token_to_slot."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║          Sparse KV Cache Manager v6 - 16-bit Token-to-Slot Index            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  v5 (1-bit residence bitmap):                                                ║
║  ──────────────────────────────                                              ║
║                                                                              ║
║    token_residence_bitmap[max_tokens/32] = { bit0, bit1, bit2, ... }        ║
║    device_buffer_tokens[H] = { token_at_slot_0, token_at_slot_1, ... }      ║
║                                                                              ║
║    Hit check:                                                                ║
║      1. is_resident = (bitmap[t/32] >> (t%32)) & 1        O(1)              ║
║      2. for s in 0..H:                                    O(H) ← SLOW!      ║
║           if device_buffer_tokens[s] == t: slot = s                         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  v6 (16-bit token_to_slot):                                                  ║
║  ────────────────────────────                                                ║
║                                                                              ║
║    token_to_slot[max_tokens] = { slot_for_token_0, slot_for_token_1, ... }  ║
║    (0xFFFF = not present, 0 to H-1 = valid slot index)                      ║
║                                                                              ║
║    Hit check:                                                                ║
║      slot = token_to_slot[t]                              O(1)              ║
║      is_hit = (slot != 0xFFFF)                            O(1)              ║
║      device_loc = device_buffer_locs[slot]                O(1)              ║
║                                                                              ║
║    Total: O(1) for residence + slot lookup!                                 ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Memory Trade-off:                                                           ║
║  ─────────────────                                                           ║
║                                                                              ║
║    max_tokens    v5 (1-bit)      v6 (16-bit)                                ║
║    ──────────    ──────────      ────────────                                ║
║    64K           8 KB            128 KB                                      ║
║    256K          32 KB           512 KB                                      ║
║    1M            128 KB          2 MB                                        ║
║    16M           2 MB            32 MB                                       ║
║                                                                              ║
║  Trade-off: 16x more memory for O(1) vs O(H) lookup                         ║
║  With H=4096, the O(H) scan would need 4096 iterations per hit!             ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Warp Efficiency:                                                            ║
║  ────────────────                                                            ║
║                                                                              ║
║  uint16 access pattern:                                                      ║
║    - Warp loads 32 × uint16 = 64 bytes = 1 cache line                       ║
║    - If tokens are sequential, access is perfectly coalesced                 ║
║    - Same efficiency as uint32 bitmap (one transaction per warp)             ║
║                                                                              ║
║  Example (sequential top_k_tokens = [100, 101, 102, ...]):                   ║
║    All 32 lanes load token_to_slot[100..131] in one coalesced transaction   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    visualize_v6_design()
