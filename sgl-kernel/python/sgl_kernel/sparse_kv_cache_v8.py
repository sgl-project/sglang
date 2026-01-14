"""
Sparse KV Cache Manager v8 - Minimal Shared Memory

Key improvements over v7:
- Eliminate s_warp_miss_count and s_warp_hit_count
- Derive counts from s_miss_bitmap using __popc()
- hit_count = num_valid - miss_count (complementary)

Shared memory reduction:
- v7: s_warp_miss_count[NUM_WARPS] + s_warp_hit_count[NUM_WARPS]
- v8: Neither needed (derived from s_miss_bitmap)
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
 * Sparse Cache Manager v8 - Minimal Shared Memory
 * 
 * Key insight: s_miss_bitmap[w] already contains all info needed!
 *   miss_count = __popc(s_miss_bitmap[w])
 *   hit_count = num_valid_in_warp - miss_count
 * 
 * Eliminated shared memory:
 *   - s_warp_miss_count (derive from bitmap)
 *   - s_warp_hit_count (derive from bitmap)
 */
template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE>
__global__ void sparse_cache_manager_v8_kernel(
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
    
    // ===== Shared Memory (reduced from v7) =====
    // Miss bitmap: stores ballot results, used to derive counts
    __shared__ uint32_t s_miss_bitmap[NUM_WARPS_FOR_TOPK];
    // Hit slots collection
    __shared__ int32_t s_hit_slots[NUM_TOP_K];
    // Prefix sums (computed from bitmap)
    __shared__ int32_t s_warp_hit_prefix[NUM_WARPS_FOR_TOPK + 1];
    __shared__ int32_t s_warp_miss_prefix[NUM_WARPS_FOR_TOPK + 1];
    // Per-token miss offset
    __shared__ int32_t s_local_miss_offset[NUM_TOP_K];
    // Protected bitmap
    __shared__ uint32_t s_protected_bitmap[BITMAP_WORDS_BUFFER];
    // Evictable slots
    __shared__ int32_t s_evictable_slots[NUM_TOP_K];
    __shared__ int32_t s_total_misses;
    __shared__ int32_t s_total_hits;
    
    // Initialize
    if (tid == 0) {
        s_total_misses = 0;
        s_total_hits = 0;
    }
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
        uint16_t slot = token_to_slot[my_token];
        is_hit = (slot != NOT_PRESENT);
        
        if (is_hit) {
            my_device_slot = static_cast<int32_t>(slot);
            top_k_device_locs[my_token_idx] = device_buffer_locs[my_device_slot];
        }
    }
    
    // Warp-level ballot
    const unsigned int hit_mask = __ballot_sync(0xFFFFFFFF, has_valid_token && is_hit);
    const unsigned int miss_mask = __ballot_sync(0xFFFFFFFF, has_valid_token && !is_hit);
    
    const int local_hit_offset = __popc(hit_mask & lanes_before);
    const int local_miss_offset = __popc(miss_mask & lanes_before);
    
    // Store local miss offset
    if (has_valid_token && !is_hit) {
        s_local_miss_offset[my_token_idx] = local_miss_offset;
    }
    
    // Warp leader stores miss bitmap only (counts derived later)
    if (lane_id == 0 && warp_id < NUM_WARPS_FOR_TOPK) {
        s_miss_bitmap[warp_id] = miss_mask;
    }
    __syncthreads();
    
    // ===== Phase 2: Compute Prefix Sums from Bitmap =====
    if (tid == 0) {
        s_warp_hit_prefix[0] = 0;
        s_warp_miss_prefix[0] = 0;
        
        for (int w = 0; w < NUM_WARPS_FOR_TOPK; w++) {
            // Number of valid tokens in this warp
            const int first_token = w * WARP_SIZE;
            const int num_valid = min(WARP_SIZE, NUM_TOP_K - first_token);
            
            // Derive counts from bitmap
            const int miss_count = __popc(s_miss_bitmap[w]);
            const int hit_count = num_valid - miss_count;
            
            s_warp_hit_prefix[w + 1] = s_warp_hit_prefix[w] + hit_count;
            s_warp_miss_prefix[w + 1] = s_warp_miss_prefix[w] + miss_count;
        }
        
        s_total_hits = s_warp_hit_prefix[NUM_WARPS_FOR_TOPK];
        s_total_misses = s_warp_miss_prefix[NUM_WARPS_FOR_TOPK];
    }
    __syncthreads();
    
    // ===== Phase 3: Collect Hit Slots =====
    if (has_valid_token && is_hit && warp_id < NUM_WARPS_FOR_TOPK) {
        const int global_hit_idx = s_warp_hit_prefix[warp_id] + local_hit_offset;
        s_hit_slots[global_hit_idx] = my_device_slot;
    }
    __syncthreads();
    
    // Early exit if all hits
    if (s_total_misses == 0) {
        return;
    }
    
    // ===== Phase 4: Build Protected Bitmap =====
    for (int i = tid; i < BITMAP_WORDS_BUFFER; i += BLOCK_SIZE) {
        s_protected_bitmap[i] = 0;
    }
    __syncthreads();
    
    if (tid == 0) {
        for (int i = 0; i < s_total_hits; i++) {
            const int slot = s_hit_slots[i];
            s_protected_bitmap[slot / 32] |= (1u << (slot % 32));
        }
    }
    __syncthreads();
    
    // ===== Phase 5: Find Evictable Slots =====
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
                is_evictable = !((prot_bitmap >> prot_bit) & 1);
            }
            
            const unsigned int evict_mask = __ballot_sync(0xFFFFFFFF, is_evictable);
            const int num_evictable = __popc(evict_mask);
            const int need = s_total_misses - found;
            const int use_count = min(num_evictable, need);
            
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
        const uint32_t miss_word = s_miss_bitmap[warp_id];
        const bool is_miss = (miss_word >> lane_id) & 1;
        
        if (is_miss) {
            const int global_slot = s_warp_miss_prefix[warp_id] + s_local_miss_offset[my_token_idx];
            
            if (global_slot < s_total_misses) {
                const int evict_slot = s_evictable_slots[global_slot];
                const int32_t old_token = device_buffer_tokens[evict_slot];
                
                if (old_token >= 0 && old_token < max_tokens) {
                    token_to_slot[old_token] = NOT_PRESENT;
                }
                
                token_to_slot[my_token] = static_cast<uint16_t>(evict_slot);
                device_buffer_tokens[evict_slot] = my_token;
                top_k_device_locs[my_token_idx] = device_buffer_locs[evict_slot];
                
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

void sparse_cache_manager_v8_cuda(
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
        sparse_cache_manager_v8_kernel<256, 64, 512><<<1, 256>>>(
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
        sparse_cache_manager_v8_kernel<256, 128, 1024><<<1, 256>>>(
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
        sparse_cache_manager_v8_kernel<256, 256, 4096><<<1, 256>>>(
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
void sparse_cache_manager_v8_cuda(
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
            name='sparse_cache_v8',
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=['sparse_cache_manager_v8_cuda'],
            verbose=False,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
        )
    return _module


class SparseCacheManagerV8:
    """
    Sparse KV Cache Manager v8 - Minimal Shared Memory
    
    Eliminated from v7:
    - s_warp_miss_count (derived from s_miss_bitmap via __popc)
    - s_warp_hit_count (derived: num_valid - miss_count)
    """
    
    NOT_PRESENT = 0xFFFF
    
    def __init__(
        self,
        max_tokens: int,
        hot_buffer_size: int,
        item_size_bytes: int,
        device: str = "cuda",
    ):
        assert hot_buffer_size <= 65535
        
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
        
        self._module.sparse_cache_manager_v8_cuda(
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


def visualize_v8_optimization():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║         Sparse KV Cache Manager v8 - Minimal Shared Memory                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  v7 Shared Memory:                                                           ║
║  ─────────────────                                                           ║
║    s_miss_bitmap[NUM_WARPS]      ← Stores ballot results                     ║
║    s_warp_miss_count[NUM_WARPS]  ← REDUNDANT! Can derive from bitmap         ║
║    s_warp_hit_count[NUM_WARPS]   ← REDUNDANT! Can derive from bitmap         ║
║                                                                              ║
║  v8 Optimization:                                                            ║
║  ────────────────                                                            ║
║    s_miss_bitmap[NUM_WARPS]      ← Keep (stores ballot results)              ║
║                                                                              ║
║    // Derive counts in prefix sum phase:                                     ║
║    for (int w = 0; w < NUM_WARPS; w++) {                                     ║
║        int num_valid = min(32, NUM_TOP_K - w * 32);                          ║
║        int miss_count = __popc(s_miss_bitmap[w]);  // From bitmap!           ║
║        int hit_count = num_valid - miss_count;     // Complementary!         ║
║        ...                                                                   ║
║    }                                                                         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Shared Memory Savings (NUM_WARPS = 8):                                      ║
║  ──────────────────────────────────────                                      ║
║    v7: s_warp_miss_count[8] + s_warp_hit_count[8] = 64 bytes                 ║
║    v8: 0 bytes                                                               ║
║                                                                              ║
║  Key Insight:                                                                ║
║  ────────────                                                                ║
║    __ballot_sync returns a 32-bit mask where each bit = one lane's result   ║
║    __popc(mask) = count of set bits = count of true results                 ║
║                                                                              ║
║    Since we already store the ballot result in s_miss_bitmap,                ║
║    the count is implicitly stored! Just use __popc() to extract it.          ║
║                                                                              ║
║    And since valid tokens are partitioned into hits and misses:              ║
║      hit_count + miss_count = num_valid_tokens                               ║
║      hit_count = num_valid - miss_count                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    visualize_v8_optimization()
