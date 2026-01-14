"""
Sparse KV Cache Manager v10 - Single Kernel with Warp Chunking

Key idea: Each warp processes multiple chunks of 32 tokens in a loop.
- Warp w handles tokens: w*32, w*32+BLOCK_SIZE, w*32+2*BLOCK_SIZE, ...
- All processing stays in one kernel (no multi-kernel overhead)
- Requires shared memory large enough for NUM_TOP_K arrays

Shared memory sizing (for NUM_TOP_K=4096, HOT_BUFFER_SIZE=4096):
- s_hit_slots[4096] = 16KB
- s_local_miss_offset[4096] = 16KB
- s_miss_bitmap[128] = 512B
- s_protected_bitmap[128] = 512B
- Other = ~1KB
Total: ~34KB (fits in 48KB shared memory)
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
 * Sparse Cache Manager v10 - Single Kernel with Warp Chunking
 * 
 * Each warp processes multiple chunks of 32 tokens:
 *   Warp w processes: tokens[w*32 + chunk*BLOCK_SIZE] for chunk = 0, 1, 2, ...
 * 
 * This allows handling large NUM_TOP_K in a single kernel.
 */
template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE>
__global__ void sparse_cache_manager_v10_kernel(
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
    static_assert(HOT_BUFFER_SIZE <= 65535, "HOT_BUFFER_SIZE must fit in uint16");
    
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    constexpr int CHUNKS_PER_WARP = (NUM_TOP_K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int BITMAP_WORDS_TOPK = (NUM_TOP_K + 31) / 32;
    constexpr int BITMAP_WORDS_BUFFER = (HOT_BUFFER_SIZE + 31) / 32;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const unsigned int lanes_before = (1u << lane_id) - 1;
    
    // ===== Shared Memory =====
    __shared__ uint32_t s_miss_bitmap[BITMAP_WORDS_TOPK];      // Miss flags as bitmap
    __shared__ int32_t s_hit_slots[NUM_TOP_K];                  // Slot for each hit
    __shared__ int32_t s_local_miss_offset[NUM_TOP_K];          // Local offset within chunk
    __shared__ int32_t s_chunk_miss_count[CHUNKS_PER_WARP * NUM_WARPS];  // Misses per chunk
    __shared__ int32_t s_chunk_hit_count[CHUNKS_PER_WARP * NUM_WARPS];   // Hits per chunk
    __shared__ uint32_t s_protected_bitmap[BITMAP_WORDS_BUFFER];
    __shared__ int32_t s_evictable_slots[NUM_TOP_K];
    __shared__ int32_t s_total_misses;
    __shared__ int32_t s_total_hits;
    __shared__ int32_t s_chunk_miss_prefix[CHUNKS_PER_WARP * NUM_WARPS + 1];
    __shared__ int32_t s_chunk_hit_prefix[CHUNKS_PER_WARP * NUM_WARPS + 1];
    
    // Initialize
    if (tid == 0) {
        s_total_misses = 0;
        s_total_hits = 0;
    }
    for (int i = tid; i < BITMAP_WORDS_TOPK; i += BLOCK_SIZE) {
        s_miss_bitmap[i] = 0;
    }
    for (int i = tid; i < NUM_TOP_K; i += BLOCK_SIZE) {
        s_hit_slots[i] = -1;
    }
    for (int i = tid; i < CHUNKS_PER_WARP * NUM_WARPS; i += BLOCK_SIZE) {
        s_chunk_miss_count[i] = 0;
        s_chunk_hit_count[i] = 0;
    }
    __syncthreads();
    
    // ===== Phase 1: Hit Detection (each warp processes multiple chunks) =====
    for (int chunk = 0; chunk < CHUNKS_PER_WARP; chunk++) {
        // Token index for this thread in this chunk
        const int token_idx = warp_id * WARP_SIZE + chunk * BLOCK_SIZE + lane_id;
        const bool has_valid_token = token_idx < NUM_TOP_K;
        
        int32_t my_token = 0;
        bool is_hit = false;
        int32_t my_device_slot = -1;
        
        if (has_valid_token) {
            my_token = top_k_tokens[token_idx];
            uint16_t slot = token_to_slot[my_token];
            is_hit = (slot != NOT_PRESENT);
            
            if (is_hit) {
                my_device_slot = static_cast<int32_t>(slot);
                top_k_device_locs[token_idx] = device_buffer_locs[my_device_slot];
                s_hit_slots[token_idx] = my_device_slot;
            }
        }
        
        // Warp-level ballot
        const unsigned int hit_mask = __ballot_sync(0xFFFFFFFF, has_valid_token && is_hit);
        const unsigned int miss_mask = __ballot_sync(0xFFFFFFFF, has_valid_token && !is_hit);
        
        const int warp_hit_count = __popc(hit_mask);
        const int warp_miss_count = __popc(miss_mask);
        const int local_miss_offset = __popc(miss_mask & lanes_before);
        
        // Store local miss offset
        if (has_valid_token && !is_hit) {
            s_local_miss_offset[token_idx] = local_miss_offset;
        }
        
        // Warp leader stores chunk counts and miss bitmap
        if (lane_id == 0) {
            const int chunk_idx = chunk * NUM_WARPS + warp_id;
            s_chunk_miss_count[chunk_idx] = warp_miss_count;
            s_chunk_hit_count[chunk_idx] = warp_hit_count;
        }
        
        // Store miss bitmap (one word per warp-chunk)
        // Bitmap word index = token_idx / 32, but we need to handle chunk layout
        if (lane_id == 0) {
            const int bitmap_word = (warp_id * WARP_SIZE + chunk * BLOCK_SIZE) / 32;
            if (bitmap_word < BITMAP_WORDS_TOPK) {
                s_miss_bitmap[bitmap_word] = miss_mask;
            }
        }
    }
    __syncthreads();
    
    // ===== Phase 2: Prefix Sum over all chunks =====
    if (tid == 0) {
        s_chunk_miss_prefix[0] = 0;
        s_chunk_hit_prefix[0] = 0;
        
        const int total_chunks = CHUNKS_PER_WARP * NUM_WARPS;
        for (int c = 0; c < total_chunks; c++) {
            s_chunk_miss_prefix[c + 1] = s_chunk_miss_prefix[c] + s_chunk_miss_count[c];
            s_chunk_hit_prefix[c + 1] = s_chunk_hit_prefix[c] + s_chunk_hit_count[c];
        }
        
        s_total_misses = s_chunk_miss_prefix[total_chunks];
        s_total_hits = s_chunk_hit_prefix[total_chunks];
    }
    __syncthreads();
    
    // Early exit if all hits
    if (s_total_misses == 0) {
        return;
    }
    
    // ===== Phase 3: Collect hit slots into contiguous array =====
    // We already have s_hit_slots[token_idx] for hits, now compact them
    // Actually, we can build protected bitmap directly from s_hit_slots
    
    // ===== Phase 4: Build Protected Bitmap =====
    for (int i = tid; i < BITMAP_WORDS_BUFFER; i += BLOCK_SIZE) {
        s_protected_bitmap[i] = 0;
    }
    __syncthreads();
    
    // Each thread checks its hit slots and marks protected
    for (int chunk = 0; chunk < CHUNKS_PER_WARP; chunk++) {
        const int token_idx = warp_id * WARP_SIZE + chunk * BLOCK_SIZE + lane_id;
        if (token_idx < NUM_TOP_K) {
            const int slot = s_hit_slots[token_idx];
            if (slot >= 0) {
                const int word = slot / 32;
                const int bit = slot % 32;
                atomicOr(&s_protected_bitmap[word], 1u << bit);
            }
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
    for (int chunk = 0; chunk < CHUNKS_PER_WARP; chunk++) {
        const int token_idx = warp_id * WARP_SIZE + chunk * BLOCK_SIZE + lane_id;
        if (token_idx >= NUM_TOP_K) continue;
        
        // Check if this token is a miss
        const int bitmap_word = token_idx / 32;
        const int bitmap_bit = token_idx % 32;
        const bool is_miss = (s_miss_bitmap[bitmap_word] >> bitmap_bit) & 1;
        
        if (is_miss) {
            // Compute global miss index
            const int chunk_idx = chunk * NUM_WARPS + warp_id;
            const int global_miss_idx = s_chunk_miss_prefix[chunk_idx] + s_local_miss_offset[token_idx];
            
            if (global_miss_idx < s_total_misses) {
                const int evict_slot = s_evictable_slots[global_miss_idx];
                const int32_t my_token = top_k_tokens[token_idx];
                const int32_t old_token = device_buffer_tokens[evict_slot];
                
                // Clear old token
                if (old_token >= 0 && old_token < max_tokens) {
                    token_to_slot[old_token] = NOT_PRESENT;
                }
                
                // Set new token
                token_to_slot[my_token] = static_cast<uint16_t>(evict_slot);
                device_buffer_tokens[evict_slot] = my_token;
                top_k_device_locs[token_idx] = device_buffer_locs[evict_slot];
                
                // Copy data
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

void sparse_cache_manager_v10_cuda(
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
    
    // Select template based on sizes
    // Each configuration supports up to NUM_TOP_K tokens
    if (num_top_k <= 256 && hot_buffer_size <= 512) {
        sparse_cache_manager_v10_kernel<256, 256, 512><<<1, 256>>>(
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
    } else if (num_top_k <= 1024 && hot_buffer_size <= 2048) {
        sparse_cache_manager_v10_kernel<256, 1024, 2048><<<1, 256>>>(
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
    } else if (num_top_k <= 4096 && hot_buffer_size <= 4096) {
        // ~34KB shared memory
        sparse_cache_manager_v10_kernel<256, 4096, 4096><<<1, 256>>>(
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
        // For very large NUM_TOP_K, use 8192 max (~68KB shared memory)
        // Requires GPU with 96KB+ shared memory per block
        sparse_cache_manager_v10_kernel<256, 8192, 8192><<<1, 256>>>(
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
void sparse_cache_manager_v10_cuda(
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
            name='sparse_cache_v10',
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=['sparse_cache_manager_v10_cuda'],
            verbose=False,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
        )
    return _module


class SparseCacheManagerV10:
    """
    Sparse KV Cache Manager v10 - Single Kernel with Warp Chunking
    
    Each warp processes multiple chunks of 32 tokens in a loop:
      Warp w handles: tokens[w*32 + chunk*BLOCK_SIZE] for chunk = 0, 1, 2, ...
    
    Benefits over v9 (multi-kernel):
    - No kernel launch overhead
    - Better cache utilization (data stays in shared memory)
    - Simpler control flow
    
    Limitations:
    - Shared memory must fit all arrays
    - NUM_TOP_K limited by shared memory size (~8K for 96KB shared mem)
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
        
        self._module.sparse_cache_manager_v10_cuda(
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


def visualize_v10_chunking():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║        Sparse KV Cache Manager v10 - Single Kernel with Warp Chunking        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Warp Chunking Pattern (BLOCK_SIZE=256, 8 warps):                            ║
║  ────────────────────────────────────────────────                            ║
║                                                                              ║
║  Chunk 0:  Warp0→[0-31]   Warp1→[32-63]   ... Warp7→[224-255]               ║
║  Chunk 1:  Warp0→[256-287] Warp1→[288-319] ... Warp7→[480-511]              ║
║  Chunk 2:  Warp0→[512-543] Warp1→[544-575] ... Warp7→[736-767]              ║
║  ...                                                                         ║
║                                                                              ║
║  Each warp iterates: for (chunk = 0; chunk < CHUNKS_PER_WARP; chunk++)       ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Shared Memory Layout (NUM_TOP_K=4096, HOT_BUFFER_SIZE=4096):                ║
║  ─────────────────────────────────────────────────────────────               ║
║                                                                              ║
║    s_miss_bitmap[128]           = 512 bytes    (4096 / 32 words)            ║
║    s_hit_slots[4096]            = 16 KB                                      ║
║    s_local_miss_offset[4096]    = 16 KB                                      ║
║    s_chunk_miss_count[128]      = 512 bytes    (16 chunks × 8 warps)        ║
║    s_chunk_hit_count[128]       = 512 bytes                                  ║
║    s_chunk_miss_prefix[129]     = 516 bytes                                  ║
║    s_chunk_hit_prefix[129]      = 516 bytes                                  ║
║    s_protected_bitmap[128]      = 512 bytes                                  ║
║    s_evictable_slots[4096]      = 16 KB                                      ║
║    ─────────────────────────────────────────                                 ║
║    Total: ~51 KB (fits in 96KB shared memory on modern GPUs)                ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Comparison:                                                                 ║
║  ───────────                                                                 ║
║                                                                              ║
║            │ v8 (single)  │ v9 (multi-kernel) │ v10 (chunking)             ║
║  ──────────┼──────────────┼───────────────────┼────────────────             ║
║  NUM_TOP_K │ ≤256         │ Unlimited         │ ≤8192 (shared mem)         ║
║  Kernels   │ 1            │ 4                 │ 1                           ║
║  Overhead  │ Minimal      │ Kernel launches   │ Loop overhead               ║
║  Memory    │ Shared only  │ Global + Shared   │ Shared only                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    visualize_v10_chunking()
