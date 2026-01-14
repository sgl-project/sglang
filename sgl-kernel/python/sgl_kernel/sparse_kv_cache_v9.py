"""
Sparse KV Cache Manager v9 - Scalable Multi-Kernel for Large NUM_TOP_K

For NUM_TOP_K > 1024, we use multiple kernels:
  Kernel 1: Hit detection (parallel, one thread per token)
  Kernel 2: Prefix sum on miss counts (single block)
  Kernel 3: Build protected bitmap + find evictable slots (single block)
  Kernel 4: Assignment + copy (parallel, one thread per miss)

Data flow:
  top_k_tokens[K] → hit_flags[K], miss_flags[K], hit_slots[K]
                  → prefix_sum → miss_indices[num_misses]
                  → protected_bitmap → evictable_slots[num_misses]
                  → assignments + copies
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
 * Kernel 1: Hit Detection (parallel across all tokens)
 * 
 * Each thread handles one token from top_k_tokens.
 * Outputs:
 *   - is_hit[K]: 1 if hit, 0 if miss
 *   - hit_slots[K]: slot index if hit, -1 if miss
 *   - top_k_device_locs[K]: filled for hits
 *   - num_misses: atomic counter
 */
__global__ void hit_detection_kernel(
    const int32_t* __restrict__ top_k_tokens,
    const uint16_t* __restrict__ token_to_slot,
    const int64_t* __restrict__ device_buffer_locs,
    int32_t* __restrict__ is_hit,
    int32_t* __restrict__ hit_slots,
    int64_t* __restrict__ top_k_device_locs,
    int32_t* __restrict__ num_misses,
    int32_t num_top_k,
    int32_t max_tokens
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_top_k) return;
    
    const int32_t token = top_k_tokens[idx];
    const uint16_t slot = token_to_slot[token];
    
    if (slot != NOT_PRESENT) {
        // Hit
        is_hit[idx] = 1;
        hit_slots[idx] = static_cast<int32_t>(slot);
        top_k_device_locs[idx] = device_buffer_locs[slot];
    } else {
        // Miss
        is_hit[idx] = 0;
        hit_slots[idx] = -1;
        atomicAdd(num_misses, 1);
    }
}

/**
 * Kernel 2: Compute miss indices via prefix sum
 * 
 * For each miss, compute its global index among all misses.
 * Uses a simple sequential scan (could optimize with parallel prefix sum for very large K).
 */
__global__ void compute_miss_indices_kernel(
    const int32_t* __restrict__ is_hit,
    int32_t* __restrict__ miss_global_idx,
    int32_t num_top_k
) {
    // Simple sequential scan (single thread)
    // For very large K, could use parallel prefix sum
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int miss_count = 0;
        for (int i = 0; i < num_top_k; i++) {
            if (is_hit[i] == 0) {
                miss_global_idx[i] = miss_count;
                miss_count++;
            } else {
                miss_global_idx[i] = -1;  // Not a miss
            }
        }
    }
}

/**
 * Kernel 2b: Parallel prefix sum for miss indices (for large K)
 * 
 * Uses block-level parallel scan with global reduction.
 */
template <int BLOCK_SIZE>
__global__ void parallel_prefix_sum_kernel(
    const int32_t* __restrict__ is_hit,
    int32_t* __restrict__ miss_global_idx,
    int32_t* __restrict__ block_sums,
    int32_t num_top_k
) {
    __shared__ int32_t s_data[BLOCK_SIZE];
    __shared__ int32_t s_is_miss[BLOCK_SIZE];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * BLOCK_SIZE + tid;
    
    // Load: 1 if miss, 0 if hit
    int is_miss = (gid < num_top_k && is_hit[gid] == 0) ? 1 : 0;
    s_is_miss[tid] = is_miss;
    s_data[tid] = is_miss;
    __syncthreads();
    
    // Parallel prefix sum (Blelloch scan)
    // Up-sweep
    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < BLOCK_SIZE) {
            s_data[idx] += s_data[idx - stride];
        }
        __syncthreads();
    }
    
    // Store block sum and clear last element
    if (tid == BLOCK_SIZE - 1) {
        block_sums[blockIdx.x] = s_data[BLOCK_SIZE - 1];
        s_data[BLOCK_SIZE - 1] = 0;
    }
    __syncthreads();
    
    // Down-sweep
    for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < BLOCK_SIZE) {
            int temp = s_data[idx - stride];
            s_data[idx - stride] = s_data[idx];
            s_data[idx] += temp;
        }
        __syncthreads();
    }
    
    // Write output
    if (gid < num_top_k) {
        if (s_is_miss[tid]) {
            miss_global_idx[gid] = s_data[tid];
        } else {
            miss_global_idx[gid] = -1;
        }
    }
}

/**
 * Kernel 2c: Add block offsets to get global indices
 */
__global__ void add_block_offsets_kernel(
    int32_t* __restrict__ miss_global_idx,
    const int32_t* __restrict__ block_prefix,
    int32_t num_top_k,
    int32_t block_size
) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_top_k) return;
    
    const int block_id = gid / block_size;
    if (miss_global_idx[gid] >= 0) {
        miss_global_idx[gid] += block_prefix[block_id];
    }
}

/**
 * Kernel 3: Build protected bitmap and find evictable slots
 * 
 * Single block kernel that:
 * 1. Builds protected bitmap from hit_slots
 * 2. Scans for evictable slots
 */
template <int BLOCK_SIZE>
__global__ void find_evictable_kernel(
    const int32_t* __restrict__ hit_slots,
    const int32_t* __restrict__ is_hit,
    int32_t* __restrict__ evictable_slots,
    int32_t num_top_k,
    int32_t hot_buffer_size,
    int32_t num_misses
) {
    // Protected bitmap in shared memory (supports up to 32K slots)
    constexpr int MAX_BITMAP_WORDS = 1024;  // 32K slots max
    __shared__ uint32_t s_protected[MAX_BITMAP_WORDS];
    __shared__ int32_t s_found;
    
    const int tid = threadIdx.x;
    const int num_words = (hot_buffer_size + 31) / 32;
    
    // Initialize
    for (int i = tid; i < num_words; i += BLOCK_SIZE) {
        s_protected[i] = 0;
    }
    if (tid == 0) s_found = 0;
    __syncthreads();
    
    // Build protected bitmap from hit slots
    for (int i = tid; i < num_top_k; i += BLOCK_SIZE) {
        if (is_hit[i]) {
            int slot = hit_slots[i];
            if (slot >= 0 && slot < hot_buffer_size) {
                atomicOr(&s_protected[slot / 32], 1u << (slot % 32));
            }
        }
    }
    __syncthreads();
    
    // Find evictable slots (first warp does the scan)
    if (tid < WARP_SIZE) {
        const int lane_id = tid;
        const unsigned int lanes_before = (1u << lane_id) - 1;
        int found = 0;
        int base_slot = 0;
        
        while (found < num_misses && base_slot < hot_buffer_size) {
            int my_slot = base_slot + lane_id;
            bool is_evictable = false;
            
            if (my_slot < hot_buffer_size) {
                uint32_t word = s_protected[my_slot / 32];
                is_evictable = !((word >> (my_slot % 32)) & 1);
            }
            
            unsigned int evict_mask = __ballot_sync(0xFFFFFFFF, is_evictable);
            int num_evictable = __popc(evict_mask);
            int need = num_misses - found;
            int use_count = min(num_evictable, need);
            
            if (is_evictable) {
                int my_pos = __popc(evict_mask & lanes_before);
                if (my_pos < use_count) {
                    evictable_slots[found + my_pos] = my_slot;
                }
            }
            
            found += use_count;
            base_slot += WARP_SIZE;
        }
    }
}

/**
 * Kernel 4: Assignment and copy (parallel across misses)
 * 
 * Each thread handles one miss token.
 */
__global__ void assignment_kernel(
    const int32_t* __restrict__ top_k_tokens,
    const int32_t* __restrict__ is_hit,
    const int32_t* __restrict__ miss_global_idx,
    const int32_t* __restrict__ evictable_slots,
    int32_t* __restrict__ device_buffer_tokens,
    uint16_t* __restrict__ token_to_slot,
    const int64_t* __restrict__ host_cache_locs,
    const int64_t* __restrict__ device_buffer_locs,
    const char* __restrict__ host_cache,
    char* __restrict__ device_buffer,
    int64_t* __restrict__ top_k_device_locs,
    int32_t num_top_k,
    int32_t max_tokens,
    int64_t item_size_bytes
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_top_k) return;
    if (is_hit[idx]) return;  // Only process misses
    
    const int32_t token = top_k_tokens[idx];
    const int32_t miss_idx = miss_global_idx[idx];
    const int32_t evict_slot = evictable_slots[miss_idx];
    const int32_t old_token = device_buffer_tokens[evict_slot];
    
    // Clear old token
    if (old_token >= 0 && old_token < max_tokens) {
        token_to_slot[old_token] = NOT_PRESENT;
    }
    
    // Set new token
    token_to_slot[token] = static_cast<uint16_t>(evict_slot);
    device_buffer_tokens[evict_slot] = token;
    top_k_device_locs[idx] = device_buffer_locs[evict_slot];
    
    // Copy data
    const int64_t host_loc = host_cache_locs[token];
    const int64_t device_loc = device_buffer_locs[evict_slot];
    const char* src = host_cache + host_loc * item_size_bytes;
    char* dst = device_buffer + device_loc * item_size_bytes;
    
    for (int64_t b = 0; b < item_size_bytes; b++) {
        dst[b] = src[b];
    }
}

/**
 * Main launcher that orchestrates all kernels
 */
void sparse_cache_manager_v9_cuda(
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
    
    auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(top_k_tokens.device());
    
    // Allocate intermediate buffers
    auto is_hit = torch::zeros({num_top_k}, options_int32);
    auto hit_slots = torch::full({num_top_k}, -1, options_int32);
    auto miss_global_idx = torch::zeros({num_top_k}, options_int32);
    auto num_misses_tensor = torch::zeros({1}, options_int32);
    
    constexpr int BLOCK_SIZE = 256;
    const int num_blocks_topk = (num_top_k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Kernel 1: Hit detection
    hit_detection_kernel<<<num_blocks_topk, BLOCK_SIZE>>>(
        top_k_tokens.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(token_to_slot.data_ptr<int16_t>()),
        device_buffer_locs.data_ptr<int64_t>(),
        is_hit.data_ptr<int32_t>(),
        hit_slots.data_ptr<int32_t>(),
        top_k_device_locs.data_ptr<int64_t>(),
        num_misses_tensor.data_ptr<int32_t>(),
        num_top_k,
        max_tokens
    );
    
    // Get num_misses
    int32_t num_misses = num_misses_tensor.item<int32_t>();
    
    if (num_misses == 0) {
        return;  // All hits, nothing more to do
    }
    
    // Kernel 2: Compute miss indices
    if (num_top_k <= 4096) {
        // Small enough for sequential scan
        compute_miss_indices_kernel<<<1, 1>>>(
            is_hit.data_ptr<int32_t>(),
            miss_global_idx.data_ptr<int32_t>(),
            num_top_k
        );
    } else {
        // Use parallel prefix sum for large K
        auto block_sums = torch::zeros({num_blocks_topk}, options_int32);
        auto block_prefix = torch::zeros({num_blocks_topk}, options_int32);
        
        parallel_prefix_sum_kernel<BLOCK_SIZE><<<num_blocks_topk, BLOCK_SIZE>>>(
            is_hit.data_ptr<int32_t>(),
            miss_global_idx.data_ptr<int32_t>(),
            block_sums.data_ptr<int32_t>(),
            num_top_k
        );
        
        // Compute block prefix sums (on CPU for simplicity, or another kernel for GPU)
        auto block_sums_cpu = block_sums.cpu();
        auto block_prefix_cpu = torch::zeros({num_blocks_topk}, torch::kInt32);
        int32_t running_sum = 0;
        for (int i = 0; i < num_blocks_topk; i++) {
            block_prefix_cpu[i] = running_sum;
            running_sum += block_sums_cpu[i].item<int32_t>();
        }
        block_prefix = block_prefix_cpu.to(top_k_tokens.device());
        
        add_block_offsets_kernel<<<num_blocks_topk, BLOCK_SIZE>>>(
            miss_global_idx.data_ptr<int32_t>(),
            block_prefix.data_ptr<int32_t>(),
            num_top_k,
            BLOCK_SIZE
        );
    }
    
    // Allocate evictable slots
    auto evictable_slots = torch::zeros({num_misses}, options_int32);
    
    // Kernel 3: Find evictable slots
    find_evictable_kernel<256><<<1, 256>>>(
        hit_slots.data_ptr<int32_t>(),
        is_hit.data_ptr<int32_t>(),
        evictable_slots.data_ptr<int32_t>(),
        num_top_k,
        hot_buffer_size,
        num_misses
    );
    
    // Kernel 4: Assignment and copy
    assignment_kernel<<<num_blocks_topk, BLOCK_SIZE>>>(
        top_k_tokens.data_ptr<int32_t>(),
        is_hit.data_ptr<int32_t>(),
        miss_global_idx.data_ptr<int32_t>(),
        evictable_slots.data_ptr<int32_t>(),
        device_buffer_tokens.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(token_to_slot.data_ptr<int16_t>()),
        host_cache_locs.data_ptr<int64_t>(),
        device_buffer_locs.data_ptr<int64_t>(),
        reinterpret_cast<const char*>(host_cache.data_ptr()),
        reinterpret_cast<char*>(device_buffer.data_ptr()),
        top_k_device_locs.data_ptr<int64_t>(),
        num_top_k,
        max_tokens,
        item_size_bytes
    );
}
'''

CPP_SOURCE = r'''
void sparse_cache_manager_v9_cuda(
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
            name='sparse_cache_v9',
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=['sparse_cache_manager_v9_cuda'],
            verbose=False,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
        )
    return _module


class SparseCacheManagerV9:
    """
    Sparse KV Cache Manager v9 - Scalable for Large NUM_TOP_K
    
    Uses multi-kernel approach:
    1. Hit detection (parallel, one thread per token)
    2. Prefix sum for miss indices
    3. Find evictable slots (single block)
    4. Assignment + copy (parallel, one thread per miss)
    
    Supports NUM_TOP_K up to millions of tokens.
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
        
        self._module.sparse_cache_manager_v9_cuda(
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


def visualize_v9_multi_kernel():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║       Sparse KV Cache Manager v9 - Scalable Multi-Kernel Approach            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Why Multi-Kernel for Large NUM_TOP_K?                                       ║
║  ─────────────────────────────────────                                       ║
║                                                                              ║
║  Single-kernel limits:                                                       ║
║  • Shared memory: s_hit_slots[K], s_miss_offset[K] don't fit for K > 1024   ║
║  • Block size: Max 1024 threads per block                                    ║
║  • Global sync: Need all hits before finding evictable slots                 ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Multi-Kernel Pipeline:                                                      ║
║  ──────────────────────                                                      ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐     ║
║  │ Kernel 1: Hit Detection (parallel, K threads across many blocks)   │     ║
║  │   • Each thread: check token_to_slot[token]                        │     ║
║  │   • Output: is_hit[K], hit_slots[K], atomicAdd(&num_misses)        │     ║
║  └─────────────────────────────────────────────────────────────────────┘     ║
║                              ↓                                               ║
║  ┌─────────────────────────────────────────────────────────────────────┐     ║
║  │ Kernel 2: Prefix Sum (compute miss indices)                         │     ║
║  │   • Small K (≤4096): sequential scan                                │     ║
║  │   • Large K: parallel Blelloch scan + block reduction               │     ║
║  │   • Output: miss_global_idx[K]                                      │     ║
║  └─────────────────────────────────────────────────────────────────────┘     ║
║                              ↓                                               ║
║  ┌─────────────────────────────────────────────────────────────────────┐     ║
║  │ Kernel 3: Find Evictable Slots (single block)                       │     ║
║  │   • Build protected bitmap from hit_slots                           │     ║
║  │   • Warp-parallel scan for unprotected slots                        │     ║
║  │   • Output: evictable_slots[num_misses]                             │     ║
║  └─────────────────────────────────────────────────────────────────────┘     ║
║                              ↓                                               ║
║  ┌─────────────────────────────────────────────────────────────────────┐     ║
║  │ Kernel 4: Assignment + Copy (parallel, K threads)                   │     ║
║  │   • Each miss thread: get evictable_slots[miss_global_idx[i]]      │     ║
║  │   • Update token_to_slot, device_buffer_tokens                      │     ║
║  │   • Copy data from host to device                                   │     ║
║  └─────────────────────────────────────────────────────────────────────┘     ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Scalability:                                                                ║
║  ────────────                                                                ║
║                                                                              ║
║  NUM_TOP_K    v8 (single kernel)    v9 (multi-kernel)                       ║
║  ─────────    ─────────────────     ────────────────────                     ║
║  256          ✓ (optimal)           ✓ (slight overhead)                      ║
║  1024         ✓ (shared mem limit)  ✓                                        ║
║  4096         ✗                     ✓                                        ║
║  16384        ✗                     ✓                                        ║
║  1M+          ✗                     ✓                                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    visualize_v9_multi_kernel()
