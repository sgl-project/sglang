"""
Test script for load_cache kernel WITHOUT token_residence_mapping.

Instead of O(1) lookup via token_residence_mapping, we do runtime hit checking
by scanning device_buffer_tokens using warp-parallel search.

Trade-off:
- Saves: num_requests × max_tokens × 2 bytes (can be 100s of MB)
- Costs: O(HOT_BUFFER_SIZE / 32) scan per token instead of O(1) lookup
"""

import pytest
import torch
from torch.utils.cpp_extension import load_inline

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

CUDA_SOURCE = r'''
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

constexpr int WARP_SIZE = 32;

__device__ __forceinline__ void
transfer_item_warp(int32_t lane_id, const void* src_addr, void* dst_addr, int64_t item_size_bytes) {
    const uint64_t* __restrict__ src = static_cast<const uint64_t*>(src_addr);
    uint64_t* __restrict__ dst = static_cast<uint64_t*>(dst_addr);
    const int total_chunks = item_size_bytes / sizeof(uint64_t);

    #pragma unroll
    for (int j = lane_id; j < total_chunks; j += WARP_SIZE) {
        dst[j] = src[j];
    }
}

/**
 * Warp-parallel search for a token in the buffer.
 * Returns slot index if found, -1 otherwise.
 * 
 * All lanes must participate (convergent).
 */
__device__ __forceinline__ int32_t warp_find_token(
    int32_t lane_id,
    int32_t target_token,
    const int32_t* __restrict__ s_buffer_tokens,
    int32_t buffer_size
) {
    const unsigned int lanes_before = (1u << lane_id) - 1;
    
    for (int base = 0; base < buffer_size; base += WARP_SIZE) {
        int slot = base + lane_id;
        bool match = (slot < buffer_size) && (s_buffer_tokens[slot] == target_token);
        unsigned int match_mask = __ballot_sync(0xFFFFFFFF, match);
        
        if (match_mask != 0) {
            // Found! Return first matching slot
            return base + (__ffs(match_mask) - 1);
        }
    }
    return -1;  // Not found
}

/**
 * Kernel without token_residence_mapping.
 * 
 * Hit checking is done by scanning device_buffer_tokens at runtime.
 * This saves memory but costs O(H/32) scan per token.
 * 
 * Data structures:
 *   device_buffer_tokens[H]: which token is at each slot (-1 = empty)
 *   (NO token_residence_mapping!)
 */
template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE>
__global__ void load_cache_no_mapping_kernel(
    const int32_t* __restrict__ top_k_tokens,
    int32_t* __restrict__ device_buffer_tokens,
    const int64_t* __restrict__ host_cache_locs,
    const int64_t* __restrict__ device_buffer_locs,
    const void* __restrict__ host_cache,
    void* __restrict__ device_buffer,
    int64_t* __restrict__ top_k_device_locs,
    int64_t item_size_bytes
) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const unsigned int lanes_before = (1u << lane_id) - 1;
    
    // Load device_buffer_tokens into shared memory for fast scanning
    __shared__ int32_t s_buffer_tokens[HOT_BUFFER_SIZE];
    __shared__ bool s_protected[HOT_BUFFER_SIZE];
    __shared__ int32_t s_hit_slots[NUM_TOP_K];      // Slot for hits (-1 for misses)
    __shared__ int32_t s_miss_indices[NUM_TOP_K];   // Indices of miss tokens in top_k
    __shared__ int32_t s_evictable_slots[NUM_TOP_K];
    __shared__ int32_t s_num_misses;
    
    // Initialize
    if (tid == 0) {
        s_num_misses = 0;
    }
    for (int i = tid; i < HOT_BUFFER_SIZE; i += BLOCK_SIZE) {
        s_buffer_tokens[i] = device_buffer_tokens[i];
        s_protected[i] = false;
    }
    for (int i = tid; i < NUM_TOP_K; i += BLOCK_SIZE) {
        s_hit_slots[i] = -1;
    }
    __syncthreads();
    
    // ===== Phase 1: Hit Detection via Warp-Parallel Scan =====
    // Each warp handles one token at a time
    for (int token_idx = warp_id; token_idx < NUM_TOP_K; token_idx += NUM_WARPS) {
        int32_t my_token = top_k_tokens[token_idx];
        
        // Warp-parallel search in s_buffer_tokens
        int32_t found_slot = warp_find_token(lane_id, my_token, s_buffer_tokens, HOT_BUFFER_SIZE);
        
        // Lane 0 processes the result
        if (lane_id == 0) {
            if (found_slot >= 0) {
                // Hit
                s_hit_slots[token_idx] = found_slot;
                s_protected[found_slot] = true;
                top_k_device_locs[token_idx] = device_buffer_locs[found_slot];
            } else {
                // Miss - record for later
                int miss_idx = atomicAdd(&s_num_misses, 1);
                s_miss_indices[miss_idx] = token_idx;
            }
        }
    }
    __syncthreads();
    
    // Early exit if all hits
    if (s_num_misses == 0) {
        return;
    }
    
    // ===== Phase 2: Find Evictable Slots =====
    if (warp_id == 0) {
        int found = 0;
        int base_slot = 0;
        
        while (found < s_num_misses && base_slot < HOT_BUFFER_SIZE) {
            int my_slot = base_slot + lane_id;
            bool is_evictable = (my_slot < HOT_BUFFER_SIZE) && !s_protected[my_slot];
            
            unsigned int evict_mask = __ballot_sync(0xFFFFFFFF, is_evictable);
            int num_evictable = __popc(evict_mask);
            int need = s_num_misses - found;
            int use_count = min(num_evictable, need);
            
            if (is_evictable) {
                int my_pos = __popc(evict_mask & lanes_before);
                if (my_pos < use_count) {
                    s_evictable_slots[found + my_pos] = my_slot;
                }
            }
            
            found += use_count;
            base_slot += WARP_SIZE;
        }
    }
    __syncthreads();
    
    // ===== Phase 3a: Metadata Update (all threads parallel) =====
    for (int i = tid; i < s_num_misses; i += BLOCK_SIZE) {
        int token_idx = s_miss_indices[i];
        int32_t miss_token = top_k_tokens[token_idx];
        int evict_slot = s_evictable_slots[i];
        
        // Update device_buffer_tokens (no token_residence_mapping to update!)
        device_buffer_tokens[evict_slot] = miss_token;
        s_buffer_tokens[evict_slot] = miss_token;  // Update shared copy too
        top_k_device_locs[token_idx] = device_buffer_locs[evict_slot];
    }
    __syncthreads();
    
    // ===== Phase 3b: Data Loading (warp-cooperative) =====
    for (int i = warp_id; i < s_num_misses; i += NUM_WARPS) {
        int token_idx = s_miss_indices[i];
        int32_t miss_token = top_k_tokens[token_idx];
        int evict_slot = s_evictable_slots[i];
        
        int64_t src_offset = host_cache_locs[miss_token] * item_size_bytes;
        int64_t dst_offset = device_buffer_locs[evict_slot] * item_size_bytes;
        
        transfer_item_warp(
            lane_id,
            static_cast<const char*>(host_cache) + src_offset,
            static_cast<char*>(device_buffer) + dst_offset,
            item_size_bytes
        );
    }
}

void load_cache_no_mapping_launcher(
    torch::Tensor top_k_tokens,
    torch::Tensor device_buffer_tokens,
    torch::Tensor host_cache_locs,
    torch::Tensor device_buffer_locs,
    torch::Tensor host_cache,
    torch::Tensor device_buffer,
    torch::Tensor top_k_device_locs,
    int64_t item_size_bytes,
    int64_t num_top_k,
    int64_t hot_buffer_size
) {
    if (num_top_k <= 64 && hot_buffer_size <= 128) {
        load_cache_no_mapping_kernel<256, 64, 128><<<1, 256>>>(
            top_k_tokens.data_ptr<int32_t>(),
            device_buffer_tokens.data_ptr<int32_t>(),
            host_cache_locs.data_ptr<int64_t>(),
            device_buffer_locs.data_ptr<int64_t>(),
            host_cache.data_ptr<uint8_t>(),
            device_buffer.data_ptr<uint8_t>(),
            top_k_device_locs.data_ptr<int64_t>(),
            item_size_bytes
        );
    } else {
        load_cache_no_mapping_kernel<256, 256, 512><<<1, 256>>>(
            top_k_tokens.data_ptr<int32_t>(),
            device_buffer_tokens.data_ptr<int32_t>(),
            host_cache_locs.data_ptr<int64_t>(),
            device_buffer_locs.data_ptr<int64_t>(),
            host_cache.data_ptr<uint8_t>(),
            device_buffer.data_ptr<uint8_t>(),
            top_k_device_locs.data_ptr<int64_t>(),
            item_size_bytes
        );
    }
}
'''

CPP_SOURCE = r'''
void load_cache_no_mapping_launcher(
    torch::Tensor top_k_tokens,
    torch::Tensor device_buffer_tokens,
    torch::Tensor host_cache_locs,
    torch::Tensor device_buffer_locs,
    torch::Tensor host_cache,
    torch::Tensor device_buffer,
    torch::Tensor top_k_device_locs,
    int64_t item_size_bytes,
    int64_t num_top_k,
    int64_t hot_buffer_size
);
'''

_module = None

def get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name='test_load_cache_no_mapping',
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=['load_cache_no_mapping_launcher'],
            verbose=False,
            extra_cuda_cflags=['-O3'],
        )
    return _module


class CacheManagerNoMapping:
    """Cache manager without token_residence_mapping (saves memory)."""
    
    def __init__(
        self,
        max_tokens: int,
        hot_buffer_size: int,
        item_size_bytes: int,
        num_top_k: int,
        device: str = "cuda",
    ):
        self.max_tokens = max_tokens
        self.hot_buffer_size = hot_buffer_size
        self.item_size_bytes = item_size_bytes
        self.num_top_k = num_top_k
        self.device = device
        
        # Only need device_buffer_tokens (NO token_residence_mapping!)
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
            self._module = get_module()
    
    def reset(self):
        self.device_buffer_tokens.fill_(-1)
    
    def process_topk(
        self,
        top_k_tokens: torch.Tensor,
        host_cache: torch.Tensor,
        device_buffer: torch.Tensor,
    ) -> torch.Tensor:
        self._ensure_compiled()
        
        if top_k_tokens.dtype != torch.int32:
            top_k_tokens = top_k_tokens.to(torch.int32)
        
        top_k_device_locs = torch.zeros(
            self.num_top_k, dtype=torch.int64, device=self.device
        )
        
        self._module.load_cache_no_mapping_launcher(
            top_k_tokens,
            self.device_buffer_tokens,
            self.host_cache_locs,
            self.device_buffer_locs,
            host_cache,
            device_buffer,
            top_k_device_locs,
            self.item_size_bytes,
            self.num_top_k,
            self.hot_buffer_size,
        )
        
        return top_k_device_locs
    
    def is_resident(self, token: int) -> bool:
        """Check if token is resident (requires scan)."""
        return (self.device_buffer_tokens == token).any().item()
    
    def get_memory_saved(self, num_requests: int = 1) -> int:
        """Calculate memory saved by not having token_residence_mapping."""
        # Would need: num_requests × max_tokens × 2 bytes (int16)
        return num_requests * self.max_tokens * 2


class PythonReferenceNoMapping:
    """Python reference without token_residence_mapping."""
    
    def __init__(self, max_tokens: int, hot_buffer_size: int, item_size_bytes: int):
        self.max_tokens = max_tokens
        self.hot_buffer_size = hot_buffer_size
        self.item_size_bytes = item_size_bytes
        
        self.device_buffer_tokens = [-1] * hot_buffer_size
        self.device_buffer_locs = list(range(hot_buffer_size))
        self.host_cache_locs = list(range(max_tokens))
    
    def reset(self):
        self.device_buffer_tokens = [-1] * self.hot_buffer_size
    
    def find_token(self, token: int) -> int:
        """Find slot containing token, or -1 if not found."""
        for slot, t in enumerate(self.device_buffer_tokens):
            if t == token:
                return slot
        return -1
    
    def process_topk(self, top_k_tokens: list, host_cache: bytes, 
                    device_buffer: bytearray) -> list:
        top_k_device_locs = [0] * len(top_k_tokens)
        
        # Phase 1: Hit detection via scan
        hits = []
        misses = []
        protected_slots = set()
        
        for i, token in enumerate(top_k_tokens):
            slot = self.find_token(token)
            if slot >= 0:
                hits.append((i, token, slot))
                top_k_device_locs[i] = self.device_buffer_locs[slot]
                protected_slots.add(slot)
            else:
                misses.append((i, token))
        
        # Phase 2: Find evictable slots
        evictable = []
        for slot in range(self.hot_buffer_size):
            if slot not in protected_slots:
                evictable.append(slot)
                if len(evictable) >= len(misses):
                    break
        
        # Phase 3: Assignment and copy
        for idx, (top_k_idx, token) in enumerate(misses):
            evict_slot = evictable[idx]
            
            self.device_buffer_tokens[evict_slot] = token
            top_k_device_locs[top_k_idx] = self.device_buffer_locs[evict_slot]
            
            src = self.host_cache_locs[token] * self.item_size_bytes
            dst = self.device_buffer_locs[evict_slot] * self.item_size_bytes
            device_buffer[dst:dst + self.item_size_bytes] = \
                host_cache[src:src + self.item_size_bytes]
        
        return top_k_device_locs


class TestLoadCacheNoMapping:
    """Tests for kernel without token_residence_mapping."""
    
    @pytest.fixture
    def config(self):
        return {
            "max_tokens": 1000,
            "hot_buffer_size": 128,
            "item_size_bytes": 64,
            "num_top_k": 64,
        }
    
    def create_test_data(self, config):
        max_tokens = config["max_tokens"]
        item_size = config["item_size_bytes"]
        
        host_cache_np = bytearray(max_tokens * item_size)
        for t in range(max_tokens):
            for i in range(item_size):
                host_cache_np[t * item_size + i] = (t + i) % 256
        
        host_cache = torch.tensor(list(host_cache_np), dtype=torch.uint8).pin_memory()
        device_buffer = torch.zeros(
            config["hot_buffer_size"] * item_size, dtype=torch.uint8, device="cuda"
        )
        
        return host_cache_np, host_cache, device_buffer
    
    def test_all_misses(self, config):
        """All tokens are cache misses."""
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        ref = PythonReferenceNoMapping(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManagerNoMapping(
            config["max_tokens"], config["hot_buffer_size"],
            config["item_size_bytes"], config["num_top_k"]
        )
        
        tokens = list(range(config["num_top_k"]))
        top_k = torch.tensor(tokens, dtype=torch.int32, device="cuda")
        
        device_ref = bytearray(config["hot_buffer_size"] * config["item_size_bytes"])
        ref_locs = ref.process_topk(tokens, host_np, device_ref)
        cuda_locs = cuda.process_topk(top_k, host_torch, device_torch)
        torch.cuda.synchronize()
        
        assert cuda_locs.cpu().tolist() == ref_locs
    
    def test_all_hits(self, config):
        """All tokens are cache hits."""
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        ref = PythonReferenceNoMapping(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManagerNoMapping(
            config["max_tokens"], config["hot_buffer_size"],
            config["item_size_bytes"], config["num_top_k"]
        )
        
        tokens = list(range(config["num_top_k"]))
        top_k = torch.tensor(tokens, dtype=torch.int32, device="cuda")
        device_ref = bytearray(config["hot_buffer_size"] * config["item_size_bytes"])
        
        # First pass: load
        ref.process_topk(tokens, host_np, device_ref)
        cuda.process_topk(top_k, host_torch, device_torch)
        torch.cuda.synchronize()
        
        # Second pass: all hits
        ref_locs = ref.process_topk(tokens, host_np, device_ref)
        cuda_locs = cuda.process_topk(top_k, host_torch, device_torch)
        torch.cuda.synchronize()
        
        assert cuda_locs.cpu().tolist() == ref_locs
    
    def test_mixed(self, config):
        """Mix of hits and misses."""
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        ref = PythonReferenceNoMapping(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManagerNoMapping(
            config["max_tokens"], config["hot_buffer_size"],
            config["item_size_bytes"], config["num_top_k"]
        )
        
        device_ref = bytearray(config["hot_buffer_size"] * config["item_size_bytes"])
        
        # First pass
        tokens1 = list(range(config["num_top_k"]))
        top_k1 = torch.tensor(tokens1, dtype=torch.int32, device="cuda")
        ref.process_topk(tokens1, host_np, device_ref)
        cuda.process_topk(top_k1, host_torch, device_torch)
        torch.cuda.synchronize()
        
        # Second pass: half hits, half misses
        half = config["num_top_k"] // 2
        tokens2 = list(range(half)) + list(range(100, 100 + half))
        top_k2 = torch.tensor(tokens2, dtype=torch.int32, device="cuda")
        
        ref_locs = ref.process_topk(tokens2, host_np, device_ref)
        cuda_locs = cuda.process_topk(top_k2, host_torch, device_torch)
        torch.cuda.synchronize()
        
        assert cuda_locs.cpu().tolist() == ref_locs
    
    def test_memory_savings(self, config):
        """Verify memory savings calculation."""
        cuda = CacheManagerNoMapping(
            config["max_tokens"], config["hot_buffer_size"],
            config["item_size_bytes"], config["num_top_k"]
        )
        
        # Single request
        saved_single = cuda.get_memory_saved(num_requests=1)
        assert saved_single == config["max_tokens"] * 2  # 1000 * 2 = 2KB
        
        # 1000 requests
        saved_batch = cuda.get_memory_saved(num_requests=1000)
        assert saved_batch == 1000 * config["max_tokens"] * 2  # 2MB
    
    def test_data_integrity(self, config):
        """Verify copied data is correct."""
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        cuda = CacheManagerNoMapping(
            config["max_tokens"], config["hot_buffer_size"],
            config["item_size_bytes"], config["num_top_k"]
        )
        
        tokens = list(range(config["num_top_k"]))
        top_k = torch.tensor(tokens, dtype=torch.int32, device="cuda")
        
        cuda_locs = cuda.process_topk(top_k, host_torch, device_torch)
        torch.cuda.synchronize()
        
        device_cpu = device_torch.cpu().numpy()
        locs = cuda_locs.cpu().tolist()
        
        for i, token in enumerate(tokens):
            loc = locs[i]
            offset = loc * config["item_size_bytes"]
            
            expected = [(token + j) % 256 for j in range(config["item_size_bytes"])]
            actual = list(device_cpu[offset:offset + config["item_size_bytes"]])
            
            assert actual == expected, f"Token {token} data mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
