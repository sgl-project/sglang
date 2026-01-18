"""
Test script for load_cache kernel with LINEAR SCAN hit detection.

No hash table - just load top_k into shared memory and scan.
Uses broadcast reads (all lanes read same address) - NO bank conflicts.

Trade-off vs hash:
- Simpler code, no hash table overhead
- O(H × K) shared memory reads vs O(H) for hash
- Better for small K, worse for large K
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

    for (int j = lane_id; j < total_chunks; j += WARP_SIZE) {
        dst[j] = src[j];
    }
}

/**
 * Linear scan hit detection kernel.
 * 
 * Algorithm:
 * 1. Load top_k_tokens into shared memory
 * 2. Each warp processes WARP_SIZE buffer slots at a time
 * 3. For each slot, broadcast-scan through top_k to find match
 * 4. Collect hits and evictable slots
 * 
 * Bank conflicts: NONE (broadcast reads)
 * Complexity: O(H × K) shared memory reads
 */
template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE>
__global__ void load_cache_linear_kernel(
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
    
    // Load top_k into shared memory
    __shared__ int32_t s_top_k[NUM_TOP_K];
    __shared__ int32_t s_hit_slots[NUM_TOP_K];    // buffer slot for each top_k (-1 = miss)
    __shared__ int32_t s_evictable_slots[NUM_TOP_K];
    __shared__ int32_t s_miss_indices[NUM_TOP_K];
    __shared__ int32_t s_num_evictable;
    __shared__ int32_t s_num_misses;
    
    // ===== Phase 1: Initialize =====
    for (int i = tid; i < NUM_TOP_K; i += BLOCK_SIZE) {
        s_top_k[i] = top_k_tokens[i];
        s_hit_slots[i] = -1;
    }
    if (tid == 0) {
        s_num_evictable = 0;
        s_num_misses = 0;
    }
    __syncthreads();
    
    // ===== Phase 2: Scan buffer, find hits using BROADCAST reads =====
    // Each warp processes WARP_SIZE buffer slots simultaneously
    // For each slot, ALL lanes scan through top_k together (broadcast)
    
    for (int base_slot = warp_id * WARP_SIZE; base_slot < HOT_BUFFER_SIZE; base_slot += NUM_WARPS * WARP_SIZE) {
        int slot = base_slot + lane_id;
        bool valid_slot = (slot < HOT_BUFFER_SIZE);
        
        // Each lane loads its buffer token
        int32_t my_token = valid_slot ? device_buffer_tokens[slot] : -1;
        
        // Search for my_token in top_k using BROADCAST
        // All lanes read the same s_top_k[i] - NO bank conflicts!
        int found_top_k_idx = -1;
        
        #pragma unroll 4
        for (int i = 0; i < NUM_TOP_K; i++) {
            int32_t top_k_token = s_top_k[i];  // BROADCAST: all lanes read same address
            if (top_k_token == my_token && my_token >= 0) {
                found_top_k_idx = i;
                break;  // Found match
            }
        }
        
        // Record hits (each lane may have found a different top_k entry)
        if (found_top_k_idx >= 0) {
            s_hit_slots[found_top_k_idx] = slot;
        }
        
        // Collect evictable slots using warp-cooperative pattern
        bool is_evictable = valid_slot && (found_top_k_idx < 0);
        unsigned int evict_mask = __ballot_sync(0xFFFFFFFF, is_evictable);
        
        if (evict_mask != 0) {
            int evict_count = __popc(evict_mask);
            int my_prefix = __popc(evict_mask & ((1u << lane_id) - 1));
            
            int warp_offset;
            if (lane_id == 0) {
                warp_offset = atomicAdd(&s_num_evictable, evict_count);
            }
            warp_offset = __shfl_sync(0xFFFFFFFF, warp_offset, 0);
            
            if (is_evictable && warp_offset + my_prefix < NUM_TOP_K) {
                s_evictable_slots[warp_offset + my_prefix] = slot;
            }
        }
    }
    __syncthreads();
    
    // ===== Phase 3: Identify misses and set hit outputs =====
    for (int base_i = warp_id * WARP_SIZE; base_i < NUM_TOP_K; base_i += NUM_WARPS * WARP_SIZE) {
        int i = base_i + lane_id;
        bool valid = (i < NUM_TOP_K);
        int slot = valid ? s_hit_slots[i] : 0;
        bool is_hit = valid && (slot >= 0);
        bool is_miss = valid && !is_hit;
        
        if (is_hit) {
            top_k_device_locs[i] = device_buffer_locs[slot];
        }
        
        unsigned int miss_mask = __ballot_sync(0xFFFFFFFF, is_miss);
        
        if (miss_mask != 0) {
            int miss_count = __popc(miss_mask);
            int my_prefix = __popc(miss_mask & ((1u << lane_id) - 1));
            
            int warp_offset;
            if (lane_id == 0) {
                warp_offset = atomicAdd(&s_num_misses, miss_count);
            }
            warp_offset = __shfl_sync(0xFFFFFFFF, warp_offset, 0);
            
            if (is_miss) {
                s_miss_indices[warp_offset + my_prefix] = i;
            }
        }
    }
    __syncthreads();
    
    if (s_num_misses == 0) return;
    
    // ===== Phase 4: Metadata update for misses =====
    for (int i = tid; i < s_num_misses; i += BLOCK_SIZE) {
        int top_k_idx = s_miss_indices[i];
        int32_t token = s_top_k[top_k_idx];
        int evict_slot = s_evictable_slots[i];
        
        device_buffer_tokens[evict_slot] = token;
        top_k_device_locs[top_k_idx] = device_buffer_locs[evict_slot];
    }
    __syncthreads();
    
    // ===== Phase 5: Data loading (warp-cooperative) =====
    for (int i = warp_id; i < s_num_misses; i += NUM_WARPS) {
        int top_k_idx = s_miss_indices[i];
        int32_t token = s_top_k[top_k_idx];
        int evict_slot = s_evictable_slots[i];
        
        int64_t src_offset = host_cache_locs[token] * item_size_bytes;
        int64_t dst_offset = device_buffer_locs[evict_slot] * item_size_bytes;
        
        transfer_item_warp(
            lane_id,
            static_cast<const char*>(host_cache) + src_offset,
            static_cast<char*>(device_buffer) + dst_offset,
            item_size_bytes
        );
    }
}

void load_cache_linear_launcher(
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
        load_cache_linear_kernel<256, 64, 128><<<1, 256>>>(
            top_k_tokens.data_ptr<int32_t>(),
            device_buffer_tokens.data_ptr<int32_t>(),
            host_cache_locs.data_ptr<int64_t>(),
            device_buffer_locs.data_ptr<int64_t>(),
            host_cache.data_ptr<uint8_t>(),
            device_buffer.data_ptr<uint8_t>(),
            top_k_device_locs.data_ptr<int64_t>(),
            item_size_bytes
        );
    } else if (num_top_k <= 256 && hot_buffer_size <= 512) {
        load_cache_linear_kernel<256, 256, 512><<<1, 256>>>(
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
        load_cache_linear_kernel<256, 512, 2048><<<1, 256>>>(
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
void load_cache_linear_launcher(
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
            name='test_load_cache_linear',
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=['load_cache_linear_launcher'],
            verbose=False,
            extra_cuda_cflags=['-O3'],
        )
    return _module


class CacheManagerLinear:
    """Cache manager with linear scan hit detection."""
    
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
        
        self._module.load_cache_linear_launcher(
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


class PythonReferenceLinear:
    """Python reference implementation."""
    
    def __init__(self, max_tokens: int, hot_buffer_size: int, item_size_bytes: int):
        self.max_tokens = max_tokens
        self.hot_buffer_size = hot_buffer_size
        self.item_size_bytes = item_size_bytes
        
        self.device_buffer_tokens = [-1] * hot_buffer_size
        self.device_buffer_locs = list(range(hot_buffer_size))
        self.host_cache_locs = list(range(max_tokens))
    
    def reset(self):
        self.device_buffer_tokens = [-1] * self.hot_buffer_size
    
    def process_topk(self, top_k_tokens: list, host_cache: bytes, 
                    device_buffer: bytearray) -> list:
        top_k_device_locs = [0] * len(top_k_tokens)
        top_k_set = set(top_k_tokens)
        
        hit_slots = [-1] * len(top_k_tokens)
        evictable = []
        
        for slot, token in enumerate(self.device_buffer_tokens):
            if token in top_k_set:
                top_k_idx = top_k_tokens.index(token)
                hit_slots[top_k_idx] = slot
            else:
                evictable.append(slot)
        
        misses = []
        for i, slot in enumerate(hit_slots):
            if slot >= 0:
                top_k_device_locs[i] = self.device_buffer_locs[slot]
            else:
                misses.append(i)
        
        for miss_idx, top_k_idx in enumerate(misses):
            token = top_k_tokens[top_k_idx]
            evict_slot = evictable[miss_idx]
            
            self.device_buffer_tokens[evict_slot] = token
            top_k_device_locs[top_k_idx] = self.device_buffer_locs[evict_slot]
            
            src = self.host_cache_locs[token] * self.item_size_bytes
            dst = self.device_buffer_locs[evict_slot] * self.item_size_bytes
            device_buffer[dst:dst + self.item_size_bytes] = \
                host_cache[src:src + self.item_size_bytes]
        
        return top_k_device_locs


class TestLoadCacheLinear:
    """Tests for linear scan kernel."""
    
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
        
        ref = PythonReferenceLinear(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManagerLinear(
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
        
        ref = PythonReferenceLinear(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManagerLinear(
            config["max_tokens"], config["hot_buffer_size"],
            config["item_size_bytes"], config["num_top_k"]
        )
        
        tokens = list(range(config["num_top_k"]))
        top_k = torch.tensor(tokens, dtype=torch.int32, device="cuda")
        device_ref = bytearray(config["hot_buffer_size"] * config["item_size_bytes"])
        
        ref.process_topk(tokens, host_np, device_ref)
        cuda.process_topk(top_k, host_torch, device_torch)
        torch.cuda.synchronize()
        
        ref_locs = ref.process_topk(tokens, host_np, device_ref)
        cuda_locs = cuda.process_topk(top_k, host_torch, device_torch)
        torch.cuda.synchronize()
        
        assert cuda_locs.cpu().tolist() == ref_locs
    
    def test_mixed(self, config):
        """Mix of hits and misses."""
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        ref = PythonReferenceLinear(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManagerLinear(
            config["max_tokens"], config["hot_buffer_size"],
            config["item_size_bytes"], config["num_top_k"]
        )
        
        device_ref = bytearray(config["hot_buffer_size"] * config["item_size_bytes"])
        
        tokens1 = list(range(config["num_top_k"]))
        top_k1 = torch.tensor(tokens1, dtype=torch.int32, device="cuda")
        ref.process_topk(tokens1, host_np, device_ref)
        cuda.process_topk(top_k1, host_torch, device_torch)
        torch.cuda.synchronize()
        
        half = config["num_top_k"] // 2
        tokens2 = list(range(half)) + list(range(100, 100 + half))
        top_k2 = torch.tensor(tokens2, dtype=torch.int32, device="cuda")
        
        ref_locs = ref.process_topk(tokens2, host_np, device_ref)
        cuda_locs = cuda.process_topk(top_k2, host_torch, device_torch)
        torch.cuda.synchronize()
        
        assert cuda_locs.cpu().tolist() == ref_locs
    
    def test_data_integrity(self, config):
        """Verify copied data is correct."""
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        cuda = CacheManagerLinear(
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
    
    def test_complexity_comparison(self):
        """Compare theoretical complexity of linear vs hash."""
        configs = [
            {"K": 64, "H": 128},
            {"K": 256, "H": 512},
            {"K": 512, "H": 2048},
        ]
        
        print("\nComplexity comparison (shared memory reads):")
        print(f"{'K':>6} {'H':>6} {'Linear O(H×K)':>15} {'Hash O(H×2)':>15} {'Ratio':>10}")
        print("-" * 55)
        
        for cfg in configs:
            K, H = cfg["K"], cfg["H"]
            linear = H * K
            hash_ops = H * 2  # ~2 probes average with 50% load factor
            ratio = linear / hash_ops
            print(f"{K:>6} {H:>6} {linear:>15} {hash_ops:>15} {ratio:>10.1f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
