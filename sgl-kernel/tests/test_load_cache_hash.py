"""
Test script for load_cache kernel with hash-based hit detection.

Instead of scanning buffer for each top_k token O(K × H/32), we:
1. Build hash table from top_k_tokens: O(K)
2. Scan device_buffer_tokens once, lookup each in hash: O(H)
Total: O(K + H) instead of O(K × H/32)

This is faster when K is large relative to H/32.
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
 * Hash-based hit detection kernel with warp-cooperative hash building.
 * 
 * Algorithm:
 * 1. Build hash table from top_k_tokens using warp-cooperative insertion (NO atomics)
 * 2. Scan device_buffer_tokens once:
 *    - For each slot, lookup token in hash table
 *    - If found: it's a hit, record slot for that top_k_index
 *    - If not found: slot is evictable
 * 3. Collect misses (top_k entries with no hit)
 * 4. Assign evictable slots to misses
 * 5. Copy data
 * 
 * Complexity: O(K + H) instead of O(K × H/32)
 */
template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE>
__global__ void load_cache_hash_kernel(
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
    constexpr int HASH_SIZE = NUM_TOP_K * 2;  // 2x load factor for good performance
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Hash table: token -> top_k_index
    __shared__ int32_t s_hash_keys[HASH_SIZE];    // token (-1 = empty)
    __shared__ int16_t s_hash_vals[HASH_SIZE];    // top_k index
    
    // Results
    __shared__ int32_t s_hit_slots[NUM_TOP_K];    // slot for each top_k (-1 = miss)
    __shared__ int32_t s_evictable_slots[NUM_TOP_K];
    __shared__ int32_t s_miss_indices[NUM_TOP_K]; // which top_k indices are misses
    __shared__ int32_t s_num_misses;
    __shared__ int32_t s_num_evictable;
    
    // ===== Phase 1: Initialize =====
    if (tid == 0) {
        s_num_misses = 0;
        s_num_evictable = 0;
    }
    for (int i = tid; i < HASH_SIZE; i += BLOCK_SIZE) {
        s_hash_keys[i] = -1;
    }
    for (int i = tid; i < NUM_TOP_K; i += BLOCK_SIZE) {
        s_hit_slots[i] = -1;
    }
    __syncthreads();
    
    // ===== Phase 2: Build hash table using warp-cooperative insertion =====
    // Warp 0 builds the hash table - NO atomics needed
    // O(K) iterations, each with warp-parallel empty slot search
    if (warp_id == 0) {
        for (int i = 0; i < NUM_TOP_K; i++) {
            int32_t token = top_k_tokens[i];
            uint32_t base_hash = static_cast<uint32_t>(token) % HASH_SIZE;
            
            // Warp-parallel search for empty slot
            for (int probe = 0; probe < HASH_SIZE; probe += WARP_SIZE) {
                uint32_t slot = (base_hash + probe + lane_id) % HASH_SIZE;
                bool is_empty = (s_hash_keys[slot] == -1);
                unsigned int mask = __ballot_sync(0xFFFFFFFF, is_empty);
                
                if (mask != 0) {
                    // Found empty slot(s) - use first one
                    int first_lane = __ffs(mask) - 1;
                    uint32_t winner_slot = (base_hash + probe + first_lane) % HASH_SIZE;
                    if (lane_id == 0) {
                        s_hash_keys[winner_slot] = token;
                        s_hash_vals[winner_slot] = static_cast<int16_t>(i);
                    }
                    __syncwarp();
                    break;
                }
            }
        }
    }
    __syncthreads();
    
    // ===== Phase 3: Scan buffer with warp-cooperative collection =====
    // Each warp processes chunks of the buffer
    // Uses ballot + prefix sum to collect evictable slots (one atomicAdd per warp, not per slot)
    for (int base_slot = warp_id * WARP_SIZE; base_slot < HOT_BUFFER_SIZE; base_slot += NUM_WARPS * WARP_SIZE) {
        int slot = base_slot + lane_id;
        bool valid_slot = (slot < HOT_BUFFER_SIZE);
        int32_t token = valid_slot ? device_buffer_tokens[slot] : -1;
        bool is_hit = false;
        
        if (token >= 0 && valid_slot) {
            // Lookup token in hash table - O(1) average
            uint32_t hash = static_cast<uint32_t>(token) % HASH_SIZE;
            
            while (s_hash_keys[hash] != -1) {
                if (s_hash_keys[hash] == token) {
                    int top_k_idx = s_hash_vals[hash];
                    s_hit_slots[top_k_idx] = slot;
                    is_hit = true;
                    break;
                }
                hash = (hash + 1) % HASH_SIZE;
            }
        }
        
        // Warp-cooperative evictable slot collection
        bool is_evictable = valid_slot && !is_hit;
        unsigned int evict_mask = __ballot_sync(0xFFFFFFFF, is_evictable);
        
        if (evict_mask != 0) {
            int evict_count = __popc(evict_mask);
            int my_prefix = __popc(evict_mask & ((1u << lane_id) - 1));
            
            // One atomicAdd per warp (not per thread)
            int warp_offset;
            if (lane_id == 0) {
                warp_offset = atomicAdd(&s_num_evictable, evict_count);
            }
            warp_offset = __shfl_sync(0xFFFFFFFF, warp_offset, 0);
            
            // Write evictable slots using prefix sum
            if (is_evictable && warp_offset + my_prefix < NUM_TOP_K) {
                s_evictable_slots[warp_offset + my_prefix] = slot;
            }
        }
    }
    __syncthreads();
    
    // ===== Phase 4: Identify misses and set hit outputs =====
    // Warp-cooperative miss collection (one atomicAdd per warp)
    for (int base_i = warp_id * WARP_SIZE; base_i < NUM_TOP_K; base_i += NUM_WARPS * WARP_SIZE) {
        int i = base_i + lane_id;
        bool valid = (i < NUM_TOP_K);
        int slot = valid ? s_hit_slots[i] : 0;
        bool is_hit = valid && (slot >= 0);
        bool is_miss = valid && !is_hit;
        
        // Set output for hits
        if (is_hit) {
            top_k_device_locs[i] = device_buffer_locs[slot];
        }
        
        // Warp-cooperative miss collection
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
    
    // Early exit if all hits
    if (s_num_misses == 0) {
        return;
    }
    
    // ===== Phase 5: Metadata update for misses =====
    for (int i = tid; i < s_num_misses; i += BLOCK_SIZE) {
        int top_k_idx = s_miss_indices[i];
        int32_t token = top_k_tokens[top_k_idx];
        int evict_slot = s_evictable_slots[i];
        
        device_buffer_tokens[evict_slot] = token;
        top_k_device_locs[top_k_idx] = device_buffer_locs[evict_slot];
    }
    __syncthreads();
    
    // ===== Phase 6: Data loading (warp-cooperative) =====
    for (int i = warp_id; i < s_num_misses; i += NUM_WARPS) {
        int top_k_idx = s_miss_indices[i];
        int32_t token = top_k_tokens[top_k_idx];
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

void load_cache_hash_launcher(
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
        load_cache_hash_kernel<256, 64, 128><<<1, 256>>>(
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
        load_cache_hash_kernel<256, 256, 512><<<1, 256>>>(
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
        load_cache_hash_kernel<256, 512, 2048><<<1, 256>>>(
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
void load_cache_hash_launcher(
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
            name='test_load_cache_hash',
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=['load_cache_hash_launcher'],
            verbose=False,
            extra_cuda_cflags=['-O3'],
        )
    return _module


class CacheManagerHash:
    """Cache manager with hash-based hit detection."""
    
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
        
        self._module.load_cache_hash_launcher(
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


class PythonReferenceHash:
    """Python reference with hash-based approach."""
    
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
        
        # Build hash set from top_k_tokens (token -> top_k_index)
        top_k_set = {token: i for i, token in enumerate(top_k_tokens)}
        
        # Scan buffer, find hits and evictable slots
        hit_slots = [-1] * len(top_k_tokens)
        evictable = []
        
        for slot, token in enumerate(self.device_buffer_tokens):
            if token in top_k_set:
                # Hit
                top_k_idx = top_k_set[token]
                hit_slots[top_k_idx] = slot
            else:
                # Evictable
                evictable.append(slot)
        
        # Process hits
        misses = []
        for i, slot in enumerate(hit_slots):
            if slot >= 0:
                top_k_device_locs[i] = self.device_buffer_locs[slot]
            else:
                misses.append(i)
        
        # Assign evictable slots to misses
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


class TestLoadCacheHash:
    """Tests for hash-based kernel."""
    
    @pytest.fixture
    def config(self):
        return {
            "max_tokens": 1000,
            "hot_buffer_size": 128,
            "item_size_bytes": 64,
            "num_top_k": 64,
        }
    
    @pytest.fixture
    def large_config(self):
        """Larger config where hash approach shines."""
        return {
            "max_tokens": 10000,
            "hot_buffer_size": 512,
            "item_size_bytes": 64,
            "num_top_k": 256,
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
        
        ref = PythonReferenceHash(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManagerHash(
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
        
        ref = PythonReferenceHash(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManagerHash(
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
        
        ref = PythonReferenceHash(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManagerHash(
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
    
    def test_large_config(self, large_config):
        """Test with larger config where hash is more beneficial."""
        host_np, host_torch, device_torch = self.create_test_data(large_config)
        
        ref = PythonReferenceHash(
            large_config["max_tokens"], large_config["hot_buffer_size"], 
            large_config["item_size_bytes"]
        )
        cuda = CacheManagerHash(
            large_config["max_tokens"], large_config["hot_buffer_size"],
            large_config["item_size_bytes"], large_config["num_top_k"]
        )
        
        tokens = list(range(large_config["num_top_k"]))
        top_k = torch.tensor(tokens, dtype=torch.int32, device="cuda")
        device_ref = bytearray(large_config["hot_buffer_size"] * large_config["item_size_bytes"])
        
        ref_locs = ref.process_topk(tokens, host_np, device_ref)
        cuda_locs = cuda.process_topk(top_k, host_torch, device_torch)
        torch.cuda.synchronize()
        
        assert cuda_locs.cpu().tolist() == ref_locs
    
    def test_data_integrity(self, config):
        """Verify copied data is correct."""
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        cuda = CacheManagerHash(
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
    
    def test_complexity_comparison(self, large_config):
        """Compare theoretical complexity of scan vs hash."""
        K = large_config["num_top_k"]  # 256
        H = large_config["hot_buffer_size"]  # 512
        
        # Warp-parallel scan: O(K × H/32)
        scan_ops = K * (H // 32)  # 256 × 16 = 4096
        
        # Hash: O(K + H)
        hash_ops = K + H  # 256 + 512 = 768
        
        print(f"\nComplexity for K={K}, H={H}:")
        print(f"  Warp-scan: O(K × H/32) = {scan_ops}")
        print(f"  Hash:      O(K + H)    = {hash_ops}")
        print(f"  Hash is {scan_ops / hash_ops:.1f}x better")
        
        assert hash_ops < scan_ops, "Hash should be more efficient for this config"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
