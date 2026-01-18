"""
Test script for load_cache kernel with SHUFFLE ROTATION hit detection.

Algorithm:
- Each warp holds 32 buffer tokens (one per lane)
- Load 32 top_k tokens into registers (one per lane)
- Use __shfl_sync to rotate top_k across lanes
- 32 rotations = 32×32 = 1024 comparisons per chunk
- No hash table needed, memory efficient

Complexity: O(H × K) but with fast shuffle operations
Memory: K × 16 bytes (vs K × 24 bytes for hash)
Best for: K ≤ 2048 where hash memory is tight
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
 * Shuffle-rotation based hit detection kernel.
 * 
 * Key insight: Use __shfl_sync to rotate top_k tokens across lanes,
 * enabling 32×32 all-to-all comparisons with just 32 shuffle operations.
 * 
 * Memory efficient: No hash table needed.
 * 
 * Algorithm:
 * 1. Load top_k into shared memory
 * 2. Each warp processes 32 buffer slots simultaneously
 * 3. For each top_k chunk of 32:
 *    - Load into registers (one per lane)
 *    - Rotate 32 times using shuffle
 *    - Each rotation: 32 parallel comparisons
 * 4. Collect hits and evictable slots
 * 5. Process misses
 */
template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE>
__global__ void load_cache_shuffle_kernel(
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
    
    // Shared memory for top_k tokens and results
    __shared__ int32_t s_top_k[NUM_TOP_K];
    __shared__ int32_t s_hit_slots[NUM_TOP_K];      // buffer slot for each top_k (-1 = miss)
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
    
    // ===== Phase 2: Shuffle-rotation hit detection =====
    // Each warp processes 32 buffer slots at a time
    
    for (int buf_base = warp_id * WARP_SIZE; buf_base < HOT_BUFFER_SIZE; buf_base += NUM_WARPS * WARP_SIZE) {
        int buf_slot = buf_base + lane_id;
        bool valid_slot = (buf_slot < HOT_BUFFER_SIZE);
        
        // Each lane loads one buffer token
        int32_t my_buffer_token = valid_slot ? device_buffer_tokens[buf_slot] : -1;
        
        // Track if this buffer token was found in top_k
        int my_found_top_k_idx = -1;
        
        // Process top_k in chunks of 32
        for (int top_k_base = 0; top_k_base < NUM_TOP_K; top_k_base += WARP_SIZE) {
            // Each lane loads one top_k token into register
            int top_k_idx = top_k_base + lane_id;
            int32_t my_top_k_token = (top_k_idx < NUM_TOP_K) ? s_top_k[top_k_idx] : -1;
            
            // 32 shuffle rotations for all-to-all comparison
            #pragma unroll
            for (int rot = 0; rot < WARP_SIZE; rot++) {
                // Get top_k token from lane (lane_id + rot) % 32
                int src_lane = (lane_id + rot) & 31;  // Fast modulo for power of 2
                int32_t rotated_top_k = __shfl_sync(0xFFFFFFFF, my_top_k_token, src_lane);
                int rotated_top_k_idx = top_k_base + src_lane;
                
                // Compare my buffer token against rotated top_k token
                if (my_buffer_token >= 0 && my_buffer_token == rotated_top_k && rotated_top_k_idx < NUM_TOP_K) {
                    my_found_top_k_idx = rotated_top_k_idx;
                }
            }
        }
        
        // Record hits: if my buffer token matched a top_k token
        if (my_found_top_k_idx >= 0) {
            s_hit_slots[my_found_top_k_idx] = buf_slot;
        }
        
        // Collect evictable slots using warp-cooperative pattern
        bool is_evictable = valid_slot && (my_found_top_k_idx < 0);
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
                s_evictable_slots[warp_offset + my_prefix] = buf_slot;
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
        
        // Set output for hits
        if (is_hit) {
            top_k_device_locs[i] = device_buffer_locs[slot];
        }
        
        // Collect misses using warp-cooperative pattern
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

void load_cache_shuffle_launcher(
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
        load_cache_shuffle_kernel<256, 64, 128><<<1, 256>>>(
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
        load_cache_shuffle_kernel<256, 256, 512><<<1, 256>>>(
            top_k_tokens.data_ptr<int32_t>(),
            device_buffer_tokens.data_ptr<int32_t>(),
            host_cache_locs.data_ptr<int64_t>(),
            device_buffer_locs.data_ptr<int64_t>(),
            host_cache.data_ptr<uint8_t>(),
            device_buffer.data_ptr<uint8_t>(),
            top_k_device_locs.data_ptr<int64_t>(),
            item_size_bytes
        );
    } else if (num_top_k <= 512 && hot_buffer_size <= 2048) {
        load_cache_shuffle_kernel<256, 512, 2048><<<1, 256>>>(
            top_k_tokens.data_ptr<int32_t>(),
            device_buffer_tokens.data_ptr<int32_t>(),
            host_cache_locs.data_ptr<int64_t>(),
            device_buffer_locs.data_ptr<int64_t>(),
            host_cache.data_ptr<uint8_t>(),
            device_buffer.data_ptr<uint8_t>(),
            top_k_device_locs.data_ptr<int64_t>(),
            item_size_bytes
        );
    } else if (num_top_k <= 1024 && hot_buffer_size <= 4096) {
        load_cache_shuffle_kernel<256, 1024, 4096><<<1, 256>>>(
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
        // Large K: 2048
        load_cache_shuffle_kernel<256, 2048, 8192><<<1, 256>>>(
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
void load_cache_shuffle_launcher(
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
            name='test_load_cache_shuffle',
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=['load_cache_shuffle_launcher'],
            verbose=False,
            extra_cuda_cflags=['-O3'],
        )
    return _module


class CacheManagerShuffle:
    """Cache manager with shuffle-rotation hit detection."""
    
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
        
        self._module.load_cache_shuffle_launcher(
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


class PythonReferenceShuffle:
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
        top_k_set = {token: i for i, token in enumerate(top_k_tokens)}
        
        hit_slots = [-1] * len(top_k_tokens)
        evictable = []
        
        # Scan buffer, find hits
        for slot, token in enumerate(self.device_buffer_tokens):
            if token in top_k_set:
                top_k_idx = top_k_set[token]
                hit_slots[top_k_idx] = slot
            else:
                evictable.append(slot)
        
        # Process hits
        misses = []
        for i, slot in enumerate(hit_slots):
            if slot >= 0:
                top_k_device_locs[i] = self.device_buffer_locs[slot]
            else:
                misses.append(i)
        
        # Process misses
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


class TestLoadCacheShuffle:
    """Tests for shuffle-rotation kernel."""
    
    @pytest.fixture
    def small_config(self):
        return {
            "max_tokens": 1000,
            "hot_buffer_size": 128,
            "item_size_bytes": 64,
            "num_top_k": 64,
        }
    
    @pytest.fixture
    def medium_config(self):
        return {
            "max_tokens": 5000,
            "hot_buffer_size": 512,
            "item_size_bytes": 64,
            "num_top_k": 256,
        }
    
    @pytest.fixture
    def large_config(self):
        """Large K where shuffle rotation shines vs hash memory."""
        return {
            "max_tokens": 10000,
            "hot_buffer_size": 2048,
            "item_size_bytes": 64,
            "num_top_k": 512,
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
    
    def test_all_misses(self, small_config):
        """All tokens are cache misses."""
        config = small_config
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        ref = PythonReferenceShuffle(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManagerShuffle(
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
    
    def test_all_hits(self, small_config):
        """All tokens are cache hits."""
        config = small_config
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        ref = PythonReferenceShuffle(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManagerShuffle(
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
    
    def test_mixed(self, small_config):
        """Mix of hits and misses."""
        config = small_config
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        ref = PythonReferenceShuffle(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManagerShuffle(
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
    
    def test_medium_config(self, medium_config):
        """Test with medium K=256."""
        config = medium_config
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        ref = PythonReferenceShuffle(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManagerShuffle(
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
    
    def test_large_config(self, large_config):
        """Test with large K=512."""
        config = large_config
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        ref = PythonReferenceShuffle(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManagerShuffle(
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
    
    def test_data_integrity(self, small_config):
        """Verify copied data is correct."""
        config = small_config
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        cuda = CacheManagerShuffle(
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
    
    def test_multiple_iterations(self, small_config):
        """Test multiple process_topk calls with changing tokens."""
        config = small_config
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        ref = PythonReferenceShuffle(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManagerShuffle(
            config["max_tokens"], config["hot_buffer_size"],
            config["item_size_bytes"], config["num_top_k"]
        )
        
        device_ref = bytearray(config["hot_buffer_size"] * config["item_size_bytes"])
        
        # Multiple iterations with shifting tokens
        for iteration in range(5):
            base = iteration * 10
            tokens = list(range(base, base + config["num_top_k"]))
            top_k = torch.tensor(tokens, dtype=torch.int32, device="cuda")
            
            ref_locs = ref.process_topk(tokens, host_np, device_ref)
            cuda_locs = cuda.process_topk(top_k, host_torch, device_torch)
            torch.cuda.synchronize()
            
            assert cuda_locs.cpu().tolist() == ref_locs, f"Mismatch at iteration {iteration}"
    
    def test_memory_comparison(self):
        """Print memory usage comparison."""
        print("\n=== Memory Usage Comparison ===")
        print(f"{'K':<8} {'Shuffle (KB)':<15} {'Hash (KB)':<15} {'Savings':<10}")
        print("-" * 50)
        
        for K in [64, 256, 512, 1024, 2048]:
            shuffle_mem = K * 16 / 1024  # K × 16 bytes
            hash_mem = K * 24 / 1024     # K × 24 bytes
            savings = (hash_mem - shuffle_mem) / hash_mem * 100
            print(f"{K:<8} {shuffle_mem:<15.1f} {hash_mem:<15.1f} {savings:<10.1f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
