"""
Test script for load_cache_to_device_buffer CUDA kernel.

Tests:
1. All hits - tokens already in device buffer
2. All misses - all tokens need to be loaded
3. Mixed hits/misses
4. Edge cases (empty, single token, full buffer)
5. Data copy correctness
6. Multiple iterations (state consistency)
7. Large NUM_TOP_K with chunking
"""

import pytest
import torch
from torch.utils.cpp_extension import load_inline

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

CUDA_SOURCE = r'''
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

constexpr int WARP_SIZE = 32;
constexpr uint16_t NOT_PRESENT = 0xFFFF;

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

__device__ __forceinline__ int warp_inclusive_scan(int* s_data, int lane_id, int offset, int count, int accumulator) {
    int idx = lane_id + offset;
    int val = (idx < count) ? s_data[idx] : 0;
    
    #pragma unroll
    for (int i = 1; i < 32; i *= 2) {
        int n = __shfl_up_sync(0xffffffff, val, i);
        if (lane_id >= i) val += n;
    }
    val += accumulator;
    if (idx < count) {
        s_data[idx] = val;
    }
    accumulator = __shfl_sync(0xffffffff, val, 31);
    return accumulator;
}

template <int BLOCK_SIZE, int MAX_TOP_K, int HOT_BUFFER_SIZE>
__global__ void load_cache_to_device_buffer(
    const uint32_t* __restrict__ top_k_tokens,
    uint32_t* __restrict__ device_buffer_tokens,
    uint16_t* __restrict__ token_residence_mapping,
    const int64_t* __restrict__ host_cache_locs,
    const int64_t* __restrict__ device_buffer_locs,
    const void* __restrict__ host_cache,
    void* __restrict__ device_buffer,
    int64_t* __restrict__ top_k_device_locs,
    uint32_t max_tokens,
    int64_t item_size_bytes,
    int32_t actual_num_top_k) {  // Added: actual number of tokens
    
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    // Use MAX_TOP_K for shared memory sizing, actual_num_top_k for bounds
    const int num_chunks = (actual_num_top_k + WARP_SIZE - 1) / WARP_SIZE;
    constexpr int MAX_CHUNKS = (MAX_TOP_K + WARP_SIZE - 1) / WARP_SIZE;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const unsigned int lanes_before = (1u << lane_id) - 1;

    __shared__ bool s_protected_bitmap[HOT_BUFFER_SIZE];
    __shared__ int32_t s_chunk_miss_offset[MAX_CHUNKS + 1];
    __shared__ int32_t s_missed_tokens[MAX_TOP_K];
    __shared__ int16_t s_missed_tokens_idx[MAX_TOP_K];
    __shared__ int32_t s_evictable_slots[MAX_TOP_K];
    __shared__ int32_t s_total_misses;

    if (tid == 0) {
        s_total_misses = 0;
    }
    for (int i = tid; i < HOT_BUFFER_SIZE; i += BLOCK_SIZE) {
        s_protected_bitmap[i] = false;
    }
    for (int i = tid; i < num_chunks + 1; i += BLOCK_SIZE) {
        s_chunk_miss_offset[i] = 0;
    }
    __syncthreads();

    // ===== Phase 1: Hit and Miss Detection ====
    const int iterations_per_warp = (num_chunks + NUM_WARPS - 1) / NUM_WARPS;
    for (int iter = 0; iter < iterations_per_warp; iter++) {
        int chunk_idx = warp_id + iter * NUM_WARPS;
        bool has_valid_chunk = chunk_idx < num_chunks;

        const int chunk_token_start = chunk_idx * WARP_SIZE;
        const int my_token_idx = chunk_token_start + lane_id;
        const bool has_valid_token = has_valid_chunk && (my_token_idx < actual_num_top_k);

        int32_t my_token = 0;
        bool is_hit = false;
        bool is_miss = false;
        int32_t my_device_slot = -1;
        int local_miss_offset = 0;

        if (has_valid_token) {
            my_token = top_k_tokens[my_token_idx];
            uint16_t slot = token_residence_mapping[my_token];
            is_hit = (slot != NOT_PRESENT);

            if (is_hit) {
                my_device_slot = static_cast<int32_t>(slot);
                s_protected_bitmap[my_device_slot] = true;
                top_k_device_locs[my_token_idx] = device_buffer_locs[my_device_slot];
            } else {
                is_miss = true;
            }
        }

        const unsigned int miss_mask = __ballot_sync(0xFFFFFFFF, is_miss);
        if (has_valid_chunk) {
            local_miss_offset = __popc(miss_mask & lanes_before);
            const int warp_miss_count = __popc(miss_mask);
            if (lane_id == 0) {
                s_chunk_miss_offset[chunk_idx + 1] = warp_miss_count;
            }
        }
        __syncthreads();

        if (warp_id == 0) {
            s_total_misses = warp_inclusive_scan(
                s_chunk_miss_offset, lane_id, iter * NUM_WARPS + 1, num_chunks + 1, s_total_misses);
        }
        __syncthreads();

        if (is_miss) {
            int miss_offset = s_chunk_miss_offset[chunk_idx] + local_miss_offset;
            s_missed_tokens[miss_offset] = my_token;
            s_missed_tokens_idx[miss_offset] = static_cast<int16_t>(my_token_idx);
        }
        __syncthreads();
    }

    // ===== Phase 2: Find Evictable Slots =====
    if (warp_id == 0) {
        int found = 0;
        int base_slot = 0;

        while (found < s_total_misses && base_slot < HOT_BUFFER_SIZE) {
            const int my_slot = base_slot + lane_id;
            bool is_evictable = false;

            if (my_slot < HOT_BUFFER_SIZE) {
                is_evictable = !s_protected_bitmap[my_slot];
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

    // ===== Phase 3: Assignment + Direct Data Copy =====
    for (int warp_iter = 0; warp_iter < (s_total_misses + NUM_WARPS - 1) / NUM_WARPS; warp_iter++) {
        int token_idx = warp_id + warp_iter * NUM_WARPS;
        bool has_work = token_idx < s_total_misses;
        int64_t src_offset = 0;
        int64_t dst_offset = 0;
        
        if (has_work) {
            const int32_t miss_token = s_missed_tokens[token_idx];
            const int evict_slot = s_evictable_slots[token_idx];
            const int top_k_idx = s_missed_tokens_idx[token_idx];
            const int32_t old_token = device_buffer_tokens[evict_slot];
            
            if (old_token >= 0 && old_token < max_tokens) {
                token_residence_mapping[old_token] = NOT_PRESENT;
            }
            token_residence_mapping[miss_token] = static_cast<uint16_t>(evict_slot);
            device_buffer_tokens[evict_slot] = miss_token;
            top_k_device_locs[top_k_idx] = device_buffer_locs[evict_slot];
            src_offset = host_cache_locs[miss_token] * item_size_bytes;
            dst_offset = device_buffer_locs[evict_slot] * item_size_bytes;
        }

        #pragma unroll
        for (int src_lane = 0; src_lane < WARP_SIZE; src_lane++) {
            bool should_copy = __shfl_sync(0xFFFFFFFF, has_work, src_lane);
            if (should_copy) {
                int64_t final_src_offset = __shfl_sync(0xFFFFFFFF, src_offset, src_lane);
                int64_t final_dst_offset = __shfl_sync(0xFFFFFFFF, dst_offset, src_lane);
                transfer_item_warp(
                    lane_id,
                    static_cast<const char*>(host_cache) + final_src_offset,
                    static_cast<char*>(device_buffer) + final_dst_offset,
                    item_size_bytes);
            }
        }
    }
}

// Launcher functions for different configurations
void load_cache_launcher_64_128(
    torch::Tensor top_k_tokens,
    torch::Tensor device_buffer_tokens,
    torch::Tensor token_residence_mapping,
    torch::Tensor host_cache_locs,
    torch::Tensor device_buffer_locs,
    torch::Tensor host_cache,
    torch::Tensor device_buffer,
    torch::Tensor top_k_device_locs,
    int64_t max_tokens,
    int64_t item_size_bytes,
    int64_t actual_num_top_k
) {
    load_cache_to_device_buffer<256, 64, 128><<<1, 256>>>(
        reinterpret_cast<uint32_t*>(top_k_tokens.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(device_buffer_tokens.data_ptr<int32_t>()),
        reinterpret_cast<uint16_t*>(token_residence_mapping.data_ptr<int16_t>()),
        host_cache_locs.data_ptr<int64_t>(),
        device_buffer_locs.data_ptr<int64_t>(),
        host_cache.data_ptr<uint8_t>(),
        device_buffer.data_ptr<uint8_t>(),
        top_k_device_locs.data_ptr<int64_t>(),
        static_cast<uint32_t>(max_tokens),
        item_size_bytes,
        static_cast<int32_t>(actual_num_top_k)
    );
}

void load_cache_launcher_256_512(
    torch::Tensor top_k_tokens,
    torch::Tensor device_buffer_tokens,
    torch::Tensor token_residence_mapping,
    torch::Tensor host_cache_locs,
    torch::Tensor device_buffer_locs,
    torch::Tensor host_cache,
    torch::Tensor device_buffer,
    torch::Tensor top_k_device_locs,
    int64_t max_tokens,
    int64_t item_size_bytes,
    int64_t actual_num_top_k
) {
    load_cache_to_device_buffer<256, 256, 512><<<1, 256>>>(
        reinterpret_cast<uint32_t*>(top_k_tokens.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(device_buffer_tokens.data_ptr<int32_t>()),
        reinterpret_cast<uint16_t*>(token_residence_mapping.data_ptr<int16_t>()),
        host_cache_locs.data_ptr<int64_t>(),
        device_buffer_locs.data_ptr<int64_t>(),
        host_cache.data_ptr<uint8_t>(),
        device_buffer.data_ptr<uint8_t>(),
        top_k_device_locs.data_ptr<int64_t>(),
        static_cast<uint32_t>(max_tokens),
        item_size_bytes,
        static_cast<int32_t>(actual_num_top_k)
    );
}
'''

CPP_SOURCE = r'''
void load_cache_launcher_64_128(
    torch::Tensor top_k_tokens,
    torch::Tensor device_buffer_tokens,
    torch::Tensor token_residence_mapping,
    torch::Tensor host_cache_locs,
    torch::Tensor device_buffer_locs,
    torch::Tensor host_cache,
    torch::Tensor device_buffer,
    torch::Tensor top_k_device_locs,
    int64_t max_tokens,
    int64_t item_size_bytes,
    int64_t actual_num_top_k
);

void load_cache_launcher_256_512(
    torch::Tensor top_k_tokens,
    torch::Tensor device_buffer_tokens,
    torch::Tensor token_residence_mapping,
    torch::Tensor host_cache_locs,
    torch::Tensor device_buffer_locs,
    torch::Tensor host_cache,
    torch::Tensor device_buffer,
    torch::Tensor top_k_device_locs,
    int64_t max_tokens,
    int64_t item_size_bytes,
    int64_t actual_num_top_k
);
'''

_module = None

def get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name='test_load_cache',
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=['load_cache_launcher_64_128', 'load_cache_launcher_256_512'],
            verbose=False,
            extra_cuda_cflags=['-O3'],
        )
    return _module


class PythonReference:
    """Python reference implementation for verification."""
    
    NOT_PRESENT = -1  # 0xFFFF in uint16, but -1 in int16 (same bit pattern)
    
    def __init__(self, max_tokens: int, hot_buffer_size: int, item_size_bytes: int):
        self.max_tokens = max_tokens
        self.hot_buffer_size = hot_buffer_size
        self.item_size_bytes = item_size_bytes
        
        # State
        self.token_residence_mapping = [self.NOT_PRESENT] * max_tokens
        self.device_buffer_tokens = [self.NOT_PRESENT] * hot_buffer_size  # Use NOT_PRESENT as "empty"
        self.device_buffer_locs = list(range(hot_buffer_size))
        self.host_cache_locs = list(range(max_tokens))
    
    def reset(self):
        self.token_residence_mapping = [self.NOT_PRESENT] * self.max_tokens
        self.device_buffer_tokens = [self.NOT_PRESENT] * self.hot_buffer_size
    
    def process_topk(self, top_k_tokens: list, host_cache: bytes, device_buffer: bytearray):
        """
        Process top-k tokens and return device locations.
        Also performs data copy.
        """
        top_k_device_locs = [0] * len(top_k_tokens)
        
        # Phase 1: Identify hits and misses
        hits = []  # (top_k_idx, token, slot)
        misses = []  # (top_k_idx, token)
        protected_slots = set()
        
        for i, token in enumerate(top_k_tokens):
            slot = self.token_residence_mapping[token]
            if slot != self.NOT_PRESENT:
                # Hit
                hits.append((i, token, slot))
                top_k_device_locs[i] = self.device_buffer_locs[slot]
                protected_slots.add(slot)
            else:
                # Miss
                misses.append((i, token))
        
        # Phase 2: Find evictable slots
        evictable_slots = []
        for slot in range(self.hot_buffer_size):
            if slot not in protected_slots:
                evictable_slots.append(slot)
                if len(evictable_slots) >= len(misses):
                    break
        
        # Phase 3: Assignment and copy
        for miss_idx, (top_k_idx, token) in enumerate(misses):
            evict_slot = evictable_slots[miss_idx]
            old_token = self.device_buffer_tokens[evict_slot]
            
            # Clear old token
            if old_token != self.NOT_PRESENT and old_token < self.max_tokens:
                self.token_residence_mapping[old_token] = self.NOT_PRESENT
            
            # Set new token
            self.token_residence_mapping[token] = evict_slot
            self.device_buffer_tokens[evict_slot] = token
            top_k_device_locs[top_k_idx] = self.device_buffer_locs[evict_slot]
            
            # Copy data
            src_offset = self.host_cache_locs[token] * self.item_size_bytes
            dst_offset = self.device_buffer_locs[evict_slot] * self.item_size_bytes
            device_buffer[dst_offset:dst_offset + self.item_size_bytes] = \
                host_cache[src_offset:src_offset + self.item_size_bytes]
        
        return top_k_device_locs


class CUDAKernel:
    """Wrapper for CUDA kernel."""
    
    NOT_PRESENT = -1  # 0xFFFF in uint16, but -1 in int16 (same bit pattern)
    
    def __init__(self, max_tokens: int, hot_buffer_size: int, item_size_bytes: int, 
                 num_top_k: int = 64, device: str = "cuda"):
        self.max_tokens = max_tokens
        self.hot_buffer_size = hot_buffer_size
        self.item_size_bytes = item_size_bytes
        self.num_top_k = num_top_k
        self.device = device
        
        # State tensors
        self.token_residence_mapping = torch.full(
            (max_tokens,), self.NOT_PRESENT, dtype=torch.int16, device=device
        )
        self.device_buffer_tokens = torch.full(
            (hot_buffer_size,), self.NOT_PRESENT, dtype=torch.int32, device=device
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
        self.token_residence_mapping.fill_(self.NOT_PRESENT)
        self.device_buffer_tokens.fill_(self.NOT_PRESENT)
    
    def process_topk(self, top_k_tokens: torch.Tensor, host_cache: torch.Tensor, 
                     device_buffer: torch.Tensor) -> torch.Tensor:
        self._ensure_compiled()
        
        if not isinstance(top_k_tokens, torch.Tensor):
            top_k_tokens = torch.tensor(top_k_tokens, dtype=torch.int32, device=self.device)
        elif top_k_tokens.dtype != torch.int32:
            top_k_tokens = top_k_tokens.to(torch.int32)
        
        if top_k_tokens.device.type != 'cuda':
            top_k_tokens = top_k_tokens.to(self.device)
        
        top_k_size = top_k_tokens.numel()
        top_k_device_locs = torch.zeros(top_k_size, dtype=torch.int64, device=self.device)
        
        if self.num_top_k <= 64 and self.hot_buffer_size <= 128:
            self._module.load_cache_launcher_64_128(
                top_k_tokens,
                self.device_buffer_tokens,
                self.token_residence_mapping,
                self.host_cache_locs,
                self.device_buffer_locs,
                host_cache,
                device_buffer,
                top_k_device_locs,
                self.max_tokens,
                self.item_size_bytes,
                top_k_size,  # actual number of tokens
            )
        else:
            self._module.load_cache_launcher_256_512(
                top_k_tokens,
                self.device_buffer_tokens,
                self.token_residence_mapping,
                self.host_cache_locs,
                self.device_buffer_locs,
                host_cache,
                device_buffer,
                top_k_device_locs,
                self.max_tokens,
                self.item_size_bytes,
                top_k_size,  # actual number of tokens
            )
        
        return top_k_device_locs


class TestLoadCacheToDeviceBuffer:
    """Test suite for load_cache_to_device_buffer kernel."""
    
    @pytest.fixture
    def small_config(self):
        """Small configuration for basic tests."""
        return {
            "max_tokens": 1000,
            "hot_buffer_size": 100,
            "item_size_bytes": 64,  # Must be multiple of 8
            "num_top_k": 32,
        }
    
    @pytest.fixture
    def medium_config(self):
        """Medium configuration."""
        return {
            "max_tokens": 10000,
            "hot_buffer_size": 500,
            "item_size_bytes": 128,
            "num_top_k": 200,
        }
    
    def create_test_data(self, config):
        """Create test data for both Python reference and CUDA kernel."""
        max_tokens = config["max_tokens"]
        hot_buffer_size = config["hot_buffer_size"]
        item_size_bytes = config["item_size_bytes"]
        
        # Create host cache with unique data per token
        host_cache_np = bytearray(max_tokens * item_size_bytes)
        for token in range(max_tokens):
            offset = token * item_size_bytes
            # Fill with token-specific pattern
            for i in range(item_size_bytes):
                host_cache_np[offset + i] = (token + i) % 256
        
        # PyTorch tensors
        host_cache_torch = torch.tensor(list(host_cache_np), dtype=torch.uint8, device="cpu").pin_memory()
        device_buffer_torch = torch.zeros(hot_buffer_size * item_size_bytes, dtype=torch.uint8, device="cuda")
        
        return host_cache_np, host_cache_torch, device_buffer_torch
    
    def test_all_misses(self, small_config):
        """Test with all cache misses (empty buffer)."""
        config = small_config
        host_cache_np, host_cache_torch, device_buffer_torch = self.create_test_data(config)
        
        # Create reference and CUDA implementations
        ref = PythonReference(config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"])
        cuda_kernel = CUDAKernel(config["max_tokens"], config["hot_buffer_size"], 
                                  config["item_size_bytes"], config["num_top_k"])
        
        # Test tokens (all new)
        top_k_tokens = list(range(10, 10 + config["num_top_k"]))
        top_k_torch = torch.tensor(top_k_tokens, dtype=torch.int32, device="cuda")
        
        # Python reference
        device_buffer_ref = bytearray(config["hot_buffer_size"] * config["item_size_bytes"])
        ref_locs = ref.process_topk(top_k_tokens, host_cache_np, device_buffer_ref)
        
        # CUDA kernel
        cuda_locs = cuda_kernel.process_topk(top_k_torch, host_cache_torch, device_buffer_torch)
        torch.cuda.synchronize()
        
        # Verify locations match
        cuda_locs_list = cuda_locs.cpu().tolist()
        assert cuda_locs_list == ref_locs, f"Locations mismatch: CUDA={cuda_locs_list}, Ref={ref_locs}"
        
        # Verify data copy
        device_buffer_cpu = device_buffer_torch.cpu().numpy()
        for i, token in enumerate(top_k_tokens):
            loc = ref_locs[i]
            offset = loc * config["item_size_bytes"]
            expected = host_cache_np[token * config["item_size_bytes"]:(token + 1) * config["item_size_bytes"]]
            actual = device_buffer_cpu[offset:offset + config["item_size_bytes"]]
            assert list(actual) == list(expected), f"Data mismatch for token {token}"
    
    def test_all_hits(self, small_config):
        """Test with all cache hits."""
        config = small_config
        host_cache_np, host_cache_torch, device_buffer_torch = self.create_test_data(config)
        
        ref = PythonReference(config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"])
        cuda_kernel = CUDAKernel(config["max_tokens"], config["hot_buffer_size"], 
                                  config["item_size_bytes"], config["num_top_k"])
        
        # First pass - load some tokens
        top_k_tokens = list(range(10, 10 + config["num_top_k"]))
        top_k_torch = torch.tensor(top_k_tokens, dtype=torch.int32, device="cuda")
        
        device_buffer_ref = bytearray(config["hot_buffer_size"] * config["item_size_bytes"])
        ref.process_topk(top_k_tokens, host_cache_np, device_buffer_ref)
        cuda_kernel.process_topk(top_k_torch, host_cache_torch, device_buffer_torch)
        torch.cuda.synchronize()
        
        # Second pass - same tokens (all hits)
        ref_locs2 = ref.process_topk(top_k_tokens, host_cache_np, device_buffer_ref)
        cuda_locs2 = cuda_kernel.process_topk(top_k_torch, host_cache_torch, device_buffer_torch)
        torch.cuda.synchronize()
        
        cuda_locs2_list = cuda_locs2.cpu().tolist()
        assert cuda_locs2_list == ref_locs2, f"All-hits locations mismatch"
    
    def test_mixed_hits_misses(self, small_config):
        """Test with mix of hits and misses."""
        config = small_config
        host_cache_np, host_cache_torch, device_buffer_torch = self.create_test_data(config)
        
        ref = PythonReference(config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"])
        cuda_kernel = CUDAKernel(config["max_tokens"], config["hot_buffer_size"], 
                                  config["item_size_bytes"], config["num_top_k"])
        
        # First pass
        top_k_tokens1 = list(range(0, config["num_top_k"]))
        top_k_torch1 = torch.tensor(top_k_tokens1, dtype=torch.int32, device="cuda")
        
        device_buffer_ref = bytearray(config["hot_buffer_size"] * config["item_size_bytes"])
        ref.process_topk(top_k_tokens1, host_cache_np, device_buffer_ref)
        cuda_kernel.process_topk(top_k_torch1, host_cache_torch, device_buffer_torch)
        torch.cuda.synchronize()
        
        # Second pass - half old, half new
        half = config["num_top_k"] // 2
        top_k_tokens2 = list(range(0, half)) + list(range(100, 100 + half))
        top_k_torch2 = torch.tensor(top_k_tokens2, dtype=torch.int32, device="cuda")
        
        ref_locs2 = ref.process_topk(top_k_tokens2, host_cache_np, device_buffer_ref)
        cuda_locs2 = cuda_kernel.process_topk(top_k_torch2, host_cache_torch, device_buffer_torch)
        torch.cuda.synchronize()
        
        cuda_locs2_list = cuda_locs2.cpu().tolist()
        assert cuda_locs2_list == ref_locs2, f"Mixed hits/misses locations mismatch"
    
    def test_eviction(self, small_config):
        """Test that eviction works correctly when buffer is full."""
        config = small_config.copy()
        config["hot_buffer_size"] = 20  # Small buffer to force eviction
        config["num_top_k"] = 10
        
        host_cache_np, host_cache_torch, device_buffer_torch = self.create_test_data(config)
        
        ref = PythonReference(config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"])
        cuda_kernel = CUDAKernel(config["max_tokens"], config["hot_buffer_size"], 
                                  config["item_size_bytes"], config["num_top_k"])
        
        device_buffer_ref = bytearray(config["hot_buffer_size"] * config["item_size_bytes"])
        
        # Fill buffer with tokens 0-19
        for batch in range(2):
            tokens = list(range(batch * 10, batch * 10 + 10))
            top_k_torch = torch.tensor(tokens, dtype=torch.int32, device="cuda")
            ref.process_topk(tokens, host_cache_np, device_buffer_ref)
            cuda_kernel.process_topk(top_k_torch, host_cache_torch, device_buffer_torch)
        torch.cuda.synchronize()
        
        # Now request tokens 0-4 (hits) and 100-104 (misses, require eviction)
        top_k_tokens = list(range(0, 5)) + list(range(100, 105))
        top_k_torch = torch.tensor(top_k_tokens, dtype=torch.int32, device="cuda")
        
        ref_locs = ref.process_topk(top_k_tokens, host_cache_np, device_buffer_ref)
        cuda_locs = cuda_kernel.process_topk(top_k_torch, host_cache_torch, device_buffer_torch)
        torch.cuda.synchronize()
        
        cuda_locs_list = cuda_locs.cpu().tolist()
        assert cuda_locs_list == ref_locs, f"Eviction test locations mismatch"
        
        # Verify all locations are unique
        assert len(set(cuda_locs_list)) == len(cuda_locs_list), "Duplicate locations found!"
    
    def test_multiple_iterations(self, small_config):
        """Test state consistency across multiple iterations."""
        config = small_config
        host_cache_np, host_cache_torch, device_buffer_torch = self.create_test_data(config)
        
        ref = PythonReference(config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"])
        cuda_kernel = CUDAKernel(config["max_tokens"], config["hot_buffer_size"], 
                                  config["item_size_bytes"], config["num_top_k"])
        
        device_buffer_ref = bytearray(config["hot_buffer_size"] * config["item_size_bytes"])
        
        import random
        random.seed(42)
        
        for iteration in range(10):
            # Random tokens
            tokens = random.sample(range(config["max_tokens"]), config["num_top_k"])
            top_k_torch = torch.tensor(tokens, dtype=torch.int32, device="cuda")
            
            ref_locs = ref.process_topk(tokens, host_cache_np, device_buffer_ref)
            cuda_locs = cuda_kernel.process_topk(top_k_torch, host_cache_torch, device_buffer_torch)
            torch.cuda.synchronize()
            
            cuda_locs_list = cuda_locs.cpu().tolist()
            assert cuda_locs_list == ref_locs, f"Iteration {iteration}: locations mismatch"
    
    def test_large_num_top_k(self, medium_config):
        """Test with larger NUM_TOP_K requiring warp chunking."""
        config = medium_config
        host_cache_np, host_cache_torch, device_buffer_torch = self.create_test_data(config)
        
        ref = PythonReference(config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"])
        cuda_kernel = CUDAKernel(config["max_tokens"], config["hot_buffer_size"], 
                                  config["item_size_bytes"], config["num_top_k"])
        
        device_buffer_ref = bytearray(config["hot_buffer_size"] * config["item_size_bytes"])
        
        # Test with 200 tokens (requires multiple chunks with 8 warps)
        tokens = list(range(config["num_top_k"]))
        top_k_torch = torch.tensor(tokens, dtype=torch.int32, device="cuda")
        
        ref_locs = ref.process_topk(tokens, host_cache_np, device_buffer_ref)
        cuda_locs = cuda_kernel.process_topk(top_k_torch, host_cache_torch, device_buffer_torch)
        torch.cuda.synchronize()
        
        cuda_locs_list = cuda_locs.cpu().tolist()
        assert cuda_locs_list == ref_locs, f"Large NUM_TOP_K test failed"
    
    def test_data_integrity(self, small_config):
        """Verify that copied data is correct."""
        config = small_config
        config["num_top_k"] = 16
        host_cache_np, host_cache_torch, device_buffer_torch = self.create_test_data(config)
        
        cuda_kernel = CUDAKernel(config["max_tokens"], config["hot_buffer_size"], 
                                  config["item_size_bytes"], config["num_top_k"])
        
        # Request specific tokens
        tokens = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        top_k_torch = torch.tensor(tokens, dtype=torch.int32, device="cuda")
        
        cuda_locs = cuda_kernel.process_topk(top_k_torch, host_cache_torch, device_buffer_torch)
        torch.cuda.synchronize()
        
        # Verify data for each token
        device_buffer_cpu = device_buffer_torch.cpu().numpy()
        cuda_locs_list = cuda_locs.cpu().tolist()
        
        for i, token in enumerate(tokens):
            loc = cuda_locs_list[i]
            offset = loc * config["item_size_bytes"]
            
            # Expected data
            expected = []
            for j in range(config["item_size_bytes"]):
                expected.append((token + j) % 256)
            
            actual = list(device_buffer_cpu[offset:offset + config["item_size_bytes"]])
            assert actual == expected, f"Data integrity failed for token {token}: expected {expected[:8]}..., got {actual[:8]}..."
    
    def test_residence_mapping_consistency(self, small_config):
        """Verify token_residence_mapping is consistent with device_buffer_tokens."""
        config = small_config
        host_cache_np, host_cache_torch, device_buffer_torch = self.create_test_data(config)
        
        cuda_kernel = CUDAKernel(config["max_tokens"], config["hot_buffer_size"], 
                                  config["item_size_bytes"], config["num_top_k"])
        
        import random
        random.seed(123)
        
        for _ in range(5):
            tokens = random.sample(range(config["max_tokens"]), config["num_top_k"])
            top_k_torch = torch.tensor(tokens, dtype=torch.int32, device="cuda")
            cuda_kernel.process_topk(top_k_torch, host_cache_torch, device_buffer_torch)
            torch.cuda.synchronize()
        
        # Check consistency
        residence = cuda_kernel.token_residence_mapping.cpu().numpy()
        buffer_tokens = cuda_kernel.device_buffer_tokens.cpu().numpy()
        
        for token in range(config["max_tokens"]):
            slot = residence[token]
            if slot != cuda_kernel.NOT_PRESENT:
                # Slot should contain this token
                assert buffer_tokens[slot] == token, \
                    f"Token {token} maps to slot {slot}, but slot contains {buffer_tokens[slot]}"
        
        for slot in range(config["hot_buffer_size"]):
            token = buffer_tokens[slot]
            if token != cuda_kernel.NOT_PRESENT and token < config["max_tokens"]:
                # Token should map to this slot
                assert residence[token] == slot, \
                    f"Slot {slot} contains token {token}, but token maps to slot {residence[token]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
