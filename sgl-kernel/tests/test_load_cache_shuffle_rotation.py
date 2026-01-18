"""
Test script for load_cache_to_device_buffer kernel with shuffle rotation hit detection.

Tests the optimized CUDA kernel that:
1. Uses shuffle rotation for all-to-all comparison (no hash table)
2. Uses warp-cooperative prefix sums for offset calculation
3. Efficiently manages KV cache hits/misses/evictions
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
constexpr int32_t TOKEN_HIT = -1;  // Sentinel value; assumes valid tokens are >= 0

__device__ __forceinline__ void
transfer_item_warp(int32_t lane_id, const void* src_addr, void* dst_addr, int64_t item_size_bytes) {
  const uint64_t* __restrict__ src = static_cast<const uint64_t*>(src_addr);
  uint64_t* __restrict__ dst = static_cast<uint64_t*>(dst_addr);
  const int total_chunks = item_size_bytes / sizeof(uint64_t);

#pragma unroll
  for (int j = lane_id; j < total_chunks; j += WARP_SIZE) {
    uint64_t tmp;
    asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src + j) : "memory");
    asm volatile("st.global.cg.b64 [%0],%1;" ::"l"(dst + j), "l"(tmp) : "memory");
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
  return __shfl_sync(0xffffffff, val, 31);
}

template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE>
__global__ void load_cache_to_device_buffer(
    const int32_t* __restrict__ top_k_tokens,
    int32_t* __restrict__ device_buffer_tokens,
    const int64_t* __restrict__ host_cache_locs,
    const int64_t* __restrict__ device_buffer_locs,
    const void* __restrict__ host_cache,
    void* __restrict__ device_buffer,
    int64_t* __restrict__ top_k_device_locs,
    int64_t item_size_bytes) {
    
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  constexpr int NUM_TOKEN_CHUNKS = (NUM_TOP_K + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int NUM_BUFFER_CHUNKS = (HOT_BUFFER_SIZE + WARP_SIZE - 1) / WARP_SIZE;

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const unsigned int lanes_before = (1u << lane_id) - 1;

  __shared__ int32_t s_top_k_tokens[NUM_TOP_K];
  __shared__ int32_t s_chunk_offset[NUM_BUFFER_CHUNKS + 1];
  __shared__ int32_t s_missed_tokens[NUM_TOP_K];
  __shared__ int32_t s_evictable_slots[NUM_TOP_K];
  __shared__ int32_t s_total_misses;

  // Initialize shared memory
  if (tid == 0) {
    s_total_misses = 0;
  }
  for (int i = tid; i < NUM_TOP_K; i += BLOCK_SIZE) {
    s_top_k_tokens[i] = top_k_tokens[i];
  }
  for (int i = tid; i < NUM_BUFFER_CHUNKS + 1; i += BLOCK_SIZE) {
    s_chunk_offset[i] = 0;
  }
  __syncthreads();

  // ===== Phase 1: Scan buffer, find hits and collect evictable slots =====
  constexpr int ITERATIONS_PER_WARP_BUFFER = (NUM_BUFFER_CHUNKS + NUM_WARPS - 1) / NUM_WARPS;
  int total_evictable = 0;
  
  for (int iter = 0; iter < ITERATIONS_PER_WARP_BUFFER; iter++) {
    const int chunk_idx = warp_id + iter * NUM_WARPS;
    const bool has_valid_chunk = chunk_idx < NUM_BUFFER_CHUNKS;
    const int buf_slot = chunk_idx * WARP_SIZE + lane_id;
    const bool has_valid_slot = has_valid_chunk && (buf_slot < HOT_BUFFER_SIZE);

    int32_t my_buffer_token = has_valid_slot ? device_buffer_tokens[buf_slot] : TOKEN_HIT;
    int my_found_top_k_idx = -1;

    // Shuffle rotation hit detection
    if (has_valid_chunk) {
      for (int top_k_base = 0; top_k_base < NUM_TOP_K; top_k_base += WARP_SIZE) {
        int top_k_idx = top_k_base + lane_id;
        int32_t my_top_k_token = (top_k_idx < NUM_TOP_K) ? s_top_k_tokens[top_k_idx] : TOKEN_HIT;

#pragma unroll
        for (int rot = 0; rot < WARP_SIZE; rot++) {
          int src_lane = (lane_id + rot) & 31;
          int32_t rotated_top_k = __shfl_sync(0xFFFFFFFF, my_top_k_token, src_lane);
          int rotated_top_k_idx = top_k_base + src_lane;

          if (my_buffer_token >= 0 && my_buffer_token == rotated_top_k && rotated_top_k_idx < NUM_TOP_K) {
            my_found_top_k_idx = rotated_top_k_idx;
          }
        }
      }
    }

    // Record hits
    if (my_found_top_k_idx >= 0) {
      s_top_k_tokens[my_found_top_k_idx] = TOKEN_HIT;
      top_k_device_locs[my_found_top_k_idx] = device_buffer_locs[buf_slot];
    }
    __syncthreads();

    // Count and collect evictable slots
    const bool is_evictable = has_valid_slot && (my_found_top_k_idx == -1);
    int local_evictable_offset = 0;
    
    if (has_valid_chunk) {
      const unsigned int evictable_mask = __ballot_sync(0xFFFFFFFF, is_evictable);
      local_evictable_offset = __popc(evictable_mask & lanes_before);
      if (lane_id == 0) {
        s_chunk_offset[chunk_idx + 1] = __popc(evictable_mask);
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      total_evictable = warp_inclusive_scan(
          s_chunk_offset, lane_id, iter * NUM_WARPS + 1, NUM_BUFFER_CHUNKS + 1, total_evictable);
    }
    __syncthreads();

    if (is_evictable) {
      int offset = s_chunk_offset[chunk_idx] + local_evictable_offset;
      if (offset < NUM_TOP_K) {
        s_evictable_slots[offset] = buf_slot;
      }
    }
    __syncthreads();
  }

  // ===== Phase 2: Find missed tokens and assign to evictable slots =====
  for (int i = tid; i < NUM_BUFFER_CHUNKS + 1; i += BLOCK_SIZE) {
    s_chunk_offset[i] = 0;
  }
  __syncthreads();

  constexpr int ITERATIONS_PER_WARP_TOKEN = (NUM_TOKEN_CHUNKS + NUM_WARPS - 1) / NUM_WARPS;
  
  for (int iter = 0; iter < ITERATIONS_PER_WARP_TOKEN; iter++) {
    const int chunk_idx = warp_id + iter * NUM_WARPS;
    const bool has_valid_chunk = chunk_idx < NUM_TOKEN_CHUNKS;
    const int my_token_idx = chunk_idx * WARP_SIZE + lane_id;
    const bool has_valid_token = has_valid_chunk && (my_token_idx < NUM_TOP_K);

    int32_t my_token = 0;
    bool is_miss = false;

    if (has_valid_token) {
      is_miss = (s_top_k_tokens[my_token_idx] != TOKEN_HIT);
      if (is_miss) {
        my_token = s_top_k_tokens[my_token_idx];
      }
    }

    const unsigned int miss_mask = __ballot_sync(0xFFFFFFFF, is_miss);
    int local_miss_offset = 0;
    
    if (has_valid_chunk) {
      local_miss_offset = __popc(miss_mask & lanes_before);
      if (lane_id == 0) {
        s_chunk_offset[chunk_idx + 1] = __popc(miss_mask);
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      s_total_misses = warp_inclusive_scan(
          s_chunk_offset, lane_id, iter * NUM_WARPS + 1, NUM_TOKEN_CHUNKS + 1, s_total_misses);
    }
    __syncthreads();

    if (is_miss) {
      int miss_offset = s_chunk_offset[chunk_idx] + local_miss_offset;
      int evict_slot = s_evictable_slots[miss_offset];
      s_missed_tokens[miss_offset] = my_token;
      top_k_device_locs[my_token_idx] = device_buffer_locs[evict_slot];
      device_buffer_tokens[evict_slot] = my_token;
    }
    __syncthreads();
  }

  // ===== Phase 3: Copy data for misses =====
  for (int miss_idx = warp_id; miss_idx < s_total_misses; miss_idx += NUM_WARPS) {
    const int32_t miss_token = s_missed_tokens[miss_idx];
    const int evict_slot = s_evictable_slots[miss_idx];

    const int64_t src_offset = host_cache_locs[miss_token] * item_size_bytes;
    const int64_t dst_offset = device_buffer_locs[evict_slot] * item_size_bytes;

    transfer_item_warp(
        lane_id,
        static_cast<const char*>(host_cache) + src_offset,
        static_cast<char*>(device_buffer) + dst_offset,
        item_size_bytes);
  }
}

// Launcher function
void load_cache_launcher(
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
    // Select appropriate template instantiation
    if (num_top_k <= 64 && hot_buffer_size <= 128) {
        load_cache_to_device_buffer<256, 64, 128><<<1, 256>>>(
            top_k_tokens.data_ptr<int32_t>(),
            device_buffer_tokens.data_ptr<int32_t>(),
            host_cache_locs.data_ptr<int64_t>(),
            device_buffer_locs.data_ptr<int64_t>(),
            host_cache.data_ptr<uint8_t>(),
            device_buffer.data_ptr<uint8_t>(),
            top_k_device_locs.data_ptr<int64_t>(),
            item_size_bytes
        );
    } else if (num_top_k <= 128 && hot_buffer_size <= 256) {
        load_cache_to_device_buffer<256, 128, 256><<<1, 256>>>(
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
        load_cache_to_device_buffer<256, 256, 512><<<1, 256>>>(
            top_k_tokens.data_ptr<int32_t>(),
            device_buffer_tokens.data_ptr<int32_t>(),
            host_cache_locs.data_ptr<int64_t>(),
            device_buffer_locs.data_ptr<int64_t>(),
            host_cache.data_ptr<uint8_t>(),
            device_buffer.data_ptr<uint8_t>(),
            top_k_device_locs.data_ptr<int64_t>(),
            item_size_bytes
        );
    } else if (num_top_k <= 512 && hot_buffer_size <= 1024) {
        load_cache_to_device_buffer<256, 512, 1024><<<1, 256>>>(
            top_k_tokens.data_ptr<int32_t>(),
            device_buffer_tokens.data_ptr<int32_t>(),
            host_cache_locs.data_ptr<int64_t>(),
            device_buffer_locs.data_ptr<int64_t>(),
            host_cache.data_ptr<uint8_t>(),
            device_buffer.data_ptr<uint8_t>(),
            top_k_device_locs.data_ptr<int64_t>(),
            item_size_bytes
        );
    } else if (num_top_k <= 1024 && hot_buffer_size <= 2048) {
        load_cache_to_device_buffer<256, 1024, 2048><<<1, 256>>>(
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
        // Large configuration
        load_cache_to_device_buffer<256, 2048, 4096><<<1, 256>>>(
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
void load_cache_launcher(
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
    """Compile and cache the CUDA module."""
    global _module
    if _module is None:
        _module = load_inline(
            name='test_load_cache_shuffle_rotation',
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=['load_cache_launcher'],
            verbose=False,
            extra_cuda_cflags=['-O3'],
        )
    return _module


class CacheManager:
    """CUDA cache manager using shuffle rotation kernel."""
    
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
        
        # Initialize device buffer tokens to -1 (empty)
        self.device_buffer_tokens = torch.full(
            (hot_buffer_size,), -1, dtype=torch.int32, device=device
        )
        # Device buffer locations: identity mapping
        self.device_buffer_locs = torch.arange(
            hot_buffer_size, dtype=torch.int64, device=device
        )
        # Host cache locations: identity mapping
        self.host_cache_locs = torch.arange(
            max_tokens, dtype=torch.int64, device=device
        )
        
        self._module = None
    
    def _ensure_compiled(self):
        if self._module is None:
            self._module = get_module()
    
    def reset(self):
        """Reset buffer to empty state."""
        self.device_buffer_tokens.fill_(-1)
    
    def process_topk(
        self,
        top_k_tokens: torch.Tensor,
        host_cache: torch.Tensor,
        device_buffer: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process top_k tokens: find hits, evict, and load misses.
        
        Args:
            top_k_tokens: [num_top_k] int32 tensor of token IDs
            host_cache: [max_tokens * item_size_bytes] uint8 tensor (pinned memory)
            device_buffer: [hot_buffer_size * item_size_bytes] uint8 tensor
            
        Returns:
            top_k_device_locs: [num_top_k] int64 tensor of device buffer locations
        """
        self._ensure_compiled()
        
        if top_k_tokens.dtype != torch.int32:
            top_k_tokens = top_k_tokens.to(torch.int32)
        
        top_k_device_locs = torch.zeros(
            self.num_top_k, dtype=torch.int64, device=self.device
        )
        
        self._module.load_cache_launcher(
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


class PythonReference:
    """Python reference implementation for correctness verification."""
    
    def __init__(self, max_tokens: int, hot_buffer_size: int, item_size_bytes: int):
        self.max_tokens = max_tokens
        self.hot_buffer_size = hot_buffer_size
        self.item_size_bytes = item_size_bytes
        
        # Buffer tokens: -1 means empty
        self.device_buffer_tokens = [-1] * hot_buffer_size
        # Identity mapping for locations
        self.device_buffer_locs = list(range(hot_buffer_size))
        self.host_cache_locs = list(range(max_tokens))
    
    def reset(self):
        self.device_buffer_tokens = [-1] * self.hot_buffer_size
    
    def process_topk(
        self, 
        top_k_tokens: list, 
        host_cache: bytes, 
        device_buffer: bytearray
    ) -> list:
        """
        Reference implementation of cache management.
        
        Returns:
            top_k_device_locs: list of device buffer locations for each top_k token
        """
        num_top_k = len(top_k_tokens)
        top_k_device_locs = [0] * num_top_k
        
        # Build set for fast lookup
        top_k_set = {token: idx for idx, token in enumerate(top_k_tokens)}
        
        # Phase 1: Scan buffer, find hits and evictable slots
        hit_slots = {}  # top_k_idx -> buffer_slot
        evictable_slots = []
        
        for slot, token in enumerate(self.device_buffer_tokens):
            if token in top_k_set:
                # Hit: this buffer slot contains a wanted token
                top_k_idx = top_k_set[token]
                hit_slots[top_k_idx] = slot
            else:
                # Evictable: slot is empty or contains unwanted token
                evictable_slots.append(slot)
        
        # Phase 2: Process hits and identify misses
        misses = []
        for i in range(num_top_k):
            if i in hit_slots:
                # Hit
                slot = hit_slots[i]
                top_k_device_locs[i] = self.device_buffer_locs[slot]
            else:
                # Miss
                misses.append(i)
        
        # Phase 3: Assign misses to evictable slots and copy data
        for miss_idx, top_k_idx in enumerate(misses):
            token = top_k_tokens[top_k_idx]
            evict_slot = evictable_slots[miss_idx]
            
            # Update buffer token
            self.device_buffer_tokens[evict_slot] = token
            
            # Set output location
            top_k_device_locs[top_k_idx] = self.device_buffer_locs[evict_slot]
            
            # Copy data from host cache to device buffer
            src_offset = self.host_cache_locs[token] * self.item_size_bytes
            dst_offset = self.device_buffer_locs[evict_slot] * self.item_size_bytes
            device_buffer[dst_offset:dst_offset + self.item_size_bytes] = \
                host_cache[src_offset:src_offset + self.item_size_bytes]
        
        return top_k_device_locs


class TestLoadCacheShuffleRotation:
    """Test suite for shuffle rotation kernel."""
    
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
        return {
            "max_tokens": 10000,
            "hot_buffer_size": 1024,
            "item_size_bytes": 64,
            "num_top_k": 512,
        }
    
    def create_test_data(self, config):
        """Create host cache and device buffer for testing."""
        max_tokens = config["max_tokens"]
        item_size = config["item_size_bytes"]
        buffer_size = config["hot_buffer_size"]
        
        # Create host cache with unique data per token
        # Pattern: token i has bytes [(i + j) % 256 for j in range(item_size)]
        host_cache_np = bytearray(max_tokens * item_size)
        for t in range(max_tokens):
            for j in range(item_size):
                host_cache_np[t * item_size + j] = (t + j) % 256
        
        # Create torch tensors
        host_cache = torch.tensor(list(host_cache_np), dtype=torch.uint8).pin_memory()
        device_buffer = torch.zeros(buffer_size * item_size, dtype=torch.uint8, device="cuda")
        
        return host_cache_np, host_cache, device_buffer
    
    def test_all_misses(self, small_config):
        """Test case: all tokens are cache misses (empty buffer)."""
        config = small_config
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        ref = PythonReference(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManager(
            config["max_tokens"], config["hot_buffer_size"],
            config["item_size_bytes"], config["num_top_k"]
        )
        
        # All new tokens
        tokens = list(range(config["num_top_k"]))
        top_k = torch.tensor(tokens, dtype=torch.int32, device="cuda")
        
        device_ref = bytearray(config["hot_buffer_size"] * config["item_size_bytes"])
        ref_locs = ref.process_topk(tokens, host_np, device_ref)
        cuda_locs = cuda.process_topk(top_k, host_torch, device_torch)
        torch.cuda.synchronize()
        
        assert cuda_locs.cpu().tolist() == ref_locs, "Location mismatch for all misses"
        
        # Verify buffer tokens updated
        expected_tokens = ref.device_buffer_tokens
        actual_tokens = cuda.device_buffer_tokens.cpu().tolist()
        for i, token in enumerate(tokens):
            assert token in actual_tokens, f"Token {token} not found in buffer"
    
    def test_all_hits(self, small_config):
        """Test case: all tokens are cache hits."""
        config = small_config
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        ref = PythonReference(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManager(
            config["max_tokens"], config["hot_buffer_size"],
            config["item_size_bytes"], config["num_top_k"]
        )
        
        tokens = list(range(config["num_top_k"]))
        top_k = torch.tensor(tokens, dtype=torch.int32, device="cuda")
        device_ref = bytearray(config["hot_buffer_size"] * config["item_size_bytes"])
        
        # First pass: load tokens
        ref.process_topk(tokens, host_np, device_ref)
        cuda.process_topk(top_k, host_torch, device_torch)
        torch.cuda.synchronize()
        
        # Second pass: all hits
        ref_locs = ref.process_topk(tokens, host_np, device_ref)
        cuda_locs = cuda.process_topk(top_k, host_torch, device_torch)
        torch.cuda.synchronize()
        
        assert cuda_locs.cpu().tolist() == ref_locs, "Location mismatch for all hits"
    
    def test_mixed_hits_misses(self, small_config):
        """Test case: mix of hits and misses."""
        config = small_config
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        ref = PythonReference(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManager(
            config["max_tokens"], config["hot_buffer_size"],
            config["item_size_bytes"], config["num_top_k"]
        )
        
        device_ref = bytearray(config["hot_buffer_size"] * config["item_size_bytes"])
        
        # First pass: load tokens 0-63
        tokens1 = list(range(config["num_top_k"]))
        top_k1 = torch.tensor(tokens1, dtype=torch.int32, device="cuda")
        ref.process_topk(tokens1, host_np, device_ref)
        cuda.process_topk(top_k1, host_torch, device_torch)
        torch.cuda.synchronize()
        
        # Second pass: half old (hits), half new (misses)
        half = config["num_top_k"] // 2
        tokens2 = list(range(half)) + list(range(100, 100 + half))
        top_k2 = torch.tensor(tokens2, dtype=torch.int32, device="cuda")
        
        ref_locs = ref.process_topk(tokens2, host_np, device_ref)
        cuda_locs = cuda.process_topk(top_k2, host_torch, device_torch)
        torch.cuda.synchronize()
        
        assert cuda_locs.cpu().tolist() == ref_locs, "Location mismatch for mixed hits/misses"
    
    def test_data_integrity(self, small_config):
        """Verify that copied data is correct."""
        config = small_config
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        cuda = CacheManager(
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
            
            assert actual == expected, f"Data mismatch for token {token}"
    
    def test_medium_config(self, medium_config):
        """Test with medium configuration (K=256, H=512)."""
        config = medium_config
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        ref = PythonReference(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManager(
            config["max_tokens"], config["hot_buffer_size"],
            config["item_size_bytes"], config["num_top_k"]
        )
        
        tokens = list(range(config["num_top_k"]))
        top_k = torch.tensor(tokens, dtype=torch.int32, device="cuda")
        device_ref = bytearray(config["hot_buffer_size"] * config["item_size_bytes"])
        
        ref_locs = ref.process_topk(tokens, host_np, device_ref)
        cuda_locs = cuda.process_topk(top_k, host_torch, device_torch)
        torch.cuda.synchronize()
        
        assert cuda_locs.cpu().tolist() == ref_locs, "Location mismatch for medium config"
    
    def test_large_config(self, large_config):
        """Test with large configuration (K=512, H=1024)."""
        config = large_config
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        ref = PythonReference(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManager(
            config["max_tokens"], config["hot_buffer_size"],
            config["item_size_bytes"], config["num_top_k"]
        )
        
        tokens = list(range(config["num_top_k"]))
        top_k = torch.tensor(tokens, dtype=torch.int32, device="cuda")
        device_ref = bytearray(config["hot_buffer_size"] * config["item_size_bytes"])
        
        ref_locs = ref.process_topk(tokens, host_np, device_ref)
        cuda_locs = cuda.process_topk(top_k, host_torch, device_torch)
        torch.cuda.synchronize()
        
        assert cuda_locs.cpu().tolist() == ref_locs, "Location mismatch for large config"
    
    def test_multiple_iterations(self, small_config):
        """Test multiple sequential process_topk calls."""
        config = small_config
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        ref = PythonReference(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManager(
            config["max_tokens"], config["hot_buffer_size"],
            config["item_size_bytes"], config["num_top_k"]
        )
        
        device_ref = bytearray(config["hot_buffer_size"] * config["item_size_bytes"])
        
        # Multiple iterations with shifting token windows
        for iteration in range(5):
            base = iteration * 20
            tokens = list(range(base, base + config["num_top_k"]))
            top_k = torch.tensor(tokens, dtype=torch.int32, device="cuda")
            
            ref_locs = ref.process_topk(tokens, host_np, device_ref)
            cuda_locs = cuda.process_topk(top_k, host_torch, device_torch)
            torch.cuda.synchronize()
            
            assert cuda_locs.cpu().tolist() == ref_locs, f"Mismatch at iteration {iteration}"
    
    def test_non_sequential_tokens(self, small_config):
        """Test with non-sequential token IDs."""
        config = small_config
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        ref = PythonReference(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManager(
            config["max_tokens"], config["hot_buffer_size"],
            config["item_size_bytes"], config["num_top_k"]
        )
        
        # Non-sequential tokens with gaps
        tokens = [i * 7 for i in range(config["num_top_k"])]
        top_k = torch.tensor(tokens, dtype=torch.int32, device="cuda")
        device_ref = bytearray(config["hot_buffer_size"] * config["item_size_bytes"])
        
        ref_locs = ref.process_topk(tokens, host_np, device_ref)
        cuda_locs = cuda.process_topk(top_k, host_torch, device_torch)
        torch.cuda.synchronize()
        
        assert cuda_locs.cpu().tolist() == ref_locs, "Mismatch for non-sequential tokens"
    
    def test_buffer_state_consistency(self, small_config):
        """Verify buffer state is consistent after operations."""
        config = small_config
        host_np, host_torch, device_torch = self.create_test_data(config)
        
        ref = PythonReference(
            config["max_tokens"], config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda = CacheManager(
            config["max_tokens"], config["hot_buffer_size"],
            config["item_size_bytes"], config["num_top_k"]
        )
        
        device_ref = bytearray(config["hot_buffer_size"] * config["item_size_bytes"])
        
        # Process multiple batches
        for batch in range(3):
            tokens = list(range(batch * 30, batch * 30 + config["num_top_k"]))
            top_k = torch.tensor(tokens, dtype=torch.int32, device="cuda")
            
            ref.process_topk(tokens, host_np, device_ref)
            cuda.process_topk(top_k, host_torch, device_torch)
            torch.cuda.synchronize()
        
        # Compare buffer states
        ref_tokens = set(t for t in ref.device_buffer_tokens if t >= 0)
        cuda_tokens = set(t.item() for t in cuda.device_buffer_tokens if t.item() >= 0)
        
        assert ref_tokens == cuda_tokens, "Buffer state mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
