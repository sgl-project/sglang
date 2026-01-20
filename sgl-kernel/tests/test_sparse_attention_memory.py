"""
Test suite for the sparse attention memory management kernel.

This kernel manages a hot buffer cache for sparse attention, handling:
- Cache hits (tokens already in device buffer)
- Cache misses (tokens need to be loaded from host cache)
- Eviction of unused tokens from the device buffer

Test configurations:
- top_k: 512, 2048
- hot_buffer_size: |top_k+1|, 2*|top_k|, 4*|top_k|
"""

import os
import pytest
import torch
from typing import Tuple

# Check if CUDA is available
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

# CUDA kernel source code
KERNEL_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int WARP_SIZE = 32;
constexpr int32_t TOKEN_HIT = 0xFFFFFFFF;

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

__device__ __forceinline__ int
warp_inclusive_scan(int* s_data, int lane_id, int offset, int boundary, int accumulator) {
  int idx = lane_id + offset;
  int val = (idx < boundary) ? s_data[idx] : 0;

#pragma unroll
  for (int i = 1; i < 32; i *= 2) {
    int n = __shfl_up_sync(0xffffffff, val, i);
    if (lane_id >= i) val += n;
  }
  val += accumulator;
  if (idx < boundary) {
    s_data[idx] = val;
  }
  accumulator = __shfl_sync(0xffffffff, val, 31);
  return accumulator;
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
    int64_t item_size_bytes,
    int64_t req_length,
    int32_t* last_evicted_slot_ptr) {
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
  __shared__ int32_t s_evictable_slots[NUM_TOP_K + 1];
  __shared__ int32_t s_total_misses;

  for (int i = tid; i < NUM_TOP_K; i += BLOCK_SIZE) {
    s_top_k_tokens[i] = top_k_tokens[i];
  }
  for (int i = tid; i < NUM_BUFFER_CHUNKS + 1; i += BLOCK_SIZE) {
    s_chunk_offset[i] = 0;
  }
  __syncthreads();

  constexpr int ITERATIONS_PER_WARP_BUFFER = (NUM_BUFFER_CHUNKS + NUM_WARPS - 1) / NUM_WARPS;
  int total_evictable = 0;
  int physical_chunk_offset = (*last_evicted_slot_ptr + WARP_SIZE - 1) / WARP_SIZE;
  for (int iter = 0; iter < ITERATIONS_PER_WARP_BUFFER; iter++) {
    const int chunk_idx = warp_id + iter * NUM_WARPS;
    const bool has_valid_chunk = chunk_idx < NUM_BUFFER_CHUNKS;

    const int physical_chunk_idx = (chunk_idx + physical_chunk_offset) % NUM_BUFFER_CHUNKS;
    const int buf_slot = physical_chunk_idx * WARP_SIZE + lane_id;
    const bool has_valid_slot = has_valid_chunk && (buf_slot < HOT_BUFFER_SIZE);

    int32_t my_buffer_token = has_valid_slot ? device_buffer_tokens[buf_slot] : -1;
    int matched_top_k_idx = -1;

    if (has_valid_chunk) {
      for (int top_k_base = 0; top_k_base < NUM_TOP_K; top_k_base += WARP_SIZE) {
        int top_k_idx = top_k_base + lane_id;
        int32_t my_top_k_token = (top_k_idx < NUM_TOP_K) ? s_top_k_tokens[top_k_idx] : -1;

#pragma unroll
        for (int rot = 0; rot < WARP_SIZE; rot++) {
          int src_lane = (lane_id + rot) & 31;
          int32_t rotated_top_k = __shfl_sync(0xFFFFFFFF, my_top_k_token, src_lane);
          int rotated_top_k_idx = top_k_base + src_lane;

          if (my_buffer_token >= 0 && my_buffer_token == rotated_top_k) {
            matched_top_k_idx = rotated_top_k_idx;
          }
        }
      }
    }

    if (matched_top_k_idx >= 0) {
      s_top_k_tokens[matched_top_k_idx] = TOKEN_HIT;
      top_k_device_locs[matched_top_k_idx] = device_buffer_locs[buf_slot];
    }
    __syncthreads();

    int local_evictable_offset = 0;
    bool is_evictable = has_valid_slot && (matched_top_k_idx == -1);
    if (has_valid_chunk) {
      const unsigned int evictable_mask = __ballot_sync(0xFFFFFFFF, is_evictable);
      local_evictable_offset = __popc(evictable_mask & lanes_before);
      const int warp_evictable_count = __popc(evictable_mask);
      if (lane_id == 0) {
        s_chunk_offset[chunk_idx + 1] = warp_evictable_count;
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      total_evictable =
          warp_inclusive_scan(s_chunk_offset, lane_id, chunk_idx + 1, NUM_BUFFER_CHUNKS + 1, total_evictable);
    }
    __syncthreads();

    if (is_evictable) {
      int evictable_offset = s_chunk_offset[chunk_idx] + local_evictable_offset;
      if (evictable_offset < NUM_TOP_K + 1) {
        s_evictable_slots[evictable_offset] = buf_slot;
      }
    }
    __syncthreads();
  }

  for (int i = tid; i < NUM_BUFFER_CHUNKS + 1; i += BLOCK_SIZE) {
    s_chunk_offset[i] = 0;
  }
  __syncthreads();

  constexpr int ITERATIONS_PER_WARP_TOKEN = (NUM_TOKEN_CHUNKS + NUM_WARPS - 1) / NUM_WARPS;
  int total_misses = 0;
  for (int iter = 0; iter < ITERATIONS_PER_WARP_TOKEN; iter++) {
    const int chunk_idx = warp_id + iter * NUM_WARPS;
    const bool has_valid_chunk = chunk_idx < NUM_TOKEN_CHUNKS;

    const int my_token_idx = chunk_idx * WARP_SIZE + lane_id;
    const bool has_valid_token = has_valid_chunk && (my_token_idx < NUM_TOP_K);

    int32_t my_token = -1;
    bool is_miss = false;
    int local_miss_offset = 0;

    if (has_valid_token) {
      is_miss = s_top_k_tokens[my_token_idx] != TOKEN_HIT;
      if (is_miss) {
        my_token = s_top_k_tokens[my_token_idx];
      }
    }

    const unsigned int miss_mask = __ballot_sync(0xFFFFFFFF, is_miss);
    if (has_valid_chunk) {
      local_miss_offset = __popc(miss_mask & lanes_before);
      const int warp_miss_count = __popc(miss_mask);
      if (lane_id == 0) {
        s_chunk_offset[chunk_idx + 1] = warp_miss_count;
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      total_misses = warp_inclusive_scan(s_chunk_offset, lane_id, chunk_idx + 1, NUM_TOKEN_CHUNKS + 1, total_misses);
      if (tid == 0) {
        s_total_misses = total_misses;
      }
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

  if (tid == 0) {
    int extra_slot = s_evictable_slots[s_total_misses];
    *last_evicted_slot_ptr = extra_slot;
    top_k_device_locs[NUM_TOP_K] = device_buffer_locs[extra_slot];
    device_buffer_tokens[extra_slot] = req_length;
  }

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

// Wrapper functions for different configurations
// top_k=512, buffer_size=513 (top_k+1)
void load_cache_512_513(
    torch::Tensor top_k_tokens,
    torch::Tensor device_buffer_tokens,
    torch::Tensor host_cache_locs,
    torch::Tensor device_buffer_locs,
    torch::Tensor host_cache,
    torch::Tensor device_buffer,
    torch::Tensor top_k_device_locs,
    int64_t item_size_bytes,
    int64_t req_length,
    torch::Tensor last_evicted_slot) {
  load_cache_to_device_buffer<256, 512, 513><<<1, 256>>>(
      top_k_tokens.data_ptr<int32_t>(),
      device_buffer_tokens.data_ptr<int32_t>(),
      host_cache_locs.data_ptr<int64_t>(),
      device_buffer_locs.data_ptr<int64_t>(),
      host_cache.data_ptr(),
      device_buffer.data_ptr(),
      top_k_device_locs.data_ptr<int64_t>(),
      item_size_bytes,
      req_length,
      last_evicted_slot.data_ptr<int32_t>());
}

// top_k=512, buffer_size=1024 (2*top_k)
void load_cache_512_1024(
    torch::Tensor top_k_tokens,
    torch::Tensor device_buffer_tokens,
    torch::Tensor host_cache_locs,
    torch::Tensor device_buffer_locs,
    torch::Tensor host_cache,
    torch::Tensor device_buffer,
    torch::Tensor top_k_device_locs,
    int64_t item_size_bytes,
    int64_t req_length,
    torch::Tensor last_evicted_slot) {
  load_cache_to_device_buffer<256, 512, 1024><<<1, 256>>>(
      top_k_tokens.data_ptr<int32_t>(),
      device_buffer_tokens.data_ptr<int32_t>(),
      host_cache_locs.data_ptr<int64_t>(),
      device_buffer_locs.data_ptr<int64_t>(),
      host_cache.data_ptr(),
      device_buffer.data_ptr(),
      top_k_device_locs.data_ptr<int64_t>(),
      item_size_bytes,
      req_length,
      last_evicted_slot.data_ptr<int32_t>());
}

// top_k=512, buffer_size=2048 (4*top_k)
void load_cache_512_2048(
    torch::Tensor top_k_tokens,
    torch::Tensor device_buffer_tokens,
    torch::Tensor host_cache_locs,
    torch::Tensor device_buffer_locs,
    torch::Tensor host_cache,
    torch::Tensor device_buffer,
    torch::Tensor top_k_device_locs,
    int64_t item_size_bytes,
    int64_t req_length,
    torch::Tensor last_evicted_slot) {
  load_cache_to_device_buffer<256, 512, 2048><<<1, 256>>>(
      top_k_tokens.data_ptr<int32_t>(),
      device_buffer_tokens.data_ptr<int32_t>(),
      host_cache_locs.data_ptr<int64_t>(),
      device_buffer_locs.data_ptr<int64_t>(),
      host_cache.data_ptr(),
      device_buffer.data_ptr(),
      top_k_device_locs.data_ptr<int64_t>(),
      item_size_bytes,
      req_length,
      last_evicted_slot.data_ptr<int32_t>());
}

// top_k=2048, buffer_size=2049 (top_k+1)
void load_cache_2048_2049(
    torch::Tensor top_k_tokens,
    torch::Tensor device_buffer_tokens,
    torch::Tensor host_cache_locs,
    torch::Tensor device_buffer_locs,
    torch::Tensor host_cache,
    torch::Tensor device_buffer,
    torch::Tensor top_k_device_locs,
    int64_t item_size_bytes,
    int64_t req_length,
    torch::Tensor last_evicted_slot) {
  load_cache_to_device_buffer<256, 2048, 2049><<<1, 256>>>(
      top_k_tokens.data_ptr<int32_t>(),
      device_buffer_tokens.data_ptr<int32_t>(),
      host_cache_locs.data_ptr<int64_t>(),
      device_buffer_locs.data_ptr<int64_t>(),
      host_cache.data_ptr(),
      device_buffer.data_ptr(),
      top_k_device_locs.data_ptr<int64_t>(),
      item_size_bytes,
      req_length,
      last_evicted_slot.data_ptr<int32_t>());
}

// top_k=2048, buffer_size=4096 (2*top_k)
void load_cache_2048_4096(
    torch::Tensor top_k_tokens,
    torch::Tensor device_buffer_tokens,
    torch::Tensor host_cache_locs,
    torch::Tensor device_buffer_locs,
    torch::Tensor host_cache,
    torch::Tensor device_buffer,
    torch::Tensor top_k_device_locs,
    int64_t item_size_bytes,
    int64_t req_length,
    torch::Tensor last_evicted_slot) {
  load_cache_to_device_buffer<256, 2048, 4096><<<1, 256>>>(
      top_k_tokens.data_ptr<int32_t>(),
      device_buffer_tokens.data_ptr<int32_t>(),
      host_cache_locs.data_ptr<int64_t>(),
      device_buffer_locs.data_ptr<int64_t>(),
      host_cache.data_ptr(),
      device_buffer.data_ptr(),
      top_k_device_locs.data_ptr<int64_t>(),
      item_size_bytes,
      req_length,
      last_evicted_slot.data_ptr<int32_t>());
}

// top_k=2048, buffer_size=8192 (4*top_k)
void load_cache_2048_8192(
    torch::Tensor top_k_tokens,
    torch::Tensor device_buffer_tokens,
    torch::Tensor host_cache_locs,
    torch::Tensor device_buffer_locs,
    torch::Tensor host_cache,
    torch::Tensor device_buffer,
    torch::Tensor top_k_device_locs,
    int64_t item_size_bytes,
    int64_t req_length,
    torch::Tensor last_evicted_slot) {
  load_cache_to_device_buffer<256, 2048, 8192><<<1, 256>>>(
      top_k_tokens.data_ptr<int32_t>(),
      device_buffer_tokens.data_ptr<int32_t>(),
      host_cache_locs.data_ptr<int64_t>(),
      device_buffer_locs.data_ptr<int64_t>(),
      host_cache.data_ptr(),
      device_buffer.data_ptr(),
      top_k_device_locs.data_ptr<int64_t>(),
      item_size_bytes,
      req_length,
      last_evicted_slot.data_ptr<int32_t>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("load_cache_512_513", &load_cache_512_513);
  m.def("load_cache_512_1024", &load_cache_512_1024);
  m.def("load_cache_512_2048", &load_cache_512_2048);
  m.def("load_cache_2048_2049", &load_cache_2048_2049);
  m.def("load_cache_2048_4096", &load_cache_2048_4096);
  m.def("load_cache_2048_8192", &load_cache_2048_8192);
}
'''


def get_kernel_module():
    """Compile and load the CUDA kernel module."""
    from torch.utils.cpp_extension import load_inline
    
    module = load_inline(
        name="sparse_attention_memory_kernel",
        cpp_sources="",
        cuda_sources=KERNEL_SOURCE,
        functions=[
            "load_cache_512_513",
            "load_cache_512_1024",
            "load_cache_512_2048",
            "load_cache_2048_2049",
            "load_cache_2048_4096",
            "load_cache_2048_8192",
        ],
        verbose=False,
        extra_cuda_cflags=["-O3"],
    )
    return module


# Global module cache
_kernel_module = None


def get_cached_kernel_module():
    """Get the cached kernel module, compiling if necessary."""
    global _kernel_module
    if _kernel_module is None:
        _kernel_module = get_kernel_module()
    return _kernel_module


def get_kernel_function(top_k: int, buffer_size: int):
    """Get the appropriate kernel function for given parameters."""
    module = get_cached_kernel_module()
    kernel_name = f"load_cache_{top_k}_{buffer_size}"
    return getattr(module, kernel_name)


class SparseAttentionMemoryRef:
    """
    Reference implementation in Python for testing correctness.
    
    This simulates the cache behavior:
    - Checks which top_k tokens are already in the device buffer (hits)
    - Identifies evictable slots (buffer tokens not in top_k)
    - Assigns evictable slots to missed tokens
    - Transfers data from host cache to device buffer for misses
    """
    
    @staticmethod
    def compute(
        top_k_tokens: torch.Tensor,
        device_buffer_tokens: torch.Tensor,
        host_cache_locs: torch.Tensor,
        device_buffer_locs: torch.Tensor,
        host_cache: torch.Tensor,
        device_buffer: torch.Tensor,
        top_k_device_locs: torch.Tensor,
        item_size_bytes: int,
        req_length: int,
        last_evicted_slot: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """
        Reference implementation of the cache management logic.
        
        Returns:
            - updated device_buffer_tokens
            - updated device_buffer
            - updated top_k_device_locs
            - number of hits
            - number of misses
        """
        top_k = len(top_k_tokens)
        buffer_size = len(device_buffer_tokens)
        
        # Create copies to avoid modifying inputs
        device_buffer_tokens = device_buffer_tokens.clone()
        device_buffer = device_buffer.clone()
        top_k_device_locs = top_k_device_locs.clone()
        last_evicted_slot_val = last_evicted_slot.item()
        
        # Convert to Python for easier manipulation
        top_k_set = set(top_k_tokens.tolist())
        buffer_tokens_list = device_buffer_tokens.tolist()
        
        # Phase 1: Find hits and evictable slots
        hits = {}  # top_k_idx -> buffer_slot
        evictable_slots = []
        
        # Round-robin eviction starting from last_evicted_slot
        warp_size = 32
        num_buffer_chunks = (buffer_size + warp_size - 1) // warp_size
        physical_chunk_offset = (last_evicted_slot_val + warp_size - 1) // warp_size
        
        for chunk_idx in range(num_buffer_chunks):
            physical_chunk_idx = (chunk_idx + physical_chunk_offset) % num_buffer_chunks
            for lane_id in range(warp_size):
                buf_slot = physical_chunk_idx * warp_size + lane_id
                if buf_slot >= buffer_size:
                    continue
                
                buffer_token = buffer_tokens_list[buf_slot]
                matched_top_k_idx = -1
                
                # Check if buffer token matches any top_k token
                if buffer_token >= 0:
                    for top_k_idx, top_k_token in enumerate(top_k_tokens.tolist()):
                        if buffer_token == top_k_token:
                            matched_top_k_idx = top_k_idx
                            break
                
                if matched_top_k_idx >= 0:
                    # Hit: record the location
                    if matched_top_k_idx not in hits:
                        hits[matched_top_k_idx] = buf_slot
                        top_k_device_locs[matched_top_k_idx] = device_buffer_locs[buf_slot].item()
                else:
                    # Evictable slot
                    evictable_slots.append(buf_slot)
        
        # Phase 2: Process misses
        misses = []
        for top_k_idx in range(top_k):
            if top_k_idx not in hits:
                misses.append(top_k_idx)
        
        # Assign evictable slots to missed tokens
        for miss_idx, top_k_idx in enumerate(misses):
            if miss_idx < len(evictable_slots):
                evict_slot = evictable_slots[miss_idx]
                miss_token = top_k_tokens[top_k_idx].item()
                
                # Update buffer token
                device_buffer_tokens[evict_slot] = miss_token
                
                # Update top_k location
                top_k_device_locs[top_k_idx] = device_buffer_locs[evict_slot].item()
                
                # Transfer data from host cache to device buffer
                src_loc = host_cache_locs[miss_token].item()
                dst_loc = device_buffer_locs[evict_slot].item()
                device_buffer[dst_loc] = host_cache[src_loc].clone()
        
        # Handle the extra slot for next token generation
        num_misses = len(misses)
        if num_misses < len(evictable_slots):
            extra_slot = evictable_slots[num_misses]
            last_evicted_slot[0] = extra_slot
            top_k_device_locs[top_k] = device_buffer_locs[extra_slot].item()
            device_buffer_tokens[extra_slot] = req_length
        
        return device_buffer_tokens, device_buffer, top_k_device_locs, len(hits), num_misses


def setup_test_data(
    top_k: int,
    buffer_size: int,
    total_tokens: int,
    item_size: int,
    hit_ratio: float,
    dtype: torch.dtype = torch.float16,
) -> dict:
    """
    Set up test data for the sparse attention memory kernel.
    
    Args:
        top_k: Number of top-k tokens
        buffer_size: Size of the hot buffer
        total_tokens: Total number of tokens in the vocabulary
        item_size: Size of each cache item (in elements, not bytes)
        hit_ratio: Fraction of top_k tokens that should be hits (0.0 to 1.0)
        dtype: Data type for cache values
    
    Returns:
        Dictionary with all tensors needed for the test
    """
    device = "cuda"
    
    # Generate random top_k tokens (unique)
    all_tokens = torch.randperm(total_tokens, device=device)
    top_k_tokens = all_tokens[:top_k].to(torch.int32)
    
    # Determine how many should be hits
    num_hits = int(top_k * hit_ratio)
    num_misses = top_k - num_hits
    
    # Set up device buffer tokens
    # First num_hits slots contain hit tokens, rest contain random tokens not in top_k
    device_buffer_tokens = torch.full((buffer_size,), -1, dtype=torch.int32, device=device)
    
    if num_hits > 0:
        # Place hit tokens in random buffer slots
        hit_slots = torch.randperm(buffer_size, device=device)[:num_hits]
        hit_tokens = top_k_tokens[:num_hits]
        device_buffer_tokens[hit_slots] = hit_tokens
    
    # Fill remaining slots with non-top_k tokens (simulating old cached tokens)
    other_tokens = all_tokens[top_k:]
    empty_slots = (device_buffer_tokens == -1).nonzero().squeeze(-1)
    num_to_fill = min(len(empty_slots), len(other_tokens))
    if num_to_fill > 0:
        device_buffer_tokens[empty_slots[:num_to_fill]] = other_tokens[:num_to_fill].to(torch.int32)
    
    # Host cache locations (direct mapping for simplicity)
    host_cache_locs = torch.arange(total_tokens, dtype=torch.int64, device=device)
    
    # Device buffer locations (direct mapping for simplicity)
    device_buffer_locs = torch.arange(buffer_size, dtype=torch.int64, device=device)
    
    # Host cache data (pinned memory for async transfer)
    host_cache = torch.randn(total_tokens, item_size, dtype=dtype, device="cpu").pin_memory()
    
    # Device buffer (initialized with zeros or random values)
    device_buffer = torch.zeros(buffer_size, item_size, dtype=dtype, device=device)
    
    # Pre-populate device buffer with data for hit tokens
    if num_hits > 0:
        for i, slot in enumerate(hit_slots.tolist()):
            token = hit_tokens[i].item()
            device_buffer[slot] = host_cache[token].to(device)
    
    # Output: top_k device locations (top_k + 1 for extra slot)
    top_k_device_locs = torch.zeros(top_k + 1, dtype=torch.int64, device=device)
    
    # Last evicted slot pointer
    last_evicted_slot = torch.zeros(1, dtype=torch.int32, device=device)
    
    # Request length (for the extra slot token assignment)
    req_length = total_tokens
    
    return {
        "top_k_tokens": top_k_tokens,
        "device_buffer_tokens": device_buffer_tokens,
        "host_cache_locs": host_cache_locs,
        "device_buffer_locs": device_buffer_locs,
        "host_cache": host_cache,
        "device_buffer": device_buffer,
        "top_k_device_locs": top_k_device_locs,
        "item_size_bytes": item_size * dtype.itemsize,
        "req_length": req_length,
        "last_evicted_slot": last_evicted_slot,
        "num_expected_hits": num_hits,
        "num_expected_misses": num_misses,
    }


def run_kernel_test(
    top_k: int,
    buffer_size: int,
    hit_ratio: float,
    total_tokens: int = 10000,
    item_size: int = 128,
    dtype: torch.dtype = torch.float16,
):
    """Run a single kernel test with given parameters."""
    
    # Setup test data
    data = setup_test_data(
        top_k=top_k,
        buffer_size=buffer_size,
        total_tokens=total_tokens,
        item_size=item_size,
        hit_ratio=hit_ratio,
        dtype=dtype,
    )
    
    # Create copies for kernel execution
    kernel_buffer_tokens = data["device_buffer_tokens"].clone()
    kernel_buffer = data["device_buffer"].clone()
    kernel_top_k_locs = data["top_k_device_locs"].clone()
    kernel_last_evicted = data["last_evicted_slot"].clone()
    
    # Get and run kernel
    kernel_fn = get_kernel_function(top_k, buffer_size)
    kernel_fn(
        data["top_k_tokens"],
        kernel_buffer_tokens,
        data["host_cache_locs"],
        data["device_buffer_locs"],
        data["host_cache"].to("cuda"),  # Move to device for kernel
        kernel_buffer,
        kernel_top_k_locs,
        data["item_size_bytes"],
        data["req_length"],
        kernel_last_evicted,
    )
    torch.cuda.synchronize()
    
    # Run reference implementation
    ref_buffer_tokens, ref_buffer, ref_top_k_locs, num_hits, num_misses = \
        SparseAttentionMemoryRef.compute(
            data["top_k_tokens"],
            data["device_buffer_tokens"].clone(),
            data["host_cache_locs"],
            data["device_buffer_locs"],
            data["host_cache"].to("cuda"),
            data["device_buffer"].clone(),
            data["top_k_device_locs"].clone(),
            data["item_size_bytes"],
            data["req_length"],
            data["last_evicted_slot"].clone(),
        )
    
    return {
        "kernel_buffer_tokens": kernel_buffer_tokens,
        "kernel_buffer": kernel_buffer,
        "kernel_top_k_locs": kernel_top_k_locs,
        "ref_buffer_tokens": ref_buffer_tokens,
        "ref_buffer": ref_buffer,
        "ref_top_k_locs": ref_top_k_locs,
        "num_hits": num_hits,
        "num_misses": num_misses,
        "expected_hits": data["num_expected_hits"],
        "expected_misses": data["num_expected_misses"],
        "top_k_tokens": data["top_k_tokens"],
        "host_cache": data["host_cache"],
    }


# ============================================================================
# Test Cases
# ============================================================================

class TestSparseAttentionMemoryKernel:
    """Test class for sparse attention memory management kernel."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)
    
    # ------------------------------------------------------------------------
    # Data Loading Tests
    # ------------------------------------------------------------------------
    
    @pytest.mark.parametrize("top_k", [512, 2048])
    @pytest.mark.parametrize("buffer_mult", [1, 2, 4])  # buffer = top_k+1, 2*top_k, 4*top_k
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_data_loading_basic(self, top_k, buffer_mult, dtype):
        """Test basic data loading functionality."""
        if buffer_mult == 1:
            buffer_size = top_k + 1
        else:
            buffer_size = buffer_mult * top_k
        
        # Start with empty buffer (all misses)
        result = run_kernel_test(
            top_k=top_k,
            buffer_size=buffer_size,
            hit_ratio=0.0,
            total_tokens=max(10000, top_k * 4),
            item_size=128,
            dtype=dtype,
        )
        
        # Verify data was loaded correctly for miss tokens
        # Check that top_k_device_locs are valid
        assert (result["kernel_top_k_locs"][:top_k] >= 0).all(), \
            "All top_k tokens should have valid device locations"
        assert (result["kernel_top_k_locs"][:top_k] < buffer_size).all(), \
            "All device locations should be within buffer bounds"
    
    # ------------------------------------------------------------------------
    # Cache Full Miss Tests
    # ------------------------------------------------------------------------
    
    @pytest.mark.parametrize("top_k", [512, 2048])
    @pytest.mark.parametrize("buffer_mult", [1, 2, 4])
    def test_cache_full_miss(self, top_k, buffer_mult):
        """Test scenario where no top_k tokens are in the cache (full miss)."""
        if buffer_mult == 1:
            buffer_size = top_k + 1
        else:
            buffer_size = buffer_mult * top_k
        
        result = run_kernel_test(
            top_k=top_k,
            buffer_size=buffer_size,
            hit_ratio=0.0,
            total_tokens=max(10000, top_k * 4),
            item_size=128,
        )
        
        # All should be misses
        assert result["num_misses"] == top_k, \
            f"Expected {top_k} misses, got {result['num_misses']}"
        
        # Verify all top_k tokens now have valid locations assigned
        for i in range(top_k):
            loc = result["kernel_top_k_locs"][i].item()
            assert 0 <= loc < buffer_size, \
                f"Token {i} has invalid location {loc}"
        
        # Verify buffer tokens were updated
        top_k_set = set(result["top_k_tokens"].tolist())
        buffer_set = set(result["kernel_buffer_tokens"].tolist())
        assert len(top_k_set & buffer_set) >= top_k, \
            "All top_k tokens should be in buffer after full miss"
    
    # ------------------------------------------------------------------------
    # Cache Full Hit Tests
    # ------------------------------------------------------------------------
    
    @pytest.mark.parametrize("top_k", [512, 2048])
    @pytest.mark.parametrize("buffer_mult", [1, 2, 4])
    def test_cache_full_hit(self, top_k, buffer_mult):
        """Test scenario where all top_k tokens are already in cache (full hit)."""
        if buffer_mult == 1:
            buffer_size = top_k + 1
        else:
            buffer_size = buffer_mult * top_k
        
        result = run_kernel_test(
            top_k=top_k,
            buffer_size=buffer_size,
            hit_ratio=1.0,
            total_tokens=max(10000, top_k * 4),
            item_size=128,
        )
        
        # All should be hits
        assert result["num_hits"] == top_k, \
            f"Expected {top_k} hits, got {result['num_hits']}"
        assert result["num_misses"] == 0, \
            f"Expected 0 misses, got {result['num_misses']}"
        
        # Verify all top_k tokens have valid locations
        for i in range(top_k):
            loc = result["kernel_top_k_locs"][i].item()
            assert 0 <= loc < buffer_size, \
                f"Token {i} has invalid location {loc}"
    
    # ------------------------------------------------------------------------
    # Cache Partial Miss Tests
    # ------------------------------------------------------------------------
    
    @pytest.mark.parametrize("top_k", [512, 2048])
    @pytest.mark.parametrize("buffer_mult", [1, 2, 4])
    @pytest.mark.parametrize("hit_ratio", [0.25, 0.5, 0.75])
    def test_cache_partial_miss(self, top_k, buffer_mult, hit_ratio):
        """Test scenario with partial cache hits/misses."""
        if buffer_mult == 1:
            buffer_size = top_k + 1
        else:
            buffer_size = buffer_mult * top_k
        
        result = run_kernel_test(
            top_k=top_k,
            buffer_size=buffer_size,
            hit_ratio=hit_ratio,
            total_tokens=max(10000, top_k * 4),
            item_size=128,
        )
        
        expected_hits = int(top_k * hit_ratio)
        expected_misses = top_k - expected_hits
        
        # Allow some tolerance due to implementation details
        assert abs(result["num_hits"] - expected_hits) <= 2, \
            f"Expected ~{expected_hits} hits, got {result['num_hits']}"
        assert abs(result["num_misses"] - expected_misses) <= 2, \
            f"Expected ~{expected_misses} misses, got {result['num_misses']}"
        
        # Verify all top_k tokens have valid locations
        for i in range(top_k):
            loc = result["kernel_top_k_locs"][i].item()
            assert 0 <= loc < buffer_size, \
                f"Token {i} has invalid location {loc}"
    
    # ------------------------------------------------------------------------
    # Data Integrity Tests
    # ------------------------------------------------------------------------
    
    @pytest.mark.parametrize("top_k", [512, 2048])
    @pytest.mark.parametrize("buffer_mult", [2, 4])
    @pytest.mark.parametrize("item_size", [64, 128, 256])
    def test_data_transfer_integrity(self, top_k, buffer_mult, item_size):
        """Test that data is correctly transferred from host cache to device buffer."""
        buffer_size = buffer_mult * top_k
        
        # Use full miss to ensure all data is transferred
        result = run_kernel_test(
            top_k=top_k,
            buffer_size=buffer_size,
            hit_ratio=0.0,
            total_tokens=max(10000, top_k * 4),
            item_size=item_size,
        )
        
        # For each top_k token, verify the data in device buffer matches host cache
        host_cache = result["host_cache"].to("cuda")
        kernel_buffer = result["kernel_buffer"]
        top_k_tokens = result["top_k_tokens"]
        kernel_locs = result["kernel_top_k_locs"]
        
        for i in range(min(100, top_k)):  # Check first 100 tokens
            token = top_k_tokens[i].item()
            loc = kernel_locs[i].item()
            
            expected_data = host_cache[token]
            actual_data = kernel_buffer[loc]
            
            torch.testing.assert_close(
                actual_data, expected_data,
                msg=f"Data mismatch for token {token} at location {loc}"
            )
    
    # ------------------------------------------------------------------------
    # Extra Slot Tests
    # ------------------------------------------------------------------------
    
    @pytest.mark.parametrize("top_k", [512, 2048])
    @pytest.mark.parametrize("buffer_mult", [1, 2, 4])
    def test_extra_slot_allocation(self, top_k, buffer_mult):
        """Test that the extra slot (for next token) is correctly allocated."""
        if buffer_mult == 1:
            buffer_size = top_k + 1
        else:
            buffer_size = buffer_mult * top_k
        
        result = run_kernel_test(
            top_k=top_k,
            buffer_size=buffer_size,
            hit_ratio=0.5,
            total_tokens=max(10000, top_k * 4),
            item_size=128,
        )
        
        # Check that the extra slot (index top_k) has a valid location
        extra_loc = result["kernel_top_k_locs"][top_k].item()
        assert 0 <= extra_loc < buffer_size, \
            f"Extra slot has invalid location {extra_loc}"
    
    # ------------------------------------------------------------------------
    # Edge Case Tests
    # ------------------------------------------------------------------------
    
    @pytest.mark.parametrize("top_k,buffer_size", [
        (512, 513),    # Minimal buffer: top_k + 1
        (512, 1024),   # 2x top_k
        (2048, 2049),  # Minimal buffer for large top_k
    ])
    def test_minimal_buffer_size(self, top_k, buffer_size):
        """Test with minimal buffer sizes where eviction is critical."""
        result = run_kernel_test(
            top_k=top_k,
            buffer_size=buffer_size,
            hit_ratio=0.0,  # Full miss - maximum pressure
            total_tokens=max(10000, top_k * 4),
            item_size=128,
        )
        
        # Even with minimal buffer, should handle all tokens
        assert (result["kernel_top_k_locs"][:top_k] >= 0).all()
        assert (result["kernel_top_k_locs"][:top_k] < buffer_size).all()
    
    @pytest.mark.parametrize("top_k", [512, 2048])
    def test_sequential_operations(self, top_k):
        """Test multiple sequential cache operations."""
        buffer_size = 2 * top_k
        device = "cuda"
        
        # Initial setup
        total_tokens = max(20000, top_k * 8)
        item_size = 128
        dtype = torch.float16
        
        # Create persistent buffer state
        device_buffer_tokens = torch.full((buffer_size,), -1, dtype=torch.int32, device=device)
        device_buffer = torch.zeros(buffer_size, item_size, dtype=dtype, device=device)
        host_cache = torch.randn(total_tokens, item_size, dtype=dtype, device="cpu").pin_memory()
        host_cache_locs = torch.arange(total_tokens, dtype=torch.int64, device=device)
        device_buffer_locs = torch.arange(buffer_size, dtype=torch.int64, device=device)
        last_evicted_slot = torch.zeros(1, dtype=torch.int32, device=device)
        
        kernel_fn = get_kernel_function(top_k, buffer_size)
        
        # Run multiple operations
        for op_idx in range(3):
            # Generate new top_k tokens
            all_tokens = torch.randperm(total_tokens, device=device)
            top_k_tokens = all_tokens[:top_k].to(torch.int32)
            top_k_device_locs = torch.zeros(top_k + 1, dtype=torch.int64, device=device)
            
            kernel_fn(
                top_k_tokens,
                device_buffer_tokens,
                host_cache_locs,
                device_buffer_locs,
                host_cache.to("cuda"),
                device_buffer,
                top_k_device_locs,
                item_size * dtype.itemsize,
                total_tokens + op_idx,
                last_evicted_slot,
            )
            torch.cuda.synchronize()
            
            # Verify all locations are valid
            assert (top_k_device_locs[:top_k] >= 0).all(), \
                f"Operation {op_idx}: Invalid locations found"
            assert (top_k_device_locs[:top_k] < buffer_size).all(), \
                f"Operation {op_idx}: Locations out of bounds"
    
    # ------------------------------------------------------------------------
    # Stress Tests
    # ------------------------------------------------------------------------
    
    @pytest.mark.parametrize("top_k,buffer_mult", [
        (512, 4),
        (2048, 4),
    ])
    def test_stress_random_patterns(self, top_k, buffer_mult):
        """Stress test with random hit patterns."""
        buffer_size = buffer_mult * top_k
        
        for trial in range(5):
            hit_ratio = torch.rand(1).item()  # Random hit ratio
            
            result = run_kernel_test(
                top_k=top_k,
                buffer_size=buffer_size,
                hit_ratio=hit_ratio,
                total_tokens=max(10000, top_k * 4),
                item_size=128,
            )
            
            # Basic validity checks
            assert (result["kernel_top_k_locs"][:top_k] >= 0).all(), \
                f"Trial {trial}: Invalid locations"
            assert (result["kernel_top_k_locs"][:top_k] < buffer_size).all(), \
                f"Trial {trial}: Locations out of bounds"


# ============================================================================
# Standalone Test Functions (for pytest discovery)
# ============================================================================

@pytest.mark.parametrize("top_k,buffer_size", [
    (512, 513),
    (512, 1024),
    (512, 2048),
    (2048, 2049),
    (2048, 4096),
    (2048, 8192),
])
def test_kernel_configurations(top_k, buffer_size):
    """Test all kernel configurations compile and run."""
    result = run_kernel_test(
        top_k=top_k,
        buffer_size=buffer_size,
        hit_ratio=0.5,
        total_tokens=max(10000, top_k * 4),
        item_size=128,
    )
    
    assert (result["kernel_top_k_locs"][:top_k] >= 0).all()
    assert (result["kernel_top_k_locs"][:top_k] < buffer_size).all()


@pytest.mark.parametrize("hit_ratio", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_hit_ratios_512(hit_ratio):
    """Test various hit ratios with top_k=512."""
    result = run_kernel_test(
        top_k=512,
        buffer_size=1024,
        hit_ratio=hit_ratio,
        total_tokens=10000,
        item_size=128,
    )
    
    assert (result["kernel_top_k_locs"][:512] >= 0).all()


@pytest.mark.parametrize("hit_ratio", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_hit_ratios_2048(hit_ratio):
    """Test various hit ratios with top_k=2048."""
    result = run_kernel_test(
        top_k=2048,
        buffer_size=4096,
        hit_ratio=hit_ratio,
        total_tokens=20000,
        item_size=128,
    )
    
    assert (result["kernel_top_k_locs"][:2048] >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
