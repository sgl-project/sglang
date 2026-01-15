"""
Test script for batched load_cache_to_device_buffer CUDA kernel.

Multiple requests processed in parallel:
- Each block handles one request
- Per-request: top_k_tokens, device_buffer_tokens, token_residence_mapping, 
               host_cache_locs, device_buffer_locs, top_k_device_locs
- Shared: host_cache, device_buffer
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

/**
 * Batched kernel: each block handles one request
 * 
 * Memory layout (2D contiguous arrays):
 *   top_k_tokens:           [num_requests, NUM_TOP_K]
 *   device_buffer_tokens:   [num_requests, HOT_BUFFER_SIZE]
 *   token_residence_mapping:[num_requests, max_tokens]
 *   host_cache_locs:        [num_requests, max_tokens]
 *   device_buffer_locs:     [num_requests, HOT_BUFFER_SIZE]
 *   top_k_device_locs:      [num_requests, NUM_TOP_K]
 * 
 * Shared across requests:
 *   host_cache:             [total_cache_entries, item_size_bytes]
 *   device_buffer:          [total_buffer_entries, item_size_bytes]
 */
template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE>
__global__ void load_cache_to_device_buffer_batched(
    const uint32_t* __restrict__ top_k_tokens,           // [num_requests, NUM_TOP_K]
    uint32_t* __restrict__ device_buffer_tokens,         // [num_requests, HOT_BUFFER_SIZE]
    uint16_t* __restrict__ token_residence_mapping,      // [num_requests, max_tokens]
    const int64_t* __restrict__ host_cache_locs,         // [num_requests, max_tokens]
    const int64_t* __restrict__ device_buffer_locs,      // [num_requests, HOT_BUFFER_SIZE]
    const void* __restrict__ host_cache,                 // Shared
    void* __restrict__ device_buffer,                    // Shared
    int64_t* __restrict__ top_k_device_locs,             // [num_requests, NUM_TOP_K]
    int32_t max_tokens,
    int64_t item_size_bytes,
    int32_t num_requests
) {
    // Each block handles one request
    const int request_id = blockIdx.x;
    if (request_id >= num_requests) return;
    
    // Calculate offsets for this request's data
    const uint32_t* my_top_k_tokens = top_k_tokens + request_id * NUM_TOP_K;
    uint32_t* my_device_buffer_tokens = device_buffer_tokens + request_id * HOT_BUFFER_SIZE;
    uint16_t* my_token_residence_mapping = token_residence_mapping + request_id * max_tokens;
    const int64_t* my_host_cache_locs = host_cache_locs + request_id * max_tokens;
    const int64_t* my_device_buffer_locs = device_buffer_locs + request_id * HOT_BUFFER_SIZE;
    int64_t* my_top_k_device_locs = top_k_device_locs + request_id * NUM_TOP_K;
    
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    constexpr int NUM_CHUNKS = (NUM_TOP_K + WARP_SIZE - 1) / WARP_SIZE;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const unsigned int lanes_before = (1u << lane_id) - 1;

    __shared__ bool s_protected_bitmap[HOT_BUFFER_SIZE];
    __shared__ int32_t s_chunk_miss_offset[NUM_CHUNKS + 1];
    __shared__ int32_t s_missed_tokens[NUM_TOP_K];
    __shared__ int16_t s_missed_tokens_idx[NUM_TOP_K];
    __shared__ int32_t s_evictable_slots[NUM_TOP_K];
    __shared__ int32_t s_total_misses;

    if (tid == 0) {
        s_total_misses = 0;
    }
    for (int i = tid; i < HOT_BUFFER_SIZE; i += BLOCK_SIZE) {
        s_protected_bitmap[i] = false;
    }
    for (int i = tid; i < NUM_CHUNKS + 1; i += BLOCK_SIZE) {
        s_chunk_miss_offset[i] = 0;
    }
    __syncthreads();

    // ===== Phase 1: Hit and Miss Detection ====
    constexpr int ITERATIONS_PER_WARP = (NUM_CHUNKS + NUM_WARPS - 1) / NUM_WARPS;
    for (int iter = 0; iter < ITERATIONS_PER_WARP; iter++) {
        int chunk_idx = warp_id + iter * NUM_WARPS;
        bool has_valid_chunk = chunk_idx < NUM_CHUNKS;

        const int chunk_token_start = chunk_idx * WARP_SIZE;
        const int my_token_idx = chunk_token_start + lane_id;
        const bool has_valid_token = has_valid_chunk && (my_token_idx < NUM_TOP_K);

        int32_t my_token = 0;
        bool is_hit = false;
        bool is_miss = false;
        int32_t my_device_slot = -1;
        int local_miss_offset = 0;

        if (has_valid_token) {
            my_token = my_top_k_tokens[my_token_idx];
            uint16_t slot = my_token_residence_mapping[my_token];
            is_hit = (slot != NOT_PRESENT);

            if (is_hit) {
                my_device_slot = static_cast<int32_t>(slot);
                s_protected_bitmap[my_device_slot] = true;
                my_top_k_device_locs[my_token_idx] = my_device_buffer_locs[my_device_slot];
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
                s_chunk_miss_offset, lane_id, iter * NUM_WARPS + 1, NUM_CHUNKS + 1, s_total_misses);
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
            const int32_t old_token = my_device_buffer_tokens[evict_slot];
            
            if (old_token >= 0 && old_token < max_tokens) {
                my_token_residence_mapping[old_token] = NOT_PRESENT;
            }
            my_token_residence_mapping[miss_token] = static_cast<uint16_t>(evict_slot);
            my_device_buffer_tokens[evict_slot] = miss_token;
            my_top_k_device_locs[top_k_idx] = my_device_buffer_locs[evict_slot];
            
            // Use per-request host_cache_locs and device_buffer_locs
            // but shared host_cache and device_buffer
            src_offset = my_host_cache_locs[miss_token] * item_size_bytes;
            dst_offset = my_device_buffer_locs[evict_slot] * item_size_bytes;
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

// Launcher for batched kernel
void load_cache_batched_launcher(
    torch::Tensor top_k_tokens,              // [num_requests, NUM_TOP_K]
    torch::Tensor device_buffer_tokens,      // [num_requests, HOT_BUFFER_SIZE]
    torch::Tensor token_residence_mapping,   // [num_requests, max_tokens]
    torch::Tensor host_cache_locs,           // [num_requests, max_tokens]
    torch::Tensor device_buffer_locs,        // [num_requests, HOT_BUFFER_SIZE]
    torch::Tensor host_cache,                // Shared
    torch::Tensor device_buffer,             // Shared
    torch::Tensor top_k_device_locs,         // [num_requests, NUM_TOP_K]
    int64_t max_tokens,
    int64_t item_size_bytes,
    int64_t num_top_k,
    int64_t hot_buffer_size
) {
    const int num_requests = top_k_tokens.size(0);
    
    // Select kernel based on sizes
    if (num_top_k <= 64 && hot_buffer_size <= 128) {
        load_cache_to_device_buffer_batched<256, 64, 128><<<num_requests, 256>>>(
            reinterpret_cast<uint32_t*>(top_k_tokens.data_ptr<int32_t>()),
            reinterpret_cast<uint32_t*>(device_buffer_tokens.data_ptr<int32_t>()),
            reinterpret_cast<uint16_t*>(token_residence_mapping.data_ptr<int16_t>()),
            host_cache_locs.data_ptr<int64_t>(),
            device_buffer_locs.data_ptr<int64_t>(),
            host_cache.data_ptr<uint8_t>(),
            device_buffer.data_ptr<uint8_t>(),
            top_k_device_locs.data_ptr<int64_t>(),
            static_cast<int32_t>(max_tokens),
            item_size_bytes,
            num_requests
        );
    } else {
        load_cache_to_device_buffer_batched<256, 256, 512><<<num_requests, 256>>>(
            reinterpret_cast<uint32_t*>(top_k_tokens.data_ptr<int32_t>()),
            reinterpret_cast<uint32_t*>(device_buffer_tokens.data_ptr<int32_t>()),
            reinterpret_cast<uint16_t*>(token_residence_mapping.data_ptr<int16_t>()),
            host_cache_locs.data_ptr<int64_t>(),
            device_buffer_locs.data_ptr<int64_t>(),
            host_cache.data_ptr<uint8_t>(),
            device_buffer.data_ptr<uint8_t>(),
            top_k_device_locs.data_ptr<int64_t>(),
            static_cast<int32_t>(max_tokens),
            item_size_bytes,
            num_requests
        );
    }
}
'''

CPP_SOURCE = r'''
void load_cache_batched_launcher(
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
    int64_t num_top_k,
    int64_t hot_buffer_size
);
'''

_module = None

def get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name='test_load_cache_batched',
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=['load_cache_batched_launcher'],
            verbose=False,
            extra_cuda_cflags=['-O3'],
        )
    return _module


class BatchedCacheManager:
    """Manages batched sparse KV cache for multiple requests."""
    
    NOT_PRESENT = -1  # 0xFFFF as int16
    
    def __init__(
        self,
        num_requests: int,
        max_tokens: int,
        hot_buffer_size: int,
        item_size_bytes: int,
        num_top_k: int,
        device: str = "cuda",
    ):
        self.num_requests = num_requests
        self.max_tokens = max_tokens
        self.hot_buffer_size = hot_buffer_size
        self.item_size_bytes = item_size_bytes
        self.num_top_k = num_top_k
        self.device = device
        
        # Per-request state: [num_requests, ...]
        self.token_residence_mapping = torch.full(
            (num_requests, max_tokens), self.NOT_PRESENT, dtype=torch.int16, device=device
        )
        self.device_buffer_tokens = torch.full(
            (num_requests, hot_buffer_size), self.NOT_PRESENT, dtype=torch.int32, device=device
        )
        
        # Per-request location mappings
        # Each request has its own mapping from token -> host cache location
        # and slot -> device buffer location
        self.host_cache_locs = torch.zeros(
            (num_requests, max_tokens), dtype=torch.int64, device=device
        )
        self.device_buffer_locs = torch.zeros(
            (num_requests, hot_buffer_size), dtype=torch.int64, device=device
        )
        
        # Initialize with default identity mappings (can be customized per request)
        for r in range(num_requests):
            self.host_cache_locs[r] = torch.arange(max_tokens, dtype=torch.int64, device=device)
            self.device_buffer_locs[r] = torch.arange(hot_buffer_size, dtype=torch.int64, device=device)
        
        self._module = None
    
    def _ensure_compiled(self):
        if self._module is None:
            self._module = get_module()
    
    def reset(self):
        """Reset all state."""
        self.token_residence_mapping.fill_(self.NOT_PRESENT)
        self.device_buffer_tokens.fill_(self.NOT_PRESENT)
    
    def reset_request(self, request_id: int):
        """Reset state for a specific request."""
        self.token_residence_mapping[request_id].fill_(self.NOT_PRESENT)
        self.device_buffer_tokens[request_id].fill_(self.NOT_PRESENT)
    
    def set_host_cache_locs(self, request_id: int, locs: torch.Tensor):
        """Set host cache location mapping for a specific request."""
        self.host_cache_locs[request_id] = locs.to(self.device)
    
    def set_device_buffer_locs(self, request_id: int, locs: torch.Tensor):
        """Set device buffer location mapping for a specific request."""
        self.device_buffer_locs[request_id] = locs.to(self.device)
    
    def process_batch(
        self,
        top_k_tokens: torch.Tensor,  # [num_requests, num_top_k]
        host_cache: torch.Tensor,
        device_buffer: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process a batch of requests.
        
        Args:
            top_k_tokens: [num_requests, num_top_k] tokens for each request
            host_cache: Shared CPU cache
            device_buffer: Shared GPU buffer
            
        Returns:
            top_k_device_locs: [num_requests, num_top_k] device locations
        """
        self._ensure_compiled()
        
        if top_k_tokens.dtype != torch.int32:
            top_k_tokens = top_k_tokens.to(torch.int32)
        
        batch_size = top_k_tokens.size(0)
        assert batch_size <= self.num_requests, f"Batch size {batch_size} > max {self.num_requests}"
        
        top_k_device_locs = torch.zeros(
            (batch_size, self.num_top_k), dtype=torch.int64, device=self.device
        )
        
        self._module.load_cache_batched_launcher(
            top_k_tokens,
            self.device_buffer_tokens[:batch_size],
            self.token_residence_mapping[:batch_size],
            self.host_cache_locs[:batch_size],
            self.device_buffer_locs[:batch_size],
            host_cache,
            device_buffer,
            top_k_device_locs,
            self.max_tokens,
            self.item_size_bytes,
            self.num_top_k,
            self.hot_buffer_size,
        )
        
        return top_k_device_locs


class PythonBatchedReference:
    """Python reference for batched cache management."""
    
    NOT_PRESENT = -1
    
    def __init__(self, num_requests: int, max_tokens: int, hot_buffer_size: int, item_size_bytes: int):
        self.num_requests = num_requests
        self.max_tokens = max_tokens
        self.hot_buffer_size = hot_buffer_size
        self.item_size_bytes = item_size_bytes
        
        # Per-request state
        self.token_residence_mapping = [[self.NOT_PRESENT] * max_tokens for _ in range(num_requests)]
        self.device_buffer_tokens = [[self.NOT_PRESENT] * hot_buffer_size for _ in range(num_requests)]
        self.host_cache_locs = [list(range(max_tokens)) for _ in range(num_requests)]
        self.device_buffer_locs = [list(range(hot_buffer_size)) for _ in range(num_requests)]
    
    def reset(self):
        for r in range(self.num_requests):
            self.token_residence_mapping[r] = [self.NOT_PRESENT] * self.max_tokens
            self.device_buffer_tokens[r] = [self.NOT_PRESENT] * self.hot_buffer_size
    
    def process_request(self, request_id: int, top_k_tokens: list, 
                       host_cache: bytes, device_buffer: bytearray) -> list:
        """Process a single request."""
        top_k_device_locs = [0] * len(top_k_tokens)
        
        # Phase 1: Identify hits and misses
        hits = []
        misses = []
        protected_slots = set()
        
        for i, token in enumerate(top_k_tokens):
            slot = self.token_residence_mapping[request_id][token]
            if slot != self.NOT_PRESENT:
                hits.append((i, token, slot))
                top_k_device_locs[i] = self.device_buffer_locs[request_id][slot]
                protected_slots.add(slot)
            else:
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
            old_token = self.device_buffer_tokens[request_id][evict_slot]
            
            if old_token != self.NOT_PRESENT and old_token >= 0:
                self.token_residence_mapping[request_id][old_token] = self.NOT_PRESENT
            
            self.token_residence_mapping[request_id][token] = evict_slot
            self.device_buffer_tokens[request_id][evict_slot] = token
            top_k_device_locs[top_k_idx] = self.device_buffer_locs[request_id][evict_slot]
            
            # Copy data (using shared host_cache and device_buffer)
            src_offset = self.host_cache_locs[request_id][token] * self.item_size_bytes
            dst_offset = self.device_buffer_locs[request_id][evict_slot] * self.item_size_bytes
            device_buffer[dst_offset:dst_offset + self.item_size_bytes] = \
                host_cache[src_offset:src_offset + self.item_size_bytes]
        
        return top_k_device_locs
    
    def process_batch(self, top_k_tokens_batch: list, host_cache: bytes, 
                     device_buffer: bytearray) -> list:
        """Process a batch of requests."""
        results = []
        for r, top_k_tokens in enumerate(top_k_tokens_batch):
            locs = self.process_request(r, top_k_tokens, host_cache, device_buffer)
            results.append(locs)
        return results


class TestBatchedLoadCache:
    """Test suite for batched load_cache_to_device_buffer kernel."""
    
    @pytest.fixture
    def config(self):
        return {
            "num_requests": 4,
            "max_tokens": 500,
            "hot_buffer_size": 128,
            "item_size_bytes": 64,
            "num_top_k": 64,
        }
    
    def create_test_data(self, config, total_cache_size=None):
        """Create shared host cache and device buffer."""
        if total_cache_size is None:
            total_cache_size = config["max_tokens"]
        
        # Shared host cache with unique data per entry
        host_cache_np = bytearray(total_cache_size * config["item_size_bytes"])
        for entry in range(total_cache_size):
            offset = entry * config["item_size_bytes"]
            for i in range(config["item_size_bytes"]):
                host_cache_np[offset + i] = (entry + i) % 256
        
        host_cache_torch = torch.tensor(
            list(host_cache_np), dtype=torch.uint8, device="cpu"
        ).pin_memory()
        
        # Shared device buffer
        device_buffer_torch = torch.zeros(
            total_cache_size * config["item_size_bytes"], 
            dtype=torch.uint8, device="cuda"
        )
        
        return host_cache_np, host_cache_torch, device_buffer_torch
    
    def test_batch_all_misses(self, config):
        """Test batch with all misses for all requests."""
        host_cache_np, host_cache_torch, device_buffer_torch = self.create_test_data(config)
        
        ref = PythonBatchedReference(
            config["num_requests"], config["max_tokens"],
            config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda_mgr = BatchedCacheManager(
            config["num_requests"], config["max_tokens"],
            config["hot_buffer_size"], config["item_size_bytes"],
            config["num_top_k"]
        )
        
        # Create batch of top_k tokens (each request has different tokens)
        top_k_batch = []
        for r in range(config["num_requests"]):
            tokens = list(range(r * config["num_top_k"], (r + 1) * config["num_top_k"]))
            top_k_batch.append(tokens)
        
        top_k_torch = torch.tensor(top_k_batch, dtype=torch.int32, device="cuda")
        
        # Reference
        device_buffer_ref = bytearray(len(host_cache_np))
        ref_locs_batch = ref.process_batch(top_k_batch, host_cache_np, device_buffer_ref)
        
        # CUDA
        cuda_locs = cuda_mgr.process_batch(top_k_torch, host_cache_torch, device_buffer_torch)
        torch.cuda.synchronize()
        
        # Verify
        for r in range(config["num_requests"]):
            cuda_locs_r = cuda_locs[r].cpu().tolist()
            assert cuda_locs_r == ref_locs_batch[r], f"Request {r} locations mismatch"
    
    def test_batch_all_hits(self, config):
        """Test batch with all hits for all requests."""
        host_cache_np, host_cache_torch, device_buffer_torch = self.create_test_data(config)
        
        ref = PythonBatchedReference(
            config["num_requests"], config["max_tokens"],
            config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda_mgr = BatchedCacheManager(
            config["num_requests"], config["max_tokens"],
            config["hot_buffer_size"], config["item_size_bytes"],
            config["num_top_k"]
        )
        
        device_buffer_ref = bytearray(len(host_cache_np))
        
        # First pass - load tokens
        top_k_batch = []
        for r in range(config["num_requests"]):
            tokens = list(range(r * config["num_top_k"], (r + 1) * config["num_top_k"]))
            top_k_batch.append(tokens)
        
        top_k_torch = torch.tensor(top_k_batch, dtype=torch.int32, device="cuda")
        
        ref.process_batch(top_k_batch, host_cache_np, device_buffer_ref)
        cuda_mgr.process_batch(top_k_torch, host_cache_torch, device_buffer_torch)
        torch.cuda.synchronize()
        
        # Second pass - same tokens (all hits)
        ref_locs_batch = ref.process_batch(top_k_batch, host_cache_np, device_buffer_ref)
        cuda_locs = cuda_mgr.process_batch(top_k_torch, host_cache_torch, device_buffer_torch)
        torch.cuda.synchronize()
        
        for r in range(config["num_requests"]):
            cuda_locs_r = cuda_locs[r].cpu().tolist()
            assert cuda_locs_r == ref_locs_batch[r], f"Request {r} all-hits mismatch"
    
    def test_batch_mixed(self, config):
        """Test batch with mix of hits and misses."""
        host_cache_np, host_cache_torch, device_buffer_torch = self.create_test_data(config)
        
        ref = PythonBatchedReference(
            config["num_requests"], config["max_tokens"],
            config["hot_buffer_size"], config["item_size_bytes"]
        )
        cuda_mgr = BatchedCacheManager(
            config["num_requests"], config["max_tokens"],
            config["hot_buffer_size"], config["item_size_bytes"],
            config["num_top_k"]
        )
        
        device_buffer_ref = bytearray(len(host_cache_np))
        
        # First pass - load tokens 0-63 for each request
        top_k_batch1 = []
        for r in range(config["num_requests"]):
            # Use same token range for all requests (each request has its own state)
            tokens = list(range(config["num_top_k"]))
            top_k_batch1.append(tokens)
        
        top_k_torch1 = torch.tensor(top_k_batch1, dtype=torch.int32, device="cuda")
        ref.process_batch(top_k_batch1, host_cache_np, device_buffer_ref)
        cuda_mgr.process_batch(top_k_torch1, host_cache_torch, device_buffer_torch)
        torch.cuda.synchronize()
        
        # Second pass - half old (0-31), half new (100-131)
        # All within max_tokens=500
        top_k_batch2 = []
        half = config["num_top_k"] // 2
        for r in range(config["num_requests"]):
            old_tokens = list(range(half))  # 0-31 (hits)
            new_tokens = list(range(100, 100 + half))  # 100-131 (misses)
            top_k_batch2.append(old_tokens + new_tokens)
        
        top_k_torch2 = torch.tensor(top_k_batch2, dtype=torch.int32, device="cuda")
        
        ref_locs_batch = ref.process_batch(top_k_batch2, host_cache_np, device_buffer_ref)
        cuda_locs = cuda_mgr.process_batch(top_k_torch2, host_cache_torch, device_buffer_torch)
        torch.cuda.synchronize()
        
        for r in range(config["num_requests"]):
            cuda_locs_r = cuda_locs[r].cpu().tolist()
            assert cuda_locs_r == ref_locs_batch[r], f"Request {r} mixed test mismatch"
    
    def test_independent_requests(self, config):
        """Test that requests are independent (don't interfere with each other)."""
        host_cache_np, host_cache_torch, device_buffer_torch = self.create_test_data(config)
        
        cuda_mgr = BatchedCacheManager(
            config["num_requests"], config["max_tokens"],
            config["hot_buffer_size"], config["item_size_bytes"],
            config["num_top_k"]
        )
        
        # Process multiple batches, each request builds up its own cache
        for iteration in range(3):
            top_k_batch = []
            for r in range(config["num_requests"]):
                # Each request has different token ranges
                base = r * 100 + iteration * config["num_top_k"]
                tokens = list(range(base, base + config["num_top_k"]))
                top_k_batch.append(tokens)
            
            top_k_torch = torch.tensor(top_k_batch, dtype=torch.int32, device="cuda")
            cuda_locs = cuda_mgr.process_batch(top_k_torch, host_cache_torch, device_buffer_torch)
            torch.cuda.synchronize()
            
            # Verify each request got unique locations
            for r in range(config["num_requests"]):
                locs = cuda_locs[r].cpu().tolist()
                assert len(set(locs)) == len(locs), f"Request {r} has duplicate locations"
    
    def test_data_integrity_batched(self, config):
        """Verify data is correctly copied for all requests."""
        # Create larger shared buffers to avoid overlap between requests
        total_buffer_entries = config["num_requests"] * config["hot_buffer_size"]
        host_cache_np, host_cache_torch, device_buffer_torch = self.create_test_data(
            config, total_cache_size=total_buffer_entries
        )
        
        cuda_mgr = BatchedCacheManager(
            config["num_requests"], config["max_tokens"],
            config["hot_buffer_size"], config["item_size_bytes"],
            config["num_top_k"]
        )
        
        # Give each request a different region in the shared device_buffer
        # Request r uses device locations [r*hot_buffer_size, (r+1)*hot_buffer_size)
        for r in range(config["num_requests"]):
            offset = r * config["hot_buffer_size"]
            locs = torch.arange(offset, offset + config["hot_buffer_size"], 
                               dtype=torch.int64, device="cuda")
            cuda_mgr.device_buffer_locs[r] = locs
        
        # Each request loads same tokens (0 to num_top_k-1)
        # But writes to different device buffer regions
        top_k_batch = []
        for r in range(config["num_requests"]):
            tokens = list(range(config["num_top_k"]))
            top_k_batch.append(tokens)
        
        top_k_torch = torch.tensor(top_k_batch, dtype=torch.int32, device="cuda")
        cuda_locs = cuda_mgr.process_batch(top_k_torch, host_cache_torch, device_buffer_torch)
        torch.cuda.synchronize()
        
        # Verify data for each request
        device_buffer_cpu = device_buffer_torch.cpu().numpy()
        
        for r in range(config["num_requests"]):
            tokens = top_k_batch[r]
            locs = cuda_locs[r].cpu().tolist()
            
            for i, token in enumerate(tokens):
                loc = locs[i]
                offset = loc * config["item_size_bytes"]
                
                # Expected data based on host_cache_locs mapping (identity by default)
                host_loc = token  # Using default identity mapping
                expected = []
                for j in range(config["item_size_bytes"]):
                    expected.append((host_loc + j) % 256)
                
                actual = list(device_buffer_cpu[offset:offset + config["item_size_bytes"]])
                assert actual == expected, \
                    f"Request {r}, token {token}, loc {loc}: data mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
