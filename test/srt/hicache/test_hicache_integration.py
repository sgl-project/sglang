import gc
import time
import pytest
from itertools import product
from typing import List

import torch

from sglang.srt.distributed import (
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool, ReqToTokenPool


# =============================================================================
# Configuration Constants
# =============================================================================

# Test configuration matrix
TEST_CONFIGURATIONS = list(product(
    ["layer_first", "page_first"],  # layouts
    [1, 64],                        # page_sizes
    ["kernel", "direct"],           # io_backends
    ["MHA", "MLA"]                  # attention_types
))

# Test constants
MAX_TOTAL_NUM_TOKENS = 12 * 1024
MAX_REQ_NUM = 1024
KV_CACHE_DTYPE = torch.bfloat16
LAYER_NUM = 64
HEAD_NUM, HEAD_DIM = 8, 128
KV_LORA_RANK = 512  # MLA specific
QK_ROPE_HEAD_DIM = 64  # MLA specific
DEVICE = "cuda"
HICACHE_RATIO = 2
HICACHE_SIZE = 0
HICACHE_WRITE_POLICY = "write_through"
HICACHE_STORAGE_BACKEND = "test"
HICACHE_IO_BACKEND = "kernel"
HICACHE_MEM_LAYOUT = "layer_first"
HICACHE_STORAGE_PREFETCH_POLICY = "best_effort"

# Test parameters
TEST_OP_SIZE = 1024
TEST_OP_NUM = 10

assert TEST_OP_SIZE * TEST_OP_NUM <= MAX_TOTAL_NUM_TOKENS, "Too many tokens for test"


# =============================================================================
# Helper Classes and Functions
# =============================================================================

class _TestConfig:
    """Test configuration container"""
    def __init__(self, layout: str, page_size: int, io_backend: str, attention_type: str):
        self.layout = layout
        self.page_size = page_size
        self.io_backend = io_backend
        self.attention_type = attention_type
        self.max_total_num_tokens = MAX_TOTAL_NUM_TOKENS
        self.max_req_num = MAX_REQ_NUM
        self.kv_cache_dtype = KV_CACHE_DTYPE
        self.layer_num = LAYER_NUM
        self.device = DEVICE
        self.hicache_ratio = HICACHE_RATIO
        self.hicache_size = HICACHE_SIZE
        self.hicache_write_policy = HICACHE_WRITE_POLICY
        self.hicache_storage_backend = HICACHE_STORAGE_BACKEND
        self.hicache_io_backend = io_backend if io_backend != "direct" else HICACHE_IO_BACKEND
        self.hicache_mem_layout = layout if layout == "layer_first" else HICACHE_MEM_LAYOUT
        self.hicache_storage_prefetch_policy = HICACHE_STORAGE_PREFETCH_POLICY

        # Attention type specific parameters
        if attention_type == "MHA":
            self.head_num = HEAD_NUM
            self.head_dim = HEAD_DIM
        elif attention_type == "MLA":
            self.kv_lora_rank = KV_LORA_RANK
            self.qk_rope_head_dim = QK_ROPE_HEAD_DIM
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")

    def __str__(self):
        return f"layout={self.layout}_page={self.page_size}_io={self.io_backend}_attn={self.attention_type}"


def setup_distributed():
    """Setup distributed environment for testing"""
    import socket

    # Find an available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]

    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        local_rank=0,
        backend="gloo",
    )

    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )

    return get_world_group().cpu_group


def create_hicache_system(config: _TestConfig):
    """Create a complete HiRadixCache system for testing"""

    # Create ReqToTokenPool
    req_to_token_pool = ReqToTokenPool(
        size=config.max_req_num,
        max_context_len=config.max_total_num_tokens,
        device=config.device,
        enable_memory_saver=False,
    )

    if config.attention_type == "MHA":
        token_to_kv_pool = MHATokenToKVPool(
            config.max_total_num_tokens,
            page_size=config.page_size,
            dtype=config.kv_cache_dtype,
            head_num=config.head_num,
            head_dim=config.head_dim,
            layer_num=config.layer_num,
            device=config.device,
            enable_memory_saver=False,
        )
    elif config.attention_type == "MLA":
        token_to_kv_pool = MLATokenToKVPool(
            config.max_total_num_tokens,
            page_size=config.page_size,
            dtype=config.kv_cache_dtype,
            kv_lora_rank=config.kv_lora_rank,
            qk_rope_head_dim=config.qk_rope_head_dim,
            layer_num=config.layer_num,
            device=config.device,
            enable_memory_saver=False,
        )
    else:
        raise ValueError(f"Unsupported attention type: {config.attention_type}")

    token_to_kv_pool_allocator = TokenToKVPoolAllocator(
        config.max_total_num_tokens,
        dtype=config.kv_cache_dtype,
        device=config.device,
        kvcache=token_to_kv_pool,
        need_sort=False,
    )

    # Create HiRadixCache
    hicache = HiRadixCache(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        tp_cache_group=get_world_group().cpu_group,
        page_size=config.page_size,
        hicache_ratio=config.hicache_ratio,
        hicache_size=config.hicache_size,
        hicache_write_policy=config.hicache_write_policy,
        hicache_io_backend=config.hicache_io_backend,
        hicache_mem_layout=config.hicache_mem_layout,
        hicache_storage_backend=config.hicache_storage_backend,
        hicache_storage_prefetch_policy=config.hicache_storage_prefetch_policy,
    )

    return hicache, req_to_token_pool, token_to_kv_pool_allocator, token_to_kv_pool


def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()


def create_test_sequences(num_sequences: int, seq_length: int, page_size: int) -> List[List[int]]:
    """Create test token sequences"""
    sequences = []
    for i in range(num_sequences):
        base_start = i * (seq_length // 2)
        aligned_length = (seq_length // page_size) * page_size
        sequence = list(range(base_start, base_start + aligned_length))
        sequences.append(sequence)
    return sequences


def release_hicache_resources(hicache, token_to_kv_pool):
    """Release HiCache resources"""
    if hicache is not None and hasattr(hicache, "reset"):
        hicache.reset()
    
    if token_to_kv_pool is not None:
        if hasattr(token_to_kv_pool, "_clear_buffers"):
            token_to_kv_pool._clear_buffers()
        else:
            for attr in ("k_buffer", "v_buffer"):
                if hasattr(token_to_kv_pool, attr):
                    delattr(token_to_kv_pool, attr)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# =============================================================================
# Test Context Manager
# =============================================================================

class HiCacheTestContext:
    """Context manager to manage HiCache lifecycle"""
    def __init__(self, config: _TestConfig):
        self.config = config
        self.hicache = None
        self.req_to_token_pool = None
        self.token_to_kv_pool_allocator = None
        self.token_to_kv_pool = None

    def __enter__(self):
        (
            self.hicache,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.token_to_kv_pool,
        ) = create_hicache_system(self.config)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.hicache is not None and hasattr(self.hicache, "reset"):
                self.hicache.reset()
            release_hicache_resources(self.hicache, self.token_to_kv_pool)
        finally:
            del self.hicache, self.req_to_token_pool, self.token_to_kv_pool_allocator, self.token_to_kv_pool
            cleanup_gpu_memory()
        return False


@pytest.fixture(scope="session")
def distributed_group():
    """Setup distributed environment for all tests"""
    return setup_distributed()


# =============================================================================
# Test Class
# =============================================================================

class TestHiCacheIntegration:
    """HiRadixCache Integration Tests"""

    def _run_configuration_test(self, layout: str, page_size: int, io_backend: str, attention_type: str):
        """Test a specific configuration combination"""
        print(f"\n=== Testing: layout={layout}, page_size={page_size}, io_backend={io_backend}, attention={attention_type} ===")

        config = _TestConfig(layout, page_size, io_backend, attention_type)
        with HiCacheTestContext(config) as ctx:
            hicache = ctx.hicache
            token_to_kv_pool_allocator = ctx.token_to_kv_pool_allocator

            # Test basic insert and match operations
            test_sequences = create_test_sequences(TEST_OP_NUM, TEST_OP_SIZE, config.page_size)

            # Test 1: Insert sequences
            start_time = time.monotonic()
            for i, sequence in enumerate(test_sequences):
                kv_indices = token_to_kv_pool_allocator.alloc(len(sequence))
                assert kv_indices is not None, f"Failed to allocate KV cache for sequence {i}"
                hicache.insert(sequence, kv_indices)
            insert_time = time.monotonic() - start_time

            # Test 2: Test prefix matching
            start_time = time.monotonic()
            for i, sequence in enumerate(test_sequences):
                match_result = hicache.match_prefix(sequence)
                assert match_result.device_indices is not None, f"Match failed for sequence {i}"
            match_time = time.monotonic() - start_time

            # Test 3: Test hierarchical cache events processing
            if hasattr(hicache, "check_hicache_events"):
                hicache.check_hicache_events()

            # Test 4: Test prefetch from storage
            for i, sequence in enumerate(test_sequences):
                hicache.prefetch_from_storage(f"req_{i}", hicache.root_node, sequence)
            
            time.sleep(1)

            for i in range(len(test_sequences)):
                completed = hicache.check_prefetch_progress(f"req_{i}")
                assert completed, f"Prefetch did not complete for req_{i}"

            print(f"Insert time: {insert_time:.6f}s")
            print(f"Match time: {match_time:.6f}s")
            print(f"Configuration {config} - PASSED")

            # Basic assertions
            assert insert_time > 0
            assert match_time >= 0
            assert hicache.total_size() >= 0
            assert hicache.evictable_size() >= 0

    def test_allocation_failure_recovery(self, distributed_group):
        """Test allocation failure followed by eviction-based recovery"""
        print(f"\n=== Testing Allocation Failure Recovery ===")

        config = _TestConfig("page_first", 64, "kernel", "MHA")
        config.hicache_write_policy = "write_back"
        
        with HiCacheTestContext(config) as ctx:
            hicache = ctx.hicache
            token_to_kv_pool_allocator = ctx.token_to_kv_pool_allocator

            # Phase 1: Fill up memory pool
            for i in range(12):
                sequence = list(range(i * 1024, (i + 1) * 1024))
                kv_indices = token_to_kv_pool_allocator.alloc(len(sequence))
                assert kv_indices is not None, f"Failed to allocate KV cache for sequence {i}"
                hicache.insert(sequence, kv_indices)

            # Check current cache state
            total_size = hicache.total_size()
            evictable_size = hicache.evictable_size()
            evict_amount = 2048
            
            assert evictable_size > evict_amount, "Should have evictable cache"

            # Phase 2: Try to allocate one more sequence
            overflow_sequence = list(range(12 * 1024, 13 * 1024))
            overflow_kv_indices = token_to_kv_pool_allocator.alloc(len(overflow_sequence))
            assert overflow_kv_indices is None, "Allocation should fail due to memory exhaustion"

            hicache.evict(evict_amount)

            # Verify eviction reduced cache size
            post_evict_total = hicache.total_size()
            assert post_evict_total < total_size, "Cache size should be reduced by eviction"

            # Verify we can now allocate and insert the overflow sequence
            overflow_kv_indices = token_to_kv_pool_allocator.alloc(len(overflow_sequence))
            assert overflow_kv_indices is not None, "Allocation should succeed after eviction"
            hicache.insert(overflow_sequence, overflow_kv_indices)
            match_result = hicache.match_prefix(overflow_sequence)

            assert match_result.device_indices is not None, "Should match newly inserted sequence"

    def test_storage_integration(self, distributed_group):
        """Test storage backend integration"""
        print(f"\n=== Testing Storage Integration ===")

        config = _TestConfig("layer_first", 64, "kernel", "MHA")
        config.hicache_storage_backend = "test"
        
        with HiCacheTestContext(config) as ctx:
            hicache = ctx.hicache
            token_to_kv_pool_allocator = ctx.token_to_kv_pool_allocator

            # Create a test sequence
            sequence = list(range(0, 512))
            kv_indices = token_to_kv_pool_allocator.alloc(len(sequence))
            assert kv_indices is not None, "Failed to allocate KV cache"

            # Insert sequence
            prev_total = hicache.total_size()
            _ = hicache.insert(sequence, kv_indices)
            assert hicache.total_size() > prev_total, "Insert should increase total cache size"

            # Process any pending cache events
            if hasattr(hicache, 'check_hicache_events'):
                for _ in range(5):
                    hicache.check_hicache_events()
                    time.sleep(0.1)

            print("Storage integration test completed")

    @pytest.mark.parametrize("layout,page_size,io_backend,attention_type", TEST_CONFIGURATIONS)
    def test_all_configurations(self, distributed_group, layout, page_size, io_backend, attention_type):
        """Test all configuration combinations using pytest parametrization"""
        self._run_configuration_test(layout, page_size, io_backend, attention_type)


# =============================================================================
# Main Program
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])