"""Test is_mooncake_registerable property for host pools.

The property delegates to self.allocator.is_mooncake_compatible. A pool
is Mooncake-registerable only when backed by MooncakeHostTensorAllocator.
"""

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.pool_host.common import HostTensorAllocator
from sglang.srt.mem_cache.memory_pool_host import (
    HostPoolGroup,
    LogicalHostPool,
    MHATokenToKVPoolHost,
    PoolEntry,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool


class _MockMooncakeAllocator(HostTensorAllocator):
    """Mimics MooncakeHostTensorAllocator for testing."""

    @property
    def is_mooncake_compatible(self) -> bool:
        return True

    def allocate(self, dims, dtype, device):
        return torch.empty(dims, dtype=dtype)


class TestIsMooncakeRegisterable(unittest.TestCase):
    """Test the is_mooncake_registerable property across different pool types."""

    def test_host_kvcache_with_mooncake_allocator_is_true(self):
        """HostKVCache with Mooncake-compatible allocator is registerable."""

        class ConcreteHostKVCache(HostKVCache):
            def get_size_per_token(self):
                return 1024

            def init_kv_buffer(self):
                return torch.empty((1, 1, 1, 1), dtype=torch.float16)

            def load_to_device_per_layer(self, *args, **kwargs):
                pass

            def backup_from_device_all_layer(self, *args, **kwargs):
                pass

            def get_data_page(self, *args, **kwargs):
                return None

            def get_dummy_flat_data_page(self):
                return torch.empty(0, dtype=torch.uint8)

            def set_from_flat_data_page(self, *args, **kwargs):
                pass

        device_pool = Mock(spec=MHATokenToKVPool)
        device_pool.store_dtype = torch.float16
        device_pool.size = 100
        device_pool.start_layer = 0
        device_pool.end_layer = 1

        host_pool = ConcreteHostKVCache(
            device_pool=device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=64,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
        )
        # Patch allocator with Mooncake-compatible one
        host_pool.allocator = _MockMooncakeAllocator()
        self.assertTrue(host_pool.is_mooncake_registerable)

    def test_host_kvcache_with_default_allocator_is_false(self):
        """HostKVCache with default (mmap) allocator is NOT registerable."""

        class ConcreteHostKVCache(HostKVCache):
            def get_size_per_token(self):
                return 1024

            def init_kv_buffer(self):
                return torch.empty((1, 1, 1, 1), dtype=torch.float16)

            def load_to_device_per_layer(self, *args, **kwargs):
                pass

            def backup_from_device_all_layer(self, *args, **kwargs):
                pass

            def get_data_page(self, *args, **kwargs):
                return None

            def get_dummy_flat_data_page(self):
                return torch.empty(0, dtype=torch.uint8)

            def set_from_flat_data_page(self, *args, **kwargs):
                pass

        device_pool = Mock(spec=MHATokenToKVPool)
        device_pool.store_dtype = torch.float16
        device_pool.size = 100
        device_pool.start_layer = 0
        device_pool.end_layer = 1

        host_pool = ConcreteHostKVCache(
            device_pool=device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=64,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
        )
        # Default HostTensorAllocator is not Mooncake-compatible
        self.assertFalse(host_pool.is_mooncake_registerable)

    def test_logical_host_pool_false(self):
        """LogicalHostPool has is_mooncake_registerable=False (no buffer)."""
        logical_pool = LogicalHostPool(size=1024, page_size=64)
        self.assertFalse(logical_pool.is_mooncake_registerable)

    def test_mha_token_to_kv_pool_host_with_mooncake_allocator(self):
        """MHATokenToKVPoolHost with Mooncake allocator is registerable."""
        device_pool = Mock(spec=MHATokenToKVPool)
        device_pool.store_dtype = torch.float16
        device_pool.size = 100
        device_pool.start_layer = 0
        device_pool.end_layer = 1
        device_pool.head_num = 8
        device_pool.head_dim = 128

        host_pool = MHATokenToKVPoolHost(
            device_pool=device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=64,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
        )
        # Patch allocator
        host_pool.allocator = _MockMooncakeAllocator()
        self.assertTrue(host_pool.is_mooncake_registerable)

    def test_host_pool_group_delegates_to_anchor(self):
        """HostPoolGroup delegates is_mooncake_registerable to anchor pool."""
        mock_pool_true = Mock()
        mock_pool_true.is_mooncake_registerable = True
        mock_pool_true.layout = "layer_first"
        mock_pool_true.page_size = 64
        mock_pool_true.device = "cpu"
        mock_pool_true.size = 1024

        mock_pool_false = Mock()
        mock_pool_false.is_mooncake_registerable = False
        mock_pool_false.layout = "layer_first"
        mock_pool_false.page_size = 64
        mock_pool_false.device = "cpu"
        mock_pool_false.size = 1024

        entry_true = PoolEntry(
            name="KV",
            host_pool=mock_pool_true,
            device_pool=None,
            is_primary_index_anchor=True,
            layer_mapper=lambda x: x,
        )
        group_true = HostPoolGroup([entry_true])
        self.assertTrue(group_true.is_mooncake_registerable)

        entry_false = PoolEntry(
            name="KV",
            host_pool=mock_pool_false,
            device_pool=None,
            is_primary_index_anchor=True,
            layer_mapper=lambda x: x,
        )
        group_false = HostPoolGroup([entry_false])
        self.assertFalse(group_false.is_mooncake_registerable)


if __name__ == "__main__":
    unittest.main()
