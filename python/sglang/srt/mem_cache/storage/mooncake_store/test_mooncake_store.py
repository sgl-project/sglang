import logging
import uuid

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def make_hicache_storage_config(
    *,
    is_mla_model: bool,
    tp_rank: int,
    tp_size: int,
) -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=is_mla_model,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=None,
    )


def generate_batch_query_keys(kv_num: int):
    return ["test_" + str(uuid.uuid4()) for _ in range(kv_num)]


def create_mock_host_kv_cache(
    buffer_size,
    entries_per_page=2,
    page_elements=1,
    dtype=torch.float32,
):
    """Create a mock HostKVCache-like object for testing."""
    buffer = torch.randn(buffer_size, dtype=dtype)

    class MockHostKVCache:
        def __init__(self, buffer, entries_per_page, page_elements):
            self.kv_buffer = buffer
            self.layout = "page_first"
            self.page_size = 1  # Simple page size for testing
            self.entries_per_page = entries_per_page
            self.page_elements = page_elements

        def get_page_buffer_meta(self, indices):
            """Mock implementation of get_page_buffer_meta."""
            ptr_list = []
            element_size_list = []
            for idx in indices:
                page_idx = int(idx)
                page_offset = page_idx * self.entries_per_page * self.page_elements
                for entry_idx in range(self.entries_per_page):
                    offset = page_offset + entry_idx * self.page_elements
                    ptr_list.append(self.kv_buffer[offset:].data_ptr())
                    element_size_list.append(
                        self.page_elements * self.kv_buffer.element_size()
                    )
            return ptr_list, element_size_list

        def get_ksize_per_token(self):
            return (
                self.entries_per_page
                * self.page_elements
                * self.kv_buffer.element_size()
            )

    return MockHostKVCache(buffer, entries_per_page, page_elements), buffer


def test_single_operation():
    """Test the set API with a single key-value pair."""
    print("=" * 100)
    print("Testing single operation")

    buffer_size = 1024 * 1024 * 16  # 16MB
    value_elements = 1024
    store = MooncakeStore(
        make_hicache_storage_config(is_mla_model=False, tp_rank=0, tp_size=1)
    )
    mock_host_kv_cache, buffer = create_mock_host_kv_cache(
        buffer_size,
        entries_per_page=2,
        page_elements=value_elements,
    )

    # Register the memory pool host - this is the proper workflow
    store.register_mem_pool_host(mock_host_kv_cache)

    value_size = value_elements * buffer.element_size()

    key = str(uuid.uuid4())
    set_slice = buffer[:value_elements]
    get_slice = buffer[value_elements : 2 * value_elements]
    set_location = set_slice.data_ptr()
    get_location = get_slice.data_ptr()

    # Test set operation
    result = store.set(key, target_location=set_location, target_sizes=value_size)
    assert result is True, f"❌set operation failed for key: {key}"

    # Test exists operation
    assert store.exists(key), f"❌key {key} should exist after set operation"

    # Test get operation
    result = store.get(key, target_location=get_location, target_sizes=value_size)
    assert result is True, f"❌get operation failed for key: {key}"

    # Compare the data using proper tensor indices
    assert torch.allclose(
        set_slice, get_slice, atol=1e-6
    ), f"❌get operation failed for key: {key}"

    logger.info(f"✅ Single operation passed")


def test_batch_operation(config: HiCacheStorageConfig):
    """Test the batch set/get APIs with multiple key-value pairs."""
    print("=" * 100)
    print(f"Testing batch operation with config: {config}")

    buffer_size = 1024 * 1024 * 16  # 16MB
    value_elements = 256
    kv_num = 13
    entries_per_page = 1 if config.is_mla_model else 2
    store = MooncakeStore(config)
    mock_host_kv_cache, buffer = create_mock_host_kv_cache(
        buffer_size,
        entries_per_page=entries_per_page,
        page_elements=value_elements,
    )

    store.register_mem_pool_host(mock_host_kv_cache)

    keys = generate_batch_query_keys(kv_num)
    set_slices = [
        buffer[i * value_elements : (i + 1) * value_elements]
        for i in range(kv_num * entries_per_page)
    ]
    set_indices = torch.arange(kv_num)

    # Test batch set operation
    result = store.batch_set_v1(keys, set_indices)
    assert all(result), "batch set operation failed"

    # Test batch exists operation
    assert (
        store.batch_exists(keys) == kv_num
    ), "keys should exist after batch set operation"

    # Test batch get operation
    get_slices = [
        buffer[
            (kv_num * entries_per_page + i)
            * value_elements : (kv_num * entries_per_page + i + 1)
            * value_elements
        ]
        for i in range(kv_num * entries_per_page)
    ]
    get_indices = torch.arange(kv_num, 2 * kv_num)
    result = store.batch_get_v1(keys, get_indices)
    assert all(result), "❌batch get operation failed"
    for i in range(kv_num * entries_per_page):
        assert torch.allclose(
            set_slices[i], get_slices[i], atol=1e-6
        ), f"❌batch get operation failed for key: {keys[i // entries_per_page]}"

    logger.info(f"✅ Batch operation passed")


if __name__ == "__main__":
    test_single_operation()
    test_batch_operation(
        make_hicache_storage_config(is_mla_model=False, tp_rank=0, tp_size=1)
    )
    test_batch_operation(
        make_hicache_storage_config(is_mla_model=True, tp_rank=0, tp_size=1)
    )
    test_batch_operation(
        make_hicache_storage_config(is_mla_model=False, tp_rank=1, tp_size=4)
    )
    test_batch_operation(
        make_hicache_storage_config(is_mla_model=True, tp_rank=3, tp_size=8)
    )
    logger.info(f"✅ All tests passed")
