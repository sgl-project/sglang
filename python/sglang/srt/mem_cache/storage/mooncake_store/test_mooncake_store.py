import logging
import uuid

import torch
from mooncake_store import MooncakeStore

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_batch_query_keys(kv_num: int, config: HiCacheStorageConfig):
    keys = []
    for _ in range(kv_num):
        key = "test_" + str(uuid.uuid4())
        keys.append(key)
    set_keys = []
    for key in keys:
        if config.is_mla_model:
            set_keys.append(key + "_k")
        else:
            set_keys.append(key + f"_{config.tp_rank}_k")
            set_keys.append(key + f"_{config.tp_rank}_v")
    get_keys = set_keys
    exist_keys = keys
    return set_keys, get_keys, exist_keys


def test_single_operation():
    """Test the set API with a single key-value pair."""
    print("=" * 100)
    print("Testing single operation")

    buffer_size = 1024 * 1024 * 16  # 16MB
    value_elements = 1024
    store = MooncakeStore()
    buffer = torch.randn(buffer_size, dtype=torch.float32)
    store.register_buffer(buffer)
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
    store = MooncakeStore(config)
    buffer = torch.randn(buffer_size, dtype=torch.float32)
    store.register_buffer(buffer)
    value_size = value_elements * buffer.element_size()

    set_keys, get_keys, exist_keys = generate_batch_query_keys(kv_num, config)
    set_slices = [
        buffer[i * value_elements : (i + 1) * value_elements]
        for i in range(len(set_keys))
    ]
    set_locations = [set_slice.data_ptr() for set_slice in set_slices]
    target_sizes = [value_size for _ in range(len(set_keys))]

    # Test batch set operation
    result = store.batch_set(
        set_keys, target_locations=set_locations, target_sizes=target_sizes
    )
    assert result is True, f"❌batch set operation failed"

    # Test batch exists operation
    assert store.batch_exists(
        exist_keys
    ), f"❌keys should exist after batch set operation"

    # Test batch get operation
    get_slices = [
        buffer[
            (len(set_keys) + i)
            * value_elements : (len(set_keys) + i + 1)
            * value_elements
        ]
        for i in range(len(get_keys))
    ]
    get_locations = [get_slice.data_ptr() for get_slice in get_slices]
    result = store.batch_get(
        get_keys, target_locations=get_locations, target_sizes=target_sizes
    )
    assert result == kv_num, f"❌batch get operation failed"
    for i in range(len(get_keys)):
        assert torch.allclose(
            set_slices[i], get_slices[i], atol=1e-6
        ), f"❌batch get operation failed for key: {get_keys[i]}"

    logger.info(f"✅ Batch operation passed")


if __name__ == "__main__":
    test_single_operation()
    test_batch_operation(
        HiCacheStorageConfig(
            is_mla_model=False,
            tp_rank=0,
            tp_size=1,
            model_name=None,
            is_page_first_layout=True,
        )
    )
    test_batch_operation(
        HiCacheStorageConfig(
            is_mla_model=True,
            tp_rank=0,
            tp_size=1,
            model_name=None,
            is_page_first_layout=True,
        )
    )
    test_batch_operation(
        HiCacheStorageConfig(
            is_mla_model=False,
            tp_rank=1,
            tp_size=4,
            model_name=None,
            is_page_first_layout=True,
        )
    )
    test_batch_operation(
        HiCacheStorageConfig(
            is_mla_model=True,
            tp_rank=3,
            tp_size=8,
            model_name=None,
            is_page_first_layout=True,
        )
    )
    logger.info(f"✅ All tests passed")
