import uuid

import torch
from mooncake_store import MooncakeStore

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)


def create_host_mem_pool(config: HiCacheStorageConfig, buffer: torch.Tensor):
    """Create a HostKVCache instance for testing."""
    page_size = 1
    head_num, head_dim, layer_num = 32, 128, 1

    # Simple capacity calculation
    elements_per_token = (
        head_dim * head_dim if config.is_mla_model else head_num * head_dim * 2
    )
    token_capacity = buffer.numel() // elements_per_token

    pool_class = MLATokenToKVPool if config.is_mla_model else MHATokenToKVPool
    host_class = MLATokenToKVPoolHost if config.is_mla_model else MHATokenToKVPoolHost

    kv_pool = pool_class(
        token_capacity,
        page_size,
        buffer.dtype,
        head_num,
        head_dim,
        layer_num,
        "cpu",
        False,
        0,
        layer_num,
    )
    return host_class(kv_pool, 2.0, 0, page_size, "page_first")


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
    config = HiCacheStorageConfig(
        is_mla_model=False,
        tp_rank=0,
        tp_size=1,
        model_name=None,
        is_page_first_layout=True,
    )

    store = MooncakeStore(config)
    buffer = torch.randn(buffer_size, dtype=torch.float32, pin_memory=True)
    host_mem_pool = create_host_mem_pool(config, buffer)
    store.register_mem_pool_host(host_mem_pool)

    kv_buffer = host_mem_pool.kv_buffer
    print(f"Buffer shape: {kv_buffer.shape}, dtype: {kv_buffer.dtype}")

    # Use flat buffer to avoid complex shape calculations
    buffer_flat = kv_buffer.flatten()
    value_elements = min(
        128, buffer_flat.numel() // 4
    )  # Leave space for both set and get

    set_data = torch.randn(value_elements, dtype=torch.float32)
    set_location = kv_buffer.data_ptr()
    get_location = set_location + value_elements * buffer_flat.element_size()
    value_size = value_elements * buffer_flat.element_size()

    # Copy test data to buffer
    buffer_flat[:value_elements] = set_data
    key = str(uuid.uuid4())

    # Test operations
    assert store.set(key, target_location=set_location, target_sizes=value_size)
    assert store.exists(key)
    assert store.get(key, target_location=get_location, target_sizes=value_size)

    # Verify data
    get_data = buffer_flat[value_elements : 2 * value_elements]
    assert torch.allclose(
        set_data, get_data, atol=1e-6
    ), "Data mismatch after get operation"

    print(f"✅ Single operation passed")


def test_batch_operation(config: HiCacheStorageConfig):
    """Test the batch set/get APIs with multiple key-value pairs."""
    print("=" * 100)
    print(f"Testing batch operation with config: {config}")

    buffer_size = 1024 * 1024 * 16  # 16MB
    kv_num = 13

    if not hasattr(config, "extra_config") or config.extra_config is None:
        config.extra_config = {}

    store = MooncakeStore(config)
    buffer = torch.randn(buffer_size, dtype=torch.float32, pin_memory=True)
    host_mem_pool = create_host_mem_pool(config, buffer)
    store.register_mem_pool_host(host_mem_pool)

    kv_buffer = host_mem_pool.kv_buffer
    buffer_flat = kv_buffer.flatten()
    print(f"Buffer shape: {kv_buffer.shape}, dtype: {kv_buffer.dtype}")

    set_keys, get_keys, exist_keys = generate_batch_query_keys(kv_num, config)

    # Calculate value size to fit in buffer
    max_elements_per_key = buffer_flat.numel() // (len(set_keys) + len(get_keys) + 2)
    value_elements = min(128, max_elements_per_key)
    value_size = value_elements * buffer_flat.element_size()

    print(f"Using {value_elements} elements per key")

    # Prepare test data and locations
    set_data_list = []
    set_locations = []
    get_locations = []

    for i in range(len(set_keys)):
        # Set data and location
        set_data = torch.randn(value_elements, dtype=torch.float32)
        set_data_list.append(set_data)

        set_offset = i * value_elements
        set_location = kv_buffer.data_ptr() + set_offset * buffer_flat.element_size()
        set_locations.append(set_location)

        # Copy data to buffer
        buffer_flat[set_offset : set_offset + value_elements] = set_data

    for i in range(len(get_keys)):
        get_offset = (len(set_keys) + i) * value_elements
        get_location = kv_buffer.data_ptr() + get_offset * buffer_flat.element_size()
        get_locations.append(get_location)

    target_sizes = [value_size for _ in range(len(set_keys))]

    # Test batch operations
    assert store.batch_set(
        set_keys, target_locations=set_locations, target_sizes=target_sizes
    )
    assert store.batch_exists(exist_keys)
    assert (
        store.batch_get(
            get_keys, target_locations=get_locations, target_sizes=target_sizes
        )
        == kv_num
    )

    # Compare the data
    for i in range(len(get_keys)):
        get_offset = (len(set_keys) + i) * value_elements
        get_data = buffer_flat[get_offset : get_offset + value_elements]
        assert torch.allclose(
            set_data_list[i], get_data, atol=1e-6
        ), f"Data mismatch for key {get_keys[i]}"

    print(f"✅ Batch operation passed")


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
    print(f"✅ All tests passed")
