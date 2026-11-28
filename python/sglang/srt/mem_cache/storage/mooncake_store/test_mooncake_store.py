import ctypes
import logging
import uuid
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
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
    assert torch.allclose(set_slice, get_slice, atol=1e-6), (
        f"❌get operation failed for key: {key}"
    )

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
    assert store.batch_exists(keys) == kv_num, (
        "keys should exist after batch set operation"
    )

    # Test batch get operation
    get_slices = [
        buffer[
            (kv_num * entries_per_page + i) * value_elements : (
                kv_num * entries_per_page + i + 1
            )
            * value_elements
        ]
        for i in range(kv_num * entries_per_page)
    ]
    get_indices = torch.arange(kv_num, 2 * kv_num)
    result = store.batch_get_v1(keys, get_indices)
    assert all(result), "❌batch get operation failed"
    for i in range(kv_num * entries_per_page):
        assert torch.allclose(set_slices[i], get_slices[i], atol=1e-6), (
            f"❌batch get operation failed for key: {keys[i // entries_per_page]}"
        )

    logger.info(f"✅ Batch operation passed")

    logger.info(f"✅ Batch operation passed")


class FakeMooncakeStoreClient:
    """Dict-backed stand-in for the MooncakeDistributedStore pybind client."""

    def __init__(self):
        self.objects = {}
        self.registered_buffers = []
        self.put_calls = []
        self.put_rpc_calls = 0

    def register_buffer(self, ptr, size):
        self.registered_buffers.append((ptr, size))
        return 0

    def batch_is_exist(self, key_strs):
        return [1 if key in self.objects else 0 for key in key_strs]

    def batch_put_from(self, key_strs, buffer_ptrs, buffer_sizes, config=None):
        group_ids = getattr(config, "group_ids", None) if config is not None else None
        results = []
        self.put_rpc_calls += 1
        for i, (key, ptr, size) in enumerate(zip(key_strs, buffer_ptrs, buffer_sizes)):
            self.objects[key] = ctypes.string_at(ptr, size)
            self.put_calls.append((key, group_ids[i] if group_ids else None))
            results.append(0)
        return results

    def batch_get_into(self, key_strs, buffer_ptrs, buffer_sizes):
        results = []
        for key, ptr, size in zip(key_strs, buffer_ptrs, buffer_sizes):
            data = self.objects.get(key)
            if data is None:
                results.append(-1)
            else:
                ctypes.memmove(ptr, data, size)
                results.append(size)
        return results


class MockMambaPoolHost:
    """Byte-faithful stand-in for MambaPoolHost with registered slot side states."""

    def __init__(self, size, num_layers=2, t_elem=9, c_elem=7, s_elem=5):
        self.page_size = 1
        self.temporal_buffer = torch.randn(
            size, num_layers, t_elem, dtype=torch.bfloat16
        )
        self.conv_buffer = [
            torch.randn(size, num_layers, c_elem, dtype=torch.bfloat16)
            for _ in range(2)
        ]
        # Presence is all the backend probes; the tensors themselves are unused.
        self.slot_state_device_tensors = [torch.randn(size, 3)]
        self.slot_state_buffers = [torch.randn(size, s_elem, dtype=torch.float32)]
        self.size_per_token = num_layers * (t_elem + 2 * c_elem) * 2 + s_elem * 4

    def get_hybrid_pool_buffer(self):
        return [self.temporal_buffer, *self.conv_buffer, *self.slot_state_buffers]

    def _iter_page_tensors(self, index):
        yield self.temporal_buffer[index]
        for conv in self.conv_buffer:
            yield conv[index]
        for state in self.slot_state_buffers:
            yield state[index : index + self.page_size]

    def get_data_page(self, index, flat=True):
        data_page = torch.cat(
            [
                t.contiguous().view(torch.uint8).reshape(-1)
                for t in self._iter_page_tensors(index)
            ]
        )
        return data_page.flatten() if flat else data_page

    def set_from_flat_data_page(self, index, data_page):
        flat_bytes = data_page.contiguous().view(torch.uint8).reshape(-1)
        expected = self.page_size * self.size_per_token
        assert flat_bytes.numel() == expected, (
            f"page has {flat_bytes.numel()} bytes, expected {expected}"
        )
        start = 0
        for tensor in self._iter_page_tensors(index):
            num_bytes = tensor.numel() * tensor.element_size()
            tensor_bytes = flat_bytes[start : start + num_bytes]
            start += num_bytes
            if tensor_bytes.storage_offset() % tensor.element_size():
                tensor_bytes = tensor_bytes.clone()
            restored = tensor_bytes.view(dtype=tensor.dtype).reshape(tensor.shape)
            tensor.copy_(restored)

    def get_page_buffer_meta(self, indices):
        # Mirrors MambaPoolHost: must never be reached while side states exist.
        raise NotImplementedError(
            "Mamba storage zero-copy does not carry registered slot side states"
        )


def _make_wholepage_store(fake_client, cap_bytes):
    store = MooncakeStore.__new__(MooncakeStore)
    store.store = fake_client
    store.config_prefix = "testprefix"
    store.is_mla_backend = False
    store.local_rank = 0
    store.pp_rank = 0
    store.pp_size = 1
    store.enable_pp = False
    store.mha_suffix = "0"
    store.mla_suffix = ""
    store.should_split_heads = False
    store.split_factor = 0
    store.registered_pools = {}
    store.mem_pool_host = SimpleNamespace(kv_buffer=None)
    store._use_group_semantics = True
    store._replicate_config_cls = lambda: SimpleNamespace(group_ids=None)
    store._supports_session_ids = True
    store._wholepage_staging_cap_bytes = cap_bytes
    store._wholepage_staging = None
    store._wholepage_retired = []
    return store


def test_wholepage_mamba_v2():
    """Whole-page IO fallback for mamba pools with registered slot side states.

    Runs without a Mooncake server: the pybind store client is replaced by a
    dict-backed fake.
    """
    print("=" * 100)
    print("Testing whole-page mamba v2 IO (slot side states)")

    fake_client = FakeMooncakeStoreClient()
    store = _make_wholepage_store(fake_client, cap_bytes=256 * 1024 * 1024)
    pool = MockMambaPoolHost(size=64)
    store.register_mem_host_pool_v2(pool, PoolName.MAMBA)
    assert any(
        ptr == pool.temporal_buffer.data_ptr()
        for ptr, _ in fake_client.registered_buffers
    ), "pool buffers must be registered"

    num_pages = 13
    keys = generate_batch_query_keys(num_pages)
    host_indices = torch.arange(num_pages)
    transfers = [
        PoolTransfer(
            name=PoolName.MAMBA,
            keys=keys,
            host_indices=host_indices,
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
    ]

    # Backup: one whole-page object per slot, zero-copy meta never touched.
    results = store.batch_set_v2(transfers)
    assert all(results[PoolName.MAMBA]), "wholepage batch set failed"
    expected_keys = [f"testprefix_{key}_0_wholepage" for key in keys]
    assert set(fake_client.objects) == set(expected_keys), (
        f"unexpected object keys: {sorted(fake_client.objects)[:3]}"
    )
    assert fake_client.put_rpc_calls == 1, "one batch RPC for the full transfer"
    for key, group_id in fake_client.put_calls:
        assert group_id == f"sglang-hicache:{key.removesuffix('_0_wholepage')}", (
            f"group id mismatch for {key}: {group_id}"
        )

    # Hit detection agrees with the written key layout.
    exist_result = store.batch_exists_v2(keys, transfers)
    assert exist_result.kv_hit_pages == num_pages
    assert exist_result.extra_pool_hit_pages[PoolName.MAMBA] == num_pages
    assert exist_result.restorable_prefix_pages == list(range(1, num_pages + 1))

    # Restore into different slots; bytes must roundtrip exactly.
    restore_indices = torch.arange(num_pages, 2 * num_pages)
    get_transfers = [
        PoolTransfer(
            name=PoolName.MAMBA,
            keys=keys,
            host_indices=restore_indices,
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
    ]
    get_results = store.batch_get_v2(get_transfers)
    assert all(get_results[PoolName.MAMBA]), "wholepage batch get failed"
    for i in range(num_pages):
        assert torch.equal(
            pool.get_data_page(i, flat=True),
            pool.get_data_page(int(restore_indices[i]), flat=True),
        ), f"roundtrip mismatch for page {i}"

    # Re-backup only writes pages the store is missing.
    fake_client.put_calls.clear()
    fake_client.objects.pop(expected_keys[5])
    results = store.batch_set_v2(transfers)
    assert all(results[PoolName.MAMBA]), "dedup batch set failed"
    assert len(fake_client.put_calls) == 1, "only the missing page may be re-put"
    assert fake_client.put_calls[0][0] == expected_keys[5]

    # A small staging cap forces chunked RPCs; the large staging buffer is reused.
    fake_client.objects.clear()
    fake_client.put_calls.clear()
    store._wholepage_staging_cap_bytes = 2 * pool.size_per_token
    results = store.batch_set_v2(transfers)
    assert all(results[PoolName.MAMBA]), "chunked batch set failed"
    assert fake_client.put_rpc_calls == 1 + 1 + 7, (
        f"expected 7 chunked RPCs, got {fake_client.put_rpc_calls - 2}"
    )
    staging_numel = store._wholepage_staging.numel()
    assert staging_numel == num_pages * pool.size_per_token, "staging was not reused"

    # A larger transfer grows the staging buffer and retires the old one.
    keys2 = generate_batch_query_keys(20)
    transfers2 = [
        PoolTransfer(
            name=PoolName.MAMBA,
            keys=keys2,
            host_indices=torch.arange(20),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
    ]
    store._wholepage_staging_cap_bytes = 256 * 1024 * 1024
    results = store.batch_set_v2(transfers2)
    assert all(results[PoolName.MAMBA]), "grown batch set failed"
    assert store._wholepage_staging.numel() == 20 * pool.size_per_token
    assert len(store._wholepage_retired) == 1
    assert store._wholepage_retired[0].numel() == staging_numel

    # Pools without side states keep the zero-copy component key layout.
    pool_no_side = MockMambaPoolHost(size=8)
    pool_no_side.slot_state_device_tensors = []
    store.registered_pools[PoolName.MAMBA] = pool_no_side
    component_keys, key_multiplier = store._get_hybrid_page_component_keys(
        ["k1"], transfers[0]
    )
    assert component_keys == ["k1_0_temporal", "k1_0_conv_0", "k1_0_conv_1"], (
        f"zero-copy key layout changed: {component_keys}"
    )
    assert key_multiplier == 3

    logger.info("✅ Whole-page mamba v2 test passed")


if __name__ == "__main__":
    test_wholepage_mamba_v2()
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
