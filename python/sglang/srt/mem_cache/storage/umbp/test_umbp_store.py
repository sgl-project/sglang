#!/usr/bin/env python3
"""Unit tests for UMBPStore with mocked HostKVCache."""

import ctypes
import tempfile
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock


@dataclass
class MockStorageConfig:
    tp_rank: int = 0
    tp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1
    is_mla_model: bool = False
    is_page_first_layout: bool = True
    model_name: str = "test-model"
    tp_lcm_size: Optional[int] = None
    should_split_heads: bool = False
    extra_config: Optional[dict] = None


class MockHostKVCache:
    """Mock HostKVCache that simulates page_first layout with real buffers."""

    def __init__(self, num_pages=4, page_size=1, element_size=1024):
        self.layout = "page_first"
        self.page_size = page_size
        self.element_size = element_size  # bytes per K or V per page

        total_bytes = num_pages * 2 * element_size  # K+V for each page
        self._buffer = (ctypes.c_char * total_bytes)()
        self._buffer_ptr = ctypes.addressof(self._buffer)
        self.kv_buffer = MagicMock()
        self.kv_buffer.data_ptr.return_value = self._buffer_ptr

    def get_page_buffer_meta(self, indices):
        """Return (ptr_list, element_size_list) for MHA page_first layout.

        For page_first MHA: alternating K, V pointers per page.
        """
        ptr_list = []
        pages = list(range(0, len(indices), self.page_size))

        for page_start in pages:
            page_idx = (
                indices[page_start] if hasattr(indices, "__getitem__") else page_start
            )
            # K pointer
            k_ptr = self._buffer_ptr + page_idx * 2 * self.element_size
            # V pointer
            v_ptr = k_ptr + self.element_size
            ptr_list.append(k_ptr)
            ptr_list.append(v_ptr)

        return ptr_list, self.element_size

    def fill_page(self, page_idx, k_val, v_val):
        """Fill a page's K and V with specific byte values."""
        k_offset = page_idx * 2 * self.element_size
        v_offset = k_offset + self.element_size
        ctypes.memset(self._buffer_ptr + k_offset, k_val, self.element_size)
        ctypes.memset(self._buffer_ptr + v_offset, v_val, self.element_size)

    def read_page_k(self, page_idx):
        """Read K data for a page."""
        k_offset = page_idx * 2 * self.element_size
        return bytes(ctypes.string_at(self._buffer_ptr + k_offset, self.element_size))

    def read_page_v(self, page_idx):
        """Read V data for a page."""
        v_offset = page_idx * 2 * self.element_size + self.element_size
        return bytes(ctypes.string_at(self._buffer_ptr + v_offset, self.element_size))


def make_indices(indices):
    """Create a list that acts like a torch.Tensor of indices."""
    return indices


def test_basic_set_get():
    print("test_basic_set_get... ", end="")
    from sglang.srt.mem_cache.storage.umbp.umbp_store import UMBPStore

    config = MockStorageConfig(
        extra_config={"dram_capacity_bytes": 1024 * 1024, "ssd_enabled": False}
    )
    store = UMBPStore(config)

    mem_pool = MockHostKVCache(num_pages=4, page_size=1, element_size=512)
    store.register_mem_pool_host(mem_pool)

    # Fill page 0 with data
    mem_pool.fill_page(0, ord("A"), ord("B"))

    # Set: store page 0 data
    keys = ["hash_page_0"]
    indices = make_indices([0])
    result = store.batch_set_v1(keys, indices)
    assert len(result) == 1
    assert result[0] is True, f"Set failed: {result}"

    # Clear the buffer to prove get actually reads from store
    mem_pool.fill_page(0, 0, 0)

    # Get: restore page 0 data
    result = store.batch_get_v1(keys, indices)
    assert len(result) == 1
    assert result[0] is True, f"Get failed: {result}"

    # Verify data restored
    k_data = mem_pool.read_page_k(0)
    v_data = mem_pool.read_page_v(0)
    assert k_data == bytes([ord("A")] * 512), "K data mismatch"
    assert v_data == bytes([ord("B")] * 512), "V data mismatch"

    print("PASSED")


def test_batch_set_get_multiple_pages():
    print("test_batch_set_get_multiple_pages... ", end="")
    from sglang.srt.mem_cache.storage.umbp.umbp_store import UMBPStore

    config = MockStorageConfig(
        extra_config={"dram_capacity_bytes": 4 * 1024 * 1024, "ssd_enabled": False}
    )
    store = UMBPStore(config)

    mem_pool = MockHostKVCache(num_pages=4, page_size=1, element_size=256)
    store.register_mem_pool_host(mem_pool)

    # Fill pages with distinct data
    for i in range(4):
        mem_pool.fill_page(i, ord("A") + i, ord("a") + i)

    keys = [f"hash_{i}" for i in range(4)]
    indices = make_indices([0, 1, 2, 3])

    # Set all 4 pages
    set_results = store.batch_set_v1(keys, indices)
    assert all(set_results), f"Batch set failed: {set_results}"

    # Clear buffer
    for i in range(4):
        mem_pool.fill_page(i, 0, 0)

    # Get all 4 pages
    get_results = store.batch_get_v1(keys, indices)
    assert all(get_results), f"Batch get failed: {get_results}"

    # Verify each page
    for i in range(4):
        k = mem_pool.read_page_k(i)
        v = mem_pool.read_page_v(i)
        assert k[0] == ord("A") + i, f"Page {i} K mismatch"
        assert v[0] == ord("a") + i, f"Page {i} V mismatch"

    print("PASSED")


def test_batch_exists():
    print("test_batch_exists... ", end="")
    from sglang.srt.mem_cache.storage.umbp.umbp_store import UMBPStore

    config = MockStorageConfig(
        extra_config={"dram_capacity_bytes": 1024 * 1024, "ssd_enabled": False}
    )
    store = UMBPStore(config)

    mem_pool = MockHostKVCache(num_pages=4, page_size=1, element_size=256)
    store.register_mem_pool_host(mem_pool)

    # Store first 2 pages
    for i in range(2):
        mem_pool.fill_page(i, ord("X"), ord("Y"))

    keys_to_set = [f"exists_{i}" for i in range(2)]
    indices = make_indices([0, 1])
    store.batch_set_v1(keys_to_set, indices)

    # Check exists: first 2 exist, 3rd does not
    all_keys = [f"exists_{i}" for i in range(3)]
    count = store.batch_exists(all_keys)
    assert count == 2, f"Expected 2 consecutive, got {count}"

    print("PASSED")


def test_dedup_on_set():
    print("test_dedup_on_set... ", end="")
    from sglang.srt.mem_cache.storage.umbp.umbp_store import UMBPStore

    config = MockStorageConfig(
        extra_config={"dram_capacity_bytes": 1024 * 1024, "ssd_enabled": False}
    )
    store = UMBPStore(config)

    mem_pool = MockHostKVCache(num_pages=2, page_size=1, element_size=256)
    store.register_mem_pool_host(mem_pool)

    mem_pool.fill_page(0, ord("A"), ord("B"))

    # Set once
    keys = ["dedup_key"]
    indices = make_indices([0])
    store.batch_set_v1(keys, indices)

    # Set again — should succeed (dedup)
    mem_pool.fill_page(0, ord("X"), ord("Y"))  # Different data
    result = store.batch_set_v1(keys, indices)
    assert result[0] is True

    # Get should return original data (dedup means second set was skipped)
    mem_pool.fill_page(0, 0, 0)
    store.batch_get_v1(keys, indices)
    k = mem_pool.read_page_k(0)
    assert k[0] == ord("A"), f"Expected original data 'A', got {chr(k[0])}"

    print("PASSED")


def test_clear():
    print("test_clear... ", end="")
    from sglang.srt.mem_cache.storage.umbp.umbp_store import UMBPStore

    config = MockStorageConfig(
        extra_config={"dram_capacity_bytes": 1024 * 1024, "ssd_enabled": False}
    )
    store = UMBPStore(config)

    mem_pool = MockHostKVCache(num_pages=2, page_size=1, element_size=256)
    store.register_mem_pool_host(mem_pool)

    mem_pool.fill_page(0, ord("C"), ord("D"))
    store.batch_set_v1(["clear_key"], make_indices([0]))

    assert store.exists("clear_key_0_k")
    store.clear()
    assert not store.exists("clear_key_0_k")

    print("PASSED")


def test_legacy_interface():
    print("test_legacy_interface... ", end="")
    from sglang.srt.mem_cache.storage.umbp.umbp_store import UMBPStore

    config = MockStorageConfig(
        extra_config={"dram_capacity_bytes": 1024 * 1024, "ssd_enabled": False}
    )
    store = UMBPStore(config)

    # Direct set/get/exists via legacy interface
    data = (ctypes.c_char * 256)(*([b"Z"] * 256))
    ptr = ctypes.addressof(data)

    assert store.set("legacy_key", target_location=ptr, target_sizes=256)
    assert store.exists("legacy_key")

    buf = (ctypes.c_char * 256)()
    result = store.get(
        "legacy_key", target_location=ctypes.addressof(buf), target_sizes=256
    )
    assert result is not None
    assert buf[0] == b"Z"

    print("PASSED")


def test_segmented_layout_basic():
    print("test_segmented_layout_basic... ", end="")
    from sglang.srt.mem_cache.storage.umbp.umbp_store import UMBPStore

    with tempfile.TemporaryDirectory(prefix="umbp_segmented_") as ssd_dir:
        config = MockStorageConfig(
            extra_config={
                "dram_capacity_bytes": 1024 * 1024,
                "ssd_enabled": True,
                "ssd_storage_dir": ssd_dir,
                "ssd_capacity_bytes": 16 * 1024 * 1024,
            }
        )
        store = UMBPStore(config)

        mem_pool = MockHostKVCache(num_pages=2, page_size=1, element_size=256)
        store.register_mem_pool_host(mem_pool)
        mem_pool.fill_page(0, ord("M"), ord("N"))

        keys = ["seg_hash_0"]
        indices = make_indices([0])
        assert store.batch_set_v1(keys, indices) == [True]
        mem_pool.fill_page(0, 0, 0)
        assert store.batch_get_v1(keys, indices) == [True]
        assert mem_pool.read_page_k(0)[0] == ord("M")
        assert mem_pool.read_page_v(0)[0] == ord("N")
        store.clear()

    print("PASSED")


if __name__ == "__main__":
    print("=== UMBPStore Tests ===")
    test_basic_set_get()
    test_batch_set_get_multiple_pages()
    test_batch_exists()
    test_dedup_on_set()
    test_clear()
    test_legacy_interface()
    test_segmented_layout_basic()
    print("All UMBPStore tests passed!")
