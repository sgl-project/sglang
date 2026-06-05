"""CPU-only round-trip tests for the INDEXER_TOPK HiCache sidecar pool (#26975).

Exercise DSAIndexerTopkPoolHost in isolation: the host<->host backup/load path
and the L3 flat-page serialization path, using lightweight fakes for the
capturer and anchor host pool. No GPU required.
"""

import unittest

import torch

from sglang.srt.mem_cache.memory_pool_host import DSAIndexerTopkPoolHost
from sglang.srt.state_capturer.indexer_topk import (
    get_global_indexer_capturer,
    set_global_indexer_capturer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeHostCache:
    """Stand-in for BaseHostCache: a plain (non-pinned) CPU buffer."""

    def __init__(self, num_tokens, num_layers, topk_size):
        self.buffer = torch.zeros(
            (num_tokens, num_layers, topk_size), dtype=torch.int32
        )


class _FakeCapturer:
    """Duck-typed IndexerTopkCapturer: exposes num_layers/topk_size/host_cache."""

    def __init__(self, num_tokens, num_layers, topk_size):
        self.num_layers = num_layers
        self.topk_size = topk_size
        self.host_cache = _FakeHostCache(num_tokens, num_layers, topk_size)


class _FakeAnchorHost:
    def __init__(self, size, page_size, page_num):
        self.size = size
        self.page_size = page_size
        self.page_num = page_num


class TestDSAIndexerTopkPool(unittest.TestCase):
    PAGE_SIZE = 4
    NUM_LAYERS = 3
    TOPK_SIZE = 5
    # capturer index space (KV slots) and pool slots are independent ranges.
    CAP_TOKENS = 64
    POOL_SIZE = 64  # multiple of page_size

    def setUp(self):
        self._prev_capturer = get_global_indexer_capturer()
        self.capturer = _FakeCapturer(self.CAP_TOKENS, self.NUM_LAYERS, self.TOPK_SIZE)
        set_global_indexer_capturer(self.capturer)
        anchor = _FakeAnchorHost(
            size=self.POOL_SIZE,
            page_size=self.PAGE_SIZE,
            page_num=self.POOL_SIZE // self.PAGE_SIZE,
        )
        # pin_memory=False keeps this CPU-only (pinned alloc needs CUDA).
        self.pool = DSAIndexerTopkPoolHost(
            device_pool=None,
            anchor_host=anchor,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
        )

    def tearDown(self):
        set_global_indexer_capturer(self._prev_capturer)

    def _fill_capturer(self, slots):
        """Write deterministic, slot-unique topk data into the capturer buffer."""
        buf = self.capturer.host_cache.buffer
        for i, s in enumerate(slots.tolist()):
            base = (s + 1) * 1000
            buf[s] = (
                torch.arange(
                    self.NUM_LAYERS * self.TOPK_SIZE, dtype=torch.int32
                ).reshape(self.NUM_LAYERS, self.TOPK_SIZE)
                + base
            )

    def test_l2_round_trip_to_new_slots(self):
        """capturer[old] -> backup -> pool; pool -> per-layer load -> capturer[new]."""
        old_slots = torch.tensor([8, 9, 10, 11], dtype=torch.int64)  # one page
        host_slots = torch.tensor([0, 1, 2, 3], dtype=torch.int64)  # pool page 0
        new_slots = torch.tensor([20, 21, 22, 23], dtype=torch.int64)

        self._fill_capturer(old_slots)
        expected = self.capturer.host_cache.buffer[old_slots].clone()

        # backup: capturer[old_slots] -> pool[host_slots]
        self.pool.backup_from_device_all_layer(
            None, host_slots, old_slots, io_backend="direct"
        )
        self.assertTrue(torch.equal(self.pool.buffer[host_slots], expected))

        # simulate slot recycling: capturer rows wiped
        self.capturer.host_cache.buffer[old_slots] = 0

        # load: pool[host_slots] -> capturer[new_slots], one layer at a time
        for layer_id in range(self.NUM_LAYERS):
            self.pool.load_to_device_per_layer(
                None, host_slots, new_slots, layer_id, io_backend="direct"
            )

        self.assertTrue(
            torch.equal(self.capturer.host_cache.buffer[new_slots], expected),
            "restored capturer rows at new slots must equal the original topk",
        )

    def test_empty_indices_noop(self):
        empty = torch.tensor([], dtype=torch.int64)
        # Must not raise and must not touch buffers.
        self.pool.backup_from_device_all_layer(None, empty, empty, io_backend="direct")
        self.pool.load_to_device_per_layer(None, empty, empty, 0, io_backend="direct")

    def test_flat_data_page_round_trip(self):
        """L3 path: get_data_page(flat) -> set_from_flat_data_page restores bytes."""
        # Fill pool page 1 directly with known data.
        page_idx = 1
        start = page_idx * self.PAGE_SIZE
        src = (
            torch.arange(
                self.PAGE_SIZE * self.NUM_LAYERS * self.TOPK_SIZE, dtype=torch.int32
            ).reshape(self.PAGE_SIZE, self.NUM_LAYERS, self.TOPK_SIZE)
            + 777
        )
        self.pool.buffer[start : start + self.PAGE_SIZE] = src

        flat = self.pool.get_data_page(start, flat=True)
        self.assertEqual(
            flat.numel(), self.PAGE_SIZE * self.NUM_LAYERS * self.TOPK_SIZE
        )

        # get_data_page may return a view; clone before mutating the underlying
        # buffer so the flat snapshot survives the wipe.
        flat = flat.clone()

        # Wipe and restore from the flat page.
        self.pool.buffer[start : start + self.PAGE_SIZE] = 0
        self.pool.set_from_flat_data_page(start, flat)
        self.assertTrue(
            torch.equal(self.pool.buffer[start : start + self.PAGE_SIZE], src)
        )

    def test_dummy_flat_data_page_shape(self):
        dummy = self.pool.get_dummy_flat_data_page()
        self.assertEqual(
            dummy.numel(), self.PAGE_SIZE * self.NUM_LAYERS * self.TOPK_SIZE
        )
        self.assertTrue(torch.equal(dummy, torch.zeros_like(dummy)))

    def test_page_buffer_meta_strides(self):
        indices = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11], dtype=torch.int64)
        ptrs, sizes = self.pool.get_page_buffer_meta(indices)
        page_bytes = self.PAGE_SIZE * self.NUM_LAYERS * self.TOPK_SIZE * 4
        self.assertEqual(sizes, [page_bytes, page_bytes])
        self.assertEqual(ptrs[1] - ptrs[0], 2 * page_bytes)  # page 0 vs page 2

    def test_size_per_token(self):
        self.assertEqual(
            self.pool.get_size_per_token(), self.NUM_LAYERS * self.TOPK_SIZE * 4
        )
        self.assertEqual(
            self.pool.get_ksize_per_token(), self.pool.get_size_per_token()
        )


if __name__ == "__main__":
    unittest.main()
