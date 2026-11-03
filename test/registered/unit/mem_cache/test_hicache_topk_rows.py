"""HiCache keeps routed-experts rows with their KV: a write copies the rows of
its device slots to host slots, and a load-back copies them into the new
device slots before get_topk reads them."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hicache_topk_rows import HostTopkRows
from sglang.srt.state_capturer.base import BaseTopkCapturer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

HOST_SLOTS = [10, 11, 12]


@unittest.skipIf(not torch.cuda.is_available(), "needs CUDA")
class TestHiCacheTopkRows(CustomTestCase):
    def setUp(self):
        self.capturer = BaseTopkCapturer(
            num_tokens=64,
            max_batch_size=8,
            num_layers=2,
            topk_size=3,
            device="cuda",
            name="routed_experts",
        )
        self.rows = HostTopkRows(host_size=32, capturer=self.capturer, device="cuda")
        self.stream = torch.cuda.Stream()
        self.expected = torch.arange(18, dtype=torch.int32).view(3, 2, 3)

    def tearDown(self):
        self.capturer.destroy()

    def _copy(self, device_slots):
        return HostTopkRows.copy_indices(
            device_indices=torch.tensor(device_slots, device="cuda"),
            host_indices=torch.tensor(HOST_SLOTS),
            stream=self.stream,
        )

    def _write_prefix(self):
        """Rows of device slots 5..7 are backed up to host slots 10..12, then
        the device slots are reused by another request."""
        self.capturer.host_cache.buffer[[5, 6, 7]] = self.expected
        self.rows.finish_write(self._copy([5, 6, 7]))
        self.capturer.host_cache.buffer[[5, 6, 7]] = -1

    def _load(self, device_slots):
        row_copy = self._copy(device_slots)
        self.rows.add_load(row_copy)
        return row_copy

    def _read(self, slots):
        pool = SimpleNamespace(
            req_to_token=torch.tensor([slots + [0]], dtype=torch.int64, device="cuda")
        )
        return self.capturer.get_topk(
            req_pool_idx=0, seqlen=len(slots) + 1, req_to_token_pool=pool
        )

    def test_load_back_restores_rows_before_read(self):
        self._write_prefix()
        self._load([20, 21, 22])
        self.assertTrue(torch.equal(self._read([20, 21, 22]), self.expected))

    def test_load_ack_restores_rows_once(self):
        self._write_prefix()
        row_copy = self._load([30, 31, 32])
        self.rows.finish_load(row_copy)
        self.assertTrue(
            torch.equal(self.capturer.host_cache.buffer[[30, 31, 32]], self.expected)
        )
        # The slots are reused before the next read; an acked load must not
        # be applied again.
        self.capturer.host_cache.buffer[[30, 31, 32]] = 7
        self._read([30, 31, 32])
        self.assertTrue((self.capturer.host_cache.buffer[[30, 31, 32]] == 7).all())

    def test_reset_drops_pending_copies(self):
        self._write_prefix()
        self._load([40, 41, 42])
        self.rows.reset()
        self.rows.apply_pending_loads()
        self.assertTrue((self.capturer.host_cache.buffer[[40, 41, 42]] == 0).all())

    def test_host_tier_rows_are_freed_with_the_capturer(self):
        self.assertIsNotNone(self.capturer.host_tier_cache.buffer)
        self.capturer.destroy()
        self.assertIsNone(self.capturer.host_tier_cache.buffer)


if __name__ == "__main__":
    unittest.main()
