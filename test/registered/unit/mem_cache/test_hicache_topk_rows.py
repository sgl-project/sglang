"""HiCache keeps routed-experts rows with their KV: a write copies the rows of
its device slots to host slots, and a load-back copies them into the new
device slots before get_topk reads them."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_topk_rows import HostTopkRows
from sglang.srt.state_capturer.base import BaseTopkCapturer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

HOST_SLOTS = [10, 11, 12]
# A DCP (or page-interleaved) host pool: 16 physical rows, 64 logical slots.
HOST_POOL = SimpleNamespace(size=16, logical_size=64)


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
        with mock.patch(
            "sglang.srt.mem_cache.hicache_topk_rows.get_global_experts_capturer",
            return_value=self.capturer,
        ):
            self.rows = HostTopkRows.create(host_pool=HOST_POOL, device="cuda")
        self.stream = torch.cuda.Stream()
        self.expected = torch.arange(18, dtype=torch.int32).view(3, 2, 3)

    def tearDown(self):
        self.capturer.destroy()

    def _copy(self, device_slots, host_slots=HOST_SLOTS):
        return self.rows.copy_indices(
            device_indices=torch.tensor(device_slots, device="cuda"),
            host_indices=torch.tensor(host_slots),
            stream=self.stream,
        )

    def _launch(self, device_slots, rows):
        """An overlap-scheduled forward writing rows of device_slots; they land
        in the capturer's host cache at the output's finalize()."""
        batch = SimpleNamespace(
            out_cache_loc_virtual=None,
            out_cache_loc=torch.tensor(device_slots, device="cuda"),
        )
        with mock.patch.object(
            self.capturer, "_get_local_slice", return_value=rows.cuda()
        ):
            output = self.capturer.on_forward_end(
                forward_batch=batch,
                can_run_graph=False,
                cuda_graph_batch=None,
                no_copy_to_cpu=True,
            )
        output.map_device_tensors(lambda tensor: tensor.cpu())
        return output

    def _write_prefix(self, host_slots=HOST_SLOTS):
        """Rows of device slots 5..7 are backed up to host slots 10..12, then
        the device slots are reused by another request."""
        self.capturer.host_cache.buffer[[5, 6, 7]] = self.expected
        self.rows.finish_write(self._copy([5, 6, 7], host_slots))
        self.capturer.host_cache.buffer[[5, 6, 7]] = -1

    def _load(self, device_slots, host_slots=HOST_SLOTS):
        row_copy = self._copy(device_slots, host_slots)
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

    def test_host_slots_past_the_physical_pool_size(self):
        logical_slots = [61, 62, 63]
        self._write_prefix(logical_slots)
        self._load([20, 21, 22], logical_slots)
        self.assertTrue(torch.equal(self._read([20, 21, 22]), self.expected))

    def test_write_waits_for_the_rows_of_the_forward_in_flight(self):
        # Overlap scheduling: a finished request is cached while the next
        # forward still writes the row of its last slot (7).
        self.capturer.host_cache.buffer[[5, 6]] = self.expected[:2]
        self.capturer.host_cache.buffer[7] = -1
        in_flight = self._launch([7], self.expected[2:])
        row_copy = self._copy([5, 6, 7])
        self.rows.finish_write(row_copy)  # the write ack comes first
        in_flight.finalize()
        self.capturer.host_cache.buffer[[5, 6, 7]] = -1
        self._load([20, 21, 22])
        self.assertTrue(torch.equal(self._read([20, 21, 22]), self.expected))

    def test_write_issued_after_the_rows_landed_copies_at_its_ack(self):
        self._launch([7], self.expected[2:]).finalize()
        self.capturer.host_cache.buffer[[5, 6]] = self.expected[:2]
        self.rows.finish_write(self._copy([5, 6, 7]))
        # A later forward reuses the freed slots.
        self._launch([5, 6, 7], torch.full_like(self.expected, -1)).finalize()
        self._load([20, 21, 22])
        self.assertTrue(torch.equal(self._read([20, 21, 22]), self.expected))

    def test_write_waiting_on_a_dropped_output_copies_before_newer_rows(self):
        self.capturer.host_cache.buffer[[5, 6, 7]] = self.expected
        # This output is never finalized.
        self._launch([7], torch.full_like(self.expected[2:], -1))
        self.rows.finish_write(self._copy([5, 6, 7]))
        self._launch([5, 6, 7], torch.full_like(self.expected, -1)).finalize()
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
