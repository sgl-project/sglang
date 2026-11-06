"""HiCache carry of captured routing rows between device and host KV slots."""

import random
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.state_capturer import hicache_carry
from sglang.srt.state_capturer.base import TopkCaptureOutput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

LAYERS, TOPK, DEVICE_ROWS, HOST_ROWS = 3, 4, 512, 1024


def _capturer(name="routed_experts"):
    buffer = torch.zeros((DEVICE_ROWS, LAYERS, TOPK), dtype=torch.int32)
    return SimpleNamespace(
        host_cache=SimpleNamespace(
            buffer=buffer, num_layers=LAYERS, topk_size=TOPK, name=name
        )
    )


def _rows(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 1 << 20, (n, LAYERS, TOPK), generator=g, dtype=torch.int32)


def _slots(*ranges):
    return torch.cat([torch.arange(a, b) for a, b in ranges]).to(torch.int64)


def _req(slots):
    pool = SimpleNamespace(req_to_token=slots.reshape(1, -1).to(torch.int32))
    return SimpleNamespace(
        kv=SimpleNamespace(req_pool_idx=0), extend_end=slots.numel()
    ), pool


class _CarryTestCase(CustomTestCase):
    def setUp(self):
        hicache_carry.reset()
        self.experts = _capturer("routed_experts")
        self.indexer = _capturer("indexer_topk")
        patcher = mock.patch.object(
            hicache_carry, "_capturers", return_value=[self.experts, self.indexer]
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(hicache_carry.reset)

    def _write(self, slots, rows):
        for capturer in (self.experts, self.indexer):
            capturer.host_cache.buffer[slots] = rows

    def _output(self, slots, rows):
        """A capture output created by a forward and finalized later, as under overlap."""
        return [
            TopkCaptureOutput(out_cache_loc=slots, topk=rows, host_cache=c.host_cache)
            for c in (self.experts, self.indexer)
        ]

    def _assert_device(self, slots, rows):
        for capturer in (self.experts, self.indexer):
            self.assertTrue(torch.equal(capturer.host_cache.buffer[slots], rows))


class TestBackupAndLoad(_CarryTestCase):
    def test_a_host_hit_restores_the_rows_into_new_device_slots(self):
        """#26975: load-back moves KV to new device slots, which must get its routing."""
        for device, host, new_device in (
            (_slots((0, 128)), _slots((300, 428)), _slots((200, 328))),
            (_slots((5, 9), (40, 44)), _slots((900, 908)), _slots((60, 64), (7, 11))),
        ):
            rows = _rows(device.numel(), seed=1)
            self._write(device, rows)
            hicache_carry.backup(device, host, HOST_ROWS)
            self._write(device, torch.zeros_like(rows))

            hicache_carry.load(host, new_device)

            self._assert_device(new_device, rows)

    def test_a_host_row_that_was_never_backed_up_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "never backed up"):
            hicache_carry.load(_slots((10, 20)), _slots((0, 10)))

    def test_a_failed_overwrite_does_not_expose_the_previous_rows(self):
        device, host = _slots((0, 8)), _slots((100, 108))
        self._write(device, _rows(8, seed=2))
        hicache_carry.backup(device, host, HOST_ROWS)

        with mock.patch.object(hicache_carry, "_copy_rows", side_effect=OSError):
            with self.assertRaises(OSError):
                hicache_carry.backup(device, host, HOST_ROWS)

        with self.assertRaisesRegex(RuntimeError, "never backed up"):
            hicache_carry.load(host, device)

    def test_a_larger_host_pool_keeps_rows_already_carried(self):
        device, host = _slots((0, 8)), _slots((0, 8))
        rows = _rows(8, seed=3)
        self._write(device, rows)
        hicache_carry.backup(device, host, 16)
        hicache_carry.backup(_slots((8, 9)), _slots((600, 601)), HOST_ROWS)

        hicache_carry.load(host, _slots((100, 108)))

        self._assert_device(_slots((100, 108)), rows)

    def test_mismatched_slot_counts_are_refused(self):
        with self.assertRaisesRegex(ValueError, "device slots for"):
            hicache_carry.backup(_slots((0, 4)), _slots((0, 5)), HOST_ROWS)


class TestInflightChunks(_CarryTestCase):
    def _stash(self, req_slots, device, host):
        req, pool = _req(req_slots)
        with hicache_carry.deferring_inflight(req, lambda: pool):
            hicache_carry.backup(device, host, HOST_ROWS)

    def test_a_chunk_stashed_before_its_capture_is_finalized_carries_the_final_rows(
        self,
    ):
        """Overlap scheduling stashes the chunk while its forward's rows are still pending."""
        slots, host = _slots((0, 64)), _slots((500, 564))
        final = _rows(64, seed=4)
        outputs = self._output(slots, final)

        self._stash(slots, slots, host)
        with self.assertRaisesRegex(RuntimeError, "never backed up"):
            hicache_carry.load(host, _slots((200, 264)))

        for output in outputs:
            output.finalize()
        hicache_carry.load(host, _slots((200, 264)))

        self._assert_device(_slots((200, 264)), final)

    def test_without_the_deferral_the_rows_carried_are_stale(self):
        """Control: the pre-fix order backs up the rows before the forward wrote them."""
        slots, host = _slots((0, 64)), _slots((500, 564))
        outputs = self._output(slots, _rows(64, seed=5))
        hicache_carry.backup(slots, host, HOST_ROWS)
        for output in outputs:
            output.finalize()

        hicache_carry.load(host, _slots((200, 264)))

        self._assert_device(
            _slots((200, 264)), torch.zeros((64, LAYERS, TOPK), dtype=torch.int32)
        )

    def test_only_the_stashed_requests_rows_wait(self):
        req_slots, other = _slots((0, 32)), _slots((100, 132))
        other_rows = _rows(32, seed=6)
        self._write(other, other_rows)
        outputs = self._output(req_slots, _rows(32, seed=7))

        self._stash(req_slots, torch.cat([req_slots, other]), _slots((500, 564)))

        hicache_carry.load(_slots((532, 564)), _slots((300, 332)))
        self._assert_device(_slots((300, 332)), other_rows)
        with self.assertRaisesRegex(RuntimeError, "never backed up"):
            hicache_carry.load(_slots((500, 532)), _slots((200, 232)))
        for output in outputs:
            output.finalize()

    def test_it_waits_for_every_output_in_flight_at_the_stash(self):
        """Pipelining can hold several of the request's chunks in flight at once."""
        first, second = _slots((0, 32)), _slots((32, 64))
        first_rows, second_rows = _rows(32, seed=8), _rows(32, seed=9)
        first_outputs = self._output(first, first_rows)
        second_outputs = self._output(second, second_rows)
        host = _slots((500, 564))

        self._stash(_slots((0, 64)), _slots((0, 64)), host)
        for output in first_outputs:
            output.finalize()
        with self.assertRaisesRegex(RuntimeError, "never backed up"):
            hicache_carry.load(host, _slots((200, 264)))
        for output in second_outputs:
            output.finalize()

        hicache_carry.load(host, _slots((200, 264)))
        self._assert_device(_slots((200, 264)), torch.cat([first_rows, second_rows]))

    def test_a_host_row_backed_up_again_is_not_overwritten_by_the_late_deferral(self):
        slots, host = _slots((0, 16)), _slots((500, 516))
        outputs = self._output(slots, _rows(16, seed=10))
        self._stash(slots, slots, host)

        newer, newer_rows = _slots((100, 116)), _rows(16, seed=11)
        self._write(newer, newer_rows)
        hicache_carry.backup(newer, host, HOST_ROWS)
        for output in outputs:
            output.finalize()

        hicache_carry.load(host, _slots((200, 216)))
        self._assert_device(_slots((200, 216)), newer_rows)

    def test_nothing_is_deferred_when_no_capture_is_in_flight(self):
        slots, host = _slots((0, 16)), _slots((500, 516))
        rows = _rows(16, seed=12)
        self._write(slots, rows)

        self._stash(slots, slots, host)

        hicache_carry.load(host, _slots((200, 216)))
        self._assert_device(_slots((200, 216)), rows)
        self.assertFalse(hicache_carry._deferred)

    def test_randomized_overlap_schedules_restore_the_routing_each_token_produced(self):
        """Chunks run, are stashed before or after their result, evicted and loaded back."""
        rng = random.Random(0)
        free_device = list(range(DEVICE_ROWS))
        free_host = list(range(HOST_ROWS))
        expected = {}  # host slot -> rows its token produced
        checked = 0
        for step in range(400):
            n = rng.randint(1, 24)
            if len(free_device) < 2 * n or len(free_host) < n:
                break
            device = torch.tensor([free_device.pop() for _ in range(n)])
            rows = _rows(n, seed=1000 + step)
            outputs = self._output(device, rows)
            host = torch.tensor([free_host.pop() for _ in range(n)])
            stash_first = rng.random() < 0.5
            if not stash_first:
                for output in outputs:
                    output.finalize()
            self._stash(device, device, host)
            if stash_first:
                for output in outputs:
                    output.finalize()
            for slot, row in zip(host.tolist(), rows):
                expected[slot] = row
            # Evict: the device slots go to another request.
            self._write(device, torch.zeros_like(rows))
            free_device[:0] = device.tolist()
            if rng.random() < 0.7:
                target = torch.tensor([free_device.pop() for _ in range(n)])
                hicache_carry.load(host, target)
                for capturer in (self.experts, self.indexer):
                    restored = capturer.host_cache.buffer[target]
                    self.assertTrue(
                        torch.equal(
                            restored, torch.stack([expected[s] for s in host.tolist()])
                        )
                    )
                checked += n
                free_device[:0] = target.tolist()
            # The host copy is dropped; its slots serve a later eviction.
            free_host[:0] = host.tolist()
        self.assertGreater(checked, 1000)
        self.assertFalse(hicache_carry._deferred)


if __name__ == "__main__":
    unittest.main()
