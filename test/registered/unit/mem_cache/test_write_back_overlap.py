"""CPU regressions for HiCache decode-overlap ownership and rollback."""

import unittest
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.hybrid_cache.write_back_overlap import WriteBackOverlap
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestWriteBackOverlap(CustomTestCase):
    def test_blocking_write_drains_deferred_full_ack_before_forward(self):
        controller = Mock(ack_write_queue=[])
        controller._num_tokens_by_pool.return_value = {}
        controller._transfer_num_bytes.return_value = 0
        full_done = Mock()
        op = SimpleNamespace(node_ids=[1], device_indices=torch.arange(64))
        overlap = object.__new__(WriteBackOverlap)
        overlap.controller = controller
        overlap.after_forward = [(op, [], full_done)]
        overlap.background = deque()
        cache = SimpleNamespace(
            cache_controller=controller,
            write_back_overlap=overlap,
            ongoing_write_through={1: object()},
            _fenced_write_acks=[],
            _log_write_ack_metrics=Mock(),
        )

        def finish(node):
            full_done.synchronize.assert_called_once_with()
            cache.ongoing_write_through.pop(node)

        cache._finish_write_through_ack = finish
        UnifiedRadixCache.writing_check(cache, write_back=True)
        UnifiedRadixCache.writing_check(cache, write_back=True)
        self.assertFalse(cache.ongoing_write_through)
        self.assertFalse(overlap.after_forward)
        self.assertFalse(controller.ack_write_queue)
        full_done.synchronize.assert_called_once_with()

    def test_ranks_disagree_about_eviction_and_both_fall_back(self):
        barrier = Barrier(2, timeout=5)
        ballots = [None, None]

        def select(rank):
            def reduce(vote, op):
                self.assertEqual(op, torch.distributed.ReduceOp.MIN)
                ballots[rank] = vote.clone()
                barrier.wait()
                vote.copy_(torch.minimum(*ballots))

            overlap = object.__new__(WriteBackOverlap)
            overlap.cache = SimpleNamespace(
                tp_world_size=2,
                full_evictable_size=lambda: 64 if rank == 0 else 0,
                _all_reduce=reduce,
            )
            overlap._pick_leaves = lambda _: [
                (1, torch.arange(64), {}, SimpleNamespace(state={}))
            ]
            overlap._can_stage = lambda _: True
            overlap._allocate_transfers = Mock()
            result = overlap._select_transfers(None, 64, enabled=True)
            overlap._allocate_transfers.assert_not_called()
            return result

        with ThreadPoolExecutor(2) as workers:
            self.assertEqual(list(workers.map(select, range(2))), [(False, None)] * 2)

    def test_successful_local_allocation_rolls_back_on_peer_failure(self):
        host_ids = torch.arange(64)
        op = SimpleNamespace(host_indices=host_ids, pool_transfers=[])
        overlap = object.__new__(WriteBackOverlap)
        overlap.controller = Mock(write_queue=[op])
        overlap.cache = SimpleNamespace(_execute_kv_backup=lambda *args: host_ids)
        overlap._unanimous = lambda locally_ready: False
        leaf = (
            1,
            torch.arange(64),
            {},
            SimpleNamespace(full=SimpleNamespace(cpu=host_ids)),
        )

        self.assertIsNone(overlap._allocate_transfers([leaf]))
        self.assertFalse(overlap.controller.write_queue)
        overlap.controller.mem_pool_host.free.assert_called_once_with(host_ids)


if __name__ == "__main__":
    unittest.main()
