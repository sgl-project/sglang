"""Unit tests for the HiCache TP/PP write-through & load-back consensus.

Covers the sglang#28429 fix that removes the write-through eviction-desync
hang: host / device pool occupancy is per-rank physical state, so a naive
per-rank alloc can succeed on some ranks and fail on others, diverging the
radix tree and desyncing the next forward's TP collectives.

The fix splits the controller write into a two-phase reserve/commit/abort and
makes ``HiRadixCache.write_backup`` / ``HiRadixCache.load_back`` reach an
``all_reduce(MIN)`` consensus so every rank makes the SAME enqueue / load
decision (commit on all ranks or abort on all ranks).

These are CPU-only unit tests: they exercise the real reserve/commit/abort
logic and the real ``write_backup`` / ``load_back`` control flow by calling the
unbound methods against light fake-self carriers (no CUDA, no NCCL).
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import (
    CacheOperation,
    HiCacheController,
    WriteReservation,
)
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, IncLockRefResult
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _indices(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.int64)


def _make_all_reduce(other_ranks_min: int = 1):
    """Fake ``_all_reduce`` simulating an all_reduce(MIN) whose minimum value
    contributed by the *other* ranks is ``other_ranks_min``.

    other_ranks_min=1 -> every other rank succeeded (result == this rank's value)
    other_ranks_min=0 -> at least one other rank failed (result forced to 0)
    """

    def _all_reduce(data: torch.Tensor, op) -> None:
        data.fill_(min(int(data.item()), other_ranks_min))

    return _all_reduce


class _FakeHostPool:
    """Host pool stand-in tracking alloc/free with a fixed capacity."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.freed = []
        self.alloc_calls = 0

    def alloc(self, n: int):
        self.alloc_calls += 1
        if n <= self.capacity:
            self.capacity -= n
            return _indices(n)
        return None

    def free(self, indices) -> None:
        self.freed.append(indices)
        self.capacity += len(indices)


def _make_controller(capacity: int) -> HiCacheController:
    """Real HiCacheController exercising the two-phase reserve/commit/abort."""
    controller = HiCacheController.__new__(HiCacheController)
    controller.mem_pool_host = _FakeHostPool(capacity)
    controller.write_queue = []
    controller.start_writing = mock.Mock()
    return controller


class TestReserveCommitAbort(unittest.TestCase):
    """Phase-split contract of the base HiCacheController."""

    def test_reserve_success_no_side_effects(self):
        controller = _make_controller(capacity=8)
        reservation = controller.reserve_write(_indices(4), node_id=7)

        self.assertIsInstance(reservation, WriteReservation)
        self.assertEqual(reservation.node_id, 7)
        self.assertEqual(len(reservation.host_indices), 4)
        # reserve must NOT schedule DMA or enqueue an ack.
        self.assertEqual(controller.write_queue, [])
        controller.start_writing.assert_not_called()

    def test_reserve_failure_returns_none(self):
        controller = _make_controller(capacity=2)
        self.assertIsNone(controller.reserve_write(_indices(4)))
        self.assertEqual(controller.write_queue, [])

    def test_commit_enqueues_and_starts_writing(self):
        controller = _make_controller(capacity=8)
        reservation = controller.reserve_write(_indices(4), node_id=3, priority=5)
        host_indices = controller.commit_write(reservation)

        self.assertIs(host_indices, reservation.host_indices)
        self.assertEqual(len(controller.write_queue), 1)
        self.assertIsInstance(controller.write_queue[0], CacheOperation)
        controller.start_writing.assert_called_once()

    def test_abort_frees_host_and_leaves_queue_empty(self):
        controller = _make_controller(capacity=8)
        reservation = controller.reserve_write(_indices(4))
        controller.abort_write(reservation)

        self.assertEqual(controller.mem_pool_host.freed, [reservation.host_indices])
        # capacity fully restored, nothing enqueued.
        self.assertEqual(controller.mem_pool_host.capacity, 8)
        self.assertEqual(controller.write_queue, [])
        controller.start_writing.assert_not_called()

    def test_write_is_reserve_plus_commit(self):
        controller = _make_controller(capacity=8)
        host_indices = controller.write(_indices(4), node_id=1)

        self.assertIsNotNone(host_indices)
        self.assertEqual(len(controller.write_queue), 1)
        controller.start_writing.assert_called_once()

    def test_write_returns_none_on_alloc_failure(self):
        controller = _make_controller(capacity=1)
        self.assertIsNone(controller.write(_indices(4)))
        self.assertEqual(controller.write_queue, [])


def _make_write_backup_self(*, tp_world_size, capacity, other_ranks_min=1):
    """Fake-self for HiRadixCache.write_backup backed by a real controller."""
    fake = SimpleNamespace()
    fake.root_node = object()
    fake.cache_controller = _make_controller(capacity)
    fake.tp_world_size = tp_world_size
    fake.pp_size = 1
    fake.evict_host = mock.Mock()
    fake._get_extra_pools = lambda: {}
    fake._all_reduce = _make_all_reduce(other_ranks_min)
    fake._track_write_through_node = mock.Mock()
    fake.inc_lock_ref = mock.Mock()
    return fake


def _make_node(fake, *, value_len=4):
    node = SimpleNamespace()
    node.parent = fake.root_node
    node.id = 42
    node.value = _indices(value_len)
    node.key = list(range(value_len))
    node.host_value = None
    node.backuped = True
    return node


class TestWriteBackupConsensus(unittest.TestCase):
    def test_multi_rank_all_agree_commits(self):
        fake = _make_write_backup_self(tp_world_size=2, capacity=8, other_ranks_min=1)
        node = _make_node(fake)

        ret = HiRadixCache.write_backup(fake, node)

        self.assertEqual(ret, 4)
        self.assertEqual(len(fake.cache_controller.write_queue), 1)
        self.assertIsNotNone(node.host_value)
        fake.inc_lock_ref.assert_called_once_with(node)

    def test_multi_rank_other_rank_failed_aborts_local(self):
        # This rank reserved fine, but a peer failed -> everyone must abort.
        fake = _make_write_backup_self(tp_world_size=2, capacity=8, other_ranks_min=0)
        node = _make_node(fake)

        ret = HiRadixCache.write_backup(fake, node)

        self.assertEqual(ret, 0)
        # committed nothing, and freed the reservation this rank made.
        self.assertEqual(fake.cache_controller.write_queue, [])
        self.assertEqual(len(fake.cache_controller.mem_pool_host.freed), 1)
        fake.inc_lock_ref.assert_not_called()

    def test_multi_rank_local_failed_no_abort_no_commit(self):
        # This rank could not reserve (capacity 0, evict is a no-op) -> return 0
        # without a commit; abort must NOT be called (there is no reservation).
        fake = _make_write_backup_self(tp_world_size=2, capacity=0, other_ranks_min=1)
        node = _make_node(fake)

        ret = HiRadixCache.write_backup(fake, node)

        self.assertEqual(ret, 0)
        self.assertEqual(fake.cache_controller.write_queue, [])
        self.assertEqual(fake.cache_controller.mem_pool_host.freed, [])
        fake.evict_host.assert_called_once()

    def test_evict_host_retry_succeeds_then_commits(self):
        # First reserve fails (capacity=0), evict_host frees enough memory,
        # second reserve succeeds, consensus passes -> commit.
        fake = _make_write_backup_self(tp_world_size=2, capacity=0, other_ranks_min=1)
        node = _make_node(fake, value_len=4)

        # evict_host side-effect: simulate freeing host slots so retry works.
        def _evict_frees_capacity(num_tokens):
            fake.cache_controller.mem_pool_host.capacity += num_tokens

        fake.evict_host = mock.Mock(side_effect=_evict_frees_capacity)

        ret = HiRadixCache.write_backup(fake, node)

        self.assertEqual(ret, 4)
        fake.evict_host.assert_called_once_with(4)
        # Two alloc attempts: first fails, second succeeds after evict.
        self.assertEqual(fake.cache_controller.mem_pool_host.alloc_calls, 2)
        self.assertEqual(len(fake.cache_controller.write_queue), 1)
        self.assertIsNotNone(node.host_value)
        fake.inc_lock_ref.assert_called_once_with(node)

    def test_evict_host_retry_succeeds_but_consensus_aborts(self):
        # First reserve fails, evict_host frees memory, second reserve succeeds,
        # but a peer rank failed its reserve -> consensus aborts locally too.
        fake = _make_write_backup_self(tp_world_size=2, capacity=0, other_ranks_min=0)
        node = _make_node(fake, value_len=4)

        def _evict_frees_capacity(num_tokens):
            fake.cache_controller.mem_pool_host.capacity += num_tokens

        fake.evict_host = mock.Mock(side_effect=_evict_frees_capacity)

        ret = HiRadixCache.write_backup(fake, node)

        self.assertEqual(ret, 0)
        fake.evict_host.assert_called_once_with(4)
        self.assertEqual(fake.cache_controller.mem_pool_host.alloc_calls, 2)
        # Reserved succeeded after evict, but consensus said abort -> free it.
        self.assertEqual(len(fake.cache_controller.mem_pool_host.freed), 1)
        self.assertEqual(fake.cache_controller.write_queue, [])
        fake.inc_lock_ref.assert_not_called()

    def test_single_rank_skips_consensus_and_commits(self):
        fake = _make_write_backup_self(tp_world_size=1, capacity=8)
        # Guard against accidental collective use on a single rank.
        fake._all_reduce = mock.Mock(
            side_effect=AssertionError("no all_reduce on 1 rank")
        )
        node = _make_node(fake)

        ret = HiRadixCache.write_backup(fake, node)

        self.assertEqual(ret, 4)
        self.assertEqual(len(fake.cache_controller.write_queue), 1)

    def test_write_back_mode_skips_consensus(self):
        # write_back=True (host->device flush) must not run the write-through
        # consensus even under TP>1, and must not take the lock ref.
        fake = _make_write_backup_self(tp_world_size=2, capacity=8)
        fake._all_reduce = mock.Mock(
            side_effect=AssertionError("no consensus in write_back")
        )
        node = _make_node(fake)

        ret = HiRadixCache.write_backup(fake, node, write_back=True)

        self.assertEqual(ret, 4)
        self.assertEqual(len(fake.cache_controller.write_queue), 1)
        fake.inc_lock_ref.assert_not_called()


def _make_load_back_self(*, tp_world_size, load_results, other_ranks_min=1):
    """Fake-self for HiRadixCache.load_back.

    load_results: list passed as side_effect to cache_controller.load.
    """
    fake = SimpleNamespace()
    fake.tp_world_size = tp_world_size
    fake.pp_size = 1
    fake.load_back_threshold = 1

    controller = SimpleNamespace()
    controller.load = mock.Mock(side_effect=list(load_results))
    controller.mem_pool_device_allocator = SimpleNamespace(free=mock.Mock())
    fake.cache_controller = controller

    fake._get_extra_pools = lambda: {}
    fake._all_reduce = _make_all_reduce(other_ranks_min)
    fake.inc_lock_ref = mock.Mock(return_value=IncLockRefResult(delta=0))
    fake.dec_lock_ref = mock.Mock()
    fake.evict = mock.Mock()
    fake.ongoing_load_back = {}
    fake._record_store_event = mock.Mock()
    fake.evictable_size_ = 0
    fake.metrics_collector = None
    return fake


def _make_evicted_node(*, host_len=4):
    ancestor = SimpleNamespace(evicted=False)
    node = SimpleNamespace(
        evicted=True,
        backuped=True,
        parent=ancestor,
        id=99,
        host_value=_indices(host_len),
        value=None,
        protect_host=mock.Mock(),
        release_host=mock.Mock(),
    )
    return node


class TestLoadBackConsensus(unittest.TestCase):
    def test_multi_rank_all_load_finalizes(self):
        fake = _make_load_back_self(
            tp_world_size=2, load_results=[_indices(4)], other_ranks_min=1
        )
        node = _make_evicted_node()

        ret = HiRadixCache.load_back(fake, node)

        self.assertIsNotNone(ret)
        self.assertEqual(len(ret), 4)
        self.assertIn(node.id, fake.ongoing_load_back)
        self.assertIsNotNone(node.value)
        node.release_host.assert_called_once()
        fake.cache_controller.mem_pool_device_allocator.free.assert_not_called()
        # Multi-rank load-back must never evict-and-retry.
        fake.evict.assert_not_called()

    def test_multi_rank_other_rank_failed_rolls_back(self):
        # Local load succeeded, but a peer failed -> free device + release + None.
        fake = _make_load_back_self(
            tp_world_size=2, load_results=[_indices(4)], other_ranks_min=0
        )
        node = _make_evicted_node()

        ret = HiRadixCache.load_back(fake, node)

        self.assertIsNone(ret)
        fake.cache_controller.mem_pool_device_allocator.free.assert_called_once()
        node.release_host.assert_called_once()
        fake.dec_lock_ref.assert_called_once()
        self.assertNotIn(node.id, fake.ongoing_load_back)

    def test_multi_rank_local_failed_no_evict_and_rolls_back(self):
        # Local load fails -> no evict-retry under TP>1, roll back, no device free.
        fake = _make_load_back_self(
            tp_world_size=2, load_results=[None], other_ranks_min=1
        )
        node = _make_evicted_node()

        ret = HiRadixCache.load_back(fake, node)

        self.assertIsNone(ret)
        fake.evict.assert_not_called()
        fake.cache_controller.mem_pool_device_allocator.free.assert_not_called()
        node.release_host.assert_called_once()
        fake.dec_lock_ref.assert_called_once()

    def test_single_rank_evicts_and_retries(self):
        # Single rank: first load fails, evict, retry succeeds -> finalize.
        fake = _make_load_back_self(tp_world_size=1, load_results=[None, _indices(4)])
        fake._all_reduce = mock.Mock(
            side_effect=AssertionError("no all_reduce on 1 rank")
        )
        node = _make_evicted_node()

        ret = HiRadixCache.load_back(fake, node)

        self.assertIsNotNone(ret)
        fake.evict.assert_called_once()
        self.assertEqual(fake.cache_controller.load.call_count, 2)
        self.assertIn(node.id, fake.ongoing_load_back)


if __name__ == "__main__":
    unittest.main(verbosity=2)
