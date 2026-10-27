"""Striped HiCache host tier: forward-entry drain and gather-gated load acks."""

import ast
import inspect
import os
import tempfile
import textwrap
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers import cache_controller as cache_controller_module
from sglang.srt.managers.cache_controller import HiCacheAck, LayerDoneCounter
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.allocator.unified_sub_pool import MultiEndedAllocator
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host.striped_host import (
    StripedGatherError,
    StripedGatherExecutor,
    StripedHostPlacement,
    StripedLoadConsensusError,
    StripedLoadDrain,
    StripedLoadNotDrainedError,
    TorchDistStripedGatherTransport,
    build_striped_load_plan,
    plan_gathered,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

NUM_RANKS = 4


class _FakeEvent:
    """CPU stand-in for a device event; ``ready`` models an unfinished copy."""

    def __init__(self, ready=True):
        self.ready = ready
        self.waited = []

    def record(self, stream=None):
        pass

    def query(self):
        return self.ready

    def synchronize(self):
        assert self.ready, "synchronize on an unfinished event"

    def wait(self, stream=None):
        self.waited.append(stream)

    def elapsed_time(self, other):
        return 0.0


_FAKE_DEVICE_MODULE = SimpleNamespace(Event=_FakeEvent)


def _worker(drain):
    worker = TpModelWorker.__new__(TpModelWorker)
    worker.hicache_load_drain = None
    worker.register_hicache_load_drain(drain)
    return worker


def _batch(mode):
    return SimpleNamespace(forward_mode=mode, hicache_consumer_index=-1)


# ---- Divergence aborts before any gather ----


def _layers(seed):
    gen = torch.Generator().manual_seed(seed)
    return (
        (
            (torch.randint(0, 256, (16, 64), dtype=torch.uint8, generator=gen), 64),
            (torch.randint(0, 256, (16, 32), dtype=torch.uint8, generator=gen), 32),
        ),
    )


def _small_plan(placement, seq, num_pages=5):
    host_pages = torch.arange(num_pages)
    host = (host_pages[:, None] * 4 + torch.arange(4)).flatten()
    dev = (torch.arange(num_pages)[:, None] * 4 + torch.arange(4)).flatten()
    small = StripedHostPlacement(placement.num_ranks, placement.rank, 4)
    return build_striped_load_plan(
        small, host_indices=host, device_indices=dev, seq=seq, layers=_layers(seq)
    )


def _diverge(rank, group, scenario):
    placement = StripedHostPlacement(num_ranks=NUM_RANKS, rank=rank, page_size=4)
    staging = NUM_RANKS * 96 * (2 if scenario == "staging" and rank == 1 else 1)
    executor = StripedGatherExecutor(
        placement,
        TorchDistStripedGatherTransport(group),
        device=torch.device("cpu"),
        staging_bytes=staging,
    )
    drain = StripedLoadDrain(executor, group)
    drain.bind_forward_entry()
    drain.enqueue(_small_plan(placement, 0), None)
    if scenario == "count" and rank == 2:
        drain.enqueue(_small_plan(placement, 1), None)
    try:
        drain.drain()
    except StripedLoadConsensusError:
        raised = True
    else:
        raised = False
    assert raised, (scenario, rank)
    assert executor.num_gathers == 0
    assert drain.pending_count == (2 if scenario == "count" and rank == 2 else 1)


def _divergence_worker(rank, rendezvous):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=NUM_RANKS,
        timeout=timedelta(seconds=60),
    )
    group = torch.distributed.new_group(list(range(NUM_RANKS)))
    try:
        _diverge(rank, group, "count")
        # Same count and seq, different chunking (staging) on one rank.
        _diverge(rank, group, "staging")
    finally:
        torch.distributed.destroy_process_group()


class TestConsensusDivergenceGloo(unittest.TestCase):
    def test_pending_count_or_gather_shape_divergence_aborts_every_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
            torch.multiprocessing.spawn(
                _divergence_worker,
                args=(f"{directory}/store",),
                nprocs=NUM_RANKS,
                join=True,
            )


# ---- Local checks ----


class _SoloTransport:
    world_size = 1
    rank = 0

    def all_gather(self, recv, send):
        raise AssertionError("N=1 issues no collective")


def _solo_drain(on_gathered=None, bind=True):
    placement = StripedHostPlacement(num_ranks=1, rank=0, page_size=4)
    executor = StripedGatherExecutor(
        placement, _SoloTransport(), device=torch.device("cpu"), staging_bytes=4096
    )
    drain = StripedLoadDrain(executor, None, on_gathered=on_gathered)
    if bind:
        drain.bind_forward_entry()
    return placement, drain


class TestDrainLocal(unittest.TestCase):
    def test_fifo_seq_and_callbacks(self):
        seen = []
        placement, drain = _solo_drain(lambda plan, done: seen.append(plan.seq))
        plans = [_small_plan(placement, seq) for seq in range(3)]
        h2d = [_FakeEvent() for _ in plans]
        for plan, event in zip(plans, h2d):
            drain.enqueue(plan, event)
        with self.assertRaises(StripedGatherError):
            drain.enqueue(_small_plan(placement, 7), None)
        self.assertEqual(drain.pending_seqs(), [0, 1, 2])
        self.assertEqual(drain.drain(agreed=True), 3)
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(drain.pending_count, 0)
        for plan, event in zip(plans, h2d):
            self.assertTrue(plan_gathered(plan))
            self.assertEqual(event.waited, [None])
        # Nothing pending: no gather, no callback.
        self.assertEqual(drain.drain(agreed=True), 0)
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(drain.next_seq(), 3)
        drain.reset()
        self.assertEqual(drain.next_seq(), 0)

    def test_consensus_fields_and_check(self):
        placement, drain = _solo_drain()
        self.assertEqual(drain.consensus_fields()[:4], [0, 0, -1, 1])
        drain.enqueue(_small_plan(placement, 0), None)
        fields = drain.consensus_fields()
        self.assertEqual(fields[:4], [1, -1, 0, 0])
        self.assertEqual(drain.check_reduced(fields), 1)
        for index in range(len(fields)):
            bad = list(fields)
            bad[index] -= 1
            with self.assertRaises(StripedLoadConsensusError):
                drain.check_reduced(bad)
        with self.assertRaises(StripedLoadConsensusError):
            drain.check_reduced(fields[:4])
        # The signature digest tracks the collective shape, not just count and seq.
        _, other = _solo_drain()
        other.enqueue(_small_plan(placement, 0, num_pages=4), None)
        self.assertNotEqual(other.consensus_fields()[4], fields[4])

    def test_enqueue_refused_until_bound_to_the_forward_entry(self):
        placement, drain = _solo_drain(bind=False)
        with self.assertRaises(StripedGatherError):
            drain.enqueue(_small_plan(placement, 0), None)
        self.assertEqual((drain.pending_count, drain.next_seq()), (0, 0))
        _worker(drain)  # register_hicache_load_drain binds it
        drain.enqueue(_small_plan(placement, 0), None)
        self.assertEqual(drain.pending_count, 1)

    def test_gather_waits_the_plan_ready_event(self):
        """The plan's device tensors are built on the enqueue stream."""
        placement, drain = _solo_drain()
        ready, h2d = _FakeEvent(), _FakeEvent()
        with mock.patch.object(drain, "_record_plan_ready", return_value=ready):
            drain.enqueue(_small_plan(placement, 0), h2d)
        waits = []
        real_gather_plan = drain.executor.gather_plan

        def gather_plan(plan, *, stream, wait_events=()):
            waits.append(tuple(wait_events))
            return real_gather_plan(plan, stream=stream, wait_events=wait_events)

        with mock.patch.object(drain.executor, "gather_plan", gather_plan):
            self.assertEqual(drain.drain(agreed=True), 1)
        self.assertEqual(waits, [(ready, h2d)])
        self.assertEqual((ready.waited, h2d.waited), ([None], [None]))


class TestAckGating(unittest.TestCase):
    def _ack(self, h2d_ready=True, plan=None):
        return HiCacheAck(
            start_event=_FakeEvent(),
            finish_event=_FakeEvent(h2d_ready),
            node_ids=[1],
            gather_plan=plan,
        )

    def test_ready_count_requires_gather_done(self):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        placement, _ = _solo_drain()
        plan = _small_plan(placement, 0)
        queue = [self._ack(), self._ack(plan=plan), self._ack()]
        self.assertEqual(cache._count_ready_acks(queue), 1)
        plan.gather_done_event = _FakeEvent(ready=False)
        self.assertEqual(cache._count_ready_acks(queue), 1)
        plan.gather_done_event.ready = True
        self.assertEqual(cache._count_ready_acks(queue), 3)
        queue[1] = self._ack(h2d_ready=False, plan=plan)
        self.assertEqual(cache._count_ready_acks(queue), 1)

    def test_load_back_event_done_requires_gather(self):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        placement, _ = _solo_drain()
        plan = _small_plan(placement, 0)
        cc = HybridCacheController.__new__(HybridCacheController)
        with mock.patch.object(
            cache_controller_module, "device_module", _FAKE_DEVICE_MODULE
        ):
            cc.layer_done_counter = LayerDoneCounter(2)
        cc.striped_load_drain = object()
        cc._striped_plans_by_producer = {0: plan}
        cache.cache_controller = cc
        self.assertFalse(cache.is_load_back_event_done(0))
        plan.gather_done_event = _FakeEvent()
        self.assertTrue(cache.is_load_back_event_done(0))
        self.assertTrue(cache.is_load_back_event_done(-1))


class TestOrphanPolicy(unittest.TestCase):
    """A load still pending at a scheduler check skipped its forward gather."""

    def _cache(self, drain):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.cache_controller = SimpleNamespace(
            striped_load_drain=drain, load_fence_stream=None
        )
        return cache

    def _agree(self, cache, drain):
        # What _sync_hicache_ready_counts stores from the MIN-reduced fields.
        cache._agreed_striped_pending = drain.check_reduced(drain.consensus_fields())

    def test_aggregated_drain_refuses_an_orphan(self):
        placement, drain = _solo_drain()
        cache = self._cache(drain)
        self._agree(cache, drain)
        cache._drain_orphan_striped_loads()  # nothing pending: no-op
        plan = _small_plan(placement, 0)
        drain.enqueue(plan, None)
        self._agree(cache, drain)
        with self.assertRaises(StripedLoadNotDrainedError):
            cache._drain_orphan_striped_loads()
        self.assertFalse(plan.gather_started)
        self.assertEqual(drain.pending_count, 1)


class TestControllerWiring(unittest.TestCase):
    def test_gather_fences_device_frees_and_moves(self):
        """Peer rows land on the forward stream after the H2D: the device
        allocator must order frees and page moves after each plan's gather,
        without releasing the pending load-page reservation (a load key does)."""
        cc = HybridCacheController.__new__(HybridCacheController)
        placement, solo = _solo_drain()
        cc.host_stripe = placement
        cc.tp_group = None
        allocator = MultiEndedAllocator.__new__(MultiEndedAllocator)
        allocator._capacity_epoch = 0
        allocator._hicache_transfer_done_events = {}
        allocator._pending_hicache_load_pages = 3
        cc.mem_pool_device_allocator = allocator
        drain = cc.install_striped_load_drain(solo.executor)
        _worker(drain)
        plans = [_small_plan(placement, seq) for seq in range(2)]
        for plan in plans:
            drain.enqueue(plan, _FakeEvent())
        self.assertEqual(drain.drain(agreed=True), 2)
        self.assertEqual(
            allocator._hicache_transfer_done_events,
            {(id(cc), "striped_gather"): plans[-1].gather_done_event},
        )
        self.assertIsNotNone(plans[-1].gather_done_event)
        self.assertEqual(allocator._pending_hicache_load_pages, 3)

    def test_stock_controller_builds_no_plan(self):
        cc = HybridCacheController.__new__(HybridCacheController)
        cc.host_stripe = None
        self.assertIsNone(cc._enqueue_striped_load(None, [], None, 0))
        self.assertTrue(cc.striped_load_gathered(0))


class TestForwardEntryHook(unittest.TestCase):
    def test_only_extend_batches_drain(self):
        drain = mock.MagicMock()
        worker = _worker(drain)
        for mode in ForwardMode:
            drain.reset_mock()
            worker.drain_hicache_loads(_batch(mode))
            self.assertEqual(
                drain.drain.call_count,
                int(mode.is_extend_without_speculative()),
                mode,
            )
        worker.hicache_load_drain = None
        worker.drain_hicache_loads(_batch(ForwardMode.EXTEND))

    def test_split_prefill_refused_with_a_drain(self):
        """pdmux split prefill has no forward-entry gather: fail closed."""
        drain = mock.MagicMock()
        worker = _worker(drain)
        batch = SimpleNamespace(forward_mode=ForwardMode.SPLIT_PREFILL, split_index=0)
        with self.assertRaises(NotImplementedError):
            worker.forward_batch_split_prefill(batch)
        drain.drain.assert_not_called()

    def test_drain_runs_at_the_forward_entry(self):
        """Right after set_hicache_consumer, before the forward batch is built."""
        tree = ast.parse(
            textwrap.dedent(inspect.getsource(TpModelWorker.forward_batch_generation))
        )
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        lines = {}
        for node in calls:
            lines.setdefault(node.func.attr, []).append(node.lineno)
        self.assertEqual(len(lines["drain_hicache_loads"]), 1)
        drain_line = lines["drain_hicache_loads"][0]
        self.assertEqual(drain_line, lines["set_hicache_consumer"][0] + 1)
        self.assertLess(drain_line, min(lines["init_new"]))
        self.assertLess(drain_line, min(lines["forward"]))


if __name__ == "__main__":
    unittest.main()
