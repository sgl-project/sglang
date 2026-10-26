"""The native ACK publisher is shared by all policies, roles and cache modes."""

import tempfile
import unittest
from collections import deque
from datetime import timedelta
from pathlib import Path
from queue import Queue
from types import MethodType, SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.srt.managers.cache_controller import HiCacheController, PrefetchAck
from sglang.srt.managers.pp_prefetch_scheduler import PPPrefetchConsensus
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def _controller(rank, size):
    cc = NS(
        pp_group=None,
        attn_cp_group=None,
        attn_tp_group=None,
        tp_group=None,
        pp_prefetch_ack_queue=Queue(),
        ack_prefetch_queue=Queue(),
    )
    cc._publish_prefetch_ack = MethodType(HiCacheController._publish_prefetch_ack, cc)
    with (
        patch.object(dist, "get_rank", return_value=rank),
        patch.object(dist, "get_world_size", return_value=size),
    ):
        cc.pp_prefetch_consensus = PPPrefetchConsensus(cc)
    return cc


def _gloo_completion(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=6,
        timeout=timedelta(seconds=90),
    )
    try:
        pp_groups = [dist.new_group(list(range(t, 6, 2))) for t in range(2)]
        tp_groups = [dist.new_group([2 * p, 2 * p + 1]) for p in range(3)]
        pp, tp = rank // 2, rank % 2
        cc = HybridCacheController.__new__(HybridCacheController)
        cc.pp_group, cc.tp_group = pp_groups[tp], tp_groups[pp]
        cc.attn_cp_group, cc.attn_tp_group = None, cc.tp_group
        cc.prefetch_completion_sync_groups = [cc.tp_group]
        cc.pp_prefetch_ack_queue, cc.ack_prefetch_queue = Queue(), Queue()
        cc.pp_prefetch_consensus = PPPrefetchConsensus(cc)
        operation = NS(is_pp_broadcast=False)
        acks = [
            PrefetchAck("r", operation, completed_tokens=128 - rank * 16),
            PrefetchAck(
                "r",
                operation,
                pool_hits={
                    PoolName.SWA: 1 if pp == 1 else 2,
                    PoolName.MAMBA: 0 if pp == 2 else 1,
                },
            ),
            PrefetchAck("r", operation, completed_req=True),
        ]
        for ack in acks:
            cc._reduce_prefetch_ack(ack)

        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tree_core = NS(enable_storage=True)
        cache.cache_controller, cache.enable_storage = cc, True
        cache.pp_group, cache.pp_rank, cache.pp_size = cc.pp_group, pp, 3
        cache.work_list = []
        s = NS(
            enable_hierarchical_cache=True,
            tree_cache=cache,
            world_group=NS(cpu_group=dist.group.WORLD),
            pp_loop_size=4,
        )
        for name in (
            "_pp_prefetch_step",
            "_pp_recv_pyobj_from_prev_stage",
            "_pp_send_pyobj_to_next_stage",
        ):
            setattr(s, name, MethodType(getattr(SchedulerPPMixin, name), s))
        topology = NS(
            attn_tp_rank=tp,
            attn_cp_rank=0,
            attn_dp_rank=0,
            attn_cp_size=1,
            attn_tp_size=2,
            tp_size=2,
            pp_rank=pp,
            pp_size=3,
        )

        def stage_broadcast(value):
            values = [value]
            dist.broadcast_object_list(values, src=2 * pp, group=cc.tp_group)
            return values[0]

        with (
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
                return_value=topology,
            ),
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.attn_cp_tp_broadcast_pyobj",
                stage_broadcast,
            ),
        ):
            request_work = []
            for tick in range(24):
                # Native request relay shares WORLD with completion, but has its
                # own tag. Downstream observes requests before completion ticks.
                if pp:
                    assert s._pp_recv_pyobj_from_prev_stage() == [tick]
                for i, ack in enumerate(acks):
                    if tick == 2 * pp + tp + i:
                        cc.pp_prefetch_ack_queue.put(ack)
                s._pp_prefetch_step()
                # Native cache progress remains PP0-led and may consume every
                # committed ACK now; no peer waits for a not-yet-published ACK.
                cache._drain_async_work()
                counts = torch.tensor([tick, cc.ack_prefetch_queue.qsize()])
                cache._pp_sync(counts)
                assert counts.tolist() == [tick, cc.ack_prefetch_queue.qsize()]
                for work in request_work:
                    work.work.wait()
                if pp < 2:
                    request_work = s._pp_send_pyobj_to_next_stage(
                        [tick], async_send=True
                    )

            # Drain the bounded empty tail of the continuously running ring.
            if pp == 0:
                from sglang.srt.distributed.communication_tags import P2PTag

                for _ in range(s.pp_loop_size):
                    s._pp_recv_pyobj_from_prev_stage(
                        tag=P2PTag.HICACHE_PREFETCH_COMPLETION
                    )
            tail_work = [
                w for batch in cc.pp_prefetch_consensus.send_work for w in batch
            ]
            for work in tail_work + request_work:
                work.work.wait()
            cache._drain_async_work()
        assert not cc.pp_prefetch_consensus.pending
        assert not cc.pp_prefetch_consensus.inflight
        assert cc.pp_prefetch_ack_queue.empty()
        results = [cc.ack_prefetch_queue.get_nowait() for _ in range(3)]
        assert all(ack.operation is operation for ack in results)
        assert results[0].completed_tokens == 48
        assert results[1].pool_hits[PoolName.SWA] == 1
        assert results[1].pool_hits[PoolName.MAMBA] == 0
        assert results[2].completed_req
    finally:
        dist.destroy_process_group()


class TestPrefetchConsensus(unittest.TestCase):
    def test_slow_stage_defers_ack_without_blocking_or_publishing_terminal(self):
        for size in (2, 3, 8):
            with self.subTest(pp=size):
                controllers = [_controller(i, size) for i in range(size)]
                inboxes = [deque() for _ in range(size)]
                operations = [object() for _ in controllers]
                for i, cc in enumerate(controllers[:-1]):
                    cc.pp_prefetch_ack_queue.put(
                        PrefetchAck("r", operations[i], completed_tokens=128)
                    )
                    cc.pp_prefetch_ack_queue.put(
                        PrefetchAck("r", operations[i], completed_req=True)
                    )
                for tick in range(6 * size):
                    if tick == 2 * size:
                        cc = controllers[-1]
                        cc.pp_prefetch_ack_queue.put(
                            PrefetchAck("r", operations[-1], completed_tokens=64)
                        )
                    if tick == 3 * size:
                        cc.pp_prefetch_ack_queue.put(
                            PrefetchAck("r", operations[-1], completed_req=True)
                        )
                    for i, cc in enumerate(controllers):

                        def send(update, i=i):
                            inboxes[(i + 1) % size].append(update)
                            return []

                        cc.pp_prefetch_consensus.advance(inboxes[i].popleft, send, size)
                    if tick < 2 * size:
                        self.assertTrue(
                            all(c.ack_prefetch_queue.empty() for c in controllers)
                        )
                for i, cc in enumerate(controllers):
                    self.assertFalse(cc.pp_prefetch_consensus.pending)
                    self.assertFalse(cc.pp_prefetch_consensus.inflight)
                    progress, terminal = [
                        cc.ack_prefetch_queue.get_nowait() for _ in range(2)
                    ]
                    self.assertEqual(progress.completed_tokens, 64)
                    self.assertIs(progress.operation, operations[i])
                    self.assertTrue(terminal.completed_req)

    def test_tp2_pp3_real_ring_and_native_cache_event_order(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _gloo_completion,
                args=(str(Path(directory) / "completion"),),
                nprocs=6,
            )

    def test_reset_and_detach_discard_unpublished_results(self):
        controllers = [_controller(i, 3) for i in range(3)]
        inboxes = [deque() for _ in controllers]
        old, new = object(), object()
        for cc in controllers:
            cc.pp_prefetch_ack_queue.put(PrefetchAck("r", old, completed_tokens=128))
        for tick in range(12):
            if tick == 2:
                for cc in controllers:
                    cc.pp_prefetch_consensus.reset()
                    cc.pp_prefetch_ack_queue.put(
                        PrefetchAck("r", new, completed_tokens=64)
                    )
            for i, cc in enumerate(controllers):

                def send(update, i=i):
                    inboxes[(i + 1) % 3].append(update)
                    return []

                cc.pp_prefetch_consensus.advance(inboxes[i].popleft, send, 3)
        for cc in controllers:
            result = cc.ack_prefetch_queue.get_nowait()
            self.assertIs(result.operation, new)
            self.assertEqual(result.completed_tokens, 64)
            self.assertTrue(cc.ack_prefetch_queue.empty())
            # Native detach cleans up operations directly, without ACK drain.
            cc.pp_prefetch_ack_queue.put(PrefetchAck("r", new, completed_tokens=128))
            cc.pp_prefetch_ack_queue.put(PrefetchAck("r", new, completed_req=True))
            cc.pp_prefetch_consensus.reset()
            self.assertFalse(cc.pp_prefetch_consensus.pending)
            self.assertTrue(cc.pp_prefetch_ack_queue.empty())
            self.assertTrue(cc.ack_prefetch_queue.empty())


if __name__ == "__main__":
    unittest.main()
