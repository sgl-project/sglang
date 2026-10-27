"""The native ACK publisher is shared by all policies, roles and cache modes."""

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from queue import Queue
from types import MethodType
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.srt.managers.cache_controller import PrefetchAck
from sglang.srt.managers.pp_prefetch_scheduler import PPPrefetchConsensus
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import published_topology

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


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
        stale, fresh = NS(is_pp_broadcast=False), NS(is_pp_broadcast=False)
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

        def stage_broadcast(value):
            values = [value]
            dist.broadcast_object_list(values, src=2 * pp, group=cc.tp_group)
            return values[0]

        with (
            published_topology(tp_size=2, pp_size=3, ranks={"world_rank": rank}),
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
                if tick == 16:
                    cc.pp_prefetch_ack_queue.put(
                        PrefetchAck("old", stale, completed_tokens=128)
                    )
                if tick == 18:
                    # Reset with a proposal in flight and an ACK still queued.
                    # The old return arrives after reset, before the fresh one.
                    cc.pp_prefetch_ack_queue.put(
                        PrefetchAck("old", stale, completed_req=True)
                    )
                    cc.pp_prefetch_consensus.reset()
                    assert cc.pp_prefetch_ack_queue.empty()
                    cc.pp_prefetch_ack_queue.put(
                        PrefetchAck("new", fresh, completed_tokens=64)
                    )
                s._pp_prefetch_step()
                if tick < 5:  # The slowest TP rank has no progress ACK yet.
                    assert cc.ack_prefetch_queue.empty()
                if tick < 7:
                    assert not any(a.completed_req for a in cc.ack_prefetch_queue.queue)
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
        results = [cc.ack_prefetch_queue.get_nowait() for _ in range(4)]
        assert all(ack.operation is operation for ack in results[:3])
        assert results[0].completed_tokens == 48
        assert results[1].pool_hits[PoolName.SWA] == 1
        assert results[1].pool_hits[PoolName.MAMBA] == 0
        assert results[2].completed_req
        assert results[3].operation is fresh
        assert results[3].completed_tokens == 64
        assert cc.ack_prefetch_queue.empty()
    finally:
        dist.destroy_process_group()


class TestPrefetchConsensus(unittest.TestCase):
    def test_tp2_pp3_slow_stage_reset_and_native_cache_event_order(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _gloo_completion,
                args=(str(Path(directory) / "completion"),),
                nprocs=6,
            )


if __name__ == "__main__":
    unittest.main()
