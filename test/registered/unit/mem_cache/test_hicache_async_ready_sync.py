from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Ack:
    def __init__(self, name, ready=False):
        self.name = name
        self.ready = ready
        self.finish_event = self

    def query(self):
        return self.ready

    def synchronize(self):
        assert self.ready, f"{self.name} consumed before it finished"

    def __repr__(self):
        return self.name


class _FakeGroup:
    """MIN all_reduce across ranks; the result is applied when a rank waits."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.rounds = []
        self.issued = [0] * world_size

    def all_reduce(self, rank, tensor):
        idx = self.issued[rank]
        self.issued[rank] += 1
        if idx == len(self.rounds):
            self.rounds.append({})
        self.rounds[idx][rank] = tensor
        group = self

        class _Work:
            def wait(self_inner):
                peers = group.rounds[idx]
                assert len(peers) == group.world_size, "wait before all ranks issued"
                tensor.copy_(torch.stack(list(peers.values())).min(dim=0).values)

        return _Work()


class _Rank:
    _count_ready_acks = UnifiedRadixCache._count_ready_acks
    _deferred_consume_count = staticmethod(UnifiedRadixCache._deferred_consume_count)
    _check_hicache_events_deferred = UnifiedRadixCache._check_hicache_events_deferred
    _drop_deferred_ready_sync = UnifiedRadixCache._drop_deferred_ready_sync
    _consume_deferred_ready_sync = UnifiedRadixCache._consume_deferred_ready_sync
    _deferred_ready_sync_group = UnifiedRadixCache._deferred_ready_sync_group

    def __init__(self, rank, group):
        self.rank = rank
        self.group = group
        self.enable_storage = False
        self._hcs_state = object()
        self.cache_controller = SimpleNamespace(ack_write_queue=[], ack_load_queue=[])
        self.tree_core = SimpleNamespace(write_back_duplicate_reclaim_digest=0)
        self._hcs_pending = None
        self.written = []
        self.loaded = []

    def writing_check(self, write_back=False, finish_count=None):
        q = self.cache_controller.ack_write_queue
        if write_back:
            for ack in q:
                ack.ready = True
                self.written.append(ack.name)
            q.clear()
            return
        for _ in range(finish_count):
            ack = q.pop(0)
            ack.finish_event.synchronize()
            self.written.append(ack.name)

    def loading_check(self, finish_count=None):
        q = self.cache_controller.ack_load_queue
        for _ in range(finish_count):
            ack = q.pop(0)
            ack.finish_event.synchronize()
            self.loaded.append(ack.name)

    def step(self):
        def fake_all_reduce(tensor, op=None, group=None, async_op=False):
            assert op == torch.distributed.ReduceOp.MIN and async_op
            return self.group.all_reduce(self.rank, tensor)

        with mock.patch.object(torch.distributed, "all_reduce", fake_all_reduce):
            self._check_hicache_events_deferred(group=None)


def _ranks(n=2):
    group = _FakeGroup(n)
    return [_Rank(r, group) for r in range(n)]


def _step_all(ranks):
    for r in ranks:
        r.step()


def _enqueue(ranks, kind, names):
    """Same acks on every rank, independent objects, initially not ready."""
    out = []
    for r in ranks:
        acks = [_Ack(n) for n in names]
        getattr(r.cache_controller, f"ack_{kind}_queue").extend(acks)
        out.append(acks)
    return out


def test_consume_count_finds_agreed_ack():
    a, b, c = _Ack("a"), _Ack("b"), _Ack("c")
    snap = [a, b, c]
    assert UnifiedRadixCache._deferred_consume_count([a, b, c], snap, 0) == 0
    assert UnifiedRadixCache._deferred_consume_count([a, b, c], snap, 2) == 2
    # Head already partly consumed: count is relative to the current head.
    assert UnifiedRadixCache._deferred_consume_count([b, c], snap, 2) == 1
    # Agreed ack already consumed by a drain: consume nothing.
    assert UnifiedRadixCache._deferred_consume_count([c], snap, 2) == 0
    assert UnifiedRadixCache._deferred_consume_count([], snap, 3) == 0


def test_result_is_consumed_one_step_later():
    ranks = _ranks()
    acks = _enqueue(ranks, "write", ["w0", "w1"])
    for per_rank in acks:
        for ack in per_rank:
            ack.ready = True

    _step_all(ranks)
    assert all(r.written == [] for r in ranks)

    _step_all(ranks)
    assert all(r.written == ["w0", "w1"] for r in ranks)


def test_ranks_stay_in_lockstep_under_skew():
    ranks = _ranks()
    w = _enqueue(ranks, "write", ["w0", "w1", "w2"])
    ld = _enqueue(ranks, "load", ["l0", "l1"])
    # Rank 0 is ahead: everything ready. Rank 1 only has the first of each.
    for ack in w[0] + ld[0]:
        ack.ready = True
    w[1][0].ready = True
    ld[1][0].ready = True

    _step_all(ranks)  # step N: issue MIN([3, 1]) = 1 write, MIN([2, 1]) = 1 load

    # Between steps rank 1 catches up and new acks land at the tail.
    for ack in w[1] + ld[1]:
        ack.ready = True
    new_w = _enqueue(ranks, "write", ["w3"])
    for per_rank in new_w:
        per_rank[0].ready = True

    _step_all(ranks)  # step N + 1: consume the agreed prefix of step N only
    for r in ranks:
        assert r.written == ["w0"]
        assert r.loaded == ["l0"]

    _step_all(ranks)
    for r in ranks:
        assert r.written == ["w0", "w1", "w2", "w3"]
        assert r.loaded == ["l0", "l1"]
    assert ranks[0].written == ranks[1].written


def test_drain_between_steps_is_not_double_consumed():
    ranks = _ranks()
    acks = _enqueue(ranks, "write", ["w0", "w1"])
    for per_rank in acks:
        for ack in per_rank:
            ack.ready = True

    _step_all(ranks)  # agreed: 2 writes ready in step N

    # A blocking write-back drain on every rank consumes the queue, then a new
    # (not yet ready) ack is enqueued before step N + 1.
    for r in ranks:
        r.writing_check(write_back=True)
    late = _enqueue(ranks, "write", ["w2"])

    _step_all(ranks)  # must not pop w2 on the strength of step N's count of 2
    for r in ranks:
        assert r.written == ["w0", "w1"]
        assert [a.name for a in r.cache_controller.ack_write_queue] == ["w2"]

    for per_rank in late:
        per_rank[0].ready = True
    _step_all(ranks)
    _step_all(ranks)
    for r in ranks:
        assert r.written == ["w0", "w1", "w2"]


def test_digest_divergence_asserts_one_step_later():
    ranks = _ranks()
    ranks[1].tree_core.write_back_duplicate_reclaim_digest = 7
    _step_all(ranks)
    with pytest.raises(AssertionError, match="duplicate-reclaim victims diverged"):
        ranks[0].step()


def test_runtime_storage_attach_falls_back_to_sync():
    ranks = _ranks()
    acks = _enqueue(ranks, "write", ["w0"])
    for per_rank in acks:
        per_rank[0].ready = True
    _step_all(ranks)
    for r in ranks:
        assert r._deferred_ready_sync_group() is r._hcs_state

    for r in ranks:
        r.enable_storage = True
    for r in ranks:
        assert r._deferred_ready_sync_group() is None
        assert r.written == ["w0"]
        assert r._hcs_pending is None
        r.enable_storage = False
        assert r._deferred_ready_sync_group() is None


def test_drop_waits_and_clears_pending():
    ranks = _ranks()
    _step_all(ranks)
    for r in ranks:
        assert r._hcs_pending is not None
        r._drop_deferred_ready_sync()
        assert r._hcs_pending is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
