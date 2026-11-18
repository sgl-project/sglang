"""External linker tickets, admission, and ownership over two Gloo stages."""

import tempfile
import threading
import time
import unittest
from array import array
from concurrent.futures import Future
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_unified_cache_linker import _FakeLinker

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache import fixtures
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class _StageStore(_FakeLinker):
    load_mode = "prefetch"
    prefetch_policy = "wait_complete"

    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.first_lookup = threading.Event()
        self.reads = []
        self.requests_by_first_page = {}
        self.slow_read = None

    def lookup(self, rid, transfers):
        self.requests_by_first_page[transfers[0].keys[0]] = rid
        if rid == "hit":
            self.first_lookup.set()
            return [1, 3, 5] if self.rank == 0 else [2, 3, 5]
        if rid == "miss" and self.rank == 1:
            return []
        return [2]

    def prefetch(self, transfers):
        keys = transfers[0].keys
        rid = self.requests_by_first_page[keys[0]]
        self.reads.append((rid, len(keys)))
        done = Future()
        if rid == "slow":
            done.set_running_or_notify_cancel()
            self.slow_read = done
        else:
            done.set_result(rid != "failed-get" or self.rank == 0)
        return done


def _request(rid, first_token, size=7):
    return Req(
        rid=rid,
        origin_input_text=None,
        origin_input_ids=array("q", range(first_token, first_token + size)),
        sampling_params=SamplingParams(max_new_tokens=1),
    )


def _drain_until(cache, condition):
    deadline = time.monotonic() + 20
    while True:
        cache.check_hicache_events()
        ready = torch.tensor(int(condition()))
        dist.all_reduce(ready, op=dist.ReduceOp.MIN)
        if ready.item():
            return
        assert time.monotonic() < deadline, "stage did not reach the expected state"


def _check_overlapping_reads(cache, store, allocator):
    """A running GET must leave the next ticket free to start and survive a skip."""
    cache.prefetch_stop_policy = "timeout"
    cache.prefetch_timeout_base = cache.prefetch_timeout_per_page = 0
    slow, fast = _request("slow", 40), _request("fast", 50)
    tickets = cache.linker.pp_tickets
    assert tickets.bind(slow, expected=True)
    assert tickets.bind(fast, expected=True)
    _drain_until(cache, lambda: ("fast", 2) in store.reads)
    assert store.slow_read.running()
    if store.rank == 0:
        assert not tickets.can_admit(slow.cache_request_handle)
    cache.check_hicache_events()
    assert tickets.can_admit(slow.cache_request_handle)
    assert store.slow_read.running()
    if store.rank == 0:
        store.slow_read.set_result(True)
    dist.barrier()
    cache.check_hicache_events()
    assert slow.cache_request_handle in cache.linker.prefetches
    assert allocator.available_size() == 32 - 5 - 2 - 2
    if store.rank == 1:
        store.slow_read.set_result(True)
    _drain_until(cache, lambda: not tickets.pending)
    assert not cache.linker.prefetches
    for req, expected in ((slow, 0), (fast, 2)):
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(req.origin_input_ids[:-1]))
        )
        assert match.device_prefix_len == expected
    assert allocator.available_size() == 32 - 5 - 2


def _check_blocked_follower_lookup(cache, store):
    tickets = cache.linker.pp_tickets
    for policy in ("best_effort", "timeout"):
        cache.prefetch_stop_policy = policy
        cache.prefetch_timeout_base = cache.prefetch_timeout_per_page = 0
        entered, resume, timed_out = (threading.Event() for _ in range(3))
        lookup = store.lookup

        def blocked_lookup(rid, transfers):
            if store.rank == 1:
                entered.set()
                if not resume.wait(10):
                    timed_out.set()
            return lookup(rid, transfers)

        with patch.object(store, "lookup", blocked_lookup):
            req = _request(policy, 60)
            handle = req.cache_request_handle
            assert tickets.bind(req, expected=True)
            try:
                if store.rank == 0:
                    tickets.states[handle].lookup.result(timeout=10)
                else:
                    assert entered.wait(10)
                dist.barrier()
                # PP0 has voted its local query ready. The follower must still
                # return from its scheduler poll while its own query is blocked.
                cache.check_hicache_events()
                if store.rank == 0:
                    assert not tickets.can_admit(handle)
                # PP1's candidate loop did not reach the blocked request. It
                # still applies PP0's stop decision in the next regular tick.
                cache.check_hicache_events()
                assert tickets.can_admit(handle)
            finally:
                resume.set()
            blocked = torch.tensor(int(timed_out.is_set()))
            dist.all_reduce(blocked, op=dist.ReduceOp.MAX)
            assert not blocked.item(), "scheduler waited on a follower's storage query"
            _drain_until(cache, lambda: not tickets.pending)
            assert all(rid != policy for rid, _ in store.reads)


def _check_layer_wise(cache, allocator, rank):
    store = _FakeLinker()
    store.load_mode = "layer_wise"
    queries = []
    probed_suffix = threading.Event()

    def lookup(rid, transfers):
        pages = len(transfers[0].keys)
        queries.append((rid, pages))
        if pages == 4:
            probed_suffix.set()
            return [2, 4]  # Queried after PP0's two local pages.
        return [] if rid == "gap" else [1, 3, 4]  # From this stage's root.

    store.lookup = lookup
    cache.init_cache_linker(store)
    tickets = cache.linker.pp_tickets
    gap, hit = _request("gap", 70), _request("layer-hit", 70)
    if rank == 0:
        cache.insert(
            InsertParams(
                key=RadixKey(gap.origin_input_ids[:2]), value=allocator.alloc(2)
            )
        )
    with patch.object(store, "prefetch", wraps=store.prefetch) as full_get:
        if rank == 1:
            assert probed_suffix.wait(10), "early lookup did not use PP0's L1 boundary"
        for req in (gap, hit):
            assert tickets.bind(req, expected=True)
        _drain_until(cache, lambda: not tickets.pending)
        assert gap.external_cache_hit_length == 0
        assert hit.external_cache_hit_length == 4
        for req, expected in ((gap, 0), (hit, 2 if rank == 0 else 4)):
            req.init_next_round_input(cache, cow_mamba=False)
            assert req.host_hit_length == expected, (rank, req.rid, req.host_hit_length)
            tickets.release(req.cache_request_handle)
        if rank == 1:
            assert [pages for rid, pages in queries if rid == "gap"] == [4, 6]
        full_get.assert_not_called()


def _check_unbound_and_retracted_tickets(cache, store):
    tickets = cache.linker.pp_tickets
    req = _request("retry", 80)
    rejected = req.cache_request_handle
    if store.rank == 1:
        tickets.skip_unbound(req.rid)
    dist.barrier()  # The follower rejected the relay before receiving the ticket.
    if store.rank == 0:
        assert tickets.bind(req, expected=False)
    empty = _request("empty", 90, size=1)
    for retracted in (empty, req):
        retracted.reset_for_retract()
        retracted.advance_cache_request_handle()
        cache.linker.prefetch(retracted)
        assert retracted.cache_request_handle in tickets.pending

    if store.rank == 0:
        assert not tickets.can_admit(rejected)
    # PP1 already rejected the request and never polls it for admission.
    # The ordered exchange carries three tickets despite that asymmetry.
    cache.check_hicache_events()
    assert tickets.can_admit(rejected)
    reads_before = len(store.reads)
    _drain_until(cache, lambda: not tickets.pending)
    assert store.reads[reads_before:] == [("retry", 2)]
    assert rejected not in cache.prefetch_loaded_tokens_by_reqid
    assert empty.cache_request_handle not in cache.prefetch_loaded_tokens_by_reqid
    assert cache.prefetch_loaded_tokens_by_reqid[req.cache_request_handle] == 2

    # Stop with a locked L1 anchor on PP0 and no matching request on PP1.
    protected = cache.full_protected_size()
    if store.rank == 0:
        assert tickets.bind(_request("pending-reset", 80), expected=False)
        assert cache.full_protected_size() > protected
    dist.barrier()
    cache.linker.reset()
    assert not tickets.pending and not tickets.states
    assert cache.full_protected_size() == protected
    active = cache.linker.pp_tickets
    following = _request("after-reset", 100)
    assert active.bind(following, expected=True)
    _drain_until(cache, lambda: not active.pending)
    assert cache.prefetch_loaded_tokens_by_reqid[following.cache_request_handle] == 2


def _run_stage(rank, init_method):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=init_method,
        timeout=timedelta(seconds=40),
    )
    with patch.object(fixtures, "get_device", return_value="cpu"):
        cache, allocator, _ = fixtures.build_fixture(
            CacheConfig(page_size=1, num_layers=1, kv_size=32),
            tree_core_backend="python",
        )
    cache.pp_group = dist.group.WORLD
    cache.pp_size, cache.pp_rank = 2, rank
    store = _StageStore(rank)
    cache.init_cache_linker(store)
    tickets = cache.linker.pp_tickets
    try:
        # Stage 1 can reserve four slots. Although both stores offer page 5,
        # only page 3 is both restorable and backed by local reservations.
        held = allocator.alloc(28) if rank == 1 else None
        requests = [
            _request(rid, i * 10)
            for i, rid in enumerate(("hit", "miss", "failed-get", "next"))
        ]
        if rank == 1:
            # Hold the real request relay until this stage has queried its store.
            assert store.first_lookup.wait(20), "ticket did not precede request relay"
            assert not store.reads
        for req in requests:
            assert tickets.bind(req, expected=rank == 1)
            assert not tickets.can_admit(req.cache_request_handle)

        deadline = time.monotonic() + 30
        while not tickets.can_admit(requests[-1].cache_request_handle):
            cache.check_hicache_events()
            if held is not None and tickets.can_admit(requests[0].cache_request_handle):
                allocator.free(held)
                held = None
            assert time.monotonic() < deadline, "ticket queue did not drain"

        assert store.reads == [("hit", 3), ("failed-get", 2), ("next", 2)]
        for req, expected in zip(requests, (3, 0, 0, 2)):
            assert tickets.can_admit(req.cache_request_handle)
            result = cache.match_prefix(
                MatchPrefixParams(key=RadixKey(req.origin_input_ids[:-1]))
            )
            assert result.device_prefix_len == expected, (rank, req.rid, result)
            tickets.release(req.cache_request_handle)
        assert not cache.linker.prefetches
        assert allocator.available_size() == 32 - 3 - 2
        _check_overlapping_reads(cache, store, allocator)
        _check_blocked_follower_lookup(cache, store)
        _check_unbound_and_retracted_tickets(cache, store)
        cache.linker.close()
        _check_layer_wise(cache, allocator, rank)
    finally:
        if store.slow_read is not None and not store.slow_read.done():
            store.slow_read.set_result(False)
        cache.linker.close()
        dist.destroy_process_group()


class TestLinkerPPTickets(CustomTestCase):
    def test_prefetch_and_layer_wise_admission_across_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            rendezvous = (Path(tmp) / "init").as_uri()
            mp.spawn(_run_stage, args=(rendezvous,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
