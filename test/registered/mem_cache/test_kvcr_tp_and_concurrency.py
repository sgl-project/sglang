"""KVCRStore under TP>1 colocation and under concurrent in-flight operations.

Everything shipped so far was validated at TP=1 with one request at a time, so
the two assumptions most likely to be wrong in production are untested:

  1. **Rank colocation (TP, and DP on top of it).** Every
     ``(dp, attn_cp, attn_tp)`` rank of one engine builds its own KVCRStore in
     its own process on the same host, and each rank holds a *different* slice
     of every attention head. Two things therefore have to follow the rank
     rather than the config: the port this rank *binds* (a collision is silent
     -- ``ZmqPeerControlChannel`` binds from the progress thread, so the engine
     still starts and registers, and only peer fetches break), and the port this
     rank *dials* on the source (a mismatch is worse than silent: the fetch
     succeeds and returns another rank's shard, because block keys are token
     hashes carrying no rank identity). The bind side takes the full coordinate;
     the dial side takes only the within-DP part, because the router has already
     resolved which DP rank of the source to talk to.

  2. **Concurrent operations.** ``poll_completed()`` both drains a queue and
     advances state machines, so the source pump and ``_drain_until`` race for
     every completion. ``_completed_ops`` exists to keep the loser's result from
     being dropped; these tests drive that path directly with several waiters in
     flight rather than trusting it by inspection.

CPU-only, no torch pool, no real NIXL, no kvcr wheel: KVCRStore is built without
a mem_pool so ``_build_kvcr`` never runs, and the core is a fake. That keeps this
on the cheapest CI tier alongside the schema tests.

    python -m pytest test/registered/mem_cache/test_kvcr_tp_and_concurrency.py -v
"""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
from unittest import mock

import torch

from kvcr.policy import FIFOPolicy, LRUPolicy
from kvcr.types import OpEntryStatus
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.storage.kvcr import kvcr_store
from sglang.srt.mem_cache.storage.kvcr.kvcr_store import KVCRStore
from sglang.srt.mem_cache.storage.kvcr.router_hint import ROUTER_HINT_KEY

try:
    from sglang.test.ci.ci_register import register_cpu_ci

    register_cpu_ci(est_time=10, suite="base-a-test-cpu")
except Exception:  # pragma: no cover - registration is CI-only
    pass


_BASE_CONTROL_PORT = 25000


def _storage_config(
    tp_rank: int,
    tp_size: int,
    dp_rank: int = 0,
    dp_size: int = 1,
    attn_cp_rank: int = 0,
    attn_cp_size: int = 1,
    **extra,
) -> HiCacheStorageConfig:
    """One scheduler's config.

    ``tp_rank``/``tp_size`` are attention-scoped whenever DP attention is on --
    ``cache_controller._generate_storage_config`` substitutes ``attn_tp_*`` for
    them there -- so a DP=2/attn_tp=2 engine is spelled here as four configs with
    ``tp_size=2``, not ``tp_size=4``.
    """
    extra_config = {
        "local_dram_bytes": 1 << 20,
        "control_host": "127.0.0.1",
        "control_port": _BASE_CONTROL_PORT,
        "control_advertise_host": "127.0.0.1",
        "enable_remote_hint": True,
    }
    extra_config.update(extra)
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="test-model",
        dp_rank=dp_rank,
        dp_size=dp_size,
        extra_config=extra_config,
    )


def _store(
    tp_rank: int,
    tp_size: int,
    dp_rank: int = 0,
    dp_size: int = 1,
    attn_cp_rank: int = 0,
    attn_cp_size: int = 1,
    **extra,
) -> KVCRStore:
    """A KVCRStore with no mem_pool, so the core is never constructed.

    Everything these tests assert on -- agent name, control endpoint, listen
    port -- is decided from config and rank before any pool is registered.
    """
    return KVCRStore(
        _storage_config(
            tp_rank,
            tp_size,
            dp_rank,
            dp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=attn_cp_size,
            **extra,
        ),
        mem_pool=None,
    )


def _hint_extra_info(endpoint: str) -> HiCacheStorageExtraInfo:
    """An extra_info carrying one router hint, as the controller threads it."""
    return HiCacheStorageExtraInfo(
        extra_info={
            ROUTER_HINT_KEY: {
                "source_control_endpoint": endpoint,
                "block_hashes": ["0123456789abcdef"],
            }
        }
    )


class FakeEntry:
    """One completed block key. _drain_until reads ``.success`` off each entry.

    ``status`` mirrors the core's ``OpEntryResult``, where ``success`` is
    ``status is SUCCESS`` -- so DROPPED and FAILED are both falsy and only the
    status tells them apart. It defaults to ``None`` so the many cases here that
    only care about pass/fail stay readable, which also stands in for an entry
    type that predates the field.
    """

    def __init__(self, success: bool = True, status=None) -> None:
        self.success = success
        if status is not None:
            self.status = status


def _per_op_residue(store: KVCRStore, op_handle: int) -> List[str]:
    """Names of every store container that still keys on ``op_handle``.

    Written as a scan rather than as a check of one named attribute because the
    guarded property is about the *class* of state, not about a particular
    field: no bookkeeping keyed by op handle may outlive the operation. Bounding
    that is what makes the store safe in a scheduler-lifetime process, and a
    rewrite is free to rename or reshape the containers.
    """
    return sorted(
        name
        for name, value in vars(store).items()
        if isinstance(value, (dict, set)) and op_handle in value
    )


def _raise_fault(*_args, **_kwargs):
    raise RuntimeError("kvcr core fault")


class _ExplodingKVCR:
    """A core whose every entry point raises.

    Stands in for the faults that arrive as an exception rather than as a
    result: a version skew that moved a method, a NIXL agent in a bad state, an
    internal assertion. ``FakeKVCR`` cannot model these -- it reports failure
    through the normal result path, which is exactly the path that does not run
    when the core raises.
    """

    deposit = _raise_fault
    deliver = _raise_fault
    discard_hint = _raise_fault
    poll_completed = _raise_fault
    query = _raise_fault
    submit_hint = _raise_fault


class FakeKVCR:
    """Records deliver() calls and reports completions only when told to.

    Nothing completes on its own: a test calls ``finish(op_handle, keys)`` to
    place a result in the queue, so the interleaving between waiters is chosen by
    the test rather than by timing.
    """

    def __init__(self) -> None:
        self._pending: List[Tuple[int, Dict]] = []
        self._lock = threading.Lock()
        self.poll_calls = 0
        self.next_handle = 100

    def deliver(self, destinations, request_id=None) -> int:
        with self._lock:
            self.next_handle += 1
            return self.next_handle

    def deliver_and_finish(self, keys: List[str]) -> int:
        """Complete the op inside the submit call, as a local-tier hit does.

        The local tier moves bytes with a NIXL transfer addressed to our own
        agent, which the progress thread can retire between the submit
        returning and the caller's first poll -- microseconds against a 5 ms
        pump interval. So this is the ordinary case, not a contrived one.
        """
        handle = self.deliver(destinations={})
        self.finish(handle, keys)
        return handle

    def finish(self, op_handle: int, keys: List[str]) -> None:
        with self._lock:
            self._pending.append((op_handle, {key: FakeEntry() for key in keys}))

    def poll_completed(self):
        with self._lock:
            self.poll_calls += 1
            drained = self._pending
            self._pending = []
            return drained


class TPColocationTest(unittest.TestCase):
    """Two ranks of one engine, same host, same extra_config."""

    def test_agent_name_is_unique_per_rank(self):
        rank0 = _store(0, 2)
        rank1 = _store(1, 2)

        self.assertNotEqual(rank0._agent_name, rank1._agent_name)
        self.assertIn("tp0", rank0._agent_name)
        self.assertIn("tp1", rank1._agent_name)

    def test_agent_name_is_unique_across_restarts_of_the_same_rank(self):
        """A restarted rank must not reuse a name a peer may still have cached."""
        first = _store(0, 2)
        second = _store(0, 2)

        self.assertNotEqual(first._agent_name, second._agent_name)

    def test_control_port_is_offset_by_tp_rank(self):
        """The whole point: one configured port must not mean one bound port.

        ZmqPeerControlChannel binds from the progress thread, so two ranks
        sharing a port fail in the background -- the engine comes up, registers,
        advertises an endpoint, and only the peer fetches break. Offsetting by
        rank is what the dynamo side already assumes when it publishes one
        endpoint per DP rank.
        """
        ports = {rank: _store(rank, 4)._control_port() for rank in range(4)}

        self.assertEqual(
            ports,
            {
                0: _BASE_CONTROL_PORT,
                1: _BASE_CONTROL_PORT + 1,
                2: _BASE_CONTROL_PORT + 2,
                3: _BASE_CONTROL_PORT + 3,
            },
        )

    def test_ephemeral_control_port_is_not_offset(self):
        """port 0 means "ask the OS", and the OS already guarantees distinctness.

        Offsetting an ephemeral port would land on an arbitrary in-use port.

        Local-only, because ``KVCRBackendConfig`` refuses port 0 together with
        ``enable_remote_hint`` -- an OS-assigned port cannot be registered for
        peers to dial (see ``test_kvcr_config_validation.py``). Colocated ranks
        still each bind one, so they must still not collide.
        """
        rank0 = _store(0, 2, control_port=0, enable_remote_hint=False)
        rank1 = _store(1, 2, control_port=0, enable_remote_hint=False)

        self.assertNotEqual(rank0._control_port(), rank1._control_port())
        self.assertGreater(rank0._control_port(), 0)
        self.assertGreater(rank1._control_port(), 0)

    def test_tp1_keeps_the_configured_port_verbatim(self):
        """The single-rank case must stay byte-identical to what was validated."""
        self.assertEqual(_store(0, 1)._control_port(), _BASE_CONTROL_PORT)


class SourceEndpointRankTest(unittest.TestCase):
    """Which source port each rank dials for a hint-driven fetch.

    Regression for a wrong-shard bug seen on real hardware at TP=2: both ranks
    of the target dialed the single endpoint dynamo advertises, so rank 1 pulled
    rank 0's KV. It does not fail -- block keys are token hashes with no rank
    identity, so rank 1 accepts the shard and the request reports a full cache
    hit. The only outward symptom was that greedy decoding produced different
    text than the same prefix computed locally.

    dynamo cannot fix this on its side: it indexes workers by ``(worker_id,
    dp_rank)`` and has no TP concept, so the endpoint it advertises is
    necessarily the engine's base. Realigning it onto the local rank is the
    consumer's job, and mirrors ``_control_port()`` on the bind side.
    """

    def _dialed(self, tp_rank: int, tp_size: int, endpoint: str) -> str:
        store = _store(tp_rank, tp_size)
        hint = store._parse_hint(_hint_extra_info(endpoint))
        return hint.source_control_endpoint

    def test_each_rank_dials_the_matching_rank_of_the_source(self):
        dialed = {
            rank: self._dialed(rank, 4, "tcp://10.0.0.7:25000") for rank in range(4)
        }

        self.assertEqual(
            dialed,
            {
                0: "tcp://10.0.0.7:25000",
                1: "tcp://10.0.0.7:25001",
                2: "tcp://10.0.0.7:25002",
                3: "tcp://10.0.0.7:25003",
            },
        )

    def test_tp1_dials_the_advertised_endpoint_unchanged(self):
        """The validated single-rank path must not move."""
        self.assertEqual(
            self._dialed(0, 1, "tcp://10.0.0.7:25000"), "tcp://10.0.0.7:25000"
        )

    def test_rank_alignment_survives_an_ipv6_source_host(self):
        """The offset is applied to the port, not to whatever precedes it.

        A bracketed IPv6 literal contains colons of its own, so splitting on the
        first colon (or on any colon) rewrites the address instead of the port.
        """
        self.assertEqual(
            self._dialed(1, 2, "tcp://[fd00::1]:25000"), "tcp://[fd00::1]:25001"
        )

    def test_an_unparseable_endpoint_drops_the_hint(self):
        """A rank that cannot align must fetch nothing, not fetch wrongly.

        Returning the endpoint verbatim would put rank 1 back on rank 0's shard
        -- the exact silent corruption this alignment exists to prevent. Dropping
        the hint costs a recompute and stays correct.
        """
        store = _store(1, 2)

        self.assertIsNone(store._parse_hint(_hint_extra_info("tcp://10.0.0.7")))


class DPColocationTest(unittest.TestCase):
    """Attention DP: several DP ranks of one engine on one host.

    SGLang runs one scheduler per ``(dp, attn_cp, attn_tp)`` rank, so a DP=2 /
    attn_tp=2 engine is four processes sharing one ``extra_config`` -- twice as
    many as the TP-only case the port offset was written for. Offsetting by
    ``tp_rank`` alone makes DP rank 1 land on DP rank 0's two ports, which is the
    same silent bind collision ``TPColocationTest`` covers, just reached through
    the other dimension.

    dynamo already advertises one endpoint per DP rank, so it is the side that
    has to know the stride between those blocks; these tests pin the stride the
    engine actually uses so a change here shows up as a failure rather than as a
    wrong-shard fetch on hardware.
    """

    def test_every_scheduler_of_a_dp_engine_binds_a_distinct_port(self):
        """The whole point: four processes, four ports, no overlap."""
        ports = [
            _store(tp_rank, 2, dp_rank, 2)._control_port()
            for dp_rank in range(2)
            for tp_rank in range(2)
        ]

        self.assertEqual(len(set(ports)), len(ports))

    def test_dp_ranks_occupy_consecutive_port_blocks(self):
        """DP rank r owns ``[base + r*attn_tp_size, +attn_tp_size)``.

        This is the layout dynamo mirrors when it publishes per-DP-rank
        endpoints, so it is a contract between the two repos, not an internal
        detail: dynamo must stride by ``attn_tp_size``, not by 1.
        """
        ports = {
            (dp_rank, tp_rank): _store(tp_rank, 2, dp_rank, 2)._control_port()
            for dp_rank in range(2)
            for tp_rank in range(2)
        }

        self.assertEqual(
            ports,
            {
                (0, 0): _BASE_CONTROL_PORT,
                (0, 1): _BASE_CONTROL_PORT + 1,
                (1, 0): _BASE_CONTROL_PORT + 2,
                (1, 1): _BASE_CONTROL_PORT + 3,
            },
        )

    def test_attn_cp_is_part_of_the_rank_coordinate(self):
        """CP shards attention too, so two CP ranks must not share a port.

        With DP=2 / attn_cp=2 / attn_tp=1 the engine is again four schedulers,
        and a formula that only multiplies ``dp_rank`` by ``tp_size`` would give
        every DP rank the same two ports.
        """
        ports = {
            (dp_rank, cp_rank): _store(
                0, 1, dp_rank, 2, attn_cp_rank=cp_rank, attn_cp_size=2
            )._control_port()
            for dp_rank in range(2)
            for cp_rank in range(2)
        }

        self.assertEqual(
            ports,
            {
                (0, 0): _BASE_CONTROL_PORT,
                (0, 1): _BASE_CONTROL_PORT + 1,
                (1, 0): _BASE_CONTROL_PORT + 2,
                (1, 1): _BASE_CONTROL_PORT + 3,
            },
        )

    def test_dp_disabled_keeps_the_tp_only_offsets(self):
        """Without DP attention, ``tp_rank`` already spans the whole engine.

        ``cache_controller`` reports ``dp_size=1`` and the full ``tp_rank`` then,
        so re-deriving the offset from ``dp_rank * attn_tp_size`` would be wrong.
        This is the configuration validated on hardware; it must not move.
        """
        ports = {rank: _store(rank, 4)._control_port() for rank in range(4)}

        self.assertEqual(
            ports,
            {
                0: _BASE_CONTROL_PORT,
                1: _BASE_CONTROL_PORT + 1,
                2: _BASE_CONTROL_PORT + 2,
                3: _BASE_CONTROL_PORT + 3,
            },
        )

    def test_agent_name_distinguishes_dp_ranks(self):
        """Same TP rank, different DP rank: NIXL still needs two names.

        Peers key their remote-agent tables by name, so a collision here would
        make one rank's registration overwrite the other's.
        """
        rank0 = _store(0, 2, dp_rank=0, dp_size=2)
        rank1 = _store(0, 2, dp_rank=1, dp_size=2)

        self.assertNotEqual(rank0._agent_name, rank1._agent_name)

    def test_the_dialed_source_port_ignores_our_dp_rank(self):
        """The router already picked the source DP rank; we only add our own
        within-DP offset.

        The hint's endpoint names one specific ``(worker, dp_rank)`` source that
        the router chose for its cached prefix -- our own DP rank says nothing
        about which of the source's DP ranks holds that prefix. Adding it would
        walk off into a different DP group's port block, and because block keys
        are token hashes with no rank identity, that fetch would succeed and
        return the wrong shard.
        """
        dialed = {
            (dp_rank, tp_rank): _store(tp_rank, 2, dp_rank, 2)
            ._parse_hint(_hint_extra_info("tcp://10.0.0.7:25000"))
            .source_control_endpoint
            for dp_rank in range(2)
            for tp_rank in range(2)
        }

        self.assertEqual(
            dialed,
            {
                (0, 0): "tcp://10.0.0.7:25000",
                (0, 1): "tcp://10.0.0.7:25001",
                (1, 0): "tcp://10.0.0.7:25000",
                (1, 1): "tcp://10.0.0.7:25001",
            },
        )


class UnusableHostPoolTest(unittest.TestCase):
    """Host pool layouts the backend cannot run against must fail at startup."""

    def test_a_per_layer_pool_refuses_to_start_the_backend(self):
        """A pool with no single kv_buffer tensor cannot be NIXL-registered.

        ``framework_dram`` is the one registration covering the engine's host KV
        pool, and *both* directions address that pool: deposit names host pages
        as transfer sources, deliver names them as destinations. Even the
        local-tier copy goes through NIXL (a loopback transfer to our own
        agent), so there is no direction left that works unregistered.

        The old code logged a warning, passed ``framework_dram=None``, and
        started anyway -- so a DeepSeek-V4-style per-layer pool would come up
        healthy and then name unregistered memory in every transfer it ever
        issued.
        """
        store = _store(0, 1)
        # Probe-able (so the local tier sizes fine) but kv_buffer is a per-layer
        # list, which is the layout with no single contiguous region.
        pool = SimpleNamespace(
            page_size=64,
            kv_buffer=[object(), object()],
            get_page_buffer_meta=lambda indices: ([1024, 2048], [16, 16]),
        )

        with self.assertRaises(RuntimeError) as raised:
            store.register_mem_pool_host(pool)

        self.assertIn("kv_buffer", str(raised.exception))

    def test_a_failed_page_probe_refuses_to_start_the_backend(self):
        """The local DRAM tier is the backend's only storage, so it is required.

        ``_local_dram_region`` sizes that tier from a probe of the host pool's
        zero-copy meta, and the probe can come back empty (a pool layout we do
        not understand). Leaving the tier unconfigured is worse than not
        starting: ``deposit`` then fails each entry *individually*, which reads
        downstream as "the cache never hits" -- indistinguishable from a cold
        server -- while every peer serve silently has nothing to offer, because
        this backend never hands out framework memory (see ``pin_adapter``).
        """
        store = _store(0, 1)
        # A registrable kv_buffer, so this gets past the framework_dram gate and
        # fails on the probe alone -- page_size is set, so the probe is reached,
        # and the empty meta is what makes it come back unusable.
        pool = SimpleNamespace(
            page_size=64,
            kv_buffer=torch.empty(64, dtype=torch.uint8),
            get_page_buffer_meta=lambda indices: ([], []),
        )

        with self.assertRaises(RuntimeError) as raised:
            store.register_mem_pool_host(pool)

        self.assertIn("local DRAM tier", str(raised.exception))


class ConcurrentDrainTest(unittest.TestCase):
    """_drain_until and the source pump racing for the same completion queue."""

    def setUp(self) -> None:
        self.store = _store(0, 1)
        self.core = FakeKVCR()
        self.store._kvcr = self.core

    def test_completion_drained_by_the_pump_still_reaches_its_waiter(self):
        """The exact case _completed_ops exists for.

        The pump drains the queue on its interval, so a get's own completion is
        routinely observed by a thread that is not waiting on it. If the pump
        dropped it, the waiter would spin to its deadline and report a miss --
        a silent recompute, not an error.

        The waiter is registered before the pump runs because that is the
        production ordering: ``_submit_and_wait`` registers under the same lock
        it holds across the submit, so no completion can exist before its
        waiter does. Only the *pop* is racy, and that is what this drives.
        """
        self.store._register_waiter(101)
        self.core.finish(101, ["seg-a"])

        self.store._poll_once(self.core)  # pump drains it first
        result = self.store._drain_until(101, timeout_s=1.0)

        self.assertEqual(result, {"seg-a": True})

    def test_concurrent_waiters_each_get_their_own_result(self):
        """Two in-flight gets must not read each other's entries."""
        results: Dict[int, Optional[Dict]] = {}
        barrier = threading.Barrier(3)

        def wait_for(handle: int) -> None:
            # Registering before the barrier mirrors _submit_and_wait, which
            # registers under the lock it holds across the submit -- so the
            # completions this test queues cannot predate their waiters.
            self.store._register_waiter(handle)
            barrier.wait()
            results[handle] = self.store._drain_until(handle, timeout_s=5.0)

        threads = [
            threading.Thread(target=wait_for, args=(handle,))
            for handle in (201, 202)
        ]
        for thread in threads:
            thread.start()
        barrier.wait()

        # Completed out of order, and both landing in one poll round.
        self.core.finish(202, ["seg-b"])
        self.core.finish(201, ["seg-a"])
        for thread in threads:
            thread.join(timeout=10)

        self.assertEqual(results, {201: {"seg-a": True}, 202: {"seg-b": True}})

    def test_stash_does_not_leak_after_delivery(self):
        """A result handed to its waiter must be removed from the stash.

        _completed_ops is keyed by op handle and the core hands out a fresh one
        per operation, so a stash that only ever grows is an unbounded leak in a
        long-running server.
        """
        self.core.finish(301, ["seg-a"])
        self.store._drain_until(301, timeout_s=1.0)

        self.assertNotIn(301, self.store._completed_ops)

    def test_timeout_returns_empty_rather_than_raising(self):
        """A late peer must degrade to a recompute, not kill the prefetch thread.

        _drain_until runs on HiCache's prefetch daemon; an exception there takes
        out storage prefetching for the whole engine.
        """
        result = self.store._drain_until(401, timeout_s=0.05)

        self.assertFalse(result)

    def test_a_completion_arriving_after_its_waiter_gave_up_is_dropped(self):
        """A timed-out op's late result must not sit in the stash forever.

        Timing out does not cancel anything -- ``kvcr.abort()`` is a no-op stub
        -- so the op stays in flight and may still report, but its waiter is
        gone and nothing will ever pop it again. Every entry stashed this way is
        permanent, and the store lives as long as the scheduler, so a server
        that times out under load leaks a dict entry (and every MemDescriptor it
        references) per timeout.

        Timeouts are the expected mode here, not an exotic one: the whole point
        of the remote path is fetching from a peer that may be slow or gone.
        """
        self.store._drain_until(501, timeout_s=0.05)
        self.core.finish(501, ["seg-a"])

        self.store._poll_once(self.core)

        self.assertNotIn(501, self.store._completed_ops)
        self.assertEqual(self.store.stats()["late_completions_dropped"], 1)

    def test_an_op_that_never_reports_leaves_nothing_behind(self):
        """The timeout path must not assume the op eventually reports.

        Measured against the real core, in this backend's exact call shape
        (empty-key ``submit_hint`` then ``deliver`` with engine host pages): a
        remote pull whose source has gone silent parks in KVCR's
        WAITING_TERMINAL state, which is left only on a ``write_done``
        notification that a dead peer never sends. 6/6 such delivers never
        reported their op handle at all -- not late, never.

        So any bookkeeping keyed on the *abandoned* handle and pruned when its
        late result arrives is pruned by an event that does not happen: one
        permanent entry per timed-out request, for the life of the scheduler.
        Keying on live waiters instead makes a never-reporting op cost nothing.

        What turns this red: reintroducing an abandoned-handle tombstone, or any
        other per-op record that outlives ``_drain_until`` on the timeout path.
        """
        for handle in range(601, 606):
            self.store._drain_until(handle, timeout_s=0.01)

        # Nothing is ever finish()ed: this source is gone for good.
        self.store._poll_once(self.core)

        for handle in range(601, 606):
            self.assertEqual(_per_op_residue(self.store, handle), [])

    def test_a_pump_cannot_poll_between_a_submit_and_its_registration(self):
        """Registration must precede the submit's completion, not follow it.

        ``_poll_once`` drops completions nobody is waiting on, which makes the
        window between "core hands back a handle" and "caller registers as its
        waiter" load-bearing. A local-tier hit retires inside that window (the
        tier moves bytes by a NIXL transfer to our own agent -- microseconds
        against the pump's 5 ms interval), so a pump polling there would see a
        completion with no waiter, drop it, and the caller would then sit out
        the full ``get_timeout_s`` before reporting a miss on an op that
        succeeded -- turning the *fastest* path into the slowest one.

        ``_submit_and_wait`` closes the window by holding ``_poll_lock`` across
        the submit and the registration, which is what the first assertion
        checks: a pump that tries to poll mid-submit makes no progress. The
        interleaving is forced rather than raced -- the completion is queued
        before the submit returns and a real pump poll is launched into the
        window -- so registering after the submit instead fails every run.
        """
        pumped = threading.Event()

        def submit_with_a_pump_racing_it() -> int:
            handle = self.core.deliver_and_finish(["seg-a"])
            pump = threading.Thread(
                target=lambda: (self.store._poll_once(self.core), pumped.set())
            )
            pump.start()
            self.addCleanup(pump.join)
            self.assertFalse(
                pumped.wait(timeout=0.5),
                "a pump polled while the submit was still in flight",
            )
            return handle

        handle, result = self.store._submit_and_wait(submit_with_a_pump_racing_it)

        self.assertEqual(result, {"seg-a": True})
        self.assertEqual(_per_op_residue(self.store, handle), [])


class CloseTest(unittest.TestCase):
    """Shutdown ordering between our pump thread and the KVCR core."""

    def test_the_core_is_not_closed_under_a_pump_stuck_in_a_poll(self):
        """A pump inside poll_completed() is walking state close() frees.

        ``kvcr.close()`` tears down the progress thread and the local tier, so a
        pump that is mid-``poll_completed()`` when that happens is reading freed
        KVCR state -- a use-after-free reachable through NIXL, not a benign late
        tick. The pump only ever misses its stop flag while it is inside a poll,
        so a join that times out is precisely the dangerous case, and the old
        code's fire-and-forget ``join(timeout=...)`` closed anyway.

        Leaking the core on a process that is already shutting down is the
        strictly safer branch, so close() declines rather than proceeds.
        """
        store = _store(0, 1)
        core = SimpleNamespace(closed=False)
        core.close = lambda: setattr(core, "closed", True)
        store._kvcr = core
        wedged = threading.Event()
        store._pump_thread = threading.Thread(target=wedged.wait, daemon=True)
        store._pump_thread.start()
        self.addCleanup(wedged.set)

        with mock.patch.object(kvcr_store, "_PUMP_JOIN_TIMEOUT_S", 0.05):
            store.close()

        self.assertFalse(core.closed)
        # And the store still holds the core, so nothing later mistakes it for
        # an already-released handle.
        self.assertIs(store._kvcr, core)

    def test_a_pump_that_stops_lets_the_core_close(self):
        """The refusal must be conditional, or close() never closes anything.

        Guards the branch above degrading into an unconditional leak.
        """
        store = _store(0, 1)
        core = SimpleNamespace(closed=False)
        core.close = lambda: setattr(core, "closed", True)
        store._kvcr = core
        store._pump_thread = threading.Thread(
            target=store._pump_stop.wait, daemon=True
        )
        store._pump_thread.start()

        store.close()

        self.assertTrue(core.closed)
        self.assertIsNone(store._kvcr)


class HintRequestIdTest(unittest.TestCase):
    """Ids scoping the core's per-request hint table."""

    def test_two_gets_sharing_a_prefix_get_different_ids(self):
        """Same-prefix concurrency is the normal case, and it used to collide.

        ``batch_get_v2`` registers the hint under this id and unregisters it in
        a ``finally``. When the id was derived from the hint's content -- source
        endpoint plus block hashes -- two concurrent requests for the *same*
        prefix produced the same id, so whichever finished first called
        ``discard_hint`` on the id the other was still fetching against. The
        loser's remaining segments then find no hint, report MISS, and get
        recomputed: a silent halving of the cache hit rate under exactly the
        load the feature exists for (many requests sharing one hot prefix).

        Nothing distinguishes these two calls but the call itself, so the id
        cannot be a function of the hint.
        """
        store = _store(0, 1)
        extra_info = _hint_extra_info("tcp://10.0.0.7:25000")
        store._kvcr = SimpleNamespace(submit_hint=lambda *a, **k: None)

        first = store._register_hint(extra_info)
        second = store._register_hint(extra_info)

        self.assertIsNotNone(first)
        self.assertNotEqual(first, second)


class RemoteFailureTest(unittest.TestCase):
    """A remote source that is slow, gone, or lying must degrade to recompute.

    Every one of these is a normal event for a P2P cache, not an exotic one: the
    source is another server that can be restarted, drained, or simply evict the
    prefix between the router's index and our fetch. The requirement is uniform
    and narrow -- report the pages as not-loaded and return, so HiCache
    recomputes them. Raising out of ``batch_get_v2`` would kill the prefetch
    thread that HiCache never restarts, taking down *local* L3 caching for the
    life of the process; hanging would stall it just as permanently.
    """

    def setUp(self) -> None:
        self.store = _store(0, 1)
        self.store._segments_per_page = 1
        self.store._slot_size = 16

    def _transfer(self, keys: List[str]) -> SimpleNamespace:
        """A PoolTransfer-alike whose host descriptors always resolve."""
        return SimpleNamespace(name="t0", keys=keys)

    def test_a_source_that_never_answers_reports_a_miss_rather_than_hanging(self):
        """The deliver completes for nobody, and the deadline must end it.

        This is the dead-source case: the router named a peer that has since
        been killed, so no completion ever arrives. ``kvcr.abort()`` is a stub
        that cancels nothing, so the deadline in ``_drain_until`` is the only
        thing standing between a dead peer and a permanently wedged prefetch
        thread.
        """
        self.store._kvcr = FakeKVCR()  # finish() is never called
        self.store._host_descriptors = lambda transfer: {"seg-a": object()}

        started = time.monotonic()
        results = self.store._deliver_transfer(
            self._transfer(["page-a"]), request_id="req-1"
        )
        elapsed = time.monotonic() - started

        self.assertEqual(results, [False])
        # Bounded by the configured get timeout, not by the caller giving up.
        self.assertLess(elapsed, self.store._config.get_timeout_s + 5.0)

    def test_a_partially_delivered_page_counts_as_not_loaded(self):
        """A page is all-or-nothing: its segments are not independently usable.

        One host page fans out into ``segments_per_page`` KVCR block keys, and
        attention reads the whole page. Reporting True when only some segments
        arrived would tell HiCache the page is cached, and the model would then
        attend over whatever was in the un-filled half of that page -- silent
        wrong output rather than a slow correct one.
        """
        self.store._segments_per_page = 2
        core = FakeKVCR()
        self.store._kvcr = core
        self.store._host_descriptors = lambda transfer: {"s0": object(), "s1": object()}
        self.store._page_segment_keys = lambda key: ["s0", "s1"]

        def deliver_then_half_fail(destinations, request_id=None):
            handle = 101
            core._pending.append((handle, {"s0": FakeEntry(True)}))
            return handle

        core.deliver = deliver_then_half_fail
        results = self.store._deliver_transfer(
            self._transfer(["page-a"]), request_id="req-1"
        )

        self.assertEqual(results, [False])

    def test_an_unparseable_hint_is_ignored_rather_than_raising(self):
        """A malformed hint must not take the prefetch thread down with it.

        The hint crosses a process and a repository boundary -- it is minted by
        the dynamo router, carried through the scheduler, and decoded here -- so
        a schema skew between the two sides is a deployment reality, not a
        programming error. The safe reading of a hint we cannot decode is that
        there is no hint: fetch locally and recompute the rest.
        """
        extra_info = HiCacheStorageExtraInfo(
            extra_info={ROUTER_HINT_KEY: {"source_control_endpoint": 17}}
        )

        self.assertIsNone(self.store._parse_hint(extra_info))

    def test_a_hint_covering_nothing_leaves_the_prefix_at_zero(self):
        """A stale hint must not promise a prefix the get cannot deliver.

        ``batch_exists_v2`` is the gate: the controller only issues gets for the
        prefix reported here, and it allocates host memory for it. Counting a
        page whose hash the hint does not cover would reserve memory for pages
        that then miss -- the memory is released again only after a full
        deliver round trip against a source that never had them.
        """
        self.store._kvcr = FakeKVCR()
        self.store._locally_resident = lambda segment_keys: False
        extra_info = _hint_extra_info("tcp://10.0.0.7:25000")

        result = self.store.batch_exists_v2(
            ["hash-the-hint-does-not-cover"], extra_info=extra_info
        )

        self.assertEqual(result.kv_hit_pages, 0)
        self.assertEqual(self.store.stats()["exists_hint_covered_nothing"], 1)


class RaisingCoreTest(unittest.TestCase):
    """A core that raises must degrade to a miss, never out of the store.

    ``RemoteFailureTest`` covers a *peer* that fails, which the core reports
    through its normal result path. This is the other source: the core itself
    raising -- a version skew that moved a method, a NIXL agent in a bad state,
    an internal assertion. It arrives as an exception rather than a result, so
    none of the miss-reporting logic runs.

    Nothing above this backend catches it.
    ``test_hicache_storage_thread_survival.py`` drives HiCache's three storage
    loops to show what happens then: each one ends on the first exception, they
    are unsupervised daemons so nothing restarts them, and they take the only
    ``append_host_mem_release`` and ``ack_backup_queue`` producers with them --
    so a single transient fault silently disables L2 and L3 *and* leaks host
    pages for the life of the process.

    What turns these red: removing a ``@_fail_closed`` decorator from any
    HiCacheStorage entry point.
    """

    def setUp(self) -> None:
        self.store = _store(0, 1)
        self.store._segments_per_page = 1
        self.store._slot_size = 16
        self.store._kvcr = _ExplodingKVCR()

    def _transfer(self, keys: List[str]) -> SimpleNamespace:
        return SimpleNamespace(name="t0", keys=keys)

    def test_a_raising_deposit_reports_every_page_unstored(self):
        self.store._host_descriptors = lambda transfer: {"seg-a": object()}

        results = self.store.batch_set_v2([self._transfer(["page-a", "page-b"])])

        self.assertEqual(results, {"t0": [False, False]})
        self.assertEqual(self.store.stats()["faults_batch_set_v2"], 1)

    def test_a_raising_deliver_reports_every_page_unloaded(self):
        self.store._host_descriptors = lambda transfer: {"seg-a": object()}

        results = self.store.batch_get_v2([self._transfer(["page-a"])])

        self.assertEqual(results, {"t0": [False]})
        self.assertEqual(self.store.stats()["faults_batch_get_v2"], 1)

    def test_a_raising_query_reports_no_available_prefix(self):
        """``batch_exists_v2`` gates the whole path, so its miss must be zero.

        Any other answer is worse than a miss: the controller allocates host
        memory for the prefix reported here and only releases it after a full
        deliver round trip, so a non-zero prefix from a store that cannot serve
        it reserves memory to fetch nothing.
        """
        result = self.store.batch_exists_v2(["page-a", "page-b"])

        self.assertEqual(result.kv_hit_pages, 0)
        self.assertEqual(self.store.stats()["faults_batch_exists_v2"], 1)

    def test_the_v1_shapes_survive_it_too(self):
        """``batch_*_v1`` is what HiRadixCache calls, and it has its own shape.

        v1 returns a flat list and ``batch_exists`` an int, so a guard that only
        covered v2 would return v2's dict here and the caller would read it as
        an unexpected type rather than as a miss.
        """
        self.store._host_descriptors = lambda transfer: {"seg-a": object()}

        self.assertEqual(
            self.store.batch_set_v1(["a", "b"], torch.arange(2)), [False, False]
        )
        self.assertEqual(
            self.store.batch_get_v1(["a", "b"], torch.arange(2)), [False, False]
        )
        self.assertEqual(self.store.batch_exists(["a", "b"]), 0)
        self.assertFalse(self.store.exists("a"))

    def test_a_repeating_fault_is_counted_every_time_but_logged_once(self):
        """The faults this guards for repeat once per prefetch.

        A traceback per occurrence would be the loudest thing in the log and say
        nothing new after the first, but dropping the count would hide how bad
        it is -- so the count is exact and the log is throttled.
        """
        self.store._host_descriptors = lambda transfer: {"seg-a": object()}

        with self.assertLogs(kvcr_store.__name__, "WARNING") as logs:
            for _ in range(5):
                self.store.batch_get_v2([self._transfer(["page-a"])])

        self.assertEqual(self.store.stats()["faults_batch_get_v2"], 5)
        self.assertEqual(len(logs.records), 1)

    def test_a_raising_discard_hint_does_not_lose_a_completed_fetch(self):
        """The pages are already in host memory when ``discard_hint`` runs.

        It sits in a ``finally`` after every transfer has completed, so letting
        it raise would convert a fetch that succeeded into a recompute -- and on
        the exception path it would replace the original fault with a less
        informative one.
        """
        core = FakeKVCR()
        core.discard_hint = _raise_fault
        core.submit_hint = lambda *a, **k: None
        self.store._kvcr = core
        self.store._host_descriptors = lambda transfer: {"seg-a": object()}
        self.store._page_segment_keys = lambda key: ["seg-a"]
        original_deliver = core.deliver

        def deliver_and_succeed(destinations, request_id=None):
            handle = original_deliver(destinations, request_id)
            core.finish(handle, ["seg-a"])
            return handle

        core.deliver = deliver_and_succeed

        results = self.store.batch_get_v2(
            [self._transfer(["page-a"])],
            _hint_extra_info("tcp://10.0.0.7:25000"),
        )

        self.assertEqual(results, {"t0": [True]})
        self.assertEqual(self.store.stats()["faults_discard_hint"], 1)


class StatsTest(unittest.TestCase):
    """The counters that make a silent remote path diagnosable."""

    def test_a_hinted_get_and_an_unhinted_one_are_counted_apart(self):
        """These two look identical downstream and have opposite causes.

        A request that arrives without a hint means the router did not name a
        source -- its index has no better peer, which is an upstream fact. A
        request that arrives *with* a hint and still loads nothing means the
        fetch failed, which is ours. Both end as "cache miss" in every metric
        sglang exports, so without this split the first symptom of a broken
        remote path is an unattributable throughput regression.
        """
        store = _store(0, 1)
        store._kvcr = SimpleNamespace(submit_hint=lambda *a, **k: None)

        store._register_hint(_hint_extra_info("tcp://10.0.0.7:25000"))
        store._register_hint(None)

        stats = store.stats()
        self.assertEqual(stats["get_with_hint"], 1)
        self.assertEqual(stats["get_without_hint"], 1)

    def test_a_policy_drop_and_a_failure_are_counted_apart(self):
        """DROPPED and FAILED are both falsy, and mean opposite things.

        ``OpEntryResult.success`` is ``status is SUCCESS``, so a block the
        policy declined because the tier is full reads identically to a block
        whose transfer broke -- everywhere downstream, including the fault
        counters the injection run reads as evidence. The first is the tier
        working as configured; the second is a defect. Without this split,
        tuning a policy until it drops more looks exactly like introducing a
        fault.
        """
        store = _store(0, 1)

        store._note_entry_statuses(
            {
                "ok": FakeEntry(True, status=OpEntryStatus.SUCCESS),
                "full": FakeEntry(False, status=OpEntryStatus.DROPPED),
                "broken": FakeEntry(False, status=OpEntryStatus.FAILED),
            }
        )

        stats = store.stats()
        self.assertEqual(stats["entries_dropped_by_policy"], 1)
        self.assertEqual(stats["entries_failed"], 1)

    def test_an_entry_without_a_status_counts_as_a_failure(self):
        """The classification must not depend on a field the core may not have.

        ``OpEntryStatus.DROPPED`` arrived in kvcr e3a816e; the store still has
        to work against a core that reports only ``success``. Treating a
        status-less falsy entry as DROPPED would be the unsafe direction --
        real faults would silently land in the "policy did this on purpose"
        bucket -- so it counts as a failure.
        """
        store = _store(0, 1)

        store._note_entry_statuses({"legacy": FakeEntry(False)})

        stats = store.stats()
        self.assertEqual(stats["entries_failed"], 1)
        self.assertNotIn("entries_dropped_by_policy", stats)


class PolicySelectionTest(unittest.TestCase):
    """The local-tier policy is named by config, never left to the core.

    The core's default is not a stable interface: it was FIFO through kvcr
    ``abb13bf`` and LRU as of ``e3a816e``. An unchanged config therefore
    re-tunes eviction on a core bump, which silently invalidates any throughput
    number taken before it -- the failure mode is a benchmark that no longer
    describes the system, with nothing in the config or the log having changed.
    """

    def test_default_is_explicit_not_inherited(self):
        store = _store(0, 1)
        self.assertEqual(store._config.policy, "lru")

    def test_builtin_names_match_the_core_classes(self):
        """These strings are also vLLM's ``_BUILTIN_POLICIES`` keys.

        Both engines' configs are read by the same people comparing the same
        arms, so a name has to mean one thing across them; and each maps to a
        class this repo does not own.
        """
        self.assertIs(kvcr_store._resolve_policy("fifo").__class__, FIFOPolicy)
        self.assertIs(kvcr_store._resolve_policy("lru").__class__, LRUPolicy)

    def test_an_unknown_bare_name_is_rejected_with_the_valid_ones(self):
        """A typo must not fall through to the core's default.

        Silently ignoring it is the whole failure this config exists to
        prevent, and the misconfiguration is invisible afterwards.
        """
        with self.assertRaises(ValueError) as caught:
            kvcr_store._resolve_policy("lfu")
        self.assertIn("lru", str(caught.exception))

    def test_a_qualified_path_to_a_non_policy_is_rejected(self):
        """An external policy is imported by path, so the type is unchecked.

        Handing the core a non-policy object fails later, from inside a
        placement decision on a prefetch, where it reads as a core bug.
        """
        with self.assertRaises(TypeError):
            kvcr_store._resolve_policy("collections.OrderedDict")


if __name__ == "__main__":
    unittest.main()
