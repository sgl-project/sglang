"""KVCRStore under TP/DP rank colocation and under concurrent operations.

Two assumptions the TP=1 single-request validation never exercised:

  1. **Rank colocation.** Every ``(dp, attn_cp, attn_tp)`` rank builds its own
     KVCRStore on the same host holding a different head slice. The port it
     *binds* must follow the full coordinate; the port it *dials* must follow
     only the within-DP part, since the router already picked the source DP
     rank. Both failures are silent: block keys are token hashes carrying no
     rank identity, so a wrong dial succeeds and returns another rank's shard.

  2. **Concurrent operations.** ``poll_completed()`` both drains a queue and
     advances state machines, so the source pump and ``_drain_until`` race for
     every completion; ``_completed_ops`` keeps the loser's result.

CPU-only: no mem_pool, so ``_build_kvcr`` never runs and the core is a fake.

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

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.kvcr.router_hint import (
    ROUTER_HINT_KEY,
    SOURCE_LOCATIONS_ACTION_TYPE,
    SOURCE_LOCATIONS_ACTION_VERSION,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

try:
    from kvcr.policy import FIFOPolicy, LRUPolicy
    from kvcr.types import OpEntryStatus, QueryStatus

    from sglang.srt.mem_cache.storage.kvcr import kvcr_store
    from sglang.srt.mem_cache.storage.kvcr.kvcr_store import KVCRStore

    _HAS_KVCR = True
except ImportError:  # pragma: no cover - wheel not installed on this tier
    _HAS_KVCR = False

# A module-level raise would be shorter, but SkipTest outside a test is an
# uncaught exception: the CI runner invokes this file as a subprocess and reads
# its exit code, so it would fail the whole CPU suite rather than skipping.
_needs_kvcr = unittest.skipUnless(_HAS_KVCR, "nvidia-kvcr wheel not installed")


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

    ``tp_rank``/``tp_size`` are attention-scoped whenever DP attention is on, so a
    DP=2/attn_tp=2 engine is four configs with ``tp_size=2``.
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
    """A KVCRStore with no mem_pool, so the core is never constructed."""
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
    """One router hint, shaped as the v0.1 envelope the router actually sends."""
    return HiCacheStorageExtraInfo(
        extra_info={
            ROUTER_HINT_KEY: {
                "protocol_version": "0.1",
                "message_id": "test-envelope",
                "actions": [
                    {
                        "action_id": "src-0",
                        "action_type": SOURCE_LOCATIONS_ACTION_TYPE,
                        "action_version": SOURCE_LOCATIONS_ACTION_VERSION,
                        "payload": {
                            "source_control_endpoint": endpoint,
                            "block_hashes": ["0123456789abcdef"],
                        },
                    }
                ],
            }
        }
    )


class FakeEntry:
    """One completed block key, shaped like the core's ``OpEntryResult``.

    ``success`` is derived from ``status`` as the core derives it, so DROPPED and
    FAILED are both falsy and only the status tells them apart.
    """

    def __init__(self, success: bool = True, status=None) -> None:
        if status is None:
            status = OpEntryStatus.SUCCESS if success else OpEntryStatus.FAILED
        self.status = status

    @property
    def success(self) -> bool:
        return self.status is OpEntryStatus.SUCCESS


def _per_op_residue(store: KVCRStore, op_handle: int) -> List[str]:
    """Names of every store container that still keys on ``op_handle``.

    A scan, not a check of one named field: no bookkeeping keyed by op handle may
    outlive the operation, however a rewrite reshapes the containers.
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

    ``FakeKVCR`` reports failure through the normal result path, which is exactly
    the path an exception skips.
    """

    deposit = _raise_fault
    deliver = _raise_fault
    discard_hint = _raise_fault
    poll_completed = _raise_fault
    query = _raise_fault
    submit_hint = _raise_fault


class FakeKVCR:
    """Records deliver() calls and reports completions only when told to.

    Nothing completes on its own, so the interleaving between waiters is chosen by
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

        The local tier moves bytes over a NIXL loopback, retired between the submit
        returning and the caller's first poll -- microseconds against a 5 ms pump.
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


@_needs_kvcr
class TPColocationTest(unittest.TestCase):
    """Two ranks of one engine, same host, same extra_config."""

    def test_the_agent_name_separates_every_rank_and_every_restart(self):
        """Peers key their remote-agent tables by name, so any collision makes
        one registration overwrite another's. The name must therefore carry the
        full rank coordinate, and a fresh id so a restarted rank does not reuse
        a name its peers still have cached.
        """
        tp0, tp1 = _store(0, 2), _store(1, 2)
        dp1 = _store(0, 2, dp_rank=1, dp_size=2)
        restarted = _store(0, 2)

        self.assertIn("tp0", tp0._agent_name)
        self.assertIn("tp1", tp1._agent_name)
        names = [
            tp0._agent_name,
            tp1._agent_name,
            dp1._agent_name,
            restarted._agent_name,
        ]
        self.assertEqual(len(set(names)), len(names))

    def test_control_port_is_offset_by_tp_rank(self):
        """One configured port must not mean one bound port.

        A collision fails in the background: the engine comes up and only peer fetches
        break. This is also the layout dynamo assumes when publishing per-rank
        endpoints.
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
        """Port 0 means "ask the OS", which already guarantees distinctness.

        Offsetting it would land on an arbitrary in-use port. Local-only, because
        ``KVCRBackendConfig`` refuses port 0 together with ``enable_remote_hint``.
        """
        rank0 = _store(0, 2, control_port=0, enable_remote_hint=False)
        rank1 = _store(1, 2, control_port=0, enable_remote_hint=False)

        self.assertNotEqual(rank0._control_port(), rank1._control_port())
        self.assertGreater(rank0._control_port(), 0)
        self.assertGreater(rank1._control_port(), 0)


@_needs_kvcr
class SourceEndpointRankTest(unittest.TestCase):
    """Which source port each rank dials for a hint-driven fetch.

    Regression for a wrong-shard bug at TP=2: both ranks dialed the single endpoint
    dynamo advertises, so rank 1 accepted rank 0's shard and reported a full hit.
    The only symptom was generated text differing from the same prefix computed
    locally. dynamo cannot fix it there -- it indexes by ``(worker_id, dp_rank)``
    and has no TP concept, so realigning is the consumer's job.
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

    def test_rank_alignment_survives_an_ipv6_source_host(self):
        """The offset applies to the port, not to whatever precedes it.

        A bracketed IPv6 literal contains colons of its own.
        """
        self.assertEqual(
            self._dialed(1, 2, "tcp://[fd00::1]:25000"), "tcp://[fd00::1]:25001"
        )

    def test_an_unparseable_endpoint_drops_the_hint(self):
        """A rank that cannot align must fetch nothing, not fetch wrongly.

        Returning the endpoint verbatim puts rank 1 back on rank 0's shard.
        """
        store = _store(1, 2)

        self.assertIsNone(store._parse_hint(_hint_extra_info("tcp://10.0.0.7")))


@_needs_kvcr
class DPColocationTest(unittest.TestCase):
    """Attention DP: several DP ranks of one engine on one host.

    One scheduler per ``(dp, attn_cp, attn_tp)`` rank, all sharing one
    ``extra_config``. Offsetting by ``tp_rank`` alone puts DP rank 1 on DP rank 0's
    ports -- the same silent collision reached through the other dimension.
    """

    def test_dp_ranks_occupy_consecutive_port_blocks(self):
        """DP rank r owns ``[base + r*attn_tp_size, +attn_tp_size)``.

        A contract with dynamo, which must stride by ``attn_tp_size``, not by 1.
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

        A formula that only multiplies ``dp_rank`` by ``tp_size`` gives every DP rank
        the same two ports here.
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

    def test_the_dialed_source_port_ignores_our_dp_rank(self):
        """The router already picked the source DP rank; we add only our own
        within-DP offset.

        Our DP rank says nothing about which of the source's DP ranks holds the prefix.
        Adding it walks into another DP group's port block, and that fetch succeeds.
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


@_needs_kvcr
class UnusableHostPoolTest(unittest.TestCase):
    """Host pool layouts the backend cannot run against must fail at startup."""

    def test_a_per_layer_pool_refuses_to_start_the_backend(self):
        """A pool with no single kv_buffer tensor cannot be NIXL-registered.

        ``framework_dram`` is the one registration both directions address -- even the
        local-tier copy is a NIXL loopback. Warning and starting anyway let a
        DeepSeek-V4-style per-layer pool name unregistered memory in every transfer.
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

        Unconfigured, ``deposit`` fails each entry individually, which reads downstream
        as a cold server while every peer serve silently has nothing to offer.
        """
        store = _store(0, 1)
        # A registrable kv_buffer, so this gets past the framework_dram gate and
        # fails on the probe alone.
        pool = SimpleNamespace(
            page_size=64,
            kv_buffer=torch.empty(64, dtype=torch.uint8),
            get_page_buffer_meta=lambda indices: ([], []),
        )

        with self.assertRaises(RuntimeError) as raised:
            store.register_mem_pool_host(pool)

        self.assertIn("local DRAM tier", str(raised.exception))


@_needs_kvcr
class SidecarPoolTest(unittest.TestCase):
    """A pool this backend cannot address must never be reported as served.

    A hybrid KV stack (DSA/MiniMax indexer, Mamba, SWA) sends one ``PoolTransfer``
    per pool, the sidecar one derived from the KV pool's indices and hashes. This
    backend resolves addresses through ``mem_pool_host``, which forwards to the
    anchor pool -- so a sidecar transfer moves KV bytes into KV pages, reports
    success, and leaves the sidecar untouched. ``True`` for a page never written
    makes ``_sync_and_clamp_prefetch_result`` skip the clamp and the model attends
    over an indexer page holding KV bytes. Every entry point must score the pool a
    miss on its own; guarding only some leaves a fail-open path.
    """

    def _store_with_core(self) -> KVCRStore:
        store = _store(0, 1)
        store._kvcr = FakeKVCR()
        store._segments_per_page = 1
        return store

    def _transfers(self):
        """One KV transfer and one sidecar transfer, as a hybrid stack sends."""
        return [
            PoolTransfer(name=PoolName.KV, keys=["p0", "p1"]),
            PoolTransfer(
                name=PoolName.INDEXER,
                keys=["p0", "p1"],
                indices_from_pool=PoolName.KV,
            ),
        ]

    def test_registering_a_sidecar_pool_refuses_to_start_the_backend(self):
        """Rejecting at startup is what turns wrong output into a failed launch."""
        store = _store(0, 1)

        with self.assertRaises(RuntimeError) as raised:
            store.register_mem_host_pool_v2(SimpleNamespace(), PoolName.INDEXER)

        self.assertIn("indexer", str(raised.exception))

    def test_registering_the_kv_pool_is_still_accepted(self):
        """The guard must reject by pool, not reject every v2 registration."""
        store = _store(0, 1)
        pool = SimpleNamespace()

        store.register_mem_host_pool_v2(pool, PoolName.KV)

        self.assertIs(store.registered_pools[PoolName.KV], pool)

    def test_a_sidecar_get_is_a_miss_and_never_reaches_the_core(self):
        """Asserting the deliver never ran separates this guard from
        ``_fail_closed``, which would also produce all-False and is not the same fix.
        """
        store = self._store_with_core()
        delivered = []

        def record_and_succeed(transfer, request_id):
            delivered.append(transfer.name)
            return [True] * len(transfer.keys)

        store._deliver_transfer = record_and_succeed

        results = store.batch_get_v2(self._transfers())

        self.assertEqual(results[str(PoolName.INDEXER)], [False, False])
        self.assertEqual(delivered, [PoolName.KV])

    def test_a_sidecar_set_is_a_miss_and_never_reaches_the_core(self):
        """The write side fails open one step later: the sidecar's host page is
        marked offloaded, HiCache evicts it, and the only copy is gone.
        """
        store = self._store_with_core()
        deposited = []

        def record_and_succeed(transfer):
            deposited.append(transfer.name)
            return [True] * len(transfer.keys)

        store._deposit_transfer = record_and_succeed

        results = store.batch_set_v2(self._transfers())

        self.assertEqual(results[str(PoolName.INDEXER)], [False, False])
        self.assertEqual(deposited, [PoolName.KV])

    def test_a_sidecar_pool_makes_the_whole_prefix_unavailable(self):
        """``batch_exists_v2`` reports one ``kv_hit_pages`` for the request and the
        controller issues gets for that prefix across all pools, so reporting the KV
        pages held would promise sidecar pages this backend cannot serve.
        """
        store = self._store_with_core()
        store._kvcr = SimpleNamespace(
            query=lambda keys: [(QueryStatus.HIT, None)] * len(keys)
        )

        result = store.batch_exists_v2(["p0", "p1"], self._transfers())

        self.assertEqual(result.kv_hit_pages, 0)
        self.assertEqual(result.extra_pool_hit_pages, {})

    def test_a_kv_only_prefix_is_still_reported(self):
        """The prefix must collapse on a sidecar pool, not on every request."""
        store = self._store_with_core()
        store._kvcr = SimpleNamespace(
            query=lambda keys: [(QueryStatus.HIT, None)] * len(keys)
        )

        result = store.batch_exists_v2(
            ["p0", "p1"], [PoolTransfer(name=PoolName.KV, keys=["p0", "p1"])]
        )

        self.assertEqual(result.kv_hit_pages, 2)


@_needs_kvcr
class UnaddressableParallelismTest(unittest.TestCase):
    """Rank coordinates a KVCR block key cannot encode must fail at startup.

    A block key is ``sha256(token ids)#<segment>`` and a hint carries an endpoint
    plus page hashes, so nothing on the wire says which model slice produced the
    bytes; ``_rank_port_offset`` separates ``(dp, attn_cp, attn_tp)`` by port
    instead. Pipeline rank and head splitting have no such separation -- same port
    *and* same key, so the fetch lands another rank's bytes in pages the model
    attends over.
    """

    def _config(self, **overrides) -> HiCacheStorageConfig:
        config = _storage_config(0, 1)
        for field, value in overrides.items():
            setattr(config, field, value)
        return config

    def test_pipeline_parallelism_refuses_to_start_the_backend(self):
        with self.assertRaises(RuntimeError) as raised:
            KVCRStore(self._config(pp_size=2), mem_pool=None)

        self.assertIn("pipeline parallelism", str(raised.exception))

    def test_split_heads_refuses_to_start_the_backend(self):
        with self.assertRaises(RuntimeError) as raised:
            KVCRStore(
                self._config(should_split_heads=True, tp_lcm_size=8), mem_pool=None
            )

        self.assertIn("heterogeneous TP", str(raised.exception))

    def test_tp_lcm_size_alone_is_not_what_gets_rejected(self):
        """Only ``should_split_heads`` means the keys actually differ per rank.

        ``cache_controller`` accepts ``tp_lcm_size`` everywhere but resolves it to
        False for a rank-replicated model (MLA) or a non-``page_head`` layout, where
        the pooled key is correct. Keying the guard on it would refuse those.
        """
        store = KVCRStore(
            self._config(tp_lcm_size=8, should_split_heads=False), mem_pool=None
        )
        self.assertIsNotNone(store)


@_needs_kvcr
class ConcurrentDrainTest(unittest.TestCase):
    """_drain_until and the source pump racing for the same completion queue."""

    def setUp(self) -> None:
        self.store = _store(0, 1)
        self.core = FakeKVCR()
        self.store._kvcr = self.core

    def test_completion_drained_by_the_pump_still_reaches_its_waiter(self):
        """The case ``_completed_ops`` exists for: the pump routinely observes a
        get's completion, and dropping it makes the waiter spin to its deadline and
        report a miss.
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
            threading.Thread(target=wait_for, args=(handle,)) for handle in (201, 202)
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

    def test_timeout_returns_empty_rather_than_raising(self):
        """A late peer must degrade to a recompute, not kill the prefetch thread.

        _drain_until runs on HiCache's prefetch daemon; an exception there takes
        out storage prefetching for the whole engine.
        """
        result = self.store._drain_until(401, timeout_s=0.05)

        self.assertFalse(result)

    def test_a_completion_arriving_after_its_waiter_gave_up_is_dropped(self):
        """A timed-out op's late result must not sit in the stash forever.

        Timing out cancels nothing -- ``kvcr.abort()`` is a no-op stub -- so the op may
        still report with no waiter left to pop it, and the store lives as long as the
        scheduler.
        """
        self.store._drain_until(501, timeout_s=0.05)
        self.core.finish(501, ["seg-a"])

        self.store._poll_once(self.core)

        self.assertNotIn(501, self.store._completed_ops)
        self.assertEqual(self.store.stats()["late_completions_dropped"], 1)

    def test_an_op_that_never_reports_leaves_nothing_behind(self):
        """The timeout path must not assume the op eventually reports.

        Measured against the real core: a remote pull whose source went silent parks in
        WAITING_TERMINAL awaiting a ``write_done`` a dead peer never sends, and 6/6
        never reported their handle at all. So bookkeeping keyed on the abandoned
        handle and pruned on its late result is pruned by an event that never happens.
        """
        for handle in range(601, 606):
            self.store._drain_until(handle, timeout_s=0.01)

        # Nothing is ever finish()ed: this source is gone for good.
        self.store._poll_once(self.core)

        for handle in range(601, 606):
            self.assertEqual(_per_op_residue(self.store, handle), [])

    def test_a_pump_cannot_poll_between_a_submit_and_its_registration(self):
        """Registration must precede the submit's completion, not follow it.

        ``_poll_once`` drops completions nobody waits on, and a local-tier hit retires
        inside that window, so a pump polling there costs the caller a full
        ``get_timeout_s`` on an op that succeeded. ``_submit_and_wait`` holds
        ``_poll_lock`` across both; the interleaving is forced, not raced.
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
        # Also the delivered-result case of the no-residue rule above: a stash
        # that only grows is an unbounded leak in a scheduler-lifetime process.
        self.assertEqual(_per_op_residue(self.store, handle), [])


@_needs_kvcr
class CloseTest(unittest.TestCase):
    """Shutdown ordering between our pump thread and the KVCR core."""

    def _store_with_pump(self, pump_target):
        store = _store(0, 1)
        core = SimpleNamespace(closed=False)
        core.close = lambda: setattr(core, "closed", True)
        store._kvcr = core
        store._pump_thread = threading.Thread(target=pump_target, daemon=True)
        store._pump_thread.start()
        return store, core

    def test_the_core_is_not_closed_under_a_pump_stuck_in_a_poll(self):
        """A pump inside poll_completed() is walking state close() frees.

        ``kvcr.close()`` tears down the progress thread and the local tier, so this is
        a use-after-free reachable through NIXL. The pump only misses its stop flag
        while inside a poll, so a join that times out is precisely the dangerous case.
        """
        wedged = threading.Event()
        store, core = self._store_with_pump(wedged.wait)
        self.addCleanup(wedged.set)

        with mock.patch.object(kvcr_store, "_PUMP_JOIN_TIMEOUT_S", 0.05):
            store.close()

        self.assertFalse(core.closed)
        # And the store still holds the core, so nothing later mistakes it for
        # an already-released handle.
        self.assertIs(store._kvcr, core)

    def test_a_pump_that_stops_lets_the_core_close(self):
        """The refusal must be conditional, or close() never closes anything."""
        store = _store(0, 1)
        core = SimpleNamespace(closed=False)
        core.close = lambda: setattr(core, "closed", True)
        store._kvcr = core
        store._pump_thread = threading.Thread(target=store._pump_stop.wait, daemon=True)
        store._pump_thread.start()

        store.close()

        self.assertTrue(core.closed)
        self.assertIsNone(store._kvcr)


@_needs_kvcr
class HintRequestIdTest(unittest.TestCase):
    """Ids scoping the core's per-request hint table."""

    def test_two_gets_sharing_a_prefix_get_different_ids(self):
        """Same-prefix concurrency is the normal case, and it used to collide.

        With the id derived from the hint's content, whichever request finished first
        called ``discard_hint`` on the id the other was still fetching against; the
        loser's remaining segments then miss and get recomputed.
        """
        store = _store(0, 1)
        extra_info = _hint_extra_info("tcp://10.0.0.7:25000")
        store._kvcr = SimpleNamespace(submit_hint=lambda *a, **k: None)

        first = store._register_hint(extra_info)
        second = store._register_hint(extra_info)

        self.assertIsNotNone(first)
        self.assertNotEqual(first, second)


@_needs_kvcr
class RemoteFailureTest(unittest.TestCase):
    """A remote source that is slow, gone, or lying must degrade to recompute.

    All three are normal for a P2P cache. Raising out of ``batch_get_v2`` kills the
    prefetch thread HiCache never restarts, taking down *local* L3 for the life of
    the process; hanging stalls it just as permanently.
    """

    def setUp(self) -> None:
        self.store = _store(0, 1)
        self.store._segments_per_page = 1
        self.store._slot_size = 16

    def _transfer(self, keys: List[str]) -> SimpleNamespace:
        """A PoolTransfer-alike whose host descriptors always resolve."""
        return SimpleNamespace(name=PoolName.KV, keys=keys)

    def test_a_source_that_never_answers_reports_a_miss_rather_than_hanging(self):
        """``kvcr.abort()`` cancels nothing, so ``_drain_until``'s deadline is all
        that stands between a dead peer and a permanently wedged prefetch thread.
        """
        self.store._kvcr = FakeKVCR()  # finish() is never called
        self.store._host_descriptors = lambda transfer: (
            {"seg-a": object()},
            [["seg-a"]],
        )

        started = time.monotonic()
        results = self.store._deliver_transfer(
            self._transfer(["page-a"]), request_id="req-1"
        )
        elapsed = time.monotonic() - started

        self.assertEqual(results, [False])
        # Bounded by the configured get timeout, not by the caller giving up.
        self.assertLess(elapsed, self.store._config.get_timeout_s + 5.0)

    def test_a_partially_delivered_page_counts_as_not_loaded(self):
        """A page is all-or-nothing: attention reads the whole page, so reporting
        True when only some of its segments arrived makes the model attend over
        whatever was in the un-filled half.
        """
        self.store._segments_per_page = 2
        core = FakeKVCR()
        self.store._kvcr = core
        self.store._host_descriptors = lambda transfer: (
            {"s0": object(), "s1": object()},
            [["s0", "s1"]],
        )

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
        """The hint crosses a process and a repository boundary, so schema skew is
        a deployment reality; the safe reading of an undecodable hint is no hint.
        """
        extra_info = HiCacheStorageExtraInfo(
            extra_info={ROUTER_HINT_KEY: {"source_control_endpoint": 17}}
        )

        self.assertIsNone(self.store._parse_hint(extra_info))

    def test_a_hint_covering_nothing_leaves_the_prefix_at_zero(self):
        """``batch_exists_v2`` is the gate: the controller allocates host memory
        for the prefix reported here, released only after a full deliver round trip.
        """
        self.store._kvcr = FakeKVCR()
        self.store._locally_resident = lambda segment_keys: False
        extra_info = _hint_extra_info("tcp://10.0.0.7:25000")

        result = self.store.batch_exists_v2(
            ["hash-the-hint-does-not-cover"], extra_info=extra_info
        )

        self.assertEqual(result.kv_hit_pages, 0)
        self.assertEqual(self.store.stats()["exists_hint_covered_nothing"], 1)


@_needs_kvcr
class RaisingCoreTest(unittest.TestCase):
    """A core that raises must degrade to a miss, never out of the store.

    ``RemoteFailureTest`` covers a *peer* failing, which the core reports through
    its normal result path. This is the core itself raising, which skips that path
    entirely, and nothing above this backend catches it: each HiCache storage loop
    catches only ``Empty``, so it ends on the first exception, unsupervised, and
    takes the only ``append_host_mem_release`` and ``ack_backup_queue`` producers
    with it.

    What turns these red: removing a ``@_fail_closed`` decorator.
    """

    def setUp(self) -> None:
        self.store = _store(0, 1)
        self.store._segments_per_page = 1
        self.store._slot_size = 16
        self.store._kvcr = _ExplodingKVCR()
        self.store._host_descriptors = lambda transfer: (
            {"seg-a": object()},
            [["seg-a"]],
        )

    def _transfer(self, keys: List[str]) -> SimpleNamespace:
        return SimpleNamespace(name=PoolName.KV, keys=keys)

    def test_the_transfer_paths_report_every_page_unserved(self):
        set_results = self.store.batch_set_v2([self._transfer(["page-a", "page-b"])])
        get_results = self.store.batch_get_v2([self._transfer(["page-a"])])

        self.assertEqual(set_results, {str(PoolName.KV): [False, False]})
        self.assertEqual(get_results, {str(PoolName.KV): [False]})
        self.assertEqual(self.store.stats()["faults_batch_set_v2"], 1)
        self.assertEqual(self.store.stats()["faults_batch_get_v2"], 1)

    def test_a_raising_query_reports_no_available_prefix(self):
        """``batch_exists_v2`` gates the whole path, and any answer but zero makes
        the controller allocate host memory it releases only after a deliver round trip.
        """
        result = self.store.batch_exists_v2(["page-a", "page-b"])

        self.assertEqual(result.kv_hit_pages, 0)
        self.assertEqual(self.store.stats()["faults_batch_exists_v2"], 1)

    def test_the_v1_shapes_survive_it_too(self):
        """``batch_*_v1`` is what HiRadixCache calls and has its own shape: a flat
        list, and an int from ``batch_exists``. A v2-only guard returns a dict the
        caller reads as an unexpected type rather than as a miss.
        """
        self.assertEqual(
            self.store.batch_set_v1(["a", "b"], torch.arange(2)), [False, False]
        )
        self.assertEqual(
            self.store.batch_get_v1(["a", "b"], torch.arange(2)), [False, False]
        )
        self.assertEqual(self.store.batch_exists(["a", "b"]), 0)
        self.assertFalse(self.store.exists("a"))

    def test_a_repeating_fault_is_counted_every_time_but_logged_once(self):
        """These faults repeat once per prefetch: a traceback each would say
        nothing new after the first, and dropping the count would hide how bad it is.
        """
        with self.assertLogs(kvcr_store.__name__, "WARNING") as logs:
            for _ in range(5):
                self.store.batch_get_v2([self._transfer(["page-a"])])

        self.assertEqual(self.store.stats()["faults_batch_get_v2"], 5)
        self.assertEqual(len(logs.records), 1)

    def test_a_raising_discard_hint_does_not_lose_a_completed_fetch(self):
        """The pages are already in host memory when ``discard_hint`` runs.

        It sits in a ``finally`` after every transfer completed, so letting it raise
        converts a successful fetch into a recompute.
        """
        core = FakeKVCR()
        core.discard_hint = _raise_fault
        core.submit_hint = lambda *a, **k: None
        self.store._kvcr = core
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

        self.assertEqual(results, {str(PoolName.KV): [True]})
        self.assertEqual(self.store.stats()["faults_discard_hint"], 1)


@_needs_kvcr
class StatsTest(unittest.TestCase):
    """The counters that make a silent remote path diagnosable."""

    def test_a_hinted_get_and_an_unhinted_one_are_counted_apart(self):
        """These look identical downstream and have opposite causes: no hint is an
        upstream fact, a hint that loads nothing is ours. Both end as "cache miss" in
        every metric sglang exports.
        """
        store = _store(0, 1)
        store._kvcr = SimpleNamespace(submit_hint=lambda *a, **k: None)

        store._register_hint(_hint_extra_info("tcp://10.0.0.7:25000"))
        store._register_hint(None)

        stats = store.stats()
        self.assertEqual(stats["get_with_hint"], 1)
        self.assertEqual(stats["get_without_hint"], 1)

    def test_a_policy_drop_and_a_failure_are_counted_apart(self):
        """DROPPED and FAILED are both falsy and mean opposite things: a tier
        working as configured versus a defect. Without the split, tuning a policy until
        it drops more looks exactly like introducing a fault.
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


@_needs_kvcr
class PolicySelectionTest(unittest.TestCase):
    """The local-tier policy is named by config, never left to the core.

    The core's default is not a stable interface -- FIFO through kvcr ``abb13bf``,
    LRU as of ``e3a816e`` -- so an unchanged config re-tunes eviction on a core
    bump and silently invalidates any throughput number taken before it.
    """

    def test_default_is_explicit_not_inherited(self):
        store = _store(0, 1)
        self.assertEqual(store._config.policy, "lru")

    def test_builtin_names_match_the_core_classes(self):
        """These strings are also vLLM's ``_BUILTIN_POLICIES`` keys.

        Both engines' configs are read by the same people comparing the same arms, and
        each maps to a class this repo does not own.
        """
        self.assertIs(kvcr_store._resolve_policy("fifo").__class__, FIFOPolicy)
        self.assertIs(kvcr_store._resolve_policy("lru").__class__, LRUPolicy)

    def test_an_unknown_bare_name_is_rejected_with_the_valid_ones(self):
        """A typo must not fall through to the core's default -- silently ignoring
        it is the whole failure this config exists to prevent.
        """
        with self.assertRaises(ValueError) as caught:
            kvcr_store._resolve_policy("lfu")
        self.assertIn("lru", str(caught.exception))

    def test_a_qualified_path_to_a_non_policy_is_rejected(self):
        """An external policy is imported by path, so the type is unchecked, and a
        non-policy object fails later from inside a placement decision.
        """
        with self.assertRaises(TypeError):
            kvcr_store._resolve_policy("collections.OrderedDict")


@_needs_kvcr
class SourcePumpFaultToleranceTest(unittest.TestCase):
    """The pump is what makes this worker usable as a P2P source.

    ``_poll_completed`` advances a peer's ``start_write`` through pin and transfer,
    and on an idle worker the pump is its only caller. A pump that exits on one
    transient NIXL or ZMQ error retires the worker as a source for the life of the
    process, while the engine keeps serving and peers see only a cold cache.
    """

    def _run_pump(self, store, poll, *, stop_after: float = 0.5) -> None:
        """Drive the real loop body until it returns or the guard time elapses."""
        store._poll_once = poll
        store._kvcr = SimpleNamespace()
        stopper = threading.Timer(stop_after, store._pump_stop.set)
        stopper.start()
        try:
            store._source_pump_func()
        finally:
            stopper.cancel()
            store._pump_stop.set()

    def test_a_transient_fault_does_not_end_the_pump(self):
        calls = []

        def poll(_kvcr):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("transient NIXL error")

        store = _store(0, 1)
        self._run_pump(store, poll)

        # Polling continued past the raise, which is the whole guard.
        self.assertGreater(len(calls), 1)
        self.assertEqual(store.stats().get("source_pump_dead", 0), 0)

    def test_a_persistent_fault_ends_the_pump_and_is_visible(self):
        """The negative branch: retrying forever on a dead core is not a fix.

        The counter is the only trace an operator gets -- past this point the worker is
        silently source-dead.
        """
        calls = []

        def poll(_kvcr):
            calls.append(1)
            raise RuntimeError("core is gone")

        store = _store(0, 1)
        with self.assertLogs(
            "sglang.srt.mem_cache.storage.kvcr.kvcr_store", "ERROR"
        ) as logs:
            self._run_pump(store, poll)

        self.assertEqual(len(calls), kvcr_store._PUMP_MAX_CONSECUTIVE_FAULTS)
        self.assertEqual(store.stats()["source_pump_dead"], 1)
        self.assertEqual(
            store.stats()["source_pump_faults"],
            kvcr_store._PUMP_MAX_CONSECUTIVE_FAULTS,
        )
        self.assertIn("P2P source", "\n".join(logs.output))

    def test_the_fault_streak_resets_on_a_successful_poll(self):
        """Counting total rather than *consecutive* faults would eventually kill
        the pump on a healthy worker that saw an occasional blip.
        """
        calls = []

        def poll(_kvcr):
            calls.append(1)
            # Fail every other poll: never two in a row, but many in total.
            if len(calls) % 2 == 1:
                raise RuntimeError("intermittent")

        store = _store(0, 1)
        self._run_pump(store, poll)

        self.assertGreater(
            store.stats()["source_pump_faults"],
            kvcr_store._PUMP_MAX_CONSECUTIVE_FAULTS,
        )
        self.assertEqual(store.stats().get("source_pump_dead", 0), 0)


if __name__ == "__main__":
    unittest.main()
