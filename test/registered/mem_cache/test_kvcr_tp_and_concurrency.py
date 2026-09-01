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

import importlib.util
import threading
import time
import unittest
from types import SimpleNamespace
from typing import Dict, List, Tuple
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

# ``kvcr`` absent is a legitimate tier configuration: the CPU suite runs without
# the wheel. ``kvcr`` present but this adapter failing to import is not -- it
# means the public API moved out from under us, and one guard covering both
# reports every case below as skipped-green while running none of them. Split
# the two conditions so only the first one skips.
_KVCR_INSTALLED = importlib.util.find_spec("kvcr") is not None
_KVCR_IMPORT_ERROR = None

if _KVCR_INSTALLED:
    try:
        from kvcr.types import OpEntryStatus, QueryStatus

        from sglang.srt.mem_cache.storage.kvcr import kvcr_store
        from sglang.srt.mem_cache.storage.kvcr.kvcr_store import KVCRStore

        _HAS_KVCR = True
    except ImportError as error:  # pragma: no cover - reported by the test below
        _HAS_KVCR = False
        _KVCR_IMPORT_ERROR = f"{type(error).__name__}: {error}"
else:  # pragma: no cover - wheel not installed on this tier
    _HAS_KVCR = False

# A module-level raise would be shorter, but SkipTest outside a test is an
# uncaught exception: the CI runner invokes this file as a subprocess and reads
# its exit code, so it would fail the whole CPU suite rather than skipping.
_needs_kvcr = unittest.skipUnless(_HAS_KVCR, "nvidia-kvcr wheel not installed")


class TestKVCRImports(unittest.TestCase):
    """Fail loudly when an installed kvcr no longer satisfies this adapter."""

    @unittest.skipUnless(_KVCR_INSTALLED, "nvidia-kvcr wheel not installed")
    def test_adapter_imports_against_the_installed_kvcr(self):
        self.assertIsNone(
            _KVCR_IMPORT_ERROR,
            "kvcr is installed but KVCRStore does not import against it, so "
            "every real-backend case in this file silently skipped: "
            f"{_KVCR_IMPORT_ERROR}",
        )


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
    over an indexer page holding KV bytes. Every entry point -- registration, get,
    set, exists -- must score the pool a miss on its own.
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

    def test_timeout_returns_empty_rather_than_raising(self):
        """A late peer must degrade to a recompute, not kill the prefetch thread.

        _drain_until runs on HiCache's prefetch daemon; an exception there takes
        out storage prefetching for the whole engine.
        """
        result = self.store._drain_until(401, timeout_s=0.05)

        self.assertFalse(result)


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


if __name__ == "__main__":
    unittest.main()
