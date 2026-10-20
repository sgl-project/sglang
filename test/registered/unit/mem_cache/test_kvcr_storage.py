"""CPU contracts for indexed HiCache IO; native NIXL is a separate integration gate."""

import ctypes
import json
import mmap
import os
import threading
import types
import unittest
from contextlib import ExitStack
from enum import Enum
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch

from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    StorageOperation,
)
from sglang.srt.mem_cache.storage.kvcr.delivery import (
    Delivery,
    block_chunks,
    requested_results,
    run_window,
    validate_destination_spans,
)
from sglang.srt.mem_cache.storage.kvcr.diagnostics import DiagnosticStats
from sglang.srt.mem_cache.storage.kvcr.hints import peer_hint
from sglang.srt.mem_cache.storage.kvcr.host_layout import (
    block_key,
    describe_pool,
    page_slots,
    resume_boundaries,
    storage_namespace,
)
from sglang.srt.mem_cache.storage.kvcr.kvcr_store import KVCRStore, _allocate_dram_pool
from sglang.srt.mem_cache.storage.kvcr.numa_pool import (
    HOST_PROOF_ATTR,
    audit_registered_host,
    load_config,
    node_option,
    prove_range,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _HostPool:
    def __init__(self, sizes=(8, 3), page_size=2, pages=8):
        self.page_size, self.size = page_size, pages * page_size
        self.layout, self.dtype = "page_first", torch.uint8
        self.buffers = [torch.zeros((pages, size), dtype=self.dtype) for size in sizes]
        self.kv_buffer = self.buffers[0]

    def get_page_buffer_meta(self, indices):
        pointers, sizes = [], []
        for start in indices.tolist()[:: self.page_size]:
            for buffer in self.buffers:
                slot = start // self.page_size
                pointers.append(buffer[slot].data_ptr())
                sizes.append(buffer.shape[1])
        return pointers, sizes


class _MemoryRef(msgspec.Struct, frozen=True, kw_only=True):
    end_point_name: str
    label: str
    element_index: int


class _QueryStatus(Enum):
    HIT = "hit"
    FETCHABLE = "fetchable"
    MISS = "miss"


class _CopyTransport:
    """External API substitute that copies the addresses selected by real adapter logic."""

    def __init__(self, layouts):
        self.components = {
            part.name: part for p in layouts.values() for part in p.components
        }
        self.storage, self.completed, self.hints = {}, {}, {}
        self.next_handle = 0

    def deposit(self, blocks):
        for key, refs in blocks.items():
            self.storage[key] = {
                ref.label: ctypes.string_at(
                    self.components[ref.label].address
                    + ref.element_index * self.components[ref.label].stride,
                    self.components[ref.label].size,
                )
                for ref in refs
            }
        return self._complete({key: SimpleNamespace(success=True) for key in blocks})

    def deliver(self, blocks, request_id=None):
        results = {}
        for key, refs in blocks.items():
            data = self.storage.get(key)
            ok = data is not None and all(ref.label in data for ref in refs)
            if ok:
                for ref in refs:
                    part = self.components[ref.label]
                    ctypes.memmove(
                        part.address + ref.element_index * part.stride,
                        data[ref.label],
                        part.size,
                    )
            results[key] = SimpleNamespace(success=ok)
        return self._complete(results)

    def _complete(self, result):
        self.next_handle += 1
        self.completed[self.next_handle] = result
        return self.next_handle

    def poll_completed(self):
        result = list(self.completed.items())
        self.completed.clear()
        return result

    def query(self, keys, request_id=None):
        return [
            (_QueryStatus.HIT if k in self.storage else _QueryStatus.MISS, None)
            for k in keys
        ]

    def submit_hint(self, hint, request_id=None):
        self.hints[request_id] = hint

    def discard_hint(self, request_id):
        del self.hints[request_id]


class _DeferredCopyTransport(_CopyTransport):
    """Queue public operations; perform real byte writes only at completion poll."""

    def __init__(self, layouts, peers=None):
        super().__init__(layouts)
        self.pending, self.reads, self.writes, self.events = {}, [], [], []
        self.peers, self.max_pending = peers, 0
        self.result_hook = None
        self._copying = False

    def deliver(self, blocks, request_id=None):
        self.next_handle += 1
        handle = self.next_handle
        hint = self.hints.get(request_id)
        source = (
            hint["actions"][0]["payload"]["source_control_endpoint"] if hint else None
        )
        self.pending[handle] = (dict(blocks), request_id, source)
        self.reads.append((handle, dict(blocks), request_id, source))
        self.events.append(("submit", handle))
        self.max_pending = max(self.max_pending, len(self.pending))
        return handle

    def _complete(self, result):
        return result if self._copying else super()._complete(result)

    def poll_completed(self):
        completed = super().poll_completed()
        # Reverse order deliberately exercises the adapter's shared mailbox.
        for handle in reversed(list(self.pending)):
            blocks, request_id, source = self.pending[handle]
            if request_id is not None and request_id not in self.hints:
                raise AssertionError("Hint discarded before native writes completed")
            prior = self.storage
            try:
                if self.peers is not None:
                    self.storage = self.peers.get(source, {})
                self._copying = True
                result = super().deliver(blocks, request_id)
            finally:
                self._copying = False
                self.storage = prior
            for key, refs in blocks.items():
                if result[key].success:
                    self.writes.append((source, key, tuple(refs)))
            if self.result_hook is not None:
                result = self.result_hook(handle, blocks, result)
            completed.append((handle, result))
            del self.pending[handle]
            self.events.append(("complete", handle))
        return completed

    def discard_hint(self, request_id):
        if any(item[1] == request_id for item in self.pending.values()):
            raise AssertionError("Discarding a hint with pending writes")
        self.events.append(("discard", request_id))
        super().discard_hint(request_id)


class _BoundedCopyTransport(_CopyTransport):
    """External bounded LRU with the native API's tail-first alignment contract."""

    def __init__(self, layouts, capacity):
        super().__init__(layouts)
        self.capacity, self.recency, self.clock = capacity, {}, 0

    def deposit(self, blocks):
        self.clock += 1
        for key in blocks:
            if key in self.storage:
                continue
            pool = key.rsplit(b"#", 1)[1]
            occupants = [
                item for item in self.storage if item.rsplit(b"#", 1)[1] == pool
            ]
            if len(occupants) >= self.capacity:
                victim = min(occupants, key=self.recency.__getitem__)
                del self.storage[victim]
                del self.recency[victim]
            self.recency[key] = (self.clock, 0)
        return super().deposit(blocks)

    def align_sequence(self, ordered_keys, use_current_time=False):
        self.clock += 1
        for position, key in enumerate(dict.fromkeys(ordered_keys)):
            if key in self.storage:
                self.recency[key] = (self.clock, -position)


class _PeerCopyTransport(_CopyTransport):
    """Route actual CPU byte copies by the public request-scoped source hint."""

    def __init__(self, layouts, peers):
        super().__init__(layouts)
        self.peers, self.reads = peers, []

    def deliver(self, blocks, request_id=None):
        source = self.hints[request_id]["actions"][0]["payload"][
            "source_control_endpoint"
        ]
        self.reads.append((source, tuple(blocks)))
        prior = self.storage
        try:
            self.storage = self.peers.get(source, {})
            return super().deliver(blocks, request_id)
        finally:
            self.storage = prior


class TestKVCRStorage(CustomTestCase):
    def setUp(self):
        self.config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=2,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="glm-test",
        )
        self.pool = _HostPool()
        self.backend = KVCRStore(self.config, self.pool)
        self.backend._namespace = storage_namespace(self.config, self.backend._layouts)
        self.transport = self.backend._runner = _CopyTransport(self.backend._layouts)
        self.keys = [f"{i + 1:064x}" for i in range(4)]
        types_module = types.ModuleType("kvcr.types")
        types_module.MemoryRef, types_module.QueryStatus = _MemoryRef, _QueryStatus
        self.stub = patch.dict(
            "sys.modules",
            {"kvcr": types.ModuleType("kvcr"), "kvcr.types": types_module},
        )
        self.stub.start()
        self.addCleanup(self.stub.stop)

    @unittest.skipUnless(hasattr(os, "fork"), "Private-map isolation requires fork")
    def test_dram_pool_is_private_and_owns_writable_bytes(self):
        """Reverting to mmap's shared default leaks a child's writes to the owner."""
        with _allocate_dram_pool(8192) as allocation:
            allocation[0] = 17
            child = os.fork()
            if child == 0:
                allocation[0] = 29
                os._exit(0)
            _, status = os.waitpid(child, 0)
            self.assertEqual(status, 0)
            self.assertEqual(allocation[0], 17)
            address = ctypes.addressof(ctypes.c_char.from_buffer(allocation))
            ctypes.memset(address + 4096, 43, 4096)
            self.assertEqual(allocation[4096:], bytes([43]) * 4096)
        self.assertTrue(allocation.closed)

    def test_dram_pool_remains_usable_when_hugepage_advice_is_rejected(self):
        """THP is an allocation hint, never a new storage-readiness requirement."""

        class AdviceRejected(mmap.mmap):
            def madvise(self, *_args):
                raise OSError("THP is not supported")

        with (
            patch(
                "sglang.srt.mem_cache.storage.kvcr.kvcr_store.mmap.mmap",
                AdviceRejected,
            ),
            _allocate_dram_pool(8192) as allocation,
        ):
            allocation[:4] = b"kvcr"
            address = ctypes.addressof(ctypes.c_char.from_buffer(allocation))
            self.assertEqual(ctypes.string_at(address, 4), b"kvcr")
        self.assertTrue(allocation.closed)

    def test_whole_page_components_round_trip_to_different_slots(self):
        """A restore must move every unequal-sized component, not only primary KV."""
        for part, buffer in enumerate(self.pool.buffers):
            buffer[0].fill_(17 + part)
            buffer[1].fill_(29 + part)
        self.assertEqual(
            self.backend.batch_set_v1(self.keys[:2], torch.arange(4)), [True, True]
        )
        self.assertEqual(
            self.backend.batch_get_v1(self.keys[:2], torch.arange(8, 12)), [True, True]
        )
        for buffer in self.pool.buffers:
            self.assertTrue(torch.equal(buffer[:2], buffer[4:6]))
        # Missing one component must not produce a valid page.
        identity = block_key(self.keys[0], self.backend._namespace, "kv")
        self.transport.storage[identity].pop("kv_1")
        self.assertEqual(
            self.backend.batch_get_v1(self.keys[:1], torch.arange(12, 14)), [False]
        )

    def test_invalid_startup_budget_is_rejected_before_registration(self):
        """An invalid optional deadline must not reach native registration."""
        for budget in (0, -1, True, 1.5, "30000", None):
            with self.subTest(budget=budget):
                self.config.extra_config = {"startup_timeout_seconds": budget}
                with self.assertRaisesRegex(ValueError, "startup_timeout_seconds"):
                    KVCRStore(self.config, self.pool)

    def test_invalid_descriptor_coalescing_is_rejected_before_registration(self):
        """A truthy string/integer must not silently enable a copy-order change."""
        for value in (0, 1, None, "true", []):
            with self.subTest(value=value):
                self.config.extra_config = {"coalesce_remote_dram_descriptors": value}
                with self.assertRaisesRegex(
                    ValueError, "coalesce_remote_dram_descriptors"
                ):
                    KVCRStore(self.config, self.pool)

    def test_invalid_ucx_split_is_rejected_before_registration(self):
        """Only native descriptor counts, not booleans or coercible values, bind."""
        for value in (0, -1, True, False, 1.5, "32", None, 2**32):
            with self.subTest(value=value):
                self.config.extra_config = {"nixl_ucx_split_batch_size": value}
                with self.assertRaisesRegex(ValueError, "nixl_ucx_split_batch_size"):
                    KVCRStore(self.config, self.pool)

    def test_logical_anchor_queries_intersect_physical_all_page_pools(self):
        """A DSv4 logical anchor proves nothing without its physical sidecars."""
        backend = KVCRStore(self.config, SimpleNamespace(kv_buffer=None))
        for name in (
            PoolName.DEEPSEEK_V4_C4,
            PoolName.DEEPSEEK_V4_C4_INDEXER,
            PoolName.SWA,
            PoolName.DEEPSEEK_V4_C4_STATE,
        ):
            backend.register_mem_host_pool_v2(_HostPool(), name)
        backend._namespace = storage_namespace(self.config, backend._layouts)
        transport = backend._runner = _CopyTransport(backend._layouts)
        self.assertEqual(backend.batch_exists(self.keys), 0)

        for name, count in (
            (PoolName.DEEPSEEK_V4_C4, 4),
            (PoolName.DEEPSEEK_V4_C4_INDEXER, 2),
        ):
            for key in self.keys[:count]:
                transport.storage[block_key(key, backend._namespace, str(name))] = {}
        # Missing SWA/state pages do not constrain the bare all-pages probe.
        result = backend.batch_exists_v2(self.keys)
        self.assertEqual(result.kv_hit_pages, 2)
        self.assertEqual(result.restorable_prefix_pages, [1, 2])

    def test_bare_logical_anchor_without_all_page_evidence_is_a_miss(self):
        """Empty or trailing-only registrations cannot prove a full prefix."""
        for pools in ((), (PoolName.SWA, PoolName.MAMBA)):
            with self.subTest(pools=pools):
                backend = KVCRStore(self.config, SimpleNamespace(kv_buffer=None))
                for name in pools:
                    backend.register_mem_host_pool_v2(_HostPool(), name)
                backend._namespace = storage_namespace(self.config, backend._layouts)
                backend._runner = _CopyTransport(backend._layouts)
                self.assertEqual(backend.batch_exists(self.keys), 0)
                self.assertEqual(backend.batch_exists([]), 0)

    def test_constructor_failure_is_fatal_and_retains_unsafe_mappings(self):
        """A startup error cannot leave wait_complete behind a dead IO thread.

        A quiescent failure may unwind mappings, but a nonquiescent failure
        must retain them until process exit. The unset optional deadline must
        also work with older KVCR configuration constructors.
        """

        class StartupError(RuntimeError):
            pass

        original_mmap = mmap.mmap
        for error_type in (RuntimeError, StartupError):
            for budget, coalesce, split in (
                (None, None, None),
                (180, None, None),
                (None, False, None),
                (None, True, None),
                (180, True, None),
                (None, None, 32),
                (180, True, 32),
                (None, True, 1),
            ):
                with self.subTest(
                    error_type=error_type, budget=budget, coalesce=coalesce, split=split
                ):
                    self.config.extra_config = {"local_dram_bytes": 128}
                    if budget is not None:
                        self.config.extra_config["startup_timeout_seconds"] = budget
                    if coalesce is not None:
                        self.config.extra_config["coalesce_remote_dram_descriptors"] = (
                            coalesce
                        )
                    if split is not None:
                        self.config.extra_config["nixl_ucx_split_batch_size"] = split
                    backend = KVCRStore(self.config, self.pool)
                    captured, mappings = [], []

                    def construct(captured=captured, error_type=error_type, **kwargs):
                        captured.append(vars(kwargs["config"]))
                        raise error_type("registration failed")

                    def allocate(*args, mappings=mappings, **kwargs):
                        allocation = original_mmap(*args, **kwargs)
                        mappings.append(allocation)
                        return allocation

                    modules = {
                        "kvcr": SimpleNamespace(
                            KVCR=construct, KVCRBindings=SimpleNamespace
                        ),
                        "kvcr.config": SimpleNamespace(
                            KVCRBackendConfigs=SimpleNamespace,
                            KVCRConfig=SimpleNamespace,
                            LocalDramOptions=SimpleNamespace,
                        ),
                        "kvcr.control_channels": SimpleNamespace(
                            ZmqPeerControlChannel=lambda **kwargs: SimpleNamespace(
                                endpoint="tcp://127.0.0.1:32100", close=lambda: None
                            )
                        ),
                        "kvcr.types": SimpleNamespace(
                            KVCRStartupError=StartupError,
                            RegionDescriptor=SimpleNamespace,
                        ),
                    }
                    try:
                        with (
                            patch.dict("sys.modules", modules),
                            patch(
                                "sglang.srt.mem_cache.storage.kvcr.kvcr_store.mmap.mmap",
                                side_effect=allocate,
                            ),
                            patch(
                                "sglang.srt.mem_cache.storage.kvcr.kvcr_store.os._exit",
                                side_effect=SystemExit(1),
                            ),
                            self.assertLogs(
                                "sglang.srt.mem_cache.storage.kvcr.kvcr_store",
                                level="CRITICAL",
                            ),
                            self.assertRaises(SystemExit) as fatal,
                        ):
                            backend._start()
                        self.assertEqual(fatal.exception.code, 1)
                        self.assertIsNone(backend._runner)
                        self.assertIsNone(backend._poll_thread)
                        self.assertTrue(mappings)
                        self.assertEqual(
                            [allocation.closed for allocation in mappings],
                            [error_type is RuntimeError] * len(mappings),
                        )
                        if budget is None:
                            self.assertNotIn("startup_timeout_seconds", captured[0])
                        else:
                            self.assertEqual(
                                captured[0]["startup_timeout_seconds"], budget
                            )
                        if coalesce is True:
                            self.assertIs(
                                captured[0]["coalesce_remote_dram_descriptors"], True
                            )
                        else:
                            self.assertNotIn(
                                "coalesce_remote_dram_descriptors", captured[0]
                            )
                        if split is None:
                            self.assertNotIn("nixl_ucx_split_batch_size", captured[0])
                        else:
                            self.assertEqual(
                                captured[0]["nixl_ucx_split_batch_size"], split
                            )
                    finally:
                        for allocation in mappings:
                            allocation.close()

    def test_byte_verification_covers_every_successful_page_component(self):
        """Diagnostic digests follow physical bytes, not slot IDs or only primary KV."""
        self.backend._verify_transfer_bytes = True
        self.pool.buffers[0][0].fill_(17)
        self.pool.buffers[1][0].fill_(29)
        with self.assertLogs(
            "sglang.srt.mem_cache.storage.kvcr.kvcr_store", level="INFO"
        ) as captured:
            self.backend.batch_set_v1(self.keys[:1], torch.arange(2))
            self.backend.batch_get_v1(self.keys[:1], torch.arange(8, 10))
            # Changing only the unequal-sized side component changes the digest.
            self.pool.buffers[1][4, 1] ^= 1
            self.backend.batch_set_v1(self.keys[:1], torch.arange(8, 10))
            missing = self.backend.batch_get_v1(self.keys[1:2], torch.arange(12, 14))
        records = [
            json.loads(line.split("KVCR L3 byte verification ", 1)[1])
            for line in captured.output
        ]
        self.assertEqual(records[0]["digests"], records[1]["digests"])
        self.assertNotEqual(records[1]["digests"], records[2]["digests"])
        self.assertEqual(records[3]["digests"], {})
        self.assertEqual(missing, [False])

    def test_metrics_count_complete_logical_pages_and_reset_intervals(self):
        """A failed sidecar excludes its page, not successfully moved KV bytes."""
        self.config.enable_storage_metrics = True
        indexer = _HostPool(sizes=(5,))
        backend = KVCRStore(self.config, self.pool)
        backend.register_mem_host_pool_v2(indexer, PoolName.INDEXER)
        backend._namespace = storage_namespace(self.config, backend._layouts)
        transport = backend._runner = _CopyTransport(backend._layouts)
        transfers = [
            PoolTransfer(name, keys=self.keys[:2], host_indices=torch.arange(4))
            for name in (PoolName.KV, PoolName.INDEXER)
        ]
        with patch(
            "sglang.srt.mem_cache.storage.kvcr.kvcr_store.time.perf_counter",
            side_effect=[10.0, 12.0, 20.0, 24.0],
        ):
            backend.batch_set_v2(transfers)
            # Two primary pages exist, but only one has its required indexer.
            transport.storage.pop(
                block_key(self.keys[1], backend._namespace, "indexer")
            )
            restored = backend.batch_get_v2(
                [
                    PoolTransfer(
                        name, keys=self.keys[:3], host_indices=torch.arange(8, 14)
                    )
                    for name in (PoolName.KV, PoolName.INDEXER)
                ]
            )
        self.assertEqual(restored["kv"], [True, True, False])
        self.assertEqual(restored["indexer"], [True, False, False])
        snapshot = backend.get_stats()
        self.assertEqual(snapshot.backup_pgs, [2])
        self.assertEqual(snapshot.prefetch_pgs, [1])
        self.assertAlmostEqual(snapshot.backup_bandwidth[0], 32 / 2 / 1e9)
        self.assertAlmostEqual(snapshot.prefetch_bandwidth[0], 27 / 4 / 1e9)
        self.assertEqual(backend.get_stats().prefetch_pgs, [])
        self.assertEqual(backend.get_stats().backup_pgs, [])

    def test_prefix_alignment_preserves_parents_before_tail_allocation(self):
        """Refreshing after deposit is too late: allocation already evicted a parent.

        A hot prefix exists in both KV and indexer while an unrelated backup
        fills each three-page L3 pool. Backing up its next page must preserve
        every parent in every pool, not only the primary KV. The unaligned
        control loses the root; alignment evicts the unrelated page instead.
        Native bounded-LRU behavior is also covered by the manual NIXL gate.
        """
        for aligned in (False, True):
            with self.subTest(aligned=aligned):
                config = SimpleNamespace(**vars(self.config))
                config.extra_config = {"align_storage_prefix": aligned}
                backend = KVCRStore(config, self.pool)
                indexer = _HostPool(sizes=(5,))
                backend.register_mem_host_pool_v2(indexer, PoolName.INDEXER)
                backend._namespace = storage_namespace(config, backend._layouts)
                backend._runner = _BoundedCopyTransport(backend._layouts, 3)
                for pool in (self.pool, indexer):
                    for part, buffer in enumerate(pool.buffers):
                        for slot in range(4):
                            buffer[slot].fill_(11 + slot + part)

                def transfers(keys, indices):
                    return [
                        PoolTransfer(name, keys=keys, host_indices=indices)
                        for name in (PoolName.KV, PoolName.INDEXER)
                    ]

                backend.batch_set_v2(transfers(self.keys[:2], torch.arange(4)))
                # Unrelated newest page competes with the reused prefix.
                backend.batch_set_v2(transfers(self.keys[2:3], torch.arange(4, 6)))
                backend.batch_set_v2(
                    transfers(self.keys[3:], torch.arange(6, 8)),
                    HiCacheStorageExtraInfo(prefix_keys=self.keys[:2]),
                )
                chain = self.keys[:2] + self.keys[3:]
                hit = backend.batch_exists_v2(chain, [PoolTransfer(PoolName.INDEXER)])
                self.assertEqual(hit.kv_hit_pages, 3 if aligned else 0)
                if aligned:
                    restored = backend.batch_get_v2(
                        transfers(chain, torch.arange(8, 14))
                    )
                    self.assertTrue(all(all(values) for values in restored.values()))
                    self.assertEqual(
                        backend.batch_exists_v2(
                            self.keys[2:3], [PoolTransfer(PoolName.INDEXER)]
                        ).kv_hit_pages,
                        0,
                    )
                    for pool in (self.pool, indexer):
                        for buffer in pool.buffers:
                            self.assertTrue(torch.equal(buffer[[0, 1, 3]], buffer[4:7]))

    def test_alignment_config_rejects_truthy_non_booleans(self):
        for value in ("false", 1, None):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    ValueError, "align_storage_prefix must be a boolean"
                ),
            ):
                config = SimpleNamespace(**vars(self.config))
                config.extra_config = {"align_storage_prefix": value}
                KVCRStore(config, self.pool)

    def test_diagnostic_intervals_preserve_labels_and_bound_sample_memory(self):
        """Unpolled telemetry must not grow with operations or merge distinct scopes."""
        stats = DiagnosticStats(max_series=3)
        for _ in range(10000):
            stats.observe_histogram("duration", 0.001, ("source_write", "success"))
        stats.observe_histogram("duration", 0.02, ("source_write", "success"))
        stats.increase_counter("bytes", 512, ("source_write",))
        stats.increase_counter("bytes", 128, ("source_write",))
        stats.increase_counter("bytes", 256, ("local_fill",))
        # An unexpected series storm drops observations instead of growing
        # without bound or failing an otherwise valid KV transfer.
        for index in range(100):
            stats.set_gauge("unexpected", index, (str(index),))
        snapshot = stats.reduce()
        self.assertEqual(snapshot["dropped_observations"], 100)
        result = {
            tuple(json.loads(k)[:2]) + (tuple(json.loads(k)[2]), json.loads(k)[3]): v
            for k, v in snapshot.items()
            if k != "dropped_observations"
        }
        hist = ("histogram", "duration", ("source_write", "success"))
        self.assertEqual(result[(*hist, "count")], 10001)
        self.assertAlmostEqual(result[(*hist, "sum")], 10.02)
        self.assertEqual(result[(*hist, "bucket_2")], 10000)
        self.assertEqual(result[(*hist, "bucket_5")], 1)
        self.assertEqual(result[("counter", "bytes", ("source_write",), "value")], 640)
        self.assertEqual(result[("counter", "bytes", ("local_fill",), "value")], 256)
        self.assertLessEqual(len(result), 9)
        self.assertTrue(stats.is_empty())
        self.assertEqual(stats.reduce(), {})

    def test_diagnostic_io_reports_terminal_miss_without_changing_restoration(self):
        """Diagnostics must not turn a failed physical page into a successful hit."""
        self.backend._diagnostic_telemetry = True
        self.backend._control_endpoint = "tcp://source:32000"
        self.pool.buffers[0][0].fill_(43)
        self.pool.buffers[1][0].fill_(57)
        with self.assertLogs(
            "sglang.srt.mem_cache.storage.kvcr.kvcr_store", level="INFO"
        ) as captured:
            self.backend.batch_set_v1(self.keys[:1], torch.arange(2))
            restored = self.backend.batch_get_v1(self.keys[:2], torch.arange(8, 12))
        records = [
            json.loads(line.split("KVCR L3 operation ", 1)[1])
            for line in captured.output
        ]
        self.assertEqual(restored, [True, False])
        for buffer in self.pool.buffers:
            self.assertTrue(torch.equal(buffer[0], buffer[4]))
        self.assertEqual(records[0]["failed_keys"], [])
        self.assertEqual(
            records[1]["failed_keys"],
            [block_key(self.keys[1], self.backend._namespace, "kv").decode()],
        )
        self.assertEqual(records[1]["requested_blocks"], 2)
        self.assertEqual(records[1]["requested_spans"], 4)
        self.assertGreaterEqual(records[1]["wait_ms"], 0)
        for record in records:
            self.assertAlmostEqual(
                record["prepare_ms"],
                sum(
                    record[field]
                    for field in ("descriptor_ms", "align_before_ms", "hint_ms")
                ),
            )

    def test_diagnostic_config_rejects_truthy_non_booleans(self):
        for value in ("false", 1, None):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    ValueError, "diagnostic_telemetry must be a boolean"
                ),
            ):
                config = SimpleNamespace(**vars(self.config))
                config.extra_config = {"diagnostic_telemetry": value}
                KVCRStore(config, self.pool)

    def _controller_with_indexer(
        self, *, combined=True, pages=4, batch_pages=128, pipeline_window=1
    ):
        config = SimpleNamespace(**vars(self.config))
        config.extra_config = {
            "combine_page_reads": combined,
            "prefetch_batch_pages": batch_pages,
            "delivery_pipeline_window": pipeline_window,
        }
        pools = {
            PoolName.KV: _HostPool(pages=pages * 2),
            PoolName.INDEXER: _HostPool(sizes=(5,), pages=pages * 2),
        }
        backend = KVCRStore(config, pools[PoolName.KV])
        backend.register_mem_host_pool_v2(pools[PoolName.INDEXER], PoolName.INDEXER)
        backend._namespace = storage_namespace(config, backend._layouts)
        transport = backend._runner = _CopyTransport(backend._layouts)
        controller = HiCacheController.__new__(HiCacheController)
        controller.storage_backend = backend
        controller.page_size = 2
        controller.page_get_func = controller._page_get_zero_copy
        controller.prefetch_sync_queue = Queue()
        operation = PrefetchOperation("combined-page-read", list(range(pages * 2)))
        operation.host_indices = torch.arange(pages * 2, pages * 4)
        operation.hash_value = (
            self.keys if pages == 4 else [f"{page:064x}" for page in range(pages)]
        )
        operation.pool_transfers = [
            PoolTransfer(PoolName.INDEXER, indices_from_pool=PoolName.KV)
        ]
        return controller, operation, pools, transport

    def test_combined_controller_reads_restore_all_components_in_one_completion(self):
        """Coalescing changes submissions, not KV/indexer bytes or valid prefix length."""
        for combined in (False, True):
            with self.subTest(combined=combined):
                c, operation, pools, transport = self._controller_with_indexer(
                    combined=combined
                )
                for pool in pools.values():
                    for part, buffer in enumerate(pool.buffers):
                        for slot in range(4):
                            buffer[slot].fill_(13 + slot + part)
                c.storage_backend.batch_set_v2(
                    [
                        PoolTransfer(name, keys=self.keys, host_indices=torch.arange(8))
                        for name in pools
                    ]
                )
                before = transport.next_handle
                hits = c._page_transfer_kv_batch(
                    operation,
                    self.keys,
                    operation.host_indices,
                    None,
                    operation.pool_transfers,
                )
                self.assertEqual(hits, 4)
                self.assertEqual(transport.next_handle - before, 1 if combined else 2)
                for pool in pools.values():
                    for buffer in pool.buffers:
                        self.assertTrue(torch.equal(buffer[:4], buffer[4:8]))

    def test_combined_partial_pool_miss_keeps_every_batch_ack(self):
        """A sidecar hole clamps the prefix, but later skipped batches still get acks.

        Returning KV's full hit count admits incomplete indexer data. Returning
        early from the outer transfer loop strands ranks waiting for later acks.
        """
        c, operation, pools, transport = self._controller_with_indexer(batch_pages=2)
        for pool in pools.values():
            for buffer in pool.buffers:
                buffer[:4].fill_(63)
        c.storage_backend.batch_set_v2(
            [
                PoolTransfer(name, keys=self.keys, host_indices=torch.arange(8))
                for name in pools
            ]
        )
        transport.storage.pop(
            block_key(self.keys[1], c.storage_backend._namespace, "indexer")
        )
        with self.assertLogs(level="WARNING"):
            completed = c._page_transfer(operation)
        self.assertEqual(completed, 1)
        acks = [c.prefetch_sync_queue.get_nowait() for _ in range(2)]
        self.assertEqual([ack.completed_tokens for ack in acks], [2, 2])
        self.assertTrue(c.prefetch_sync_queue.empty())
        for pool in pools.values():
            for buffer in pool.buffers:
                self.assertTrue(torch.equal(buffer[0], buffer[4]))
                self.assertEqual(int(buffer[6].sum()), 0)

    def test_prefetch_batch_size_preserves_bytes_and_ack_prefix_after_sidecar_hole(
        self,
    ):
        """Larger batches reduce completions without admitting incomplete pages.

        A hole inside either the first or middle batch must stop the usable
        prefix and leave later batches untouched, while emitting the same ACK
        count as a successful peer. Default and opt-in paths restore identical
        bytes, including a short final batch.
        """
        pages = 1100
        cases = (
            (128, None, 9, [256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2200]),
            (512, None, 3, [1024, 2048, 2200]),
            (512, 600, 2, [1024, 1200, 1200]),
            (1024, None, 2, [2048, 2200]),
            (1024, 600, 1, [1200, 1200]),
        )
        for batch_pages, missing, submissions, expected_acks in cases:
            with self.subTest(batch_pages=batch_pages, missing=missing):
                c, op, pools, transport = self._controller_with_indexer(
                    pages=pages, batch_pages=batch_pages
                )
                for pool in pools.values():
                    for part, buffer in enumerate(pool.buffers):
                        values = (
                            torch.arange(pages * buffer.shape[1]).reshape(
                                pages, buffer.shape[1]
                            )
                            + part * 17
                        ) % 251
                        buffer[:pages].copy_(values.to(torch.uint8))
                c.storage_backend.batch_set_v2(
                    [
                        PoolTransfer(
                            name,
                            keys=op.hash_value,
                            host_indices=torch.arange(pages * 2),
                        )
                        for name in pools
                    ]
                )
                if missing is not None:
                    transport.storage.pop(
                        block_key(
                            op.hash_value[missing],
                            c.storage_backend._namespace,
                            "indexer",
                        )
                    )
                before = transport.next_handle
                with (
                    self.assertLogs(level="WARNING")
                    if missing is not None
                    else ExitStack()
                ):
                    restored = c._page_transfer(op)
                self.assertEqual(restored, pages if missing is None else missing)
                self.assertEqual(transport.next_handle - before, submissions)
                acks = []
                while not c.prefetch_sync_queue.empty():
                    acks.append(c.prefetch_sync_queue.get_nowait().completed_tokens)
                self.assertEqual(acks, expected_acks)
                for pool in pools.values():
                    for buffer in pool.buffers:
                        self.assertTrue(
                            torch.equal(
                                buffer[:restored], buffer[pages : pages + restored]
                            )
                        )
                        if missing is not None:
                            self.assertEqual(int(buffer[pages + 1024 :].sum()), 0)
                if missing is not None:
                    self.assertEqual(
                        int(pools[PoolName.INDEXER].buffers[0][pages + missing].sum()),
                        0,
                    )

    def test_prefetch_batch_config_rejects_values_that_break_collective_sequence(self):
        for value in (0, -1, True, False, 1.5, "512", None):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    ValueError, "prefetch_batch_pages must be a positive integer"
                ),
            ):
                config = SimpleNamespace(**vars(self.config))
                config.extra_config = {"prefetch_batch_pages": value}
                KVCRStore(config, self.pool)

    def test_query_batch_default_and_optin_do_not_change_payload_batching(self):
        self.assertEqual(self.backend.query_batch_pages, 128)
        for pages in (1, 128, 2048):
            with self.subTest(pages=pages):
                config = SimpleNamespace(**vars(self.config))
                config.extra_config = {"query_batch_pages": pages}
                store = KVCRStore(config, self.pool)
                self.addCleanup(store.close)
                self.assertEqual(store.query_batch_pages, pages)
                self.assertEqual(store.prefetch_batch_pages, 128)
                self.assertEqual(store._delivery_pipeline_window, 1)
                self.assertEqual(set(store._layouts), {"kv"})

    def test_query_batch_config_rejects_non_integer_or_unbounded_values(self):
        for value in (0, -1, True, False, 1.5, "2048", None, 2049):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    ValueError, "query_batch_pages must be an integer from 1 to 2048"
                ),
            ):
                config = SimpleNamespace(**vars(self.config))
                config.extra_config = {"query_batch_pages": value}
                KVCRStore(config, self.pool)

    def test_query_batch_optin_rejects_MLA_and_logical_anchor(self):
        for mla, pool in ((True, self.pool), (False, SimpleNamespace(kv_buffer=None))):
            with (
                self.subTest(mla=mla),
                self.assertRaisesRegex(
                    ValueError,
                    "query_batch_pages opt-in requires a non-MLA KV-only layout",
                ),
            ):
                config = SimpleNamespace(**vars(self.config))
                config.is_mla_model = mla
                config.extra_config = {"query_batch_pages": 2048}
                KVCRStore(config, pool)

    def test_query_batch_optin_rejects_sidecars_before_mutating_layouts(self):
        config = SimpleNamespace(**vars(self.config))
        config.extra_config = {"query_batch_pages": 2048}
        store = KVCRStore(config, self.pool)
        self.addCleanup(store.close)
        for name in (PoolName.INDEXER, PoolName.MAMBA):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    ValueError,
                    "query_batch_pages opt-in requires a non-MLA KV-only layout",
                ),
            ):
                store.register_mem_host_pool_v2(_HostPool(), name)
            self.assertEqual(set(store._layouts), {"kv"})
            self.assertEqual(set(store.registered_pools), {"kv"})

    def test_query_batch_controller_preserves_exact_prefix_and_hint_scope(self):
        pages, page_size = 4097, 2
        hashes = [f"{i + 1:064x}" for i in range(pages)]
        for batch in (128, 2048):
            for missing in (0, 1, 127, 128, 129, 2047, 2048, 2049, pages - 1, pages):
                for prefix in (None, [], ["prior-prefix"]):
                    with self.subTest(batch=batch, missing=missing, prefix=prefix):
                        calls = []
                        hint = {"request": "not-a-global-cache-key"}
                        operation = PrefetchOperation(
                            "query-prefix",
                            list(range(pages * page_size)),
                            last_hash="previous-hash",
                            prefix_keys=prefix,
                        )
                        operation.kv_hints = hint

                        def exists(keys, extra):
                            start = len(calls) * batch
                            self.assertEqual(keys, hashes[start : start + batch])
                            self.assertIs(extra.extra_info["kv_hints"], hint)
                            expected = (
                                None if prefix is None else prefix + hashes[:start]
                            )
                            self.assertEqual(extra.prefix_keys, expected)
                            calls.append(list(keys))
                            return min(len(keys), max(0, missing - start))

                        controller = HiCacheController.__new__(HiCacheController)
                        controller.page_size = page_size
                        controller.storage_backend = SimpleNamespace(
                            query_batch_pages=batch, batch_exists=exists
                        )
                        with patch(
                            "sglang.srt.managers.cache_controller.get_storage_hash_str",
                            return_value=hashes,
                        ) as codec:
                            result = controller._storage_hit_query(operation)
                        self.assertEqual(
                            result, (hashes[:missing], missing * page_size)
                        )
                        self.assertEqual(operation.all_hash_values, hashes)
                        self.assertEqual(operation.prefix_keys, prefix)
                        self.assertEqual(
                            len(calls),
                            min(missing // batch + 1, (pages + batch - 1) // batch),
                        )
                        codec.assert_called_once_with(
                            operation.token_ids, "previous-hash", page_size=page_size
                        )

    def test_query_batch_controller_empty_chain_has_no_backend_call(self):
        controller = HiCacheController.__new__(HiCacheController)
        controller.page_size = 2

        def unexpected(*args):
            self.fail("Empty input must not query storage")

        controller.storage_backend = SimpleNamespace(
            query_batch_pages=2048, batch_exists=unexpected
        )
        operation = PrefetchOperation("empty-query", [])
        with patch(
            "sglang.srt.managers.cache_controller.get_storage_hash_str", return_value=[]
        ):
            self.assertEqual(controller._storage_hit_query(operation), ([], 0))
        self.assertEqual(operation.all_hash_values, [])

    def test_combined_result_rejects_missing_or_truncated_sidecar(self):
        """Dropping a pool/result entry must not masquerade as an all-pool hit."""
        c, operation, _, _ = self._controller_with_indexer()
        malformed = [
            {"kv": [True] * 4},
            {"kv": [True] * 4, "indexer": [True]},
            {"kv": [True] * 4, "indexer": None},
            {"kv": [True] * 4, "indexer": [1] * 4},
        ]
        for result in malformed:
            with (
                self.subTest(result=result),
                patch.object(c.storage_backend, "batch_get_v2", return_value=result),
                self.assertLogs(level="ERROR"),
            ):
                self.assertEqual(
                    c._page_transfer_kv_batch(
                        operation,
                        self.keys,
                        operation.host_indices,
                        None,
                        operation.pool_transfers,
                    ),
                    0,
                )

    def test_combined_read_config_rejects_truthy_non_booleans(self):
        for value in ("false", 1, None):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    ValueError, "combine_page_reads must be a boolean"
                ),
            ):
                config = SimpleNamespace(**vars(self.config))
                config.extra_config = {"combine_page_reads": value}
                KVCRStore(config, self.pool)

    def _backup_with_indexer(self, *, combined=True, pages=130):
        base, read, pools, transport = self._controller_with_indexer(pages=pages)
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.__dict__.update(base.__dict__)
        controller.mem_pool_host = SimpleNamespace(
            kv_buffer=pools[PoolName.KV].kv_buffer, layout_lease=ExitStack
        )
        controller.backup_skip = False
        controller.page_set_func = controller._page_set_zero_copy
        controller.storage_backend.supports_combined_page_writes = combined
        operation = StorageOperation(
            torch.arange(pages * 2),
            list(range(pages * 2)),
            hash_value=read.hash_value,
            prefix_keys=["f" * 64],
            pool_transfers=[
                PoolTransfer(PoolName.INDEXER, indices_from_pool=PoolName.KV)
            ],
        )
        for pool in pools.values():
            for part, buffer in enumerate(pool.buffers):
                values = (
                    torch.arange(pages * buffer.shape[1]).reshape(pages, -1) + part * 17
                ) % 251
                buffer[:pages].copy_(values.to(torch.uint8))
        return controller, operation, pools, transport

    def test_combined_backup_preserves_bytes_and_short_final_batch(self):
        """One completion covers KV plus indexer, with the same restorable bytes."""
        for combined in (False, True):
            with self.subTest(combined=combined):
                c, operation, pools, transport = self._backup_with_indexer(
                    combined=combined
                )
                c._page_backup(operation)
                self.assertEqual(operation.completed_tokens, 260)
                self.assertEqual(
                    operation.pool_storage_result.extra_pool_hit_pages, {"indexer": 130}
                )
                self.assertEqual(transport.next_handle, 2 if combined else 3)
                restored = c.storage_backend.batch_get_v2(
                    [
                        PoolTransfer(
                            name,
                            keys=operation.hash_value,
                            host_indices=torch.arange(260, 520),
                        )
                        for name in pools
                    ]
                )
                self.assertTrue(all(all(values) for values in restored.values()))
                for pool in pools.values():
                    for buffer in pool.buffers:
                        self.assertTrue(torch.equal(buffer[:130], buffer[130:]))

    def test_combined_backup_never_acknowledges_incomplete_sidecar_batch(self):
        """Malformed or failed sidecars stop ACK growth and leave later batches idle."""
        malformed = (
            {"kv": [True] * 128},
            {"kv": [True] * 128, "indexer": [True]},
            {"kv": [True] * 128, "indexer": [1] * 128},
            {"kv": [True] * 128, "indexer": [False] + [True] * 127},
        )
        for result in malformed:
            with self.subTest(result=result):
                c, operation, _, transport = self._backup_with_indexer(pages=270)
                real_set = c.storage_backend.batch_set_v2
                batches = []

                def fail_second(
                    transfers,
                    extra_info=None,
                    batches=batches,
                    result=result,
                    real_set=real_set,
                ):
                    batches.append(list(extra_info.prefix_keys))
                    if len(batches) == 2:
                        return result
                    return real_set(transfers, extra_info)

                with (
                    patch.object(c.storage_backend, "batch_set_v2", fail_second),
                    self.assertLogs(level="WARNING"),
                ):
                    c._page_backup(operation)
                self.assertEqual(operation.completed_tokens, 256)
                self.assertEqual(len(batches), 2)
                self.assertEqual(batches[0], ["f" * 64])
                self.assertEqual(batches[1], ["f" * 64] + operation.hash_value[:128])
                self.assertEqual(transport.next_handle, 1)
                self.assertNotIn(
                    block_key(
                        operation.hash_value[256], c.storage_backend._namespace, "kv"
                    ),
                    transport.storage,
                )

    def test_combined_backup_does_not_reinterpret_trailing_sidecar_indices(self):
        """A separately indexed trailing checkpoint must retain its own source slot."""
        c, operation, pools, transport = self._backup_with_indexer()
        operation.pool_transfers = [
            PoolTransfer(
                PoolName.INDEXER,
                keys=operation.hash_value[-1:],
                host_indices=torch.arange(258, 260),
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ]
        c._page_backup(operation)
        self.assertEqual(operation.completed_tokens, 260)
        self.assertEqual(transport.next_handle, 3)
        self.assertEqual(
            operation.pool_storage_result.extra_pool_hit_pages, {"indexer": 1}
        )
        result = c.storage_backend.batch_get_v2(
            [
                PoolTransfer(
                    PoolName.INDEXER,
                    keys=operation.hash_value[-1:],
                    host_indices=torch.arange(260, 262),
                )
            ]
        )
        self.assertEqual(result, {"indexer": [True]})
        for buffer in pools[PoolName.INDEXER].buffers:
            self.assertTrue(torch.equal(buffer[129], buffer[130]))

    def test_combined_write_config_rejects_truthy_non_booleans(self):
        for value in ("false", 1, None):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    ValueError, "combine_page_writes must be a boolean"
                ),
            ):
                config = SimpleNamespace(**vars(self.config))
                config.extra_config = {"combine_page_writes": value}
                KVCRStore(config, self.pool)

    def _shared_reader(self, rank, **changes):
        config = SimpleNamespace(
            **{
                **vars(self.config),
                "tp_rank": rank,
                "is_mla_model": True,
                "extra_config": {"share_replicated_mla": True},
                **changes,
            }
        )
        store = KVCRStore(config, _HostPool())
        store._namespace = storage_namespace(config, store._layouts, shared_mla=True)
        store._runner = _CopyTransport(store._layouts)
        store._owner_endpoint = "tcp://127.0.0.1:32100"
        return store

    def test_shared_mla_namespace_is_rank_invariant_but_not_legacy_compatible(self):
        """Replicated readers share identity; ordinary shard keys remain isolated."""
        owner, reader = self._shared_reader(0), self._shared_reader(1)
        self.assertEqual(owner._namespace, reader._namespace)
        self.assertNotEqual(
            reader._namespace, storage_namespace(reader.config, reader._layouts)
        )
        changed = self._shared_reader(1, model_name="different-model")
        self.assertNotEqual(reader._namespace, changed._namespace)
        config = SimpleNamespace(**vars(owner.config))
        config.tp_rank = 1
        self.assertNotEqual(
            storage_namespace(owner.config, owner._layouts),
            storage_namespace(config, owner._layouts),
        )

    def test_shared_reader_selects_tp0_and_retains_replica_local_fallback(self):
        """Missing/disabled external hints must still allow own-replica L3 reads."""
        reader = self._shared_reader(1)
        key = block_key(self.keys[0], reader._namespace, "kv")
        extra = HiCacheStorageExtraInfo(
            extra_info={
                "kv_hints": {
                    "protocol_version": "0.1",
                    "actions": [
                        {
                            "action_type": "kv.fetch",
                            "action_version": "1.0",
                            "payload": {
                                "source_control_endpoint": "tcp://127.0.0.1:33000",
                                "block_hashes": [0],
                            },
                        }
                    ],
                }
            }
        )
        external = reader._operation_hint(extra, [key])
        self.assertEqual(
            external["actions"][0]["payload"]["source_control_endpoint"],
            "tcp://127.0.0.1:33000",
        )
        for disabled in (False, True):
            reader._extra["enable_remote_hint"] = not disabled
            own = reader._operation_hint(extra if disabled else None, [key])
            self.assertEqual(
                own["actions"][0]["payload"]["source_control_endpoint"],
                reader._owner_endpoint,
            )
            self.assertEqual(
                own["actions"][0]["payload"]["block_hashes"], [int(key[:16], 16)]
            )
        owner = self._shared_reader(0)
        self.assertIsNone(owner._operation_hint(None, [key]))

    def test_shared_mla_refuses_sharded_topologies_and_auxiliary_state(self):
        """Sharing a namespace must never alias KDA/SWA or context/head shards."""
        for changes in (
            {"is_mla_model": False},
            {"pp_size": 2},
            {"attn_cp_size": 2},
            {"should_split_heads": True},
        ):
            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(ValueError, "Shared MLA storage requires"),
            ):
                self._shared_reader(0, **changes)
        reader = self._shared_reader(1)
        with self.assertRaisesRegex(ValueError, "only supports replicated KV/indexer"):
            reader.register_mem_host_pool_v2(_HostPool(), PoolName.MAMBA)
        for value in ("false", 1, None):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    ValueError, "share_replicated_mla must be a boolean"
                ),
            ):
                self._shared_reader(0, extra_config={"share_replicated_mla": value})

    def test_shared_reader_recovers_only_failed_external_pages_from_local_owner(self):
        """An external hint cannot hide local-owner pages or turn real misses into hits.

        Both sources complete before the adapter reports success. Already
        restored external pages are not submitted again, genuinely absent
        pages stay misses, and diagnostics preserve each operation's result.
        """
        reader = self._shared_reader(1)
        reader._runner = None
        reader.register_mem_host_pool_v2(_HostPool((5,)), PoolName.INDEXER)
        reader._namespace = storage_namespace(
            reader.config, reader._layouts, shared_mla=True
        )
        reader._diagnostic_telemetry = True
        reader._control_endpoint = "tcp://127.0.0.1:32101"
        external_endpoint = "tcp://127.0.0.1:33000"
        peers = {reader._owner_endpoint: {}, external_endpoint: {}}
        for name, layout in reader._layouts.items():
            for page, source in enumerate(
                (reader._owner_endpoint, external_endpoint, external_endpoint)
            ):
                peers[source][block_key(self.keys[page], reader._namespace, name)] = {
                    part.name: bytes([17 + page * 13 + component]) * part.size
                    for component, part in enumerate(layout.components)
                }
        transport = reader._runner = _PeerCopyTransport(reader._layouts, peers)
        reader._metrics = SimpleNamespace(prefetch_pgs=[], prefetch_bandwidth=[])
        for pool in reader.registered_pools.values():
            for buffer in pool.buffers:
                buffer.fill_(163)
        extra = HiCacheStorageExtraInfo(
            extra_info={
                "kv_hints": {
                    "protocol_version": "0.1",
                    "actions": [
                        {
                            "action_type": "kv.fetch",
                            "action_version": "1.0",
                            "payload": {
                                "source_control_endpoint": external_endpoint,
                                "block_hashes": [
                                    int(key[:16], 16) for key in self.keys
                                ],
                            },
                        }
                    ],
                }
            }
        )
        with self.assertLogs(
            "sglang.srt.mem_cache.storage.kvcr.kvcr_store", level="INFO"
        ) as captured:
            restored = reader.batch_get_v2(
                [
                    PoolTransfer(name, keys=self.keys, host_indices=torch.arange(8, 16))
                    for name in reader._layouts
                ],
                extra,
            )
        self.assertEqual(
            restored, {name: [True, True, True, False] for name in reader._layouts}
        )
        for pool in reader.registered_pools.values():
            for component, buffer in enumerate(pool.buffers):
                for page in range(3):
                    self.assertTrue(
                        torch.all(buffer[4 + page] == 17 + page * 13 + component)
                    )
                self.assertTrue(torch.all(buffer[7] == 163))
        failed = tuple(
            block_key(key, reader._namespace, name)
            for name in reader._layouts
            for key in (self.keys[0], self.keys[3])
        )
        self.assertEqual(len(transport.reads), 2)
        self.assertEqual(transport.reads[1], (reader._owner_endpoint, failed))
        self.assertEqual(transport.hints, {})
        self.assertEqual(reader._metrics.prefetch_pgs, [3])
        records = [
            json.loads(line.split("KVCR L3 operation ", 1)[1])
            for line in captured.output
        ]
        self.assertEqual(
            [record["stage"] for record in records],
            ["primary", "replica-local-fallback"],
        )
        self.assertEqual(records[0]["failed_keys"], [key.decode() for key in failed])
        self.assertEqual(records[1]["requested_blocks"], 4)
        self.assertEqual(
            records[1]["failed_keys"],
            [
                block_key(self.keys[3], reader._namespace, name).decode()
                for name in reader._layouts
            ],
        )
        # Already-selected local TP0 is never retried against itself.
        with self.assertLogs(
            "sglang.srt.mem_cache.storage.kvcr.kvcr_store", level="INFO"
        ):
            self.assertEqual(
                reader.batch_get_v1(self.keys, torch.arange(8, 16)),
                [True, False, False, False],
            )
        self.assertEqual(len(transport.reads), 3)
        self.assertEqual(transport.hints, {})

    def test_shared_secondary_cannot_create_an_independent_backup(self):
        """A misrouted backup fails before submission, never creating replica-local shards."""
        reader = self._shared_reader(1)
        with self.assertRaisesRegex(RuntimeError, "must run on TP0"):
            reader.batch_set_v1(self.keys[:1], torch.arange(2))
        self.assertEqual(reader._runner.storage, {})
        self.assertEqual(reader._runner.next_handle, 0)

    def _pipeline_fixture(self, pages, *, window=4, logical=False, sidecars=()):
        config = SimpleNamespace(**vars(self.config))
        config.extra_config = (
            {} if window is None else {"delivery_pipeline_window": window}
        )
        primary = (
            SimpleNamespace(kv_buffer=None)
            if logical
            else _HostPool(pages=2 * pages + 8)
        )
        store = KVCRStore(config, primary)
        for name, sizes, page_size in sidecars:
            store.register_mem_host_pool_v2(
                _HostPool(sizes, page_size, pages=2 * pages + 8), name
            )
        store._namespace = storage_namespace(config, store._layouts)
        transport = store._runner = _DeferredCopyTransport(store._layouts)
        keys = [f"{i + 1:064x}" for i in range(pages)]
        for layout in store._layouts.values():
            for component, part in enumerate(layout.components):
                for slot in range(pages):
                    ctypes.memset(
                        part.address + slot * part.stride,
                        (slot + component * 17) % 251 + 1,
                        part.size,
                    )
        return store, transport, keys

    def _pipeline_transfers(self, store, keys, slots):
        return [
            PoolTransfer(
                name,
                keys=keys,
                host_indices=torch.tensor(
                    [
                        slot * layout.page_size + i
                        for slot in slots
                        for i in range(layout.page_size)
                    ]
                ),
            )
            for name, layout in store._layouts.items()
        ]

    def test_pipeline_config_is_strict_before_host_registration(self):
        for value in (None, True, False, 0, -1, 2, 3, 5, 1.0, 4.0, "4", [], {}):
            with self.subTest(value=value):
                config = SimpleNamespace(**vars(self.config))
                config.extra_config = {"delivery_pipeline_window": value}
                with patch.object(KVCRStore, "register_mem_pool_host") as register:
                    with self.assertRaisesRegex(ValueError, "delivery_pipeline_window"):
                        KVCRStore(config, self.pool)
                    register.assert_not_called()

    def test_pipeline_default_shape_and_unsplit_backup(self):
        """Absent/1 keeps one large public operation; opting in only splits reads."""
        pages = 1812
        for window in (None, 1, 4):
            with self.subTest(window=window):
                store, transport, keys = self._pipeline_fixture(
                    pages, window=window, sidecars=((PoolName.INDEXER, (5, 2), 1),)
                )
                store._diagnostic_telemetry = True
                store._control_endpoint = "tcp://target:32100"
                with self.assertLogs(
                    "sglang.srt.mem_cache.storage.kvcr.kvcr_store", level="INFO"
                ) as captured:
                    saved = store.batch_set_v2(
                        self._pipeline_transfers(store, keys, range(pages))
                    )
                    self.assertTrue(all(all(hits) for hits in saved.values()))
                    self.assertEqual(transport.next_handle, 1)
                    restored = store.batch_get_v2(
                        self._pipeline_transfers(store, keys, range(pages, 2 * pages))
                    )
                self.assertTrue(all(all(hits) for hits in restored.values()))
                expected = [512] * 7 + [40] if window == 4 else [2 * pages]
                self.assertEqual([len(read[1]) for read in transport.reads], expected)
                self.assertEqual(store.prefetch_batch_pages, 128)
                records = [
                    json.loads(line.split("KVCR L3 operation ", 1)[1])
                    for line in captured.output
                ]
                self.assertEqual(len(records), 1 + len(expected))
                self.assertEqual(records[0]["direction"], "deposit")
                self.assertEqual(records[0]["requested_blocks"], 2 * pages)
                if window == 4:
                    reads = records[1:]
                    self.assertEqual(
                        [r["pipeline_chunk_index"] for r in reads], list(range(8))
                    )
                    self.assertEqual(len({r["logical_start_epoch"] for r in reads}), 1)
                    for index, record in enumerate(reads):
                        self.assertEqual(record["delivery_pipeline_window"], 4)
                        self.assertLessEqual(
                            record["logical_start_epoch"], record["start_epoch"]
                        )
                        self.assertLessEqual(record["start_epoch"], record["epoch"])
                        if index:
                            self.assertTrue(
                                all(
                                    record[field] == 0
                                    for field in (
                                        "prepare_ms",
                                        "lock_wait_ms",
                                        "descriptor_ms",
                                        "align_before_ms",
                                        "hint_ms",
                                    )
                                )
                            )
                    self.assertAlmostEqual(
                        reads[0]["prepare_ms"],
                        sum(
                            reads[0][field]
                            for field in ("descriptor_ms", "align_before_ms", "hint_ms")
                        ),
                    )
                for pool in store.registered_pools.values():
                    for buffer in pool.buffers:
                        self.assertTrue(
                            torch.equal(buffer[:pages], buffer[pages : 2 * pages])
                        )

    def test_pipeline_boundaries_reverse_completions_and_arbitrary_slots(self):
        for pages in (1, 511, 512, 513, 1812, 2048, 2601):
            with self.subTest(pages=pages):
                store, transport, keys = self._pipeline_fixture(pages)
                store.batch_set_v2(self._pipeline_transfers(store, keys, range(pages)))
                transport.completed[999999] = {b"other-caller": True}
                destination = list(reversed(range(pages, 2 * pages)))
                restored = store.batch_get_v2(
                    self._pipeline_transfers(store, keys, destination)
                )
                self.assertEqual(restored, {"kv": [True] * pages})
                chunks = (pages + 511) // 512
                self.assertEqual(len(transport.reads), chunks)
                self.assertEqual(transport.max_pending, min(4, chunks))
                first_complete = next(
                    i
                    for i, event in enumerate(transport.events)
                    if event[0] == "complete"
                )
                self.assertEqual(first_complete, min(4, chunks))
                self.assertEqual(store._wait(999999), {b"other-caller": True})
                for part in store._layouts["kv"].components:
                    for source, target in enumerate(destination):
                        self.assertEqual(
                            ctypes.string_at(
                                part.address + source * part.stride, part.size
                            ),
                            ctypes.string_at(
                                part.address + target * part.stride, part.size
                            ),
                        )
                self.assertEqual(transport.pending, {})

    def test_pipeline_misses_ignore_other_chunks_keys_and_keep_one_metric_and_digest(
        self,
    ):
        pages = 1100
        store, transport, keys = self._pipeline_fixture(
            pages, sidecars=((PoolName.INDEXER, (5,), 2),)
        )
        store.batch_set_v2(self._pipeline_transfers(store, keys, range(pages)))
        missing = block_key(keys[600], store._namespace, "indexer")
        omitted = block_key(keys[10], store._namespace, "kv")
        transport.storage.pop(missing)
        transport.storage.pop(omitted)

        def corrupt_result(_handle, blocks, result):
            result.pop(omitted, None)
            # Both are real identities from this logical call, but not this handle.
            for key in (missing, omitted):
                if key not in blocks:
                    result[key] = SimpleNamespace(success=True)
            result[b"unsolicited"] = SimpleNamespace(success=True)
            return result

        transport.result_hook = corrupt_result
        store._metrics = SimpleNamespace(prefetch_pgs=[], prefetch_bandwidth=[])
        store._verify_transfer_bytes = store._diagnostic_telemetry = True
        store._control_endpoint = "tcp://target:32100"
        real_metrics = store._record_metrics
        with patch.object(store, "_record_metrics", wraps=real_metrics) as metrics:
            with self.assertLogs(
                "sglang.srt.mem_cache.storage.kvcr.kvcr_store", level="INFO"
            ) as captured:
                restored = store.batch_get_v2(
                    self._pipeline_transfers(store, keys, range(pages, 2 * pages))
                )
            metrics.assert_called_once()
            elapsed = metrics.call_args.kwargs["elapsed"]
        self.assertEqual([i for i, hit in enumerate(restored["kv"]) if not hit], [10])
        self.assertEqual(
            [i for i, hit in enumerate(restored["indexer"]) if not hit], [600]
        )
        self.assertEqual(store._metrics.prefetch_pgs, [pages - 2])
        self.assertAlmostEqual(
            store._metrics.prefetch_bandwidth[0], (16 * pages - 16) / elapsed / 1e9
        )
        operation_records = [
            json.loads(line.split("KVCR L3 operation ", 1)[1])
            for line in captured.output
            if "KVCR L3 operation " in line
        ]
        self.assertEqual(
            [r["requested_blocks"] for r in operation_records],
            [512, 512, 512, 512, 152],
        )
        self.assertEqual(
            {k for r in operation_records for k in r["failed_keys"]},
            {missing.decode(), omitted.decode()},
        )
        digests = json.loads(
            next(
                line.split("KVCR L3 byte verification ", 1)[1]
                for line in captured.output
                if "KVCR L3 byte verification " in line
            )
        )["digests"]
        self.assertEqual(len(digests), 2 * pages - 2)
        self.assertNotIn(missing.decode(), digests)
        self.assertNotIn(omitted.decode(), digests)
        self.assertEqual(transport.pending, {})
        for name, pool in store.registered_pools.items():
            for buffer in pool.buffers:
                absent = 10 if name == "kv" else 600
                self.assertTrue(
                    torch.equal(buffer[:absent], buffer[pages : pages + absent])
                )
                self.assertTrue(
                    torch.equal(
                        buffer[absent + 1 : pages],
                        buffer[pages + absent + 1 : 2 * pages],
                    )
                )
                self.assertEqual(int(buffer[pages + absent].sum()), 0)

    def test_pipeline_logical_anchor_and_independent_trailing_checkpoint_indices(self):
        pages = 600
        store, transport, keys = self._pipeline_fixture(
            pages,
            logical=True,
            sidecars=(
                (PoolName.DEEPSEEK_V4_C4, (7, 3), 2),
                (PoolName.MAMBA, (5, 2), 1),
            ),
        )
        all_pages = PoolName.DEEPSEEK_V4_C4
        transfers = [
            PoolTransfer(PoolName.KV, keys=keys, host_indices=None),
            PoolTransfer(all_pages, keys=keys, host_indices=torch.arange(2 * pages)),
            PoolTransfer(
                PoolName.MAMBA,
                keys=[keys[9], keys[-1]],
                host_indices=torch.tensor([3, 17]),
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ]
        store.batch_set_v2(transfers)
        restored = store.batch_get_v2(
            [
                PoolTransfer(PoolName.KV, keys=keys),
                PoolTransfer(
                    all_pages,
                    keys=keys,
                    host_indices=torch.arange(2 * pages, 4 * pages),
                ),
                PoolTransfer(
                    PoolName.MAMBA,
                    keys=[keys[9], keys[-1]],
                    host_indices=torch.tensor([pages + 7, pages + 31]),
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                ),
            ]
        )
        self.assertEqual(
            restored,
            {
                "kv": [True] * pages,
                str(all_pages): [True] * pages,
                "mamba": [True, True],
            },
        )
        self.assertEqual([len(read[1]) for read in transport.reads], [512, 90])
        for buffer in store.registered_pools[str(all_pages)].buffers:
            self.assertTrue(torch.equal(buffer[:pages], buffer[pages : 2 * pages]))
        for part in store._layouts["mamba"].components:
            for source, target in ((3, pages + 7), (17, pages + 31)):
                self.assertEqual(
                    ctypes.string_at(part.address + source * part.stride, part.size),
                    ctypes.string_at(part.address + target * part.stride, part.size),
                )
        boundary = store.batch_exists_v2(
            keys,
            [
                PoolTransfer(
                    PoolName.MAMBA,
                    keys=[keys[-1]],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                ),
                PoolTransfer(all_pages),
            ],
        )
        self.assertEqual(boundary.restorable_prefix_pages, [10, pages])

    def test_pipeline_duplicate_identity_and_destination_alias_preflight(self):
        for alias in (
            "identity",
            "slot",
            "cross-pool",
            "cross-component",
            "partial-component",
        ):
            with self.subTest(alias=alias):
                pages = 513
                store, transport, keys = self._pipeline_fixture(pages)
                if alias == "identity":
                    keys[-1] = keys[0]
                if alias in ("cross-pool", "cross-component", "partial-component"):
                    # Re-register real overlapping tensor views before first IO.
                    store._runner = None
                    pool = store.registered_pools["kv"]
                    if alias == "cross-pool":
                        store.register_mem_host_pool_v2(pool, PoolName.INDEXER)
                    else:
                        pool.buffers[1] = (
                            pool.buffers[0][:, :3]
                            if alias == "cross-component"
                            else pool.buffers[0][:, 6:8]
                        )
                        store._layouts["kv"] = describe_pool("kv", pool)
                    store._runner = transport = _DeferredCopyTransport(store._layouts)
                slots = list(range(pages, 2 * pages))
                if alias == "slot":
                    slots[-1] = slots[0]
                before = [
                    ctypes.string_at(p.address, (p.count - 1) * p.stride + p.size)
                    for layout in store._layouts.values()
                    for p in layout.components
                ]
                with self.assertRaisesRegex(ValueError, "Duplicate|Overlapping"):
                    store.batch_get_v2(self._pipeline_transfers(store, keys, slots))
                self.assertEqual(transport.reads, [])
                self.assertEqual(transport.hints, {})
                after = [
                    ctypes.string_at(p.address, (p.count - 1) * p.stride + p.size)
                    for layout in store._layouts.values()
                    for p in layout.components
                ]
                self.assertEqual(before, after)

    def test_pipeline_controller_partial_prefix_keeps_logical_ack_count(self):
        pages = 1812
        for missing in (None, 600):
            with self.subTest(missing=missing):
                controller, operation, pools, transport = self._controller_with_indexer(
                    pages=pages, batch_pages=2048, pipeline_window=4
                )
                for pool in pools.values():
                    for part, buffer in enumerate(pool.buffers):
                        buffer[:pages].fill_(31 + part)
                controller.storage_backend.batch_set_v2(
                    [
                        PoolTransfer(
                            name,
                            keys=operation.hash_value,
                            host_indices=torch.arange(2 * pages),
                        )
                        for name in pools
                    ]
                )
                if missing is not None:
                    transport.storage.pop(
                        block_key(
                            operation.hash_value[missing],
                            controller.storage_backend._namespace,
                            "indexer",
                        )
                    )
                with (
                    self.assertLogs(level="WARNING")
                    if missing is not None
                    else ExitStack()
                ):
                    completed = controller._page_transfer(operation)
                prefix = pages if missing is None else missing
                self.assertEqual(completed, prefix)
                self.assertEqual(
                    controller.prefetch_sync_queue.get_nowait().completed_tokens,
                    prefix * 2,
                )
                self.assertTrue(controller.prefetch_sync_queue.empty())
                self.assertEqual(
                    transport.next_handle, 9
                )  # one deposit, eight read chunks
                for pool in pools.values():
                    for buffer in pool.buffers:
                        self.assertTrue(
                            torch.equal(buffer[:prefix], buffer[pages : pages + prefix])
                        )

    def _pipeline_mla_fixture(self, pages=2200):
        store, _, keys = self._pipeline_fixture(
            pages, sidecars=((PoolName.INDEXER, (5,), 1),)
        )
        store._share_replicated_mla = True
        store.config.is_mla_model, store.config.tp_rank = True, 1
        store._source_offset = 0
        store._owner_endpoint = "tcp://owner:32100"
        store._control_endpoint = "tcp://target:32101"
        store._namespace = storage_namespace(
            store.config, store._layouts, shared_mla=True
        )
        external = "tcp://external:33000"
        peers = {external: {}, store._owner_endpoint: {}}
        for name, layout in store._layouts.items():
            for page, key in enumerate(keys):
                identity = block_key(key, store._namespace, name)
                # Owner also holds different bytes for external successes, so
                # accidentally retrying them is observable in the final bytes.
                if page % 3 != 2:
                    peers[store._owner_endpoint][identity] = {
                        part.name: bytes([73 + component]) * part.size
                        for component, part in enumerate(layout.components)
                    }
                if page % 3 == 1:
                    peers[external][identity] = {
                        part.name: bytes([29 + component]) * part.size
                        for component, part in enumerate(layout.components)
                    }
        transport = store._runner = _DeferredCopyTransport(store._layouts, peers)
        extra = HiCacheStorageExtraInfo(
            extra_info={
                "kv_hints": {
                    "protocol_version": "0.1",
                    "actions": [
                        {
                            "action_type": "kv.fetch",
                            "action_version": "1.0",
                            "payload": {
                                "source_control_endpoint": external,
                                "block_hashes": [0],
                            },
                        }
                    ],
                }
            }
        )
        return store, transport, keys, extra

    def test_pipeline_mla_retries_only_failed_chunks_after_primary_terminal(self):
        pages = 2200
        store, transport, keys, extra = self._pipeline_mla_fixture(pages)
        store._metrics = SimpleNamespace(prefetch_pgs=[], prefetch_bandwidth=[])
        store._diagnostic_telemetry = True
        with self.assertLogs(
            "sglang.srt.mem_cache.storage.kvcr.kvcr_store", level="INFO"
        ) as captured:
            restored = store.batch_get_v2(
                self._pipeline_transfers(store, keys, range(pages, 2 * pages)), extra
            )
        expected = [page % 3 != 2 for page in range(pages)]
        self.assertEqual(restored, {"kv": expected, "indexer": expected})
        self.assertEqual(store._metrics.prefetch_pgs, [sum(expected)])
        self.assertEqual(transport.max_pending, 4)
        self.assertFalse(transport.pending)
        self.assertFalse(transport.hints)
        external_reads = [r for r in transport.reads if r[3] == "tcp://external:33000"]
        local_reads = [r for r in transport.reads if r[3] == store._owner_endpoint]
        self.assertEqual(len(external_reads), 9)
        self.assertEqual(len(local_reads), 9)
        self.assertEqual(len({r[2] for r in external_reads + local_reads}), 2)
        event_index = {event: i for i, event in enumerate(transport.events)}
        self.assertLess(
            max(event_index[("complete", r[0])] for r in external_reads),
            min(event_index[("submit", r[0])] for r in local_reads),
        )
        for source, identity, _ in transport.writes:
            page = int(identity.split(b"#", 1)[0], 16) - 1
            self.assertEqual(
                source,
                "tcp://external:33000" if page % 3 == 1 else store._owner_endpoint,
            )
        self.assertEqual(len(transport.writes), 2 * sum(expected))
        for layout in store._layouts.values():
            for component, part in enumerate(layout.components):
                for page in range(pages):
                    value = (
                        0
                        if page % 3 == 2
                        else (29 if page % 3 == 1 else 73) + component
                    )
                    self.assertEqual(
                        ctypes.string_at(
                            part.address + (pages + page) * part.stride, part.size
                        ),
                        bytes([value]) * part.size,
                    )
        records = [
            json.loads(line.split("KVCR L3 operation ", 1)[1])
            for line in captured.output
        ]
        primary = {r["operation"]: r for r in records if r["stage"] == "primary"}
        self.assertEqual([r["pipeline_chunk_index"] for r in records], list(range(18)))
        self.assertEqual(len({r["logical_start_epoch"] for r in records}), 1)
        for record in records:
            self.assertLessEqual(record["logical_start_epoch"], record["start_epoch"])
            self.assertLessEqual(record["start_epoch"], record["epoch"])
            self.assertGreaterEqual(record["submit_ms"], 0)
            self.assertGreaterEqual(record["wait_ms"], 0)
            if record["stage"] != "primary":
                parent = primary[record["parent_operation"]]
                self.assertEqual(record["requested_blocks"], len(parent["failed_keys"]))
                self.assertTrue(
                    set(record["failed_keys"]) <= set(parent["failed_keys"])
                )
                self.assertEqual(record["source"], store._owner_endpoint)
        before = len(transport.reads)
        self.assertEqual(
            store.batch_get_v1(keys, torch.arange(2 * pages, 4 * pages)),
            [page % 3 != 2 for page in range(pages)],
        )
        self.assertEqual(len(transport.reads) - before, 5)  # no fallback to own owner
        self.assertTrue(
            all(r[3] == store._owner_endpoint for r in transport.reads[before:])
        )

    def test_pipeline_fatal_entry_precedes_primary_or_fallback_hint_discard(self):
        """A Python mock unwinds later; only callback ENTRY proves ordering here."""
        for phase in ("primary", "fallback"):
            for failure in (
                "unknown-submit",
                "none-handle",
                "duplicate-handle",
                "wait",
                "bookkeeping",
            ):
                with self.subTest(phase=phase, failure=failure):
                    pages = 1100
                    store, transport, keys, extra = self._pipeline_mla_fixture(pages)
                    real_deliver, real_wait = transport.deliver, store._wait
                    events, selected = [], []

                    def is_selected(
                        request_id, transport=transport, store=store, phase=phase
                    ):
                        source = transport.hints[request_id]["actions"][0]["payload"][
                            "source_control_endpoint"
                        ]
                        return (source == store._owner_endpoint) == (
                            phase == "fallback"
                        )

                    def submit(
                        blocks,
                        request_id=None,
                        real_deliver=real_deliver,
                        is_selected=is_selected,
                        selected=selected,
                        failure=failure,
                    ):
                        handle = real_deliver(blocks, request_id)
                        if is_selected(request_id):
                            selected.append(handle)
                            if len(selected) == 2:
                                if failure == "unknown-submit":
                                    raise KeyboardInterrupt(
                                        "enqueued without returning handle"
                                    )
                                if failure == "none-handle":
                                    return None
                                if failure == "duplicate-handle":
                                    return selected[0]
                        return handle

                    def wait(
                        handle, selected=selected, failure=failure, real_wait=real_wait
                    ):
                        if selected and handle == selected[0] and failure == "wait":
                            # One completion may already exist before another poll fails.
                            raise GeneratorExit("lost poll state")
                        return real_wait(handle)

                    def clock(failure=failure, selected=selected):
                        if failure == "bookkeeping" and len(selected) == 2:
                            raise SystemExit("post-submit clock failed")
                        return 1.0

                    def fatal(code, events=events, transport=transport):
                        events.append("fatal-entry")
                        self.assertEqual(code, 1)
                        self.assertTrue(transport.pending)
                        self.assertTrue(transport.hints)
                        self.assertNotIn("discard", events)
                        self.assertNotIn("ack", events)
                        raise SystemExit(code)

                    def discard(request_id, events=events, transport=transport):
                        # Simulate the unwind caused by a Python-raised exit mock.
                        events.append("discard")
                        transport.hints.pop(request_id)

                    store._diagnostic_telemetry = failure == "bookkeeping"
                    with (
                        patch.object(transport, "deliver", submit),
                        patch.object(store, "_wait", wait),
                        patch.object(transport, "discard_hint", discard),
                        patch(
                            "sglang.srt.mem_cache.storage.kvcr.delivery.time.perf_counter",
                            clock,
                        ),
                        patch(
                            "sglang.srt.mem_cache.storage.kvcr.delivery.logger.critical",
                            side_effect=RuntimeError("logging failed"),
                        ),
                        patch(
                            "sglang.srt.mem_cache.storage.kvcr.delivery.os._exit", fatal
                        ),
                        self.assertRaises(SystemExit),
                    ):
                        store.batch_get_v2(
                            self._pipeline_transfers(
                                store, keys, range(pages, 2 * pages)
                            ),
                            extra,
                        )
                        events.append("ack")
                    self.assertEqual(events[0], "fatal-entry")

    def test_pipeline_result_and_diagnostic_exceptions_only_after_all_writes_terminal(
        self,
    ):
        for failure in ("result", "diagnostic"):
            with self.subTest(failure=failure):
                store, transport, keys = self._pipeline_fixture(2601)
                store.batch_set_v2(self._pipeline_transfers(store, keys, range(2601)))

                class ExplodingResult:
                    def __init__(inner, transport):
                        inner.transport = transport

                    @property
                    def success(inner):
                        self.assertFalse(inner.transport.pending)
                        raise ValueError("terminal result failure")

                if failure == "result":

                    def replace(_handle, _blocks, result, transport=transport):
                        return dict.fromkeys(result, ExplodingResult(transport))

                    transport.result_hook = replace
                store._diagnostic_telemetry = failure == "diagnostic"
                store._control_endpoint = "tcp://target:32100"

                def log(*_args, transport=transport, **_kwargs):
                    self.assertFalse(transport.pending)
                    raise ValueError("terminal diagnostic failure")

                with (
                    patch(
                        "sglang.srt.mem_cache.storage.kvcr.kvcr_store.logger.info", log
                    ),
                    patch(
                        "sglang.srt.mem_cache.storage.kvcr.delivery.os._exit"
                    ) as fatal,
                    self.assertRaisesRegex(ValueError, "terminal"),
                ):
                    store.batch_get_v2(
                        self._pipeline_transfers(store, keys, range(2601, 5202))
                    )
                fatal.assert_not_called()
                self.assertEqual(len(transport.reads), 6)
                self.assertFalse(transport.pending)

    def test_sparse_checkpoints_intersect_prefix_sets(self):
        """KV max=4 and checkpoint max=3 do not imply that boundary 2 is valid."""
        state = _HostPool(sizes=(5,), page_size=1)
        # Sidecar-specific pools (notably DSA indexer) expose page accessors,
        # not the KV anchor's kv_buffer attribute.
        del state.kv_buffer
        self.backend._runner = None
        self.backend.register_mem_host_pool_v2(state, PoolName.MAMBA)
        self.backend._namespace = storage_namespace(self.config, self.backend._layouts)
        self.backend._runner = self.transport = _CopyTransport(self.backend._layouts)
        self.backend.batch_set_v1(self.keys, torch.arange(8))
        self.backend.batch_set_v2(
            [
                PoolTransfer(
                    PoolName.MAMBA,
                    keys=[self.keys[0], self.keys[2]],
                    host_indices=torch.tensor([1, 3]),
                )
            ]
        )
        result = self.backend.batch_exists_v2(
            self.keys,
            [
                PoolTransfer(
                    PoolName.MAMBA,
                    keys=[self.keys[-1]],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ],
        )
        self.assertEqual(result.restorable_prefix_pages, [1, 3])
        self.assertEqual(result.kv_hit_pages, 3)

    def test_invalid_physical_pages_are_rejected_before_io(self):
        layout = describe_pool("kv", self.pool)
        for indices in ([0, 3], [1, 2], [-2, -1], [16, 17], [0]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                page_slots(layout, self.keys[:1], torch.tensor(indices))
        self.assertFalse(self.transport.storage)

    def test_completion_drain_preserves_another_callers_result(self):
        """One poll drain must not drop a concurrent backup/prefetch completion."""
        self.transport.completed = {11: {"one": True}, 12: {"two": True}}
        self.assertEqual(self.backend._wait(12), {"two": True})
        self.assertEqual(self.backend._wait(11), {"one": True})

    def test_hint_is_rank_aligned_and_discarded(self):
        info = HiCacheStorageExtraInfo(
            extra_info={
                "kv_hints": {
                    "protocol_version": "0.1",
                    "message_id": "r",
                    "actions": [
                        {
                            "action_id": "a",
                            "action_type": "kv.fetch",
                            "action_version": "1.0",
                            "payload": {
                                "source_control_endpoint": "tcp://source:32000",
                                "block_hashes": [(1 << 63) + 1],
                            },
                        }
                    ],
                }
            }
        )
        self.assertEqual(
            peer_hint(info, 3)["actions"][0]["payload"]["source_control_endpoint"],
            "tcp://source:32003",
        )
        with self.backend._hint(info) as request_id:
            self.assertEqual(len(self.transport.hints), 1)
            self.assertIn(request_id, self.transport.hints)
        self.assertEqual(self.transport.hints, {})
        info.extra_info["kv_hints"]["actions"][0]["payload"]["block_hashes"] = [True]
        self.assertIsNone(peer_hint(info, 0))

    def test_namespace_separates_model_format_and_shard_not_replica(self):
        namespace = storage_namespace(self.config, self.backend._layouts)
        self.config.dp_rank = 1
        self.assertEqual(
            storage_namespace(self.config, self.backend._layouts), namespace
        )
        for attribute, value in (("model_name", "other-model"), ("tp_rank", 1)):
            with self.subTest(attribute=attribute):
                config = SimpleNamespace(**vars(self.config))
                setattr(config, attribute, value)
                self.assertNotEqual(
                    storage_namespace(config, self.backend._layouts), namespace
                )

    def test_trailing_window_holes_do_not_become_contiguous_prefixes(self):
        self.assertEqual(resume_boundaries([False, True, True, False], trailing=2), [3])
        self.assertEqual(resume_boundaries([True, False, True], trailing=None), [1])


class TestKVCRDeliveryWindow(CustomTestCase):
    """Dependency-free boundary cases also runnable without loading Torch.

    The CPU suite imports normally. A minimal local gate can load these exact
    fixture/class definitions and the adapter class without its import graph.
    """

    def _fixture(self, pages=2601, *, diagnostic=False):
        buffers = [(ctypes.c_ubyte * (pages * size))() for size in (5, 3)]
        parts = tuple(
            SimpleNamespace(
                name=f"kv_{i}",
                address=ctypes.addressof(buffer),
                size=size,
                stride=size,
                count=pages,
            )
            for i, (buffer, size) in enumerate(zip(buffers, (5, 3)))
        )
        layouts = {"kv": SimpleNamespace(components=parts)}
        store = KVCRStore.__new__(KVCRStore)
        store._diagnostic_telemetry = diagnostic
        store._completion = threading.Condition(threading.RLock())
        store._completed = {}
        transport = store._runner = _DeferredCopyTransport(layouts)
        blocks = {
            f"page-{page}".encode(): [
                SimpleNamespace(label=part.name, element_index=page) for part in parts
            ]
            for page in range(pages)
        }
        for page, key in enumerate(blocks):
            transport.storage[key] = {
                part.name: bytes([(page + i * 17) % 251 + 1]) * part.size
                for i, part in enumerate(parts)
            }
        return store, transport, blocks, layouts, buffers

    def test_window_uses_actual_bytes_for_boundaries_short_tail_and_reverse_results(
        self,
    ):
        for count in (1, 511, 512, 513, 1812, 2048, 2601):
            with self.subTest(count=count):
                store, transport, blocks, layouts, buffers = self._fixture(count)
                transport.completed[999999] = {b"other": True}
                validate_destination_spans(blocks, layouts)
                completed = run_window(
                    store,
                    (Delivery(chunk) for chunk in block_chunks(blocks)),
                    None,
                    set(),
                )
                self.assertEqual(len(requested_results(completed)), count)
                self.assertEqual(transport.max_pending, min(4, (count + 511) // 512))
                self.assertEqual(
                    [len(d.blocks) for d in completed],
                    [512] * (count // 512) + ([count % 512] if count % 512 else []),
                )
                for i, buffer in enumerate(buffers):
                    size = layouts["kv"].components[i].size
                    expected = b"".join(
                        bytes([(page + i * 17) % 251 + 1]) * size
                        for page in range(count)
                    )
                    self.assertEqual(bytes(buffer), expected)
                self.assertEqual(store._wait(999999), {b"other": True})
                self.assertFalse(transport.pending)

    def test_overlap_preflight_covers_aliases_across_chunk_boundaries(self):
        for alias in ("slot", "component", "partial", "pool"):
            with self.subTest(alias=alias):
                _store, transport, blocks, layouts, buffers = self._fixture(513)
                refs = blocks[b"page-512"]
                first = layouts["kv"].components[0]
                if alias == "slot":
                    refs[0].element_index = 0
                else:
                    part = SimpleNamespace(**vars(first))
                    part.name = "sidecar"
                    part.address += 1 if alias == "partial" else 0
                    part.size = 2
                    layouts["sidecar"] = SimpleNamespace(components=(part,))
                    new_ref = SimpleNamespace(label=part.name, element_index=0)
                    if alias == "pool":
                        blocks[b"other-pool"] = [new_ref]
                    else:
                        refs.append(new_ref)
                with self.assertRaisesRegex(ValueError, "Overlapping"):
                    validate_destination_spans(blocks, layouts)
                self.assertEqual(transport.reads, [])
                self.assertTrue(all(not any(buffer) for buffer in buffers))

    def test_unrequested_successes_never_override_missing_or_false_own_results(self):
        store, transport, blocks, _, buffers = self._fixture(1100)
        transport.storage.pop(b"page-10")
        transport.storage.pop(b"page-600")

        def result_hook(_handle, chunk, result):
            result.pop(b"page-10", None)
            for key in (b"page-10", b"page-600"):
                if key not in chunk:
                    result[key] = SimpleNamespace(success=True)
            return result

        transport.result_hook = result_hook
        completed = run_window(
            store, (Delivery(c) for c in block_chunks(blocks)), None, set()
        )
        result = requested_results(completed)
        self.assertNotIn(b"page-10", result)
        self.assertFalse(result[b"page-600"].success)
        for buffer, size in zip(buffers, (5, 3)):
            self.assertEqual(bytes(buffer[10 * size : 11 * size]), bytes(size))
            self.assertEqual(bytes(buffer[600 * size : 601 * size]), bytes(size))
            self.assertTrue(any(buffer[601 * size : 602 * size]))
        self.assertFalse(transport.pending)

    def _assert_fatal(self, store, transport, blocks, action):
        """Assert ENTRY ordering, not production finally suppression by Python mocks."""
        hint = {
            "actions": [{"payload": {"source_control_endpoint": "tcp://source:32100"}}]
        }
        events = []

        def fatal(code):
            events.append("fatal")
            self.assertEqual(code, 1)
            self.assertTrue(transport.pending)
            self.assertTrue(transport.hints)
            self.assertNotIn("discard", events)
            self.assertNotIn("reuse", events)
            raise SystemExit(code)

        def discard(request_id):
            events.append("discard")
            transport.hints.pop(request_id)

        with (
            patch.object(store, "_operation_hint", return_value=hint),
            patch.object(transport, "discard_hint", discard),
            patch(
                "sglang.srt.mem_cache.storage.kvcr.delivery.logger.critical",
                side_effect=RuntimeError("logger failed"),
            ),
            patch("sglang.srt.mem_cache.storage.kvcr.delivery.os._exit", fatal),
            self.assertRaises(SystemExit),
        ):
            with store._hint(None, blocks) as request_id:
                action(request_id)
            events.append("reuse")
        self.assertEqual(events[0], "fatal")

    def test_fatal_entry_for_unknown_submit_and_invalid_or_duplicate_handles(self):
        for invalid in (None, True, 0, -1, "7", [], "duplicate", "unknown"):
            with self.subTest(invalid=invalid):
                store, transport, blocks, _, buffers = self._fixture(1100)
                real_deliver = transport.deliver
                handles = []

                def submit(
                    chunk,
                    request_id=None,
                    real_deliver=real_deliver,
                    handles=handles,
                    invalid=invalid,
                ):
                    handle = real_deliver(chunk, request_id)
                    handles.append(handle)
                    if len(handles) == 2:
                        if invalid == "unknown":
                            raise KeyboardInterrupt("enqueue then raise")
                        return handles[0] if invalid == "duplicate" else invalid
                    return handle

                with patch.object(transport, "deliver", submit):
                    self._assert_fatal(
                        store,
                        transport,
                        blocks,
                        lambda request, store=store, blocks=blocks: run_window(
                            store,
                            (Delivery(c) for c in block_chunks(blocks)),
                            request,
                            set(),
                        ),
                    )
                self.assertTrue(all(not any(buffer) for buffer in buffers))

    def test_fatal_entry_for_late_bookkeeping_and_iterator_failures(self):
        class BrokenSet(set):
            def add(self, _handle):
                raise MemoryError("lost handle while adding")

        for failure in ("seen", "clock", "iterator"):
            with self.subTest(failure=failure):
                store, transport, blocks, _, buffers = self._fixture(
                    1100, diagnostic=True
                )

                def deliveries(blocks=blocks, failure=failure):
                    for i, chunk in enumerate(block_chunks(blocks)):
                        if i == 2 and failure == "iterator":
                            raise GeneratorExit("late chunk preparation failed")
                        yield Delivery(chunk)

                def clock(failure=failure, transport=transport):
                    if failure == "clock" and transport.pending:
                        raise SystemExit("post-submit timestamp failed")
                    return 1.0

                with patch(
                    "sglang.srt.mem_cache.storage.kvcr.delivery.time.perf_counter",
                    clock,
                ):
                    self._assert_fatal(
                        store,
                        transport,
                        blocks,
                        lambda request, store=store, failure=failure: run_window(
                            store,
                            deliveries(),
                            request,
                            BrokenSet() if failure == "seen" else set(),
                        ),
                    )
                self.assertTrue(all(not any(buffer) for buffer in buffers))

    def test_fatal_entry_after_partial_completion_then_poll_failure(self):
        store, transport, blocks, _, buffers = self._fixture(2601)
        real_poll = transport.poll_completed
        polls = []

        def poll():
            polls.append(True)
            if len(polls) > 1:
                raise KeyboardInterrupt("poll failed with writes still pending")
            # Complete just the last handle; _wait must retain it while polling
            # for its requested first handle. Other writes remain outstanding.
            last = next(reversed(transport.pending))
            pending = transport.pending
            transport.pending = {last: pending.pop(last)}
            try:
                return real_poll()
            finally:
                transport.pending = pending

        with patch.object(transport, "poll_completed", poll):
            self._assert_fatal(
                store,
                transport,
                blocks,
                lambda request: run_window(
                    store, (Delivery(c) for c in block_chunks(blocks)), request, set()
                ),
            )
        self.assertEqual(len(transport.writes), 512)
        self.assertEqual(bytes(buffers[0][:5]), bytes(5))
        self.assertTrue(any(buffers[0][1536 * 5 : 2048 * 5]))

    def test_handle_reuse_after_completed_window_is_fatal(self):
        store, transport, blocks, _, _ = self._fixture(2601)
        real_deliver = transport.deliver
        handles = []

        def submit(chunk, request_id=None):
            handle = real_deliver(chunk, request_id)
            handles.append(handle)
            return handles[0] if len(handles) == 5 else handle

        with patch.object(transport, "deliver", submit):
            self._assert_fatal(
                store,
                transport,
                blocks,
                lambda request: run_window(
                    store, (Delivery(c) for c in block_chunks(blocks)), request, set()
                ),
            )
        self.assertEqual(len(transport.writes), 2048)

    def test_native_timestamps_bracket_submit_and_completion_only_when_enabled(self):
        for diagnostic in (False, True):
            with self.subTest(diagnostic=diagnostic):
                store, transport, blocks, _, _ = self._fixture(
                    1812, diagnostic=diagnostic
                )
                clock_events = []
                ticks = iter(range(100, 200))

                def epoch(ticks=ticks, clock_events=clock_events, transport=transport):
                    value = next(ticks)
                    clock_events.append(
                        (value, len(transport.reads), len(transport.writes))
                    )
                    return value

                with patch(
                    "sglang.srt.mem_cache.storage.kvcr.delivery.time.time", epoch
                ):
                    completed = run_window(
                        store, (Delivery(c) for c in block_chunks(blocks)), None, set()
                    )
                if diagnostic:
                    self.assertEqual(len(clock_events), 8)
                    self.assertEqual(
                        [event[1] for event in clock_events[:4]], [0, 1, 2, 3]
                    )
                    self.assertTrue(all(event[2] == 1812 for event in clock_events[4:]))
                    self.assertTrue(
                        all(d.prepared_epoch < d.finished_epoch for d in completed)
                    )
                else:
                    self.assertEqual(clock_events, [])
                    self.assertTrue(
                        all(
                            d.prepared_epoch is None and d.finished_epoch is None
                            for d in completed
                        )
                    )


class TestKVCROwnedNUMAProof(CustomTestCase):
    """Reject placement or registration claims that do not cover owned bytes."""

    def _mapping(self, *, shared=True):
        page = mmap.PAGESIZE
        start, end = 16 * page, 18 * page
        span = f"{start:x}-{end:x}"
        tail = (
            "rw-s 00000000 00:01 42 /dev/zero (deleted)"
            if shared
            else ("rw-p 00000000 00:00 0")
        )
        maps = f"{span} {tail}\n"
        numa = f"{start:x} bind:1 dirty=2 N1=2\n"
        kib = 2 * page // 1024
        smaps = maps + (
            f"Size: {kib} kB\nRss: {kib} kB\nAnonHugePages: 0 kB\n"
            f"KernelPageSize: {page // 1024} kB\nShmemPmdMapped: 0 kB\n"
        )
        return maps, numa, smaps, start, 2 * page

    def test_node_config_rejects_boolean_coercion_and_non_objects(self):
        for value in (True, False, -1, 65536, 1.0, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                node_option(value, "host_pool_numa_node")
        for raw in ("[]", "1", "null", [], "@"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                load_config(raw)
        self.assertEqual(load_config(None), {})
        self.assertEqual(
            load_config('{"host_pool_numa_node": 0}'), {"host_pool_numa_node": 0}
        )

    def test_fully_resident_owned_mapping_and_covered_subrange(self):
        for shared in (True, False):
            with self.subTest(shared=shared):
                maps, numa, smaps, address, length = self._mapping(shared=shared)
                proof = prove_range(maps, numa, smaps, address, length, 1, shared)
                self.assertEqual(proof["node_pages"], {1: 2})
                self.assertTrue(proof["all_covering_pages_resident_on_requested_node"])
                subrange = prove_range(
                    maps, numa, smaps, address, length // 2, 1, shared
                )
                self.assertEqual(subrange["covering_end"], address + length)
                self.assertEqual(subrange["length"], length // 2)

    def test_wrong_node_incomplete_residency_and_wrong_sharing_fail_closed(self):
        maps, numa, smaps, address, length = self._mapping()
        cases = [
            (maps, numa.replace("N1=2", "N0=2"), smaps, True),
            (maps, numa.replace("N1=2", "N1=1"), smaps, True),
            (maps, numa.replace("bind:1", "default"), smaps, True),
            (maps, numa, smaps.replace(f"Rss: {length // 1024} kB", "Rss: 0 kB"), True),
            (maps, numa, smaps, False),
        ]
        for m, n, s, shared in cases:
            with self.subTest(numa=n, shared=shared), self.assertRaises(RuntimeError):
                prove_range(m, n, s, address, length, 1, shared)
        with self.assertRaises(RuntimeError):
            prove_range(maps, numa, smaps, address, length + mmap.PAGESIZE, 1, True)

    def test_cuda_registration_requires_exact_contiguous_owned_coverage(self):
        maps, numa, smaps, address, length = self._mapping()
        proof = prove_range(maps, numa, smaps, address, length, 1, True)
        tensor = SimpleNamespace(
            data_ptr=lambda: address, numel=lambda: length, element_size=lambda: 1
        )
        setattr(tensor, HOST_PROOF_ATTR, proof)
        target = "sglang.srt.mem_cache.storage.kvcr.numa_pool.snapshot_range"
        with patch(target, return_value=proof):
            audit_registered_host(tensor, [(address, length)])
            for ranges in (
                [],
                [(address, length - 1)],
                [(address + 1, length)],
                [(address, length + 1)],
                [(address, length), (address, 1)],
            ):
                with self.subTest(ranges=ranges), self.assertRaises(RuntimeError):
                    audit_registered_host(tensor, ranges)
        with patch(target, return_value=dict(proof, node_pages={0: 2})):
            with self.assertRaisesRegex(RuntimeError, "changed during CUDA"):
                audit_registered_host(tensor, [(address, length)])


if __name__ == "__main__":
    unittest.main()
