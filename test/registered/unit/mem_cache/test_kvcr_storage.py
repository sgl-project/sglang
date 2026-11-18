"""CPU contracts for indexed HiCache IO; the native NIXL transport is faked."""

import copy
import ctypes
import dataclasses
import mmap
import sys
import types
import unittest
from array import array
from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch

from sglang.srt.disaggregation.kv_events import namespaced_block_hash
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.managers.kv_hints import decode_kv_hints_envelope
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.events import KVCacheEventRecorder
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    HiCacheStorageFailureInjector,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    StorageOperation,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.kvcr.endpoints import control_address
from sglang.srt.mem_cache.storage.kvcr.hints import peer_hint
from sglang.srt.mem_cache.storage.kvcr.host_layout import (
    block_key,
    describe_pool,
    page_slots,
    resume_boundaries,
    storage_namespace,
)
from sglang.srt.mem_cache.storage.kvcr.kvcr_store import (
    KVCRStore,
    _allocate_dram_pool,
    _bool_option,
)
from sglang.srt.mem_cache.storage_key_binding import storage_key_to_event_hashes
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.utils import (
    get_event_hash_str,
    get_hash_str,
    get_storage_hash_str,
    hash_str_to_int64,
)
from sglang.srt.runtime_context import get_parallel
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
        # KVCR's public API requires typed envelopes and typed actions.
        if not isinstance(hint, msgspec.Struct) or not all(
            isinstance(action, msgspec.Struct) for action in hint.actions
        ):
            raise ValueError("invalid router hint")
        self.hints[request_id] = msgspec.to_builtins(hint)

    def discard_hint(self, request_id):
        del self.hints[request_id]


class _StrictPeerTransport(_CopyTransport):
    """Native-style hint membership at the external dependency boundary."""

    def __init__(self, layouts, peers):
        super().__init__(layouts)
        self.peers, self.reads = peers, []

    def _authorized(self, request_id):
        hint = self.hints.get(request_id)
        if hint is None:
            return {}
        payload = hint["actions"][0]["payload"]
        allowed = set(payload["block_hashes"])
        return {
            key: data
            for key, data in self.peers.get(
                payload["source_control_endpoint"], {}
            ).items()
            if int(key[:16], 16) in allowed
        }

    def query(self, keys, request_id=None):
        available = self._authorized(request_id)
        return [
            (_QueryStatus.FETCHABLE if key in available else _QueryStatus.MISS, None)
            for key in keys
        ]

    def deliver(self, blocks, request_id=None):
        self.reads.append(tuple(blocks))
        prior = self.storage
        try:
            self.storage = {**self._authorized(request_id), **prior}
            return super().deliver(blocks, request_id)
        finally:
            self.storage = prior


def _hint_gate(backend, recorder, page_size=2):
    """The UnifiedRadixCache state its kv_hints gate reads, and nothing else."""
    cache = object.__new__(UnifiedRadixCache)
    cache.tree_core = SimpleNamespace(
        enable_storage=True, page_size=page_size, kv_events=recorder
    )
    cache.cache_controller = SimpleNamespace(storage_backend=backend)
    return cache


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
        # A standalone replica: no DP controller numbered this process.
        self.enterContext(get_parallel().override(dp_rank=None))
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

    def test_rank_local_backup_preserves_upstream_writer_selection(self):
        controller = object.__new__(HiCacheController)
        controller.storage_config = SimpleNamespace(is_storage_writer=False)
        controller.storage_backend = SimpleNamespace(requires_rank_local_backup=False)
        self.assertTrue(controller.backup_skip)
        controller.storage_backend.requires_rank_local_backup = True
        self.assertFalse(controller.backup_skip)
        controller.storage_backend.requires_rank_local_backup = False
        controller.storage_config.is_storage_writer = True
        self.assertFalse(controller.backup_skip)

    def test_dcp_storage_fails_closed(self):
        config = copy.copy(self.config)
        config.dcp_size = 2
        with self.assertRaisesRegex(ValueError, "does not support DCP"):
            KVCRStore(config, self.pool)

    def test_replica_dp_storage_config_disambiguates_hint_sources(self):
        """Independent DP replicas must not advertise the same source address."""
        # Without attention DP, HiCache hands every replica dp_rank=0.
        config = dataclasses.replace(
            self.config, extra_config={"control_advertise_host": "source.example"}
        )
        backends = []
        for rank in (0, 1):
            with get_parallel().override(dp_rank=rank):
                backends.append(KVCRStore(config, self.pool))
                self.addCleanup(backends[-1].close)
        self.assertEqual([b.config.dp_rank for b in backends], [0, 1])
        self.assertEqual(
            [control_address(b.config, b._extra) for b in backends],
            [("source.example", 32100), ("source.example", 32102)],
        )

    def test_cp_shards_without_attention_dp_use_dense_offsets(self):
        """TP4/CP2 source endpoints must cover exactly four shards."""
        backends = []
        for rank in range(4):
            cp_rank, attn_tp_rank = divmod(rank, 2)
            # Without attention DP, HiCache numbers TP across the CP group.
            config = dataclasses.replace(
                self.config,
                tp_rank=rank,
                tp_size=4,
                attn_cp_rank=cp_rank,
                attn_cp_size=2,
                extra_config={"control_advertise_host": "source.example"},
            )
            with get_parallel().override(attn_tp_rank=attn_tp_rank, attn_tp_size=2):
                backends.append(KVCRStore(config, self.pool))
                self.addCleanup(backends[-1].close)
        self.assertEqual([b._offset for b in backends], [0, 1, 2, 3])
        self.assertEqual(
            [control_address(b.config, b._extra)[1] for b in backends],
            [32100, 32101, 32102, 32103],
        )

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

    def test_cache_salt_isolates_v2_kv_indexer_and_mamba_bytes(self):
        """Removing the namespace seed must expose a cross-tenant L3 hit here."""
        self.backend = KVCRStore(self.config, self.pool)
        pools = {PoolName.KV: self.pool}
        for name, sizes in ((PoolName.INDEXER, (5,)), (PoolName.MAMBA, (9, 3))):
            pools[name] = _HostPool(sizes=sizes)
            self.backend.register_mem_host_pool_v2(pools[name], name)
        self.backend._namespace = storage_namespace(self.config, self.backend._layouts)
        self.transport = self.backend._runner = _CopyTransport(self.backend._layouts)

        def keys(salt):
            return get_storage_hash_str(
                RadixKey(array("q", range(1, 9)), cache_salt=salt), page_size=2
            )

        def transfers(hashes, destination=False):
            return [
                PoolTransfer(
                    name,
                    host_indices=(
                        torch.arange(2, 4) if destination else torch.arange(2)
                    )
                    if name == PoolName.MAMBA
                    else (torch.arange(8, 16) if destination else torch.arange(8)),
                    keys=hashes[-1:] if name == PoolName.MAMBA else hashes,
                    hit_policy=PoolHitPolicy.TRAILING_PAGES
                    if name == PoolName.MAMBA
                    else PoolHitPolicy.ALL_PAGES,
                )
                for name in pools
            ]

        a, b = keys("tenant-a"), keys("tenant-b")
        self.assertTrue(set(a).isdisjoint(b))
        for number, pool in enumerate(pools.values()):
            for component, buffer in enumerate(pool.buffers):
                for page in range(1 if list(pools)[number] == PoolName.MAMBA else 4):
                    buffer[page].fill_(31 + 7 * number + component + page)
        written = self.backend.batch_set_v2(transfers(a))
        self.assertTrue(all(all(flags) for flags in written.values()))
        positive = self.backend.batch_exists_v2(a, transfers(a)[1:])
        self.assertEqual(positive.kv_hit_pages, 4)
        self.assertEqual(positive.restorable_prefix_pages, [4])
        negative = self.backend.batch_exists_v2(b, transfers(b)[1:])
        self.assertEqual(negative.kv_hit_pages, 0)
        self.assertFalse(negative.restorable_prefix_pages)
        missed = self.backend.batch_get_v2(transfers(b, destination=True))
        self.assertFalse(any(any(flags) for flags in missed.values()))
        for name, pool in pools.items():
            for buffer in pool.buffers:
                self.assertFalse(
                    torch.count_nonzero(
                        buffer[1:2] if name == PoolName.MAMBA else buffer[4:8]
                    )
                )
        restored = self.backend.batch_get_v2(transfers(a, destination=True))
        self.assertTrue(all(all(flags) for flags in restored.values()))
        for name, pool in pools.items():
            for buffer in pool.buffers:
                self.assertTrue(
                    torch.equal(buffer[0:1], buffer[1:2])
                    if name == PoolName.MAMBA
                    else torch.equal(buffer[:4], buffer[4:8])
                )

    def test_storage_controller_keeps_salt_in_lookup_and_prefix_continuation(self):
        """List-converting a RadixKey at the storage boundary must fail closed."""
        a = RadixKey(array("q", range(1, 9)), cache_salt="tenant-a")
        b = RadixKey(array("q", range(1, 9)), cache_salt="tenant-b")
        hashes = get_storage_hash_str(a, page_size=2)
        self.assertEqual(self.backend.batch_set_v1(hashes, torch.arange(8)), [True] * 4)
        for controller_type in (HiCacheController, HybridCacheController):
            controller = controller_type.__new__(controller_type)
            controller.page_size, controller.storage_backend = 2, self.backend
            for key, expected in ((a, 8), (b, 0)):
                operation = StorageOperation(None, key)
                operation.assume_stored = False
                self.assertEqual(controller._storage_hit_query(operation)[1], expected)
            head = get_storage_hash_str(a[:4], page_size=2)
            tail = StorageOperation(None, a[4:], last_hash=head[-1], prefix_keys=head)
            tail.assume_stored = False
            found, count = controller._storage_hit_query(tail)
            self.assertEqual(found, hashes[2:])
            self.assertEqual(count, 4)

    def test_storage_salt_namespaces_preserve_legacy_and_avoid_join_collisions(self):
        """Tenant/adapter tuple framing must not collapse distinct identities."""
        tokens = array("q", range(1, 9))
        plain = get_storage_hash_str(RadixKey(tokens), page_size=2)
        self.assertEqual(plain, get_hash_str(tokens, page_size=2))
        self.assertEqual(
            plain, get_storage_hash_str(RadixKey(tokens, cache_salt=""), page_size=2)
        )
        identities = (
            (None, "tenant-a"),
            (None, "tenant-b"),
            (None, "租户-α"),
            ("a", "bc"),
            ("ab", "c"),
            ("adapter-a", "tenant-a"),
        )
        chains = [
            get_storage_hash_str(
                RadixKey(tokens, extra_key=extra, cache_salt=salt), page_size=2
            )
            for extra, salt in identities
        ]
        for page in range(len(plain)):
            self.assertEqual(
                len({plain[page], *(chain[page] for chain in chains)}), 1 + len(chains)
            )

    def test_peer_hint_translates_only_bound_requested_pages(self):
        key = RadixKey(array("q", range(8)), cache_salt="tenant-a")
        mapping = storage_key_to_event_hashes(key, 2, None, None, namespace_seed=None)
        pages = list(mapping)
        hint = {
            "protocol_version": "0.1",
            "actions": [
                {
                    "action_type": "kv.fetch",
                    "action_version": "1.0",
                    "payload": {
                        "source_control_endpoint": "tcp://peer:32100",
                        "block_hashes": list(mapping.values())[:2],
                    },
                }
            ],
        }
        saved = copy.deepcopy(hint)
        info = HiCacheStorageExtraInfo(
            extra_info={"kv_hints": hint, "storage_key_to_event_hash": mapping}
        )
        physical = [block_key(h, "0" * 64, "kv") for h in pages]
        translated = peer_hint(info, 1, physical)
        payload = translated["actions"][0]["payload"]
        self.assertEqual(payload["source_control_endpoint"], "tcp://peer:32101")
        self.assertEqual(payload["block_hashes"], [int(h[:16], 16) for h in pages[:2]])
        self.assertEqual(hint, saved)
        other = get_storage_hash_str(
            RadixKey(array("q", range(8)), cache_salt="tenant-b"), page_size=2
        )
        self.assertIsNone(
            peer_hint(info, 0, [block_key(h, "0" * 64, "kv") for h in other])
        )
        for invalid in ({}, [], {pages[0]: True}, {pages[0]: -1}):
            info.extra_info["storage_key_to_event_hash"] = invalid
            self.assertIsNone(peer_hint(info, 0, physical))

    def test_salted_peer_prefetch_restores_kv_and_indexer_bytes(self):
        """Separate owners, strict advertised membership, actual continuation bytes."""
        writer = KVCRStore(self.config, self.pool)
        source_pools = {PoolName.KV: self.pool}
        for name, sizes in ((PoolName.INDEXER, (5,)), (PoolName.MAMBA, (9, 3))):
            source_pools[name] = _HostPool(sizes=sizes)
            writer.register_mem_host_pool_v2(source_pools[name], name)
        writer._namespace = storage_namespace(self.config, writer._layouts)
        source = writer._runner = _CopyTransport(writer._layouts)
        key = RadixKey(array("q", range(8)), cache_salt="tenant-a")
        hashes = get_storage_hash_str(key, page_size=2)

        def transfers(pools, hashes=hashes):
            return [
                PoolTransfer(
                    name,
                    host_indices=torch.arange(2 if name == PoolName.MAMBA else 8),
                    keys=hashes[-1:] if name == PoolName.MAMBA else hashes,
                    hit_policy=PoolHitPolicy.TRAILING_PAGES
                    if name == PoolName.MAMBA
                    else PoolHitPolicy.ALL_PAGES,
                    indices_from_pool=PoolName.KV if name == PoolName.INDEXER else None,
                )
                for name in pools
            ]

        for number, pool in enumerate(source_pools.values()):
            for component, buffer in enumerate(pool.buffers):
                for page in range(1 if pool is source_pools[PoolName.MAMBA] else 4):
                    buffer[page].fill_(31 + 7 * number + component + page)
        written = writer.batch_set_v2(transfers(source_pools))
        self.assertTrue(all(all(flags) for flags in written.values()))
        destination_pools = {
            name: _HostPool(sizes=tuple(b.shape[1] for b in pool.buffers))
            for name, pool in source_pools.items()
        }
        reader = KVCRStore(
            dataclasses.replace(self.config, extra_config={"enable_remote_hint": True}),
            destination_pools[PoolName.KV],
        )
        for name, pool in destination_pools.items():
            if name != PoolName.KV:
                reader.register_mem_host_pool_v2(pool, name)
        reader._namespace = storage_namespace(self.config, reader._layouts)
        transport = reader._runner = _StrictPeerTransport(
            reader._layouts, {"tcp://peer:32100": source.storage}
        )
        c = HybridCacheController.__new__(HybridCacheController)
        c.page_size, c.storage_backend = 2, reader
        c.page_get_func = c._page_get_zero_copy
        c.pp_prefetch_command_group = None
        c.prefetch_queue, c.prefetch_sync_queue = Queue(), Queue()
        c.storage_stop_event = Event()
        # The source publishes event IDs namespaced by the request's salt.
        recorder = KVCacheEventRecorder(enabled=False, page_size=2, dynamo_format=True)
        seed = recorder.namespace_seed(None, "tenant-a")
        published = [
            namespaced_block_hash(hash_str_to_int64(h), namespace_seed=seed)
            & ((1 << 64) - 1)
            for h in get_event_hash_str(key, page_size=2)
        ]
        hint = decode_kv_hints_envelope(
            {
                "protocol_version": "0.1",
                "message_id": "m",
                "actions": [
                    {
                        "action_id": "a",
                        "action_type": "kv.fetch",
                        "action_version": "1.0",
                        "payload": {
                            "source_control_endpoint": "tcp://peer:32100",
                            "block_hashes": published,
                        },
                    }
                ],
            }
        )

        # Build read metadata through the scheduler-side hint gate.
        gate = _hint_gate(reader, recorder)

        def submit(rid, radix_key, pool_transfers):
            metadata = gate._storage_read_metadata(hint, radix_key, [], None)
            self.assertIn("storage_key_to_event_hash", metadata)
            operation = c.submit_prefetch(
                CacheRequestHandle(rid, 0),
                radix_key,
                None,
                None,
                [],
                pool_transfers,
                storage_read_metadata=metadata,
            ).operation
            self.assertIs(c.prefetch_queue.get_nowait(), operation)
            return operation

        operation = submit("peer", key, transfers(destination_pools)[1:2])
        operation.hash_value, count = c._storage_hit_query(operation)
        self.assertEqual(count, 8)
        operation.host_indices = torch.arange(8)
        c._page_transfer_with_stable_layout(operation)
        self.assertEqual(c.prefetch_sync_queue.get_nowait().completed_tokens, 8)
        for name in (PoolName.KV, PoolName.INDEXER):
            for expected, actual in zip(
                source_pools[name].buffers, destination_pools[name].buffers
            ):
                self.assertTrue(torch.equal(expected[:4], actual[:4]))
        self.assertFalse(transport.storage)  # All bytes came from the other owner.
        self.assertFalse(transport.hints)  # No request authorization leaked.

        # A hinted page does not prove that the peer holds the Mamba checkpoint.
        hybrid = submit("hybrid", key, transfers(destination_pools)[1:])
        self.assertEqual(c._storage_hit_query(hybrid)[1], 0)
        self.assertEqual(
            reader.batch_exists_v2(
                hashes,
                transfers(destination_pools)[1:],
                HiCacheStorageExtraInfo(extra_info=hybrid.storage_read_metadata),
            ).extra_pool_hit_pages,
            {"kv": 4, "indexer": 4, "mamba": 0},
        )

        other = RadixKey(array("q", range(8)), cache_salt="tenant-b")
        denied = submit("other", other, None)
        self.assertEqual(c._storage_hit_query(denied)[1], 0)
        before = {
            name: [buffer.clone() for buffer in pool.buffers]
            for name, pool in destination_pools.items()
        }
        foreign = transfers(destination_pools, get_storage_hash_str(other, page_size=2))
        missed = reader.batch_get_v2(
            foreign,
            extra_info=HiCacheStorageExtraInfo(extra_info=denied.storage_read_metadata),
        )
        self.assertFalse(any(any(flags) for flags in missed.values()))
        for name, pool in destination_pools.items():
            for prior, actual in zip(before[name], pool.buffers):
                self.assertTrue(torch.equal(prior, actual))

    def test_invalid_startup_budget_is_rejected_before_registration(self):
        """An invalid optional deadline must not reach native registration."""
        for budget in (0, -1, True, 1.5, "30000", None):
            with self.subTest(budget=budget):
                self.config.extra_config = {"startup_timeout_seconds": budget}
                with self.assertRaisesRegex(ValueError, "startup_timeout_seconds"):
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
            for budget in (None, 180):
                with self.subTest(error_type=error_type, budget=budget):
                    self.config.extra_config = {"local_dram_bytes": 128}
                    if budget is not None:
                        self.config.extra_config["startup_timeout_seconds"] = budget
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
                            KVCR=construct,
                            KVCRBindings=SimpleNamespace,
                            ZmqPeerControlChannel=lambda **kwargs: SimpleNamespace(
                                endpoint="tcp://127.0.0.1:32100", close=lambda: None
                            ),
                        ),
                        "kvcr.config": SimpleNamespace(
                            KVCRBackendConfigs=SimpleNamespace,
                            KVCRConfig=SimpleNamespace,
                            LocalDramOptions=SimpleNamespace,
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
                    finally:
                        for allocation in mappings:
                            allocation.close()

    @unittest.skipUnless(sys.platform == "linux", "Linux smaps policy")
    def test_private_dram_pool_excludes_thp_without_disabling_worker_thp(self):
        """A plain anonymous mmap can be THP-eligible and stall source writes.

        The KVCR pool alone must request ordinary pages; process-wide THP
        remains unchanged for model allocations.
        """
        if not (
            hasattr(mmap, "MAP_PRIVATE")
            and hasattr(mmap, "MAP_ANONYMOUS")
            and hasattr(mmap, "MADV_NOHUGEPAGE")
        ):
            self.skipTest("Pool-local MADV_NOHUGEPAGE is unavailable")
        before = Path("/proc/self/status").read_text()
        allocation = _allocate_dram_pool(2 << 20)
        try:
            address = ctypes.addressof(ctypes.c_char.from_buffer(allocation))
            mapping = []
            inside = False
            for line in Path("/proc/self/smaps").read_text().splitlines():
                if "-" in line.split(" ", 1)[0]:
                    bounds = line.split(" ", 1)[0]
                    start, end = (int(value, 16) for value in bounds.split("-"))
                    if inside:
                        break
                    inside = start <= address < end
                if inside:
                    mapping.append(line)
            self.assertTrue(mapping, "KVCR pool VMA was not found")
            self.assertIn("rw-p", mapping[0])
            flags = next(line for line in mapping if line.startswith("VmFlags:"))
            self.assertIn("nh", flags.split())
            self.assertNotIn("hg", flags.split())
            eligible = next(line for line in mapping if line.startswith("THPeligible:"))
            self.assertEqual(eligible.split()[1], "0")
            current_thp = (
                Path("/proc/self/status")
                .read_text()
                .split("THP_enabled:")[1]
                .split()[0]
            )
            previous_thp = before.split("THP_enabled:")[1].split()[0]
            self.assertEqual(current_thp, previous_thp)
        finally:
            allocation.close()

    def test_boolean_options_are_strict_and_default_to_false(self):
        name = "enable_remote_hint"
        self.assertIs(_bool_option({}, name), False)
        self.assertIs(_bool_option({name: True}, name), True)
        self.assertIs(_bool_option({name: False}, name), False)
        for value in ("false", 1, None):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, f"{name} must be a boolean"),
            ):
                KVCRStore(
                    dataclasses.replace(self.config, extra_config={name: value}),
                    self.pool,
                )

    def test_remote_hints_are_opt_in_through_failure_injection(self):
        """Without consumes_kv_hints, HiCache never forwards request hints."""
        hinted = KVCRStore(
            dataclasses.replace(self.config, extra_config={"enable_remote_hint": True}),
            self.pool,
        )
        for backend, expected in ((self.backend, False), (hinted, True)):
            wrapped = HiCacheStorageFailureInjector(backend, 0.0, 0.0)
            self.assertIs(backend.consumes_kv_hints(), expected)
            self.assertIs(wrapped.consumes_kv_hints(), expected)
            # MLA elects one writer; a rank-local store needs every rank's copy.
            self.assertIs(wrapped.requires_rank_local_backup, True)
            controller = object.__new__(HiCacheController)
            controller.storage_config = SimpleNamespace(is_storage_writer=False)
            controller.storage_backend = wrapped
            self.assertFalse(controller.backup_skip)

    def test_unified_radix_cache_forwards_hints_only_when_enabled(self):
        """The real URC gate binds hints only for a backend that reads them."""
        hinted = KVCRStore(
            dataclasses.replace(self.config, extra_config={"enable_remote_hint": True}),
            self.pool,
        )
        recorder = KVCacheEventRecorder(enabled=False, page_size=2, dynamo_format=True)
        hint = decode_kv_hints_envelope(
            {
                "protocol_version": "0.1",
                "message_id": "m",
                "actions": [
                    {
                        "action_id": "a",
                        "action_type": "kv.fetch",
                        "action_version": "1.0",
                        "payload": {
                            "source_control_endpoint": "tcp://peer:32100",
                            "block_hashes": [1],
                        },
                    }
                ],
            }
        )
        salted = RadixKey(array("q", range(4)), cache_salt="tenant-a")
        plain = RadixKey(array("q", range(4)))
        for backend, enabled in ((self.backend, False), (hinted, True)):
            for wrap in (False, True):
                if wrap:
                    backend = HiCacheStorageFailureInjector(backend, 0.0, 0.0)
                gate = _hint_gate(backend, recorder)
                with self.subTest(enabled=enabled, wrapped=wrap):
                    self.assertIs(gate.forwards_kv_hints(), enabled)
                    metadata = gate._storage_read_metadata(hint, salted, [], None)
                    if not enabled:
                        self.assertIsNone(metadata)
                        continue
                    self.assertIs(metadata["kv_hints"], hint)
                    self.assertEqual(
                        metadata["storage_key_to_event_hash"],
                        storage_key_to_event_hashes(
                            salted,
                            2,
                            [],
                            None,
                            namespace_seed=recorder.namespace_seed(None, "tenant-a"),
                        ),
                    )
                    # Unsalted keys keep router hashes; no binding is built.
                    self.assertEqual(
                        gate._storage_read_metadata(hint, plain, [], None),
                        {"kv_hints": hint},
                    )
                    self.assertIsNone(
                        gate._storage_read_metadata(None, salted, [], None)
                    )

    def test_fetchable_trailing_state_is_not_a_resume_boundary(self):
        """A router hint names logical pages, not the peer's Mamba checkpoints."""
        self.backend._runner = None
        self.backend.register_mem_host_pool_v2(_HostPool(sizes=(5,)), PoolName.MAMBA)
        self.backend._namespace = storage_namespace(self.config, self.backend._layouts)
        self.backend._runner = SimpleNamespace(
            query=lambda keys, request_id=None: (
                [(_QueryStatus.FETCHABLE, None)] * len(keys)
            )
        )
        self.assertEqual(self.backend.batch_exists(self.keys), 4)
        mamba = PoolTransfer(
            PoolName.MAMBA,
            keys=self.keys[-1:],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        result = self.backend.batch_exists_v2(self.keys, [mamba])
        self.assertEqual(result.extra_pool_hit_pages, {"kv": 4, "mamba": 0})
        self.assertEqual(result.kv_hit_pages, 0)
        self.assertFalse(result.restorable_prefix_pages)

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
        # The default backend ignores hints even when a caller passes one.
        with self.backend._hint(info) as request_id:
            self.assertIsNone(request_id)
            self.assertEqual(self.transport.hints, {})
        hinted = KVCRStore(
            dataclasses.replace(self.config, extra_config={"enable_remote_hint": True}),
            self.pool,
        )
        transport = hinted._runner = _CopyTransport(hinted._layouts)
        with hinted._hint(info) as request_id:
            self.assertEqual(len(transport.hints), 1)
            self.assertIn(request_id, transport.hints)
        self.assertEqual(transport.hints, {})
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


if __name__ == "__main__":
    unittest.main()
