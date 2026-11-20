"""CPU contracts for indexed HiCache IO; the native NIXL transport is faked."""

import atexit
import copy
import ctypes
import dataclasses
import functools
import hashlib
import json
import mmap
import re
import sys
import time
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
import zmq

from sglang.srt.disaggregation.kv_events import (
    BlockRemoved,
    BlockStored,
    KVEventBatch,
    StorageMedium,
    namespaced_block_hash,
)
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.managers.kv_hints import decode_kv_hints_envelope
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, CacheRequestHandle
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
from sglang.srt.mem_cache.storage.kvcr.endpoints import (
    control_address,
    require_routable_host,
)
from sglang.srt.mem_cache.storage.kvcr.hints import peer_hint
from sglang.srt.mem_cache.storage.kvcr.host_layout import (
    block_key,
    describe_pool,
    page_slots,
    resume_boundaries,
    storage_namespace,
)
from sglang.srt.mem_cache.storage.kvcr.inventory import KVCRInventory
from sglang.srt.mem_cache.storage.kvcr.kvcr_store import (
    KVCRStore,
    _allocate_dram_pool,
    _bool_option,
)
from sglang.srt.mem_cache.storage_events import StorageEventReducer
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
    cache.cache_controller = SimpleNamespace(
        storage_backend=backend, storage_config=SimpleNamespace(dcp_size=1)
    )
    cache.pp_size = 1
    cache._init_storage_kv_events()
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
        # Discovery reports each replica once, under its own DP rank.
        for backend in backends:
            backend._extra["enable_remote_hint"] = backend._remote_hint = True
        self.assertEqual(
            [b.get_kv_hint_source() for b in backends],
            [
                {
                    "dp_rank": rank,
                    "capabilities": ["router_hint"],
                    "source_control_endpoint": f"tcp://source.example:{port}",
                }
                for rank, port in ((0, 32100), (1, 32102))
            ],
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
        # The same dense shards gate inventory: all four must hold a page.
        self.assertEqual([b.kv_event_shard for b in backends], [0, 1, 2, 3])
        self.assertEqual([b.kv_event_shard_count for b in backends], [4] * 4)
        batches = {backend.kv_event_shard: [] for backend in backends}
        self.assertEqual(StorageEventReducer(4).update(batches), [])

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


# KVCR's cache owner: <ns[:32]>:<ns[32:]>/<fixed cluster>/<process uuid>.
_CACHE_OWNER = re.compile(r"^([0-9a-f]{32}:[0-9a-f]{32})/0{16}/[0-9a-f]{32}$")


def _owned_store(block=17, parent=None, ownership="kvcr"):
    return BlockStored(
        block_hashes=[block],
        parent_block_hash=parent,
        token_ids=[1, 2],
        block_size=2,
        lora_id=None,
        medium=StorageMedium.CPU,
        ownership=ownership,
    )


def _owned_removal(block=17):
    return BlockRemoved(
        block_hashes=[block], medium=StorageMedium.CPU, ownership="kvcr"
    )


def _residency(keys, removed=False):
    """A native KVCR InventoryEvent for the local DRAM tier."""
    return SimpleNamespace(keys=tuple(keys), removed=removed)


class TestKVCRInventory(CustomTestCase):
    """KVCR's per-shard join of native residency and framework identities."""

    def _inventory(self, *pools):
        inventory = KVCRInventory()
        inventory.configure("ns", pools)
        return inventory

    def test_native_residency_not_submission_drives_events(self):
        inventory = self._inventory("kv", "index")
        keys = [b"page#ns#kv", b"page#ns#index"]
        framework = _owned_store(ownership=None)
        held = inventory.register({"page": framework}, keys)
        self.assertEqual(inventory.take(), [])
        inventory.update(_residency(keys[:1]))
        self.assertEqual(inventory.take(), [])
        inventory.update(_residency(keys[1:]))
        self.assertEqual(inventory.take(), [_owned_store()])
        self.assertIsNone(framework.ownership)
        inventory.release(held)
        self.assertEqual(inventory.take(), [])
        inventory.update(_residency(keys[:1], removed=True))
        self.assertEqual(inventory.take(), [_owned_removal()])
        inventory.update(_residency(keys[1:], removed=True))
        self.assertEqual(inventory._metadata, {})

    def test_trailing_pools_and_other_namespaces_do_not_gate_a_page(self):
        inventory = self._inventory("kv")
        held = inventory.register(
            {"page": _owned_store()}, [b"page#ns#kv", b"page#ns#swa", b"page#other#kv"]
        )
        inventory.update(_residency([b"page#other#kv", b"page#ns#swa", b"junk"]))
        self.assertEqual(inventory.take(), [])
        inventory.update(_residency([b"page#ns#kv"]))
        self.assertEqual(inventory.take(), [_owned_store()])
        inventory.release(held)
        inventory.update(_residency([b"page#ns#swa"], removed=True))
        self.assertEqual(inventory.take(), [])

    def test_failed_io_does_not_retain_metadata_or_advertise(self):
        inventory = self._inventory("kv")
        held = inventory.register({"page": _owned_store()}, [b"page#ns#kv"])
        inventory.release(held)
        self.assertEqual(inventory.take(), [])
        self.assertEqual(inventory._metadata, {})
        # Residency without an identity is never advertised.
        inventory.update(_residency([b"page#ns#kv"]))
        self.assertEqual(inventory.take(), [])

    def test_published_page_is_removed_under_its_stored_identity(self):
        inventory = self._inventory("kv")
        inventory.register({"page": _owned_store(17)}, [b"page#ns#kv"])
        inventory.update(_residency([b"page#ns#kv"]))
        inventory.register({"page": _owned_store(18)}, [b"page#ns#kv"])
        inventory.update(_residency([b"page#ns#kv"], removed=True))
        self.assertEqual(inventory.take(), [_owned_store(17), _owned_removal(17)])

    def test_rejects_multi_page_identities_and_trailing_only_layouts(self):
        inventory = self._inventory("kv")
        two_pages = _owned_store()
        two_pages.block_hashes = [1, 2]
        with self.assertRaisesRegex(ValueError, "one store payload per page"):
            inventory.register({"page": two_pages}, [b"page#ns#kv"])
        with self.assertRaisesRegex(ValueError, "ALL_PAGES pool"):
            KVCRInventory().configure("ns", [])

    def test_wire_coalescing_cannot_change_inventory_removal_identity(self):
        inventory = self._inventory("kv")
        keys = [b"first#ns#kv", b"second#ns#kv"]
        held = inventory.register(
            {"first": _owned_store(17), "second": _owned_store(18, parent=17)}, keys
        )
        inventory.update(_residency(keys))
        inventory.release(held)
        reducer = StorageEventReducer(1)
        recorder = KVCacheEventRecorder(enabled=True, page_size=2)
        for event in reducer.update({0: inventory.take()}):
            recorder.enqueue(event)
        self.assertEqual(recorder.take()[0].block_hashes, [17, 18])
        inventory.update(_residency(keys[:1], removed=True))
        removals = reducer.update({0: inventory.take()})
        self.assertEqual([event.block_hashes for event in removals], [[17]])


class _Residency(_CopyTransport):
    """Native KVCR stand-in that reports LOCAL_G2 residency to the sink."""

    def __init__(self, layouts, sink):
        super().__init__(layouts)
        self.sink, self.fail = sink, set()

    def _report(self, keys, removed, tier="DRAM"):
        self.sink(
            SimpleNamespace(
                keys=tuple(keys), tier=SimpleNamespace(value=tier), removed=removed
            )
        )

    def deposit(self, blocks):
        kept = {k: v for k, v in blocks.items() if k not in self.fail}
        handle = super().deposit(kept)
        self.completed[handle].update(
            {k: SimpleNamespace(success=False) for k in self.fail if k in blocks}
        )
        # KVCR publishes committed residency before it completes the deposit.
        self._report(kept, removed=False)
        self._report(kept, removed=False, tier="G3")
        return handle

    def evict(self, keys):
        for key in keys:
            self.storage.pop(key)
        self._report(keys, removed=True)

    def close(self):
        pass


class TestKVCRInventoryPublication(CustomTestCase):
    """KVCR DRAM residency reaches a KV-aware router the way Dynamo reads it."""

    HOST = "10.1.2.3"

    def setUp(self):
        self.config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="glm-test",
        )
        self.enterContext(get_parallel().override(dp_rank=None))
        self.pool = _HostPool()
        types_module = types.ModuleType("kvcr.types")
        types_module.MemoryRef, types_module.QueryStatus = _MemoryRef, _QueryStatus
        self.enterContext(
            patch.dict(
                "sys.modules",
                {"kvcr": types.ModuleType("kvcr"), "kvcr.types": types_module},
            )
        )
        self.keys = [f"{i + 1:064x}" for i in range(4)]

    def _extra(self, **overrides):
        return {
            "enable_remote_hint": True,
            "publish_inventory": True,
            "control_advertise_host": self.HOST,
            "control_port": 29017,
            **overrides,
        }

    def _backend(self, config=None, **overrides):
        config = config or dataclasses.replace(
            self.config, extra_config=self._extra(**overrides)
        )
        backend = KVCRStore(config, self.pool)
        self.addCleanup(backend.close)
        return backend

    def _running(self, backend):
        """Emulate _start's namespace and inventory wiring with a fake runner."""
        backend._namespace = storage_namespace(backend.config, backend._layouts)
        backend._inventory.configure(backend._namespace, ["kv"])
        backend._runner = _Residency(backend._layouts, backend._record_inventory)
        return backend._runner

    def _metadata(self, pages, *, parent=None):
        events = {}
        for index, page in enumerate(pages):
            events[page] = BlockStored(
                block_hashes=[100 + index],
                parent_block_hash=parent if index == 0 else 99 + index,
                token_ids=[index, index],
                block_size=2,
                lora_id=None,
                medium=StorageMedium.GPU,
            )
        return events

    def _write(self, backend, keys, metadata):
        return backend.batch_set_v1(
            keys,
            torch.arange(len(keys) * 2, dtype=torch.int64),
            HiCacheStorageExtraInfo(extra_info={"kv_event_metadata": metadata}),
        )

    def test_hint_source_uses_backend_resolved_address_without_starting_io(self):
        config = copy.copy(self.config)
        config.tp_size = 2
        config.extra_config = {
            "enable_remote_hint": True,
            "control_advertise_host": "2001:db8::1",
            "control_port": 29017,
        }
        # Replica 2 of a TP2 deployment, without attention DP.
        with get_parallel().override(dp_rank=2):
            backend = self._backend(config)
        source = backend.get_kv_hint_source()
        self.assertEqual(
            source,
            {
                "dp_rank": 2,
                "capabilities": ["router_hint"],
                "source_control_endpoint": "tcp://[2001:db8::1]:29021",
            },
        )
        host, port = control_address(backend.config, backend._extra)
        self.assertEqual(source["source_control_endpoint"], f"tcp://{host}:{port}")
        self.assertIsNone(backend._runner)
        # No router source without remote hints, an explicit advertised host
        # (the default is loopback), or on any but the replica's first shard.
        backend._remote_hint = False
        self.assertIsNone(backend.get_kv_hint_source())
        backend._remote_hint = True
        backend._extra.pop("control_advertise_host")
        self.assertIsNone(backend.get_kv_hint_source())
        backend._extra["control_advertise_host"] = "worker-a"
        backend._offset = 1
        self.assertIsNone(backend.get_kv_hint_source())

    def test_hint_source_rejects_invalid_advertised_coordinates(self):
        for extra in (
            {"control_advertise_host": "0.0.0.0"},
            {"control_advertise_host": "tcp://worker-a"},
            {"control_advertise_host": "worker-a", "control_port": True},
            {"control_advertise_host": "worker-a", "control_port": 65535},
        ):
            with (
                self.subTest(extra=extra),
                self.assertRaises(ValueError),
                get_parallel().override(dp_rank=1),
            ):
                config = copy.copy(self.config)
                config.extra_config = {"enable_remote_hint": True, **extra}
                self._backend(config).get_kv_hint_source()

    def test_inventory_requires_a_routable_hint_source(self):
        for extra in (
            {"publish_inventory": True},
            {"publish_inventory": True, "control_advertise_host": self.HOST},
            {"publish_inventory": True, "enable_remote_hint": True},
        ):
            config = dataclasses.replace(self.config, extra_config=extra)
            with (
                self.subTest(extra=extra),
                self.assertRaisesRegex(ValueError, "publish_inventory requires"),
            ):
                KVCRStore(config, self.pool)
        # Dynamo's state agent subscribes from another host.
        for host in ("127.0.0.1", "127.1", "localhost", "::1", "::ffff:127.0.0.1"):
            with (
                self.subTest(host=host),
                self.assertRaisesRegex(ValueError, "must not be loopback"),
            ):
                self._backend(control_advertise_host=host)
            # Loopback stays valid for hints alone.
            self.assertTrue(
                self._backend(
                    control_advertise_host=host, publish_inventory=False
                ).get_kv_hint_source()
            )
        with self.assertRaisesRegex(ValueError, "publish_inventory must be a boolean"):
            self._backend(publish_inventory="true")

    def test_rejects_replica_dp_combined_with_attention_dp(self):
        """Replicas would share attention-DP ranks and control ports."""
        module = "sglang.srt.mem_cache.storage.kvcr.kvcr_store"
        config = dataclasses.replace(self.config, dp_rank=1, extra_config={})
        with (
            patch(f"{module}.is_dp_attention_enabled", return_value=True),
            get_parallel().override(dp_size=2),
            self.assertRaisesRegex(ValueError, "replica DP"),
        ):
            KVCRStore(config, self.pool)
        with (
            patch(f"{module}.is_dp_attention_enabled", return_value=True),
            get_parallel().override(dp_size=1),
        ):
            self.assertEqual(self._backend(config).config.dp_rank, 1)

    def test_routable_host_rejects_every_loopback_form(self):
        for host in ("10.0.0.1", "worker-a", "[2001:db8::1]"):
            with self.subTest(host=host):
                require_routable_host(host)
        for host in (
            "127.0.0.1",
            "127.1",
            "2130706433",
            "localhost.",
            "0.0.0.0",
            "[::1]",
            "[::]",
            "[::ffff:127.0.0.1]",
            "*",
        ):
            with self.subTest(host=host), self.assertRaises(ValueError):
                require_routable_host(host)

    def test_cache_owner_is_canonical_engine_lifetime_and_inventory_only(self):
        first, second = self._backend(), self._backend()
        sources = [first.get_kv_hint_source(), second.get_kv_hint_source()]
        for source in sources:
            self.assertEqual(source["capabilities"], ["router_hint", "kvcr_inventory"])
            self.assertRegex(source["cache_owner_id"], _CACHE_OWNER)
        owners = [source["cache_owner_id"] for source in sources]
        # Same model and shard: one indexer domain, the storage namespace.
        namespace = storage_namespace(first.config, first._layouts)
        self.assertEqual(
            {owner.split("/", 1)[0] for owner in owners},
            {f"{namespace[:32]}:{namespace[32:]}"},
        )
        # A new incarnation per engine process, fixed for its lifetime.
        self.assertNotEqual(owners[0], owners[1])
        self.assertEqual(first.get_kv_hint_source()["cache_owner_id"], owners[0])
        self.assertNotIn(
            "cache_owner_id",
            self._backend(publish_inventory=False).get_kv_hint_source(),
        )
        # The advertised domain names exactly the pools it serves.
        with self.assertRaisesRegex(ValueError, "before discovery"):
            first.register_mem_host_pool_v2(_HostPool(), PoolName.MAMBA)

    def test_native_dram_residency_drives_owned_events(self):
        backend = self._backend()
        transport = self._running(backend)
        first, second = self.keys[:2]
        metadata = self._metadata([first])
        self.assertEqual(self._write(backend, [first, second], metadata), [True] * 2)
        self.assertEqual(backend.pending_kv_events(), 1)
        (stored,) = backend.take_kv_events()
        self.assertEqual(backend.take_kv_events(), [])
        # Identity from the framework, owner and medium from KVCR.
        self.assertEqual(stored.block_hashes, [100])
        self.assertEqual(stored.token_ids, [0, 0])
        self.assertEqual((stored.ownership, stored.medium), ("kvcr", "CPU_PINNED"))
        self.assertIsNone(metadata[first].ownership)
        # A resident page keeps its identity after the IO; the page without
        # metadata is resident but never advertised.
        self.assertEqual(set(backend._inventory._metadata), {first})
        transport.evict([block_key(first, backend._namespace, "kv")])
        (removed,) = backend.take_kv_events()
        self.assertEqual(
            removed,
            BlockRemoved(block_hashes=[100], medium="CPU_PINNED", ownership="kvcr"),
        )
        self.assertEqual(backend._inventory._metadata, {})

    def test_failed_or_unannotated_io_advertises_nothing(self):
        backend = self._backend()
        transport = self._running(backend)
        transport.fail = {block_key(self.keys[0], backend._namespace, "kv")}
        self.assertEqual(
            self._write(backend, self.keys[:1], self._metadata(self.keys[:1])),
            [False],
        )
        self.assertEqual(backend.take_kv_events(), [])
        self.assertEqual(backend._inventory._metadata, {})
        # Writes without metadata (e.g. decode offload) are stored, not advertised.
        transport.fail = set()
        self.assertEqual(
            backend.batch_set_v1(
                self.keys[:1], torch.arange(2, dtype=torch.int64), None
            ),
            [True],
        )
        self.assertEqual(backend.take_kv_events(), [])
        # A non-publishing backend never receives or keeps residency.
        quiet = self._backend(publish_inventory=False)
        self.assertEqual(
            (quiet.publishes_kv_events, quiet.pending_kv_events()), (False, 0)
        )
        self.assertEqual(quiet.take_kv_events(), [])

    def test_start_wires_inventory_sink_only_when_publishing(self):
        class StartupError(RuntimeError):
            pass

        for publish in (True, False):
            with self.subTest(publish=publish):
                backend = self._backend(publish_inventory=publish, local_dram_bytes=128)
                bindings = []

                def construct(captured=bindings, **kwargs):
                    captured.append(vars(kwargs["bindings"]))
                    raise RuntimeError("stop after construction")

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
                with (
                    patch.dict("sys.modules", modules),
                    patch(
                        "sglang.srt.mem_cache.storage.kvcr.kvcr_store.os._exit",
                        side_effect=SystemExit(1),
                    ),
                    self.assertLogs(
                        "sglang.srt.mem_cache.storage.kvcr.kvcr_store",
                        level="CRITICAL",
                    ),
                    self.assertRaises(SystemExit),
                ):
                    backend._start()
                if publish:
                    self.assertEqual(
                        bindings[0]["inventory_sink"], backend._record_inventory
                    )
                    self.assertEqual(backend._inventory.namespace, backend._namespace)
                    self.assertEqual(backend._inventory.required_pools, {"kv"})
                else:
                    # Older KVCR builds construct without the field.
                    self.assertNotIn("inventory_sink", bindings[0])

    def test_unified_radix_cache_publishes_only_for_inventory(self):
        for publish in (True, False):
            with self.subTest(publish=publish):
                backend = self._backend(publish_inventory=publish)
                recorder = KVCacheEventRecorder(enabled=False, page_size=2)
                cache = _hint_gate(backend, recorder)
                self.assertIs(cache.storage_kv_events_enabled, publish)
                (source,) = cache.storage_kv_hint_sources()
                self.assertEqual(source, backend.get_kv_hint_source())
                self.assertIs("kvcr_inventory" in source["capabilities"], publish)

    def test_failure_injector_delegates_inventory_hooks(self):
        backend = self._backend()
        wrapped = HiCacheStorageFailureInjector(backend, 0.0, 0.0)
        self.assertIs(wrapped.publishes_kv_events, True)
        self.assertEqual((wrapped.kv_event_shard, wrapped.kv_event_shard_count), (0, 1))
        self.assertEqual(wrapped.get_kv_hint_source(), backend.get_kv_hint_source())
        self.assertEqual(wrapped.pending_kv_events(), 0)
        self.assertEqual(wrapped.take_kv_events(), [])

    def test_init_info_and_wire_events_report_owned_residency(self):
        """The startup metadata and raw events an external state agent reads."""
        from sglang.srt.managers.scheduler_components import kv_events_publisher

        backend = self._backend()
        cache = SimpleNamespace(
            storage_kv_events_enabled=True,
            cache_controller=SimpleNamespace(storage_backend=backend),
        )
        cache.storage_kv_hint_sources = functools.partial(
            BasePrefixCache.storage_kv_hint_sources, cache
        )
        with zmq.Context.instance().socket(zmq.PUB) as probe:
            port = probe.bind_to_random_port("tcp://127.0.0.1")
        config = json.dumps(
            {
                "publisher": "zmq",
                "endpoint": f"tcp://*:{port}",
                "topic": "kv",
                "format": "dynamo",
            }
        )
        parallel = SimpleNamespace(
            pp_rank=0,
            attn_tp_rank=0,
            attn_cp_rank=0,
            attn_dp_size=1,
            attn_dp_rank=0,
            dp_rank=0,
        )
        with (
            patch.object(kv_events_publisher, "get_parallel", return_value=parallel),
            patch(
                "sglang.srt.disaggregation.kv_events.get_parallel",
                return_value=parallel,
            ),
        ):
            component = kv_events_publisher.SchedulerKvEventsPublisher(
                kv_events_config=config,
                tree_cache=cache,
                send_metrics_from_scheduler=None,
                max_running_requests=1,
                max_total_num_tokens=1,
                get_stats=lambda: None,
            )
        publisher = component.kv_event_publisher
        atexit.unregister(publisher.shutdown)
        self.addCleanup(publisher.shutdown)
        # As Scheduler.get_init_info reports it (page size 2, DCP 1).
        info = {
            "kv_hint_sources": cache.storage_kv_hint_sources(),
            "kv_event_sources": component.local_kv_event_sources(2),
        }
        (hint,) = info["kv_hint_sources"]
        owner = hint["cache_owner_id"]
        self.assertRegex(owner, _CACHE_OWNER)
        self.assertEqual(
            hint,
            {
                "dp_rank": 0,
                "capabilities": ["router_hint", "kvcr_inventory"],
                "source_control_endpoint": f"tcp://{self.HOST}:29017",
                "cache_owner_id": owner,
            },
        )
        topic = f"kv/owner/{hashlib.sha256(owner.encode()).hexdigest()}"
        self.assertEqual(
            info["kv_event_sources"],
            [
                {
                    "dp_rank": 0,
                    "endpoint": f"tcp://{self.HOST}:{port}",
                    "topic": topic,
                    "block_size": 2,
                    "format": "dynamo",
                }
            ],
        )

        # The events of one store and one eviction, through the real reducer
        # and the publisher's wire: [topic, u64 BE seq, [ts, events, dp_rank]].
        transport = self._running(backend)
        reducer = StorageEventReducer(backend.kv_event_shard_count)
        self._write(backend, self.keys[:1], self._metadata(self.keys[:1]))
        events = reducer.update({backend.kv_event_shard: backend.take_kv_events()})
        transport.evict([block_key(self.keys[0], backend._namespace, "kv")])
        events += reducer.update({backend.kv_event_shard: backend.take_kv_events()})
        with zmq.Context() as context, context.socket(zmq.SUB) as subscriber:
            subscriber.setsockopt_string(zmq.SUBSCRIBE, topic)
            subscriber.connect(f"tcp://127.0.0.1:{port}")
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                publisher.publish(KVEventBatch(ts=1.0, events=events))
                if subscriber.poll(100):
                    frames = subscriber.recv_multipart()
                    break
            else:
                self.fail("No event received from the advertised raw source")
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[0], topic.encode())
        self.assertEqual(len(frames[1]), 8)
        ts, wire_events, dp_rank = msgspec.msgpack.decode(frames[2])
        self.assertEqual(dp_rank, 0)
        self.assertEqual(
            [event["type"] for event in wire_events], ["BlockStored", "BlockRemoved"]
        )
        for event in wire_events:
            self.assertEqual(
                (event["ownership"], event["medium"]), ("kvcr", "CPU_PINNED")
            )
            self.assertTrue(all(type(h) is int for h in event["block_hashes"]))
        self.assertEqual(len(wire_events[0]["token_ids"]), wire_events[0]["block_size"])


if __name__ == "__main__":
    unittest.main()
