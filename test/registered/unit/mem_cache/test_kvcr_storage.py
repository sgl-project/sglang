"""CPU contracts for indexed HiCache IO; NIXL is faked except in TestRealKVCR."""

import copy
import ctypes
import dataclasses
import importlib.util
import math
import mmap
import socket
import sys
import threading
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

from sglang.srt.disaggregation.kv_events import (
    namespaced_block_hash,
)
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
from sglang.srt.mem_cache.storage.kvcr.endpoints import (
    control_address,
)
from sglang.srt.mem_cache.storage.kvcr.hints import parse_router_hint, peer_hint
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
from sglang.srt.utils.network import is_port_available
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, find_available_port

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _route(info, offset, page_keys):
    """A request's router hint for one shard, narrowed to these pages."""
    extra = info.extra_info
    hint = peer_hint(
        parse_router_hint(extra["kv_hints"], offset),
        page_keys,
        extra.get("storage_key_to_event_hash"),
    )
    return None if hint is None else msgspec.to_builtins(hint)


def _page(router_hash):
    """A full storage-page key whose router hash is router_hash."""
    return f"{router_hash:016x}" + "0" * 48


class _ControlChannelExport:
    """kvcr>=0.2.0's control channel, as construction checks it."""

    def adopt_listener(self, listener_fd):
        raise AssertionError("not started in this test")


def _kvcr_module():
    """A kvcr>=0.2.0 stand-in: construction checks only this export."""
    module = types.ModuleType("kvcr")
    module.ZmqPeerControlChannel = _ControlChannelExport
    return module


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

    def close(self):
        pass


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
            (
                _QueryStatus.HIT
                if key in self.storage
                else _QueryStatus.FETCHABLE
                if key in available
                else _QueryStatus.MISS,
                None,
            )
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
        # A standalone replica on one node: no DP controller numbered this process.
        self.enterContext(get_parallel().override(dp_rank=None, nnodes=1))
        # Real port reservations are covered by TestPortReservation.
        self.enterContext(patch.object(KVCRStore, "_reserve_ports"))
        types_module = types.ModuleType("kvcr.types")
        types_module.MemoryRef, types_module.QueryStatus = _MemoryRef, _QueryStatus
        self.stub = patch.dict(
            "sys.modules",
            {"kvcr": _kvcr_module(), "kvcr.types": types_module},
        )
        self.stub.start()
        self.addCleanup(self.stub.stop)
        self.pool = _HostPool()
        self.backend = KVCRStore(self.config, self.pool)
        self.backend._namespace = storage_namespace(self.config, self.backend._layouts)
        self.transport = self.backend._runner = _CopyTransport(self.backend._layouts)
        self.keys = [f"{i + 1:064x}" for i in range(4)]

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
        translated = _route(info, 1, pages)
        payload = translated["actions"][0]["payload"]
        self.assertEqual(payload["source_control_endpoint"], "tcp://peer:32101")
        self.assertEqual(payload["block_hashes"], [int(h[:16], 16) for h in pages[:2]])
        self.assertEqual(hint, saved)
        other = get_storage_hash_str(
            RadixKey(array("q", range(8)), cache_salt="tenant-b"), page_size=2
        )
        self.assertIsNone(_route(info, 0, other))
        # Narrowed to the batch: a page outside the hint is not named.
        self.assertEqual(
            _route(info, 0, pages[1:])["actions"][0]["payload"]["block_hashes"],
            [int(pages[1][:16], 16)],
        )
        for invalid in ({}, [], {pages[0]: True}, {pages[0]: -1}):
            info.extra_info["storage_key_to_event_hash"] = invalid
            self.assertIsNone(_route(info, 0, pages))

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

    def test_static_misconfiguration_fails_construction(self):
        """A missing package or bad option fails the attach, not every later IO."""
        no_adopt = types.ModuleType("kvcr")
        no_adopt.ZmqPeerControlChannel = object
        for module in (None, types.ModuleType("kvcr"), no_adopt):
            # Missing, or older than 0.2.0 (no top-level ZmqPeerControlChannel,
            # or a control channel that cannot adopt a reserved listener).
            with (
                self.subTest(kvcr=module),
                patch.dict("sys.modules", {"kvcr": module}),
                self.assertRaisesRegex(ImportError, "kvcr>=0.2.0"),
            ):
                KVCRStore(self.config, self.pool)
        for extra, message in (
            ({"local_dram_bytes": "8G"}, "local DRAM budget"),
            ({"local_dram_bytes": 0}, "local DRAM budget"),
            ({"local_dram_pool_pages": [1]}, "local DRAM budget"),
            ({"local_dram_pool_pages": {"kv": 0}}, "local DRAM budget"),
            ({"local_dram_pool_pages": {"kv": "100"}}, "local DRAM budget"),
            (
                {"operation_timeout_ms": 40000, "abandon_timeout_ms": 60000},
                "at least twice",
            ),
            ({"operation_timeout_ms": 0}, "operation_timeout_ms"),
            ({"operation_timeout_ms": "30000"}, "operation_timeout_ms"),
            ({"abandon_timeout_ms": True}, "abandon_timeout_ms"),
            # Too short to derive an operation deadline from.
            ({"abandon_timeout_ms": 1}, "at least twice"),
            ({"nixl_listen_port": 70000}, "nixl_listen_port"),
            ({"nixl_listen_port": True}, "nixl_listen_port"),
            ({"control_port": 70000}, "control_port"),
            # KVCR 0.2.0's control channel binds IPv4 only.
            ({"control_host": "::"}, "IPv4"),
            ({"control_host": "fd00::1"}, "IPv4"),
            (
                {"enable_remote_hint": True, "control_advertise_host": "2001:db8::1"},
                "control_advertise_host must be an IPv4",
            ),
            ({"control_port": 65535}, "control-port range"),
            ({"control_port": 40000, "nixl_listen_port": 40000}, "overlaps"),
            ({"control_advertise_host": "0.0.0.0"}, "unspecified"),
        ):
            with (
                self.subTest(extra=extra),
                self.assertRaisesRegex(ValueError, message),
            ):
                KVCRStore(
                    dataclasses.replace(self.config, extra_config=extra), self.pool
                )
        # A budget below one page fails the attach once the pools are known.
        tiny = KVCRStore(
            dataclasses.replace(self.config, extra_config={"local_dram_bytes": 1}),
            self.pool,
        )
        self.addCleanup(tiny.close)
        with self.assertRaisesRegex(ValueError, "cannot hold one page"):
            tiny.on_host_pools_registered()
        # A pool registered after the attach (a draft sidecar) is checked
        # against the shrunken split when it registers, not at the first IO.
        late = KVCRStore(
            dataclasses.replace(
                self.config,
                # 88 host bytes per pool: 12 bytes give the KV pool one page
                # alone, and no pool a page once a second one joins.
                extra_config={"local_dram_bytes": 12},
            ),
            self.pool,
        )
        self.addCleanup(late.close)
        late.on_host_pools_registered()
        with self.assertRaisesRegex(ValueError, "cannot hold one page"):
            late.register_mem_host_pool_v2(_HostPool(), "draft_kv")
        self.assertNotIn("draft_kv", late._layouts)
        # A pool name that matches nothing is reported once registration has
        # closed, so a draft pool registering after the attach is not flagged.
        module = "sglang.srt.mem_cache.storage.kvcr.kvcr_store"
        named = KVCRStore(
            dataclasses.replace(
                self.config,
                extra_config={"local_dram_pool_pages": {"draft": 4, "kv_typo": 4}},
            ),
            self.pool,
        )
        self.addCleanup(named.close)
        with self.assertNoLogs(module, "WARNING"):
            named.on_host_pools_registered()
        named.register_mem_host_pool_v2(_HostPool(), "draft")
        with self.assertLogs(module, "WARNING") as logs:
            named._warn_unmatched_pool_pages()
        self.assertIn("['kv_typo']", logs.output[0])
        # Native startup (registration, binds) still waits for the first IO.
        backend = KVCRStore(self.config, self.pool)
        self.addCleanup(backend.close)
        backend.on_host_pools_registered()
        self.assertIsNone(backend._runner)
        self.assertIsNone(backend._poll_thread)

    def test_default_deadlines_scale_with_one_storage_batch(self):
        """Large pages get more time before a slow transfer becomes uncertain."""

        def pools(page_bytes):
            component = SimpleNamespace(size=page_bytes)
            return {"kv": SimpleNamespace(components=[component])}

        def timeouts(page_bytes, **extra):
            store = KVCRStore(
                dataclasses.replace(self.config, extra_config=extra), self.pool
            )
            self.addCleanup(store.close)
            return store._timeouts(pools(page_bytes))

        # The test pool, and 8 MiB pages (1 GiB batches), keep the floors.
        self.assertEqual(self.backend._timeouts(), (5000, 30000))
        self.assertEqual(timeouts(8 << 20), (5000, 30000))
        # 64 MiB pages: an 8 GiB batch at 250 MB/s.
        operation = math.ceil(128 * (64 << 20) * 1000 / (250 * 1000**2))
        self.assertEqual(timeouts(64 << 20), (operation, 3 * operation))
        # Setting one derives the other, within KVCR's 2x rule.
        self.assertEqual(timeouts(64 << 20, abandon_timeout_ms=20000), (10000, 20000))
        self.assertEqual(timeouts(64 << 20, operation_timeout_ms=1000), (1000, 30000))
        self.assertEqual(
            timeouts(1, operation_timeout_ms=4000, abandon_timeout_ms=9000),
            (4000, 9000),
        )

    def test_remote_hints_require_one_node(self):
        """Peers address shard k from shard 0's host, so shards cannot span nodes."""
        hinted = {"enable_remote_hint": True, "control_advertise_host": "10.0.0.1"}
        with get_parallel().override(nnodes=2):
            with self.assertRaisesRegex(ValueError, "--nnodes 1"):
                KVCRStore(
                    dataclasses.replace(self.config, tp_rank=1, extra_config=hinted),
                    self.pool,
                )
            # Local-only KVCR has no cross-node addressing.
            local = KVCRStore(self.config, self.pool)
            self.addCleanup(local.close)

    def test_unknown_extra_config_keys_warn(self):
        module = "sglang.srt.mem_cache.storage.kvcr.kvcr_store"
        with self.assertLogs(module, "WARNING") as logs:
            typo = KVCRStore(
                dataclasses.replace(
                    self.config,
                    extra_config={"enable_remote_hints": True, "control_port": 32100},
                ),
                self.pool,
            )
        self.assertEqual(len(logs.output), 1)
        self.assertIn("['enable_remote_hints']", logs.output[0])
        self.assertFalse(typo.consumes_kv_hints())
        documented = {
            "enable_remote_hint": True,
            "share_replicated_kv": False,
            "control_advertise_host": "source.example",
            "control_host": "0.0.0.0",
            "control_port": 32100,
            "nixl_listen_port": 33000,
            "local_dram_bytes": 1 << 20,
            "local_dram_pool_pages": {},
            "operation_timeout_ms": 1000,
            "abandon_timeout_ms": 2000,
            "startup_timeout_seconds": 60,
            # Generic HiCache keys: decode offload passes this one through,
            # and the controller reads interface_v1 for a "dynamic" backend.
            "prefetch_threshold": 256,
            "interface_v1": 1,
        }
        with self.assertNoLogs(module, "WARNING"):
            KVCRStore(
                dataclasses.replace(self.config, extra_config=documented), self.pool
            )

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
        must retain them until process exit. An unset optional deadline keeps
        KVCR's default.
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
                            self.assertIsNone(captured[0]["startup_timeout_seconds"])
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

    def test_waiter_polls_its_own_completion(self):
        """Without a poll thread, the waiter alone drives completion."""
        results = [[], [], [(7, {"k": True})]]
        with patch.object(
            self.transport, "poll_completed", side_effect=results
        ) as poll:
            self.assertEqual(self.backend._wait(7), {"k": True})
        self.assertEqual(poll.call_count, 3)

    def test_peer_hint_no_retain_by_value(self):
        """false/null carry no retention mark; anything else skips the fetch."""
        for value, kept in (
            (False, True),
            (None, True),
            (True, False),
            ("false", False),
            (0, False),
        ):
            info = HiCacheStorageExtraInfo(
                extra_info={
                    "kv_hints": {
                        "protocol_version": "0.1",
                        "actions": [
                            {
                                "action_type": "kv.fetch",
                                "action_version": "1.0",
                                "payload": {
                                    "source_control_endpoint": "tcp://source:32000",
                                    "block_hashes": [7],
                                    "no_retain": value,
                                },
                            }
                        ],
                    }
                }
            )
            with self.subTest(no_retain=value):
                hint = _route(info, 0, [_page(7), _page(8)])
                if not kept:
                    self.assertIsNone(hint)
                    continue
                payload = hint["actions"][0]["payload"]
                self.assertEqual(payload["block_hashes"], [7])
                self.assertNotIn("no_retain", payload)

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
        page = _page((1 << 63) + 1)
        self.assertEqual(
            _route(info, 3, [page])["actions"][0]["payload"]["source_control_endpoint"],
            "tcp://source:32003",
        )
        # The default backend ignores hints even when a caller passes one.
        self.assertIsNone(self.backend._router_hint(info, [page]))
        hinted = KVCRStore(
            dataclasses.replace(self.config, extra_config={"enable_remote_hint": True}),
            self.pool,
        )
        transport = hinted._runner = _CopyTransport(hinted._layouts)
        with patch(
            "sglang.srt.mem_cache.storage.kvcr.kvcr_store.parse_router_hint",
            wraps=parse_router_hint,
        ) as parse:
            hint = hinted._router_hint(info, [page])
            # Parsed once per request envelope, however many batches it spans.
            self.assertEqual(hinted._router_hint(info, [page]), hint)
        parse.assert_called_once()
        with hinted._hint(hint) as request_id:
            self.assertEqual(len(transport.hints), 1)
            self.assertIn(request_id, transport.hints)
        self.assertEqual(transport.hints, {})
        info.extra_info["kv_hints"]["actions"][0]["payload"]["block_hashes"] = [True]
        self.assertIsNone(_route(info, 0, [page]))

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

    def test_extra_backend_tag_separates_namespace(self):
        """Same served name, different weights: the tag keeps their pages apart."""
        namespace = storage_namespace(self.config, self.backend._layouts)
        tagged = [
            storage_namespace(
                dataclasses.replace(
                    self.config, extra_config={"extra_backend_tag": tag}
                ),
                self.backend._layouts,
            )
            for tag in ("ckpt-a", "ckpt-b")
        ]
        self.assertEqual(len({namespace, *tagged}), 3)
        for tag in ("", 7):
            with (
                self.subTest(tag=tag),
                self.assertRaisesRegex(ValueError, "extra_backend_tag"),
            ):
                KVCRStore(
                    dataclasses.replace(
                        self.config, extra_config={"extra_backend_tag": tag}
                    ),
                    self.pool,
                )

    def test_trailing_window_holes_do_not_become_contiguous_prefixes(self):
        self.assertEqual(resume_boundaries([False, True, True, False], trailing=2), [3])
        self.assertEqual(resume_boundaries([True, False, True], trailing=None), [1])


class TestKVCRHintSource(CustomTestCase):
    """The router source a KV-aware router (e.g. Dynamo's) discovers."""

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
        self.enterContext(get_parallel().override(dp_rank=None, nnodes=1))
        self.enterContext(patch.object(KVCRStore, "_reserve_ports"))
        self.pool = _HostPool()
        types_module = types.ModuleType("kvcr.types")
        types_module.MemoryRef, types_module.QueryStatus = _MemoryRef, _QueryStatus
        self.enterContext(
            patch.dict(
                "sys.modules",
                {"kvcr": _kvcr_module(), "kvcr.types": types_module},
            )
        )
        self.keys = [f"{i + 1:064x}" for i in range(4)]

    def _extra(self, **overrides):
        return {
            "enable_remote_hint": True,
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

    def test_hint_source_uses_backend_resolved_address_without_starting_io(self):
        config = copy.copy(self.config)
        config.tp_size = 2
        config.extra_config = {
            "enable_remote_hint": True,
            "control_advertise_host": "192.0.2.7",
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
                "source_control_endpoint": "tcp://192.0.2.7:29021",
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

    def test_control_endpoint_exists_only_with_remote_hints(self):
        """Peers fetch through the unauthenticated control endpoint, so a
        local-only store neither binds nor serves one."""
        for hinted in (True, False):
            with self.subTest(enable_remote_hint=hinted):
                backend = self._backend(enable_remote_hint=hinted, local_dram_bytes=128)
                channels, bindings = [], []

                def channel(created=channels, **kwargs):
                    created.append(kwargs)
                    return SimpleNamespace(endpoint="tcp://x:1", close=lambda: None)

                def construct(captured=bindings, **kwargs):
                    captured.append(vars(kwargs["bindings"]))
                    raise RuntimeError("stop after construction")

                modules = {
                    "kvcr": SimpleNamespace(
                        KVCR=construct,
                        KVCRBindings=SimpleNamespace,
                        ZmqPeerControlChannel=channel,
                    ),
                    "kvcr.config": SimpleNamespace(
                        KVCRBackendConfigs=SimpleNamespace,
                        KVCRConfig=SimpleNamespace,
                        LocalDramOptions=SimpleNamespace,
                    ),
                    "kvcr.types": SimpleNamespace(
                        KVCRStartupError=type("StartupError", (RuntimeError,), {}),
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
                self.assertEqual(len(channels), int(hinted))
                if hinted:
                    self.assertIsNotNone(bindings[0]["framework_control"])
                else:
                    self.assertIsNone(bindings[0]["framework_control"])

    def test_prefix_cache_reports_the_backend_source(self):
        backend = self._backend()
        recorder = KVCacheEventRecorder(enabled=False, page_size=2)
        (source,) = _hint_gate(backend, recorder).storage_kv_hint_sources()
        self.assertEqual(source, backend.get_kv_hint_source())
        local = self._backend(enable_remote_hint=False)
        self.assertEqual(_hint_gate(local, recorder).storage_kv_hint_sources(), [])

    def test_failure_injector_delegates_the_hint_source(self):
        backend = self._backend()
        wrapped = HiCacheStorageFailureInjector(backend, 0.0, 0.0)
        self.assertEqual(wrapped.get_kv_hint_source(), backend.get_kv_hint_source())
        self.assertIs(wrapped.requires_rank_local_backup, True)
        self.assertIs(wrapped.rank_keyed_mha_drafts, True)


class _FakeChannel:
    """ZmqPeerControlChannel stand-in: records its addresses and listener."""

    def __init__(self, bind_host, bind_port, advertise_host):
        self.bind = (bind_host, bind_port)
        self.endpoint = f"tcp://{advertise_host}:{bind_port}"
        self.adopted = None

    def adopt_listener(self, listener_fd):
        self.adopted = socket.socket(fileno=listener_fd)

    def close(self):
        if self.adopted is not None:
            self.adopted.close()


class _FakeKVCR(_CopyTransport):
    """KVCR stand-in whose completions only the store's poll thread drains."""

    def __init__(self, *, config, bindings, backend_configs):
        regions = backend_configs.framework_regions
        super().__init__(
            {
                "regions": SimpleNamespace(
                    components=[
                        SimpleNamespace(
                            name=r.label, address=r.addr, size=r.size, stride=r.stride
                        )
                        for r in regions
                    ]
                )
            }
        )
        self.config, self.bindings = config, bindings
        self.backend_configs = backend_configs
        self.held, self.polls, self.closed, self.fail = {}, 0, 0, False

    def _complete(self, result):
        self.next_handle += 1
        self.held[self.next_handle] = result
        return self.next_handle

    def poll_completed(self):
        if threading.current_thread().name != "kvcr-storage-completions":
            return []
        self.polls += 1
        if self.fail:
            raise RuntimeError("progress failed")
        done, self.held = list(self.held.items()), {}
        return done

    def close(self):
        self.closed += 1
        if self.bindings.framework_control is not None:
            self.bindings.framework_control.close()


def _fake_kvcr_modules():
    kvcr = types.ModuleType("kvcr")
    kvcr.KVCR, kvcr.KVCRBindings = _FakeKVCR, SimpleNamespace
    kvcr.ZmqPeerControlChannel = _FakeChannel
    config = types.ModuleType("kvcr.config")
    config.KVCRBackendConfigs = config.KVCRConfig = SimpleNamespace
    config.LocalDramOptions = SimpleNamespace
    kvcr_types = types.ModuleType("kvcr.types")
    kvcr_types.MemoryRef, kvcr_types.QueryStatus = _MemoryRef, _QueryStatus
    kvcr_types.RegionDescriptor = SimpleNamespace
    kvcr_types.KVCRStartupError = type("KVCRStartupError", (Exception,), {})
    kvcr_types.TransferError = type("TransferError", (Exception,), {})
    return {"kvcr": kvcr, "kvcr.config": config, "kvcr.types": kvcr_types}


def _hint_payload(transport, request_id):
    return transport.hints[request_id]["actions"][0]["payload"]


class TestSharedReplicatedKV(CustomTestCase):
    """share_replicated_kv: one copy of rank-replicated pools, on the writer."""

    def setUp(self):
        self.enterContext(get_parallel().override(dp_rank=None, nnodes=1))
        self.enterContext(patch.object(KVCRStore, "_reserve_ports"))
        self.enterContext(patch.dict("sys.modules", _fake_kvcr_modules()))
        # Distinct router hashes (the leading 16 hex digits) per page.
        self.keys = [f"{i + 1:016x}" * 4 for i in range(4)]

    def _config(self, tp_rank, tp_size=2, **extra):
        return HiCacheStorageConfig(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=True,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="mla-test",
            extra_config={
                "share_replicated_kv": True,
                "local_dram_bytes": 256,
                **extra,
            },
        )

    def _store(self, tp_rank, pool=None, **extra):
        store = KVCRStore(self._config(tp_rank, **extra), pool or _HostPool())
        self.addCleanup(store.close)
        return store

    def _ready(self, store, transport):
        store._namespace, store._shared_namespace = store._namespaces()
        store._runner = transport
        return transport

    def _pair(self, *, replica_pools=(), **extra):
        writer, replica = self._store(0, **extra), self._store(1, **extra)
        for name, pool in replica_pools:
            replica.register_mem_host_pool_v2(pool, name)
        source = self._ready(writer, _CopyTransport(writer._layouts))
        peers = {replica._writer_endpoint: source.storage}
        reader = self._ready(replica, _StrictPeerTransport(replica._layouts, peers))
        return writer, replica, source, reader

    def test_sharing_is_opt_in_and_needs_one_node_without_cp(self):
        logger = "sglang.srt.mem_cache.storage.kvcr.kvcr_store"
        with self.assertLogs(logger, "WARNING") as logs:
            store = KVCRStore(
                dataclasses.replace(self._config(1), extra_config={}), _HostPool()
            )
        self.assertIn("2 copies per page", logs.output[0])
        self.assertIn("share_replicated_kv is off", logs.output[0])
        self.assertFalse(store._shared)
        self.assertTrue(store.requires_rank_local_backup)
        # The store reads attention TP/CP placement and node count from the
        # runtime topology; a stub keeps it free of the topology validator.
        cp = dataclasses.replace(self._config(1), tp_size=4, attn_cp_size=2)
        for reason, config, topology in (
            ("--nnodes > 1", self._config(1), {"nnodes": 2}),
            ("attention CP > 1", cp, {"attn_tp_rank": 1, "attn_tp_size": 2}),
        ):
            parallel = SimpleNamespace(
                **{"dp_rank": None, "dp_size": 1, "nnodes": 1, **topology}
            )
            with (
                self.subTest(reason=reason),
                patch(
                    "sglang.srt.mem_cache.storage.kvcr.kvcr_store.get_parallel",
                    return_value=parallel,
                ),
                self.assertLogs(logger, "WARNING") as logs,
            ):
                self.assertFalse(KVCRStore(config, _HostPool())._shared)
            self.assertIn(reason, logs.output[0])
        # Nothing to share: rank-sharded KV, or a single attention rank.
        for config in (
            dataclasses.replace(self._config(1), is_mla_model=False),
            self._config(0, tp_size=1),
        ):
            self.assertFalse(KVCRStore(config, _HostPool())._shared)

    def test_shared_store_elects_the_writer_shard(self):
        writer, replica = self._store(0), self._store(1)
        for store, is_writer in ((writer, True), (replica, False)):
            self.assertTrue(store._shared)
            self.assertFalse(store.requires_rank_local_backup)
            self.assertEqual(store._writer, is_writer)
            self.assertEqual(store._source_offset, 0)
            # Its control endpoint lets siblings reach the writer.
            self.assertEqual(store._idle_poll_max, 0.02)
        self.assertIsNone(writer._writer_endpoint)
        self.assertEqual(replica._writer_endpoint, "tcp://127.0.0.1:32100")
        # Siblings dial the control endpoint, which KVCR 0.2.0 binds on IPv4.
        with self.assertRaisesRegex(ValueError, "control_advertise_host"):
            KVCRStore(self._config(1, control_advertise_host="fd00::1"), _HostPool())
        # The controller elects the same writer and skips the replica's backup.
        controller = object.__new__(HiCacheController)
        for store, skipped in ((writer, False), (replica, True)):
            controller.storage_config = store.config
            controller.storage_backend = store
            self.assertEqual(controller.backup_skip, skipped)

    def test_shared_pools_use_the_writers_namespace(self):
        writer, replica = self._store(0), self._store(1)
        mamba = PoolName.MAMBA
        for store in (writer, replica):
            store.register_mem_host_pool_v2(_HostPool(sizes=(5,)), mamba)
            store._namespace, store._shared_namespace = store._namespaces()
        self.assertEqual(writer._namespace, writer._shared_namespace)
        self.assertEqual(replica._shared_namespace, writer._namespace)
        self.assertNotEqual(replica._namespace, writer._namespace)
        key = self.keys[0]
        self.assertEqual(replica._key(key, "kv"), writer._key(key, "kv"))
        self.assertNotEqual(replica._key(key, "mamba"), writer._key(key, "mamba"))
        # A store that keeps every rank's copy never pairs with a shared one.
        unshared = KVCRStore(
            dataclasses.replace(self._config(0), extra_config={}), _HostPool()
        )
        unshared.register_mem_host_pool_v2(_HostPool(sizes=(5,)), mamba)
        self.assertNotEqual(unshared._namespaces()[0], writer._namespace)

    def test_replica_restores_shared_pools_from_the_writer(self):
        writer, replica, source, reader = self._pair()
        for part, buffer in enumerate(writer.mem_pool_host.buffers):
            buffer[:2].fill_(40 + part)
        self.assertEqual(
            writer.batch_set_v1(self.keys[:2], torch.arange(4)), [True] * 2
        )
        hints = []
        submit = reader.submit_hint
        reader.submit_hint = lambda hint, request_id=None: (
            hints.append(hint),
            submit(hint, request_id),
        )
        self.assertEqual(
            replica.batch_get_v1(self.keys[:2], torch.arange(8, 12)), [True] * 2
        )
        for part, buffer in enumerate(replica.mem_pool_host.buffers):
            self.assertTrue(torch.all(buffer[4:6] == 40 + part))
        payload = hints[0].actions[0].payload
        self.assertEqual(payload["source_control_endpoint"], "tcp://127.0.0.1:32100")
        # KVCR rejects any other container type for block_hashes.
        self.assertIs(type(payload["block_hashes"]), list)
        self.assertEqual(
            payload["block_hashes"], [int(key[:16], 16) for key in self.keys[:2]]
        )
        self.assertEqual(reader.hints, {})
        # The writer reads its own DRAM and never names itself as a source.
        self.assertEqual(
            writer.batch_get_v1(self.keys[:2], torch.arange(12, 16)), [True] * 2
        )
        self.assertEqual(source.hints, {})

    def test_replica_lookup_defers_shared_pools_to_the_writer(self):
        _, replica, _, reader = self._pair(
            replica_pools=[(PoolName.MAMBA, _HostPool(sizes=(5,)))]
        )
        queries = []
        query = reader.query
        reader.query = lambda keys, request_id=None: (
            queries.append((list(keys), request_id)) or query(keys, request_id)
        )
        # KV alone: the writer's answer decides, so nothing is asked here.
        self.assertEqual(replica.batch_exists(self.keys), 4)
        self.assertEqual(queries, [])
        # Its own (rank-sharded) Mamba checkpoints come from local DRAM.
        mamba = PoolTransfer(
            PoolName.MAMBA, keys=self.keys[-1:], hit_policy=PoolHitPolicy.TRAILING_PAGES
        )
        replica.batch_set_v2(
            [
                PoolTransfer(
                    PoolName.MAMBA, keys=self.keys[1:2], host_indices=torch.arange(2)
                )
            ]
        )
        result = replica.batch_exists_v2(self.keys, [mamba])
        self.assertEqual(result.extra_pool_hit_pages, {"kv": 4, "mamba": 2})
        self.assertEqual(result.restorable_prefix_pages, [2])
        self.assertEqual(
            queries,
            [([replica._key(key, "mamba") for key in self.keys], None)],
        )

    def test_replica_reads_hinted_pages_remotely_then_from_its_writer(self):
        extra = {"enable_remote_hint": True, "control_advertise_host": "10.0.0.9"}
        writer, replica, source, reader = self._pair(**extra)
        remote = self._store(0, **extra)
        remote_source = self._ready(remote, _CopyTransport(remote._layouts))
        reader.peers["tcp://peer:32100"] = remote_source.storage
        # The remote replica holds page 0, this replica's writer page 1.
        for store, key in ((remote, self.keys[0]), (writer, self.keys[1])):
            store.mem_pool_host.buffers[0][0].fill_(7)
            self.assertEqual(store.batch_set_v1([key], torch.arange(2)), [True])
        envelope = {
            "protocol_version": "0.1",
            "message_id": "m",
            "actions": [
                {
                    "action_id": "a",
                    "action_type": "kv.fetch",
                    "action_version": "1.0",
                    "payload": {
                        # Advertised at the source replica's first shard.
                        "source_control_endpoint": "tcp://peer:32100",
                        "block_hashes": [int(k[:16], 16) for k in self.keys[:2]],
                    },
                }
            ],
        }
        info = HiCacheStorageExtraInfo(extra_info={"kv_hints": envelope})
        self.assertEqual(
            replica.batch_get_v1(self.keys[:3], torch.arange(6), info),
            [True, True, False],
        )
        ids = [replica._key(key, "kv") for key in self.keys[:3]]
        # This replica's writer first, as the writer reads its own DRAM first;
        # then its misses from the remote writer, which the hint names only
        # for page 0 (page 2 goes nowhere and misses).
        self.assertEqual(reader.reads, [tuple(ids), (ids[0], ids[2])])
        self.assertTrue(torch.all(replica.mem_pool_host.buffers[0][:2] == 7))

    def _draft_replica(self, **extra):
        mha = patch(
            "sglang.srt.mem_cache.storage.kvcr.kvcr_store.MHATokenToKVPoolHost",
            _HostPool,
        )
        with mha:
            replica = self._store(1, **extra)
            replica.register_mem_host_pool_v2(_HostPool(sizes=(6,)), PoolName.DRAFT)
        return replica

    def test_replica_hints_its_own_pools_at_its_own_shard(self):
        """An MHA draft is per rank: rank 1 reads it from the peer's rank 1."""
        extra = {"enable_remote_hint": True, "control_advertise_host": "10.0.0.9"}
        replica = self._draft_replica(**extra)
        reader = self._ready(replica, _StrictPeerTransport(replica._layouts, {}))
        draft = replica._key(self.keys[0], "draft")
        reader.peers["tcp://peer:32101"] = {draft: {"draft_0": bytes(6)}}
        envelope = {
            "protocol_version": "0.1",
            "actions": [
                {
                    "action_id": "a",
                    "action_type": "kv.fetch",
                    "action_version": "1.0",
                    "payload": {
                        "source_control_endpoint": "tcp://peer:32100",
                        "block_hashes": [int(self.keys[0][:16], 16)],
                    },
                }
            ],
        }
        info = HiCacheStorageExtraInfo(extra_info={"kv_hints": envelope})
        result = replica.batch_exists_v2(
            self.keys[:1], [PoolTransfer(PoolName.DRAFT)], info
        )
        self.assertEqual(result.extra_pool_hit_pages, {"kv": 1, "draft": 1})
        transfers = [
            PoolTransfer(
                PoolName.DRAFT, keys=self.keys[:1], host_indices=torch.arange(2)
            )
        ]
        self.assertEqual(replica.batch_get_v2(transfers, info), {"draft": [True]})
        self.assertEqual(reader.reads, [(draft,)])

    def test_replica_never_backs_up_a_shared_pool(self):
        _, replica, _, reader = self._pair()
        with self.assertRaisesRegex(ValueError, "Only the writer shard"):
            replica.batch_set_v1(self.keys[:1], torch.arange(2))
        self.assertEqual(reader.storage, {})

    def test_mha_draft_pools_stay_rank_local(self):
        controller = object.__new__(HybridCacheController)
        controller.storage_backend_type = "dynamic"
        controller.storage_config = SimpleNamespace(is_storage_writer=False)
        draft = PoolTransfer(PoolName.DRAFT)
        with patch(
            "sglang.srt.mem_cache.storage.kvcr.kvcr_store.MHATokenToKVPoolHost",
            _HostPool,
        ):
            replica = self._store(1)
            replica.register_mem_host_pool_v2(_HostPool(), PoolName.DRAFT)
        self.assertTrue(replica._shares("kv"))
        self.assertFalse(replica._shares("draft"))
        # HiCache backs it up on every rank, keyed by rank, here too.
        controller.storage_backend = replica
        with patch(
            "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller."
            "MHATokenToKVPoolHost",
            _HostPool,
        ):
            controller.mem_pool_host = SimpleNamespace(
                entry_map={
                    PoolName.DRAFT: SimpleNamespace(host_pool=_HostPool()),
                }
            )
            self.assertTrue(controller.should_backup(draft))
            controller.storage_backend = SimpleNamespace(
                requires_rank_local_backup=False, rank_keyed_mha_drafts=False
            )
            self.assertFalse(controller.should_backup(draft))

    def _started(self, store):
        store._start()
        return store._runner

    def test_start_gives_a_replica_no_shared_dram(self):
        writer, replica = self._store(0), self._store(1)
        kv_parts = ["kv_0", "kv_1"]
        runner = self._started(writer)
        self.assertEqual(
            [name for name, _, _ in runner.backend_configs.local_dram.pools], kv_parts
        )
        runner = self._started(replica)
        # Restores still land in every registered pool.
        self.assertEqual(
            [r.label for r in runner.backend_configs.framework_regions], kv_parts
        )
        self.assertIsNone(runner.backend_configs.local_dram)
        self.assertEqual(replica._maps, [])
        channel = runner.bindings.framework_control
        # Only this replica dials it: bound and advertised on loopback.
        self.assertEqual(channel.bind, ("127.0.0.1", 32101))
        self.assertEqual(channel.endpoint, "tcp://127.0.0.1:32101")
        self.assertEqual(replica.batch_exists(self.keys), 4)

    def test_start_keeps_one_shared_page_beside_rank_local_pools(self):
        # KVCR's DRAM pools must match its pool layouts.
        replica = self._store(1)
        replica.register_mem_host_pool_v2(_HostPool(sizes=(5,)), PoolName.MAMBA)
        pages = replica._pool_pages()
        runner = self._started(replica)
        pools = {
            name: length for name, _, length in runner.backend_configs.local_dram.pools
        }
        self.assertEqual(list(pools), [name for name, _ in runner.config.pool_layouts])
        self.assertEqual(pools["kv_0"], 8)
        self.assertEqual(pools["kv_1"], 3)
        self.assertEqual(pools["mamba_0"], pages["mamba"] * 5)

    def test_shared_deadlines_stay_per_rank(self):
        # 128 pages of 1 MiB at 250 MB/s: 537 ms. A writer's fan-in would
        # stretch a dead peer's stall by TP, so it does not scale them.
        pool = _HostPool(sizes=(1 << 20,))
        writer = self._store(0, pool)
        unshared = KVCRStore(
            dataclasses.replace(self._config(0), extra_config={}), pool
        )
        with patch(
            "sglang.srt.mem_cache.storage.kvcr.kvcr_store._MIN_OPERATION_TIMEOUT_MS", 1
        ):
            self.assertEqual(unshared._timeouts(), (537, 30000))
            self.assertEqual(writer._timeouts(), (537, 30000))


class TestPortReservation(CustomTestCase):
    """Taken listen ports fail the attach, before the worker reports ready."""

    def setUp(self):
        self.enterContext(get_parallel().override(dp_rank=None, nnodes=1))
        self.enterContext(patch.dict("sys.modules", _fake_kvcr_modules()))

    def _store(self, **extra):
        control = find_available_port(36100)
        nixl = find_available_port(control + 1500)
        config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="ports",
            extra_config={
                "control_port": control,
                "nixl_listen_port": nixl,
                "control_host": "127.0.0.1",
                "local_dram_bytes": 256,
                **extra,
            },
        )
        store = KVCRStore(config, _HostPool())
        self.addCleanup(store.close)
        return store, control, nixl

    def _bindable(self, port, host=""):
        with socket.socket() as probe:
            try:
                probe.bind((host, port))
            except OSError:
                return False
            return True

    def test_taken_ports_fail_the_attach(self):
        for which in ("control", "nixl"):
            with self.subTest(which=which):
                store, control, nixl = self._store(enable_remote_hint=True)
                with socket.create_server(
                    (
                        "127.0.0.1" if which == "control" else "",
                        control if which == "control" else nixl,
                    )
                ):
                    # Fatal: one rank failing a runtime attach would strand
                    # the others in its collectives.
                    with (
                        patch(
                            "sglang.srt.mem_cache.storage.kvcr.kvcr_store.os._exit",
                            side_effect=SystemExit,
                        ) as exit_,
                        self.assertRaises(SystemExit),
                        self.assertLogs(
                            "sglang.srt.mem_cache.storage.kvcr.kvcr_store", "CRITICAL"
                        ),
                    ):
                        store.on_host_pools_registered()
                    exit_.assert_called_once_with(1)
                self.assertIsNone(store._listener)
                self.assertIsNone(store._nixl_hold)

    def test_reserved_ports_pass_to_kvcr_and_close_with_the_store(self):
        store, control, nixl = self._store(enable_remote_hint=True)
        store.on_host_pools_registered()
        # Held until KVCR binds them.
        self.assertFalse(self._bindable(control, "127.0.0.1"))
        self.assertFalse(self._bindable(nixl))
        store._start()
        channel = store._runner.bindings.framework_control
        self.assertIsNotNone(channel.adopted)
        self.assertIsNone(store._listener)
        self.assertIsNone(store._nixl_hold)
        self.assertTrue(self._bindable(nixl))
        store.close()
        # Unstarted stores release what they reserved.
        store, control, nixl = self._store()
        store.on_host_pools_registered()
        self.assertIsNone(store._listener)
        self.assertFalse(self._bindable(nixl))
        store.close()
        self.assertTrue(self._bindable(nixl))


class _RefusingChannel(_FakeChannel):
    def adopt_listener(self, listener_fd):
        socket.socket(fileno=listener_fd).close()
        raise OSError("cannot adopt")


class TestAdoptFailure(CustomTestCase):
    def test_adopt_failure_is_a_startup_failure(self):
        """Handled like the other startup failures, with no fd left behind."""
        self.enterContext(get_parallel().override(dp_rank=None, nnodes=1))
        modules = _fake_kvcr_modules()
        modules["kvcr"].ZmqPeerControlChannel = _RefusingChannel
        self.enterContext(patch.dict("sys.modules", modules))
        control = find_available_port(37100)
        config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="adopt",
            extra_config={
                "enable_remote_hint": True,
                "control_host": "127.0.0.1",
                "control_port": control,
                "nixl_listen_port": find_available_port(control + 1500),
                "local_dram_bytes": 256,
            },
        )
        store = KVCRStore(config, _HostPool())
        self.addCleanup(store.close)
        store.on_host_pools_registered()
        with (
            patch(
                "sglang.srt.mem_cache.storage.kvcr.kvcr_store.os._exit",
                side_effect=SystemExit,
            ) as exit_,
            self.assertRaises(SystemExit),
            self.assertLogs("sglang.srt.mem_cache.storage.kvcr.kvcr_store", "CRITICAL"),
        ):
            store._start()
        exit_.assert_called_once_with(1)
        self.assertIsNone(store._listener)
        self.assertIsNone(store._runner)


class TestCompletionPollThread(CustomTestCase):
    """_start, _poll_loop, _wait and close() with KVCR faked at its module."""

    def setUp(self):
        self.enterContext(get_parallel().override(dp_rank=None, nnodes=1))
        self.enterContext(patch.object(KVCRStore, "_reserve_ports"))
        self.enterContext(patch.dict("sys.modules", _fake_kvcr_modules()))
        self.exits = []
        self.enterContext(
            patch(
                "sglang.srt.mem_cache.storage.kvcr.kvcr_store.os._exit",
                side_effect=self.exits.append,
            )
        )
        config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="poll",
            extra_config={"local_dram_bytes": 256},
        )
        self.pool = _HostPool()
        self.store = KVCRStore(config, self.pool)
        self.addCleanup(self.store.close)
        self.keys = [f"{i + 1:064x}" for i in range(4)]

    def test_poll_thread_completes_concurrent_backup_and_restore(self):
        # Every completion arrives through the poll thread's handoff.
        self.assertEqual(
            self.store.batch_set_v1(self.keys[:2], torch.arange(4)), [True] * 2
        )
        thread = self.store._poll_thread
        self.assertTrue(thread.is_alive())
        results = Queue()

        def run(call):
            results.put(call())

        workers = [
            threading.Thread(
                target=run,
                args=(
                    lambda: self.store.batch_set_v1(self.keys[2:], torch.arange(4, 8)),
                ),
            ),
            threading.Thread(
                target=run,
                args=(
                    lambda: self.store.batch_get_v1(self.keys[:2], torch.arange(8, 12)),
                ),
            ),
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(10)
        self.assertEqual(
            sorted([results.get_nowait(), results.get_nowait()]), [[True] * 2] * 2
        )
        self.assertEqual(self.store._completed, {})
        runner = self.store._runner
        self.store.close()
        self.assertFalse(thread.is_alive())
        self.assertEqual(runner.closed, 1)
        self.assertEqual(self.store._maps, [])
        self.store.close()
        self.assertEqual(runner.closed, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.store._start()
        self.assertEqual(self.exits, [])

    def test_idle_poll_backs_off_without_peers(self):
        self.assertEqual(self.store._idle_poll_max, 0.1)
        with patch(
            "sglang.srt.mem_cache.storage.kvcr.kvcr_store._POLL_INTERVAL_SECONDS", 0.05
        ):
            self.store.batch_set_v1(self.keys[:1], torch.arange(2))
            polls = self.store._runner.polls
            time.sleep(0.5)
        # Doubling from 50 ms to the 100 ms cap: a handful of wakeups.
        self.assertLessEqual(self.store._runner.polls - polls, 8)

    def test_progress_failure_terminates_the_worker(self):
        self.store.batch_set_v1(self.keys[:1], torch.arange(2))
        self.store._runner.fail = True
        deadline = time.monotonic() + 5
        while not self.exits and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.exits[:1], [1])

    def test_stalled_source_is_an_error_not_an_exit(self):
        logger = "sglang.srt.mem_cache.storage.kvcr.kvcr_store"
        with self.assertLogs(logger, "ERROR") as logs:
            KVCRStore._resilience_event(
                RuntimeError(
                    "KVCR source progress stalled for 6000 ms; "
                    "new source writes disabled until restart"
                )
            )
        self.assertIn("stopped serving peers", logs.output[0])
        self.assertEqual(self.exits, [])


_REAL_KVCR = sys.platform == "linux" and all(
    importlib.util.find_spec(name) is not None for name in ("kvcr", "nixl")
)


@unittest.skipUnless(_REAL_KVCR, "needs kvcr>=0.2.0 and NIXL on Linux")
class TestRealKVCR(CustomTestCase):
    """The KVCR contract the fakes above stand in for, on the installed package."""

    def test_installed_package_matches_the_store_contract(self):
        from kvcr import ZmqPeerControlChannel
        from kvcr.config import KVCRConfig

        self.assertIn("startup_timeout_seconds", KVCRConfig.__struct_fields__)
        self.assertTrue(hasattr(ZmqPeerControlChannel, "adopt_listener"))

    def test_backup_query_restore_round_trip_with_remote_hints(self):
        exits = []

        def fail_instead_of_exit(code):
            # Also reached from the poll thread, where a raise alone is lost.
            exits.append(code)
            raise RuntimeError(f"KVCR store called os._exit({code})")

        # Cleanups run last-in first-out: this check runs after close.
        self.addCleanup(lambda: self.assertEqual(exits, []))
        self.enterContext(
            patch(
                "sglang.srt.mem_cache.storage.kvcr.kvcr_store.os._exit",
                side_effect=fail_instead_of_exit,
            )
        )
        self.enterContext(get_parallel().override(dp_rank=None, nnodes=1))
        control_port = find_available_port(33100)
        config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="kvcr-real",
            extra_config={
                "enable_remote_hint": True,
                # Advertised only (TEST-NET-1); nothing dials it here.
                "control_advertise_host": "192.0.2.1",
                "control_host": "127.0.0.1",
                "control_port": control_port,
                "nixl_listen_port": find_available_port(control_port + 2000),
                "local_dram_bytes": 1 << 20,
                "startup_timeout_seconds": 60,
            },
        )
        pool = _HostPool()
        backend = KVCRStore(config, pool)
        self.addCleanup(backend.close)
        keys = [f"{i + 1:064x}" for i in range(2)]
        generator = torch.Generator().manual_seed(0)
        for buffer in pool.buffers:
            buffer[:2].copy_(
                torch.randint(
                    0, 256, buffer[:2].shape, dtype=torch.uint8, generator=generator
                )
            )
        # The first IO starts real KVCR: registration, binds, local DRAM deposit.
        self.assertEqual(backend.batch_set_v1(keys, torch.arange(4)), [True] * 2)
        self.assertIsNotNone(backend._runner)
        # Real query statuses, then a local restore into other slots.
        self.assertEqual(backend.batch_exists_v2(keys).kv_hit_pages, 2)
        self.assertEqual(backend.batch_get_v1(keys, torch.arange(8, 12)), [True] * 2)
        for buffer in pool.buffers:
            self.assertTrue(torch.equal(buffer[:2], buffer[4:6]))

    def test_local_only_round_trip_without_a_control_endpoint(self):
        """Remote hints off: KVCR runs with no control channel at all."""
        self.enterContext(get_parallel().override(dp_rank=None, nnodes=1))
        self.enterContext(
            patch(
                "sglang.srt.mem_cache.storage.kvcr.kvcr_store.os._exit",
                side_effect=lambda code: self.fail(f"os._exit({code})"),
            )
        )
        control_port = find_available_port(34100)
        config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="kvcr-real-local",
            extra_config={
                "control_port": control_port,
                "nixl_listen_port": find_available_port(control_port + 2000),
                "local_dram_bytes": 1 << 20,
            },
        )
        pool = _HostPool()
        backend = KVCRStore(config, pool)
        self.addCleanup(backend.close)
        keys = [f"{i + 1:064x}" for i in range(2)]
        for buffer in pool.buffers:
            buffer[:2].copy_(torch.randint(0, 256, buffer[:2].shape, dtype=torch.uint8))
        self.assertEqual(backend.batch_set_v1(keys, torch.arange(4)), [True] * 2)
        self.assertEqual(backend.batch_exists_v2(keys).kv_hit_pages, 2)
        self.assertEqual(backend.batch_get_v1(keys, torch.arange(8, 12)), [True] * 2)
        for buffer in pool.buffers:
            self.assertTrue(torch.equal(buffer[:2], buffer[4:6]))
        # Nothing listens on the control port.
        with socket.socket() as probe:
            self.assertNotEqual(probe.connect_ex(("127.0.0.1", control_port)), 0)

    def _shared_pair(self, model_name, *, pools=()):
        """A TP2 MLA replica's writer and second rank, on real KVCR and NIXL."""
        self.enterContext(get_parallel().override(dp_rank=None, nnodes=1))
        exits = []

        def fail_instead_of_exit(code):
            # Also reached from the poll thread, where a raise alone is lost.
            exits.append(code)
            raise RuntimeError(f"KVCR store called os._exit({code})")

        # Cleanups run last-in first-out: this check runs after the closes.
        self.addCleanup(lambda: self.assertEqual(exits, []))
        self.enterContext(
            patch(
                "sglang.srt.mem_cache.storage.kvcr.kvcr_store.os._exit",
                side_effect=fail_instead_of_exit,
            )
        )
        # Both ranks' ports: base and base + 1, free in both ranges.
        while True:
            control_port = find_available_port(35100)
            nixl_port = find_available_port(control_port + 2000)
            ports = (control_port, control_port + 1, nixl_port, nixl_port + 1)
            if all(is_port_available(port) for port in ports):
                break
        stores = []
        for tp_rank in (0, 1):
            config = HiCacheStorageConfig(
                tp_rank=tp_rank,
                tp_size=2,
                pp_rank=0,
                pp_size=1,
                attn_cp_rank=0,
                attn_cp_size=1,
                is_mla_model=True,
                enable_storage_metrics=False,
                is_page_first_layout=True,
                model_name=model_name,
                extra_config={
                    "share_replicated_kv": True,
                    "control_port": control_port,
                    "nixl_listen_port": nixl_port,
                    "local_dram_bytes": 1 << 20,
                    "startup_timeout_seconds": 60,
                },
            )
            store = KVCRStore(config, _HostPool())
            self.addCleanup(store.close)
            for name, sizes in pools:
                store.register_mem_host_pool_v2(_HostPool(sizes=sizes), name)
            # Binds (and so verifies) both listen ports, as an attach would.
            store.on_host_pools_registered()
            stores.append(store)
        return stores

    def _restore_from_the_writer(self, writer, replica):
        keys = [f"{i + 1:016x}" * 4 for i in range(2)]
        for buffer in writer.mem_pool_host.buffers:
            buffer[:2].copy_(torch.randint(0, 256, buffer[:2].shape, dtype=torch.uint8))
        self.assertEqual(writer.batch_set_v1(keys, torch.arange(4)), [True] * 2)
        # The writer answers lookups; the replica defers to it.
        self.assertEqual(writer.batch_exists(keys), 2)
        self.assertEqual(replica.batch_exists(keys), 2)
        self.assertEqual(replica.batch_get_v1(keys, torch.arange(8, 12)), [True] * 2)
        for source, restored in zip(
            writer.mem_pool_host.buffers, replica.mem_pool_host.buffers
        ):
            self.assertTrue(torch.equal(source[:2], restored[4:6]))
        # A page the writer never stored is a clean miss, not an exit.
        missing = [f"{9:016x}" * 4]
        self.assertEqual(replica.batch_get_v1(missing, torch.arange(2)), [False])
        return keys

    def test_shared_replica_restores_from_its_writer_without_dram(self):
        """Pure MLA: the replica holds no KVCR DRAM and no remote hints run."""
        writer, replica = self._shared_pair("kvcr-real-shared")
        self._restore_from_the_writer(writer, replica)
        self.assertEqual(replica._maps, [])
        self.assertEqual(len(writer._maps), 2)
        # Only this host's ranks dial the control endpoints, on loopback.
        self.assertTrue(replica._writer_endpoint.startswith("tcp://127.0.0.1:"))

    def test_shared_replica_keeps_its_own_rank_sharded_pools(self):
        """KV shared on the writer; each rank's Mamba checkpoints its own."""
        writer, replica = self._shared_pair(
            "kvcr-real-shared-mamba", pools=[(PoolName.MAMBA, (5,))]
        )
        keys = self._restore_from_the_writer(writer, replica)
        # KVCR's DRAM pools must match its layouts: one shared page each.
        self.assertEqual(len(replica._maps), 3)
        mamba = replica.registered_pools["mamba"]
        mamba.buffers[0][1].fill_(99)
        self.assertEqual(
            replica.batch_set_v2(
                [
                    PoolTransfer(
                        PoolName.MAMBA, keys=keys[1:], host_indices=torch.arange(2, 4)
                    )
                ]
            ),
            {"mamba": [True]},
        )
        result = replica.batch_exists_v2(
            keys,
            [
                PoolTransfer(
                    PoolName.MAMBA,
                    keys=keys[-1:],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ],
        )
        self.assertEqual(result.restorable_prefix_pages, [2])
        self.assertEqual(
            replica.batch_get_v2(
                [
                    PoolTransfer(PoolName.KV, keys=keys, host_indices=torch.arange(4)),
                    PoolTransfer(
                        PoolName.MAMBA, keys=keys[1:], host_indices=torch.arange(6, 8)
                    ),
                ]
            ),
            {"kv": [True, True], "mamba": [True]},
        )
        self.assertTrue(torch.all(mamba.buffers[0][3] == 99))


if __name__ == "__main__":
    unittest.main()
