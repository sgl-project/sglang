"""CPU contracts for indexed HiCache IO; native NIXL is a separate integration gate."""

import copy
import ctypes
import json
import mmap
import sys
import types
import unittest
from array import array
from contextlib import ExitStack, nullcontext
from enum import Enum
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch

from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.buffer_mode.pipeline import BufferModePipeline
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
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.kvcr.diagnostics import DiagnosticStats
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
)
from sglang.srt.mem_cache.storage_key_binding import storage_key_to_event_hashes
from sglang.srt.mem_cache.utils import (
    compute_node_event_hash_values,
    get_event_hash_str,
    get_hash_str,
    get_storage_hash_str,
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


class _StrictPeerTransport(_CopyTransport):
    """Native-style hint membership at the external dependency boundary."""

    def __init__(self, layouts, peers):
        super().__init__(layouts)
        self.peers, self.reads = peers, []
        self.claims, self.fetches = {}, []
        self.next_claim = 0

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

    def fetch(self, keys, request_id=None, expected_layout=None):
        """Model a physical source check and a held local copy, unlike query."""
        self.fetches.append(tuple(keys))
        available = {**self._authorized(request_id), **self.storage}
        result = {}
        for key in keys:
            data = available.get(key)
            if data is None or not all(
                label in data for label in expected_layout or ()
            ):
                result[key] = SimpleNamespace(success=False, release_handle=None)
                continue
            self.next_claim += 1
            self.claims[self.next_claim] = key
            self.storage[key] = copy.deepcopy(data)
            result[key] = SimpleNamespace(success=True, release_handle=self.next_claim)
        return self._complete(result)

    def release(self, handles):
        released = []
        for handle in handles:
            key = self.claims.pop(handle, None)
            if key is not None and key not in self.claims.values():
                # A second operation can still hold the same native key.
                # Remove its local copy only after the final claim releases.
                self.storage.pop(key, None)
            released.append((handle, key is not None))
        return released


class _HintSupersetTransport(_StrictPeerTransport):
    """A logical-page hint can advertise absent Mamba state as FETCHABLE."""

    def query(self, keys, request_id=None):
        hint = self.hints.get(request_id)
        hashes = (
            set(hint["actions"][0]["payload"]["block_hashes"])
            if hint is not None
            else set()
        )
        return [
            (
                _QueryStatus.HIT
                if key in self.storage
                else _QueryStatus.FETCHABLE
                if int(key[:16], 16) in hashes
                else _QueryStatus.MISS,
                None,
            )
            for key in keys
        ]


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

    def test_hint_source_uses_backend_resolved_address_without_starting_io(self):
        config = copy.copy(self.config)
        config.dp_rank, config.attn_cp_size = 2, 2
        config.extra_config = {
            "control_advertise_host": "2001:db8::1",
            "control_port": 29017,
        }
        backend = KVCRStore(config, self.pool)
        self.addCleanup(backend.close)
        source = backend.get_kv_hint_source()
        self.assertEqual(
            source,
            {
                "dp_rank": 2,
                "capabilities": ["router_hint"],
                "source_control_endpoint": "tcp://[2001:db8::1]:29025",
            },
        )
        host, port = control_address(config, backend._extra)
        self.assertEqual(source["source_control_endpoint"], f"tcp://{host}:{port}")
        self.assertIsNone(backend._runner)
        backend._extra["enable_remote_hint"] = False
        self.assertIsNone(backend.get_kv_hint_source())
        backend._extra.pop("enable_remote_hint")
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
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                config = copy.copy(self.config)
                config.dp_rank, config.extra_config = 1, extra
                backend = KVCRStore(config, self.pool)
                self.addCleanup(backend.close)
                backend.get_kv_hint_source()

    def test_confirmed_mamba_boundary_is_not_shortened_by_host_capacity(self):
        """A full-boundary Mamba claim cannot validate a smaller KV staging cut.

        Under host pressure the smaller boundary may lack a checkpoint, as in
        the unsalted GLM-5.3-Flash AgentX replay. Decline that partial hit;
        ordinary non-trailing KV hits may still use partial staging.
        """
        controller = object.__new__(HybridCacheController)
        controller.storage_backend = self.backend
        controller.page_size = 2
        controller.mem_pool_host = SimpleNamespace(
            anchor_entry=SimpleNamespace(host_pool=self.pool),
            available_size=lambda: 8,
        )
        partial_indices = torch.arange(8)

        def allocate(_operation, tokens):
            return partial_indices if tokens == 8 else None

        with (
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller."
                "uses_shared_host_layout",
                return_value=False,
            ),
            patch.object(
                controller, "alloc_prefetch_host_buffers", side_effect=allocate
            ),
        ):
            trailing = SimpleNamespace(
                pool_transfers=[
                    PoolTransfer(
                        PoolName.MAMBA,
                        keys=["checkpoint"],
                        hit_policy=PoolHitPolicy.TRAILING_PAGES,
                    )
                ]
            )
            indices, span = controller.allocate_storage_hit(
                trailing,
                12,
                allow_partial=True,
                min_tokens=2,
                evict_host=lambda _tokens: 0,
            )
            self.assertIsNone(indices)
            self.assertEqual(span, 12)

            ordinary = SimpleNamespace(pool_transfers=[])
            indices, span = controller.allocate_storage_hit(
                ordinary,
                12,
                allow_partial=True,
                min_tokens=2,
                evict_host=lambda _tokens: 0,
            )
            self.assertIs(indices, partial_indices)
            self.assertEqual(span, 8)

            # Checkpoint confirmation is a KVCR opt-in. Other backends must
            # retain their existing partial host-staging behavior even when
            # the operation has a trailing-state transfer.
            controller.storage_backend = SimpleNamespace(
                requires_lookup_confirmation=False
            )
            indices, span = controller.allocate_storage_hit(
                trailing,
                12,
                allow_partial=True,
                min_tokens=2,
                evict_host=lambda _tokens: 0,
            )
            self.assertIs(indices, partial_indices)
            self.assertEqual(span, 8)

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

    def test_peer_binding_matches_event_chain_with_prefix_and_bigram(self):
        for salt, adapter in (("tenant-a", None), ("租户-α", "adapter"), (None, "a")):
            for bigram in (False, True):
                with self.subTest(salt=salt, adapter=adapter, bigram=bigram):
                    key = RadixKey(
                        array("q", range(1, 10 if bigram else 9)),
                        extra_key=adapter,
                        cache_salt=salt,
                        is_bigram=bigram,
                    )
                    storage = get_storage_hash_str(key, page_size=2)
                    parent = SimpleNamespace(
                        parent=None,
                        key=key[:4],
                        hash_value=storage[:2],
                        event_hash_value=None,
                    )
                    child = SimpleNamespace(
                        parent=parent,
                        key=key[4:],
                        hash_value=storage[2:],
                        event_hash_value=None,
                    )
                    events = get_event_hash_str(key, page_size=2)
                    self.assertEqual(
                        compute_node_event_hash_values(child, 2), events[2:]
                    )
                    mapping = storage_key_to_event_hashes(
                        key[4:], 2, list(key.raw_token_ids()[:4]), storage[1]
                    )
                    self.assertEqual(
                        mapping,
                        dict(zip(storage[2:], (int(h[:16], 16) for h in events[2:]))),
                    )
                    self.assertEqual(
                        storage_key_to_event_hashes(key[4:], 2, last_hash=storage[1]),
                        {},
                    )
                    self.assertEqual(
                        storage_key_to_event_hashes(key[4:], 2, [1, 2, 3], storage[1]),
                        {},
                    )
                    self.assertEqual(
                        storage_key_to_event_hashes(
                            key[4:], 2, [1, 2, 3, 4], "00" * 32
                        ),
                        {},
                    )
        for salt in (None, ""):
            self.assertIsNone(
                storage_key_to_event_hashes(
                    RadixKey(array("q", range(8)), cache_salt=salt), 2
                )
            )

    def test_peer_hint_translates_only_bound_requested_pages(self):
        key = RadixKey(array("q", range(8)), cache_salt="tenant-a")
        mapping = storage_key_to_event_hashes(key, 2)
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

    def test_salted_peer_prefetch_restores_kv_indexer_and_kda_bytes(self):
        """Separate owners, strict advertised membership, actual continuation bytes."""
        self.backend = KVCRStore(self.config, self.pool)
        source_pools = {PoolName.KV: self.pool}
        for name, sizes in ((PoolName.INDEXER, (5,)), (PoolName.MAMBA, (9, 3))):
            source_pools[name] = _HostPool(sizes=sizes)
            self.backend.register_mem_host_pool_v2(source_pools[name], name)
        self.backend._namespace = storage_namespace(self.config, self.backend._layouts)
        source = self.backend._runner = _CopyTransport(self.backend._layouts)
        key = RadixKey(array("q", range(8)), cache_salt="tenant-a")
        hashes = get_storage_hash_str(key, page_size=2)

        def transfers(pools):
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
        self.assertTrue(
            all(
                all(flags)
                for flags in self.backend.batch_set_v2(transfers(source_pools)).values()
            )
        )
        destination_pools = {
            name: _HostPool(sizes=tuple(b.shape[1] for b in pool.buffers))
            for name, pool in source_pools.items()
        }
        reader = KVCRStore(self.config, destination_pools[PoolName.KV])
        for name, pool in destination_pools.items():
            if name != PoolName.KV:
                reader.register_mem_host_pool_v2(pool, name)
        reader._namespace = storage_namespace(self.config, reader._layouts)
        transport = reader._runner = _StrictPeerTransport(
            reader._layouts, {"tcp://peer:32100": source.storage}
        )
        reader.prefetch_batch_pages = 2
        c = HybridCacheController.__new__(HybridCacheController)
        c.page_size, c.storage_backend = 2, reader
        c.page_get_func = c._page_get_zero_copy
        c.pp_prefetch_command_group = None
        c.prefetch_queue, c.prefetch_sync_queue = Queue(), Queue()
        c.mem_pool_host = SimpleNamespace(entry_map={})
        hint = {
            "protocol_version": "0.1",
            "actions": [
                {
                    "action_type": "kv.fetch",
                    "action_version": "1.0",
                    "payload": {
                        "source_control_endpoint": "tcp://peer:32100",
                        "block_hashes": [
                            int(h[:16], 16)
                            for h in get_event_hash_str(key, page_size=2)
                        ],
                    },
                }
            ],
        }
        operation = c.submit_prefetch(
            CacheRequestHandle("peer", 0),
            key,
            None,
            None,
            [],
            transfers(destination_pools)[1:],
            kv_hints=hint,
        ).operation
        self.assertIs(c.prefetch_queue.get_nowait(), operation)
        operation.hash_value, count = c._storage_hit_query(operation)
        self.assertEqual(count, 8)
        operation.host_indices = torch.arange(8)
        self.assertEqual(HiCacheController._page_transfer(c, operation), 4)
        c._page_transfer_sidecar(operation, 4)
        self.assertEqual(c.prefetch_sync_queue.get_nowait().completed_tokens, 4)
        self.assertEqual(c.prefetch_sync_queue.get_nowait().completed_tokens, 8)
        self.assertEqual(
            c.prefetch_sync_queue.get_nowait().pool_hits[PoolName.MAMBA], 1
        )
        for name, pool in source_pools.items():
            for expected, actual in zip(pool.buffers, destination_pools[name].buffers):
                self.assertTrue(
                    torch.equal(
                        expected[: 1 if name == PoolName.MAMBA else 4],
                        actual[: 1 if name == PoolName.MAMBA else 4],
                    )
                )
        self.assertFalse(transport.storage)  # All bytes came from the other owner.
        self.assertFalse(transport.hints)  # No request authorization leaked.
        other = RadixKey(array("q", range(8)), cache_salt="tenant-b")
        denied = c.submit_prefetch(
            CacheRequestHandle("other", 0), other, None, None, [], None, kv_hints=hint
        ).operation
        self.assertEqual(c._storage_hit_query(denied)[1], 0)
        before = {
            name: [buffer.clone() for buffer in pool.buffers]
            for name, pool in destination_pools.items()
        }
        foreign_hashes = get_storage_hash_str(other, page_size=2)
        foreign_transfers = transfers(destination_pools)
        for transfer in foreign_transfers:
            transfer.keys = (
                foreign_hashes[-1:]
                if transfer.name == PoolName.MAMBA
                else foreign_hashes
            )
        missed = reader.batch_get_v2(
            foreign_transfers,
            extra_info=HiCacheStorageExtraInfo(
                extra_info={
                    "kv_hints": hint,
                    "storage_key_to_event_hash": denied.storage_key_to_event_hash,
                }
            ),
        )
        self.assertFalse(any(any(flags) for flags in missed.values()))
        for name, pool in destination_pools.items():
            for prior, actual in zip(before[name], pool.buffers):
                self.assertTrue(torch.equal(prior, actual))

    def test_advisory_mamba_hint_requires_source_checkpoint_and_held_copy(self):
        """A logical-page hint must not admit absent Mamba, and a confirmed
        checkpoint must survive source removal until host delivery completes.
        A cancelled queued delivery must still release that held checkpoint.
        Failed rank consensus must not leave a confirmed checkpoint claimed.
        """
        self.backend._runner = None
        source_pools = {
            PoolName.KV: self.pool,
            PoolName.INDEXER: _HostPool(sizes=(5,)),
            PoolName.MAMBA: _HostPool(sizes=(9, 3)),
        }
        for name in (PoolName.INDEXER, PoolName.MAMBA):
            self.backend.register_mem_host_pool_v2(source_pools[name], name)
        self.backend._namespace = storage_namespace(self.config, self.backend._layouts)
        source = self.backend._runner = _CopyTransport(self.backend._layouts)
        key = RadixKey(array("q", range(8)))
        hashes = get_storage_hash_str(key, page_size=2)
        for number, pool in enumerate(source_pools.values()):
            for component, buffer in enumerate(pool.buffers):
                buffer[:4].fill_(19 + number * 11 + component)
        self.backend.batch_set_v2(
            [
                PoolTransfer(name, keys=hashes, host_indices=torch.arange(8))
                for name in (PoolName.KV, PoolName.INDEXER)
            ]
            + [
                PoolTransfer(
                    PoolName.MAMBA,
                    keys=hashes[-1:],
                    host_indices=torch.arange(2),
                )
            ]
        )

        destination_pools = {
            name: _HostPool(sizes=tuple(b.shape[1] for b in pool.buffers))
            for name, pool in source_pools.items()
        }
        reader = KVCRStore(self.config, destination_pools[PoolName.KV])
        for name in (PoolName.INDEXER, PoolName.MAMBA):
            reader.register_mem_host_pool_v2(destination_pools[name], name)
        reader._namespace = storage_namespace(self.config, reader._layouts)
        transport = reader._runner = _HintSupersetTransport(
            reader._layouts, {"tcp://peer:32100": source.storage}
        )
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.page_size, controller.storage_backend = 2, reader
        controller.page_get_func = controller._page_get_zero_copy
        controller.prefetch_queue, controller.prefetch_sync_queue = Queue(), Queue()
        controller.prefetch_buffer = Queue()
        controller.pp_prefetch_command_group = None
        controller.prefetch_hits_sync_groups = []
        controller._all_reduce = lambda *_args, **_kwargs: None
        controller.mem_pool_host = SimpleNamespace(
            entry_map={}, layout_lease=lambda: nullcontext()
        )
        hint = {
            "protocol_version": "0.1",
            "actions": [
                {
                    "action_type": "kv.fetch",
                    "action_version": "1.0",
                    "payload": {
                        "source_control_endpoint": "tcp://peer:32100",
                        "block_hashes": [int(h[:16], 16) for h in hashes],
                    },
                }
            ],
        }

        def submit(end_pages):
            end_hashes = hashes[:end_pages]
            transfers = [
                PoolTransfer(
                    PoolName.INDEXER,
                    keys=end_hashes,
                    host_indices=torch.arange(8, 8 + end_pages * 2),
                    indices_from_pool=PoolName.KV,
                ),
                PoolTransfer(
                    PoolName.MAMBA,
                    keys=end_hashes[-1:],
                    host_indices=torch.arange(8, 10),
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                ),
            ]
            operation = controller.submit_prefetch(
                CacheRequestHandle(f"boundary-{end_pages}", 0),
                key[: end_pages * 2],
                None,
                None,
                [],
                transfers,
                kv_hints=hint,
            ).operation
            self.assertIs(controller.prefetch_queue.get_nowait(), operation)
            hashes_found, advisory_tokens = controller._storage_hit_query(operation)
            self.assertEqual(hashes_found, end_hashes)
            self.assertEqual(advisory_tokens, end_pages * 2)
            return operation, controller._sync_prefetch_hit_query(
                operation, advisory_tokens
            )

        middle, confirmed = submit(2)
        self.assertEqual(confirmed, 0)
        self.assertEqual(middle.pool_storage_result.restorable_prefix_pages, [])
        self.assertEqual(middle.query_pool_hit_pages[PoolName.MAMBA], 0)
        self.assertFalse(transport.claims)

        full, confirmed = submit(4)
        self.assertEqual(confirmed, 8)
        self.assertEqual(full.query_pool_hit_pages[PoolName.MAMBA], 4)
        self.assertEqual(len(transport.claims), 1)
        self.assertEqual(
            transport.fetches[0][-1], block_key(hashes[1], reader._namespace, "mamba")
        )
        # A rematch with the same RID must hold a separate operation-scoped
        # claim; finishing the first delivery cannot release the second.
        overlap, overlap_confirmed = submit(4)
        self.assertEqual(overlap_confirmed, 8)
        self.assertEqual(overlap.request_id, full.request_id)
        self.assertNotEqual(overlap.storage_lookup_scope, full.storage_lookup_scope)
        self.assertEqual(len(transport.claims), 2)
        # A hit parked by host pressure has not been delivered yet. Releasing
        # its source-backed checkpoint here would let source eviction turn the
        # later retry into a KV-only continuation.
        pipeline = object.__new__(BufferModePipeline)
        pipeline._cache = SimpleNamespace(
            cache_controller=controller,
            prefetch_threshold=2,
            evict_host=lambda _tokens: 0,
            _log_storage_prefetch_deferred=lambda *_args: None,
        )
        with (
            patch.object(
                controller, "can_fit_prefetch_host_buffers", return_value=True
            ),
            patch.object(controller, "allocate_storage_hit", return_value=(None, 8)),
            patch.object(pipeline, "release_anchor_lock"),
        ):
            self.assertFalse(
                pipeline.allocate_and_queue_prefetch(
                    full, SimpleNamespace(), hit_tokens=8, aux_hit_tokens=8
                )
            )
        self.assertFalse(full.storage_transfer_enqueued)
        self.assertEqual(len(transport.claims), 2)
        mamba_key = block_key(hashes[-1], reader._namespace, "mamba")
        source.storage.pop(mamba_key)
        full.hash_value = hashes
        full.host_indices = torch.arange(8, 16)
        controller.enqueue_prefetch_transfer(full)
        controller.release_unqueued_prefetch_claims(full)
        self.assertEqual(len(transport.claims), 2)
        controller._page_transfer(full)
        self.assertEqual(len(transport.claims), 1)
        self.assertEqual(set(reader._lookup_claims), {overlap.storage_lookup_scope})
        overlap.hash_value = hashes
        overlap.host_indices = torch.arange(8, 16)
        controller.enqueue_prefetch_transfer(overlap)
        controller.release_unqueued_prefetch_claims(overlap)
        self.assertEqual(len(transport.claims), 1)
        controller._page_transfer(overlap)
        self.assertFalse(transport.claims)
        self.assertFalse(reader._lookup_claims)
        self.assertTrue(
            torch.equal(
                source_pools[PoolName.MAMBA].buffers[0][0],
                destination_pools[PoolName.MAMBA].buffers[0][4],
            )
        )

        # An advisory full-chain hit can still fall back to a shorter source
        # checkpoint. Only the physically confirmed boundary may be admitted.
        for component, buffer in enumerate(source_pools[PoolName.MAMBA].buffers):
            buffer[0].fill_(73 + component)
        self.assertEqual(
            self.backend.batch_set_v2(
                [
                    PoolTransfer(
                        PoolName.MAMBA,
                        keys=hashes[1:2],
                        host_indices=torch.arange(2),
                    )
                ]
            )[PoolName.MAMBA],
            [True],
        )
        fallback, confirmed = submit(4)
        self.assertEqual(confirmed, 4)
        self.assertEqual(fallback.pool_storage_result.restorable_prefix_pages, [2])
        self.assertEqual(fallback.query_pool_hit_pages[PoolName.MAMBA], 2)
        self.assertEqual(len(transport.claims), 1)
        fallback.hash_value = hashes[:2]
        fallback.host_indices = torch.arange(8, 12)
        controller._page_transfer(fallback)
        self.assertFalse(transport.claims)
        self.assertTrue(
            torch.equal(
                source_pools[PoolName.MAMBA].buffers[0][0],
                destination_pools[PoolName.MAMBA].buffers[0][4],
            )
        )

        # A device-resident KV head can be trimmed after physical confirmation.
        # The trailing state still belongs to the confirmed prefix endpoint,
        # not to the narrowed KV transfer's first page or the advisory full hit.
        for trim_tokens in (2, 4):
            with self.subTest(trim_tokens=trim_tokens):
                trimmed, confirmed = submit(4)
                self.assertEqual(confirmed, 4)
                trimmed.hash_value = hashes[:2]
                trimmed.storage_hit_count = confirmed
                trimmed.storage_start = 0
                controller.trim_prefetch_full_head(trimmed, trim_tokens)
                trimmed.host_indices = torch.arange(8, 8 + confirmed - trim_tokens)
                self.assertEqual(trimmed.hash_value, hashes[trim_tokens // 2 : 2])
                self.assertEqual(trimmed.sidecar_hash_values, hashes[:2])
                self.assertEqual(trimmed.sidecar_hit_pages, 2)
                for buffer in destination_pools[PoolName.MAMBA].buffers:
                    buffer[4].zero_()
                reads_before = len(transport.reads)
                controller._page_transfer(trimmed)
                self.assertFalse(transport.claims)
                self.assertEqual(trimmed.pool_transfers[1].keys, hashes[1:2])
                self.assertTrue(
                    any(
                        block_key(hashes[1], reader._namespace, "mamba") in group
                        for group in transport.reads[reads_before:]
                    )
                )
                for source_buffer, destination_buffer in zip(
                    source_pools[PoolName.MAMBA].buffers,
                    destination_pools[PoolName.MAMBA].buffers,
                    strict=True,
                ):
                    self.assertTrue(
                        torch.equal(source_buffer[0], destination_buffer[4])
                    )

        dropped, confirmed = submit(4)
        self.assertEqual(confirmed, 4)
        self.assertEqual(len(transport.claims), 1)
        controller.release_unqueued_prefetch_claims(dropped)
        self.assertFalse(transport.claims)
        self.assertFalse(reader._lookup_claims)

        # A queue failure after confirmation must not strand the source pin.
        rejected, confirmed = submit(4)
        self.assertEqual(confirmed, 4)
        self.assertEqual(len(transport.claims), 1)
        with (
            patch.object(
                controller.prefetch_buffer,
                "put",
                side_effect=RuntimeError("prefetch queue unavailable"),
            ),
            self.assertRaisesRegex(RuntimeError, "prefetch queue unavailable"),
        ):
            controller.enqueue_prefetch_transfer(rejected)
        self.assertFalse(rejected.storage_transfer_enqueued)
        self.assertFalse(transport.claims)
        self.assertFalse(reader._lookup_claims)

        # Delivery can fail after the transfer has been queued. Its finally
        # path must still release the physical Mamba copy held by lookup.
        failed_transfer, confirmed = submit(4)
        self.assertEqual(confirmed, 4)
        self.assertEqual(len(transport.claims), 1)
        controller.enqueue_prefetch_transfer(failed_transfer)
        with (
            patch.object(
                controller,
                "_page_transfer_with_stable_layout",
                side_effect=RuntimeError("host transfer failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "host transfer failed"),
        ):
            controller._page_transfer(failed_transfer)
        self.assertFalse(transport.claims)
        self.assertFalse(reader._lookup_claims)

        # Cancellation after enqueue still reaches the IO worker. It must
        # release the source-backed claim even though no pool is delivered.
        cancelled, confirmed = submit(4)
        self.assertEqual(confirmed, 4)
        cancelled.hash_value = hashes[:2]
        cancelled.host_indices = torch.arange(8, 12)
        controller.enqueue_prefetch_transfer(cancelled)
        self.assertEqual(len(transport.claims), 1)
        cancelled.mark_terminate()
        reads_before = len(transport.reads)
        controller._page_transfer(cancelled)
        self.assertEqual(len(transport.reads), reads_before)
        self.assertFalse(transport.claims)
        self.assertFalse(reader._lookup_claims)

        # A rank that lacks the same trailing-pool requirement must not let
        # another rank admit the advisory boundary without confirmation.
        fetches_before = len(transport.fetches)

        def disagree_on_confirmation_flags(tensor, *_args, **_kwargs):
            if tensor.numel() == len(PoolName) + 5:
                tensor[-2:] = 0

        controller._all_reduce = disagree_on_confirmation_flags
        mixed, confirmed = submit(4)
        self.assertEqual(confirmed, 0)
        self.assertEqual(mixed.pool_storage_result.restorable_prefix_pages, [])
        self.assertEqual(len(transport.fetches), fetches_before)

        controller._all_reduce = lambda *_args, **_kwargs: None
        with patch.object(transport, "fetch", side_effect=RuntimeError("source lost")):
            failed, confirmed = submit(4)
        self.assertEqual(confirmed, 0)
        self.assertEqual(failed.query_pool_hit_pages[PoolName.MAMBA], 0)
        self.assertFalse(transport.claims)

        def fail_after_checkpoint_fetch(tensor, *_args, **_kwargs):
            if tensor.numel() == 2 and transport.claims:
                raise RuntimeError("rank consensus failed after Mamba fetch")

        with (
            patch.object(
                controller, "_all_reduce", side_effect=fail_after_checkpoint_fetch
            ),
            self.assertRaisesRegex(RuntimeError, "rank consensus failed"),
        ):
            submit(4)
        self.assertFalse(transport.claims)
        self.assertFalse(reader._lookup_claims)

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

    def _controller_with_indexer(self, *, combined=True, pages=4, batch_pages=128):
        config = SimpleNamespace(**vars(self.config))
        config.extra_config = {
            "combine_page_reads": combined,
            "prefetch_batch_pages": batch_pages,
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


if __name__ == "__main__":
    unittest.main()
