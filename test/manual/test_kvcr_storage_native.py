"""Manual real-NIXL gate for the HiCacheStorage backend (no model required).

Requires Linux, KVCR, NIXL/UCX and Torch. This intentionally does not substitute
the transport: two independent KVCR agents own their DRAM and exchange bytes.
Run: python test/manual/test_kvcr_storage_native.py -v
"""

import hashlib
import itertools
import json
import socket
import unittest
from contextlib import ExitStack
from copy import deepcopy
from queue import Queue
from types import SimpleNamespace

import torch

from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    StorageOperation,
)
from sglang.srt.mem_cache.storage.kvcr.kvcr_store import KVCRStore
from sglang.test.test_utils import CustomTestCase


class HostPages:
    """Stable page-first CPU buffers with unequal-sized page components."""

    def __init__(self, sizes, page_size=2, pages=16):
        self.page_size, self.size = page_size, pages * page_size
        self.layout, self.dtype = "page_first", torch.uint8
        self.buffers = [torch.zeros((pages, size), dtype=self.dtype) for size in sizes]
        self.kv_buffer = self.buffers[0]

    def get_page_buffer_meta(self, indices):
        pointers, sizes = [], []
        for start in indices.tolist()[:: self.page_size]:
            for buffer in self.buffers:
                pointers.append(buffer[start // self.page_size].data_ptr())
                sizes.append(buffer.shape[1])
        return pointers, sizes


# Port zero selects the outbound ephemeral range. NIXL/UCX can consume a
# previously probed port while initializing, before the control listener binds.
# This manual Linux fixture uses unique non-ephemeral ports in its own process;
# bind probes are still advisory, not a general cross-process reservation API.
_TEST_PORTS = itertools.count(20000)


def free_port():
    for _ in range(5000):
        port = next(_TEST_PORTS)
        if port >= 25000:
            break
        try:
            with socket.socket() as sock:
                sock.bind(("0.0.0.0", port))
                return port
        except OSError:
            continue
    raise RuntimeError("No available non-ephemeral native-fixture control port")


def free_port_pair():
    for _ in range(100):
        port = free_port()
        # Consume the neighbor as well so separately selected TP pairs cannot
        # repeatedly overlap before their listeners have been constructed.
        next(_TEST_PORTS)
        try:
            with socket.socket() as first, socket.socket() as second:
                first.bind(("0.0.0.0", port))
                second.bind(("0.0.0.0", port + 1))
                return port
        except OSError:
            continue
    raise RuntimeError("No available neighboring control ports")


class TestKVCRNativeStorage(CustomTestCase):
    def test_large_prefetch_batches_restore_every_component_through_real_nixl(self):
        """Combined backups and three reads restore every real-NIXL component.

        The target starts cold; real independent agents and the real controller
        must populate unequal KV/indexer components and produce the exact ACK
        prefix sequence. This is not a GPU inference/performance gate.
        """
        pages, page_size = 1100, 2
        keys = [hashlib.sha256(f"large-{i}".encode()).hexdigest() for i in range(pages)]
        ports = [free_port(), free_port()]
        while ports[0] == ports[1]:
            ports[1] = free_port()
        with ExitStack() as resources:
            stores, pools = [], []
            for port in ports:
                host = {
                    PoolName.KV: HostPages((8192, 4096), pages=pages),
                    PoolName.INDEXER: HostPages((2048,), pages=pages),
                }
                config = HiCacheStorageConfig(
                    tp_rank=0,
                    tp_size=1,
                    pp_rank=0,
                    pp_size=1,
                    attn_cp_rank=0,
                    attn_cp_size=1,
                    is_mla_model=True,
                    enable_storage_metrics=False,
                    is_page_first_layout=True,
                    model_name="glm53-large-read-byte-gate",
                    extra_config={
                        "control_port": port,
                        "control_advertise_host": "127.0.0.1",
                        "local_dram_pool_pages": {"kv": pages, "indexer": pages},
                        "combine_page_reads": True,
                        "combine_page_writes": True,
                        "prefetch_batch_pages": 512,
                    },
                )
                store = KVCRStore(config, host[PoolName.KV])
                store.register_mem_host_pool_v2(
                    host[PoolName.INDEXER], PoolName.INDEXER
                )
                resources.callback(store.close)
                stores.append(store)
                pools.append(host)
            source, target = stores
            for pool_index, pool in enumerate(pools[0].values()):
                for part, buffer in enumerate(pool.buffers):
                    pattern = (
                        torch.arange(pages).reshape(-1, 1) * 31
                        + torch.arange(buffer.shape[1]).reshape(1, -1) * 17
                        + pool_index * 43
                        + part * 11
                    ) % 251
                    buffer.copy_(pattern.to(torch.uint8))
            indices = torch.arange(pages * page_size)
            backup = HybridCacheController.__new__(HybridCacheController)
            backup.storage_backend, backup.page_size = source, page_size
            backup.backup_skip = False
            backup.mem_pool_host = SimpleNamespace(
                kv_buffer=pools[0][PoolName.KV].kv_buffer, layout_lease=ExitStack
            )
            stored = StorageOperation(
                indices,
                list(range(pages * page_size)),
                hash_value=keys,
                pool_transfers=[
                    PoolTransfer(PoolName.INDEXER, indices_from_pool=PoolName.KV)
                ],
            )
            backup._page_backup(stored)
            self.assertEqual(stored.completed_tokens, pages * page_size)
            self.assertEqual(
                stored.pool_storage_result.extra_pool_hit_pages, {"indexer": pages}
            )
            self.assertEqual(
                target.batch_exists_v2(
                    keys, [PoolTransfer(PoolName.INDEXER)]
                ).kv_hit_pages,
                0,
            )
            c = HiCacheController.__new__(HiCacheController)
            c.storage_backend, c.page_size = target, page_size
            c.page_get_func = c._page_get_zero_copy
            c.prefetch_sync_queue = Queue()
            op = PrefetchOperation("native-large-read", list(range(pages * page_size)))
            op.host_indices, op.hash_value = indices, keys
            op.pool_transfers = [
                PoolTransfer(PoolName.INDEXER, indices_from_pool=PoolName.KV)
            ]
            op.kv_hints = {
                "protocol_version": "0.1",
                "actions": [
                    {
                        "action_type": "kv.fetch",
                        "action_version": "1.0",
                        "payload": {
                            "source_control_endpoint": f"tcp://127.0.0.1:{ports[0]}",
                            "block_hashes": [int(key[:16], 16) for key in keys],
                        },
                    }
                ],
            }
            self.assertEqual(c._page_transfer(op), pages)
            acks = [
                c.prefetch_sync_queue.get_nowait().completed_tokens for _ in range(3)
            ]
            self.assertEqual(acks, [1024, 2048, 2200])
            self.assertTrue(c.prefetch_sync_queue.empty())
            for name in pools[0]:
                for original, restored in zip(
                    pools[0][name].buffers, pools[1][name].buffers, strict=True
                ):
                    self.assertTrue(torch.equal(original, restored))

    def test_shared_mla_real_reader_restores_from_replica_owner_and_remote_owner(self):
        """No secondary DRAM copy; local owner truth bounds advisory cold hits.

        Restore every primary/indexer byte into rank-one host slots through
        real NIXL, first from its own TP0 and then an independent replica's
        TP0, including a read mixing pages owned by the two replicas. An
        external hint must not hide pages still resident on the reader's own
        TP0. The secondary has no KVCR-owned local DRAM allocation.
        """
        keys = [hashlib.sha256(f"shared-{i}".encode()).hexdigest() for i in range(4)]
        remote_keys = [
            hashlib.sha256(f"shared-remote-{i}".encode()).hexdigest() for i in range(4)
        ]
        bases = [free_port_pair() for _ in range(2)]
        while abs(bases[0] - bases[1]) < 2:
            bases[1] = free_port_pair()
        nixl = [free_port_pair() for _ in range(2)]
        while (
            any(abs(a - b) < 2 for a in nixl for b in bases)
            or abs(nixl[0] - nixl[1]) < 2
        ):
            nixl = [free_port_pair() for _ in range(2)]
        with ExitStack() as resources:
            stores, pools = [], []
            for replica, rank in ((0, 0), (1, 0), (1, 1)):
                host = {
                    PoolName.KV: HostPages((8192, 4096)),
                    PoolName.INDEXER: HostPages((2048,)),
                }
                config = HiCacheStorageConfig(
                    tp_rank=rank,
                    tp_size=2,
                    pp_rank=0,
                    pp_size=1,
                    attn_cp_rank=0,
                    attn_cp_size=1,
                    is_mla_model=True,
                    enable_storage_metrics=False,
                    is_page_first_layout=True,
                    model_name="glm53-replicated-native",
                    extra_config={
                        "control_port": bases[replica],
                        "nixl_listen_port": nixl[replica],
                        "control_advertise_host": "127.0.0.1",
                        "local_dram_pool_pages": {"kv": 16, "indexer": 16},
                        "share_replicated_mla": True,
                    },
                )
                store = KVCRStore(config, host[PoolName.KV])
                store.register_mem_host_pool_v2(
                    host[PoolName.INDEXER], PoolName.INDEXER
                )
                resources.callback(store.close)
                stores.append(store)
                pools.append(host)
            source, owner, reader = stores
            # Real query statuses: TP0's local truth makes the cross-rank MIN
            # zero despite the secondary's advisory own-replica coverage.
            owner_hits = owner.batch_exists_v2(
                keys, [PoolTransfer(PoolName.INDEXER)]
            ).kv_hit_pages
            reader_hits = reader.batch_exists_v2(
                keys, [PoolTransfer(PoolName.INDEXER)]
            ).kv_hit_pages
            self.assertEqual(owner_hits, 0)
            self.assertEqual(reader_hits, 4)
            self.assertEqual(min(owner_hits, reader_hits), 0)
            self.assertEqual(reader._maps, [])

            def transfers(indices, page_keys=keys):
                return [
                    PoolTransfer(name, keys=page_keys, host_indices=indices)
                    for name in pools[0]
                ]

            for replica in (0, 1):
                for pool in pools[replica].values():
                    for part, buffer in enumerate(pool.buffers):
                        for page in range(4):
                            buffer[page].copy_(
                                (
                                    (
                                        torch.arange(buffer.shape[1]) * 17
                                        + replica * 31
                                        + part
                                        + page
                                    )
                                    % 251
                                ).to(torch.uint8)
                            )
                result = stores[replica].batch_set_v2(
                    transfers(torch.arange(8), remote_keys if replica == 0 else keys)
                )
                self.assertTrue(all(all(rows) for rows in result.values()))
            self.assertEqual(owner._namespace, reader._namespace)
            self.assertEqual(source._namespace, reader._namespace)
            self.assertEqual(
                owner.batch_exists_v2(
                    keys, [PoolTransfer(PoolName.INDEXER)]
                ).kv_hit_pages,
                4,
            )
            external = HiCacheStorageExtraInfo(
                extra_info={
                    "kv_hints": {
                        "protocol_version": "0.1",
                        "actions": [
                            {
                                "action_type": "kv.fetch",
                                "action_version": "1.0",
                                "payload": {
                                    "source_control_endpoint": f"tcp://127.0.0.1:{bases[0]}",
                                    "block_hashes": [
                                        int(key[:16], 16) for key in remote_keys
                                    ],
                                },
                            }
                        ],
                    }
                }
            )
            for replica, info, indices, page_keys in (
                (1, None, torch.arange(8, 16), keys),
                (0, external, torch.arange(16, 24), remote_keys),
            ):
                restored = reader.batch_get_v2(transfers(indices, page_keys), info)
                self.assertTrue(all(all(rows) for rows in restored.values()), restored)
                slot = int(indices[0]) // 2
                for name, host in pools[replica].items():
                    for expected, actual in zip(
                        host.buffers, pools[2][name].buffers, strict=True
                    ):
                        self.assertTrue(
                            torch.equal(expected[:4], actual[slot : slot + 4])
                        )
                self.assertEqual(reader._maps, [])

            # Router coverage is advisory, not a promise that every hinted
            # page remains on that source. TP0 can combine its local pages
            # with externally owned pages; its secondary must do so too.
            mixed_keys = [keys[0], remote_keys[1], keys[2], remote_keys[3]]
            mixed_info = deepcopy(external)
            mixed_info.extra_info["kv_hints"]["actions"][0]["payload"][
                "block_hashes"
            ] = [int(key[:16], 16) for key in mixed_keys]
            for store, destination in ((owner, pools[1]), (reader, pools[2])):
                with self.subTest(reader_rank=store.config.tp_rank):
                    self.assertEqual(
                        store.batch_exists_v2(
                            mixed_keys, [PoolTransfer(PoolName.INDEXER)], mixed_info
                        ).kv_hit_pages,
                        4,
                    )
                    restored = store.batch_get_v2(
                        transfers(torch.arange(24, 32), mixed_keys), mixed_info
                    )
                    self.assertEqual(
                        restored,
                        {str(name): [True] * 4 for name in pools[0]},
                    )
                    for name, host in destination.items():
                        for part, actual in enumerate(host.buffers):
                            for page, replica in enumerate((1, 0, 1, 0)):
                                self.assertTrue(
                                    torch.equal(
                                        pools[replica][name].buffers[part][page],
                                        actual[12 + page],
                                    ),
                                    (name, part, page, replica),
                                )
            self.assertEqual(reader._maps, [])

    def test_prefix_alignment_preserves_bounded_native_lru(self):
        """Real native allocation must evict an unrelated page, not a hot parent.

        Run an unaligned control and an aligned candidate at identical three-
        page capacity for both KV and indexer. Query and restored bytes, not
        a mocked callback, prove the effect of pre-allocation alignment.
        """
        keys = [hashlib.sha256(f"lru-{i}".encode()).hexdigest() for i in range(4)]
        for aligned in (False, True):
            with self.subTest(aligned=aligned), ExitStack() as resources:
                pools = {
                    PoolName.KV: HostPages((8192, 4096)),
                    PoolName.INDEXER: HostPages((2048,)),
                }
                control_port = free_port()
                nixl_port = free_port()
                while nixl_port == control_port:
                    nixl_port = free_port()
                config = HiCacheStorageConfig(
                    tp_rank=0,
                    tp_size=1,
                    pp_rank=0,
                    pp_size=1,
                    attn_cp_rank=0,
                    attn_cp_size=1,
                    is_mla_model=True,
                    enable_storage_metrics=False,
                    is_page_first_layout=True,
                    model_name="glm53-native-bounded-lru",
                    extra_config={
                        "control_port": control_port,
                        "nixl_listen_port": nixl_port,
                        "local_dram_pool_pages": {"kv": 3, "indexer": 3},
                        "align_storage_prefix": aligned,
                    },
                )
                store = KVCRStore(config, pools[PoolName.KV])
                store.register_mem_host_pool_v2(
                    pools[PoolName.INDEXER], PoolName.INDEXER
                )
                resources.callback(store.close)
                for pool in pools.values():
                    for part, buffer in enumerate(pool.buffers):
                        for slot in range(4):
                            buffer[slot].copy_(
                                (
                                    (torch.arange(buffer.shape[1]) * 37 + slot + part)
                                    % 251
                                ).to(torch.uint8)
                            )

                def transfers(page_keys, indices, names=tuple(pools)):
                    return [
                        PoolTransfer(name, keys=page_keys, host_indices=indices)
                        for name in names
                    ]

                for page_keys, indices, info in (
                    (keys[:2], torch.arange(4), None),
                    (keys[2:3], torch.arange(4, 6), None),
                    (
                        keys[3:],
                        torch.arange(6, 8),
                        HiCacheStorageExtraInfo(prefix_keys=keys[:2]),
                    ),
                ):
                    result = store.batch_set_v2(transfers(page_keys, indices), info)
                    self.assertTrue(
                        all(all(values) for values in result.values()), result
                    )
                chain = keys[:2] + keys[3:]
                self.assertEqual(
                    store.batch_exists_v2(
                        chain, [PoolTransfer(PoolName.INDEXER)]
                    ).kv_hit_pages,
                    3 if aligned else 0,
                )
                if aligned:
                    result = store.batch_get_v2(transfers(chain, torch.arange(8, 14)))
                    self.assertTrue(
                        all(all(values) for values in result.values()), result
                    )
                    self.assertEqual(
                        store.batch_exists_v2(
                            keys[2:3], [PoolTransfer(PoolName.INDEXER)]
                        ).kv_hit_pages,
                        0,
                    )
                    for pool in pools.values():
                        for buffer in pool.buffers:
                            self.assertTrue(torch.equal(buffer[[0, 1, 3]], buffer[4:7]))

    def test_remote_restore_all_components_and_missing_pool(self):
        """Remote success must populate every byte; missing indexer stays invalid."""
        self._round_trip(((8192, 4096), (2048,)), page_size=2)

    def test_glm53_page_sized_remote_restore(self):
        """Exercise the observed GLM5.3 MLA/indexer physical page byte sizes.

        This is a storage transport gate, not an inference or GPU-copy test.
        Nonuniform payloads expose within-page truncation and offset errors.
        """
        self._round_trip(((2875392,), (658944,)), page_size=64)

    def test_native_diagnostics_record_remote_bytes_and_inventory(self):
        """The flag alone is insufficient: native stats require a wired factory.

        Validate a real peer restore, then inspect the native interval handed
        back during close. Logical adapter bandwidth cannot satisfy this gate.
        """
        from kvcr.api import DURATION_METRIC, TRANSFER_BYTES_METRIC

        with self.assertLogs(
            "sglang.srt.mem_cache.storage.kvcr.kvcr_store", level="INFO"
        ) as captured:
            self._round_trip(((8192, 4096), (2048,)), page_size=2, diagnostic=True)
        snapshots = [
            json.loads(line.split("KVCR L3 native stats ", 1)[1])["metrics"]
            for line in captured.output
            if "KVCR L3 native stats " in line
        ]
        source_bytes = json.dumps(
            ("counter", TRANSFER_BYTES_METRIC, ("source_write",), "value"),
            separators=(",", ":"),
        )
        remote_duration = json.dumps(
            ("histogram", DURATION_METRIC, ("remote_deliver", "success"), "count"),
            separators=(",", ":"),
        )
        self.assertGreater(
            sum(snapshot.get(source_bytes, 0) for snapshot in snapshots), 0
        )
        self.assertGreater(
            sum(snapshot.get(remote_duration, 0) for snapshot in snapshots), 0
        )
        inventory = [
            json.loads(line.split("KVCR L3 inventory ", 1)[1])
            for line in captured.output
            if "KVCR L3 inventory " in line
        ]
        self.assertTrue(
            any(not event["removed"] and event["keys"] for event in inventory)
        )

    def _round_trip(self, sizes, page_size, diagnostic=False):
        keys = [hashlib.sha256(f"page-{i}".encode()).hexdigest() for i in range(4)]
        port = free_port()
        target_port = free_port()
        while target_port == port:
            target_port = free_port()
        source_pools = {
            PoolName.KV: HostPages(sizes[0], page_size),
            PoolName.INDEXER: HostPages(sizes[1], page_size),
        }
        target_pools = {
            PoolName.KV: HostPages(sizes[0], page_size),
            PoolName.INDEXER: HostPages(sizes[1], page_size),
        }
        with ExitStack() as resources:
            stores = []
            for control_port, pools in (
                (port, source_pools),
                (target_port, target_pools),
            ):
                config = HiCacheStorageConfig(
                    tp_rank=0,
                    tp_size=1,
                    pp_rank=0,
                    pp_size=1,
                    attn_cp_rank=0,
                    attn_cp_size=1,
                    is_mla_model=True,
                    enable_storage_metrics=False,
                    is_page_first_layout=True,
                    model_name="glm53-native-byte-gate",
                    extra_config={
                        "control_port": control_port,
                        "control_advertise_host": "127.0.0.1",
                        "local_dram_pool_pages": {"kv": 16, "indexer": 16},
                        "diagnostic_telemetry": diagnostic,
                    },
                )
                store = KVCRStore(config, pools[PoolName.KV])
                store.register_mem_host_pool_v2(
                    pools[PoolName.INDEXER], PoolName.INDEXER
                )
                resources.callback(store.close)
                stores.append(store)
            source, target = stores
            for pool_index, pool in enumerate(source_pools.values()):
                for part_index, buffer in enumerate(pool.buffers):
                    for slot in range(4):
                        pattern = torch.arange(buffer.shape[1], dtype=torch.int64)
                        pattern = (
                            pattern * 37 + 11 + 31 * pool_index + 7 * part_index + slot
                        ) % 251
                        buffer[slot].copy_(pattern.to(torch.uint8))
            written = source.batch_set_v2(
                [
                    PoolTransfer(
                        name, keys=keys, host_indices=torch.arange(4 * page_size)
                    )
                    for name in source_pools
                ]
            )
            self.assertTrue(all(all(results) for results in written.values()))
            transfers = [
                PoolTransfer(
                    name,
                    keys=keys,
                    host_indices=torch.arange(8 * page_size, 12 * page_size),
                )
                for name in target_pools
            ]
            # An empty target cannot silently manufacture a local hit.
            self.assertEqual(
                target.batch_exists_v2(
                    keys, [PoolTransfer(PoolName.INDEXER)]
                ).kv_hit_pages,
                0,
            )
            hint = HiCacheStorageExtraInfo(
                extra_info={
                    "kv_hints": {
                        "protocol_version": "0.1",
                        "message_id": "native-restore",
                        "actions": [
                            {
                                "action_id": "fetch",
                                "action_type": "kv.fetch",
                                "action_version": "1.0",
                                "payload": {
                                    "source_control_endpoint": f"tcp://127.0.0.1:{port}",
                                    "block_hashes": [int(key[:16], 16) for key in keys],
                                },
                            }
                        ],
                    }
                }
            )
            restored = target.batch_get_v2(transfers, hint)
            self.assertTrue(
                all(all(results) for results in restored.values()), restored
            )
            for name, source_pool in source_pools.items():
                for source_buffer, target_buffer in zip(
                    source_pool.buffers, target_pools[name].buffers, strict=True
                ):
                    self.assertTrue(torch.equal(source_buffer[:4], target_buffer[8:12]))
            missing_key = hashlib.sha256(b"missing-indexer").hexdigest()
            source.batch_set_v1(
                [missing_key], torch.arange(4 * page_size, 5 * page_size)
            )
            hint.extra_info["kv_hints"]["actions"][0]["payload"]["block_hashes"] = [
                int(missing_key[:16], 16)
            ]
            missing = target.batch_get_v2(
                [
                    PoolTransfer(
                        name,
                        keys=[missing_key],
                        host_indices=torch.arange(12 * page_size, 13 * page_size),
                    )
                    for name in target_pools
                ],
                hint,
            )
            self.assertEqual(missing["indexer"], [False])
            # KVCR may refuse the whole remote operation when a required key
            # is absent; no caller may infer a valid combined page from it.
            self.assertFalse(all(all(results) for results in missing.values()))


if __name__ == "__main__":
    unittest.main()
