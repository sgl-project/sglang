import threading
import unittest
from collections import Counter
from queue import Queue
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
    _storage_sharing,
    _storage_suffix,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_REPLICATED = {PoolName.KV, PoolName.INDEXER, PoolName.DRAFT}


def _sharing(dcp_size=1, **kwargs):
    pools = {
        name: SimpleNamespace(
            tp_replicated=name in _REPLICATED,
            cp_replicated=False,
            dcp_sharded=dcp_size > 1 and name in _REPLICATED,
        )
        for name in (*_REPLICATED, PoolName.MAMBA)
    }
    args = dict(
        rank_replicated=False,
        tp_rank=0,
        tp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        pp_rank=0,
        offload_owner=True,
    )
    args.update(kwargs)
    return _storage_sharing(
        pools, dcp_size=dcp_size, dcp_rank=args["tp_rank"] % dcp_size, **args
    )


class TestStorageSharing(CustomTestCase):
    def test_replicated_pools_share_one_key_and_one_writer(self):
        writers = {}
        for rank in range(8):
            suffixes, offload, tp_shared, _ = _sharing(tp_rank=rank, tp_size=8)
            self.assertEqual(tp_shared, _REPLICATED)
            self.assertEqual(
                suffixes, dict.fromkeys(_REPLICATED, "tpshared_v1_cp0_pp0")
            )
            writers[rank] = offload
        self.assertEqual(writers[0], _REPLICATED | {PoolName.MAMBA})
        for rank in range(1, 8):
            self.assertEqual(writers[rank], {PoolName.MAMBA})

    def test_each_cp_rank_keeps_its_own_shared_object(self):
        for cp in range(2):
            suffixes, _, _, _ = _sharing(tp_rank=1, tp_size=2, attn_cp_rank=cp)
            self.assertEqual(suffixes[PoolName.KV], f"tpshared_v1_cp{cp}_pp0")

    def test_single_tp_rank_or_replicated_group_keeps_group_keys(self):
        for kwargs in ({"tp_size": 1}, {"rank_replicated": True, "tp_size": 8}):
            suffixes, offload, tp_shared, cp_shared = _sharing(**kwargs)
            self.assertEqual((suffixes, tp_shared), ({}, set()))
            self.assertFalse(cp_shared)
            self.assertEqual(offload, _REPLICATED | {PoolName.MAMBA})

    def test_dcp_keys_each_rank_and_the_first_block_writes(self):
        for rank in range(16):
            suffixes, offload, _, _ = _sharing(tp_rank=rank, tp_size=16, dcp_size=8)
            self.assertEqual(
                suffixes,
                dict.fromkeys(_REPLICATED, f"tpshared_v1_dcp8r{rank % 8}_cp0_pp0"),
            )
            self.assertEqual(
                offload,
                _REPLICATED | {PoolName.MAMBA} if rank < 8 else {PoolName.MAMBA},
            )

    def test_non_owner_rank_offloads_nothing(self):
        _, offload, _, _ = _sharing(tp_rank=3, tp_size=8, offload_owner=False)
        self.assertEqual(offload, set())

    def test_dsv41_cp_rank_storage_has_one_writer_for_each_shared_key(self):
        """CP peers restore the same full pages, but rank-local SWA stays distinct.

        A DSV4.1 CP8/attention-TP1 write previously persisted eight identical
        C1/C2 KV and index pools: every CP rank was an attention-TP owner.
        """
        replicated = {
            PoolName.DEEPSEEK_V4_C1,
            PoolName.DEEPSEEK_V4_C1_INDEXER,
            PoolName.DEEPSEEK_V4_C2,
            PoolName.DEEPSEEK_V4_C2_INDEXER,
        }
        pools = {
            name: SimpleNamespace(
                tp_replicated=False,
                dcp_sharded=False,
                cp_replicated=name in replicated,
            )
            for name in (*replicated, PoolName.SWA)
        }

        for cp_size in (1, 4, 8):
            with self.subTest(cp_size=cp_size):
                written = Counter()
                readable = set()
                for cp_rank in range(cp_size):
                    args = dict(
                        rank_replicated=True,
                        tp_rank=0,
                        tp_size=1,
                        attn_cp_rank=cp_rank,
                        attn_cp_size=cp_size,
                        pp_rank=2,
                        offload_owner=True,
                        dcp_size=1,
                        dcp_rank=0,
                    )
                    suffixes, owners, _, _ = _storage_sharing(pools, **args)
                    rank_suffix = _storage_suffix(
                        rank_replicated=True,
                        tp_rank=0,
                        attn_cp_rank=cp_rank,
                        pp_rank=2,
                    )
                    # Every physical object read by any CP rank must have
                    # exactly one writer, with rank-local windows immutably
                    # separate from the new versioned full-page namespace.
                    for name in pools:
                        key = (name, suffixes.get(name, rank_suffix))
                        readable.add(key)
                        if name in owners:
                            written[key] += 1
                        if name == PoolName.SWA or cp_size == 1:
                            self.assertEqual(key[1], rank_suffix)
                        else:
                            self.assertNotEqual(key[1], rank_suffix)
                    # Other attention TP ranks must not become CP writers.
                    _, non_owner, _, _ = _storage_sharing(
                        pools,
                        **(args | {"tp_rank": 1, "tp_size": 2, "offload_owner": False}),
                    )
                    self.assertFalse(non_owner)
                self.assertEqual(set(written), readable)
                self.assertEqual(set(written.values()), {1})
                self.assertEqual(len(written), len(replicated) + cp_size)
                if cp_size > 1:
                    for _, suffix in written:
                        if suffix.startswith("cpshared"):
                            self.assertIn(f"cp{cp_size}_pp2", suffix)

    def test_noop_offload_does_not_overtake_rank_local_write(self):
        """The rank-local write's failure must ACK its own radix node, not the next.

        A non-writer can still have a pending SWA offload when the next node
        contains only shared full pages. Keep even the no-op ACK in FIFO order.
        """

        def entry(name):
            return DevicePoolEntry(
                name=name,
                indices_from_pool=name,
                device_pool=None,
                components=[[torch.zeros((2, 4), dtype=torch.uint8)]],
                layer_mapping={0: 0},
                page_size=2,
                rows_are_pages=True,
            )

        linker = object.__new__(MooncakeDirectLinker)
        linker.pool_group = DevicePoolGroup(
            [entry(PoolName.KV), entry(PoolName.SWA)], num_layers=1, page_size=2
        )
        linker.page_size = 2
        linker.offload_pools = {PoolName.SWA}
        linker.gc_frozen = True
        linker.offload_queue = Queue()
        linker.offload_results = Queue()
        linker.record_transfer_ready = lambda: SimpleNamespace(synchronize=lambda: None)
        entered, release = threading.Event(), threading.Event()

        def failed_write(transfers, extra_info):
            entered.set()
            release.wait(timeout=5)
            return {PoolName.SWA: [False]}

        linker.storage = SimpleNamespace(batch_set_v2=failed_write)
        worker = threading.Thread(target=linker.offload_thread_func, daemon=True)
        worker.start()

        def transfer(name, key):
            return PoolTransfer(name=name, keys=[key], device_indices=torch.arange(2))

        try:
            self.assertTrue(
                linker.offload(
                    [transfer(PoolName.KV, "a"), transfer(PoolName.SWA, "a")]
                )
            )
            self.assertTrue(entered.wait(timeout=5))
            self.assertTrue(linker.offload([transfer(PoolName.KV, "b")]))
            self.assertEqual(linker.num_completed_offloads(), 0)
            release.set()
            self.assertFalse(linker.offload_results.get(timeout=5))
            self.assertTrue(linker.offload_results.get(timeout=5))
        finally:
            release.set()
            linker.offload_queue.put(None)
            worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
