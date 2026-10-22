"""MooncakeStore under the unified key scheme: one object per grid chunk.

The fake store copies real bytes through the pointers MooncakeStore hands it,
so these tests check what lands in the reader's page blocks, not just keys.
"""

import ctypes
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_key_scheme import plan_unified_kv
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.pool_host.page_unified import PageUnifiedLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 4
LAYERS = 4
TOTAL_KV_HEADS = 4
HEAD_DIM = 8
PAGES = 3


class _FakeCluster:
    """Objects shared by every client, like one mooncake_master."""

    def __init__(self):
        self.objects = {}


def _fake_store_class(cluster):
    class FakeMooncakeDistributedStore:
        def setup(self, *args, **kwargs):
            return 0

        def register_buffer(self, ptr, size):
            return 0

        def put(self, key, value, *args):
            cluster.objects[key] = bytes(value)
            return 0

        def is_exist(self, key):
            return 1 if key in cluster.objects else 0

        def get(self, key):
            return cluster.objects.get(key)

        def batch_is_exist(self, keys):
            return [1 if key in cluster.objects else 0 for key in keys]

        def batch_put_from(self, keys, ptrs, sizes, *config):
            for key, ptr, size in zip(keys, ptrs, sizes):
                cluster.objects.setdefault(key, ctypes.string_at(ptr, size))
            return [0] * len(keys)

        def batch_get_into(self, keys, ptrs, sizes):
            results = []
            for key, ptr, size in zip(keys, ptrs, sizes):
                data = cluster.objects.get(key)
                if data is None or len(data) != size:
                    results.append(-1)
                    continue
                ctypes.memmove(ptr, data, size)
                results.append(size)
            return results

    return FakeMooncakeDistributedStore


class _ReplicateConfig:
    def __init__(self):
        self.group_ids = None


def _fake_mooncake_modules(cluster):
    mooncake = types.ModuleType("mooncake")
    store = types.ModuleType("mooncake.store")
    store.MooncakeDistributedStore = _fake_store_class(cluster)
    store.ReplicateConfig = _ReplicateConfig
    return {"mooncake": mooncake, "mooncake.store": store}


class _UnifiedHostPool:
    """The page_unified host pool surface MooncakeStore reads, on a CPU buffer."""

    _page_ids = HostKVCache._page_ids
    get_page_buffer_meta = HostKVCache._page_unified_buffer_meta

    def __init__(self, layout):
        self.page_unified_layout = layout
        self.page_size = layout.page_size
        self.layout = "page_unified"
        self.kv_buffer = torch.zeros(layout.page_dims(PAGES), dtype=torch.bfloat16)

    def get_ksize_per_token(self):
        return self.page_unified_layout.bytes_per_page // self.page_size


def _chunk_keys(cluster):
    # Every store instance also puts one warmup object at construction.
    return sorted(key for key in cluster.objects if "warmup" not in key)


def _page_indices(pages):
    return torch.tensor(
        [page * PAGE_SIZE + offset for page in pages for offset in range(PAGE_SIZE)]
    )


class _Rank:
    """One attention-TP rank: its plan, page_unified pool and MooncakeStore."""

    def __init__(self, cluster, *, mla, tp_rank, tp_size, head_partition):
        local_heads = 1 if mla else TOTAL_KV_HEADS // tp_size
        groups = 1 if mla else local_heads // head_partition
        self.layout = PageUnifiedLayout(
            page_size=PAGE_SIZE,
            layer_num=LAYERS,
            head_num=local_heads,
            head_group_num=groups,
            head_dim=HEAD_DIM,
            itemsize=2,
            is_mla=mla,
        )
        self.plan = plan_unified_kv(
            model_id="org/model",
            dtype="bfloat16",
            page_size=PAGE_SIZE,
            rank_replicated=mla,
            local_kv_heads=0 if mla else local_heads,
            attn_tp_rank=tp_rank,
            attn_tp_size=tp_size,
            attn_cp_size=1,
            start_layer=0,
            end_layer=LAYERS,
            is_final_stage=True,
            head_group=None if mla else head_partition,
            layer_partition=2,
        )
        self.first_head = tp_rank * local_heads
        self.pool = _UnifiedHostPool(self.layout)
        config = HiCacheStorageConfig(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=mla,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="org/model",
            extra_config={
                "master_server_address": "127.0.0.1:50051",
                "check_server": False,
                "global_segment_size": 0,
            },
            unified_suffixes=self.plan.suffixes,
            unified_chunk_spans=self.plan.chunk_spans(self.layout),
        )
        with patch.dict("sys.modules", _fake_mooncake_modules(cluster)):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            self.store = MooncakeStore(config)
        self.store.register_mem_pool_host(self.pool)

    def fill_from(self, logical, host_pages):
        """Lay logical KV pages out the way this rank's write-back would."""
        for host_page, page in zip(host_pages, logical):
            self.pool.kv_buffer[host_page] = self.expected_page(page)

    def expected_page(self, page):
        """This rank's slice of one logical page in page_unified byte order."""
        layout = self.layout
        if layout.is_mla:
            return page
        # page: (layer, 2, page_size, total_heads, dim) -> this rank's heads,
        # then (group, layer, 2, page_size, heads_in_group, dim).
        heads = page[:, :, :, self.first_head : self.first_head + layout.head_num]
        return heads.reshape(
            LAYERS,
            2,
            PAGE_SIZE,
            layout.head_group_num,
            layout.heads_per_group,
            HEAD_DIM,
        ).permute(3, 0, 1, 2, 4, 5)


def _logical_pages(mla, count):
    generator = torch.Generator().manual_seed(20261021)
    shape = (
        (LAYERS, PAGE_SIZE, HEAD_DIM)
        if mla
        else (LAYERS, 2, PAGE_SIZE, TOTAL_KV_HEADS, HEAD_DIM)
    )
    return [
        torch.randn(shape, generator=generator).to(torch.bfloat16) for _ in range(count)
    ]


class TestMooncakeUnifiedKeys(CustomTestCase):
    def test_tp2_writers_fill_a_tp1_readers_page_blocks(self):
        """Cross-TP reuse at the byte level: TP2's chunks tile TP1's pages.

        The reader sees no hit until both writers' head groups exist, then
        reassembles every page block bit-exactly at its own host pages.
        """
        cluster = _FakeCluster()
        writers = [
            _Rank(cluster, mla=False, tp_rank=r, tp_size=2, head_partition=1)
            for r in range(2)
        ]
        reader = _Rank(cluster, mla=False, tp_rank=0, tp_size=1, head_partition=1)
        keys = ["page0", "page1"]
        logical = _logical_pages(mla=False, count=len(keys))

        writers[0].fill_from(logical, host_pages=[0, 1])
        self.assertEqual(
            writers[0].store.batch_set_v1(keys, _page_indices([0, 1])), [True, True]
        )
        # Half the head groups are not a hit for a reader that needs all of them.
        self.assertEqual(reader.store.batch_exists(keys), 0)
        self.assertEqual(writers[0].store.batch_exists(keys), 2)

        writers[1].fill_from(logical, host_pages=[2, 0])
        self.assertEqual(
            writers[1].store.batch_set_v1(keys, _page_indices([2, 0])), [True, True]
        )
        self.assertEqual(reader.store.batch_exists(keys), 2)
        # Per page: 2 writers x 2 layer ranges x 2 one-head groups each.
        self.assertEqual(len(_chunk_keys(cluster)), len(keys) * 2 * 2 * 2)

        self.assertEqual(
            reader.store.batch_get_v1(keys, _page_indices([2, 1])), [True, True]
        )
        for host_page, page in zip([2, 1], logical):
            self.assertTrue(
                torch.equal(
                    reader.pool.kv_buffer[host_page], reader.expected_page(page)
                )
            )

    def test_mla_chunks_are_shared_across_tp_sizes(self):
        cluster = _FakeCluster()
        writer = _Rank(cluster, mla=True, tp_rank=0, tp_size=2, head_partition=None)
        reader = _Rank(cluster, mla=True, tp_rank=1, tp_size=4, head_partition=None)
        self.assertEqual(writer.plan.suffixes, reader.plan.suffixes)
        logical = _logical_pages(mla=True, count=1)
        writer.fill_from(logical, host_pages=[1])
        self.assertEqual(writer.store.batch_set_v1(["p"], _page_indices([1])), [True])
        self.assertEqual(reader.store.batch_get_v1(["p"], _page_indices([0])), [True])
        self.assertTrue(torch.equal(reader.pool.kv_buffer[0], logical[0]))

    def test_keys_name_chunks_not_ranks(self):
        cluster = _FakeCluster()
        rank = _Rank(cluster, mla=False, tp_rank=1, tp_size=2, head_partition=1)
        rank.store.batch_set_v1(["p"], _page_indices([0]))
        stored = _chunk_keys(cluster)
        self.assertEqual(
            stored,
            sorted(f"org-model_p_{suffix}" for suffix in rank.plan.suffixes),
        )
        self.assertTrue(all(key.endswith(("_H2", "_H3")) for key in stored))

    def test_a_missing_chunk_fails_its_page_and_ends_the_prefix(self):
        cluster = _FakeCluster()
        rank = _Rank(cluster, mla=False, tp_rank=0, tp_size=1, head_partition=2)
        keys = ["p0", "p1", "p2"]
        rank.store.batch_set_v1(keys, _page_indices([0, 1, 2]))
        del cluster.objects[f"org-model_p1_{rank.plan.suffixes[-1]}"]
        self.assertEqual(rank.store.batch_exists(keys), 1)
        self.assertEqual(
            rank.store.batch_get_v1(keys, _page_indices([0, 1, 2])),
            [True, False, True],
        )

    def test_backup_metrics_count_pages_not_chunks(self):
        cluster = _FakeCluster()
        rank = _Rank(cluster, mla=False, tp_rank=0, tp_size=1, head_partition=2)
        rank.store.enable_storage_metrics = True
        rank.store.batch_set_v1(["p0", "p1"], _page_indices([0, 1]))
        self.assertEqual(rank.store.backup_pgs, [2])
        self.assertEqual(rank.store.gb_per_page * (1 << 30), rank.layout.bytes_per_page)

    def test_spans_must_pair_with_suffixes(self):
        cluster = _FakeCluster()
        rank = _Rank(cluster, mla=False, tp_rank=0, tp_size=1, head_partition=2)
        config = rank.store.storage_config
        config.unified_chunk_spans = config.unified_chunk_spans[:-1]
        with (
            patch.dict("sys.modules", _fake_mooncake_modules(cluster)),
            self.assertRaisesRegex(ValueError, "one chunk span per unified suffix"),
        ):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            MooncakeStore(config)

    def test_hybrid_pool_transfers_are_refused(self):
        cluster = _FakeCluster()
        rank = _Rank(cluster, mla=False, tp_rank=0, tp_size=1, head_partition=2)
        rank.store.batch_set_v1(["p"], _page_indices([0]))
        transfer = PoolTransfer(name=PoolName.MAMBA, keys=["p"])
        with self.assertRaisesRegex(NotImplementedError, "hybrid pools"):
            rank.store.batch_exists_v2(["p"], [transfer])


if __name__ == "__main__":
    unittest.main()
