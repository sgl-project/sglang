"""MooncakeStore under the unified key scheme: one object per grid chunk.

The fake store copies real bytes through the pointers MooncakeStore hands it,
so these tests check what lands in the reader's page blocks, not just keys.
"""

import ctypes
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_key_scheme import (
    plan_unified_draft_sidecar,
    plan_unified_kv,
    with_packed_draft,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
    UnifiedPoolChunks,
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
        self.registered_buffers = 0
        self.registered = []

    def holds(self, ptr, size):
        return size == 0 or any(
            base <= ptr and ptr + size <= base + length
            for base, length in self.registered
        )


def _fake_store_class(cluster):
    class FakeMooncakeDistributedStore:
        def setup(self, *args, **kwargs):
            return 0

        def register_buffer(self, ptr, size):
            cluster.registered_buffers += 1
            cluster.registered.append((ptr, size))
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
            codes = []
            for key, ptr, size in zip(keys, ptrs, sizes):
                if not cluster.holds(ptr, size):
                    codes.append(-1)
                    continue
                cluster.objects.setdefault(key, ctypes.string_at(ptr, size))
                codes.append(0)
            return codes

        def batch_get_into(self, keys, ptrs, sizes):
            results = []
            for key, ptr, size in zip(keys, ptrs, sizes):
                data = cluster.objects.get(key)
                # Like Mooncake: only a buffer smaller than the object fails; a
                # larger one takes the object and reports the object's size.
                if data is None or len(data) > size or not cluster.holds(ptr, size):
                    results.append(-1)
                    continue
                ctypes.memmove(ptr, data, len(data))
                results.append(len(data))
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
        # Like the real pools: MHA reports K alone, MLA its whole latent row.
        layout = self.page_unified_layout
        return layout.bytes_per_page // layout.components // self.page_size


def _chunk_keys(cluster):
    # Every store instance also puts one warmup object at construction.
    return sorted(key for key in cluster.objects if "warmup" not in key)


def _page_indices(pages):
    return torch.tensor(
        [page * PAGE_SIZE + offset for page in pages for offset in range(PAGE_SIZE)]
    )


class _Rank:
    """One attention-TP rank of a PP stage: plan, page_unified pool and store."""

    def __init__(
        self,
        cluster,
        *,
        mla,
        tp_rank,
        tp_size,
        head_partition,
        layer_partition=2,
        start_layer=0,
        end_layer=LAYERS,
        pp_rank=0,
        pp_size=1,
        unnamed_layers=0,
        draft_layers=0,
        draft_id="draft",
    ):
        local_heads = 1 if mla else TOTAL_KV_HEADS // tp_size
        groups = 1 if mla else local_heads // head_partition
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.layout = PageUnifiedLayout(
            page_size=PAGE_SIZE,
            # Layers of the page block that the plan does not name, and packed
            # draft layers, which follow the target's.
            layer_num=end_layer - start_layer + unnamed_layers + draft_layers,
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
            start_layer=start_layer,
            end_layer=end_layer,
            is_final_stage=pp_rank == pp_size - 1,
            head_group=None if mla else head_partition,
            layer_partition=layer_partition,
        )
        self.draft_layers = draft_layers
        if draft_layers:
            self.plan = with_packed_draft(
                self.plan,
                target_layer_num=end_layer - start_layer,
                draft_layer_num=draft_layers,
                algorithm="EAGLE",
                draft_id=draft_id,
                dtype="bfloat16",
                row_width=HEAD_DIM,
                attn_tp_rank=tp_rank,
                attn_cp_size=1,
                is_final_stage=pp_rank == pp_size - 1,
            )
        self.first_head = tp_rank * local_heads
        self.pool = _UnifiedHostPool(self.layout)
        config = HiCacheStorageConfig(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
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
        self.config = config
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
        # Logical pages hold the draft layers after the model's LAYERS.
        page = torch.cat(
            [
                page[self.start_layer : self.end_layer],
                page[LAYERS : LAYERS + self.draft_layers],
            ]
        )
        if layout.is_mla:
            return page
        # page: (layer, 2, page_size, total_heads, dim) -> this rank's heads,
        # then (group, layer, 2, page_size, heads_in_group, dim).
        heads = page[:, :, :, self.first_head : self.first_head + layout.head_num]
        return heads.reshape(
            layout.layer_num,
            2,
            PAGE_SIZE,
            layout.head_group_num,
            layout.heads_per_group,
            HEAD_DIM,
        ).permute(3, 0, 1, 2, 4, 5)


def _logical_pages(mla, count, draft_layers=0):
    generator = torch.Generator().manual_seed(20261021)
    layers = LAYERS + draft_layers
    shape = (
        (layers, PAGE_SIZE, HEAD_DIM)
        if mla
        else (layers, 2, PAGE_SIZE, TOTAL_KV_HEADS, HEAD_DIM)
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

    def test_mla_chunks_are_shared_across_tp_and_pp_splits(self):
        """MLA chunks name absolute layer ranges, not a TP size or PP stage.

        A reader with no PP split needs both stages' chunks for a hit, and
        lands each stage's layers at its own local offsets.
        """
        cluster = _FakeCluster()
        stages = [
            _Rank(
                cluster,
                mla=True,
                tp_rank=0,
                tp_size=2,
                head_partition=None,
                start_layer=start,
                end_layer=end,
                pp_rank=pp_rank,
                pp_size=2,
            )
            for pp_rank, (start, end) in enumerate([(0, 2), (2, LAYERS)])
        ]
        reader = _Rank(cluster, mla=True, tp_rank=1, tp_size=4, head_partition=None)
        logical = _logical_pages(mla=True, count=1)

        stages[0].fill_from(logical, host_pages=[1])
        self.assertEqual(
            stages[0].store.batch_set_v1(["p"], _page_indices([1])), [True]
        )
        self.assertEqual(reader.store.batch_exists(["p"]), 0)

        stages[1].fill_from(logical, host_pages=[2])
        self.assertEqual(
            stages[1].store.batch_set_v1(["p"], _page_indices([2])), [True]
        )
        self.assertEqual(
            _chunk_keys(cluster),
            sorted(f"org-model_p_{suffix}" for suffix in reader.plan.suffixes),
        )
        self.assertEqual(reader.store.batch_exists(["p"]), 1)
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

    def test_a_short_chunk_object_fails_its_page(self):
        """Mooncake fills a larger buffer from a shorter object and reports
        success; the page must miss, or the rest of the span keeps stale bytes.
        """
        cluster = _FakeCluster()
        rank = _Rank(cluster, mla=False, tp_rank=0, tp_size=1, head_partition=2)
        keys = ["p0", "p1"]
        rank.store.batch_set_v1(keys, _page_indices([0, 1]))
        short_key = f"org-model_p0_{rank.plan.suffixes[0]}"
        cluster.objects[short_key] = cluster.objects[short_key][:-16]
        self.assertEqual(
            rank.store.batch_get_v1(keys, _page_indices([0, 1])), [False, True]
        )

    def test_backup_metrics_count_pages_and_bytes(self):
        """Backups report the pages touched and the bytes put, not chunk counts.

        A ragged final layer range makes chunks unequal, so rewriting only the
        evicted short chunks moves far less than their share of a page.
        """
        cluster = _FakeCluster()
        rank = _Rank(
            cluster,
            mla=False,
            tp_rank=0,
            tp_size=1,
            head_partition=2,
            layer_partition=3,
        )
        rank.store.enable_storage_metrics = True
        # The MHA pool sizes a page by K alone; a unified page holds K and V.
        self.assertEqual(rank.store.gb_per_page * (1 << 30), rank.layout.bytes_per_page)
        keys = ["p0", "p1"]
        short_chunks = [suffix for suffix in rank.plan.suffixes if "_L3-4_" in suffix]
        with patch("time.perf_counter", side_effect=[0.0, 1.0, 1.0, 2.0]):
            rank.store.batch_set_v1(keys, _page_indices([0, 1]))
            for suffix in short_chunks:
                del cluster.objects[f"org-model_p1_{suffix}"]
            rank.store.batch_set_v1(keys, _page_indices([0, 1]))
        self.assertEqual(rank.store.backup_pgs, [2, 1])
        self.assertEqual(
            rank.store.backup_bandwidth,
            [
                2 * rank.layout.bytes_per_page / (1 << 30),
                len(short_chunks) * rank.layout.layer_bytes / (1 << 30),
            ],
        )

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

    # Packed draft layers: one more chunk per head group under the draft
    # digest. The store treats it like any other chunk of the rank's list.

    def test_tp2_writers_fill_a_tp1_readers_draft_rows(self):
        cluster = _FakeCluster()
        writers = [
            _Rank(
                cluster,
                mla=False,
                tp_rank=r,
                tp_size=2,
                head_partition=1,
                draft_layers=1,
            )
            for r in range(2)
        ]
        reader = _Rank(
            cluster, mla=False, tp_rank=0, tp_size=1, head_partition=1, draft_layers=1
        )
        logical = _logical_pages(mla=False, count=1, draft_layers=1)
        for host_page, writer in zip([1, 2], writers):
            writer.fill_from(logical, host_pages=[host_page])
            self.assertEqual(
                writer.store.batch_set_v1(["p"], _page_indices([host_page])), [True]
            )
        # Each object holds the draft rows of its GLOBAL head group, K then V:
        # checked on the stored bytes, so a consistent swap cannot hide.
        draft_keys = [key for key in _chunk_keys(cluster) if "_ukd1-" in key]
        self.assertEqual(len(draft_keys), TOTAL_KV_HEADS)
        for head in range(TOTAL_KV_HEADS):
            (key,) = [k for k in draft_keys if k.endswith(f"_D0-1_H{head}")]
            rows = logical[0][LAYERS:, :, :, head].contiguous()
            self.assertEqual(
                cluster.objects[key], rows.view(torch.uint8).numpy().tobytes()
            )
        self.assertEqual(reader.store.batch_exists(["p"]), 1)
        self.assertEqual(reader.store.batch_get_v1(["p"], _page_indices([0])), [True])
        self.assertTrue(
            torch.equal(reader.pool.kv_buffer[0], reader.expected_page(logical[0]))
        )

    def test_mla_draft_rows_follow_the_final_pp_stage(self):
        cluster = _FakeCluster()
        stages = [
            _Rank(
                cluster,
                mla=True,
                tp_rank=0,
                tp_size=2,
                head_partition=None,
                start_layer=start,
                end_layer=end,
                pp_rank=pp_rank,
                pp_size=2,
                draft_layers=draft_layers,
            )
            for pp_rank, (start, end, draft_layers) in enumerate(
                [(0, 2, 0), (2, LAYERS, 1)]
            )
        ]
        reader = _Rank(
            cluster, mla=True, tp_rank=1, tp_size=4, head_partition=None, draft_layers=1
        )
        logical = _logical_pages(mla=True, count=1, draft_layers=1)
        for host_page, stage in zip([1, 2], stages):
            stage.fill_from(logical, host_pages=[host_page])
            self.assertEqual(
                stage.store.batch_set_v1(["p"], _page_indices([host_page])), [True]
            )
        self.assertFalse(any("ukd1-" in suffix for suffix in stages[0].plan.suffixes))
        (draft_key,) = [key for key in _chunk_keys(cluster) if "_ukd1-" in key]
        self.assertTrue(draft_key.endswith("_D0-1"))
        self.assertEqual(
            cluster.objects[draft_key],
            logical[0][LAYERS:].contiguous().view(torch.uint8).numpy().tobytes(),
        )
        self.assertEqual(reader.store.batch_get_v1(["p"], _page_indices([0])), [True])
        self.assertTrue(torch.equal(reader.pool.kv_buffer[0], logical[0]))

    def test_readers_with_and_without_draft_kv_share_the_target_chunks(self):
        cluster = _FakeCluster()
        kwargs = dict(mla=False, tp_rank=0, tp_size=1, head_partition=2)
        plain = _Rank(cluster, **kwargs)
        ours = _Rank(cluster, draft_layers=1, **kwargs)
        theirs = _Rank(cluster, draft_layers=1, draft_id="another-draft", **kwargs)
        logical = _logical_pages(mla=False, count=1, draft_layers=1)

        # A target-only page is a miss for a reader that stores draft KV.
        plain.fill_from(logical, host_pages=[0])
        self.assertEqual(plain.store.batch_set_v1(["p"], _page_indices([0])), [True])
        target_objects = dict(cluster.objects)
        self.assertEqual(ours.store.batch_exists(["p"]), 0)

        # Its backup adds its draft chunks and leaves the target's alone.
        ours.fill_from(logical, host_pages=[0])
        self.assertEqual(ours.store.batch_set_v1(["p"], _page_indices([0])), [True])
        added = set(cluster.objects) - set(target_objects)
        self.assertEqual(len(added), 2)
        self.assertTrue(all("_ukd1-" in key for key in added))
        self.assertEqual(
            {key: cluster.objects[key] for key in target_objects}, target_objects
        )
        self.assertEqual(ours.store.batch_exists(["p"]), 1)

        # Another draft misses on the draft chunks alone; the reader without
        # draft KV never needed them.
        self.assertEqual(theirs.store.batch_exists(["p"]), 0)
        for key in added:
            del cluster.objects[key]
        self.assertEqual(plain.store.batch_exists(["p"]), 1)
        self.assertEqual(plain.store.batch_get_v1(["p"], _page_indices([1])), [True])
        self.assertTrue(
            torch.equal(plain.pool.kv_buffer[1], plain.expected_page(logical[0]))
        )

    def test_a_missing_or_short_draft_chunk_fails_its_page(self):
        cluster = _FakeCluster()
        rank = _Rank(
            cluster, mla=False, tp_rank=0, tp_size=1, head_partition=2, draft_layers=1
        )
        keys = ["p0", "p1", "p2"]
        rank.store.batch_set_v1(keys, _page_indices([0, 1, 2]))
        draft_suffix = rank.plan.suffixes[-1]
        self.assertIn("ukd1-", draft_suffix)
        short_key = f"org-model_p0_{draft_suffix}"
        cluster.objects[short_key] = cluster.objects[short_key][:-16]
        del cluster.objects[f"org-model_p1_{draft_suffix}"]
        self.assertEqual(
            rank.store.batch_get_v1(keys, _page_indices([0, 1, 2])),
            [False, False, True],
        )
        # Existence is by key: the short object counts, the missing one cuts
        # the prefix.
        self.assertEqual(rank.store.batch_exists(keys), 1)

    def test_page_bytes_without_a_chunk_are_refused_at_registration(self):
        """Reads are zero-copy into the host page: an unnamed byte stays stale."""
        for mla in (False, True):
            with self.subTest(mla=mla):
                cluster = _FakeCluster()
                with self.assertRaisesRegex(ValueError, "unified chunk spans"):
                    _Rank(
                        cluster,
                        mla=mla,
                        tp_rank=0,
                        tp_size=1,
                        head_partition=2,
                        unnamed_layers=1,
                    )
                self.assertEqual(cluster.registered_buffers, 0)

    def test_pools_without_planned_chunks_are_refused_when_they_register(self):
        """Not at their first IO, where the refusal is only logged."""
        cluster = _FakeCluster()
        rank = _Rank(cluster, mla=False, tp_rank=0, tp_size=1, head_partition=2)
        registered = cluster.registered_buffers
        for name in (PoolName.DRAFT, PoolName.MAMBA):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    NotImplementedError, f"hybrid pools: {name} has no unified keys"
                ),
            ):
                rank.store.register_mem_host_pool_v2(object(), name)
            self.assertNotIn(name, getattr(rank.store, "registered_pools", {}))
        self.assertEqual(cluster.registered_buffers, registered)
        # The KV anchor's own v2 registration stays a no-op.
        rank.store.register_mem_host_pool_v2(object(), PoolName.KV)

    def test_hybrid_pool_transfers_are_refused(self):
        cluster = _FakeCluster()
        rank = _Rank(cluster, mla=False, tp_rank=0, tp_size=1, head_partition=2)
        rank.store.batch_set_v1(["p"], _page_indices([0]))
        transfer = PoolTransfer(name=PoolName.MAMBA, keys=["p"])
        with self.assertRaisesRegex(NotImplementedError, "hybrid pools"):
            rank.store.batch_exists_v2(["p"], [transfer])


# A draft KV pool beside the target's (DFLASH without its ring): 3 layers of
# 16-wide rows on 4 kv heads, where the target has 4 layers of 8-wide rows.
DRAFT_LAYERS = 3
DRAFT_HEAD_DIM = 16


def _draft_logical_pages(count):
    generator = torch.Generator().manual_seed(20261120)
    shape = (DRAFT_LAYERS, 2, PAGE_SIZE, TOTAL_KV_HEADS, DRAFT_HEAD_DIM)
    return [
        torch.randn(shape, generator=generator).to(torch.bfloat16) for _ in range(count)
    ]


class _DraftRank(_Rank):
    """A rank whose store also holds the planned draft KV pool."""

    def __init__(self, cluster, *, tp_rank, tp_size, head_partition, draft_id="x"):
        super().__init__(
            cluster,
            mla=False,
            tp_rank=tp_rank,
            tp_size=tp_size,
            head_partition=head_partition,
        )
        local_heads = TOTAL_KV_HEADS // tp_size
        self.draft_layout = PageUnifiedLayout(
            page_size=PAGE_SIZE,
            layer_num=DRAFT_LAYERS,
            head_num=local_heads,
            head_group_num=local_heads // head_partition,
            head_dim=DRAFT_HEAD_DIM,
            itemsize=2,
        )
        self.draft_plan = plan_unified_draft_sidecar(
            target=self.plan.namespace,
            algorithm="DFLASH",
            draft_id=draft_id,
            dtype="bfloat16",
            rank_replicated=False,
            local_kv_heads=local_heads,
            row_width=DRAFT_HEAD_DIM,
            layer_num=DRAFT_LAYERS,
            attn_tp_rank=tp_rank,
            attn_tp_size=tp_size,
            attn_cp_size=1,
            head_group=head_partition,
        )
        self.draft_pool = _UnifiedHostPool(self.draft_layout)
        # The controller adds the plan before it registers the pool.
        self.config.unified_pool_chunks[PoolName.DRAFT] = UnifiedPoolChunks(
            suffixes=tuple(self.draft_plan.suffixes),
            spans=tuple(self.draft_plan.chunk_spans(self.draft_layout)),
            is_writer=True,
        )
        registered = cluster.registered_buffers
        self.store.register_mem_host_pool_v2(self.draft_pool, PoolName.DRAFT)
        # Zero-copy IO needs the pool's buffer registered with the store.
        assert cluster.registered_buffers == registered + 1
        buffer = self.draft_pool.kv_buffer
        assert cluster.registered[-1] == (
            buffer.data_ptr(),
            buffer.numel() * buffer.element_size(),
        )

    def expected_draft_page(self, page):
        layout = self.draft_layout
        heads = page[:, :, :, self.first_head : self.first_head + layout.head_num]
        return heads.reshape(
            DRAFT_LAYERS,
            2,
            PAGE_SIZE,
            layout.head_group_num,
            layout.heads_per_group,
            DRAFT_HEAD_DIM,
        ).permute(3, 0, 1, 2, 4, 5)

    def fill_draft_from(self, logical, host_pages):
        for host_page, page in zip(host_pages, logical):
            self.draft_pool.kv_buffer[host_page] = self.expected_draft_page(page)

    def transfer(self, keys, host_pages):
        return PoolTransfer(
            name=PoolName.DRAFT, keys=keys, host_indices=_page_indices(host_pages)
        )


class TestMooncakeUnifiedDraftSidecar(CustomTestCase):
    # Stored under the model name's key prefix: "org-model_{key}_{suffix}".
    KEYS = ["page0", "page1", "page2"]

    def _write(self, cluster, tp_size, head_partition, logical, **kwargs):
        writers = [
            _DraftRank(
                cluster,
                tp_rank=r,
                tp_size=tp_size,
                head_partition=head_partition,
                **kwargs,
            )
            for r in range(tp_size)
        ]
        for writer in writers:
            writer.fill_draft_from(logical, host_pages=[2, 0, 1])
            self.assertEqual(
                writer.store.batch_set_v2([writer.transfer(self.KEYS, [2, 0, 1])]),
                {PoolName.DRAFT: [True] * 3},
            )
        return writers

    def test_draft_chunks_reshard_across_tp_sizes(self):
        for writer_tp, reader_tp, head_partition in ((2, 1, 1), (1, 2, 1), (2, 1, 2)):
            with self.subTest(writer=writer_tp, reader=reader_tp, hp=head_partition):
                cluster = _FakeCluster()
                logical = _draft_logical_pages(len(self.KEYS))
                self._write(cluster, writer_tp, head_partition, logical)
                groups = TOTAL_KV_HEADS // head_partition
                self.assertEqual(len(_chunk_keys(cluster)), len(self.KEYS) * groups)
                # Each object: one head group's draft layers, K and V, straight
                # from the logical page.
                for key, page in zip(self.KEYS, logical):
                    for group in range(groups):
                        (name,) = [
                            k
                            for k in _chunk_keys(cluster)
                            if k.startswith(f"org-model_{key}_ukd1-")
                            and k.endswith(f"_D0-{DRAFT_LAYERS}_H{group}")
                        ]
                        heads = slice(
                            group * head_partition, (group + 1) * head_partition
                        )
                        self.assertEqual(
                            cluster.objects[name],
                            page[:, :, :, heads]
                            .contiguous()
                            .view(torch.uint8)
                            .numpy()
                            .tobytes(),
                        )
                for tp_rank in range(reader_tp):
                    reader = _DraftRank(
                        cluster,
                        tp_rank=tp_rank,
                        tp_size=reader_tp,
                        head_partition=head_partition,
                    )
                    # No target objects were written: the draft pool's own
                    # verdict is complete, the page is not a hit.
                    result = reader.store.batch_exists_v2(
                        self.KEYS, [reader.transfer(self.KEYS, [1, 2, 0])]
                    )
                    self.assertEqual(result.kv_hit_pages, 0)
                    self.assertEqual(
                        reader.store.batch_get_v2(
                            [reader.transfer(self.KEYS, [1, 2, 0])]
                        ),
                        {PoolName.DRAFT: [True] * 3},
                    )
                    for host_page, page in zip([1, 2, 0], logical):
                        self.assertTrue(
                            torch.equal(
                                reader.draft_pool.kv_buffer[host_page],
                                reader.expected_draft_page(page),
                            )
                        )

    def _with_target(self, rank, logical_kv):
        rank.fill_from(logical_kv, host_pages=[0, 1, 2])
        self.assertEqual(
            rank.store.batch_set_v1(self.KEYS, _page_indices([0, 1, 2])), [True] * 3
        )

    def test_a_page_is_a_hit_only_with_every_draft_chunk(self):
        cluster = _FakeCluster()
        logical = _draft_logical_pages(len(self.KEYS))
        (writer,) = self._write(cluster, 1, 2, logical)
        self._with_target(writer, _logical_pages(mla=False, count=3))
        transfer = writer.transfer(self.KEYS, [0, 1, 2])
        result = writer.store.batch_exists_v2(self.KEYS, [transfer])
        self.assertEqual(result.kv_hit_pages, 3)
        # Without its second draft chunk, page 1 ends the prefix.
        del cluster.objects[f"org-model_page1_{writer.draft_plan.suffixes[1]}"]
        result = writer.store.batch_exists_v2(self.KEYS, [transfer])
        self.assertEqual(result.kv_hit_pages, 1)
        self.assertEqual(result.extra_pool_hit_pages[PoolName.KV], 3)
        self.assertEqual(result.extra_pool_hit_pages[PoolName.DRAFT], 1)
        self.assertEqual(
            writer.store.batch_get_v2([transfer]),
            {PoolName.DRAFT: [True, False, True]},
        )
        # Another draft's reader finds the target pages and no draft page.
        other = _DraftRank(
            cluster, tp_rank=0, tp_size=1, head_partition=2, draft_id="another"
        )
        result = other.store.batch_exists_v2(
            self.KEYS, [other.transfer(self.KEYS, [0, 1, 2])]
        )
        self.assertEqual(result.kv_hit_pages, 0)
        self.assertEqual(result.extra_pool_hit_pages[PoolName.KV], 3)

    def test_short_draft_object_fails_the_read_of_its_page(self):
        """Mooncake reads a shorter object into a larger buffer and reports
        success; the rest of the chunk's span would keep the slot's bytes."""
        cluster = _FakeCluster()
        logical = _draft_logical_pages(len(self.KEYS))
        (writer,) = self._write(cluster, 1, 2, logical)
        name = f"org-model_page1_{writer.draft_plan.suffixes[0]}"
        cluster.objects[name] = cluster.objects[name][:-2]
        reader = _DraftRank(cluster, tp_rank=0, tp_size=1, head_partition=2)
        self.assertEqual(
            reader.store.batch_get_v2([reader.transfer(self.KEYS, [0, 1, 2])]),
            {PoolName.DRAFT: [True, False, True]},
        )

    def test_partial_backup_writes_only_missing_objects_at_their_own_bytes(self):
        """Page 1's draft chunks and one of page 2's exist already: the other
        objects must still come from their own pages and spans."""
        cluster = _FakeCluster()
        logical = _draft_logical_pages(len(self.KEYS))
        writer = _DraftRank(cluster, tp_rank=0, tp_size=1, head_partition=2)
        first, second = writer.draft_plan.suffixes
        sentinels = {
            f"org-model_page1_{first}": b"sentinel-1a",
            f"org-model_page1_{second}": b"sentinel-1b",
            f"org-model_page2_{first}": b"sentinel-2a",
        }
        cluster.objects.update(sentinels)
        writer.fill_draft_from(logical, host_pages=[2, 0, 1])
        self.assertEqual(
            writer.store.batch_set_v2([writer.transfer(self.KEYS, [2, 0, 1])]),
            {PoolName.DRAFT: [True] * 3},
        )
        for name, data in sentinels.items():
            self.assertEqual(cluster.objects[name], data)
        for key, page, groups in ((0, logical[0], (0, 1)), (2, logical[2], (1,))):
            for group in groups:
                heads = slice(group * 2, (group + 1) * 2)
                self.assertEqual(
                    cluster.objects[
                        f"org-model_page{key}_{writer.draft_plan.suffixes[group]}"
                    ],
                    page[:, :, :, heads]
                    .contiguous()
                    .view(torch.uint8)
                    .numpy()
                    .tobytes(),
                )

    def test_a_plan_belongs_to_the_pool_it_was_made_for(self):
        cluster = _FakeCluster()
        rank = _DraftRank(cluster, tp_rank=0, tp_size=1, head_partition=2)
        chunks = rank.config.unified_pool_chunks[PoolName.DRAFT]
        self.assertEqual(rank.store.unified_pool_chunks, {PoolName.DRAFT: chunks})
        # The draft pool's plan does not name another pool.
        with self.assertRaisesRegex(NotImplementedError, "mamba has no unified keys"):
            rank.store.register_mem_host_pool_v2(rank.draft_pool, PoolName.MAMBA)
        # Nor may a second pool repeat the names of one that is registered.
        rank.config.unified_pool_chunks[PoolName.DRAFT_INDEXER] = chunks
        with self.assertRaisesRegex(ValueError, "draft_indexer must be nonempty"):
            rank.store.register_mem_host_pool_v2(
                rank.draft_pool, PoolName.DRAFT_INDEXER
            )
        self.assertEqual(list(rank.store.unified_pool_chunks), [PoolName.DRAFT])

    def test_two_planned_pools_keep_their_own_names_spans_and_bytes(self):
        cluster = _FakeCluster()
        rank = _DraftRank(cluster, tp_rank=0, tp_size=1, head_partition=2)
        first = rank.config.unified_pool_chunks[PoolName.DRAFT]
        # A second pool with a plan of its own: another digest, 4 chunks.
        layout = PageUnifiedLayout(
            page_size=PAGE_SIZE,
            layer_num=DRAFT_LAYERS,
            head_num=TOTAL_KV_HEADS,
            head_group_num=TOTAL_KV_HEADS,
            head_dim=DRAFT_HEAD_DIM,
            itemsize=2,
        )
        plan = plan_unified_draft_sidecar(
            target=rank.plan.namespace,
            algorithm="DFLASH",
            draft_id="second",
            dtype="bfloat16",
            rank_replicated=False,
            local_kv_heads=TOTAL_KV_HEADS,
            row_width=DRAFT_HEAD_DIM,
            layer_num=DRAFT_LAYERS,
            attn_tp_rank=0,
            attn_tp_size=1,
            attn_cp_size=1,
            head_group=1,
        )
        second = UnifiedPoolChunks(
            tuple(plan.suffixes), tuple(plan.chunk_spans(layout)), True
        )
        self.assertEqual(len(second.suffixes), 4)
        self.assertFalse(set(first.suffixes) & set(second.suffixes))
        pool = _UnifiedHostPool(layout)
        name = PoolName.DRAFT_INDEXER
        # One repeated name is enough, whichever of the earlier pool's.
        rank.config.unified_pool_chunks[name] = UnifiedPoolChunks(
            second.suffixes[:3] + first.suffixes[1:], second.spans, True
        )
        with self.assertRaisesRegex(ValueError, "draft_indexer must be nonempty"):
            rank.store.register_mem_host_pool_v2(pool, name)
        rank.config.unified_pool_chunks[name] = second
        rank.store.register_mem_host_pool_v2(pool, name)
        self.assertEqual(
            rank.store.unified_pool_chunks, {PoolName.DRAFT: first, name: second}
        )
        rank.fill_draft_from(_draft_logical_pages(3), host_pages=[0, 1, 2])
        draft_expected = rank.draft_pool.kv_buffer.clone()
        generator = torch.Generator().manual_seed(7)
        pool.kv_buffer.copy_(
            torch.randn(pool.kv_buffer.shape, generator=generator).to(torch.bfloat16)
        )
        expected = pool.kv_buffer.clone()
        transfers = [
            rank.transfer(self.KEYS, [0, 1, 2]),
            PoolTransfer(
                name=name, keys=self.KEYS, host_indices=_page_indices([0, 1, 2])
            ),
        ]
        self.assertEqual(
            rank.store.batch_set_v2(transfers),
            {PoolName.DRAFT: [True] * 3, name: [True] * 3},
        )
        self.assertEqual(
            _chunk_keys(cluster),
            sorted(
                f"org-model_{key}_{suffix}"
                for key in self.KEYS
                for suffix in first.suffixes + second.suffixes
            ),
        )
        for page, key in enumerate(self.KEYS):
            for chunks, source in ((first, draft_expected), (second, expected)):
                raw = source[page].contiguous().view(torch.uint8).numpy().tobytes()
                for suffix, (offset, length) in zip(chunks.suffixes, chunks.spans):
                    self.assertEqual(
                        cluster.objects[f"org-model_{key}_{suffix}"],
                        raw[offset : offset + length],
                    )
        pool.kv_buffer.zero_()
        rank.draft_pool.kv_buffer.zero_()
        self.assertEqual(
            rank.store.batch_get_v2(transfers),
            {PoolName.DRAFT: [True] * 3, name: [True] * 3},
        )
        self.assertTrue(torch.equal(pool.kv_buffer, expected))
        self.assertTrue(torch.equal(rank.draft_pool.kv_buffer, draft_expected))

    def test_a_planned_pools_page_must_be_one_run(self):
        """A chunk's span is an offset into its page's run, as for the KV
        pool: a pool that hands out a page in pieces would be cut wrongly."""
        cluster = _FakeCluster()
        rank = _DraftRank(cluster, tp_rank=0, tp_size=1, head_partition=2)
        rank.fill_draft_from(_draft_logical_pages(3), host_pages=[0, 1, 2])
        one_run = rank.draft_pool.get_page_buffer_meta

        def two_runs(indices):
            ptrs, sizes = one_run(indices)
            return (
                [
                    half
                    for ptr, size in zip(ptrs, sizes)
                    for half in (ptr, ptr + size // 2)
                ],
                [size // 2 for size in sizes for _ in range(2)],
            )

        rank.draft_pool.get_page_buffer_meta = two_runs
        transfers = [rank.transfer(self.KEYS, [0, 1, 2])]
        for io in (rank.store.batch_set_v2, rank.store.batch_get_v2):
            with self.subTest(io=io.__name__), self.assertRaises(AssertionError):
                io(transfers)
        self.assertEqual(_chunk_keys(cluster), [])
        rank.draft_pool.get_page_buffer_meta = one_run
        self.assertEqual(
            rank.store.batch_set_v2(transfers), {PoolName.DRAFT: [True] * 3}
        )

    def test_planned_chunks_must_name_the_pools_page_and_no_other_object(self):
        cluster = _FakeCluster()
        rank = _DraftRank(cluster, tp_rank=0, tp_size=1, head_partition=2)
        chunks = rank.config.unified_pool_chunks[PoolName.DRAFT]
        cases = {
            "gap": UnifiedPoolChunks(chunks.suffixes, chunks.spans[:1] * 2, True),
            "unpaired": UnifiedPoolChunks(chunks.suffixes[:1], chunks.spans, True),
            "duplicate": UnifiedPoolChunks(chunks.suffixes[:1] * 2, chunks.spans, True),
            "target": UnifiedPoolChunks(
                (rank.plan.suffixes[0], chunks.suffixes[1]), chunks.spans, True
            ),
        }
        for name, bad in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                fresh = _Rank(
                    _FakeCluster(), mla=False, tp_rank=0, tp_size=1, head_partition=2
                )
                fresh.config.unified_pool_chunks[PoolName.DRAFT] = bad
                fresh.store.register_mem_host_pool_v2(rank.draft_pool, PoolName.DRAFT)


if __name__ == "__main__":
    unittest.main()
