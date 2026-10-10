"""Unit tests for the SeaweedFS HiCache storage backend.

Pages move through the cache controller's own generic get/set paths and a real CPU host pool;
only the S3 client is replaced, by an in-memory store with S3's miss semantics.

Run with:
    python3 -m pytest test/registered/unit/mem_cache/test_hicache_seaweedfs_storage.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import io
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.storage import backend_factory
from sglang.srt.mem_cache.storage.seaweedfs import seaweedfs_store
from sglang.test.test_utils import CustomTestCase

PAGE_SIZE = 16
NUM_PAGES = 4


class _ClientError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class InMemoryS3:
    """The subset of the boto3 S3 client the backend uses, with S3's miss behaviour."""

    exceptions = SimpleNamespace(ClientError=_ClientError)

    def __init__(self, buckets: dict):
        self.buckets = buckets

    def _bucket(self, bucket):
        if bucket not in self.buckets:
            raise _ClientError("NoSuchBucket")
        return self.buckets[bucket]

    def head_bucket(self, Bucket):
        self._bucket(Bucket)

    def create_bucket(self, Bucket):
        self.buckets.setdefault(Bucket, {})

    def put_object(self, Bucket, Key, Body):
        self._bucket(Bucket)[Key] = bytes(Body)

    def get_object(self, Bucket, Key):
        objects = self._bucket(Bucket)
        if Key not in objects:
            raise _ClientError("NoSuchKey")
        return {"Body": io.BytesIO(objects[Key])}

    def head_object(self, Bucket, Key):
        if Key not in self._bucket(Bucket):
            raise _ClientError("404")

    def delete_object(self, Bucket, Key):
        self._bucket(Bucket).pop(Key, None)

    def delete_objects(self, Bucket, Delete):
        for obj in Delete["Objects"]:
            self._bucket(Bucket).pop(obj["Key"], None)

    def get_paginator(self, name):
        store = self

        class _Paginator:
            def paginate(self, Bucket, Prefix):
                keys = [k for k in store._bucket(Bucket) if k.startswith(Prefix)]
                yield {"Contents": [{"Key": k} for k in keys]}

        return _Paginator()


def _host_pool(layout: str) -> MHATokenToKVPoolHost:
    device_pool = SimpleNamespace(
        store_dtype=torch.bfloat16,
        size=64,
        layer_num=4,
        start_layer=0,
        end_layer=4,
        head_num=2,
        head_dim=64,
        v_head_dim=64,
        row_dim=128,
        device="cpu",
        layer_shard_enabled=False,
        hicache_write_back_staging=None,
    )
    return MHATokenToKVPoolHost(
        device_pool=device_pool,
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=PAGE_SIZE,
        layout=layout,
        pin_memory=False,
        device="cpu",
    )


def _page(pool: MHATokenToKVPoolHost, page: int) -> torch.Tensor:
    return pool.get_data_page(page * PAGE_SIZE, flat=True).view(torch.uint8).clone()


class TestSeaweedFSStore(CustomTestCase):
    endpoint = "http://seaweedfs.invalid:8333"

    def make_client(self, config):
        return InMemoryS3(self.buckets)

    def setUp(self):
        self.buckets = {}
        self.bucket = f"sgl-{uuid.uuid4().hex[:10]}"
        patcher = mock.patch.object(
            seaweedfs_store,
            "_make_s3_client",
            side_effect=lambda config: self.make_client(config),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _config(self, tp_rank=0, tp_size=1, is_mla=False, cp_rank=0, cp_size=1):
        return HiCacheStorageConfig(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=cp_rank,
            attn_cp_size=cp_size,
            is_mla_model=is_mla,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="Qwen/Qwen3-0.6B",
            extra_config={"endpoint": self.endpoint, "bucket": self.bucket},
        )

    def _store(self, **kwargs):
        return backend_factory.StorageBackendFactory.create_backend(
            backend_name="seaweedfs",
            storage_config=self._config(**kwargs),
            mem_pool_host=None,
        )

    def _controller(self, store, pool):
        store.register_mem_pool_host(pool)
        return SimpleNamespace(
            storage_backend=store, storage_host_pool=pool, page_size=PAGE_SIZE
        )

    def _backup_and_reload(
        self, store, layout, drop_page=None, set_via="generic", get_via="generic"
    ):
        """Back up NUM_PAGES random pages, wipe the pool, prefetch them back.

        set_via / get_via pick the controller path: "generic" (flat page tensors) or
        "zero_copy" (batch_get_v1 / batch_set_v1 on the host pool's own buffers).
        """
        pool = _host_pool(layout)
        ctrl = self._controller(store, pool)
        pool.kv_buffer.copy_(torch.randn(pool.kv_buffer.shape).to(torch.bfloat16))
        expected = [_page(pool, i) for i in range(NUM_PAGES)]
        keys = [f"{layout}-{uuid.uuid4().hex[:8]}-{i}" for i in range(NUM_PAGES)]
        host_indices = torch.arange(NUM_PAGES * PAGE_SIZE, dtype=torch.int64)

        page_set = {
            "generic": HiCacheController._generic_page_set,
            "zero_copy": HiCacheController._page_set_zero_copy,
        }[set_via]
        page_get = {
            "generic": HiCacheController._generic_page_get,
            "zero_copy": HiCacheController._page_get_zero_copy,
        }[get_via]
        self.assertTrue(page_set(ctrl, keys, host_indices))
        if drop_page is not None:
            store._s3.delete_object(
                Bucket=self.bucket, Key=store._object_key(keys[drop_page])
            )
        pool.kv_buffer.zero_()
        operation = SimpleNamespace(request_id="r", is_terminated=lambda: False)
        loaded = page_get(ctrl, operation, keys, host_indices)
        return pool, keys, expected, loaded

    def test_controller_round_trip_is_bit_exact_for_every_layout(self):
        store = self._store()
        for via in ("generic", "zero_copy"):
            for layout in ("layer_first", "page_first", "page_first_direct"):
                with self.subTest(via=via, layout=layout):
                    pool, _, expected, loaded = self._backup_and_reload(
                        store, layout, set_via=via, get_via=via
                    )
                    self.assertEqual(loaded, NUM_PAGES)
                    for i in range(NUM_PAGES):
                        self.assertTrue(torch.equal(_page(pool, i), expected[i]))

    def test_generic_and_zero_copy_read_each_others_pages(self):
        """Both interfaces store a page as one object with the same bytes (page-first layouts)."""
        store = self._store()
        for layout in ("page_first", "page_first_direct"):
            for set_via, get_via in (
                ("generic", "zero_copy"),
                ("zero_copy", "generic"),
            ):
                with self.subTest(layout=layout, set_via=set_via):
                    pool, _, expected, loaded = self._backup_and_reload(
                        store, layout, set_via=set_via, get_via=get_via
                    )
                    self.assertEqual(loaded, NUM_PAGES)
                    for i in range(NUM_PAGES):
                        self.assertTrue(torch.equal(_page(pool, i), expected[i]))

    def test_prefetch_stops_at_the_first_missing_page(self):
        """Pages after a miss are unusable and must not be loaded or counted."""
        store = self._store()
        for via in ("generic", "zero_copy"):
            with self.subTest(via=via):
                pool, keys, _, loaded = self._backup_and_reload(
                    store, "page_first", drop_page=2, set_via=via, get_via=via
                )
                self.assertEqual(loaded, 2)
                if via == "generic":
                    # Zero-copy fetches a batch in parallel into slots the controller
                    # releases after the count; only the generic path skips the copy.
                    self.assertFalse(_page(pool, 3).any())
                extra_info = HiCacheStorageExtraInfo(prefix_keys=[])
                self.assertEqual(store.batch_exists(keys, extra_info), 2)

    def test_zero_copy_rejects_an_object_of_the_wrong_size(self):
        """A stale or foreign object must not be copied into the host pool."""
        store = self._store()
        pool = _host_pool("page_first")
        pool.kv_buffer.zero_()
        store.register_mem_pool_host(pool)
        store._s3.put_object(
            Bucket=self.bucket, Key=store._object_key("k"), Body=b"\x7f" * 8192
        )
        host_indices = torch.arange(PAGE_SIZE, dtype=torch.int64)
        self.assertEqual(store.batch_get_v1(["k"], host_indices), [False])
        self.assertFalse(pool.kv_buffer.any())

    def test_object_of_the_wrong_size_is_a_miss_not_an_overrun(self):
        store = self._store()
        store._s3.put_object(
            Bucket=self.bucket, Key=store._object_key("k"), Body=b"\x7f" * 8192
        )
        arena = torch.zeros(8192, dtype=torch.uint8)
        self.assertIsNone(store.get("k", arena[:4096]))
        self.assertFalse(arena.any())

    def test_mha_and_context_parallel_ranks_do_not_share_objects(self):
        for kwargs in (
            [dict(tp_rank=0, tp_size=2), dict(tp_rank=1, tp_size=2)],
            [dict(cp_rank=0, cp_size=2), dict(cp_rank=1, cp_size=2)],
        ):
            with self.subTest(ranks=kwargs):
                first, second = self._store(**kwargs[0]), self._store(**kwargs[1])
                first.set("shared", torch.full((64,), 1, dtype=torch.uint8))
                second.set("shared", torch.full((64,), 2, dtype=torch.uint8))
                self.assertEqual(int(first.get("shared")[0]), 1)
                second.clear()
                self.assertTrue(first.exists("shared"))

    def test_mla_ranks_share_objects(self):
        """MLA ranks hold identical KV and only one backs up, so the others must find it."""
        writer = self._store(tp_rank=0, tp_size=2, is_mla=True)
        reader = self._store(tp_rank=1, tp_size=2, is_mla=True)
        writer.set("shared", torch.full((64,), 3, dtype=torch.uint8))
        self.assertTrue(reader.exists("shared"))

    # ---- v2 interface (hybrid models): KV plus a side pool -----------------
    # A second real host pool stands in for the side pool (Mamba/SWA/indexer):
    # the backend only touches it through the host-pool page API, so what this
    # proves is the per-pool object naming, routing and hit policies.

    def _v2_store(self, layout="page_first", **store_kwargs):
        store = self._store(**store_kwargs)
        kv, side = _host_pool(layout), _host_pool(layout)
        store.register_mem_host_pool_v2(kv, PoolName.KV)
        store.register_mem_host_pool_v2(side, PoolName.MAMBA)
        for pool in (kv, side):
            pool.kv_buffer.copy_(torch.randn(pool.kv_buffer.shape).to(torch.bfloat16))
        keys = [f"v2-{uuid.uuid4().hex[:8]}-{i}" for i in range(NUM_PAGES)]
        indices = torch.arange(NUM_PAGES * PAGE_SIZE, dtype=torch.int64)
        transfers = [
            PoolTransfer(name=PoolName.KV, host_indices=indices, keys=keys),
            PoolTransfer(name=PoolName.MAMBA, host_indices=indices, keys=keys),
        ]
        return store, kv, side, keys, transfers

    def test_v2_round_trip_is_bit_exact_per_pool(self):
        for layout in ("layer_first", "page_first", "page_first_direct"):
            with self.subTest(layout=layout):
                store, kv, side, keys, transfers = self._v2_store(layout)
                expected = {
                    p: [_page(pool, i) for i in range(NUM_PAGES)]
                    for p, pool in ((PoolName.KV, kv), (PoolName.MAMBA, side))
                }
                written = store.batch_set_v2(transfers)
                self.assertEqual(written[PoolName.KV], [True] * NUM_PAGES)
                self.assertEqual(written[PoolName.MAMBA], [True] * NUM_PAGES)
                kv.kv_buffer.zero_()
                side.kv_buffer.zero_()
                loaded = store.batch_get_v2(transfers)
                for p, pool in ((PoolName.KV, kv), (PoolName.MAMBA, side)):
                    self.assertEqual(loaded[p], [True] * NUM_PAGES)
                    for i in range(NUM_PAGES):
                        self.assertTrue(torch.equal(_page(pool, i), expected[p][i]))

    def test_v2_mla_mamba_state_stays_per_rank(self):
        # Every MLA TP rank backs up its own Mamba shard; replicated KV is shared.
        ranks = [self._v2_store(tp_rank=r, tp_size=2, is_mla=True) for r in (0, 1)]
        keys = ranks[0][3]
        for store, _, _, _, transfers in ranks:
            transfers[1].keys = keys
            self.assertEqual(
                store.batch_set_v2(transfers[1:])[PoolName.MAMBA], [True] * NUM_PAGES
            )
        for store, _, side, _, transfers in ranks:
            expected = [_page(side, i) for i in range(NUM_PAGES)]
            side.kv_buffer.zero_()
            self.assertEqual(
                store.batch_get_v2(transfers[1:])[PoolName.MAMBA], [True] * NUM_PAGES
            )
            for i in range(NUM_PAGES):
                self.assertTrue(torch.equal(_page(side, i), expected[i]))
        self.assertEqual(
            ranks[0][0]._component_key("k", PoolName.KV),
            ranks[1][0]._component_key("k", PoolName.KV),
        )

    def test_v2_kv_pages_are_the_v1_objects(self):
        # KV pages keep their v1 object names, so v1 and v2 share one cache;
        # side-pool pages are separate objects and never overwrite them.
        store, kv, side, keys, transfers = self._v2_store()
        store.register_mem_pool_host(kv)  # the v1 path reads through this pool
        store.batch_set_v2(transfers)
        self.assertEqual(store.batch_exists(keys), NUM_PAGES)
        self.assertEqual(store.batch_exists([f"{k}.mamba" for k in keys]), NUM_PAGES)
        self.assertEqual(
            store.batch_get_v1(keys, transfers[0].host_indices), [True] * NUM_PAGES
        )

    def test_v2_exists_all_pages_stops_at_a_missing_side_page(self):
        store, _, _, keys, transfers = self._v2_store()
        store.batch_set_v2(transfers)
        store._s3.delete_object(
            Bucket=self.bucket, Key=store._object_key(f"{keys[2]}.mamba")
        )
        side = PoolTransfer(
            name=PoolName.MAMBA, keys=keys, hit_policy=PoolHitPolicy.ALL_PAGES
        )
        result = store.batch_exists_v2(keys, [side])
        self.assertEqual(result.kv_hit_pages, 2)
        self.assertEqual(
            result.extra_pool_hit_pages, {PoolName.KV: NUM_PAGES, PoolName.MAMBA: 2}
        )

    def test_v2_exists_trailing_pages_needs_only_the_window(self):
        store, _, _, keys, transfers = self._v2_store()
        store.batch_set_v2(transfers)
        # Only the last page's side state matters for a one-page window, so a
        # gap earlier in the side pool does not shorten the prefix...
        store._s3.delete_object(
            Bucket=self.bucket, Key=store._object_key(f"{keys[0]}.mamba")
        )
        window = PoolTransfer(
            name=PoolName.MAMBA, keys=keys[-1:], hit_policy=PoolHitPolicy.TRAILING_PAGES
        )
        self.assertEqual(store.batch_exists_v2(keys, [window]).kv_hit_pages, NUM_PAGES)
        # ...but losing the last page's state falls back to the longest prefix
        # whose own last page still has it.
        store._s3.delete_object(
            Bucket=self.bucket, Key=store._object_key(f"{keys[-1]}.mamba")
        )
        result = store.batch_exists_v2(keys, [window])
        self.assertEqual(result.kv_hit_pages, NUM_PAGES - 1)
        self.assertEqual(result.extra_pool_hit_pages[PoolName.MAMBA], NUM_PAGES - 1)

    def test_v2_exists_without_kv_prefix_is_empty(self):
        store, _, _, keys, _ = self._v2_store()
        side = PoolTransfer(name=PoolName.MAMBA, keys=keys)
        result = store.batch_exists_v2(keys, [side])
        self.assertEqual((result.kv_hit_pages, result.extra_pool_hit_pages), (0, {}))

    def test_v2_rejects_mismatched_indices_and_unknown_pools(self):
        store, _, _, keys, _ = self._v2_store()
        short = torch.arange(PAGE_SIZE, dtype=torch.int64)  # one page for four keys
        bad = [
            PoolTransfer(name=PoolName.KV, host_indices=short, keys=keys),
            PoolTransfer(
                name=PoolName.SWA,
                host_indices=torch.arange(NUM_PAGES * PAGE_SIZE),
                keys=keys,
            ),
        ]
        for op in (store.batch_set_v2, store.batch_get_v2):
            result = op(bad)
            self.assertEqual(result[PoolName.KV], [False] * NUM_PAGES)
            self.assertEqual(result[PoolName.SWA], [False] * NUM_PAGES)
        self.assertEqual(store.batch_exists(keys), 0)

    def test_missing_endpoint_is_refused(self):
        with self.assertRaises(ValueError):
            seaweedfs_store.SeaweedFSConfig.from_extra_config({"bucket": "b"})


if __name__ == "__main__":
    unittest.main()
