"""Unit tests for the GCS HiCache storage backend.

No network or cloud SDK is used: only the GCS client factory is replaced by an
in-memory bucket with the SDK's miss behaviour. Pages are moved through the
backend's own generic get/set and v2 pool interfaces with a real CPU host
pool-shaped buffer.

Run with:
    python3 -m pytest test/registered/unit/mem_cache/test_hicache_gcs_storage.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import sys
import unittest
import uuid
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    resolve_storage_key_scheme,
)
from sglang.srt.mem_cache.storage import backend_factory
from sglang.srt.mem_cache.storage.gcs import GCSConfig, GcsStore
from sglang.srt.mem_cache.storage.gcs import gcs_store
from sglang.test.test_utils import CustomTestCase

PAGE_SIZE = 16
NUM_PAGES = 4
WIDTH = 64


class _NotFound(Exception):
    """google.api_core.exceptions.NotFound exposes the HTTP code."""

    code = 404


class _FakeBlob:
    def __init__(self, objects: dict, name: str):
        self._objects = objects
        self.name = name

    def exists(self) -> bool:
        return self.name in self._objects

    def upload_from_string(self, data, content_type=None) -> None:
        self._objects[self.name] = bytes(data)

    def download_as_bytes(self) -> bytes:
        if self.name not in self._objects:
            raise _NotFound(self.name)
        return self._objects[self.name]

    def delete(self) -> None:
        if self.name not in self._objects:
            raise _NotFound(self.name)
        del self._objects[self.name]


class _FakeBucket:
    def __init__(self):
        self.objects = {}

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self.objects, name)

    def list_blobs(self, prefix: str = ""):
        for name in list(self.objects):
            if name.startswith(prefix):
                yield _FakeBlob(self.objects, name)


class _FakeClient:
    def __init__(self):
        self.buckets = {}

    def bucket(self, name: str) -> _FakeBucket:
        return self.buckets.setdefault(name, _FakeBucket())


class _HostPool:
    """The host-pool page API the backend uses, backed by one flat buffer."""

    def __init__(self, num_pages=NUM_PAGES, page_size=PAGE_SIZE, width=WIDTH):
        self.page_size = page_size
        self.dcp_size = 1
        self.rows = torch.zeros(num_pages * page_size, width, dtype=torch.bfloat16)

    def get_data_page(self, offset, flat=False):
        page = self.rows[offset : offset + self.page_size]
        return page.reshape(-1) if flat else page

    def get_dummy_flat_data_page(self):
        return torch.empty(self.page_size * self.rows.shape[1], dtype=self.rows.dtype)

    def set_from_flat_data_page(self, offset, page):
        self.rows[offset : offset + self.page_size] = page.reshape(
            self.page_size, self.rows.shape[1]
        )


def _page(pool: _HostPool, page: int) -> torch.Tensor:
    return pool.get_data_page(page * PAGE_SIZE, flat=True).view(torch.uint8).clone()


def _bytes_per_page() -> int:
    return PAGE_SIZE * WIDTH * 2


class TestGcsStore(CustomTestCase):
    def setUp(self):
        self.client = _FakeClient()
        self.bucket = f"sgl-{uuid.uuid4().hex[:10]}"
        self._real_make_client = gcs_store._make_gcs_client
        patcher = mock.patch.object(
            gcs_store, "_make_gcs_client", side_effect=lambda config: self.client
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _config(self, unified_suffixes=None, unified_chunk_spans=None, **kwargs):
        values = dict(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="Qwen/Qwen3-0.6B",
            extra_config={"bucket": self.bucket, "prefix": "hicache-test"},
        )
        values.update(kwargs)
        if unified_suffixes is not None:
            values["unified_suffixes"] = unified_suffixes
            values["unified_chunk_spans"] = unified_chunk_spans
        return HiCacheStorageConfig(**values)

    def _store(self, **kwargs):
        return GcsStore(self._config(**kwargs))

    def _new_pool(self):
        pool = _HostPool()
        pool.rows.normal_()
        return pool

    def test_factory_creates_the_backend(self):
        store = backend_factory.StorageBackendFactory.create_backend(
            backend_name="gcs",
            storage_config=self._config(),
            mem_pool_host=None,
        )
        self.assertIsInstance(store, GcsStore)
        self.assertEqual(store.config.bucket, self.bucket)

    def test_round_trip_is_bit_exact(self):
        store = self._store()
        pool = self._new_pool()
        expected = [_page(pool, i) for i in range(NUM_PAGES)]
        keys = [f"k-{i}" for i in range(NUM_PAGES)]

        self.assertTrue(store.set(keys[0], pool.get_data_page(0, flat=True)))
        self.assertTrue(store.exists(keys[0]))
        restored = torch.empty_like(expected[0])
        self.assertIs(store.get(keys[0], restored), restored)
        self.assertTrue(torch.equal(restored, expected[0]))

        self.assertTrue(
            store.batch_set(
                keys, [pool.get_data_page(i * PAGE_SIZE) for i in range(NUM_PAGES)]
            )
        )
        outputs = [torch.empty_like(page) for page in expected]
        store.batch_get(keys, outputs)
        for output, page in zip(outputs, expected):
            self.assertTrue(torch.equal(output, page))

    def test_wrong_size_object_is_a_miss_not_an_overrun(self):
        store = self._store()
        object_name = store._object_name("truncated", store.config_suffix)
        store._bucket.blob(object_name).upload_from_string(b"\x7f" * 7)
        target = torch.zeros(_bytes_per_page(), dtype=torch.uint8)
        self.assertIsNone(store.get("truncated", target))
        self.assertFalse(target.any())

    def test_batch_exists_stops_at_the_first_missing_page(self):
        store = self._store()
        pool = self._new_pool()
        keys = [f"prefix-{i}" for i in range(NUM_PAGES)]
        store.batch_set(
            keys, [pool.get_data_page(i * PAGE_SIZE) for i in range(NUM_PAGES)]
        )
        self.assertEqual(store.batch_exists(keys), NUM_PAGES)
        store._bucket.blob(store._object_name(keys[2], store.config_suffix)).delete()
        self.assertEqual(store.batch_exists(keys), 2)

    def test_mha_and_context_parallel_ranks_do_not_share_objects(self):
        for kwargs in (
            (dict(tp_rank=0, tp_size=2), dict(tp_rank=1, tp_size=2)),
            (dict(cp_rank=0, cp_size=2), dict(cp_rank=1, cp_size=2)),
        ):
            with self.subTest(ranks=kwargs):
                first = self._store(
                    **{
                        "attn_cp_rank": kwargs[0].get("cp_rank", 0),
                        "attn_cp_size": kwargs[0].get("cp_size", 1),
                        "tp_rank": kwargs[0].get("tp_rank", 0),
                        "tp_size": kwargs[0].get("tp_size", 1),
                    }
                )
                second = self._store(
                    **{
                        "attn_cp_rank": kwargs[1].get("cp_rank", 0),
                        "attn_cp_size": kwargs[1].get("cp_size", 1),
                        "tp_rank": kwargs[1].get("tp_rank", 0),
                        "tp_size": kwargs[1].get("tp_size", 1),
                    }
                )
                first.set("shared", torch.full((64,), 1, dtype=torch.uint8))
                second.set("shared", torch.full((64,), 2, dtype=torch.uint8))
                self.assertEqual(int(first.get("shared")[0]), 1)
                self.assertNotEqual(
                    first._component_key("shared", PoolName.KV),
                    second._component_key("shared", PoolName.KV),
                )

    def test_mla_ranks_share_objects(self):
        writer = self._store(tp_rank=0, tp_size=2, is_mla_model=True)
        reader = self._store(tp_rank=1, tp_size=2, is_mla_model=True)
        writer.set("shared", torch.full((64,), 3, dtype=torch.uint8))
        self.assertTrue(reader.exists("shared"))

    def test_clear_only_touches_this_prefix(self):
        mine, other = (
            self._store(),
            self._store(
                extra_config={
                    "bucket": self.bucket,
                    "prefix": "another-deployment",
                }
            ),
        )
        mine.set("k", torch.ones(16, dtype=torch.uint8))
        other.set("k", torch.ones(16, dtype=torch.uint8))
        mine.clear()
        self.assertFalse(mine.exists("k"))
        self.assertTrue(other.exists("k"))

    def _v2_setup(self, **store_kwargs):
        store = self._store(**store_kwargs)
        kv, side = self._new_pool(), self._new_pool()
        store.register_mem_host_pool_v2(kv, PoolName.KV)
        store.register_mem_host_pool_v2(side, PoolName.MAMBA)
        keys = [f"v2-{i}" for i in range(NUM_PAGES)]
        indices = torch.arange(NUM_PAGES * PAGE_SIZE, dtype=torch.int64)
        transfers = [
            PoolTransfer(name=PoolName.KV, host_indices=indices, keys=keys),
            PoolTransfer(name=PoolName.MAMBA, host_indices=indices, keys=keys),
        ]
        return store, kv, side, keys, transfers

    def test_v2_round_trip_is_bit_exact_per_pool(self):
        store, kv, side, keys, transfers = self._v2_setup()
        expected = {
            name: [_page(pool, i) for i in range(NUM_PAGES)]
            for name, pool in ((PoolName.KV, kv), (PoolName.MAMBA, side))
        }
        written = store.batch_set_v2(transfers)
        self.assertEqual(written[PoolName.KV], [True] * NUM_PAGES)
        self.assertEqual(written[PoolName.MAMBA], [True] * NUM_PAGES)
        kv.rows.zero_()
        side.rows.zero_()
        loaded = store.batch_get_v2(transfers)
        for name, pool in ((PoolName.KV, kv), (PoolName.MAMBA, side)):
            self.assertEqual(loaded[name], [True] * NUM_PAGES)
            for i in range(NUM_PAGES):
                self.assertTrue(torch.equal(_page(pool, i), expected[name][i]))

    def test_v2_hit_policies_follow_the_kv_prefix(self):
        store, _, _, keys, transfers = self._v2_setup()
        store.batch_set_v2(transfers)
        store._bucket.blob(store._component_key(keys[2], PoolName.MAMBA)[0]).delete()

        all_pages = PoolTransfer(name=PoolName.MAMBA, keys=keys)
        result = store.batch_exists_v2(keys, [all_pages])
        self.assertEqual(result.kv_hit_pages, 2)
        self.assertEqual(
            result.extra_pool_hit_pages, {PoolName.KV: NUM_PAGES, PoolName.MAMBA: 2}
        )

        store._bucket.blob(store._component_key(keys[0], PoolName.MAMBA)[0]).delete()
        window = PoolTransfer(
            name=PoolName.MAMBA, keys=keys[-1:], hit_policy=PoolHitPolicy.TRAILING_PAGES
        )
        result = store.batch_exists_v2(keys, [window])
        self.assertEqual(result.kv_hit_pages, NUM_PAGES)

        store._bucket.blob(store._component_key(keys[-1], PoolName.MAMBA)[0]).delete()
        result = store.batch_exists_v2(keys, [window])
        # Page 2's side state is missing, so the longest restorable prefix
        # whose own trailing window is present ends at page 2.
        self.assertEqual(result.kv_hit_pages, 2)

    def test_v2_rejects_mismatched_indices_and_unknown_pools(self):
        store, _, _, keys, _ = self._v2_setup()
        short = torch.arange(PAGE_SIZE, dtype=torch.int64)
        bad = [
            PoolTransfer(name=PoolName.KV, host_indices=short, keys=keys),
            PoolTransfer(
                name=PoolName.SWA,
                host_indices=torch.arange(NUM_PAGES * PAGE_SIZE),
                keys=keys,
            ),
        ]
        for operation in (store.batch_set_v2, store.batch_get_v2):
            result = operation(bad)
            self.assertEqual(result[PoolName.KV], [False] * NUM_PAGES)
            self.assertEqual(result[PoolName.SWA], [False] * NUM_PAGES)

    def test_unified_chunks_share_across_tp_topologies(self):
        """Keys name layer/head chunks, so a different TP split reads them."""
        half = _bytes_per_page() // 2
        suffixes = ["ukv1-abc_L0-2_H0", "ukv1-abc_L0-2_H1"]
        spans = [(0, half), (half, half)]
        writer = self._store(
            tp_rank=0, tp_size=1, unified_suffixes=suffixes, unified_chunk_spans=spans
        )
        reader = self._store(
            tp_rank=1, tp_size=2, unified_suffixes=suffixes, unified_chunk_spans=spans
        )
        pool = self._new_pool()
        expected = _page(pool, 0)
        self.assertTrue(writer.set("unified", pool.get_data_page(0, flat=True)))
        self.assertTrue(reader.exists("unified"))
        restored = torch.empty_like(expected)
        self.assertIs(reader.get("unified", restored), restored)
        self.assertTrue(torch.equal(restored, expected))
        names = writer._component_key("unified", PoolName.KV)
        self.assertTrue(
            all(name.startswith("hicache-test/unified_ukv1-abc_") for name in names)
        )

    def test_unified_chunk_length_mismatch_is_rejected(self):
        suffixes = ["ukv1-abc_L0-2_H0", "ukv1-abc_L0-2_H1"]
        half = _bytes_per_page() // 2
        store = self._store(
            unified_suffixes=suffixes, unified_chunk_spans=[(0, half), (half, half)]
        )
        with self.assertRaises(ValueError):
            store.set("short", torch.ones(half, dtype=torch.uint8))

    def test_unified_sidecar_pools_are_unsupported(self):
        store = self._store(
            unified_suffixes=["ukv1-abc_L0-2"],
            unified_chunk_spans=[(0, _bytes_per_page())],
        )
        with self.assertRaises(NotImplementedError):
            store._component_key("k", PoolName.MAMBA)

    def test_shared_key_scheme_matches_the_file_backend(self):
        """The suffix comes from the base helper, not backend rank arithmetic."""
        legacy = self._config(tp_rank=1, tp_size=4)
        scheme = resolve_storage_key_scheme(legacy)
        file_backend = HiCacheFile.__new__(HiCacheFile)
        HiCacheFile._init_config_suffix(file_backend, legacy)
        self.assertEqual(scheme.suffix, "_Qwen-Qwen3-0.6B_1_4")
        self.assertEqual(file_backend.config_suffix, scheme.suffix)
        self.assertEqual(file_backend.tp_config_suffix, scheme.tp_suffix)
        self.assertEqual(self._store(tp_rank=1, tp_size=4).config_suffix, scheme.suffix)

        suffixes = ["ukv1-abc_L0-2_H0", "ukv1-abc_L0-2_H1"]
        spans = [(0, 100), (100, 100)]
        unified = self._config(unified_suffixes=suffixes, unified_chunk_spans=spans)
        scheme = resolve_storage_key_scheme(unified)
        file_backend = HiCacheFile.__new__(HiCacheFile)
        HiCacheFile._init_config_suffix(file_backend, unified)
        self.assertEqual(scheme.suffix, "_ukv1-abc_L0-2_H0")
        self.assertEqual(file_backend.config_suffix, scheme.suffix)
        self.assertEqual(
            file_backend._unified_chunk_sizes,
            {"_ukv1-abc_L0-2_H0": 100, "_ukv1-abc_L0-2_H1": 100},
        )

    def test_invalid_unified_chunk_grids_are_rejected(self):
        with self.assertRaises(ValueError):
            resolve_storage_key_scheme(self._config(unified_suffixes=["a", "b"]))
        with self.assertRaises(ValueError):
            resolve_storage_key_scheme(
                self._config(
                    unified_suffixes=["a"], unified_chunk_spans=[(0, 10), (10, 10)]
                )
            )
        with self.assertRaises(ValueError):
            resolve_storage_key_scheme(
                self._config(
                    unified_suffixes=["a", "b"],
                    unified_chunk_spans=[(0, 10), (11, 10)],
                )
            )

    def test_config_requires_a_bucket(self):
        with self.assertRaises(ValueError):
            GCSConfig.from_extra_config({"prefix": "x"})
        self.assertEqual(
            GCSConfig.from_extra_config(
                {"bucket": "b", "backend_name": "ignored", "interface_v1": 0}
            ).bucket,
            "b",
        )

    def test_missing_sdk_names_the_package_to_install(self):
        with mock.patch.dict(
            sys.modules, {"google.cloud": None, "google.cloud.storage": None}
        ):
            with self.assertRaises(ImportError) as raised:
                self._real_make_client(GCSConfig(bucket="b"))
        self.assertIn("pip install google-cloud-storage", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
