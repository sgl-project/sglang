"""Unit tests for the HiCache ``fast_file`` storage backend.

Pure CPU tests: they exercise HiCacheFastFile and its LRUFileEvictor through
the public storage interface on a temp directory.
    uv run pytest test/registered/unit/mem_cache/test_hicache_fast_file_unit.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.storage import StorageBackendFactory
from sglang.srt.mem_cache.storage.fast_file.fast_file_store import HiCacheFastFile
from sglang.srt.mem_cache.storage.fast_file.lru_file_evictor import (
    LRUFileEvictor,
    parse_size_to_bytes,
)
from sglang.test.test_utils import CustomTestCase

STORE_LOGGER = "sglang.srt.mem_cache.storage.fast_file.fast_file_store"


def _t(n_bytes: int, fill: int = 0) -> torch.Tensor:
    return torch.full((n_bytes,), fill, dtype=torch.uint8)


class _FakePageFirstPool:
    """Page-first host pool stand-in: kv_buffer[k_or_v, token, byte]."""

    def __init__(self, *, size: int = 8, page_size: int = 2):
        self.layout = "page_first"
        self.page_size = page_size
        self.size = size
        self.dtype = torch.uint8
        self.size_per_token = 4
        self.kv_buffer = torch.zeros((2, size, self.size_per_token), dtype=torch.uint8)

    def get_page_buffer_meta(self, indices):
        ptrs = []
        sizes = []
        base = self.kv_buffer.data_ptr()
        value_offset = self.size * self.size_per_token
        page_bytes = self.page_size * self.size_per_token
        for index in indices[:: self.page_size].tolist():
            key_ptr = base + index * self.size_per_token
            ptrs.extend((key_ptr, key_ptr + value_offset))
            sizes.extend((page_bytes, page_bytes))
        return ptrs, sizes

    def get_data_page(self, index, flat=True):
        page = self.kv_buffer[:, index : index + self.page_size]
        return page.flatten() if flat else page

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            (2, self.page_size, self.size_per_token), dtype=torch.uint8
        ).flatten()

    def set_from_flat_data_page(self, index, data_page):
        self.kv_buffer[:, index : index + self.page_size] = data_page.reshape(
            2, self.page_size, self.size_per_token
        )


def _make_config(
    *,
    tp_rank=0,
    tp_size=1,
    pp_rank=0,
    pp_size=1,
    attn_cp_rank=0,
    attn_cp_size=1,
    is_mla=False,
    model="testmodel",
    extra_config=None,
    enable_storage_metrics=False,
) -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
        is_mla_model=is_mla,
        enable_storage_metrics=enable_storage_metrics,
        is_page_first_layout=True,
        model_name=model,
        extra_config=extra_config,
    )


class _BackendBuilder:
    """Build HiCacheFastFile backends in fresh temp dirs and close them later.

    Background eviction is off by default (``evict_high_watermark=1.0``) so
    accounting assertions are deterministic; tests opt in explicitly.
    """

    def __init__(self, base_tmp: str):
        self.base_tmp = base_tmp
        self.backends = []

    def __call__(
        self,
        *,
        max_size=None,
        min_free=None,
        evict_high_watermark=1.0,
        evict_low_watermark=None,
        preevict_interval_ms=100_000,
        evict_batch_size=None,
        read_workers=None,
        metadata_ttl=None,
        enable_metadata_cache=None,
        stale_temp_age_s=None,
        tp_rank=0,
        tp_size=1,
        is_mla=False,
        model="testmodel",
        subdir=None,
        enable_storage_metrics=False,
    ) -> HiCacheFastFile:
        directory = os.path.join(
            self.base_tmp, subdir or f"r{tp_rank}_t{tp_size}_{time.time_ns()}"
        )
        cfg = _make_config(
            tp_rank=tp_rank,
            tp_size=tp_size,
            is_mla=is_mla,
            model=model,
            enable_storage_metrics=enable_storage_metrics,
            extra_config={
                "max_size": max_size,
                "min_free_space": min_free,
                "evict_high_watermark": evict_high_watermark,
                "evict_low_watermark": evict_low_watermark,
                "preevict_interval_ms": preevict_interval_ms,
                "evict_batch_size": evict_batch_size,
                "read_workers": read_workers,
                "metadata_ttl": metadata_ttl,
                "enable_metadata_cache": enable_metadata_cache,
                "stale_temp_age_s": stale_temp_age_s,
            },
        )
        backend = HiCacheFastFile(cfg, file_path=directory)
        self.backends.append(backend)
        return backend

    def close_all(self):
        for backend in self.backends:
            backend.close()


class FastFileTestBase(CustomTestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="hicache_fast_file_unit_")
        self.make_backend = _BackendBuilder(self.tmpdir)
        # Keep the developer's shell from leaking file-backend settings in.
        self._env_overrides = [
            envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(None),
            envs.SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE.override("0"),
            envs.SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE.override("0"),
        ]
        for cm in self._env_overrides:
            cm.__enter__()

    def tearDown(self):
        self.make_backend.close_all()
        for cm in reversed(self._env_overrides):
            cm.__exit__(None, None, None)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestParseSize(CustomTestCase):
    def test_units_and_disabled_values(self):
        self.assertEqual(parse_size_to_bytes(None), 0)
        self.assertEqual(parse_size_to_bytes("0"), 0)
        self.assertEqual(parse_size_to_bytes(""), 0)
        self.assertEqual(parse_size_to_bytes("1024"), 1024)
        self.assertEqual(parse_size_to_bytes("1k"), 1000)
        self.assertEqual(parse_size_to_bytes("1Ki"), 1024)
        self.assertEqual(parse_size_to_bytes("2Gi"), 2 * (1 << 30))
        self.assertEqual(parse_size_to_bytes("1.5G"), int(1.5 * 10**9))

    def test_invalid_sizes_raise(self):
        for value in ("none", "abc", "10XY", -1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_size_to_bytes(value)


class TestNamespaces(FastFileTestBase):
    def test_storage_root_precedence(self):
        """``storage_dir`` wins over the ``file`` backend's storage-dir env
        var, which fast_file reuses instead of defining its own."""
        env_root = os.path.join(self.tmpdir, "from-env")
        cfg_root = os.path.join(self.tmpdir, "from-config")
        with envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(env_root):
            from_env = HiCacheFastFile(_make_config())
            from_cfg = HiCacheFastFile(
                _make_config(extra_config={"storage_dir": cfg_root})
            )
        self.make_backend.backends.extend((from_env, from_cfg))
        self.assertEqual(from_env.storage_root, env_root)
        self.assertEqual(from_cfg.storage_root, cfg_root)
        self.assertTrue(os.path.isdir(from_cfg.file_path))

    def test_host_pool_layouts_use_distinct_namespaces(self):
        """The same page hash is different bytes under another layout, page
        size or dtype, so those must never share files."""
        root = os.path.join(self.tmpdir, "layouts")
        page_first = HiCacheFastFile(
            _make_config(), _FakePageFirstPool(page_size=2), file_path=root
        )
        layer_first_pool = _FakePageFirstPool(page_size=2)
        layer_first_pool.layout = "layer_first"
        layer_first = HiCacheFastFile(_make_config(), layer_first_pool, file_path=root)
        other_page_size = HiCacheFastFile(
            _make_config(), _FakePageFirstPool(page_size=4), file_path=root
        )
        self.make_backend.backends.extend((page_first, layer_first, other_page_size))
        namespaces = {
            page_first.file_path,
            layer_first.file_path,
            other_page_size.file_path,
        }
        self.assertEqual(len(namespaces), 3)

    def test_mla_ranks_share_a_namespace_and_other_ranks_do_not(self):
        owner = self.make_backend(is_mla=True, tp_rank=0, tp_size=2, subdir="mla")
        peer = self.make_backend(is_mla=True, tp_rank=1, tp_size=2, subdir="mla")
        self.assertEqual(owner.file_path, peer.file_path)
        self.assertTrue(owner.set("shared", _t(50, fill=9)))
        self.assertTrue(torch.equal(peer.get("shared", _t(50)), _t(50, fill=9)))

        rank0 = self.make_backend(tp_rank=0, tp_size=4, subdir="sharded")
        rank3 = self.make_backend(tp_rank=3, tp_size=4, subdir="sharded")
        self.assertNotEqual(rank0.file_path, rank3.file_path)
        self.assertTrue(os.path.isdir(rank3.file_path))

    def test_clear_removes_only_the_current_namespace(self):
        """Model names that are suffixes of each other ("foo" / "bar_foo")
        share a key suffix; clear must be scoped by namespace, not suffix."""
        root = os.path.join(self.tmpdir, "shared-root")
        b = HiCacheFastFile(
            _make_config(
                model="foo",
                extra_config={
                    "max_size": "300",
                    "evict_high_watermark": 1.0,
                    "metadata_ttl": -1.0,
                    "enable_metadata_cache": True,
                },
            ),
            file_path=root,
        )
        other = HiCacheFastFile(_make_config(model="bar_foo"), file_path=root)
        self.make_backend.backends.extend((b, other))
        self.assertTrue(b.set("owned", _t(100)))
        self.assertTrue(other.set("other", _t(100, fill=7)))
        owned_temp = os.path.join(
            b.file_path, f"partial{b.config_suffix}.bin.tmp.1.2.partial"
        )
        other_temp = os.path.join(
            other.file_path, f"partial{other.config_suffix}.bin.tmp.1.2.partial"
        )
        unrelated = os.path.join(root, "notes.txt")
        for path in (owned_temp, other_temp, unrelated):
            with open(path, "wb") as file:
                file.write(b"data")

        self.assertTrue(b.clear())

        self.assertFalse(os.path.exists(b._get_component_path("owned")))
        self.assertFalse(os.path.exists(owned_temp))
        self.assertTrue(os.path.exists(other._get_component_path("other")))
        self.assertTrue(os.path.exists(other_temp))
        self.assertTrue(os.path.exists(unrelated))
        self.assertTrue(torch.equal(other.get("other", _t(100)), _t(100, fill=7)))
        self.assertEqual(b._evictor.snapshot()["entries"], 0)
        self.assertEqual(b._evictor.snapshot()["total_bytes"], 0)
        self.assertFalse(b.metadata_cache.contains(b._get_suffixed_key("owned")))

    def test_clear_reports_partial_failure(self):
        b = self.make_backend(max_size="300")
        with mock.patch.object(b._evictor, "clear_storage", return_value=False):
            self.assertFalse(b.clear())

    def test_startup_removes_only_stale_temp_files_for_this_namespace(self):
        directory = os.path.join(self.tmpdir, "crash-recovery")
        config = _make_config(extra_config={"stale_temp_age_s": 10})
        namespace = os.path.join(
            directory, HiCacheFastFile._namespace_name(config, None)
        )
        os.makedirs(namespace)
        suffix = "_testmodel_0_1"
        stale_path = os.path.join(namespace, f"stale{suffix}.bin.tmp.1.2.old")
        fresh_path = os.path.join(namespace, f"fresh{suffix}.bin.tmp.1.2.new")
        other_namespace = os.path.join(directory, "namespace-other")
        os.makedirs(other_namespace)
        other_path = os.path.join(
            other_namespace, "other_othermodel_0_1.bin.tmp.1.2.old"
        )
        for path in (stale_path, fresh_path, other_path):
            with open(path, "wb") as file:
                file.write(b"x" * 32)
        old_time = time.time() - 100
        os.utime(stale_path, (old_time, old_time))
        os.utime(other_path, (old_time, old_time))

        backend = HiCacheFastFile(config, file_path=directory)
        self.make_backend.backends.append(backend)

        self.assertFalse(os.path.exists(stale_path))
        self.assertTrue(os.path.exists(fresh_path))
        self.assertTrue(os.path.exists(other_path))


class TestFormatCompatibility(FastFileTestBase):
    def test_pages_are_interchangeable_with_the_reference_backend(self):
        fast = self.make_backend()
        reference = HiCacheFile(_make_config(), file_path=fast.file_path)
        self.assertEqual(fast.config_suffix, reference.config_suffix)

        first = _t(128, fill=3)
        second = _t(128, fill=7)
        self.assertTrue(reference.set("from-reference", first))
        self.assertTrue(torch.equal(fast.get("from-reference", _t(128)), first))
        self.assertTrue(fast.set("from-fast", second))
        self.assertTrue(torch.equal(reference.get("from-fast", _t(128)), second))

    def test_factory_constructs_fast_file_backend(self):
        config = _make_config(
            extra_config={"storage_dir": os.path.join(self.tmpdir, "factory")}
        )
        backend = StorageBackendFactory.create_backend("fast_file", config, None)
        self.make_backend.backends.append(backend)
        self.assertIsInstance(backend, HiCacheFastFile)

    def test_direct_read_accepts_pages_written_by_the_staged_path(self):
        """The iovec order of a direct transfer must equal the flat page order,
        otherwise staged writers and direct readers disagree on the bytes."""
        b = self.make_backend()
        page = torch.arange(16, dtype=torch.uint8)
        self.assertTrue(b.set("legacy", page))

        pool = _FakePageFirstPool()
        b.register_mem_pool_host(pool)
        self.assertEqual(b.batch_get_v1(["legacy"], torch.tensor([6, 7])), [True])
        self.assertTrue(torch.equal(pool.kv_buffer[:, 6:8].flatten(), page))


class TestDirectIO(FastFileTestBase):
    def _no_staging(self, pool):
        return (
            mock.patch.object(
                pool, "get_data_page", side_effect=AssertionError("staging copy used")
            ),
            mock.patch.object(
                pool,
                "get_dummy_flat_data_page",
                side_effect=AssertionError("staging allocation used"),
            ),
        )

    def test_v1_reads_and_writes_the_registered_pool_in_place(self):
        b = self.make_backend(read_workers=2)
        pool = _FakePageFirstPool()
        b.register_mem_pool_host(pool)
        self.assertTrue(b._pool_direct_io[PoolName.KV])

        indices = torch.tensor([0, 1, 4, 5])
        pool.kv_buffer[:, 0:2] = torch.arange(16, dtype=torch.uint8).reshape(2, 2, 4)
        pool.kv_buffer[:, 4:6] = torch.arange(16, 32, dtype=torch.uint8).reshape(
            2, 2, 4
        )
        expected_first = pool.kv_buffer[:, 0:2].clone()
        expected_second = pool.kv_buffer[:, 4:6].clone()
        no_page, no_dummy = self._no_staging(pool)
        with no_page, no_dummy:
            self.assertEqual(b.batch_set_v1(["a", "b"], indices), [True, True])
            with open(b._get_component_path("a"), "rb") as file:
                self.assertEqual(
                    file.read(), expected_first.flatten().numpy().tobytes()
                )
            pool.kv_buffer.zero_()
            self.assertEqual(b.batch_get_v1(["a", "b"], indices), [True, True])
        self.assertTrue(torch.equal(pool.kv_buffer[:, 0:2], expected_first))
        self.assertTrue(torch.equal(pool.kv_buffer[:, 4:6], expected_second))

    def test_missing_vectored_io_falls_back_to_staging(self):
        b = self.make_backend()
        b._vector_io_supported = False
        pool = _FakePageFirstPool()
        b.register_mem_pool_host(pool)
        self.assertFalse(b._pool_direct_io[PoolName.KV])

        indices = torch.tensor([0, 1])
        pool.kv_buffer[:, 0:2].fill_(9)
        self.assertEqual(b.batch_set_v1(["page"], indices), [True])
        pool.kv_buffer.zero_()
        self.assertEqual(b.batch_get_v1(["page"], indices), [True])
        self.assertTrue(torch.all(pool.kv_buffer[:, 0:2] == 9))

    def test_v2_uses_direct_io_for_a_registered_sidecar_pool(self):
        b = self.make_backend(read_workers=2)
        pool = _FakePageFirstPool()
        b.register_mem_host_pool_v2(pool, PoolName.DRAFT)
        indices = torch.tensor([2, 3])
        pool.kv_buffer[:, 2:4] = torch.arange(16, dtype=torch.uint8).reshape(2, 2, 4)
        expected = pool.kv_buffer[:, 2:4].clone()
        transfer = PoolTransfer(
            name=PoolName.DRAFT, host_indices=indices, keys=["page"]
        )
        no_page, no_dummy = self._no_staging(pool)
        with no_page, no_dummy:
            self.assertEqual(b.batch_set_v2([transfer]), {PoolName.DRAFT: [True]})
            pool.kv_buffer[:, 2:4].zero_()
            self.assertEqual(b.batch_get_v2([transfer]), {PoolName.DRAFT: [True]})
        self.assertTrue(torch.equal(pool.kv_buffer[:, 2:4], expected))

    def test_non_page_first_pool_uses_the_staged_path(self):
        b = self.make_backend()
        pool = _FakePageFirstPool()
        pool.layout = "layer_first"
        with mock.patch.object(
            pool, "get_page_buffer_meta", side_effect=AssertionError("direct path used")
        ):
            b.register_mem_pool_host(pool)
            indices = torch.tensor([0, 1])
            pool.kv_buffer[:, 0:2].fill_(7)
            self.assertEqual(b.batch_set_v1(["a"], indices), [True])
            pool.kv_buffer.zero_()
            self.assertEqual(b.batch_get_v1(["a"], indices), [True])
        self.assertTrue(torch.all(pool.kv_buffer[:, 0:2] == 7))

    def test_logical_anchor_pool_round_trips_empty_marker_pages(self):
        """A logical (buffer-less) KV anchor stores zero-byte pages whose
        presence gates the sidecars; no special casing may break that."""
        b = self.make_backend(read_workers=2)
        b.register_mem_pool_host(
            LogicalHostPool(size=4, page_size=2, layout="page_first")
        )
        self.assertFalse(b._pool_direct_io[PoolName.KV])

        keys = ["a", "b"]
        indices = torch.arange(4)
        self.assertEqual(b.batch_set_v1(keys, indices), [True, True])
        for key in keys:
            self.assertEqual(os.path.getsize(b._get_component_path(key)), 0)
        self.assertEqual(b.batch_get_v1(keys, indices), [True, True])
        os.remove(b._get_component_path("b"))
        self.assertEqual(b.batch_get_v1(keys, indices), [True, False])

    def test_invalid_page_buffer_metadata_fails_the_batch_without_raising(self):
        """Storage I/O runs on controller threads that never catch; a bad pool
        must produce a failed batch, not an exception."""
        b = self.make_backend()
        pool = _FakePageFirstPool()
        b.register_mem_pool_host(pool)
        indices = torch.tensor([0, 1, 2, 3])
        with mock.patch.object(
            pool, "get_page_buffer_meta", return_value=([1, 2, 3], [8, 8, 8])
        ):
            self.assertEqual(b.batch_set_v1(["a", "b"], indices), [False, False])
            self.assertEqual(b.batch_get_v1(["a", "b"], indices), [False, False])
        self.assertEqual(
            b.batch_get_v1(["a", "b"], torch.tensor([0, 1])), [False, False]
        )

    def test_rank_sharded_sidecars_are_rejected_only_under_replicated_kv(self):
        cases = [
            (True, 2, PoolName.MAMBA, False, True),
            (True, 2, PoolName.DRAFT, True, True),
            (True, 1, PoolName.MAMBA, False, False),
            (False, 2, PoolName.MAMBA, False, False),
        ]
        for is_mla, tp_size, pool_name, mha_draft, expect_error in cases:
            with self.subTest(is_mla=is_mla, tp_size=tp_size, pool=pool_name):
                b = self.make_backend(is_mla=is_mla, tp_size=tp_size)
                mha_patch = (
                    mock.patch(
                        "sglang.srt.mem_cache.pool_host.mha.MHATokenToKVPoolHost",
                        _FakePageFirstPool,
                    )
                    if mha_draft
                    else mock.patch.object(b, "_tp_size", tp_size)
                )
                with mha_patch:
                    if expect_error:
                        with self.assertRaisesRegex(ValueError, "rank-sharded"):
                            b.register_mem_host_pool_v2(_FakePageFirstPool(), pool_name)
                        self.assertFalse(hasattr(b, "registered_pools"))
                    else:
                        b.register_mem_host_pool_v2(_FakePageFirstPool(), pool_name)
                        self.assertIn(pool_name, b.registered_pools)

    def test_bounded_eviction_with_side_pools_warns_but_registers(self):
        b = self.make_backend(max_size="1Mi")
        with self.assertLogs(STORE_LOGGER, level="WARNING") as logs:
            b.register_mem_host_pool_v2(_FakePageFirstPool(), PoolName.SWA)
            b.register_mem_host_pool_v2(_FakePageFirstPool(), PoolName.DRAFT)
        self.assertEqual(
            sum("independently" in line for line in logs.output), 1, logs.output
        )
        self.assertIn(PoolName.SWA, b.registered_pools)


class TestParallelReads(FastFileTestBase):
    def test_batch_get_runs_concurrently_and_preserves_order(self):
        serial = self.make_backend()
        self.assertIsNone(serial._read_executor)

        b = self.make_backend(read_workers=4)
        self.assertIsNotNone(b._read_executor)
        keys = [f"k{i}" for i in range(8)]
        values = [_t(4096, fill=i) for i in range(len(keys))]
        self.assertTrue(b.batch_set(keys, values))

        thread_ids = set()
        thread_ids_lock = threading.Lock()
        original_get = b.get

        def tracked_get(key, target_location, target_sizes=None):
            with thread_ids_lock:
                thread_ids.add(threading.get_ident())
            time.sleep(0.01)
            return original_get(key, target_location, target_sizes)

        with mock.patch.object(b, "get", side_effect=tracked_get):
            results = b.batch_get(keys, [_t(4096) for _ in keys])

        self.assertGreaterEqual(len(thread_ids), 2)
        for result, expected in zip(results, values):
            self.assertTrue(torch.equal(result, expected))

    def test_parallel_map_drains_workers_before_propagating_an_error(self):
        """A worker still writing into a caller-owned buffer after the batch
        returned would corrupt host memory; every task must finish first."""
        b = self.make_backend(read_workers=2)
        slow_started = threading.Event()
        release_slow = threading.Event()
        bad_raised = threading.Event()
        slow_finished = threading.Event()
        batch_done = threading.Event()
        errors = []

        def read(key):
            if key == "bad":
                self.assertTrue(slow_started.wait(timeout=5))
                bad_raised.set()
                raise RuntimeError("injected read failure")
            slow_started.set()
            self.assertTrue(release_slow.wait(timeout=5))
            slow_finished.set()
            return True

        def run_batch():
            try:
                b._parallel_map(read, ["bad", "slow"])
            except Exception as exc:
                errors.append(exc)
            finally:
                batch_done.set()

        caller = threading.Thread(target=run_batch)
        caller.start()
        try:
            self.assertTrue(bad_raised.wait(timeout=5))
            self.assertFalse(batch_done.wait(timeout=0.1))
            self.assertFalse(slow_finished.is_set())
        finally:
            release_slow.set()
        caller.join(timeout=5)

        self.assertFalse(caller.is_alive())
        self.assertTrue(slow_finished.is_set())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)

    def test_batch_exists_uses_targeted_stats_not_a_directory_scan(self):
        b = self.make_backend(read_workers=2)
        b.set("k1", _t(50))
        original_exists = os.path.exists
        with (
            mock.patch("os.scandir", side_effect=AssertionError("full scan used")),
            mock.patch("os.path.exists", side_effect=original_exists) as mock_exists,
        ):
            result = b.batch_exists_v2(["k1", "missing"])
        self.assertEqual(result.kv_hit_pages, 1)
        self.assertEqual(mock_exists.call_count, 2)

    def test_invalid_config_is_rejected_before_any_thread_starts(self):
        cases = [
            ({"read_workers": 0}, "read_workers"),
            ({"max_size": "100", "stale_temp_age_s": -1}, "stale_temp_age_s"),
            (
                {
                    "max_size": "100",
                    "evict_high_watermark": 0.5,
                    "evict_low_watermark": 0.9,
                },
                "evict_low_watermark",
            ),
        ]
        for extra, message in cases:
            with self.subTest(extra=extra):
                with mock.patch("threading.Thread.start") as start:
                    with self.assertRaisesRegex(ValueError, message):
                        HiCacheFastFile(
                            _make_config(extra_config=extra),
                            file_path=os.path.join(self.tmpdir, "invalid"),
                        )
                start.assert_not_called()

    def test_close_is_idempotent(self):
        b = self.make_backend(
            read_workers=2,
            max_size="100",
            evict_high_watermark=0.5,
            evict_low_watermark=0.4,
        )
        b.close()
        b.close()
        self.assertIsNone(b._read_executor)


class TestStorageMetrics(FastFileTestBase):
    def test_metrics_report_pages_and_bandwidth_then_drain(self):
        b = self.make_backend(read_workers=2, enable_storage_metrics=True)
        keys = ["a", "b", "c", "d"]
        values = [_t(4096, fill=i) for i in range(len(keys))]
        self.assertTrue(b.batch_set(keys, values))
        results = b.batch_get(keys, [_t(4096) for _ in keys])
        self.assertTrue(all(result is not None for result in results))

        stats = b.get_stats()
        self.assertEqual(stats.backup_pgs, [len(keys)])
        self.assertEqual(stats.prefetch_pgs, [len(keys)])
        self.assertGreater(stats.backup_bandwidth[0], 0)
        self.assertGreater(stats.prefetch_bandwidth[0], 0)

        drained = b.get_stats()
        self.assertEqual(drained.backup_pgs, [])
        self.assertEqual(drained.prefetch_pgs, [])

    def test_metrics_skip_duplicate_writes_and_count_partial_reads(self):
        b = self.make_backend(enable_storage_metrics=True)
        pool = _FakePageFirstPool(size=4)
        b.register_mem_host_pool_v2(pool, PoolName.SWA)
        indices = torch.tensor([0, 1, 2, 3])
        pool.kv_buffer.fill_(5)
        transfer = PoolTransfer(
            name=PoolName.SWA, host_indices=indices, keys=["a", "b"]
        )

        self.assertEqual(b.batch_set_v2([transfer]), {PoolName.SWA: [True, True]})
        self.assertEqual(b.get_stats().backup_pgs, [1 + 1])
        self.assertEqual(b.batch_set_v2([transfer]), {PoolName.SWA: [True, True]})
        self.assertEqual(b.get_stats().backup_pgs, [])

        os.remove(b._get_component_path("b", PoolName.SWA))
        pool.kv_buffer.zero_()
        with mock.patch.object(
            b, "_record_io_metrics", wraps=b._record_io_metrics
        ) as record:
            self.assertEqual(b.batch_get_v2([transfer]), {PoolName.SWA: [True, False]})
        record.assert_called_once()
        self.assertEqual(record.call_args.kwargs["pages"], 1)
        self.assertEqual(record.call_args.kwargs["num_bytes"], 16)
        self.assertEqual(b.get_stats().prefetch_pgs, [1])


class TestCapEviction(FastFileTestBase):
    def test_unconfigured_backend_does_not_track_or_evict(self):
        b = self.make_backend()
        self.assertFalse(b._evictor.enabled)
        self.assertTrue(b.set("k1", _t(50)))
        self.assertTrue(b.exists("k1"))
        self.assertEqual(b._evictor.snapshot()["entries"], 0)

    def test_lru_evicts_the_oldest_untouched_page(self):
        b = self.make_backend(max_size="300")
        for key in ("a", "b", "c"):
            self.assertTrue(b.set(key, _t(100)))
        self.assertEqual(b._evictor.snapshot()["total_bytes"], 300)
        b.get("a", _t(100))
        self.assertTrue(b.set("d", _t(100)))
        self.assertLessEqual(b._evictor.snapshot()["total_bytes"], 300)
        self.assertFalse(b.exists("b"), "b became the LRU once a was read")
        for key in ("a", "c", "d"):
            self.assertTrue(b.exists(key), f"{key} should still be present")

    def test_value_larger_than_cap_is_rejected(self):
        b = self.make_backend(max_size="100")
        self.assertFalse(b.set("too_big", _t(200)))
        self.assertFalse(b.exists("too_big"))
        self.assertEqual(b._evictor.snapshot()["entries"], 0)

    def test_repeated_set_of_the_same_key_is_accounted_once(self):
        b = self.make_backend(max_size="300")
        self.assertTrue(b.set("a", _t(100)))
        self.assertTrue(b.set("a", _t(100)))
        snapshot = b._evictor.snapshot()
        self.assertEqual(snapshot["total_bytes"], 100)
        self.assertEqual(snapshot["entries"], 1)

    def test_foreground_admission_evicts_only_the_required_space(self):
        b = self.make_backend(
            max_size="1000", evict_high_watermark=0.95, evict_low_watermark=0.85
        )
        for i in range(10):
            self.assertTrue(b.set(f"k{i}", _t(100)))

        self.assertTrue(b.set("new", _t(100)))
        snapshot = b._evictor.snapshot()
        self.assertEqual(snapshot["total_bytes"], 1000)
        self.assertEqual(snapshot["foreground_evicted_entries"], 1)
        self.assertEqual(snapshot["background_evicted_entries"], 0)
        self.assertFalse(b.exists("k0"))
        self.assertTrue(b.exists("k1"))

    def test_background_eviction_drains_from_high_to_low_watermark(self):
        b = self.make_backend(
            max_size="1000", evict_high_watermark=0.75, evict_low_watermark=0.5
        )
        for i in range(8):
            self.assertTrue(b.set(f"k{i}", _t(100)))

        self.assertEqual(b._evictor._preevict_batch(), 300)
        snapshot = b._evictor.snapshot()
        self.assertEqual(snapshot["total_bytes"], 500)
        self.assertEqual(snapshot["foreground_evicted_entries"], 0)
        self.assertEqual(snapshot["background_evicted_entries"], 3)
        for i in range(3):
            self.assertFalse(b.exists(f"k{i}"))
        for i in range(3, 8):
            self.assertTrue(b.exists(f"k{i}"))

    def test_background_eviction_is_sliced_into_bounded_batches(self):
        b = self.make_backend(
            max_size="1000",
            evict_high_watermark=0.75,
            evict_low_watermark=0.2,
            evict_batch_size=2,
        )
        for i in range(8):
            self.assertTrue(b.set(f"k{i}", _t(100)))

        self.assertEqual(b._evictor._preevict_batch(), 200)
        self.assertEqual(b._evictor.snapshot()["total_bytes"], 600)
        self.assertTrue(b._evictor.snapshot()["draining"])
        self.assertEqual(b._evictor._preevict_batch(), 200)
        self.assertEqual(b._evictor.snapshot()["total_bytes"], 400)

    def test_background_worker_drains_on_its_own(self):
        b = self.make_backend(
            max_size="1000",
            evict_high_watermark=0.75,
            evict_low_watermark=0.5,
            preevict_interval_ms=1,
        )
        for i in range(8):
            self.assertTrue(b.set(f"k{i}", _t(100)))

        deadline = time.monotonic() + 2
        while (
            b._evictor.snapshot()["total_bytes"] > 500 and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        self.assertLessEqual(b._evictor.snapshot()["total_bytes"], 500)

    def test_startup_scan_seeds_the_lru_in_mtime_order(self):
        directory = os.path.join(self.tmpdir, "seed")
        cfg = _make_config(model="seedmodel", extra_config={"max_size": "1000"})
        namespace = os.path.join(directory, HiCacheFastFile._namespace_name(cfg, None))
        os.makedirs(namespace)
        suffix = "_seedmodel_0_1"
        old_path = os.path.join(namespace, f"old{suffix}.bin")
        new_path = os.path.join(namespace, f"new{suffix}.bin")
        with open(old_path, "wb") as f:
            f.write(b"x" * 50)
        old_t = time.time() - 100
        os.utime(old_path, (old_t, old_t))
        with open(new_path, "wb") as f:
            f.write(b"y" * 70)

        b = HiCacheFastFile(cfg, file_path=directory)
        self.make_backend.backends.append(b)
        self.assertEqual(b._evictor.snapshot()["total_bytes"], 120)
        self.assertEqual(list(b._evictor._lru), [f"old{suffix}", f"new{suffix}"])

    def test_untracked_files_are_adopted_on_set_and_get(self):
        b = self.make_backend(max_size="500")
        for key, size, access in (("x", 80, "set"), ("y", 64, "get")):
            path = b._get_component_path(key)
            with open(path, "wb") as f:
                f.write(b"\x00" * size)
            self.assertNotIn(b._get_suffixed_key(key), b._evictor._lru)
            if access == "set":
                self.assertTrue(b.set(key, _t(size)))
            else:
                self.assertIsNotNone(b.get(key, _t(size)))
            self.assertIn(b._get_suffixed_key(key), b._evictor._lru)
        self.assertEqual(b._evictor.snapshot()["total_bytes"], 80 + 64)

    def test_external_delete_drops_stale_accounting_on_read(self):
        b = self.make_backend(max_size="200")
        self.assertTrue(b.set("a", _t(100)))
        os.remove(b._get_component_path("a"))

        self.assertIsNone(b.get("a", _t(100)))

        snapshot = b._evictor.snapshot()
        self.assertEqual(snapshot["entries"], 0)
        self.assertEqual(snapshot["total_bytes"], 0)

    def test_wrong_sized_page_is_a_miss_and_is_removed(self):
        for target_bytes in (200, 50):
            with self.subTest(target_bytes=target_bytes):
                b = self.make_backend(max_size="200")
                self.assertTrue(b.set("a", _t(100)))

                self.assertIsNone(b.get("a", _t(target_bytes)))

                self.assertEqual(b._evictor.snapshot()["entries"], 0)
                self.assertFalse(os.path.exists(b._get_component_path("a")))

    def test_set_replaces_a_wrong_sized_existing_page(self):
        b = self.make_backend(max_size="200")
        self.assertTrue(b.set("a", _t(100, fill=1)))
        with open(b._get_component_path("a"), "wb") as file:
            file.write(b"x" * 50)

        self.assertTrue(b.set("a", _t(100, fill=2)))

        self.assertTrue(torch.all(b.get("a", _t(100)) == 2))
        snapshot = b._evictor.snapshot()
        self.assertEqual(snapshot["entries"], 1)
        self.assertEqual(snapshot["total_bytes"], 100)

    def test_transient_read_error_keeps_the_page(self):
        b = self.make_backend(max_size="200")
        self.assertTrue(b.set("a", _t(100, fill=4)))

        with mock.patch.object(b, "_readv_exact", side_effect=OSError("injected")):
            self.assertIsNone(b.get("a", _t(100)))

        self.assertTrue(os.path.exists(b._get_component_path("a")))
        self.assertEqual(b._evictor.snapshot()["entries"], 1)
        self.assertTrue(torch.all(b.get("a", _t(100)) == 4))

    def test_pending_reservation_blocks_eviction_and_competing_admission(self):
        b = self.make_backend(max_size="100")
        pending = b._get_suffixed_key("A")
        with b._evictor._lock:
            b._evictor._lru[pending] = 60
            b._evictor._pending_writes.add(pending)
            b._evictor._total_bytes = 60

        self.assertFalse(b.set("B", _t(60)))
        self.assertIn(pending, b._evictor._lru)
        self.assertIn(pending, b._evictor._pending_writes)
        self.assertEqual(b._evictor.snapshot()["total_bytes"], 60)

    def test_same_key_writes_are_serialized_and_deduplicated(self):
        b = self.make_backend(max_size="1000")
        first_in_write = threading.Event()
        release_first = threading.Event()
        original_writev = b._writev_exact
        call_count = 0
        call_count_lock = threading.Lock()
        outcomes = {}

        def blocked_write(fd, buffers):
            nonlocal call_count
            with call_count_lock:
                call_count += 1
                call = call_count
            if call == 1:
                first_in_write.set()
                self.assertTrue(release_first.wait(timeout=5))
            else:
                raise OSError("second physical write should not run")
            return original_writev(fd, buffers)

        with mock.patch.object(b, "_writev_exact", side_effect=blocked_write):
            first = threading.Thread(
                target=lambda: outcomes.setdefault(
                    "first", b.set("same", _t(100, fill=1))
                )
            )
            second = threading.Thread(
                target=lambda: outcomes.setdefault(
                    "second", b.set("same", _t(100, fill=2))
                )
            )
            first.start()
            self.assertTrue(first_in_write.wait(timeout=5))
            second.start()
            time.sleep(0.1)
            release_first.set()
            first.join(timeout=5)
            second.join(timeout=5)

        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertTrue(outcomes["first"] and outcomes["second"])
        self.assertEqual(call_count, 1)
        self.assertTrue(torch.all(b.get("same", _t(100)) == 1))
        snapshot = b._evictor.snapshot()
        self.assertEqual(snapshot["entries"], 1)
        self.assertEqual(snapshot["pending_writes"], 0)
        self.assertFalse(any(".tmp." in name for name in os.listdir(b.file_path)))

    def test_missing_read_does_not_discard_a_pending_write(self):
        b = self.make_backend(max_size="200")
        write_started = threading.Event()
        release_write = threading.Event()
        original_writev = b._writev_exact
        outcomes = {}

        def blocked_write(fd, buffers):
            write_started.set()
            self.assertTrue(release_write.wait(timeout=5))
            return original_writev(fd, buffers)

        with mock.patch.object(b, "_writev_exact", side_effect=blocked_write):
            writer = threading.Thread(
                target=lambda: outcomes.setdefault("write", b.set("a", _t(100)))
            )
            writer.start()
            self.assertTrue(write_started.wait(timeout=5))
            try:
                self.assertIsNone(b.get("a", _t(100)))
                snapshot = b._evictor.snapshot()
                self.assertEqual(snapshot["entries"], 1)
                self.assertEqual(snapshot["total_bytes"], 100)
                self.assertEqual(snapshot["pending_writes"], 1)
            finally:
                release_write.set()
            writer.join(timeout=5)

        self.assertTrue(outcomes["write"])
        self.assertIsNotNone(b.get("a", _t(100)))
        self.assertEqual(b._evictor.snapshot()["pending_writes"], 0)


class TestMinFreeSpace(FastFileTestBase):
    def test_missing_filesystem_statistics_fail_closed(self):
        with mock.patch.object(LRUFileEvictor, "_fs_stats", return_value=None):
            with self.assertRaisesRegex(OSError, "cannot enforce min_free_space"):
                self.make_backend(min_free="100")

        b = self.make_backend(min_free="100")
        with mock.patch.object(b._evictor, "_fs_stats", return_value=None):
            self.assertFalse(b.set("nope", _t(100)))
        self.assertFalse(b.exists("nope"))

    def test_refuses_writes_that_would_drop_below_the_floor(self):
        b = self.make_backend(min_free="100")
        b._evictor._fs_stats = lambda: (1024, 150)
        self.assertFalse(b.set("nope", _t(100)))
        self.assertFalse(b.exists("nope"))

    def test_evicts_to_restore_the_floor(self):
        b = self.make_backend(min_free="100")
        suffixed = b._get_suffixed_key("victim")
        path = b._get_component_path("victim")
        with open(path, "wb") as f:
            f.write(b"v" * 80)
        b._evictor._lru[suffixed] = 80
        b._evictor._total_bytes = 80
        free = [130]
        original_remove = os.remove

        def tracked_remove(p):
            # Model a filesystem that frees space on unlink.
            if os.path.exists(p):
                free[0] += os.path.getsize(p)
            return original_remove(p)

        with (
            mock.patch.object(
                b._evictor, "_fs_stats", side_effect=lambda: (1024, free[0])
            ),
            mock.patch("os.remove", side_effect=tracked_remove),
        ):
            self.assertTrue(b.set("newk", _t(60)))
        self.assertFalse(b.exists("victim"))
        self.assertTrue(b.exists("newk"))


class TestMLAOwnerGating(FastFileTestBase):
    def test_only_rank_zero_evicts_and_creates_files_for_replicated_kv(self):
        owner = self.make_backend(max_size="200", is_mla=True, tp_rank=0, tp_size=2)
        self.assertTrue(owner._evictor.is_storage_owner)
        self.assertTrue(owner._evictor.enabled)

        peer = self.make_backend(max_size="200", is_mla=True, tp_rank=1, tp_size=2)
        self.assertFalse(peer._evictor.enabled)
        self.assertFalse(peer.set("a", _t(50)))
        self.assertFalse(peer.exists("a"))
        with open(peer._get_component_path("a"), "wb") as f:
            f.write(b"x" * 50)
        self.assertTrue(peer.set("a", _t(50)), "an existing page is still accepted")
        self.assertEqual(peer._evictor.snapshot()["entries"], 0)

        sharded = self.make_backend(max_size="200", tp_rank=3, tp_size=4)
        self.assertTrue(sharded._evictor.enabled)


class TestMetadataCacheIntegration(FastFileTestBase):
    def test_writes_reads_and_startup_scan_populate_the_cache(self):
        self.assertIsNone(self.make_backend().metadata_cache)

        directory = os.path.join(self.tmpdir, "metadata-seed")
        cfg = _make_config(
            model="seedmodel",
            extra_config={"metadata_ttl": 5.0, "enable_metadata_cache": True},
        )
        namespace = os.path.join(directory, HiCacheFastFile._namespace_name(cfg, None))
        os.makedirs(namespace)
        suffix = "_seedmodel_0_1"
        with open(os.path.join(namespace, f"k0{suffix}.bin"), "wb") as f:
            f.write(b"data")
        b = HiCacheFastFile(cfg, file_path=directory)
        self.make_backend.backends.append(b)
        self.assertTrue(b.metadata_cache.contains(f"k0{suffix}"))

        self.assertFalse(b.metadata_cache.contains(f"k1{suffix}"))
        b.set("k1", _t(50))
        self.assertTrue(b.metadata_cache.contains(f"k1{suffix}"))
        b.metadata_cache.clear()
        b.get("k1", _t(50))
        self.assertTrue(b.metadata_cache.contains(f"k1{suffix}"))

    def test_external_delete_does_not_turn_a_rewrite_into_a_noop(self):
        b = self.make_backend(metadata_ttl=-1.0, enable_metadata_cache=True)
        self.assertTrue(b.set("k1", _t(50, fill=1)))
        os.remove(b._get_component_path("k1"))

        self.assertTrue(b.set("k1", _t(50, fill=2)))

        self.assertTrue(torch.all(b.get("k1", _t(50)) == 2))

    def test_metadata_is_published_before_the_write_becomes_evictable(self):
        b = self.make_backend(
            max_size="200", metadata_ttl=-1.0, enable_metadata_cache=True
        )
        original_commit = b._evictor.commit

        def assert_metadata_then_commit(suffixed_key):
            self.assertTrue(b.metadata_cache.contains(suffixed_key))
            original_commit(suffixed_key)

        with mock.patch.object(
            b._evictor, "commit", side_effect=assert_metadata_then_commit
        ):
            self.assertTrue(b.set("k1", _t(50)))

    def test_eviction_removes_the_metadata_entry(self):
        b = self.make_backend(
            max_size="200", metadata_ttl=-1.0, enable_metadata_cache=True
        )
        suffix = b.config_suffix
        b.set("k1", _t(100))
        b.set("k2", _t(100))
        b.set("k3", _t(100))
        self.assertFalse(b.metadata_cache.contains(f"k1{suffix}"))
        self.assertTrue(b.metadata_cache.contains(f"k2{suffix}"))
        self.assertTrue(b.metadata_cache.contains(f"k3{suffix}"))

    def test_batch_exists_answers_from_the_cache_and_stats_only_misses(self):
        b = self.make_backend(metadata_ttl=5.0, enable_metadata_cache=True)
        b.set("k1", _t(50))
        b.set("k2", _t(50))

        with (
            mock.patch("os.scandir") as mock_scandir,
            mock.patch("os.path.exists", return_value=True) as mock_exists,
        ):
            self.assertEqual(b.batch_exists_v2(["k1", "k2"]).kv_hit_pages, 2)
            mock_scandir.assert_not_called()
            mock_exists.assert_not_called()

            self.assertEqual(b.batch_exists_v2(["k3"]).kv_hit_pages, 1)
            mock_scandir.assert_not_called()
            mock_exists.assert_called_once()


class TestBatchExistsV2(FastFileTestBase):
    def test_each_sidecar_is_evaluated_against_the_full_kv_prefix(self):
        b = self.make_backend(read_workers=2)
        keys = [f"k{i}" for i in range(4)]
        for key in keys:
            self.assertTrue(b.set(key, _t(8)))
        for key in keys[:2]:
            self.assertTrue(b.set(f"{key}.{PoolName.SWA}", _t(8)))
        self.assertTrue(b.set(f"{keys[3]}.{PoolName.DRAFT}", _t(8)))

        result = b.batch_exists_v2(
            keys,
            [
                PoolTransfer(
                    name=PoolName.SWA, keys=keys, hit_policy=PoolHitPolicy.ALL_PAGES
                ),
                PoolTransfer(
                    name=PoolName.DRAFT,
                    keys=[keys[-1]],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                ),
            ],
        )

        self.assertEqual(result.kv_hit_pages, 2)
        self.assertEqual(
            result.extra_pool_hit_pages,
            {PoolName.KV: 4, PoolName.SWA: 2, PoolName.DRAFT: 4},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
