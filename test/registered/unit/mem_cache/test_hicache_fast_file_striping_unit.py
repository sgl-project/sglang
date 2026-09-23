"""Unit tests for fast_file multi-root striping and parallel writes.

Pure CPU tests on temp directories, alongside test_hicache_fast_file_unit.py.
    uv run pytest test/registered/unit/mem_cache/test_hicache_fast_file_striping_unit.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import inspect
import os
import re
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
    PoolName,
)
from sglang.srt.mem_cache.storage.fast_file.fast_file_store import HiCacheFastFile
from sglang.srt.mem_cache.storage.fast_file.lru_file_evictor import (
    DiskRouter,
    LRUFileEvictor,
    StripedEvictor,
)
from sglang.test.test_utils import CustomTestCase


class _FakePageFirstPool:
    """Page-first host pool stand-in: kv_buffer[k_or_v, token, byte]."""

    def __init__(self, *, size: int = 256, page_size: int = 2, direct: bool = True):
        self.layout = "page_first" if direct else "layer_first"
        self.page_size = page_size
        self.size = size
        self.dtype = torch.uint8
        self.size_per_token = 4
        self.kv_buffer = torch.zeros((2, size, self.size_per_token), dtype=torch.uint8)

    def get_page_buffer_meta(self, indices):
        ptrs, sizes = [], []
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

    @property
    def page_bytes(self):
        return 2 * self.page_size * self.size_per_token


def _config(extra_config, **overrides) -> HiCacheStorageConfig:
    fields = dict(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="testmodel",
        extra_config=extra_config,
    )
    fields.update(overrides)
    return HiCacheStorageConfig(**fields)


class _Base(CustomTestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="hicache_fast_file_striping_")
        self.backends = []
        self._env = [
            envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(None),
            envs.SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE.override("0"),
            envs.SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE.override("0"),
        ]
        for cm in self._env:
            cm.__enter__()

    def tearDown(self):
        for backend in self.backends:
            backend.close()
        for cm in reversed(self._env):
            cm.__exit__(None, None, None)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def roots(self, *names):
        return [os.path.join(self.tmpdir, name) for name in names]

    def make(self, storage_dir, pool=None, **extra):
        extra.setdefault("evict_high_watermark", 1.0)
        extra.setdefault("preevict_interval_ms", 100_000)
        backend = HiCacheFastFile(_config({"storage_dir": storage_dir, **extra}))
        self.backends.append(backend)
        if pool is not None:
            backend.register_mem_pool_host(pool)
        return backend

    def reopen(self, backend, storage_dir, pool=None, **extra):
        backend.close()
        self.backends.remove(backend)
        return self.make(storage_dir, pool, **extra)

    @staticmethod
    def fill(pool, num_pages):
        for page in range(num_pages):
            start = page * pool.page_size
            pool.kv_buffer[:, start : start + pool.page_size] = page + 1

    @staticmethod
    def indices(pool, num_pages):
        return torch.arange(num_pages * pool.page_size)

    @staticmethod
    def pages_on(path):
        return sorted(name for name in os.listdir(path) if name.endswith(".bin"))


class TestDiskRouter(CustomTestCase):
    IDS = ["/mnt/nvme0", "/mnt/nvme1", "/mnt/nvme2", "/mnt/nvme3"]

    def test_routes_are_stable_and_do_not_depend_on_list_order(self):
        keys = [f"page{i}_model" for i in range(2000)]
        forward = DiskRouter(self.IDS)
        backward = DiskRouter(list(reversed(self.IDS)))
        again = DiskRouter(self.IDS)
        for key in keys:
            self.assertEqual(forward.route(key), again.route(key))
            self.assertEqual(
                self.IDS[forward.route(key)],
                list(reversed(self.IDS))[backward.route(key)],
            )

    def test_keys_and_buckets_are_balanced(self):
        router = DiskRouter(self.IDS)
        self.assertAlmostEqual(sum(router.shares), 1.0)
        for share in router.shares:
            self.assertAlmostEqual(share, 0.25, delta=0.03)
        counts = [0] * len(self.IDS)
        for i in range(20000):
            counts[router.route(os.urandom(8).hex())] += 1
        for root, count in enumerate(counts):
            self.assertAlmostEqual(count / 20000, router.shares[root], delta=0.03)

    def test_removing_a_root_only_moves_the_keys_it_held(self):
        full = DiskRouter(self.IDS)
        remaining = [self.IDS[0], self.IDS[1], self.IDS[3]]
        shrunk = DiskRouter(remaining)
        moved = 0
        for i in range(5000):
            key = f"k{i}"
            before = self.IDS[full.route(key)]
            after = remaining[shrunk.route(key)]
            if before != self.IDS[2]:
                self.assertEqual(before, after, key)
            else:
                moved += 1
        self.assertGreater(moved, 0)

    def test_single_root_and_invalid_lists(self):
        self.assertEqual(DiskRouter(["/only"]).route("anything"), 0)
        with self.assertRaises(ValueError):
            DiskRouter([])
        with self.assertRaises(ValueError):
            DiskRouter(["/a", "/a"])


class TestStorageRoots(_Base):
    def test_single_root_keeps_the_existing_layout_and_evictor(self):
        (root,) = self.roots("one")
        b = self.make(root)
        self.assertEqual(b.storage_roots, [root])
        self.assertEqual(b.file_paths, [b.file_path])
        self.assertEqual(os.path.dirname(b.file_path), root)
        self.assertIsInstance(b._evictor, LRUFileEvictor)
        self.assertTrue(b.set("k", torch.full((16,), 3, dtype=torch.uint8)))
        self.assertTrue(
            os.path.exists(os.path.join(b.file_path, f"k{b.config_suffix}.bin"))
        )

    def test_list_and_comma_string_are_equivalent(self):
        roots = self.roots("a", "b", "c")
        as_list = self.make(roots)
        as_text = self.make(" , ".join(roots) + ",")
        self.assertEqual(as_list.storage_roots, roots)
        self.assertEqual(as_text.storage_roots, roots)
        self.assertEqual(as_list.file_paths, as_text.file_paths)
        self.assertIsInstance(as_list._evictor, StripedEvictor)
        for path in as_list.file_paths:
            self.assertTrue(os.path.isdir(path))
            self.assertEqual(
                os.path.basename(path), os.path.basename(as_list.file_path)
            )

    def test_duplicate_or_empty_roots_are_rejected(self):
        a, b = self.roots("a", "b")
        os.makedirs(a)
        os.symlink(a, b)
        for storage_dir in ([a, a + "/"], [a, b], " , "):
            with self.subTest(storage_dir=storage_dir):
                with self.assertRaises(ValueError):
                    self.make(storage_dir)

    def test_env_var_stays_a_single_path(self):
        joined = ",".join(self.roots("x", "y"))
        with envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(joined):
            b = HiCacheFastFile(_config({}))
        self.backends.append(b)
        self.assertEqual(b.storage_roots, [joined])


class TestStripedPages(_Base):
    def test_each_page_lives_on_exactly_the_root_it_routes_to(self):
        pool = _FakePageFirstPool()
        b = self.make(self.roots("a", "b", "c"), pool, read_workers=3, write_workers=3)
        num_pages = 96
        self.fill(pool, num_pages)
        keys = [f"p{i}" for i in range(num_pages)]
        indices = self.indices(pool, num_pages)
        self.assertEqual(b.batch_set_v1(keys, indices), [True] * num_pages)

        per_root = [set(self.pages_on(path)) for path in b.file_paths]
        self.assertTrue(all(per_root), "every root should receive pages")
        for key in keys:
            name = f"{b._get_suffixed_key(key)}.bin"
            holders = [i for i, names in enumerate(per_root) if name in names]
            self.assertEqual(holders, [b._router.route(b._get_suffixed_key(key))])

        pool.kv_buffer.zero_()
        self.assertEqual(b.batch_get_v1(keys, indices), [True] * num_pages)
        for page in range(num_pages):
            start = page * pool.page_size
            self.assertTrue(
                torch.all(pool.kv_buffer[:, start : start + pool.page_size] == page + 1)
            )

    def test_existence_checks_follow_the_route(self):
        pool = _FakePageFirstPool()
        b = self.make(self.roots("a", "b"), pool, read_workers=2)
        keys = [f"p{i}" for i in range(16)]
        b.batch_set_v1(keys, self.indices(pool, len(keys)))
        self.assertTrue(all(b.exists(k) for k in keys))
        self.assertEqual(b.batch_exists_v2(keys).kv_hit_pages, len(keys))
        os.remove(b._get_component_path(keys[5]))
        self.assertFalse(b.exists(keys[5]))
        self.assertEqual(b.batch_exists_v2(keys).kv_hit_pages, 5)

    def test_clear_empties_every_root(self):
        pool = _FakePageFirstPool()
        b = self.make(self.roots("a", "b", "c"), pool)
        keys = [f"p{i}" for i in range(32)]
        b.batch_set_v1(keys, self.indices(pool, len(keys)))
        self.assertTrue(b.clear())
        for path in b.file_paths:
            self.assertEqual(self.pages_on(path), [])

    def test_metadata_cache_is_seeded_from_every_root(self):
        pool = _FakePageFirstPool()
        roots = self.roots("a", "b")
        b = self.make(roots, pool)
        keys = [f"p{i}" for i in range(16)]
        b.batch_set_v1(keys, self.indices(pool, len(keys)))
        b = self.reopen(b, roots, pool, enable_metadata_cache=True, metadata_ttl=-1)
        for key in keys:
            self.assertTrue(b.metadata_cache.contains(b._get_suffixed_key(key)))

    def test_changing_the_root_list_only_loses_moved_pages(self):
        pool = _FakePageFirstPool()
        two = self.roots("a", "b")
        three = self.roots("a", "b", "c")
        b = self.make(two, pool)
        num_pages = 64
        self.fill(pool, num_pages)
        keys = [f"p{i}" for i in range(num_pages)]
        b.batch_set_v1(keys, self.indices(pool, num_pages))
        old_home = {k: two[b._router.route(b._get_suffixed_key(k))] for k in keys}

        # Eviction is off: the startup scan still reclaims pages that the new
        # root list routes elsewhere, so they cannot leak.
        b = self.reopen(b, three, pool)
        new_home = {k: three[b._router.route(b._get_suffixed_key(k))] for k in keys}
        stayed = [k for k in keys if old_home[k] == new_home[k]]
        moved = [k for k in keys if old_home[k] != new_home[k]]
        self.assertTrue(stayed and moved)
        self.assertTrue(all(new_home[k] == three[2] for k in moved))
        for root, path in enumerate(b.file_paths):
            for name in self.pages_on(path):
                self.assertEqual(b._router.route(name[:-4]), root, name)

        results = [b.batch_get_v1([k], self.indices(pool, 1))[0] for k in keys]
        self.assertEqual(
            [k for k, ok in zip(keys, results) if ok], stayed, "moved pages must miss"
        )

    def test_every_inherited_single_directory_method_is_overridden(self):
        """HiCacheFile methods that build paths from ``self.file_path`` would
        silently cover only the first root; fast_file must override them all."""
        single_dir = re.compile(r"self\.file_path\b")
        checked = []
        for name in vars(HiCacheFile):
            member = getattr(HiCacheFile, name)
            if name == "__init__" or not inspect.isfunction(member):
                continue
            if not single_dir.search(inspect.getsource(member)):
                continue
            checked.append(name)
            with self.subTest(method=name):
                self.assertIn(name, vars(HiCacheFastFile))
                override = inspect.getsource(getattr(HiCacheFastFile, name))
                self.assertIsNone(single_dir.search(override))
        self.assertIn("_get_component_path", checked)  # the guard is not vacuous


class TestStripedEviction(_Base):
    def test_cap_is_split_by_key_share_and_enforced_per_root(self):
        pool = _FakePageFirstPool()
        page = pool.page_bytes
        cap = 30 * page
        b = self.make(self.roots("a", "b", "c"), pool, max_size=cap, write_workers=4)
        caps = [ev.max_size_bytes for ev in b._evictor.evictors]
        for root, root_cap in enumerate(caps):
            self.assertEqual(root_cap, int(cap * b._router.shares[root]))
        self.assertLessEqual(sum(caps), cap)

        for batch in range(4):
            keys = [f"b{batch}p{i}" for i in range(32)]
            b.batch_set_v1(keys, self.indices(pool, 32))
        for evictor, path in zip(b._evictor.evictors, b.file_paths):
            snapshot = evictor.snapshot()
            on_disk = sum(
                os.path.getsize(os.path.join(path, name))
                for name in self.pages_on(path)
            )
            self.assertLessEqual(snapshot["total_bytes"], evictor.max_size_bytes)
            self.assertEqual(snapshot["total_bytes"], on_disk)
            self.assertEqual(snapshot["pending_writes"], 0)
        merged = b._evictor.snapshot()
        self.assertEqual(
            merged["total_bytes"],
            sum(ev.snapshot()["total_bytes"] for ev in b._evictor.evictors),
        )

    def test_free_space_floor_applies_to_each_root(self):
        pool = _FakePageFirstPool()
        b = self.make(self.roots("a", "b"), pool, min_free_space=1 << 20)
        full, healthy = b._evictor.evictors
        with (
            mock.patch.object(full, "_fs_stats", return_value=(1 << 30, 0)),
            mock.patch.object(healthy, "_fs_stats", return_value=(1 << 30, 1 << 30)),
        ):
            keys = [f"p{i}" for i in range(32)]
            results = b.batch_set_v1(keys, self.indices(pool, len(keys)))
        for key, ok in zip(keys, results):
            root = b._router.route(b._get_suffixed_key(key))
            self.assertEqual(ok, root == 1, key)
        self.assertEqual(self.pages_on(b.file_paths[0]), [])


class TestParallelWrites(_Base):
    def test_default_is_serial_and_close_stops_the_pool(self):
        (root,) = self.roots("a")
        self.assertIsNone(self.make(root)._write_executor)
        b = self.make(root, write_workers=3)
        executor = b._write_executor
        self.assertIsNotNone(executor)
        b.close()
        self.assertIsNone(b._write_executor)
        with self.assertRaises(RuntimeError):
            executor.submit(lambda: None)

    def test_invalid_worker_counts_are_rejected(self):
        (root,) = self.roots("a")
        for name in ("read_workers", "write_workers"):
            for value in (0, -1, "x", True):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        self.make(root, **{name: value})

    def _tracked_writes(self, backend, delay=0.01, fail=()):
        names = []
        lock = threading.Lock()
        original = backend._write_page_buffers

        def tracked(key, buffers):
            with lock:
                names.append(threading.current_thread().name)
            time.sleep(delay)
            if key in fail:
                return False, 0
            return original(key, buffers)

        return names, mock.patch.object(backend, "_write_page_buffers", tracked)

    def test_direct_writes_run_on_the_write_pool_and_keep_order(self):
        pool = _FakePageFirstPool()
        (root,) = self.roots("a")
        b = self.make(root, pool, read_workers=2, write_workers=4)
        keys = [f"p{i}" for i in range(12)]
        names, patch = self._tracked_writes(b, fail={"p3", "p7"})
        with patch:
            results = b.batch_set_v1(keys, self.indices(pool, len(keys)))
        self.assertEqual(results, [k not in ("p3", "p7") for k in keys])
        self.assertGreaterEqual(len(set(names)), 2)
        self.assertTrue(all(n.startswith("HiCacheFastFileWrite-") for n in names))
        read_names, read_lock = [], threading.Lock()
        original_read = b._read_page_buffers

        def tracked_read(key, buffers):
            with read_lock:
                read_names.append(threading.current_thread().name)
            return original_read(key, buffers)

        with mock.patch.object(b, "_read_page_buffers", tracked_read):
            b.batch_get_v1(keys, self.indices(pool, len(keys)))
        self.assertTrue(all(n.startswith("HiCacheFastFileRead-") for n in read_names))

    def test_staged_writes_are_parallel_too(self):
        pool = _FakePageFirstPool(direct=False)
        b = self.make(self.roots("a", "b"), pool, write_workers=4)
        self.assertFalse(b._pool_direct_io[PoolName.KV])
        num_pages = 16
        self.fill(pool, num_pages)
        keys = [f"p{i}" for i in range(num_pages)]
        seen = set()
        lock = threading.Lock()
        original = b._staged_write_page

        def tracked(*args):
            with lock:
                seen.add(threading.current_thread().name)
            time.sleep(0.005)
            return original(*args)

        with mock.patch.object(b, "_staged_write_page", tracked):
            self.assertEqual(
                b.batch_set_v1(keys, self.indices(pool, num_pages)), [True] * num_pages
            )
        self.assertGreaterEqual(len(seen), 2)
        pool.kv_buffer.zero_()
        self.assertEqual(
            b.batch_get_v1(keys, self.indices(pool, num_pages)), [True] * num_pages
        )
        self.assertTrue(torch.all(pool.kv_buffer[:, 0:2] == 1))

    def test_a_failing_write_does_not_return_before_the_others_finish(self):
        """The controller may reuse host pages as soon as the batch returns,
        so no worker may still be reading from them."""
        (root,) = self.roots("a")
        b = self.make(root, write_workers=2)
        slow_started, release, slow_done = (threading.Event() for _ in range(3))
        errors, returned = [], threading.Event()

        def write(key):
            if key == "bad":
                self.assertTrue(slow_started.wait(timeout=5))
                raise RuntimeError("injected write failure")
            slow_started.set()
            self.assertTrue(release.wait(timeout=5))
            slow_done.set()
            return True, 1

        def run():
            try:
                b._parallel_write_map(write, ["bad", "slow"])
            except RuntimeError as exc:
                errors.append(exc)
            finally:
                returned.set()

        caller = threading.Thread(target=run)
        caller.start()
        try:
            self.assertFalse(returned.wait(timeout=0.2))
        finally:
            release.set()
        caller.join(timeout=5)
        self.assertTrue(slow_done.is_set())
        self.assertEqual(len(errors), 1)

    def test_concurrent_same_key_writes_publish_one_complete_page(self):
        pool = _FakePageFirstPool()
        b = self.make(self.roots("a", "b"), pool, write_workers=8, max_size=1 << 20)
        self.fill(pool, 16)
        indices = torch.cat([self.indices(pool, 1)] * 16)  # one page, 16 times
        results = b.batch_set_v1(["same"] * 16, indices)
        self.assertEqual(results, [True] * 16)
        path = b._get_component_path("same")
        self.assertEqual(os.path.getsize(path), pool.page_bytes)
        snapshot = b._evictor.snapshot()
        self.assertEqual(snapshot["entries"], 1)
        self.assertEqual(snapshot["total_bytes"], pool.page_bytes)
        self.assertEqual(snapshot["pending_writes"], 0)
        leftovers = [
            name
            for path in b.file_paths
            for name in os.listdir(path)
            if ".tmp." in name
        ]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
