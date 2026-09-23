"""Unit tests for fast_file parallel writes.

Pure CPU tests on temp directories, alongside test_hicache_fast_file_unit.py.
    uv run pytest test/registered/unit/mem_cache/test_hicache_fast_file_striping_unit.py -v
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
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig, PoolName
from sglang.srt.mem_cache.storage.fast_file.fast_file_store import HiCacheFastFile
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

    @staticmethod
    def fill(pool, num_pages):
        for page in range(num_pages):
            start = page * pool.page_size
            pool.kv_buffer[:, start : start + pool.page_size] = page + 1

    @staticmethod
    def indices(pool, num_pages):
        return torch.arange(num_pages * pool.page_size)


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
        (root,) = self.roots("a")
        b = self.make(root, pool, write_workers=4)
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
        (root,) = self.roots("a")
        b = self.make(root, pool, write_workers=8, max_size=1 << 20)
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
        leftovers = [name for name in os.listdir(b.file_path) if ".tmp." in name]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
