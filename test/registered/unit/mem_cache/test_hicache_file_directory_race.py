"""CPU-only directory initialization checks; no server or GPU required."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.test.test_utils import CustomTestCase


def _config(path, *, pp_rank=0, tp_rank=0):
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=2,
        pp_rank=pp_rank,
        pp_size=2,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="directory-test",
        extra_config={
            "file_storage_path": path,
            "enable_metadata_cache": False,
            "max_size": 0,
            "min_free_space": 0,
        },
    )


class TestHiCacheFileDirectoryCreation(CustomTestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ)
        self.environment.start()
        os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", None)

    def tearDown(self):
        self.environment.stop()

    def test_pipeline_rank_zero_workers_can_create_one_directory(self):
        with tempfile.TemporaryDirectory() as parent:
            path = os.path.join(parent, "shared")
            checked_absent = threading.Barrier(2)
            real_exists = os.path.exists

            def exists(candidate):
                result = real_exists(candidate)
                if os.fspath(candidate) == path:
                    # Force both workers to observe the real absent directory
                    # before either proceeds; do not fake mkdir's result.
                    checked_absent.wait(timeout=5)
                return result

            errors = []
            backends = []
            with mock.patch("os.path.exists", side_effect=exists):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [
                        pool.submit(HiCacheFile, _config(path, pp_rank=rank))
                        for rank in (0, 1)
                    ]
                    for future in futures:
                        try:
                            backends.append(future.result(timeout=10))
                        except Exception as error:
                            errors.append(error)
            self.assertEqual(errors, [])
            self.assertEqual(len(backends), 2)
            self.assertTrue(os.path.isdir(path))
            self.assertNotEqual(backends[0].config_suffix, backends[1].config_suffix)

    def test_existing_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as path:
            sentinel = os.path.join(path, "keep")
            with open(sentinel, "w") as stream:
                stream.write("unchanged")
            backend = HiCacheFile(_config(path))
            self.assertEqual(backend.file_path, path)
            with open(sentinel) as stream:
                self.assertEqual(stream.read(), "unchanged")

    def test_nonzero_tp_rank_keeps_existing_directory_policy(self):
        with tempfile.TemporaryDirectory() as parent:
            path = os.path.join(parent, "absent")
            backend = HiCacheFile(_config(path, tp_rank=1))
            self.assertEqual(backend.file_path, path)
            self.assertFalse(os.path.exists(path))

    def test_creation_permission_error_is_not_swallowed(self):
        with tempfile.TemporaryDirectory() as parent:
            path = os.path.join(parent, "absent")
            with mock.patch("os.makedirs", side_effect=PermissionError("denied")):
                with self.assertRaises(PermissionError):
                    HiCacheFile(_config(path))


if __name__ == "__main__":
    unittest.main()
