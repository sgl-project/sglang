"""
E2E test for HiCache file storage with EAGLE3 speculative decoding.

Usage:
    python3 -m pytest test/registered/hicache/test_hicache_spec_file_storage.py -v
"""

import os
import shutil
import tempfile
import time
import unittest

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.hicache_spec_storage_common import HiCacheSpecStorageMixin
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=148, stage="extra-a", runner_config="1-gpu-large")


@unittest.skipIf(is_hip(), "HiCache + EAGLE3 file-storage loadback e2e is CUDA-only.")
class TestHiCacheSpecFileStorage(HiCacheSpecStorageMixin, CustomTestCase):
    storage_backend = "file"
    expected_storage_backend = "HiCacheFile"
    storage_wait_timeout = 30

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.process = None
        cls._launch_spec_server()

    @classmethod
    def _get_spec_server_env(cls):
        return {"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir}

    @classmethod
    def _count_file_storage_pages(cls):
        try:
            filenames = os.listdir(cls.temp_dir)
        except FileNotFoundError:
            return 0, 0

        target_pages = 0
        draft_pages = 0
        for filename in filenames:
            if not filename.endswith(".bin"):
                continue
            if f".{PoolName.DRAFT}" in filename:
                draft_pages += 1
            else:
                target_pages += 1
        return target_pages, draft_pages

    @classmethod
    def _wait_for_file_storage_pages(cls):
        min_pages = (cls.input_token_len - 2 * cls.page_size) // cls.page_size
        deadline = time.monotonic() + cls.storage_wait_timeout
        target_pages = draft_pages = 0

        while time.monotonic() < deadline:
            target_pages, draft_pages = cls._count_file_storage_pages()
            if target_pages >= min_pages and draft_pages >= min_pages:
                return target_pages, draft_pages
            time.sleep(0.2)

        raise AssertionError(
            "Timed out waiting for HiCache file storage pages before restart: "
            f"{target_pages=}, {draft_pages=}, {min_pages=}"
        )

    @classmethod
    def tearDownClass(cls):
        cls._stop_spec_server()
        if hasattr(cls, "temp_dir"):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _wait_for_storage_before_restart(self):
        target_pages, draft_pages = self._wait_for_file_storage_pages()
        print(f"file_storage_before_restart: {target_pages=}, {draft_pages=}")

    def test_file_storage_loadback_keeps_spec_accept_length(self):
        self._run_storage_loadback_keeps_spec_accept_length()


if __name__ == "__main__":
    unittest.main(verbosity=2)
