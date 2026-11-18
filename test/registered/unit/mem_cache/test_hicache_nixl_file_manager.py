"""Unit tests for NixlFileManager scoped clear -- no server, no NIXL agent."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import os
import shutil
import tempfile
import unittest

from sglang.srt.mem_cache.storage.nixl.nixl_utils import (
    NixlFileManager,
    _name_matches_suffix,
)
from sglang.test.test_utils import CustomTestCase


class TestNameMatchesSuffix(CustomTestCase):
    """Derived boundary property of the suffix matcher.

    A match must land on a key-component boundary (suffix followed by end of
    name or an underscore). A "looks equivalent" rewrite to plain substring
    matching would let an instance of model ``x`` delete model ``x-2`` files.
    """

    def test_matches_at_end_and_before_component_parts(self):
        self.assertTrue(_name_matches_suffix("page-a_qwen_0_8", "_qwen_0_8"))
        self.assertTrue(_name_matches_suffix("page-a_qwen_0_8_swa_k", "_qwen_0_8"))
        self.assertTrue(_name_matches_suffix("page-a_qwen", "_qwen"))
        self.assertTrue(_name_matches_suffix("page-a_qwen_mamba_temporal", "_qwen"))

    def test_rejects_partial_model_and_rank_overlaps(self):
        self.assertFalse(_name_matches_suffix("page-a_qwen-72b_0_8", "_qwen_0_8"))
        self.assertFalse(_name_matches_suffix("page-a_qwen_0_80", "_qwen_0_8"))
        self.assertFalse(_name_matches_suffix("page-a_qwen-72b", "_qwen"))


class TestNixlFileManagerClear(CustomTestCase):
    """Regression tests for #32693.

    clear() used to delete every file under every base directory, so one
    instance's /clear_hicache_storage destroyed the cached data of all other
    models/deployments sharing an L3 mount. It must only remove files that
    carry the clearing instance's key suffix.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_nixl_file_manager_")
        self.base_dirs = [os.path.join(self.test_dir, f"disk{i}") for i in range(2)]
        self.file_manager = NixlFileManager(self.base_dirs, use_direct_io=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _write_key(self, key: str) -> str:
        path = self.file_manager.get_file_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"x")
        return path

    def test_scoped_clear_spares_other_instances(self):
        mine = [
            self._write_key("page-a_qwen_0_8"),
            self._write_key("page-b_qwen_0_8"),
            self._write_key("page-b_qwen_0_8_swa_k"),
        ]
        others = [
            self._write_key("page-a_glm_0_8"),
            self._write_key("page-a_qwen-72b_0_8"),
            self._write_key("page-a_qwen_0_82"),
        ]

        self.file_manager.clear(suffix="_qwen_0_8")

        for path in mine:
            self.assertFalse(os.path.exists(path), path)
        for path in others:
            self.assertTrue(os.path.exists(path), path)

    def test_unscoped_clear_removes_everything_and_warns(self):
        paths = [
            self._write_key("page-a_qwen_0_8"),
            self._write_key("page-a_glm_0_8"),
        ]

        with self.assertLogs(
            "sglang.srt.mem_cache.storage.nixl.nixl_utils", level="WARNING"
        ):
            self.file_manager.clear()

        for path in paths:
            self.assertFalse(os.path.exists(path), path)

    def test_degenerate_suffix_falls_back_to_unscoped(self):
        path = self._write_key("page-a_glm_0_8")

        with self.assertLogs(
            "sglang.srt.mem_cache.storage.nixl.nixl_utils", level="WARNING"
        ):
            self.file_manager.clear(suffix="_")

        self.assertFalse(os.path.exists(path), path)


if __name__ == "__main__":
    unittest.main()
