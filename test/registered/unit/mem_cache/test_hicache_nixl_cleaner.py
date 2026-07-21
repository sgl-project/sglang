"""Unit tests for the NIXL FILE L3 cleaner."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import os
import shutil
import tempfile
import unittest

from sglang.srt.mem_cache.storage.nixl.nixl_cleaner import (
    HiCacheL3Cleaner,
    _parse_group_key,
    _safe_unlink,
)
from sglang.srt.mem_cache.storage.nixl.nixl_utils import (
    NixlBackendConfig,
    NixlFileManager,
)
from sglang.test.test_utils import CustomTestCase


class TestHiCacheL3Cleaner(CustomTestCase):
    """Tests for watermark-driven cleanup over bucketed NIXL FILE layout."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_nixl_l3_cleaner_")
        self.base_dirs = [os.path.join(self.test_dir, f"disk{i}") for i in range(2)]
        self.file_manager = NixlFileManager(self.base_dirs, use_direct_io=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _write_key(self, key: str, *, mtime: float, size: int = 16) -> str:
        path = self.file_manager.get_file_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"x" * size)
        os.utime(path, (mtime, mtime))
        return path

    def test_parse_group_key_strips_rank_and_kv_suffix(self):
        """Keys for TP ranks and zero-copy K/V files share one cleanup group."""
        self.assertEqual(_parse_group_key("page-a_model_0_8"), "page-a_model")
        self.assertEqual(_parse_group_key("page-a_model_7_8_k"), "page-a_model")
        self.assertEqual(_parse_group_key("page-a_model_7_8_v"), "page-a_model")
        self.assertEqual(_parse_group_key("page-a_model_k"), "page-a_model")

    def test_tick_deletes_oldest_group_across_bucketed_dirs(self):
        """A cleaner batch deletes all files in the oldest logical key group."""
        old_keys = ["page-old_model_0_2", "page-old_model_1_2"]
        new_keys = ["page-new_model_0_2", "page-new_model_1_2"]
        old_paths = [self._write_key(key, mtime=100.0) for key in old_keys]
        new_paths = [self._write_key(key, mtime=200.0) for key in new_keys]

        cleaner = HiCacheL3Cleaner(
            self.base_dirs,
            tp_rank=0,
            high_watermark=80.0,
            low_watermark=70.0,
            recheck_groups=1,
            unlink_workers=1,
        )

        usage_calls: dict[str, int] = {}

        def fake_usage(path: str) -> float:
            usage_calls[path] = usage_calls.get(path, 0) + 1
            return 90.0 if usage_calls[path] == 1 else 60.0

        cleaner._disk_usage_pct = fake_usage

        self.assertTrue(cleaner._tick())
        self.assertFalse(any(os.path.exists(path) for path in old_paths))
        self.assertTrue(all(os.path.exists(path) for path in new_paths))

    def test_tick_ignores_non_bucket_directories(self):
        """Only hash-bucket directories are treated as NIXL FILE cache entries."""
        non_bucket = os.path.join(self.base_dirs[0], "not-a-bucket")
        os.makedirs(non_bucket, exist_ok=True)
        unrelated = os.path.join(non_bucket, "page-old_model_0_2")
        with open(unrelated, "wb") as f:
            f.write(b"x")

        cleaner = HiCacheL3Cleaner(
            self.base_dirs,
            tp_rank=0,
            high_watermark=80.0,
            low_watermark=70.0,
            unlink_workers=1,
        )
        cleaner._disk_usage_pct = lambda _path: 90.0

        self.assertFalse(cleaner._tick())
        self.assertTrue(os.path.exists(unrelated))

    def test_safe_unlink_tolerates_missing_and_os_errors(self):
        """Cleanup races should not fail the cleaner tick."""
        missing = os.path.join(self.test_dir, "missing")
        existing = os.path.join(self.test_dir, "existing")
        with open(existing, "wb") as f:
            f.write(b"abc")

        self.assertEqual(_safe_unlink(missing), (False, 0))
        self.assertEqual(_safe_unlink(self.test_dir), (False, 0))
        self.assertEqual(_safe_unlink(existing), (True, 3))
        self.assertFalse(os.path.exists(existing))

    def test_start_only_runs_on_tp_rank_zero(self):
        """Only TP rank 0 owns file cleanup for a shared storage directory."""
        cleaner = HiCacheL3Cleaner(self.base_dirs, tp_rank=1, interval_sec=0.01)
        cleaner.start()
        self.assertIsNone(cleaner._thread)

    def test_nixl_config_parses_l3_cleaner_options(self):
        """Cleaner settings are top-level NIXL config, not plugin init params."""
        cfg = NixlBackendConfig(
            {
                "use_uring": "true",
                "l3_cleaner_enabled": False,
                "l3_cleaner_high_watermark": "85",
                "l3_cleaner_low_watermark": 75,
            }
        )

        cleaner_config = cfg.get_l3_cleaner_config()
        self.assertFalse(cleaner_config["enabled"])
        self.assertEqual(cleaner_config["high_watermark"], 85.0)
        self.assertEqual(cleaner_config["low_watermark"], 75.0)
        self.assertEqual(cfg.get_backend_initparams("POSIX"), {"use_uring": "true"})

        default_config = NixlBackendConfig().get_l3_cleaner_config()
        self.assertTrue(default_config["enabled"])

    def test_nixl_config_rejects_non_boolean_l3_cleaner_enabled(self):
        """Cleaner enablement uses native config booleans only."""
        cfg = NixlBackendConfig({"l3_cleaner_enabled": "false"})

        with self.assertRaises(ValueError):
            cfg.get_l3_cleaner_config()


if __name__ == "__main__":
    unittest.main()
