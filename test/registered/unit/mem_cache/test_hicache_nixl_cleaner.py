"""Unit tests for the NIXL FILE L3 cleaner."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import os
import shutil
import tempfile
import unittest
from unittest import mock

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

    def _run_single_group_cleanup(self) -> None:
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

    def test_parse_group_key_strips_rank_and_kv_suffix(self):
        """Keys for TP ranks and zero-copy K/V files share one cleanup group."""
        self.assertEqual(_parse_group_key("page-a_model_0_8"), "page-a_model")
        self.assertEqual(_parse_group_key("page-a_model_7_8_k"), "page-a_model")
        self.assertEqual(_parse_group_key("page-a_model_7_8_v"), "page-a_model")
        self.assertEqual(_parse_group_key("page-a_model_k"), "page-a_model")

    def test_parse_group_key_strips_hybrid_component_suffix(self):
        """All hybrid component shapes share the logical page's cleanup group."""
        names = [
            "page-a_model_7_8_kv_k",
            "page-a_model_7_8_swa_k",
            "page-a_model_7_8_swa_v",
            "page-a_model_7_8_mamba_temporal",
            "page-a_model_7_8_mamba_conv_0",
            "page-a_model_7_8_indexer_2",
            "page-a_model_7_8_draft_swa",
            "page-a_model_deepseek_v4_c4_indexer_state_2",
        ]

        for name in names:
            with self.subTest(name=name):
                self.assertEqual(_parse_group_key(name), "page-a_model")

    def test_tick_deletes_oldest_group_across_bucketed_dirs(self):
        """A cleaner batch deletes all files in the oldest logical key group."""
        old_keys = ["page-old_model_0_2", "page-old_model_1_2"]
        new_keys = ["page-new_model_0_2", "page-new_model_1_2"]
        old_paths = [self._write_key(key, mtime=100.0) for key in old_keys]
        new_paths = [self._write_key(key, mtime=200.0) for key in new_keys]

        self._run_single_group_cleanup()
        self.assertFalse(any(os.path.exists(path) for path in old_paths))
        self.assertTrue(all(os.path.exists(path) for path in new_paths))

    def test_tick_deletes_hybrid_components_atomically(self):
        """Evict every pool component and TP rank for one logical page."""
        physical_suffixes = [
            "",
            "_k",
            "_v",
            "_kv_k",
            "_kv_v",
            "_swa_k",
            "_swa_v",
            "_mamba_temporal",
            "_mamba_conv_0",
        ]
        old_keys = [
            f"page-old_model_{rank}_2{suffix}"
            for rank in range(2)
            for suffix in physical_suffixes
        ]
        new_keys = [
            f"page-new_model_{rank}_2{suffix}"
            for rank in range(2)
            for suffix in physical_suffixes
        ]
        old_paths = [self._write_key(key, mtime=100.0) for key in old_keys]
        new_paths = [self._write_key(key, mtime=200.0) for key in new_keys]

        self._run_single_group_cleanup()
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

    def test_nixl_config_parses_l3_cleaner_capacity(self):
        """The capacity budget is optional and must be positive."""
        self.assertIsNone(NixlBackendConfig().get_l3_cleaner_config()["capacity_gb"])
        cfg = NixlBackendConfig({"l3_cleaner_capacity_gb": "512"})
        self.assertEqual(cfg.get_l3_cleaner_config()["capacity_gb"], 512.0)
        self.assertEqual(cfg.get_backend_initparams("POSIX"), {})
        with self.assertRaises(ValueError):
            NixlBackendConfig({"l3_cleaner_capacity_gb": 0}).get_l3_cleaner_config()

    def test_nixl_config_rejects_non_boolean_l3_cleaner_enabled(self):
        """Cleaner enablement uses native config booleans only."""
        cfg = NixlBackendConfig({"l3_cleaner_enabled": "false"})

        with self.assertRaises(ValueError):
            cfg.get_l3_cleaner_config()


def _fake_statvfs(used_pct: float, total_bytes: int = 1 << 40):
    """Return a statvfs stand-in for a filesystem at ``used_pct`` usage."""
    frsize = 4096
    blocks = total_bytes // frsize
    bavail = int(blocks * (100.0 - used_pct) / 100.0)
    return lambda _path: os.statvfs_result(
        (frsize, frsize, blocks, bavail, bavail, 0, 0, 0, 0, 255)
    )


class TestHiCacheL3CleanerCapacity(CustomTestCase):
    """Watermarks on a nearly full shared volume, with and without a budget."""

    GROUPS = 10
    FILE_SIZE = 1000

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_nixl_l3_cleaner_cap_")
        self.base_dirs = [os.path.join(self.test_dir, f"disk{i}") for i in range(2)]
        self.file_manager = NixlFileManager(self.base_dirs, use_direct_io=False)
        # Ten logical groups (two TP-rank files each), group i has mtime 100+i.
        self.group_paths = []
        for i in range(self.GROUPS):
            paths = []
            for rank in range(2):
                path = self.file_manager.get_file_path(f"page-{i:02d}_model_{rank}_2")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(b"x" * self.FILE_SIZE)
                os.utime(path, (100.0 + i, 100.0 + i))
                paths.append(path)
            self.group_paths.append(paths)
        self.own_bytes = self.GROUPS * 2 * self.FILE_SIZE
        # Other tenants' data keeps the shared volume at 90% regardless of what
        # the cleaner deletes.
        self.statvfs_patch = mock.patch(
            "sglang.srt.mem_cache.storage.nixl.nixl_cleaner.os.statvfs",
            _fake_statvfs(90.0),
        )
        self.statvfs_patch.start()

    def tearDown(self):
        self.statvfs_patch.stop()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _cleaner(self, capacity_bytes=None) -> HiCacheL3Cleaner:
        return HiCacheL3Cleaner(
            self.base_dirs,
            tp_rank=0,
            high_watermark=80.0,
            low_watermark=70.0,
            recheck_groups=1,
            unlink_workers=1,
            capacity_gb=(
                capacity_bytes / (1024**3) if capacity_bytes is not None else None
            ),
        )

    def _surviving_groups(self) -> list[int]:
        return [
            i
            for i, paths in enumerate(self.group_paths)
            if all(os.path.exists(path) for path in paths)
        ]

    def test_without_capacity_uses_filesystem_usage(self):
        """Default mode keeps statvfs semantics: a full shared volume empties L3."""
        cleaner = self._cleaner()
        with self.assertLogs(
            "sglang.srt.mem_cache.storage.nixl.nixl_cleaner", level="WARNING"
        ) as logs:
            self.assertTrue(cleaner._tick())
            self.assertFalse(cleaner._tick())
        self.assertEqual(self._surviving_groups(), [])
        shared_warnings = [
            line for line in logs.output if "l3_cleaner_capacity_gb" in line
        ]
        self.assertEqual(len(shared_warnings), 1)

    def test_capacity_below_high_watermark_keeps_everything(self):
        """Own usage at 50% of the budget deletes nothing on a 90% full volume."""
        cleaner = self._cleaner(capacity_bytes=2 * self.own_bytes)
        self.assertFalse(cleaner._tick())
        self.assertEqual(self._surviving_groups(), list(range(self.GROUPS)))

    def test_capacity_above_high_watermark_deletes_oldest_to_low_watermark(self):
        """Own usage at 100% deletes oldest groups until below 70% of budget."""
        cleaner = self._cleaner(capacity_bytes=self.own_bytes)
        self.assertTrue(cleaner._tick())
        # Each group is 10% of the budget: 100% -> 60% after four groups.
        self.assertEqual(self._surviving_groups(), list(range(4, self.GROUPS)))
        self.assertFalse(cleaner._tick())
        self.assertEqual(self._surviving_groups(), list(range(4, self.GROUPS)))

    def test_capacity_skips_scan_when_filesystem_usage_is_below_budget(self):
        """Used bytes on the filesystem bound own usage, so no scan is needed."""
        cleaner = self._cleaner(capacity_bytes=1 << 50)
        with mock.patch.object(cleaner, "_scan_base_dir") as scan:
            self.assertFalse(cleaner._tick())
        scan.assert_not_called()

    def test_rejects_non_positive_capacity(self):
        with self.assertRaises(ValueError):
            HiCacheL3Cleaner(self.base_dirs, tp_rank=0, capacity_gb=0)


if __name__ == "__main__":
    unittest.main()
