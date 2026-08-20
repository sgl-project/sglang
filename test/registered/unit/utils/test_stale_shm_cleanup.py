import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.utils import stale_shm_cleanup
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestStaleShmCleanup(CustomTestCase):
    def test_make_shm_name_contains_creator_pid_and_is_unique(self):
        first = stale_shm_cleanup.make_shm_name("broadcast")
        second = stale_shm_cleanup.make_shm_name("broadcast")

        self.assertRegex(first, rf"^sgl_shm_broadcast_{os.getpid()}_[0-9a-f]{{8}}$")
        self.assertNotEqual(first, second)

    def test_creator_pid_parses_supported_names(self):
        cases = {
            "sgl_shm_broadcast_123_abcdef12": 123,
            "sgl_shm_kind_with_underscores_456_abcdef12": 456,
            "multi_tokenizer_args_789": 789,
            "sgl_shm_broadcast_not-a-pid_abcdef12": None,
            "multi_tokenizer_args_not-a-pid": None,
            "sgl_shm_too_short": None,
            "unrelated_123": None,
        }

        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(stale_shm_cleanup._creator_pid(filename), expected)

    def test_creator_pid_rejects_process_group_values(self):
        for pid in (0, -1, -42):
            with self.subTest(pid=pid):
                self.assertIsNone(
                    stale_shm_cleanup._creator_pid(f"sgl_shm_broadcast_{pid}_abcdef12")
                )
                self.assertIsNone(
                    stale_shm_cleanup._creator_pid(f"multi_tokenizer_args_{pid}")
                )

    def test_pid_alive_handles_process_probe_outcomes(self):
        with patch.object(stale_shm_cleanup.os, "kill") as kill:
            self.assertTrue(stale_shm_cleanup._pid_alive(123))
            kill.assert_called_once_with(123, 0)

        with patch.object(stale_shm_cleanup.os, "kill", side_effect=ProcessLookupError):
            self.assertFalse(stale_shm_cleanup._pid_alive(123))

        with patch.object(stale_shm_cleanup.os, "kill", side_effect=PermissionError):
            self.assertTrue(stale_shm_cleanup._pid_alive(123))

    def test_cleanup_is_disabled_outside_ci(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shm_dir = Path(temp_dir)
            stale = self._write_segment(shm_dir, "sgl_shm_test_123_deadbeef")

            with (
                patch.object(stale_shm_cleanup, "_SHM_DIR", shm_dir),
                patch.dict(os.environ, {"SGLANG_IS_IN_CI": "false"}),
            ):
                stale_shm_cleanup._cleanup_stale_shm_impl()

            self.assertTrue(stale.exists())

    def test_cleanup_removes_only_stale_and_orphaned_segments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shm_dir = Path(temp_dir)
            current_pid = os.getpid()
            keep_names = (
                f"sgl_shm_current_{current_pid}_aaaaaaaa",
                "sgl_shm_live_4242_bbbbbbbb",
                "multi_tokenizer_args_4242",
                "sgl_shm_invalid_not-a-pid_cccccccc",
                "unrelated-segment",
            )
            remove_names = (
                "sgl_shm_dead_4343_dddddddd",
                "multi_tokenizer_args_4343",
                "sglang_loads_snapshot",
                "cuda.shm.123",
                "nccl-communicator",
                "sem.loky-worker",
            )
            for name in (*keep_names, *remove_names):
                self._write_segment(shm_dir, name)

            with (
                patch.object(stale_shm_cleanup, "_SHM_DIR", shm_dir),
                patch.object(
                    stale_shm_cleanup, "_pid_alive", side_effect=lambda pid: pid == 4242
                ),
                patch.dict(os.environ, {"SGLANG_IS_IN_CI": "true"}),
            ):
                stale_shm_cleanup._cleanup_stale_shm_impl()

            for name in keep_names:
                with self.subTest(name=name, expected="preserved"):
                    self.assertTrue((shm_dir / name).exists())
            for name in remove_names:
                with self.subTest(name=name, expected="removed"):
                    self.assertFalse((shm_dir / name).exists())

    def test_cleanup_wrapper_never_propagates_sweep_failure(self):
        with (
            patch.object(
                stale_shm_cleanup,
                "_cleanup_stale_shm_impl",
                side_effect=RuntimeError("boom"),
            ),
            self.assertLogs(stale_shm_cleanup.logger, level="WARNING") as logs,
        ):
            stale_shm_cleanup.cleanup_stale_shm()

        self.assertIn("sweep failed, continuing startup", logs.output[0])

    @staticmethod
    def _write_segment(directory: Path, name: str) -> Path:
        path = directory / name
        path.write_bytes(b"segment")
        return path


if __name__ == "__main__":
    unittest.main()
