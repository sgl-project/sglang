import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from runner_disk_preflight import main, prune_diagnostics


class DiagnosticRetentionTest(unittest.TestCase):
    def test_retention_preserves_recent_newest_and_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = Path(directory)
            now = time.time()
            for name, age in (
                ("Worker_old.log", 10),
                ("Worker_recent.log", 1),
                ("Runner_old.log", 20),
                ("Runner_newest.log", 10),
                ("other.log", 20),
            ):
                path = diag / name
                path.write_bytes(b"test")
                os.utime(path, (now - age * 86400,) * 2)
            blocks = diag / "blocks"
            blocks.mkdir()
            (blocks / "pending-upload").write_bytes(b"keep")
            self.assertEqual(prune_diagnostics(diag, now), 8)
            self.assertEqual(
                {p.name for p in diag.iterdir()},
                {"Worker_recent.log", "Runner_newest.log", "other.log", "blocks"},
            )

    def test_symlinks_are_never_followed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            diag = base / "_diag"
            diag.mkdir()
            outside = base / "valuable"
            outside.write_bytes(b"keep")
            (diag / "Worker_link.log").symlink_to(outside)
            linked_diag = base / "linked_diag"
            linked_diag.symlink_to(diag, target_is_directory=True)
            self.assertEqual(prune_diagnostics(diag, time.time()), 0)
            self.assertEqual(prune_diagnostics(linked_diag, time.time()), 0)
            self.assertEqual(outside.read_bytes(), b"keep")

    def test_quota_exhaustion_fails_even_with_filesystem_space(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temp = root / "_work" / "_temp"
            temp.mkdir(parents=True)
            (root / "_diag").mkdir()
            (root / ".runner").write_text("{}")
            with (
                patch.dict(
                    os.environ,
                    {"RUNNER_TEMP": str(temp), "GITHUB_WORKSPACE": str(temp)},
                ),
                patch("runner_disk_preflight.shutil.which", return_value=None),
                patch("runner_disk_preflight.shutil.disk_usage") as usage,
                patch(
                    "runner_disk_preflight.tempfile.TemporaryFile",
                    side_effect=OSError(122, "Disk quota exceeded"),
                ),
            ):
                usage.return_value.free = 100 * 1024**3
                with self.assertRaisesRegex(SystemExit, "Disk quota exceeded"):
                    main()


if __name__ == "__main__":
    unittest.main()
