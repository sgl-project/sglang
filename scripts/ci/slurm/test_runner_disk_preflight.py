import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from runner_disk_preflight import main, prune_diagnostics, prune_scratch


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
                    {
                        "RUNNER_TEMP": str(temp),
                        "GITHUB_WORKSPACE": str(temp),
                        "RUNNER_NAME": "runner-a",
                        "HOME": str(root),
                    },
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


class ScratchReclaimTest(unittest.TestCase):
    def test_removes_only_this_runners_and_old_unowned_dirs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            scratch = base / ".mi355x_ci"
            now = time.time()
            for name, owner, age_days in (
                ("mine-crashed", "runner-a", 0),
                ("other-runner-live", "runner-b", 0),
                ("unowned-old", None, 2),
                ("unowned-recent", None, 0),
            ):
                leg = scratch / name
                (leg / "checkout").mkdir(parents=True)
                if owner is not None:
                    (leg / ".runner_name").write_text(owner + "\n")
                os.utime(leg, (now - age_days * 86400,) * 2)
            outside = base / "valuable"
            outside.mkdir()
            (scratch / "linked").symlink_to(outside, target_is_directory=True)

            removed = prune_scratch(scratch, runner_name="runner-a", now=now)

            self.assertEqual(sorted(removed), ["mine-crashed", "unowned-old"])
            self.assertEqual(
                {p.name for p in scratch.iterdir()},
                {"other-runner-live", "unowned-recent", "linked"},
            )
            self.assertTrue(outside.is_dir())

    def test_missing_scratch_dir_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                prune_scratch(Path(directory) / "absent", runner_name="r", now=0), []
            )


if __name__ == "__main__":
    unittest.main()
