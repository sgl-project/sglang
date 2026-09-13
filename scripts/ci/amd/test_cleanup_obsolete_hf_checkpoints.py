"""CPU tests for scripts/ci/amd/cleanup_obsolete_hf_checkpoints.sh."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cleanup_obsolete_hf_checkpoints.sh",
)


def _run(hf_home: str, extra_env=None):
    env = os.environ.copy()
    env["HF_HOME"] = hf_home
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", _SCRIPT],
        cwd=tempfile.gettempdir(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


class CleanupObsoleteHfCheckpoints(unittest.TestCase):
    def test_deletes_allowlisted_dirs_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp) / "hub"
            keep = hub / "models--Qwen--Qwen3.5-397B-A17B-FP8"
            drop = hub / "models--Qwen--Qwen3-235B-A22B-Instruct-2507"
            (keep / "blobs").mkdir(parents=True)
            (drop / "blobs").mkdir(parents=True)
            (keep / "blobs" / "keep.bin").write_text("keep")
            (drop / "blobs" / "drop.bin").write_text("drop")
            (hub / "models--Qwen--Qwen3-235B-A22B-Instruct-2507.lock").write_text(
                "lock"
            )

            result = _run(tmp)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(keep.exists())
            self.assertFalse(drop.exists())
            self.assertFalse(
                (hub / "models--Qwen--Qwen3-235B-A22B-Instruct-2507.lock").exists()
            )
            self.assertIn("removed=", result.stdout)

    def test_require_hf_cache_fails_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty-home"
            empty.mkdir()
            result = _run(str(empty), extra_env={"REQUIRE_HF_CACHE": "1"})
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
