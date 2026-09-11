"""CPU tests for scripts/ci/amd/reclaim_and_seed_hf_cache.sh.

The script runs `rm -rf` against a volume the whole AMD fleet shares, so the
cases worth pinning down are the ones where it must refuse.
"""

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "reclaim_and_seed_hf_cache.sh",
)


def _make_hub(tmp: str, *names: str) -> Path:
    hub = Path(tmp) / "hub"
    for name in names:
        blobs = hub / name / "blobs"
        blobs.mkdir(parents=True)
        (blobs / "shard").write_text(name)
    return hub


def _run(hf_home: str, **env_overrides):
    env = os.environ.copy()
    env["HF_HOME"] = hf_home
    # Default to the inert configuration so each test opts in to what it needs.
    env.setdefault("RECLAIM_DIRS", "")
    env.setdefault("SEED_MODEL", "")
    # One check instead of twenty minutes of waiting for blocks that, in a
    # test, are never coming back.
    env.setdefault("SETTLE_SECONDS", "0")
    env.update({k: str(v) for k, v in env_overrides.items()})
    with tempfile.TemporaryDirectory() as workdir:
        return subprocess.run(
            ["bash", _SCRIPT],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )


class ReclaimAndSeedHfCache(unittest.TestCase):
    def test_audit_only_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            hub = _make_hub(tmp, "models--org--alpha", "models--org--beta")
            result = _run(tmp)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((hub / "models--org--alpha").exists())
            self.assertTrue((hub / "models--org--beta").exists())
            self.assertIn("audit only", result.stdout)

    def test_reclaims_listed_dirs_and_their_locks(self):
        with tempfile.TemporaryDirectory() as tmp:
            hub = _make_hub(tmp, "models--org--keep", "models--org--drop")
            (hub / "models--org--drop.lock").write_text("lock")

            result = _run(tmp, RECLAIM_DIRS="models--org--drop models--org--absent")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((hub / "models--org--keep").exists())
            self.assertFalse((hub / "models--org--drop").exists())
            self.assertFalse((hub / "models--org--drop.lock").exists())
            self.assertIn("removed=1 missing=1 failed=0", result.stdout)

    def test_refuses_a_name_that_is_a_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            hub = _make_hub(tmp, "models--org--alpha")
            for name in ("../../etc", "models--org--a/../../b", ".", "*"):
                result = _run(tmp, RECLAIM_DIRS=name)
                self.assertEqual(result.returncode, 2, f"{name}: {result.stderr}")
                self.assertIn("Refusing unsafe checkpoint name", result.stderr)
            self.assertTrue((hub / "models--org--alpha").exists())

    def test_refuses_to_delete_the_seed_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            hub = _make_hub(tmp, "models--amd--GLM-5.2-MXFP4")
            result = _run(
                tmp,
                RECLAIM_DIRS="models--amd--GLM-5.2-MXFP4",
                SEED_MODEL="amd/GLM-5.2-MXFP4",
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("Refusing to delete the seed target", result.stderr)
            self.assertTrue((hub / "models--amd--GLM-5.2-MXFP4").exists())

    def test_refuses_to_seed_what_cannot_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_hub(tmp, "models--amd--GLM-5.2-MXFP4")
            # No filesystem has an exabyte free, so this exercises the shortfall
            # branch without having to fill one up.
            result = _run(
                tmp,
                SEED_MODEL="amd/GLM-5.2-MXFP4",
                SEED_MIN_GIB=1_000_000_000,
            )
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("Refusing to start", result.stderr)
            self.assertIn("Waiting for the filesystem", result.stdout)

    def test_waits_for_space_rather_than_reading_df_straight_after_rm(self):
        # The reclaim that motivated the wait unlinked 372 GiB in under half a
        # second and df had not moved when it was read two milliseconds later.
        # A shortfall must cost the settle window before it is called one.
        with tempfile.TemporaryDirectory() as tmp:
            _make_hub(tmp, "models--amd--GLM-5.2-MXFP4")
            start = time.monotonic()
            result = _run(
                tmp,
                SEED_MODEL="amd/GLM-5.2-MXFP4",
                SEED_MIN_GIB=1_000_000_000,
                SETTLE_SECONDS=16,
            )
            waited = time.monotonic() - start
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertGreaterEqual(waited, 15)

    def test_fails_when_the_cache_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _run(os.path.join(tmp, "no-such-home"))
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("does not exist", result.stderr)


if __name__ == "__main__":
    unittest.main()
