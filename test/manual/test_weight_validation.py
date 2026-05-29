"""
Unit tests for weight validation and cache cleanup logic.

Tests the fix for issue #14754 - ensuring that missing shards do not trigger
entire cache deletion, which can cause race conditions in multi-process scenarios.
"""

import errno
import json
import os
import struct
import tempfile
import time
import unittest
from unittest import mock

from sglang.srt.model_loader.ci_weight_validation import (
    _check_index_files_exist,
    _ensure_lock_file_creatable,
    _preflight_reclaim_cache_space,
    _reclaim_stale_incomplete_blobs,
    _validate_sharded_model,
)


class TestWeightValidation(unittest.TestCase):
    """Tests for weight validation functions."""

    def test_validate_sharded_model_missing_shard(self):
        """
        Test that missing shards are detected correctly.

        This is the core test for issue #14754 fix: when a shard is missing,
        the validation should return is_valid=False with an error message
        containing "Missing", but corrupted_files should be empty (indicating
        this is a missing shard issue, not a corruption issue).

        This distinction is critical because:
        - Missing shards: should NOT delete cache (other processes may be using it)
        - Corrupted files: should delete only the corrupted files selectively
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create partial shards (missing shard 3)
            for i in [1, 2]:  # Missing shard 3
                open(
                    os.path.join(tmpdir, f"model-0000{i}-of-00003.safetensors"), "w"
                ).close()

            # Create index file
            index_data = {
                "weight_map": {
                    "layer1": "model-00001-of-00003.safetensors",
                    "layer2": "model-00002-of-00003.safetensors",
                    "layer3": "model-00003-of-00003.safetensors",
                }
            }
            with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w") as f:
                json.dump(index_data, f)

            weight_files = [
                os.path.join(tmpdir, f"model-0000{i}-of-00003.safetensors")
                for i in [1, 2]
            ]

            is_valid, error_msg, corrupted_files = _validate_sharded_model(
                tmpdir, weight_files
            )

            self.assertFalse(is_valid)
            self.assertIn("Missing", error_msg)
            # CRITICAL: corrupted_files should be empty for missing shards
            # This is what prevents entire cache deletion
            self.assertEqual(corrupted_files, [])

    def test_validate_sharded_model_all_present(self):
        """Test that complete shards pass validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create all shards with valid safetensors header
            for i in [1, 2, 3]:
                filepath = os.path.join(tmpdir, f"model-0000{i}-of-00003.safetensors")
                # Create a minimal valid safetensors file
                # Header: 8 bytes for header size + JSON header
                header = b'{"__metadata__":{}}'
                header_size = len(header)
                with open(filepath, "wb") as f:
                    f.write(struct.pack("<Q", header_size))
                    f.write(header)

            # Create index file
            index_data = {
                "weight_map": {
                    "layer1": "model-00001-of-00003.safetensors",
                    "layer2": "model-00002-of-00003.safetensors",
                    "layer3": "model-00003-of-00003.safetensors",
                }
            }
            with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w") as f:
                json.dump(index_data, f)

            weight_files = [
                os.path.join(tmpdir, f"model-0000{i}-of-00003.safetensors")
                for i in [1, 2, 3]
            ]

            is_valid, error_msg, corrupted_files = _validate_sharded_model(
                tmpdir, weight_files
            )

            self.assertTrue(is_valid)
            self.assertIsNone(error_msg)
            self.assertEqual(corrupted_files, [])

    def test_validate_sharded_model_corrupted_shard(self):
        """
        Test that corrupted shards are detected and returned in corrupted_files.

        This tests the other branch: when a file exists but is corrupted
        (invalid safetensors format), it should be added to corrupted_files
        so that selective cleanup can remove just that file.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create shard 1 as valid
            filepath1 = os.path.join(tmpdir, "model-00001-of-00003.safetensors")
            header = b'{"__metadata__":{}}'
            with open(filepath1, "wb") as f:
                f.write(struct.pack("<Q", len(header)))
                f.write(header)

            # Create shard 2 as corrupted (invalid header)
            filepath2 = os.path.join(tmpdir, "model-00002-of-00003.safetensors")
            with open(filepath2, "wb") as f:
                f.write(b"invalid data that is not a valid safetensors file")

            # Create shard 3 as valid
            filepath3 = os.path.join(tmpdir, "model-00003-of-00003.safetensors")
            with open(filepath3, "wb") as f:
                f.write(struct.pack("<Q", len(header)))
                f.write(header)

            # Create index file
            index_data = {
                "weight_map": {
                    "layer1": "model-00001-of-00003.safetensors",
                    "layer2": "model-00002-of-00003.safetensors",
                    "layer3": "model-00003-of-00003.safetensors",
                }
            }
            with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w") as f:
                json.dump(index_data, f)

            weight_files = [filepath1, filepath2, filepath3]

            is_valid, error_msg, corrupted_files = _validate_sharded_model(
                tmpdir, weight_files
            )

            self.assertFalse(is_valid)
            self.assertIn("Corrupt", error_msg)
            # The corrupted file should be identified
            self.assertEqual(len(corrupted_files), 1)
            self.assertIn("model-00002-of-00003.safetensors", corrupted_files[0])

    def test_broken_index_symlink_detected(self):
        """
        Test that broken index symlinks are detected and cause validation to fail.

        When an index file is a symlink pointing to a non-existent blob,
        validation should fail (to trigger re-download) rather than silently
        continuing and causing timeout during actual loading.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a broken symlink for the index file
            index_path = os.path.join(tmpdir, "model.safetensors.index.json")
            non_existent_blob = os.path.join(tmpdir, "blobs", "nonexistent_hash")
            os.symlink(non_existent_blob, index_path)

            # Verify it's a broken symlink
            self.assertTrue(os.path.islink(index_path))
            self.assertFalse(os.path.exists(index_path))

            # Check should fail for broken symlink
            is_valid, error_msg = _check_index_files_exist(tmpdir)

            self.assertFalse(is_valid)
            self.assertIn("Broken", error_msg)
            # The broken symlink should have been cleaned up
            self.assertFalse(os.path.exists(index_path))
            self.assertFalse(os.path.islink(index_path))


class TestCacheSpaceReclaim(unittest.TestCase):
    """Tests for the shared-CI-cache ENOSPC mitigation.

    Covers age-gated reclaim of orphaned ``*.incomplete`` partial downloads and
    the actionable out-of-space error raised when the lock file cannot be
    created.
    """

    def _make_incomplete(self, cache_dir, repo, name, age_seconds, size=1024):
        blobs = os.path.join(cache_dir, repo, "blobs")
        os.makedirs(blobs, exist_ok=True)
        path = os.path.join(blobs, name)
        with open(path, "wb") as f:
            f.write(b"x" * size)
        mtime = time.time() - age_seconds
        os.utime(path, (mtime, mtime))
        return path

    def test_reclaim_removes_only_stale_incomplete(self):
        """Stale partials (old mtime) are removed; active ones and real blobs stay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stale_a = self._make_incomplete(
                tmpdir, "models--org--modelA", "aaa.incomplete", age_seconds=7200
            )
            stale_b = self._make_incomplete(
                tmpdir, "models--org--modelB", "bbb.incomplete", age_seconds=7200
            )
            # Fresh partial == active download in another process/container.
            fresh = self._make_incomplete(
                tmpdir, "models--org--modelA", "ccc.incomplete", age_seconds=10
            )
            # A real (complete) blob must never be touched.
            real_blob = os.path.join(tmpdir, "models--org--modelA", "blobs", "ddd")
            with open(real_blob, "wb") as f:
                f.write(b"y" * 2048)

            removed, freed = _reclaim_stale_incomplete_blobs(
                tmpdir, min_age_seconds=1800
            )

            self.assertEqual(removed, 2)
            self.assertEqual(freed, 2048)  # 2 stale files * 1024 bytes
            self.assertFalse(os.path.exists(stale_a))
            self.assertFalse(os.path.exists(stale_b))
            self.assertTrue(os.path.exists(fresh))
            self.assertTrue(os.path.exists(real_blob))

    def test_reclaim_noop_when_cache_dir_missing(self):
        self.assertEqual(
            _reclaim_stale_incomplete_blobs("/nonexistent/path/xyz", 0), (0, 0)
        )

    def test_preflight_noop_when_space_healthy(self):
        """With a 0 GB threshold, free space is never 'low', so nothing is reclaimed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stale = self._make_incomplete(
                tmpdir, "models--org--modelA", "aaa.incomplete", age_seconds=7200
            )
            with mock.patch.dict(os.environ, {"SGLANG_CI_CACHE_MIN_FREE_GB": "0"}):
                _preflight_reclaim_cache_space(tmpdir, "org/modelA")
            self.assertTrue(os.path.exists(stale))

    def test_preflight_reclaims_when_space_low(self):
        """An unreachable free-space threshold forces a reclaim pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stale = self._make_incomplete(
                tmpdir, "models--org--modelA", "aaa.incomplete", age_seconds=7200
            )
            fresh = self._make_incomplete(
                tmpdir, "models--org--modelA", "bbb.incomplete", age_seconds=10
            )
            with mock.patch.dict(
                os.environ, {"SGLANG_CI_CACHE_MIN_FREE_GB": "100000000"}
            ):
                _preflight_reclaim_cache_space(tmpdir, "org/modelA")
            self.assertFalse(os.path.exists(stale))
            self.assertTrue(os.path.exists(fresh))

    def test_ensure_lock_creatable_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "download_test.lock")
            _ensure_lock_file_creatable(lock_path, tmpdir, "org/modelA")
            self.assertTrue(os.path.exists(lock_path))

    def test_ensure_lock_creatable_enospc_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "download_test.lock")

            def fake_open(*args, **kwargs):
                raise OSError(errno.ENOSPC, "No space left on device")

            with mock.patch.object(os, "open", side_effect=fake_open):
                with self.assertRaises(RuntimeError) as ctx:
                    _ensure_lock_file_creatable(lock_path, tmpdir, "org/modelA")
            self.assertIn("out of disk space", str(ctx.exception))
            self.assertIn("org/modelA", str(ctx.exception))

    def test_ensure_lock_creatable_ignores_non_enospc(self):
        """Non-ENOSPC errors are swallowed so filelock can surface them normally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "download_test.lock")

            def fake_open(*args, **kwargs):
                raise OSError(errno.EACCES, "Permission denied")

            with mock.patch.object(os, "open", side_effect=fake_open):
                # Should not raise.
                _ensure_lock_file_creatable(lock_path, tmpdir, "org/modelA")


if __name__ == "__main__":
    unittest.main()
