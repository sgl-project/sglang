"""
Unit tests for weight validation and cache cleanup logic.

Tests the fix for issue #14754 - ensuring that missing shards do not trigger
entire cache deletion, which can cause race conditions in multi-process scenarios.
"""

import json
import os
import struct
import tempfile
import unittest

from sglang.srt.model_loader.weight_validation import (
    _check_index_files_exist,
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


if __name__ == "__main__":
    unittest.main()
