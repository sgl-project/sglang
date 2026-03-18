"""
Unit tests for coordinated checkpoint prefetch.

Verifies that _prefetch_all_checkpoints correctly distributes files
across ranks, reads all bytes, and that weights are identical with
and without prefetch enabled.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import safetensors.torch
import torch

from sglang.srt.model_loader.weight_utils import (
    _prefetch_all_checkpoints,
    _prefetch_checkpoint_file,
    safetensors_weights_iterator,
)


class TestPrefetchReadsAllBytes(unittest.TestCase):
    """Verify prefetch reads the full file content, not just opens it."""

    def test_all_bytes_read(self):
        """_prefetch_checkpoint_file must read every byte in the file."""
        file_size = 16 * 1024 * 1024 * 3 + 7  # 3 full 16MB blocks + partial
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(os.urandom(file_size))
            path = f.name

        try:
            bytes_read = 0
            original_open = open

            class CountingReader:
                def __init__(self, fobj):
                    self._fobj = fobj

                def read(self, n=-1):
                    nonlocal bytes_read
                    data = self._fobj.read(n)
                    bytes_read += len(data)
                    return data

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    self._fobj.close()

            with patch(
                "builtins.open",
                lambda p, m: CountingReader(original_open(p, m)),
            ):
                _prefetch_checkpoint_file(path)

            self.assertEqual(bytes_read, file_size)
        finally:
            os.unlink(path)


class TestPrefetchDistributedOnlyReadsSubset(unittest.TestCase):
    """Verify each rank only prefetches its assigned fraction of files."""

    def _create_temp_files(self, n):
        paths = []
        for i in range(n):
            f = tempfile.NamedTemporaryFile(
                delete=False, suffix=f"-{i:05d}.safetensors"
            )
            f.write(b"x" * 1024)
            f.close()
            paths.append(f.name)
        return sorted(paths)

    def _cleanup(self, paths):
        for p in paths:
            os.unlink(p)

    @patch("torch.distributed.barrier")
    @patch("torch.distributed.get_world_size", return_value=4)
    @patch("torch.distributed.get_rank", return_value=1)
    @patch("torch.distributed.is_initialized", return_value=True)
    def test_rank_only_reads_its_assigned_files(
        self, mock_init, mock_rank, mock_world, mock_barrier
    ):
        """Rank 1 of 4 with 12 files should only read files at indices 1, 5, 9."""
        paths = self._create_temp_files(12)
        try:
            read_files = []
            original_fn = _prefetch_checkpoint_file

            def tracking_prefetch(p):
                read_files.append(p)
                original_fn(p)

            with patch(
                "sglang.srt.model_loader.weight_utils._prefetch_checkpoint_file",
                side_effect=tracking_prefetch,
            ):
                _prefetch_all_checkpoints(paths)

            expected = [paths[i] for i in [1, 5, 9]]
            self.assertEqual(sorted(read_files), sorted(expected))
            mock_barrier.assert_called_once()
        finally:
            self._cleanup(paths)

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_single_rank_reads_all_files(self, mock_init):
        """Without distributed, all files should be prefetched."""
        paths = self._create_temp_files(5)
        try:
            read_files = []
            original_fn = _prefetch_checkpoint_file

            def tracking_prefetch(p):
                read_files.append(p)
                original_fn(p)

            with patch(
                "sglang.srt.model_loader.weight_utils._prefetch_checkpoint_file",
                side_effect=tracking_prefetch,
            ):
                _prefetch_all_checkpoints(paths)

            self.assertEqual(sorted(read_files), sorted(paths))
        finally:
            self._cleanup(paths)


class TestPrefetchWeightsIdentical(unittest.TestCase):
    """Verify that loading with prefetch yields identical weights to without."""

    def _create_safetensors_files(self, tmpdir, num_shards=3):
        """Create real safetensors files with known tensor content."""
        paths = []
        for i in range(num_shards):
            tensors = {
                f"layer{i}.weight": torch.randn(32, 32),
                f"layer{i}.bias": torch.randn(32),
            }
            path = os.path.join(tmpdir, f"model-{i:05d}.safetensors")
            safetensors.torch.save_file(tensors, path)
            paths.append(path)
        return paths

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_weights_match_with_and_without_prefetch(self, _):
        """Tensors yielded must be bit-identical regardless of prefetch flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._create_safetensors_files(tmpdir)

            without = dict(safetensors_weights_iterator(paths, prefetch=False))
            with_pf = dict(safetensors_weights_iterator(paths, prefetch=True))

            self.assertEqual(set(without.keys()), set(with_pf.keys()))
            for name in without:
                torch.testing.assert_close(without[name], with_pf[name])


if __name__ == "__main__":
    unittest.main()
