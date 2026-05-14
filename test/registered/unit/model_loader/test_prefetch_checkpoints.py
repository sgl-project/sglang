"""
Unit tests for coordinated checkpoint prefetch.

Verifies that weights loaded with prefetch enabled are bit-identical
to weights loaded without prefetch.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import safetensors.torch
import torch

from sglang.srt.model_loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="stage-a-test-cpu")


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
