# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for Triton-based vision rotary position embedding kernel.

Compares the Triton kernel output against the reference PyTorch implementation
used in Qwen2-VL vision encoder.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, suite="base-b-test-1-gpu-small")


def reference_rot_pos_emb(
    grid_thw: torch.Tensor,
    freqs_cached: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Reference implementation matching Qwen2VisionTransformer.rot_pos_emb."""
    pos_ids = []
    for i in range(grid_thw.size(0)):
        t, h, w = grid_thw[i].tolist()
        t, h, w = int(t), int(h), int(w)
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        hpos_ids = (
            hpos_ids.reshape(
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
            )
            .permute(0, 2, 1, 3)
            .flatten()
        )
        wpos_ids = (
            wpos_ids.reshape(
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
            )
            .permute(0, 2, 1, 3)
            .flatten()
        )
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    pos_ids = torch.cat(pos_ids, dim=0)
    rotary_pos_emb = freqs_cached[pos_ids].flatten(1)
    return rotary_pos_emb


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton kernel")
class TestVisionRotaryTriton(CustomTestCase):
    """Test Triton vision rotary pos embedding against reference."""

    def _run_test(self, grid_thw_list, spatial_merge_size=2, half_dim=32):
        """Helper to run a single test case."""
        from sglang.srt.layers.rotary_embedding.vision_rotary_triton import (
            triton_vision_rot_pos_emb,
        )

        grid_thw = torch.tensor(grid_thw_list, dtype=torch.int32)
        max_grid_size = grid_thw[:, 1:].max().item()

        # Build frequency cache (same as Qwen2VisionRotaryEmbedding)
        theta = 10000.0
        dim = half_dim * 2
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        seq = torch.arange(max_grid_size * 2, dtype=torch.float)
        freqs_cached = torch.outer(seq, inv_freq)  # [max_seq, half_dim]

        # Reference (CPU)
        ref_output = reference_rot_pos_emb(grid_thw, freqs_cached, spatial_merge_size)

        # Triton (GPU)
        freqs_cached_gpu = freqs_cached.cuda()
        triton_output = triton_vision_rot_pos_emb(
            grid_thw, freqs_cached_gpu, spatial_merge_size
        )

        # Compare
        self.assertEqual(ref_output.shape, triton_output.shape)
        torch.testing.assert_close(
            ref_output.cuda(),
            triton_output,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_single_image_small(self):
        """Single image with small grid."""
        self._run_test([[1, 4, 6]])

    def test_single_image_square(self):
        """Single square image."""
        self._run_test([[1, 8, 8]])

    def test_single_image_with_temporal(self):
        """Single image with temporal dimension > 1 (video)."""
        self._run_test([[3, 4, 6]])

    def test_multiple_images_same_size(self):
        """Multiple images with same grid size."""
        self._run_test([[1, 4, 4], [1, 4, 4], [1, 4, 4]])

    def test_multiple_images_different_sizes(self):
        """Multiple images with different grid sizes."""
        self._run_test([[1, 4, 6], [1, 8, 8], [2, 6, 4]])

    def test_large_grid(self):
        """Larger grid to stress-test the kernel."""
        self._run_test([[1, 28, 28]])

    def test_spatial_merge_size_1(self):
        """Edge case: spatial_merge_size=1 (no merging)."""
        self._run_test([[1, 6, 6]], spatial_merge_size=1)

    def test_various_half_dims(self):
        """Test with different frequency dimensions."""
        for half_dim in [16, 32, 64]:
            with self.subTest(half_dim=half_dim):
                self._run_test([[1, 4, 6]], half_dim=half_dim)

    def test_batch_with_video(self):
        """Batch with both images and videos (different t values)."""
        self._run_test([[1, 8, 8], [4, 4, 4], [1, 6, 6]])

    def test_large_batch(self):
        """Larger batch to ensure kernel handles many images."""
        grids = [[1, 4, 4]] * 16
        self._run_test(grids)


if __name__ == "__main__":
    unittest.main()