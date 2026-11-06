"""Window layout agrees with direct enumeration for mixed image/video grids."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.qwen2_5_vl import Qwen2_5_VisionTransformer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def reference_layout(grids, window=4, merge=2):
    indices = []
    boundaries = [0]
    offset = 0
    for t, height, width in grids:
        h, w = height // merge, width // merge
        for frame in range(t):
            for y0 in range(0, h, window):
                for x0 in range(0, w, window):
                    for y in range(y0, min(y0 + window, h)):
                        for x in range(x0, min(x0 + window, w)):
                            indices.append(offset + frame * h * w + y * w + x)
                    boundaries.append(len(indices) * merge**2)
        offset += t * h * w
    return indices, boundaries


class TestQwen25WindowIndex(unittest.TestCase):
    def test_mixed_video_and_image_shapes(self):
        model = SimpleNamespace(
            window_size=112, spatial_merge_size=2, patch_size=14, spatial_merge_unit=4
        )
        for grids in (
            [(1, 16, 16)],
            [(2, 16, 24)],
            [(1, 14, 22)],
            [(1, 16, 14)],
            [(1, 14, 16)],
            [(1, 8, 8), (2, 14, 16), (1, 16, 24)],
            [(1, 48, 48)],
            [(2, 32, 32)] * 8,
            [(4, 64, 64)],
            [(5, 64, 64)],
            [(8, 64, 64)],
        ):
            with self.subTest(grids=grids):
                indices, boundaries = Qwen2_5_VisionTransformer.get_window_index(
                    model, torch.tensor(grids)
                )
                expected_indices, expected_boundaries = reference_layout(grids)
                self.assertEqual(indices.tolist(), expected_indices)
                self.assertEqual(boundaries, expected_boundaries)
                self.assertEqual(len(indices), sum(t * h * w // 4 for t, h, w in grids))


if __name__ == "__main__":
    unittest.main()
