"""Canvas M-RoPE layout for MiniCPM-V 4.7 (CPU only).

The expected positions come from the checkpoints' reference
``mrope_minicpmv4_7.py`` and pin the port to it.
"""

import unittest

import torch

from sglang.srt.multimodal.processors.minicpmv4_7 import (
    build_image_bounds,
    compute_canvas_positions,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Synthetic ids: the layout code only ever compares them for equality.
IM_START, IM_END, SLICE_START, SLICE_END, NEWLINE, PAD = 900, 901, 902, 903, 904, 905
SPECIAL_IDS = {
    "im_start_id": IM_START,
    "im_end_id": IM_END,
    "slice_start_id": SLICE_START,
    "slice_end_id": SLICE_END,
    "newline_id": NEWLINE,
}


def _canvas(input_ids, target_sizes):
    ids = torch.tensor(input_ids, dtype=torch.long)
    return compute_canvas_positions(
        ids, build_image_bounds(ids, SPECIAL_IDS), target_sizes, SPECIAL_IDS
    )


class TestMiniCPMV4_7CanvasMRoPE(CustomTestCase):
    def test_thumbnail_grid_places_rows_and_columns(self):
        # 16 visual tokens on a 4x4 grid: <image> halo sits one position below
        # the base, </image> one past the canvas.
        positions = _canvas(
            [10, 11, IM_START] + [PAD] * 16 + [IM_END, 12, 13], [(4, 4)]
        )

        self.assertEqual(positions[0].tolist(), [0, 1, 2] + [2] * 16 + [2, 7, 8])
        self.assertEqual(
            positions[1].tolist(),
            [0, 1, 1] + [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5] + [6, 7, 8],
        )
        self.assertEqual(
            positions[2].tolist(),
            [0, 1, 1] + [2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5] + [6, 7, 8],
        )

    def test_tightly_adjacent_frames_form_one_video_group(self):
        frame = [IM_START] + [PAD] * 16 + [IM_END]
        positions = _canvas([10, 11] + frame * 3 + [12, 13], [(4, 4)] * 3)

        # Each frame advances its own base instead of sharing the first one's.
        self.assertTrue((positions[0][20:36] > positions[0][3:19].max()).all())


if __name__ == "__main__":
    unittest.main()
