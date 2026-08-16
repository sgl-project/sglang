# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2 import (
    coords as magi2_coords,
)

COLUMNS = 9


class TestMagi2VideoAndTextCoords(unittest.TestCase):
    def test_text_is_offset_before_time_zero_with_a_singleton_reference(self):
        num_tokens = 7
        out = magi2_coords.text_coords(num_tokens=num_tokens)

        self.assertEqual(out.shape, (num_tokens, COLUMNS))
        self.assertTrue(
            torch.equal(
                out[:, 0],
                torch.arange(num_tokens, dtype=torch.float32) - num_tokens,
            )
        )
        self.assertLess(float(out[:, 0].max()), 0.0)
        self.assertEqual(out[:, 6].unique().tolist(), [1.0])


class TestMagi2AudioCoords(unittest.TestCase):
    def test_reference_time_compresses_by_eight_with_ceiling(self):
        for num_tokens in (1, 8, 9, 16, 17, 348):
            with self.subTest(num_tokens=num_tokens):
                out = magi2_coords.audio_coords(num_tokens=num_tokens)
                self.assertEqual(out.shape, (num_tokens, COLUMNS))
                self.assertEqual(float(out[0, 6]), float((num_tokens - 1) // 8 + 1))
                self.assertEqual(float(out[0, 3]), float(num_tokens))


class TestMagi2RefImageCoords(unittest.TestCase):
    def test_reference_images_sit_past_the_clip_with_the_documented_gap(self):
        video_time_steps = 6
        parts = magi2_coords.ref_image_coords(
            token_counts=[9, 4],
            feat_shapes=[(3, 3), (2, 2)],
            video_time_steps=video_time_steps,
        )
        self.assertEqual(len(parts), 4)

        for index, (sentinel, grid) in enumerate(zip(parts[::2], parts[1::2])):
            want_time = video_time_steps + magi2_coords.REF_IMAGE_TIME_GAP + index
            self.assertEqual(float(sentinel[0, 0]), float(want_time))
            self.assertEqual(grid[:, 0].unique().tolist(), [float(want_time)])
            self.assertGreater(want_time, video_time_steps)

    def test_sentinel_row_marks_the_whole_image_with_negative_positions(self):
        sentinel, grid = magi2_coords.ref_image_coords(
            token_counts=[9], feat_shapes=[(3, 3)], video_time_steps=4
        )
        self.assertEqual(sentinel.shape, (1, COLUMNS))
        self.assertEqual(sentinel[0, 1:3].tolist(), [-1.0, -1.0])
        self.assertEqual(sentinel[0, 3:6].tolist(), [1.0, 3.0, 3.0])
        self.assertTrue(torch.equal(sentinel[0, 3:6], sentinel[0, 6:9]))
        self.assertTrue((grid[:, 1:3] >= 0).all())


if __name__ == "__main__":
    unittest.main()
