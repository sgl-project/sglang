"""Canvas 3-D M-RoPE positions for MiniCPM-V 4.7 (CPU only).

4.7 places visual tokens on a per-image slice canvas instead of giving them
sequential positions; the expected tensors below were produced by the
checkpoints' reference ``mrope_minicpmv4_7.py`` and pin the port to it.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.processors.minicpmv4_6 import (
    MiniCPMV4_6ImageProcessor,
    MiniCPMVMediaProfile,
)
from sglang.srt.multimodal.processors.minicpmv4_7 import (
    MiniCPMV4_7MultimodalProcessor,
    _compute_canvas_single,
    _parse_mm_processor_kwargs,
    _sequential_positions,
    build_image_bounds,
    canvas_rope_delta,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Synthetic ids: the layout code only ever compares them for equality.
IM_START, IM_END, SLICE_START, SLICE_END, NEWLINE, PAD = 900, 901, 902, 903, 904, 905
SPECIAL_IDS = {
    "im_start_id": IM_START,
    "im_end_id": IM_END,
    "slice_start_id": SLICE_START,
    "slice_end_id": SLICE_END,
    "newline_id": NEWLINE,
}


def _canvas(input_ids, target_sizes, special_ids=SPECIAL_IDS):
    ids = torch.tensor(input_ids, dtype=torch.long)
    sizes = torch.tensor(target_sizes, dtype=torch.long)
    bounds = build_image_bounds(ids, special_ids)
    positions = _compute_canvas_single(
        ids, torch.arange(ids.shape[0]), bounds, sizes, special_ids
    )
    return positions, canvas_rope_delta(positions, ids.shape[0])


class TestMiniCPMV4_7CanvasMRoPE(CustomTestCase):
    def test_text_only_collapses_to_sequential_positions(self):
        positions, delta = _canvas([10, 11, 12, 13], [])

        self.assertEqual(positions.shape, (3, 4))
        self.assertEqual(positions[0].tolist(), [0, 1, 2, 3])
        self.assertEqual(positions[1].tolist(), positions[0].tolist())
        self.assertEqual(positions[2].tolist(), positions[0].tolist())
        self.assertEqual(delta, 0)

    def test_thumbnail_grid_places_rows_and_columns(self):
        # 16 visual tokens on a 4x4 grid: <image> halo sits one position below
        # the base, </image> one past the canvas.
        positions, delta = _canvas(
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
        # One past the last prompt position, so the first decoded token
        # continues from 9.
        self.assertEqual(delta, 8 + 1 - 22)

    def test_slice_is_placed_after_the_thumbnail(self):
        positions, delta = _canvas(
            [10, 11, IM_START]
            + [PAD] * 16
            + [IM_END]
            + [SLICE_START]
            + [PAD] * 16
            + [SLICE_END, 12, 13],
            [(4, 4), (4, 4)],
        )

        # The slice contributes no new canvas extent (1x1 slice grid), so it
        # reuses the thumbnail's coordinates rather than extending them.
        self.assertEqual(positions[0][19], 2)  # </image> halo
        self.assertEqual(positions[0][20], 2)  # <slice>
        self.assertEqual(positions[1][21:37].tolist(), positions[1][3:19].tolist())
        self.assertEqual(positions[2][21:37].tolist(), positions[2][3:19].tolist())
        self.assertEqual(delta, 8 + 1 - 40)

    def test_tightly_adjacent_frames_form_one_video_group(self):
        frame = [IM_START] + [PAD] * 16 + [IM_END]
        positions, _ = _canvas([10, 11] + frame * 3 + [12, 13], [(4, 4)] * 3)

        # Each frame advances its own base, so frame 2 starts past frame 1's
        # canvas instead of sharing its positions.
        first_frame = positions[0][3:19]
        second_frame = positions[0][20:36]
        self.assertTrue((second_frame > first_frame.max()).all())

    def test_positions_are_contiguous_per_row(self):
        positions, _ = _canvas(
            [10, 11, IM_START] + [PAD] * 16 + [IM_END, 12, 13], [(4, 4)]
        )

        for row in range(3):
            # Text keeps the plain 0,1,... sequence around the visual block.
            self.assertEqual(positions[row, 0].item(), 0)
            self.assertEqual(positions[row, -2].item(), 7)
            self.assertEqual(positions[row, -1].item(), 8)


class TestMiniCPMV4_7MultimodalProcessor(CustomTestCase):
    def _processor(self):
        processor = MiniCPMV4_7MultimodalProcessor.__new__(
            MiniCPMV4_7MultimodalProcessor
        )
        processor.uses_canvas_mrope = True
        processor._canvas_special_ids = dict(SPECIAL_IDS)
        return processor

    def test_target_sizes_follow_sequence_order(self):
        processor = self._processor()
        # The processor appends images and then videos, but the canvas layout
        # needs the grids in the order they appear in ``input_ids``.
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                offsets=[(30, 31)],
                model_specific_data={"tgt_size": [(4, 4)]},
            ),
            MultimodalDataItem(
                modality=Modality.VIDEO,
                offsets=[(5, 6)],
                model_specific_data={"tgt_size": [(2, 2)]},
            ),
        ]

        sizes = processor._target_sizes_in_sequence_order(items)

        self.assertEqual(sizes.tolist(), [[2, 2], [4, 4]])

    def test_missing_grids_fall_back_to_sequential_positions(self):
        processor = self._processor()
        input_ids = [10, 11, IM_START] + [PAD] * 16 + [IM_END, 12, 13]

        positions, delta = processor.compute_mrope_positions(input_ids, [])

        self.assertEqual(positions.shape, (3, len(input_ids)))
        self.assertEqual(positions[0].tolist(), list(range(len(input_ids))))
        self.assertEqual(delta.item(), 0)

    def test_bounds_and_grid_count_mismatch_falls_back(self):
        processor = self._processor()
        input_ids = [10, 11, IM_START] + [PAD] * 16 + [IM_END, 12, 13]
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                offsets=[(3, 18)],
                model_specific_data={"tgt_size": [(4, 4)]},
            ),
            MultimodalDataItem(
                modality=Modality.IMAGE,
                offsets=[(3, 18)],
                model_specific_data={"tgt_size": [(4, 4)]},
            ),
        ]

        positions, delta = processor.compute_mrope_positions(input_ids, items)

        self.assertEqual(
            positions.tolist(), _sequential_positions(len(input_ids)).tolist()
        )
        self.assertEqual(delta.item(), 0)

    def test_canvas_positions_are_returned_for_a_matching_request(self):
        processor = self._processor()
        input_ids = [10, 11, IM_START] + [PAD] * 16 + [IM_END, 12, 13]
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                offsets=[(3, 18)],
                model_specific_data={"tgt_size": [(4, 4)]},
            )
        ]

        positions, delta = processor.compute_mrope_positions(input_ids, items)

        expected, expected_delta = _canvas(input_ids, [(4, 4)])
        self.assertEqual(positions.tolist(), expected.tolist())
        self.assertEqual(delta.item(), expected_delta)


class TestPerRequestProcessorKwargs(CustomTestCase):
    def _processor(self):
        processor = MiniCPMV4_7MultimodalProcessor.__new__(
            MiniCPMV4_7MultimodalProcessor
        )
        processor.patch_size = 14
        processor.default_media_profile = MiniCPMVMediaProfile(
            image_processor=MiniCPMV4_6ImageProcessor(
                max_slice_nums=9, patch_size=14, downsample_mode="16x"
            ),
            downsample_mode="16x",
        )
        processor.image_processor = processor.default_media_profile.image_processor
        return processor

    def _request(self, mm_processor_kwargs):
        return SimpleNamespace(mm_processor_kwargs=mm_processor_kwargs)

    def test_hf_style_modality_keys_are_flattened(self):
        flat = _parse_mm_processor_kwargs(
            {"images_kwargs": {"download_sample_size": 1}, "downsample_mode": "4x"}
        )

        self.assertEqual(flat, {"download_sample_size": 1, "downsample_mode": "4x"})

    def test_non_mapping_kwargs_are_ignored(self):
        self.assertEqual(_parse_mm_processor_kwargs(None), {})
        self.assertEqual(_parse_mm_processor_kwargs("downsample_mode=4x"), {})

    def test_absent_kwargs_keep_the_default_profile(self):
        processor = self._processor()

        profile = processor._media_profile(self._request(None))

        self.assertIs(profile, processor.default_media_profile)

    def test_kwargs_build_a_request_scoped_profile(self):
        processor = self._processor()

        profile = processor._media_profile(
            self._request({"downsample_mode": "4x", "max_slice_nums": 4})
        )

        self.assertIsNot(profile, processor.default_media_profile)
        self.assertEqual(profile.downsample_mode, "4x")
        self.assertEqual(profile.pad_divisor, 4)
        self.assertEqual(profile.image_processor.max_slice_nums, 4)
        # The shared default profile must stay untouched: requests are
        # preprocessed concurrently.
        self.assertEqual(processor.default_media_profile.downsample_mode, "16x")
        self.assertEqual(processor.image_processor.max_slice_nums, 9)

    def test_invalid_values_fall_back_to_the_defaults(self):
        processor = self._processor()

        profile = processor._media_profile(
            self._request({"downsample_mode": "8x", "max_slice_nums": "many"})
        )

        self.assertIs(profile, processor.default_media_profile)

    def test_vit_merger_flag_follows_the_requested_downsample_mode(self):
        processor = self._processor()

        profile_16x = processor._media_profile(self._request({}))
        profile_4x = processor._media_profile(self._request({"downsample_mode": "4x"}))

        self.assertTrue(
            processor._item_specific_data((4, 4), profile_16x)["use_vit_merger"]
        )
        self.assertFalse(
            processor._item_specific_data((4, 4), profile_4x)["use_vit_merger"]
        )


if __name__ == "__main__":
    unittest.main()
