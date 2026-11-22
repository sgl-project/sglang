"""Regression tests for get_rope_index's per-item token membership scans."""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.rotary_embedding.mrope_rope_index import (
    get_rope_index,
    get_rope_index_qwen3_omni,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

IMAGE, VIDEO, VISION_START, VISION_END, TEXT = 11, 12, 13, 14, 7
AUDIO, AUDIO_START, AUDIO_END = 15, 16, 17


def _rope_index(input_ids, **kwargs):
    original_tolist = torch.Tensor.tolist
    calls = []

    class Recorder(list):
        def __contains__(self, item):
            calls.append(item)
            return super().__contains__(item)

    def tolist(tensor):
        values = original_tolist(tensor)
        return Recorder(values) if tensor.dim() == 1 else values

    with mock.patch.object(torch.Tensor, "tolist", tolist):
        out = get_rope_index(
            spatial_merge_size=2,
            image_token_id=IMAGE,
            video_token_id=VIDEO,
            vision_start_token_id=VISION_START,
            model_type="qwen3_vl",
            input_ids=input_ids,
            **kwargs,
        )
    return out, calls


class TestGetRopeIndexMembershipScans(CustomTestCase):
    def test_video_only_never_scans_for_image_tokens(self):
        frames, tokens_per_frame = 6, 4
        ids = [TEXT]
        for _ in range(frames):
            ids += [TEXT, VISION_START] + [VIDEO] * tokens_per_frame + [VISION_END]
        ids += [TEXT]
        (positions, delta), calls = _rope_index(
            torch.tensor([ids]),
            video_grid_thw=torch.tensor([[frames, 4, 4]]),
        )
        self.assertNotIn(IMAGE, calls)
        self.assertEqual(positions.shape, (3, 1, len(ids)))
        self.assertEqual(delta.shape, (1, 1))

    def test_image_only_never_scans_for_video_tokens(self):
        ids = [TEXT, VISION_START, IMAGE, IMAGE, IMAGE, IMAGE, VISION_END, TEXT]
        (positions, _), calls = _rope_index(
            torch.tensor([ids]),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
        )
        self.assertNotIn(VIDEO, calls)
        self.assertEqual(positions.shape, (3, 1, len(ids)))

    def test_mixed_items_keep_text_and_media_positions(self):
        ids = [
            TEXT,
            VISION_START,
            IMAGE,
            IMAGE,
            IMAGE,
            IMAGE,
            VISION_END,
            TEXT,
            VISION_START,
            VIDEO,
            VIDEO,
            VIDEO,
            VIDEO,
            VISION_END,
        ]
        (positions, delta), _ = _rope_index(
            torch.tensor([ids]),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            video_grid_thw=torch.tensor([[1, 4, 4]]),
        )
        self.assertEqual(
            positions[:, 0].tolist(),
            [
                [0, 1, 2, 2, 2, 2, 4, 5, 6, 7, 7, 7, 7, 9],
                [0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7, 8, 8, 9],
                [0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 7, 8, 9],
            ],
        )
        self.assertEqual(delta.tolist(), [[-4]])

    def test_qwen3_omni_vision_only_never_scans_for_audio_tokens(self):
        original_tolist = torch.Tensor.tolist
        calls = []

        class Recorder(list):
            def __contains__(self, item):
                calls.append(item)
                return super().__contains__(item)

        def tolist(tensor):
            values = original_tolist(tensor)
            return Recorder(values) if tensor.dim() == 1 else values

        ids = [TEXT, VISION_START, IMAGE, IMAGE, IMAGE, IMAGE, VISION_END, TEXT]
        with mock.patch.object(torch.Tensor, "tolist", tolist):
            positions, delta = get_rope_index_qwen3_omni(
                spatial_merge_size=2,
                image_token_id=IMAGE,
                video_token_id=VIDEO,
                vision_start_token_id=VISION_START,
                input_ids=torch.tensor([ids]),
                image_grid_thw=torch.tensor([[1, 4, 4]]),
                audio_token_id=AUDIO,
                audio_start_token_id=AUDIO_START,
                position_id_per_seconds=25,
            )
        self.assertNotIn(AUDIO, calls)
        self.assertEqual(positions.shape, (3, 1, len(ids)))
        self.assertEqual(delta.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
