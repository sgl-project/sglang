"""CPU coverage for mRoPE positions and DP vision-encoder helpers."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.multimodal.mm_utils import (
    _pad_mrope_vision_embeddings_for_tp_gather,
    run_dp_sharded_mrope_vision_model,
)
from sglang.srt.multimodal.mrope_rope_index import get_rope_index_ernie45
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _RecordingGather:
    def __init__(self):
        self.input = None

    def all_gather(self, input_, dim):
        self.input = input_
        rank_zero_embeddings = torch.full_like(input_, 99)
        return torch.cat([rank_zero_embeddings, input_], dim=dim)


class _Rope2dVisionTower:
    merge_kernel_size = (1, 1)
    config = SimpleNamespace(hidden_size=1)

    def __call__(self, pixel_values, grid_hw, max_seqlen):
        return pixel_values.reshape(-1, 1, 1)


class TestMropeVisionEncoderPadding(CustomTestCase):
    def test_padding_preserves_2d_embedding_prefix(self):
        embeddings = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        padded = _pad_mrope_vision_embeddings_for_tp_gather(embeddings, 5)

        self.assertEqual(padded.shape, (5, 4))
        self.assertTrue(torch.equal(padded[:3], embeddings))

    def test_padding_preserves_3d_embedding_prefix(self):
        embeddings = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

        padded = _pad_mrope_vision_embeddings_for_tp_gather(embeddings, 5)

        self.assertEqual(padded.shape, (5, 3, 4))
        self.assertTrue(torch.equal(padded[:2], embeddings))

    def test_empty_rank_gets_a_gatherable_shape(self):
        embeddings = torch.empty((0, 3, 4), dtype=torch.bfloat16)

        padded = _pad_mrope_vision_embeddings_for_tp_gather(embeddings, 5)

        self.assertEqual(padded.shape, (5, 3, 4))
        self.assertEqual(padded.dtype, torch.bfloat16)

    def test_already_full_embedding_is_not_copied(self):
        embeddings = torch.randn(5, 4)

        padded = _pad_mrope_vision_embeddings_for_tp_gather(embeddings, 5)

        self.assertIs(padded, embeddings)

    def test_dp_encoder_reconstructs_an_underfilled_rope2d_rank(self):
        gather = _RecordingGather()
        pixel_values = torch.tensor([[1], [2], [3], [4], [7]], dtype=torch.float32)

        with get_parallel().override(
            attn_tp_size=2,
            attn_tp_rank=1,
            attn_tp_group=gather,
        ):
            embeddings = run_dp_sharded_mrope_vision_model(
                _Rope2dVisionTower(),
                pixel_values,
                [[2, 2], [1, 1]],
                rope_type="rope_2d",
            )

        self.assertEqual(gather.input.shape, (4, 1, 1))
        self.assertTrue(torch.equal(gather.input[:1], torch.tensor([[[7.0]]])))
        self.assertTrue(torch.equal(embeddings[:4], torch.full((4, 1, 1), 99.0)))
        self.assertTrue(torch.equal(embeddings[4:], torch.tensor([[[7.0]]])))


class TestErnieVideoPositions(CustomTestCase):
    def test_rectangular_video_axes_and_following_text(self):
        # The first video advances farther in time, the second in width.
        # Both must start following text after the maximum of all three axes.
        expected = torch.tensor(
            [
                [
                    [0, 1, 2, 2, 3, 3, 4, 4, 5, 6],
                    [0, 1, 2, 2, 2, 3, 3, 3, 5, 6],
                ],
                [
                    [0, 1, 2, 3, 2, 3, 2, 3, 5, 6],
                    [0, 1, 2, 2, 2, 2, 2, 2, 5, 6],
                ],
                [
                    [0, 1, 2, 2, 2, 2, 2, 2, 5, 6],
                    [0, 1, 2, 3, 4, 2, 3, 4, 5, 6],
                ],
            ]
        )
        for temporal_conv_size in (1, 2, 3):
            for dtype in (torch.int32, torch.int64):
                with self.subTest(temporal_conv_size=temporal_conv_size, dtype=dtype):
                    config = SimpleNamespace(
                        im_patch_id=3,
                        video_start_token_id=4,
                        video_end_token_id=5,
                        spatial_conv_size=2,
                        temporal_conv_size=temporal_conv_size,
                    )
                    input_ids = torch.tensor(
                        [[1, 4] + [3] * 6 + [5, 2]] * 2, dtype=dtype
                    )
                    # Include a partial temporal group to pin floor division.
                    grids = torch.tensor(
                        [
                            [4 * temporal_conv_size - 1, 4, 2],
                            [3 * temporal_conv_size - 1, 2, 6],
                        ]
                    )
                    positions, deltas = get_rope_index_ernie45(
                        input_ids, config, None, grids
                    )
                    torch.testing.assert_close(positions, expected.to(dtype))
                    torch.testing.assert_close(deltas, torch.tensor([[-3], [-3]]))
                    self.assertEqual(positions.device, input_ids.device)

    def test_image_and_multiple_videos_keep_independent_offsets(self):
        config = SimpleNamespace(
            im_patch_id=3,
            video_start_token_id=4,
            video_end_token_id=5,
            spatial_conv_size=2,
            temporal_conv_size=2,
        )
        input_ids = torch.tensor([[1, 3, 3, 2, 4, 3, 3, 3, 3, 5, 2, 4, 3, 3, 3, 5]])
        positions, deltas = get_rope_index_ernie45(
            input_ids,
            config,
            torch.tensor([[1, 2, 4]]),
            torch.tensor([[4, 2, 4], [6, 2, 2]]),
        )
        expected = torch.tensor(
            [
                [0, 1, 1, 3, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13],
                [0, 1, 1, 3, 4, 5, 5, 5, 5, 7, 8, 9, 10, 10, 10, 13],
                [0, 1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 9, 10, 10, 10, 13],
            ]
        ).unsqueeze(1)
        torch.testing.assert_close(positions, expected)
        torch.testing.assert_close(deltas, torch.tensor([[-2]]))


if __name__ == "__main__":
    unittest.main()
