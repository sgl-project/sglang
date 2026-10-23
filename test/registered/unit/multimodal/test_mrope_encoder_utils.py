"""CPU coverage for mRoPE DP vision-encoder helpers."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.multimodal.mm_utils import (
    _pad_mrope_vision_embeddings_for_tp_gather,
    iter_frame_bounded_image_batches,
    run_dp_sharded_mrope_vision_model,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _RecordingGather:
    def __init__(self):
        self.input = None

    def all_gather(self, input_, dim):
        self.input = input_
        rank_zero_embeddings = torch.full_like(input_, 99)
        return torch.cat([rank_zero_embeddings, input_], dim=dim)


class _Rope2dVisionTower:
    device = torch.device("cpu")
    merge_kernel_size = (1, 1)
    config = SimpleNamespace(hidden_size=1)

    def __call__(self, pixel_values, grid_hw, max_seqlen):
        return pixel_values.reshape(-1, 1, 1)


class TestMropeVisionEncoderPadding(CustomTestCase):
    def test_frame_batches_preserve_temporal_items_and_owner_order(self):
        grids = [[3, 2, 2], [1, 2, 2], [2, 2, 2], [4, 2, 2]]
        groups = list(iter_frame_bounded_image_batches([3, 0, 2, 1], grids, 5))
        self.assertEqual(groups, [[3], [0, 2], [1]])
        with self.assertRaises(ValueError):
            list(iter_frame_bounded_image_batches([3], grids, 3))

    def test_local_encoder_can_own_preprocessing_with_tp_one(self):
        expected = torch.arange(8, dtype=torch.float32).reshape(8, 1, 1)

        def encode(indices):
            self.assertEqual(indices, [0, 1])
            return expected

        with get_parallel().override(attn_tp_size=1):
            actual = run_dp_sharded_mrope_vision_model(
                _Rope2dVisionTower(),
                None,
                [[4, 2, 2], [1, 2, 2]],
                rope_type="rope_2d",
                pool_temporal_dimension=True,
                pixel_values_device=torch.device("cpu"),
                pixel_values_dtype=torch.float32,
                encode_local_items=encode,
            )
        self.assertTrue(torch.equal(actual, expected))

    def test_local_encoder_receives_only_owned_items_before_gather(self):
        gather = _RecordingGather()

        def encode(indices):
            self.assertEqual(indices, [1])
            return torch.tensor([[[7.0]]])

        with get_parallel().override(
            attn_tp_size=2, attn_tp_rank=1, attn_tp_group=gather
        ):
            actual = run_dp_sharded_mrope_vision_model(
                _Rope2dVisionTower(),
                None,
                [[4, 2, 2], [1, 1, 1]],
                rope_type="rope_2d",
                pool_temporal_dimension=True,
                pixel_values_device=torch.device("cpu"),
                pixel_values_dtype=torch.float32,
                encode_local_items=encode,
            )
        expected = torch.tensor([[[99.0]]] * 4 + [[[7.0]]])
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(gather.input.shape, (4, 1, 1))

    def test_empty_rank_skips_local_encoding_and_receives_broadcast(self):
        class Group:
            def broadcast(self, tensor, src):
                assert src == 0
                tensor.fill_(99)

        def encode(indices):
            raise AssertionError("an empty rank must not preprocess or encode")

        with get_parallel().override(
            attn_tp_size=2, attn_tp_rank=1, attn_tp_group=Group()
        ):
            actual = run_dp_sharded_mrope_vision_model(
                _Rope2dVisionTower(),
                None,
                [[4, 2, 2]],
                rope_type="rope_2d",
                pool_temporal_dimension=True,
                pixel_values_device=torch.device("cpu"),
                pixel_values_dtype=torch.float32,
                encode_local_items=encode,
            )
        self.assertTrue(torch.equal(actual, torch.full((4, 1, 1), 99.0)))

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


if __name__ == "__main__":
    unittest.main()
