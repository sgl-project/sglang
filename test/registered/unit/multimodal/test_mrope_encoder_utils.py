"""CPU coverage for mRoPE DP vision-encoder helpers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.multimodal.mm_utils import (
    _pad_mrope_vision_embeddings_for_tp_gather,
    _should_run_mrope_local_postprocess,
    _should_use_mrope_all_gatherv,
    run_dp_sharded_mrope_vision_model,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingGather:
    def __init__(self):
        self.input = None

    def all_gather(self, input_, dim):
        self.input = input_
        rank_zero_embeddings = torch.full_like(input_, 99)
        return torch.cat([rank_zero_embeddings, input_], dim=dim)


class _RecordingAllGatherv:
    def __init__(self):
        self.input = None
        self.sizes = None

    def all_gatherv(self, input_, sizes, output):
        self.input = input_
        self.sizes = sizes
        output[: sizes[0]].fill_(99)
        output[sizes[0] :].copy_(input_)


class _RecordingBroadcast:
    def __init__(self):
        self.input = None

    def broadcast(self, input_, src):
        self.input = input_
        self.src = src
        input_.fill_(7)


class _AvailablePyNccl:
    available = True
    disabled = True


class _Rope2dVisionTower:
    merge_kernel_size = (1, 1)
    config = SimpleNamespace(hidden_size=1)
    device = torch.device("cpu")

    def __call__(self, pixel_values, grid_hw, max_seqlen):
        return pixel_values.reshape(-1, 1, 1)


class TestMropeVisionEncoderPadding(CustomTestCase):
    def test_all_gatherv_capability_does_not_depend_on_eager_state(self):
        from sglang.srt.multimodal.mm_utils import _can_use_mrope_all_gatherv

        group = SimpleNamespace(pynccl_comm=_AvailablePyNccl())
        tensor = SimpleNamespace(is_cuda=True)

        self.assertTrue(_can_use_mrope_all_gatherv(tensor, group))

    def test_variable_gather_policy_avoids_modest_imbalance(self):
        group = SimpleNamespace(pynccl_comm=_AvailablePyNccl())
        tensor = SimpleNamespace(is_cuda=True)

        self.assertTrue(_should_use_mrope_all_gatherv(tensor, group, [4] * 8))
        self.assertFalse(
            _should_use_mrope_all_gatherv(
                tensor, group, [1024, 768, 512, 384, 256, 128, 64, 32]
            )
        )
        self.assertTrue(
            _should_use_mrope_all_gatherv(tensor, group, [1024, 64, 32, 16, 8, 4, 2, 1])
        )

    def test_local_projector_policy_requires_balance_or_large_payload(self):
        tensor = torch.empty(0, 4, 1024, dtype=torch.bfloat16)

        self.assertTrue(_should_run_mrope_local_postprocess(tensor, [384] * 8))
        self.assertFalse(
            _should_run_mrope_local_postprocess(
                tensor, [1024, 768, 512, 384, 256, 128, 64, 32]
            )
        )
        self.assertTrue(
            _should_run_mrope_local_postprocess(
                tensor, [4096, 3072, 2048, 1536, 1024, 512, 256, 128]
            )
        )
        self.assertFalse(
            _should_run_mrope_local_postprocess(
                tensor, [4096, 256, 128, 64, 32, 16, 8, 4]
            )
        )

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

    def test_single_image_projects_after_raw_broadcast(self):
        broadcast = _RecordingBroadcast()
        pixel_values = torch.tensor([[1]], dtype=torch.float32)

        with get_parallel().override(
            attn_tp_size=2,
            attn_tp_rank=1,
            attn_tp_group=broadcast,
        ):
            embeddings = run_dp_sharded_mrope_vision_model(
                _Rope2dVisionTower(),
                pixel_values,
                [[1, 1]],
                rope_type="rope_2d",
                local_postprocess=lambda value: value.reshape(value.shape[0], -1) * 10,
            )

        self.assertEqual(broadcast.src, 0)
        self.assertEqual(broadcast.input.shape, (1, 1, 1))
        self.assertTrue(torch.equal(embeddings, torch.tensor([[70.0]])))

    def test_dp_encoder_projects_on_owner_before_variable_gather(self):
        gather = _RecordingAllGatherv()
        pixel_values = torch.tensor([[1], [2], [3], [4], [7]], dtype=torch.float32)

        with get_parallel().override(
            attn_tp_size=2,
            attn_tp_rank=1,
            attn_tp_group=gather,
        ), patch(
            "sglang.srt.multimodal.mm_utils._can_use_mrope_all_gatherv",
            return_value=True,
        ), patch(
            "sglang.srt.multimodal.mm_utils._should_run_mrope_local_postprocess",
            return_value=True,
        ):
            embeddings = run_dp_sharded_mrope_vision_model(
                _Rope2dVisionTower(),
                pixel_values,
                [[2, 2], [1, 1]],
                rope_type="rope_2d",
                local_postprocess=lambda value: value.reshape(value.shape[0], -1) * 10,
            )

        self.assertEqual(gather.sizes, [4, 1])
        self.assertEqual(gather.input.shape, (1, 1))
        self.assertTrue(torch.equal(gather.input, torch.tensor([[70.0]])))
        self.assertTrue(torch.equal(embeddings[:4], torch.full((4, 1), 99.0)))
        self.assertTrue(torch.equal(embeddings[4:], torch.tensor([[70.0]])))


if __name__ == "__main__":
    unittest.main()
