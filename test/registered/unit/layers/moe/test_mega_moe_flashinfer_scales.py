"""Unit tests for the ModelOpt NVFP4 -> FlashInfer MegaMoE epilogue scale mapping."""

import unittest

import torch

from sglang.srt.layers.moe.mega_moe_flashinfer import (
    compute_flashinfer_mega_moe_scales,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFlashInferMegaMoeScales(CustomTestCase):
    def test_modelopt_scales_map_to_norm_const_and_alpha(self):
        # Two EP ranks x two local experts; the fc2 input scale is stored for
        # all four global experts and sliced to this rank.
        scales = compute_flashinfer_mega_moe_scales(
            w13_input_scale=torch.tensor([0.5, 0.25]),
            w13_weight_scale_2=torch.tensor([[2.0, 2.0], [4.0, 4.0]]),
            w2_input_scale=torch.tensor([8.0, 8.0, 0.5, 0.25]),
            w2_weight_scale_2=torch.tensor([3.0, 5.0]),
            ep_rank=1,
            num_local_experts=2,
        )
        # fc1 activations share one scale: the largest input scale (TRT-LLM /
        # CuteDSL convention), so alpha = that scale x weight scale_2.
        self.assertAlmostEqual(scales.input_norm_const, 2.0)
        torch.testing.assert_close(scales.fc1_alpha, torch.tensor([1.0, 2.0]))
        # fc2 keeps the checkpoint's per-expert static input scale.
        torch.testing.assert_close(scales.fc1_norm_const, torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(scales.fc2_alpha, torch.tensor([1.5, 1.25]))
        self.assertEqual(scales.fc1_alpha.dtype, torch.float32)

    def test_shared_gate_up_scale_vector(self):
        scales = compute_flashinfer_mega_moe_scales(
            w13_input_scale=torch.tensor([1.0]),
            w13_weight_scale_2=torch.tensor([2.0, 4.0]),
            w2_input_scale=torch.tensor([1.0, 1.0]),
            w2_weight_scale_2=torch.tensor([1.0, 1.0]),
            ep_rank=0,
            num_local_experts=2,
        )
        torch.testing.assert_close(scales.fc1_alpha, torch.tensor([2.0, 4.0]))

    def test_distinct_gate_and_up_scales_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "gate and up"):
            compute_flashinfer_mega_moe_scales(
                w13_input_scale=torch.tensor([1.0]),
                w13_weight_scale_2=torch.tensor([[2.0, 3.0]]),
                w2_input_scale=torch.tensor([1.0]),
                w2_weight_scale_2=torch.tensor([1.0]),
                ep_rank=0,
                num_local_experts=1,
            )


if __name__ == "__main__":
    unittest.main()
