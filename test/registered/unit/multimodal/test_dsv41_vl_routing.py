"""Unit tests for srt/multimodal/dsv41/vl_routing"""

import unittest
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.topk import TopKConfig
from sglang.srt.multimodal.dsv41.vl_routing import vision_topk
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_EXPERTS = 16
NUM_ROUTED_TOPK = 4
ROUTED_SCALING_FACTOR = 1.5


def _make_moe(
    num_fused_shared_experts: int,
    fused_shared_experts_scaling_factor: Optional[float] = None,
):
    return SimpleNamespace(
        gate=SimpleNamespace(
            e_score_correction_bias=torch.zeros(NUM_EXPERTS),
            e_score_correction_bias_vl=torch.zeros(NUM_EXPERTS),
        ),
        config=SimpleNamespace(image_token_id=0),
        topk=SimpleNamespace(
            topk_config=TopKConfig(
                top_k=NUM_ROUTED_TOPK + num_fused_shared_experts,
                renormalize=True,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=ROUTED_SCALING_FACTOR,
                apply_routed_scaling_factor_on_output=True,
                scoring_func="sqrtsoftplus",
                fused_shared_experts_scaling_factor=fused_shared_experts_scaling_factor,
            )
        ),
    )


class TestDsv41VisionTopK(CustomTestCase):
    @patch("sglang.srt.multimodal.dsv41.vl_routing.is_cuda", return_value=False)
    def test_fused_shared_expert_slot(self, _mock_is_cuda):
        torch.manual_seed(0)
        logits = torch.randn(8, NUM_EXPERTS)

        unfused = vision_topk(_make_moe(0), logits, None)
        fused = vision_topk(_make_moe(1), logits, None)

        self.assertEqual(fused.topk_ids.shape, (8, NUM_ROUTED_TOPK + 1))
        # The shared expert occupies the slot past the routed experts and, with
        # renormalization, contributes with weight 1.0.
        torch.testing.assert_close(
            fused.topk_ids[:, -1],
            torch.full((8,), NUM_EXPERTS, dtype=fused.topk_ids.dtype),
        )
        torch.testing.assert_close(fused.topk_weights[:, -1], torch.ones(8))
        # Routing of the non-shared slots is unchanged by fusion.
        torch.testing.assert_close(fused.topk_ids[:, :-1], unfused.topk_ids)
        torch.testing.assert_close(fused.topk_weights[:, :-1], unfused.topk_weights)

    @patch("sglang.srt.multimodal.dsv41.vl_routing.is_cuda", return_value=False)
    def test_fused_shared_expert_ep_scaling(self, _mock_is_cuda):
        torch.manual_seed(0)
        logits = torch.randn(8, NUM_EXPERTS)
        ep_size = 8

        fused = vision_topk(_make_moe(1), logits, None)
        scaled = vision_topk(_make_moe(1, 1 / ep_size), logits, None)

        # Standard EP replicates the shared expert per rank, so its weight is
        # divided by ep_size while the routed slots are untouched.
        torch.testing.assert_close(scaled.topk_ids, fused.topk_ids)
        torch.testing.assert_close(
            scaled.topk_weights[:, -1], fused.topk_weights[:, -1] / ep_size
        )
        torch.testing.assert_close(
            scaled.topk_weights[:, :-1], fused.topk_weights[:, :-1]
        )


if __name__ == "__main__":
    unittest.main()
