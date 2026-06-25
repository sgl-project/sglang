"""Unit tests for fused shared-expert weight scaling on the DeepEP layout.

These tests pin the contract of ``remap_topk_for_per_rank_shared_slots`` for
the fused shared expert's topk weight on the two paths this fix covers:

  * aiter (HIP) path: routed_scaling_factor is folded into the routed weights and
    forward_deepep skips the post-MoE multiply, so the shared weight must be 1.0
    for a net 1.0x contribution.
  * post-MoE scaling path (default): the whole MoE output is multiplied by
    routed_scaling_factor afterward, so the shared weight must be 1/rsf.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import TopKConfig
from sglang.test.test_utils import CustomTestCase


class TestFusedSharedExpertScaling(CustomTestCase):
    # 256 physical routed experts over ep_size=8 -> 32 local routed per rank.
    NUM_PHYSICAL_ROUTED = 256
    EP_SIZE = 8
    EP_RANK = 0
    ROUTED_SCALING_FACTOR = 2.5

    def _run_remap(self, *, use_aiter):
        # Layout: [routed, routed, routed, shared]; the trailing column is the
        # fused shared expert. The shared weight starts at a sentinel to prove
        # the function overwrites it.
        topk_ids = torch.tensor([[5, 40, 100, 999]], dtype=torch.int32)
        routed_weights = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)
        topk_weights = torch.tensor([[1.0, 0.5, 0.25, -123.0]], dtype=torch.float32)

        topk_config = TopKConfig(
            top_k=4,
            num_fused_shared_experts=1,
            routed_scaling_factor=self.ROUTED_SCALING_FACTOR,
        )

        with (
            patch.object(topk_module, "_use_aiter", use_aiter),
            patch.object(
                topk_module,
                "get_parallel",
                return_value=SimpleNamespace(
                    moe_ep_size=self.EP_SIZE, moe_ep_rank=self.EP_RANK
                ),
            ),
        ):
            _out_ids, out_weights = topk_module.remap_topk_for_per_rank_shared_slots(
                topk_ids.clone(),
                topk_weights.clone(),
                num_fused_shared_experts=1,
                num_physical_routed_experts=self.NUM_PHYSICAL_ROUTED,
                topk_config=topk_config,
            )

        # Routed weights must never be touched by the shared-weight override.
        self.assertTrue(torch.equal(out_weights[0, :-1], routed_weights))
        return out_weights[0, -1].item()

    def test_aiter_path_uses_unit_shared_weight(self):
        # routed_scaling_factor is folded into the routed weights and the
        # post-MoE multiply is skipped -> shared weight must be 1.0, NOT 1/rsf.
        shared_weight = self._run_remap(use_aiter=True)
        self.assertAlmostEqual(shared_weight, 1.0)

    def test_post_moe_scaling_path_compensates_with_inverse_rsf(self):
        # Default path: the whole MoE output is multiplied by rsf afterward, so
        # the shared weight must be 1/rsf to net out to 1.0.
        shared_weight = self._run_remap(use_aiter=False)
        self.assertAlmostEqual(shared_weight, 1.0 / self.ROUTED_SCALING_FACTOR)

    def test_shared_expert_ids_route_to_home_rank(self):
        # Sanity: the shared slot id is placed at this rank's interleaved
        # position (ep_rank * num_local_experts + num_local_routed).
        topk_ids = torch.tensor([[5, 40, 100, 999]], dtype=torch.int32)
        topk_weights = torch.tensor([[1.0, 0.5, 0.25, 0.0]], dtype=torch.float32)
        topk_config = TopKConfig(
            top_k=4,
            num_fused_shared_experts=1,
            routed_scaling_factor=self.ROUTED_SCALING_FACTOR,
        )
        with (
            patch.object(topk_module, "_use_aiter", True),
            patch.object(
                topk_module,
                "get_parallel",
                return_value=SimpleNamespace(
                    moe_ep_size=self.EP_SIZE, moe_ep_rank=self.EP_RANK
                ),
            ),
        ):
            out_ids, _ = topk_module.remap_topk_for_per_rank_shared_slots(
                topk_ids.clone(),
                topk_weights.clone(),
                num_fused_shared_experts=1,
                num_physical_routed_experts=self.NUM_PHYSICAL_ROUTED,
                topk_config=topk_config,
            )
        num_local_routed = self.NUM_PHYSICAL_ROUTED // self.EP_SIZE  # 32
        num_local_experts = num_local_routed + 1  # 33
        expected_shared_id = self.EP_RANK * num_local_experts + num_local_routed
        self.assertEqual(out_ids[0, -1].item(), expected_shared_id)


if __name__ == "__main__":
    unittest.main()
