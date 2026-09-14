"""Regression test: on the aiter (HIP) path the fused shared expert must be
appended exactly once, and the gate must still return all K routed experts.

MiniMax-M3 (sigmoid + correction bias, top-4 routed + 1 fused shared expert,
routed_scaling_factor 2.0) routes through ``biased_topk_jit_kernel_impl`` ->
``moe_fused_gate``. That gate treats ``topk`` as the total width including the
shared slot, while ``_post_process_topk_ids`` appends the shared expert itself
whenever ``_use_aiter`` is set. Passing ``num_fused_shared_experts`` to the gate
as well produced rows of [3 routed, shared, shared]: one routed expert lost and
the shared expert counted twice (GSM8K-500 0.81 fused vs 0.88 unfused on
MI350X). The gate must be asked for K_routed plain routed experts on every aiter
path, exactly as PR #36515 already did for the per-rank shared-slot path.
"""

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.test.test_utils import CustomTestCase


class TestAiterFusedSharedTopK(CustomTestCase):
    NUM_EXPERTS = 128
    TOPK_ROUTED = 4
    ROUTED_SCALING_FACTOR = 2.0

    @unittest.skipUnless(torch.cuda.is_available(), "GPU required (Triton gate)")
    def test_aiter_path_keeps_all_routed_and_one_shared(self):
        torch.manual_seed(0)
        num_tokens = 64
        logits = torch.randn(num_tokens, self.NUM_EXPERTS, device="cuda").float()
        bias = (torch.randn(self.NUM_EXPERTS, device="cuda") * 0.1).float()
        hidden = torch.zeros(num_tokens, 16, device="cuda", dtype=torch.bfloat16)

        # HF MiniMax-M3 semantics: sigmoid scores, top-k on score + bias,
        # renormalize the raw scores of the winners, then routed_scaling_factor.
        scores = logits.sigmoid()
        ref_ids = torch.topk(scores + bias, self.TOPK_ROUTED, dim=-1).indices
        ref_w = scores.gather(1, ref_ids)
        ref_w = ref_w / ref_w.sum(-1, keepdim=True) * self.ROUTED_SCALING_FACTOR

        cfg = TopKConfig(
            top_k=self.TOPK_ROUTED + 1,
            renormalize=True,
            scoring_func="sigmoid",
            correction_bias=bias,
            num_fused_shared_experts=1,
            routed_scaling_factor=self.ROUTED_SCALING_FACTOR,
            apply_routed_scaling_factor_on_output=True,
            allow_routed_experts_capture=False,
        )
        with (
            patch.object(topk_module, "_use_aiter", True),
            patch.object(topk_module, "_is_cuda", False),
            patch.object(topk_module, "_is_hip", True),
        ):
            out = select_experts(
                hidden_states=hidden, router_logits=logits, topk_config=cfg, layer_id=0
            )
        ids, weights = out.topk_ids, out.topk_weights
        self.assertEqual(tuple(ids.shape), (num_tokens, self.TOPK_ROUTED + 1))

        routed_ids, shared_ids = ids[:, : self.TOPK_ROUTED], ids[:, self.TOPK_ROUTED]
        # Exactly one shared column, at id num_experts, weight 1.0.
        self.assertTrue(bool((shared_ids == self.NUM_EXPERTS).all()))
        self.assertTrue(bool((routed_ids < self.NUM_EXPERTS).all()))
        torch.testing.assert_close(
            weights[:, self.TOPK_ROUTED], torch.ones_like(weights[:, self.TOPK_ROUTED])
        )
        # All K routed experts survive, matching the HF top-k set and weights.
        self.assertTrue(
            torch.equal(
                torch.sort(routed_ids, dim=1).values,
                torch.sort(ref_ids.to(routed_ids.dtype), dim=1).values,
            )
        )
        order = torch.argsort(routed_ids, dim=1)
        ref_order = torch.argsort(ref_ids, dim=1)
        torch.testing.assert_close(
            weights[:, : self.TOPK_ROUTED].gather(1, order),
            ref_w.gather(1, ref_order),
            rtol=1e-4,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
