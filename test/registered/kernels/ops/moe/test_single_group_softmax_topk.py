"""With one expert group, the grouped softmax top-k must select the same experts
and weights as the ungrouped fused top-k that DeepseekV2MoE switches to on ROCm
for n_group == topk_group == 1 routers without a correction bias (Mistral-format
MoE checkpoints, DeepSeek-V2-Lite)."""

import unittest

import torch

from sglang.srt.layers.moe.topk import fused_topk, grouped_topk_gpu
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")

torch.manual_seed(0)


@unittest.skipUnless(is_hip(), "Requires AMD GPU")
class TestSingleGroupSoftmaxTopk(CustomTestCase):
    def _run_case(self, num_tokens: int, num_experts: int, topk: int) -> None:
        hidden = torch.empty(num_tokens, 64, device="cuda", dtype=torch.bfloat16)
        logits = (torch.randn(num_tokens, num_experts, device="cuda") * 2).to(
            torch.bfloat16
        )
        w_grouped, ids_grouped = grouped_topk_gpu(
            hidden_states=hidden,
            gating_output=logits,
            topk=topk,
            renormalize=True,
            num_expert_group=1,
            topk_group=1,
            scoring_func="softmax",
        )
        w_fused, ids_fused = fused_topk(
            hidden_states=hidden, gating_output=logits, topk=topk, renormalize=True
        )

        ids_grouped, order_grouped = ids_grouped.sort(dim=-1)
        ids_fused, order_fused = ids_fused.to(ids_grouped.dtype).sort(dim=-1)
        same_rows = (ids_grouped == ids_fused).all(dim=-1)
        # The grouped route rounds the softmax to bf16, so a near-tie can flip.
        self.assertGreater(same_rows.float().mean().item(), 0.995)
        torch.testing.assert_close(
            w_fused.gather(1, order_fused)[same_rows],
            w_grouped.float().gather(1, order_grouped)[same_rows],
            atol=2e-2,
            rtol=0,
        )

        ref = torch.softmax(logits.float(), dim=-1).topk(topk, dim=-1)
        ref_ids, ref_order = ref.indices.sort(dim=-1)
        ref_w = (ref.values / ref.values.sum(-1, keepdim=True)).gather(1, ref_order)
        ref_rows = (ref_ids == ids_fused).all(dim=-1)
        torch.testing.assert_close(
            w_fused.gather(1, order_fused)[ref_rows],
            ref_w[ref_rows],
            atol=1e-5,
            rtol=0,
        )

    def test_cases(self) -> None:
        for num_experts, topk in ((128, 4), (64, 6)):
            for num_tokens in (1, 4, 256, 16384):
                with self.subTest(
                    num_tokens=num_tokens, num_experts=num_experts, topk=topk
                ):
                    self._run_case(
                        num_tokens=num_tokens, num_experts=num_experts, topk=topk
                    )


if __name__ == "__main__":
    unittest.main()
