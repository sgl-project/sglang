import unittest

import torch
from sgl_kernel.common_ops import biased_grouped_topk_cpu as biased_grouped_topk
from sgl_kernel.common_ops import grouped_topk_cpu as grouped_topk

from sglang.test.test_utils import CustomTestCase


# This is used by the Deepseek-V2 model
class TestGroupedTopK(CustomTestCase):
    def _grouped_topk_native(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        num_expert_group: int = 0,
        topk_group: int = 0,
    ):

        assert (
            hidden_states.shape[0] == gating_output.shape[0]
        ), "Number of tokens mismatch"

        scores = torch.softmax(gating_output, dim=-1)
        num_token = scores.shape[0]
        group_scores = (
            scores.view(num_token, num_expert_group, -1).max(dim=-1).values
        )  # [n, n_group]
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
            1
        ]  # [n, top_k_group]

        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
            .reshape(num_token, -1)
        )  # [n, e]

        tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
        topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)

        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

    def _run_single_test(self, M, E, G, topk, topk_group, renormalize, dtype):

        # expand gating_output by M, otherwise bfloat16 fall into same value aftering truncating
        hidden_states = torch.randn(M, 100, dtype=dtype)
        gating_output = torch.randn(M, E, dtype=dtype) * 2 * M

        ref_topk_weights, ref_topk_ids = self._grouped_topk_native(
            hidden_states.float(),
            gating_output.float(),
            topk,
            renormalize,
            G,
            topk_group,
        )

        # fused version
        topk_weights, topk_ids = grouped_topk(
            hidden_states, gating_output, topk, renormalize, G, topk_group
        )

        res = torch.zeros(M, E, dtype=torch.float)
        ref = torch.zeros(M, E, dtype=torch.float)
        res.scatter_(1, topk_ids.long(), topk_weights)
        ref.scatter_(1, ref_topk_ids.long(), ref_topk_weights)
        torch.testing.assert_close(res, ref)

    def test_grouped_topk(self):
        for renormalize in [True, False]:
            self._run_single_test(123, 8, 2, 2, 1, renormalize, torch.bfloat16)
            self._run_single_test(123, 16, 4, 3, 2, renormalize, torch.bfloat16)
            self._run_single_test(123, 32, 4, 3, 2, renormalize, torch.bfloat16)
            self._run_single_test(1123, 32, 4, 3, 2, renormalize, torch.bfloat16)
            self._run_single_test(123, 64, 1, 6, 1, renormalize, torch.bfloat16)
            self._run_single_test(123, 256, 8, 4, 8, renormalize, torch.bfloat16)
            self._run_single_test(123, 160, 8, 6, 2, renormalize, torch.bfloat16)


# DeepSeek V2/V3/R1 uses biased_grouped_top
class TestBiasedGroupedTopK(CustomTestCase):
    def _biased_grouped_topk(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        correction_bias: torch.Tensor,
        topk: int,
        renormalize: bool,
        num_expert_group: int = 0,
        topk_group: int = 0,
    ):
        assert (
            hidden_states.shape[0] == gating_output.shape[0]
        ), "Number of tokens mismatch"

        scores = gating_output.sigmoid()
        num_token = scores.shape[0]
        scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(num_token, num_expert_group, -1)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )  # [n, n_group]
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
            1
        ]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
            .reshape(num_token, -1)
        )  # [n, e]
        tmp_scores = scores_for_choice.masked_fill(
            ~score_mask.bool(), float("-inf")
        )  # [n, e]
        _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
        topk_weights = scores.gather(1, topk_ids)

        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

    def _run_single_test(self, M, E, G, topk, topk_group, renormalize, dtype):

        # expand gating_output by M, otherwise bfloat16 fall into same value aftering truncating
        hidden_states = torch.randn(M, 100, dtype=dtype)
        gating_output = torch.randn(M, E, dtype=dtype) * 2 * M
        correction_bias = torch.randn(E, dtype=dtype)

        ref_topk_weights, ref_topk_ids = self._biased_grouped_topk(
            hidden_states.float(),
            gating_output.float(),
            correction_bias.float(),
            topk,
            renormalize,
            G,
            topk_group,
        )

        # fused version
        topk_weights, topk_ids = biased_grouped_topk(
            hidden_states,
            gating_output,
            correction_bias,
            topk,
            renormalize,
            G,
            topk_group,
        )

        res = torch.zeros(M, E, dtype=torch.float)
        ref = torch.zeros(M, E, dtype=torch.float)
        res.scatter_(1, topk_ids.long(), topk_weights)
        ref.scatter_(1, ref_topk_ids.long(), ref_topk_weights)
        torch.testing.assert_close(res, ref)

    def test_biased_grouped_topk(self):
        for renormalize in [True, False]:
            self._run_single_test(122, 256, 8, 8, 2, renormalize, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
