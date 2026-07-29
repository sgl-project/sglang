import unittest
from typing import Optional

import torch

from sglang.srt.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.layers.moe.topk import (
    _mask_topk_ids_padded_region,
)
from sglang.srt.layers.moe.topk import (
    biased_grouped_topk_impl as native_biased_grouped_topk,
)
from sglang.srt.layers.moe.topk import biased_topk_impl as native_biased_topk
from sglang.srt.layers.moe.topk import fused_topk_torch_native as native_fused_topk
from sglang.srt.layers.moe.topk import grouped_topk_gpu as native_grouped_topk
from sglang.srt.models.llama4 import Llama4MoE
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-b-test-cpu")
register_cpu_ci(est_time=10, suite="base-b-test-cpu-arm64")


def native_biased_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    if scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    elif scoring_func == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(gating_output).sqrt()

    num_token = scores.shape[0]
    num_experts = scores.shape[1]

    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    _, topk_ids = torch.topk(
        scores_for_choice,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )
    topk_weights = scores.gather(1, topk_ids)

    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        if routed_scaling_factor is not None:
            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor
            )

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids


# This is used by the Deepseek-V2 model
class TestGroupedTopK(CustomTestCase):
    def _run_single_test(self, M, E, G, topk, topk_group, renormalize, dtype):
        torch.manual_seed(12)

        # expand gating_output by M, otherwise bfloat16 fall into same value aftering truncating
        hidden_states = torch.randn(M, 100, dtype=dtype)
        gating_output = torch.randn(M, E, dtype=dtype) * 2 * M

        ref_topk_weights, ref_topk_ids = native_grouped_topk(
            hidden_states.float(),
            gating_output.float(),
            topk,
            renormalize,
            G,
            topk_group,
        )

        # fused version
        topk_weights, topk_ids = torch.ops.sgl_kernel.grouped_topk_cpu(
            hidden_states,
            gating_output,
            topk,
            renormalize,
            G,
            topk_group,
            0,
            None,
            None,
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
    def _run_single_test(
        self,
        M,
        E,
        G,
        topk,
        topk_group,
        renormalize,
        gating_dtype,
        bias_dtype,
        routed_scaling_factor,
    ):
        torch.manual_seed(1024)

        # expand gating_output by M, otherwise bfloat16 fall into same value aftering truncating
        hidden_states = torch.randn(M, 100, dtype=torch.bfloat16)
        gating_output = torch.randn(M, E, dtype=gating_dtype) * 2 * M
        correction_bias = torch.randn(E, dtype=bias_dtype)

        ref_topk_weights, ref_topk_ids = native_biased_grouped_topk(
            hidden_states.float(),
            gating_output.float(),
            correction_bias.float(),
            topk,
            renormalize,
            G,
            topk_group,
        )
        ref_topk_weights = (
            ref_topk_weights * routed_scaling_factor
            if routed_scaling_factor is not None
            else ref_topk_weights
        )
        # fused version
        topk_weights, topk_ids = torch.ops.sgl_kernel.biased_grouped_topk_cpu(
            hidden_states,
            gating_output,
            correction_bias,
            topk,
            renormalize,
            G,
            topk_group,
            0,
            routed_scaling_factor,
            None,
        )

        res = torch.zeros(M, E, dtype=torch.float)
        ref = torch.zeros(M, E, dtype=torch.float)
        res.scatter_(1, topk_ids.long(), topk_weights)
        ref.scatter_(1, ref_topk_ids.long(), ref_topk_weights)
        torch.testing.assert_close(res, ref)

    def test_biased_grouped_topk(self):
        for renormalize in [False]:
            for bias_dtype in [torch.float32, torch.bfloat16]:
                for gating_dtype in [torch.float32, torch.bfloat16]:
                    for routed_scaling_factor in [None, 1.125]:
                        for E_num in [128, 192, 256, 384]:
                            self._run_single_test(
                                34,
                                E_num,
                                8,
                                8,
                                2,
                                renormalize,
                                gating_dtype,
                                bias_dtype,
                                routed_scaling_factor,
                            )


class TestBiasedTopK(CustomTestCase):
    def test_biased_topk_returns_logical_ids_with_eplb_info(self):
        hidden_states = torch.ones(1, 4)
        gating_output = torch.tensor([[10.0, 9.0, 1.0, 0.0]])
        correction_bias = torch.zeros(4)
        dispatch_info = ExpertLocationDispatchInfo(
            ep_dispatch_algorithm="static",
            partial_logical_to_rank_dispatch_physical_map=torch.tensor(
                [2, 3, 0, 1], dtype=torch.int64
            ),
            partial_logical_to_all_physical_map=torch.tensor(
                [[2], [3], [0], [1]], dtype=torch.int64
            ),
            partial_logical_to_all_physical_map_num_valid=torch.ones(
                4, dtype=torch.int64
            ),
            num_physical_experts=4,
        )

        _, topk_ids = native_biased_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            correction_bias=correction_bias,
            topk=2,
            renormalize=False,
            scoring_func="sqrtsoftplus",
            expert_location_dispatch_info=dispatch_info,
        )

        torch.testing.assert_close(topk_ids, torch.tensor([[0, 1]], dtype=torch.int32))


class TestTopK(CustomTestCase):
    def _run_single_test(self, M, E, topk, renormalize, dtype):
        torch.manual_seed(1998)

        # expand gating_output by M, otherwise bfloat16 fall into same value aftering truncating
        hidden_states = torch.randn(M, 100, dtype=dtype)
        gating_output = torch.randn(M, E, dtype=dtype) * 2 * M

        ref_topk_weights, ref_topk_ids = native_fused_topk(
            hidden_states.float(),
            gating_output.float(),
            topk,
            renormalize,
        )

        # fused version
        topk_weights, topk_ids = torch.ops.sgl_kernel.topk_softmax_cpu(
            hidden_states, gating_output, topk, renormalize
        )

        res = torch.zeros(M, E, dtype=torch.float)
        ref = torch.zeros(M, E, dtype=torch.float)
        res.scatter_(1, topk_ids.long(), topk_weights)
        ref.scatter_(1, ref_topk_ids.long(), ref_topk_weights)
        torch.testing.assert_close(res, ref)

    def test_topk(self):
        for renormalize in [True, False]:
            self._run_single_test(123, 8, 2, renormalize, torch.bfloat16)
            self._run_single_test(123, 16, 3, renormalize, torch.bfloat16)
            self._run_single_test(123, 32, 3, renormalize, torch.bfloat16)
            self._run_single_test(123, 32, 3, renormalize, torch.bfloat16)
            self._run_single_test(123, 64, 6, renormalize, torch.bfloat16)
            self._run_single_test(123, 256, 4, renormalize, torch.bfloat16)
            self._run_single_test(123, 160, 6, renormalize, torch.bfloat16)


class TestCustomTopK(CustomTestCase):
    def _run_single_test(
        self, M, E, topk, renormalize, dtype, native_custom_f, fused_custom_f
    ):
        torch.manual_seed(16)

        # expand gating_output by M, otherwise bfloat16 fall into same value aftering truncating
        hidden_states = torch.randn(M, 100, dtype=dtype)
        gating_output = torch.randn(M, E, dtype=dtype) * 2 * M

        ref_topk_weights, ref_topk_ids = native_custom_f(
            hidden_states.float(),
            gating_output.float(),
            topk,
            renormalize,
        )

        # fused version
        topk_weights, topk_ids = fused_custom_f(
            hidden_states, gating_output, topk, renormalize
        )

        res = torch.zeros(M, E, dtype=torch.float)
        ref = torch.zeros(M, E, dtype=torch.float)
        res.scatter_(1, topk_ids.long(), topk_weights)
        ref.scatter_(1, ref_topk_ids.long(), ref_topk_weights)
        torch.testing.assert_close(res, ref)

    def test_custom_topk(self):
        test_custom_functions = [
            (Llama4MoE.custom_routing_function, torch.ops.sgl_kernel.topk_sigmoid_cpu)
        ]
        for native_custom_f, fused_custom_f in test_custom_functions:
            self._run_single_test(
                123, 8, 1, False, torch.bfloat16, native_custom_f, fused_custom_f
            )
            self._run_single_test(
                123, 16, 1, False, torch.bfloat16, native_custom_f, fused_custom_f
            )
            self._run_single_test(
                123, 32, 1, False, torch.bfloat16, native_custom_f, fused_custom_f
            )


# biased topk (flat, non-grouped) for DeepSeek V4
class TestBiasedTopK(CustomTestCase):
    def _run_single_test(
        self,
        M,
        E,
        topk,
        renormalize,
        gating_dtype,
        bias_dtype,
        scoring_func,
        num_fused_shared_experts=0,
        routed_scaling_factor=None,
    ):
        torch.manual_seed(2024)

        hidden_states = torch.randn(M, 100, dtype=torch.bfloat16)
        gating_output = torch.randn(M, E, dtype=gating_dtype) * 2 * M
        correction_bias = torch.randn(E, dtype=bias_dtype)

        ref_topk_weights, ref_topk_ids = native_biased_topk(
            hidden_states.float(),
            gating_output.float(),
            correction_bias.float(),
            topk,
            renormalize,
            scoring_func=scoring_func,
            num_fused_shared_experts=0,
            routed_scaling_factor=None,
        )

        # fused version - use float32 inputs for consistent comparison
        gating_float = gating_output.float()
        bias_float = correction_bias.float()
        topk_weights, topk_ids = torch.ops.sgl_kernel.biased_topk_cpu(
            hidden_states,
            gating_float,
            bias_float,
            topk,
            renormalize,
            scoring_func,
            0,  # num_fused_shared_experts
            None,  # routed_scaling_factor
            False,  # apply_routed_scaling_factor_on_output
        )

        # Ties in biased scores can cause different expert selections between
        # torch.topk and std::partial_sort, both are valid.
        # We verify correctness by checking that all selected experts have biased
        # scores >= all unselected experts (within tolerance).
        gating_float = gating_output.float()
        bias_float = correction_bias.float()
        if scoring_func == "sigmoid":
            scores = gating_float.sigmoid()
        else:
            scores = torch.nn.functional.softplus(gating_float).sqrt()
        scores_biased = scores + bias_float.unsqueeze(0)

        for i in range(M):
            selected = topk_ids[i].long()
            min_selected = scores_biased[i].gather(0, selected).min().item()
            mask = torch.ones(E, dtype=torch.bool)
            mask[selected] = False
            if mask.any():
                max_unselected = scores_biased[i][mask].max().item()
                self.assertLessEqual(max_unselected, min_selected + 1e-5)

        # For rows where the same experts are selected, weights should match
        for i in range(M):
            ref_set = set(ref_topk_ids[i].tolist())
            fused_set = set(topk_ids[i].tolist())
            if ref_set == fused_set:
                # Same experts selected, compare weights via scatter
                res_row = torch.zeros(E, dtype=torch.float)
                ref_row = torch.zeros(E, dtype=torch.float)
                res_row.scatter_(0, topk_ids[i].long(), topk_weights[i])
                ref_row.scatter_(0, ref_topk_ids[i].long(), ref_topk_weights[i])
                torch.testing.assert_close(res_row, ref_row, atol=1e-5, rtol=1e-4)

    def test_biased_topk_sigmoid(self):
        for renormalize in [True, False]:
            for bias_dtype in [torch.float32, torch.bfloat16]:
                for gating_dtype in [torch.float32, torch.bfloat16]:
                    for E in [64, 128, 256]:
                        self._run_single_test(
                            34, E, 8, renormalize, gating_dtype, bias_dtype, "sigmoid"
                        )

    def test_biased_topk_sqrtsoftplus(self):
        for renormalize in [True, False]:
            for bias_dtype in [torch.float32, torch.bfloat16]:
                for gating_dtype in [torch.float32, torch.bfloat16]:
                    for E in [64, 128, 256]:
                        self._run_single_test(
                            34,
                            E,
                            8,
                            renormalize,
                            gating_dtype,
                            bias_dtype,
                            "sqrtsoftplus",
                        )


def hash_topk_native(
    gating_output,
    tid2eid,
    topk,
    scoring_func,
    num_fused_shared_experts,
    routed_scaling_factor,
):
    M = gating_output.size(0)
    g = gating_output.float()
    if scoring_func == "softmax":
        scores = g.softmax(dim=-1)
    elif scoring_func == "sigmoid":
        scores = g.sigmoid()
    else:
        scores = torch.nn.functional.softplus(g).sqrt()

    topk_ids = torch.zeros(M, topk, dtype=torch.int32)
    topk_weights = torch.zeros(M, topk, dtype=torch.float32)

    if num_fused_shared_experts == 1:
        topk_ids[:, :-1] = tid2eid
        topk_weights[:, :-1] = scores.gather(1, tid2eid.long())
        if scoring_func != "softmax":
            topk_weights[:, :-1] /= topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        # shared expert weight = sum of routed weights / scaling_factor
        topk_weights[:, -1] = topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor
    else:
        topk_ids[:, :] = tid2eid
        topk_weights[:, :] = scores.gather(1, tid2eid.long())
        if scoring_func != "softmax":
            topk_weights[:, :] /= topk_weights[:, :].sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


# HashTopK for DeepSeek V4 (expert IDs from precomputed lookup table)
class TestHashTopK(CustomTestCase):
    def _run_single_test(
        self,
        M,
        E,
        topk,
        scoring_func,
        num_fused_shared_experts,
        routed_scaling_factor,
        gating_dtype,
    ):
        torch.manual_seed(2026)

        routed_topk = topk - num_fused_shared_experts
        # Simulate tid2eid: random expert assignments for each token
        tid2eid = torch.randint(0, E, (M, routed_topk), dtype=torch.int32)
        gating_output = torch.randn(M, E, dtype=gating_dtype) * 2 * M

        ref_topk_weights, ref_topk_ids = hash_topk_native(
            gating_output,
            tid2eid,
            topk,
            scoring_func,
            num_fused_shared_experts,
            routed_scaling_factor,
        )

        # Fused kernel
        fused_w, fused_ids = torch.ops.sgl_kernel.hash_topk_cpu(
            gating_output,
            tid2eid,
            topk,
            scoring_func,
            num_fused_shared_experts,
            E,
            routed_scaling_factor,
        )

        # Compare routed expert IDs (all slots for no-fused, or :-1 for fused)
        if num_fused_shared_experts == 1:
            torch.testing.assert_close(
                fused_ids[:, :-1], ref_topk_ids[:, :-1], atol=0, rtol=0
            )
            # Shared expert ID should be in [E, E + num_fused_shared_experts)
            self.assertTrue((fused_ids[:, -1] >= E).all())
            self.assertTrue((fused_ids[:, -1] < E + num_fused_shared_experts).all())
            # Compare routed weights (use allclose to avoid NaN relative diff from underflow)
            self.assertTrue(
                torch.allclose(
                    fused_w[:, :-1], ref_topk_weights[:, :-1], atol=2e-4, rtol=0
                ),
                f"routed weights mismatch: max abs diff = {(fused_w[:, :-1] - ref_topk_weights[:, :-1]).abs().max()}",
            )
            # Compare shared expert weights
            self.assertTrue(
                torch.allclose(
                    fused_w[:, -1], ref_topk_weights[:, -1], atol=2e-4, rtol=0
                ),
                f"shared weights mismatch: max abs diff = {(fused_w[:, -1] - ref_topk_weights[:, -1]).abs().max()}",
            )
        else:
            torch.testing.assert_close(fused_ids, ref_topk_ids, atol=0, rtol=0)
            self.assertTrue(
                torch.allclose(fused_w, ref_topk_weights, atol=2e-4, rtol=0),
                f"weights mismatch: max abs diff = {(fused_w - ref_topk_weights).abs().max()}",
            )

    def test_hash_topk_no_fused(self):
        for scoring_func in ["softmax", "sigmoid", "sqrtsoftplus"]:
            for gating_dtype in [torch.float32, torch.bfloat16]:
                for E in [64, 128, 256]:
                    self._run_single_test(34, E, 8, scoring_func, 0, 1.5, gating_dtype)

    def test_hash_topk_with_fused(self):
        for scoring_func in ["sigmoid", "sqrtsoftplus"]:
            for gating_dtype in [torch.float32, torch.bfloat16]:
                for E in [64, 128, 256]:
                    self._run_single_test(34, E, 8, scoring_func, 1, 1.5, gating_dtype)


if __name__ == "__main__":
    unittest.main()
