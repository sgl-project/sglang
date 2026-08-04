import unittest
from unittest.mock import patch

import torch
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopK,
    TopKOutputFormat,
    TritonKernelTopKOutput,
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
    def test_native_forward_preserves_standard_format(self):
        topk = TopK(top_k=2)
        hidden_states = torch.randn(1, 4)
        router_logits = torch.randn(1, 8)
        expected = StandardTopKOutput(
            torch.randn(1, 2), torch.zeros(1, 2, dtype=torch.int64), router_logits
        )

        with patch("sglang.srt.layers.moe.topk.select_experts", return_value=expected):
            output = topk.forward_native(hidden_states, router_logits)

        self.assertIs(output, expected)
        self.assertTrue(topk.topk_config.torch_native)

    def test_native_forward_preserves_bypassed_format(self):
        topk = TopK(top_k=2, output_format=TopKOutputFormat.BYPASSED)
        hidden_states = torch.randn(1, 4)
        router_logits = torch.randn(1, 8)

        output = topk.forward_native(hidden_states, router_logits)

        self.assertIsInstance(output, BypassedTopKOutput)
        self.assertIs(output.hidden_states, hidden_states)
        self.assertIs(output.router_logits, router_logits)

    def test_native_forward_preserves_backend_bypassed_format(self):
        topk = TopK(top_k=2)
        hidden_states = torch.randn(1, 4)
        router_logits = torch.randn(1, 8)

        with patch(
            "sglang.srt.layers.moe.topk.get_moe_runner_backend",
            return_value=MoeRunnerBackend.FLASHINFER_TRTLLM,
        ):
            output = topk.forward_native(hidden_states, router_logits)

        self.assertIsInstance(output, BypassedTopKOutput)
        self.assertIs(output.hidden_states, hidden_states)
        self.assertIs(output.router_logits, router_logits)

    def test_native_forward_preserves_triton_format(self):
        topk = TopK(top_k=2, output_format=TopKOutputFormat.TRITON_KERNEL)
        hidden_states = torch.randn(1, 4)
        router_logits = torch.randn(1, 8)
        routing_data, gather_idx, scatter_idx = object(), object(), object()

        with patch(
            "sglang.srt.layers.moe.topk.routing",
            return_value=(routing_data, gather_idx, scatter_idx),
            create=True,
        ):
            output = topk.forward_native(hidden_states, router_logits)

        self.assertIsInstance(output, TritonKernelTopKOutput)
        self.assertIs(output.routing_data, routing_data)
        self.assertIs(output.gather_indx, gather_idx)
        self.assertIs(output.scatter_indx, scatter_idx)

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


if __name__ == "__main__":
    unittest.main()
