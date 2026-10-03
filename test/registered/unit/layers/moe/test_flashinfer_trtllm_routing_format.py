"""The configured TRT-LLM backend consumes either logits or materialized routes."""

import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardCombineInput,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import (
    BypassedTopKOutput,
    PackedTopKOutput,
    StandardTopKOutput,
    StandardTopKOutputPacked,
    TopKConfig,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlashinferTrtllmRoutingFormat(CustomTestCase):
    def test_format_selects_routed_or_logits_for_each_quantization(self):
        from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm as trtllm

        hidden = torch.ones(2, 4)
        logits = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
        weights = torch.ones(2, 1)
        ids = torch.tensor([[1], [0]], dtype=torch.int32)
        packed_ids = torch.zeros(2, 1, dtype=torch.int32)
        config = TopKConfig(top_k=1)
        outputs = [
            ("bypassed", BypassedTopKOutput(hidden, logits, config), False),
            ("standard", StandardTopKOutput(weights, ids, None), True),
            (
                "standard_packed",
                StandardTopKOutputPacked(weights, ids, None, packed_ids),
                True,
            ),
            ("packed", PackedTopKOutput(packed_ids, None), True),
        ]
        weight = torch.empty(0)
        quantizations = {
            "fp4": trtllm.FlashInferTrtllmFp4MoeQuantInfo(
                w13_weight=weight,
                w2_weight=weight,
                w13_weight_scale=weight,
                w2_weight_scale=weight,
                g1_scale_c=weight,
                g1_alphas=weight,
                g2_alphas=weight,
                w13_input_scale_quant=weight,
                global_num_experts=2,
                local_expert_offset=0,
                local_num_experts=2,
                intermediate_size_per_partition=4,
                routing_method_type=0,
            ),
            "fp8": trtllm.FlashInferTrtllmFp8MoeQuantInfo(
                w13_weight=weight,
                w2_weight=weight,
                global_num_experts=2,
                local_expert_offset=0,
                local_num_experts=2,
                intermediate_size=4,
                routing_method_type=0,
                block_quant=True,
            ),
            "bf16": trtllm.FlashInferTrtllmBf16MoeQuantInfo(
                gemm1_weights=weight,
                gemm2_weights=weight,
                global_num_experts=2,
                local_expert_offset=0,
            ),
        }
        runner_config = MoeRunnerConfig(top_k=1)
        expected = StandardCombineInput(hidden_states=hidden)
        with ExitStack() as stack:
            kernels = {}
            for quantization in quantizations:
                kernels[quantization] = stack.enter_context(
                    patch.object(
                        trtllm,
                        f"fused_experts_none_to_flashinfer_trtllm_{quantization}",
                        Mock(return_value=expected),
                    )
                )
            for quantization, quant_info in quantizations.items():
                for name, topk_output, use_routed_topk in outputs:
                    with self.subTest(quantization=quantization, format=name):
                        for kernel in kernels.values():
                            kernel.reset_mock()
                        dispatch_output = StandardDispatchOutput(
                            hidden_states=hidden,
                            hidden_states_scale=None,
                            topk_output=topk_output,
                        )
                        result = trtllm.fused_experts_none_to_flashinfer_trtllm(
                            dispatch_output, quant_info, runner_config
                        )
                        self.assertIs(result, expected)
                        kernels[quantization].assert_called_once_with(
                            dispatch_output,
                            quant_info,
                            runner_config,
                            use_routed_topk=use_routed_topk,
                        )
                        for other, kernel in kernels.items():
                            if other != quantization:
                                kernel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
