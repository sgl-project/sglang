import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.quantization import unquant
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import deepseek_v2
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestFlashInferFusedSharedExpert(CustomTestCase):
    def test_policy(self):
        cases = (
            (ForwardMode.DECODE, 0, 4, False),
            (ForwardMode.DECODE, 1, 4, True),
            (ForwardMode.DECODE, 64, 4, True),
            (ForwardMode.TARGET_VERIFY, 64, 4, True),
            (ForwardMode.DECODE, 65, 4, False),
            (ForwardMode.DECODE, 64, 8, True),
            (ForwardMode.DECODE, 65, 8, False),
            (ForwardMode.EXTEND, 1, 4, False),
            (ForwardMode.IDLE, 1, 4, False),
            (ForwardMode.DECODE, 1, 2, False),
        )
        for mode, num_tokens, tp_size, expected in cases:
            with self.subTest(mode=mode, num_tokens=num_tokens, tp_size=tp_size):
                batch = SimpleNamespace(forward_mode=mode)
                self.assertEqual(
                    deepseek_v2._use_flashinfer_fused_shared_expert(
                        batch, num_tokens, tp_size
                    ),
                    expected,
                )

    def test_fp4_handoff_selects_fused_shared_gemm1_activation(self):
        hidden_states = torch.empty(3, 6144, dtype=torch.bfloat16)
        activated, output = object(), object()
        mm_bf16_swiglu = MagicMock(return_value=activated)
        shared = MagicMock()
        shared.gate_up_proj.flashinfer_bf16_swiglu_weight = object()
        shared.down_proj.return_value = (output, None)
        moe = SimpleNamespace(num_fused_shared_experts=0, shared_experts=shared)

        with patch.dict(
            sys.modules,
            {"flashinfer": SimpleNamespace(mm_bf16_swiglu=mm_bf16_swiglu)},
        ):
            result = DeepseekV2MoE._forward_shared_experts(
                moe,
                hidden_states,
                pre_quant_input=(torch.empty(3, 3072, dtype=torch.uint8), object()),
            )

        self.assertIs(result, output)
        shared.assert_not_called()
        mm_bf16_swiglu.assert_called_once_with(
            hidden_states,
            shared.gate_up_proj.flashinfer_bf16_swiglu_weight,
            pdl=True,
        )
        shared.down_proj.assert_called_once_with(activated)

    def test_fused_ar_scale_uses_the_routed_expert_static_scale(self):
        scale = object()
        moe = SimpleNamespace(
            _enable_flashinfer_fused_shared_expert=True,
            tp_size=4,
            experts=SimpleNamespace(w13_input_scale_quant=scale),
        )
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

        with patch.object(
            deepseek_v2, "is_in_tc_piecewise_cuda_graph", return_value=False
        ):
            result = DeepseekV2MoE.get_fused_ar_fp4_quant_scale(
                moe, torch.empty(64, 1), batch
            )
            fallback = DeepseekV2MoE.get_fused_ar_fp4_quant_scale(
                moe, torch.empty(65, 1), batch
            )
        with patch.object(
            deepseek_v2, "is_in_tc_piecewise_cuda_graph", return_value=True
        ):
            tc_graph_fallback = DeepseekV2MoE.get_fused_ar_fp4_quant_scale(
                moe, torch.empty(64, 1), batch
            )

        self.assertIs(result, scale)
        self.assertIsNone(fallback)
        self.assertIsNone(tc_graph_fallback)

    def test_prepares_fused_weight_after_checkpoint_loading(self):
        layer = torch.nn.Module()
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.randn(128, 128, dtype=torch.bfloat16), requires_grad=False
            ),
        )
        prepared = torch.empty(128, 128, dtype=torch.bfloat16)
        original_weight = layer.weight
        prepare_weight = MagicMock(return_value=prepared)
        method = unquant.UnquantizedLinearMethod()
        method._prepare_flashinfer_bf16_swiglu_weight = True

        with (
            patch.object(unquant, "_is_cpu", False),
            patch(
                "flashinfer.prepare_bf16_swiglu_weight",
                prepare_weight,
                create=True,
            ),
        ):
            method.process_weights_after_loading(layer)

        self.assertIs(layer.flashinfer_bf16_swiglu_weight, prepared)
        self.assertIs(layer.weight, original_weight)
        self.assertIn(
            "flashinfer_bf16_swiglu_weight", layer._non_persistent_buffers_set
        )
        prepare_weight.assert_called_once_with(layer.weight, input_order="gate_up")


if __name__ == "__main__":
    unittest.main()
