"""CPU tests for model-specific MoE router precision contracts."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import sglang.srt.models.kimi_linear as kimi_linear
import sglang.srt.models.mimo_v2 as mimo_v2
from sglang.srt.models.bailing_moe import BailingMoEGate
from sglang.srt.models.bailing_moe_linear import BailingMoEGate as BailingLinearMoEGate
from sglang.srt.models.ernie4 import MoEGate as Ernie4MoEGate
from sglang.srt.models.llada2 import LLaDA2MoeGate
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@contextmanager
def default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class TestCorrectionBiasDtype(CustomTestCase):
    def test_mimo_modelopt_flashinfer_keeps_fp32_bias(self):
        config = SimpleNamespace(
            n_routed_experts=8,
            hidden_size=16,
            topk_method="noaux_tc",
        )
        quant_config = SimpleNamespace(get_name=lambda: "modelopt_fp4")
        backend = SimpleNamespace(is_flashinfer_trtllm=lambda: True)

        # ``create=True`` keeps this test valid after the model-specific backend
        # exception is removed from the production module.
        with patch.object(
            mimo_v2, "get_moe_runner_backend", return_value=backend, create=True
        ):
            gate = mimo_v2.MoEGate(config=config, quant_config=quant_config)

        self.assertEqual(gate.e_score_correction_bias.dtype, torch.float32)

    def test_ernie_keeps_fp32_bias_when_model_weights_are_bf16(self):
        config = SimpleNamespace(moe_num_experts=8, hidden_size=16)
        with default_dtype(torch.bfloat16):
            gate = Ernie4MoEGate(config=config)

        self.assertEqual(gate.weight.dtype, torch.bfloat16)
        self.assertEqual(gate.e_score_correction_bias.dtype, torch.float32)

    def test_kimi_linear_keeps_fp32_bias_when_model_weights_are_bf16(self):
        class FakeGate(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        class FakeExperts(nn.Module):
            should_fuse_routed_scaling_factor_in_topk = False

            def __init__(self, *args, **kwargs):
                super().__init__()

        class FakeTopK(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        config = SimpleNamespace(
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_experts=8,
            n_routed_experts=8,
            num_experts_per_token=2,
            num_expert_group=2,
            topk_group=1,
            moe_renormalize=True,
            routed_scaling_factor=1.0,
            num_shared_experts=None,
            hidden_act="silu",
            activation_situ_beta=1.0,
            activation_situ_linear_beta=0.0,
        )

        with (
            default_dtype(torch.bfloat16),
            patch.object(kimi_linear, "ReplicatedLinear", FakeGate),
            patch.object(kimi_linear, "get_moe_impl_class", return_value=FakeExperts),
            patch.object(kimi_linear, "TopK", FakeTopK),
            patch.object(
                kimi_linear,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=1),
            ),
        ):
            moe = kimi_linear.KimiMoE(config=config)

        self.assertEqual(moe.gate.e_score_correction_bias.dtype, torch.float32)


class TestRouterLogitsDtype(CustomTestCase):
    def test_fp32_router_weights_produce_fp32_logits(self):
        gate_types = (BailingMoEGate, BailingLinearMoEGate, LLaDA2MoeGate)
        config = SimpleNamespace(
            num_experts=8,
            hidden_size=16,
            moe_router_enable_expert_bias=False,
        )
        hidden_states = torch.randn(4, 16, dtype=torch.bfloat16)

        for gate_type in gate_types:
            with self.subTest(gate_type=gate_type.__name__):
                gate = gate_type(config=config, params_dtype=torch.float32)
                logits = gate(hidden_states)
                self.assertEqual(logits.dtype, torch.float32)
                self.assertEqual(logits.shape, (4, 8))

    def test_bf16_router_weights_still_produce_bf16_logits(self):
        gate_types = (BailingMoEGate, BailingLinearMoEGate, LLaDA2MoeGate)
        config = SimpleNamespace(
            num_experts=8,
            hidden_size=16,
            moe_router_enable_expert_bias=False,
        )
        hidden_states = torch.randn(4, 16, dtype=torch.bfloat16)

        for gate_type in gate_types:
            with self.subTest(gate_type=gate_type.__name__):
                gate = gate_type(config=config, params_dtype=torch.bfloat16)
                logits = gate(hidden_states)
                self.assertEqual(logits.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
