"""Unit tests for the DeepSeek MoE gate's NPU FP32 router projection."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_loader.weight_utils import default_weight_loader  # noqa: E402
from sglang.srt.models import deepseek_v2  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _gate_config(*, hidden_size=16, n_routed_experts=8):
    return SimpleNamespace(
        architectures=["DeepseekV2ForCausalLM"],
        hidden_size=hidden_size,
        n_routed_experts=n_routed_experts,
        topk_method="greedy",
    )


def _exec_config(*, deterministic=False):
    return SimpleNamespace(
        deterministic=SimpleNamespace(
            enable_deterministic_inference=deterministic,
        )
    )


class TestDeepseekV2NpuRouter(CustomTestCase):
    def _make_gate(self, *, is_npu):
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with patch.object(deepseek_v2, "_is_npu", is_npu):
                return deepseek_v2.MoEGate(_gate_config(), quant_config=None)
        finally:
            torch.set_default_dtype(original_dtype)

    def _run_npu_gate(
        self,
        gate,
        hidden_states,
        *,
        deterministic=False,
        prefill_context_parallel=False,
    ):
        forward_batch = object() if prefill_context_parallel else None
        with (
            patch.object(deepseek_v2, "_is_npu", True),
            patch.object(deepseek_v2, "_is_cuda", False),
            patch.object(deepseek_v2, "_use_aiter", False),
            patch.object(
                deepseek_v2,
                "get_exec",
                return_value=_exec_config(deterministic=deterministic),
            ),
            patch.object(
                deepseek_v2,
                "dsa_use_prefill_cp",
                return_value=prefill_context_parallel,
            ),
            patch.object(
                deepseek_v2,
                "mla_use_prefill_cp",
                return_value=False,
            ),
        ):
            return gate(hidden_states, forward_batch=forward_batch)

    def test_bf16_writeback_differs_from_fp32_projection(self):
        torch.manual_seed(34861)
        hidden_states = torch.randn(16, 32, dtype=torch.bfloat16)
        weight = torch.randn(12, 32, dtype=torch.bfloat16)

        bf16_then_fp32 = F.linear(hidden_states, weight).float()
        fp32_reference = F.linear(hidden_states.float(), weight.float())

        self.assertFalse(torch.equal(bf16_then_fp32, fp32_reference))
        self.assertGreater(
            (bf16_then_fp32 - fp32_reference).abs().max().item(),
            0.0,
        )

    def test_npu_gate_weight_stays_fp32_after_load_and_update(self):
        gate = self._make_gate(is_npu=True)
        self.assertEqual(gate.weight.dtype, torch.float32)

        checkpoint_weight = torch.randn_like(gate.weight, dtype=torch.bfloat16)
        default_weight_loader(gate.weight, checkpoint_weight)
        self.assertEqual(gate.weight.dtype, torch.float32)
        torch.testing.assert_close(gate.weight, checkpoint_weight.float())

        updated_weight = torch.full_like(checkpoint_weight, 0.25)
        default_weight_loader(gate.weight, updated_weight)
        self.assertEqual(gate.weight.dtype, torch.float32)
        torch.testing.assert_close(gate.weight, updated_weight.float())

    def test_non_npu_gate_keeps_default_parameter_dtype(self):
        gate = self._make_gate(is_npu=False)
        self.assertEqual(gate.weight.dtype, torch.bfloat16)

    def test_regular_npu_routing_returns_fp32_reference(self):
        gate = self._make_gate(is_npu=True)
        torch.manual_seed(1)
        checkpoint_weight = torch.randn_like(gate.weight, dtype=torch.bfloat16)
        default_weight_loader(gate.weight, checkpoint_weight)
        hidden_states = torch.randn(4, gate.weight.shape[1], dtype=torch.bfloat16)

        logits = self._run_npu_gate(gate, hidden_states)
        reference = F.linear(hidden_states.float(), gate.weight)

        self.assertEqual(logits.dtype, torch.float32)
        torch.testing.assert_close(logits, reference)

    def test_all_npu_branches_use_shared_fp32_projection(self):
        hidden_states = torch.randn(4, 16, dtype=torch.bfloat16)
        cases = (
            ("regular", False, False),
            ("prefill_context_parallel", False, True),
            ("deterministic", True, False),
        )

        for name, deterministic, prefill_context_parallel in cases:
            with self.subTest(name=name):
                gate = self._make_gate(is_npu=True)
                expected = torch.randn(4, 8, dtype=torch.float32)
                with patch.object(
                    gate,
                    "_npu_router_gemm_fp32",
                    return_value=expected,
                ) as mock_projection:
                    logits = self._run_npu_gate(
                        gate,
                        hidden_states,
                        deterministic=deterministic,
                        prefill_context_parallel=prefill_context_parallel,
                    )

                self.assertIs(logits, expected)
                mock_projection.assert_called_once_with(hidden_states)

    def test_non_npu_routing_does_not_use_npu_projection(self):
        gate = self._make_gate(is_npu=False)
        hidden_states = torch.randn(4, 16, dtype=torch.bfloat16)
        with (
            patch.object(deepseek_v2, "_is_npu", False),
            patch.object(deepseek_v2, "_is_cuda", False),
            patch.object(deepseek_v2, "_use_aiter", False),
            patch.object(
                deepseek_v2,
                "get_exec",
                return_value=_exec_config(deterministic=False),
            ),
            patch.object(
                gate,
                "_npu_router_gemm_fp32",
                side_effect=AssertionError("unexpected NPU projection"),
            ) as mock_projection,
        ):
            logits = gate(hidden_states)

        self.assertEqual(logits.dtype, torch.bfloat16)
        mock_projection.assert_not_called()

    def test_runtime_weight_update_changes_next_projection(self):
        gate = self._make_gate(is_npu=True)
        hidden_states = torch.ones(2, 16, dtype=torch.bfloat16)
        default_weight_loader(
            gate.weight,
            torch.zeros_like(gate.weight, dtype=torch.bfloat16),
        )
        before_update = self._run_npu_gate(gate, hidden_states)

        default_weight_loader(
            gate.weight,
            torch.ones_like(gate.weight, dtype=torch.bfloat16),
        )
        after_update = self._run_npu_gate(gate, hidden_states)

        self.assertEqual(after_update.dtype, torch.float32)
        self.assertFalse(torch.equal(before_update, after_update))
        torch.testing.assert_close(after_update, torch.full_like(after_update, 16.0))


if __name__ == "__main__":
    unittest.main()
