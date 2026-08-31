"""Regression tests for GLM MoE gate weights used by FP32 routing."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    _model_load_weights_direct,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.glm4_moe import Glm4MoeGate
from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteGate
from sglang.test.test_utils import CustomTestCase

_CONFIG = SimpleNamespace(n_routed_experts=3, hidden_size=4)


class TestGlmMoeGateFp32Weight(CustomTestCase):
    def test_bf16_load_updates_fp32_weight_in_place(self):
        """BF16 runtime updates must overwrite the canonical FP32 gate weight."""
        hidden_states = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        initial = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        updated = initial + 1

        for gate_cls in (Glm4MoeGate, Glm4MoeLiteGate):
            with self.subTest(gate=gate_cls.__name__):
                with set_default_torch_dtype(torch.bfloat16):
                    gate = gate_cls(_CONFIG)

                self.assertEqual(gate.weight.dtype, torch.float32)
                self.assertFalse(hasattr(gate, "_weight_fp32"))
                weight_ptr = gate.weight.data_ptr()

                default_weight_loader(gate.weight, initial)
                torch.testing.assert_close(gate.weight, initial.float())

                _model_load_weights_direct(gate, [("weight", updated)])
                self.assertEqual(gate.weight.data_ptr(), weight_ptr)
                torch.testing.assert_close(gate.weight, updated.float())
                torch.testing.assert_close(
                    gate(hidden_states),
                    F.linear(hidden_states.float(), updated.float()),
                )


if __name__ == "__main__":
    unittest.main()
