"""
Unit tests for the GLM MoE gate FP32 cache refresh.

The gate projects routing logits in FP32 from a cached FP32 copy of the BF16
gate weight. Runtime weight updates (e.g. RL weight sync) write the BF16
parameter in place through its weight_loader, which must refresh the FP32
cache without reallocating it: CUDA graphs capture the cache's address.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    _model_load_weights_direct,
)
from sglang.srt.models.glm4_moe import Glm4MoeGate
from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteGate

_CONFIG = SimpleNamespace(n_routed_experts=3, hidden_size=4)


def _make_gate(gate_cls):
    gate = gate_cls(_CONFIG).to(dtype=torch.bfloat16)
    hidden_states = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    return gate, hidden_states


class TestGlmMoeGateFp32Cache(unittest.TestCase):
    GATE_CLASSES = (Glm4MoeGate, Glm4MoeLiteGate)

    def test_initial_load_materializes_cache(self):
        for gate_cls in self.GATE_CLASSES:
            with self.subTest(gate=gate_cls.__name__):
                gate, hidden_states = _make_gate(gate_cls)
                weight = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)

                gate.weight.weight_loader(gate.weight, weight)

                self.assertIsNotNone(gate._weight_fp32)
                self.assertEqual(gate._weight_fp32.dtype, torch.float32)
                torch.testing.assert_close(gate._weight_fp32, weight.float())
                torch.testing.assert_close(
                    gate(hidden_states),
                    F.linear(hidden_states.float(), weight.float()),
                )

    def test_update_refreshes_cache_in_place(self):
        for gate_cls in self.GATE_CLASSES:
            with self.subTest(gate=gate_cls.__name__):
                gate, hidden_states = _make_gate(gate_cls)
                initial = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
                gate.weight.weight_loader(gate.weight, initial)
                gate(hidden_states)
                cache_ptr = gate._weight_fp32.data_ptr()

                updated = initial + 1
                gate.weight.weight_loader(gate.weight, updated)

                self.assertEqual(gate._weight_fp32.data_ptr(), cache_ptr)
                torch.testing.assert_close(gate._weight_fp32, updated.float())
                torch.testing.assert_close(
                    gate(hidden_states),
                    F.linear(hidden_states.float(), updated.float()),
                )

    def test_update_after_lazy_forward_refreshes_in_place(self):
        # Dummy load initializes weights without weight_loader, so the first
        # forward materializes the cache; a later update must refresh it.
        for gate_cls in self.GATE_CLASSES:
            with self.subTest(gate=gate_cls.__name__):
                gate, hidden_states = _make_gate(gate_cls)
                gate.weight.data.copy_(torch.zeros(3, 4, dtype=torch.bfloat16))
                gate(hidden_states)
                cache_ptr = gate._weight_fp32.data_ptr()

                updated = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
                gate.weight.weight_loader(gate.weight, updated)

                self.assertEqual(gate._weight_fp32.data_ptr(), cache_ptr)
                torch.testing.assert_close(gate._weight_fp32, updated.float())

    def test_direct_load_refreshes_cache_in_place(self):
        # update_weights_from_tensor(load_format="direct") bypasses
        # param.weight_loader; the post_direct_write hook must resync the cache.
        for gate_cls in self.GATE_CLASSES:
            with self.subTest(gate=gate_cls.__name__):
                gate, hidden_states = _make_gate(gate_cls)
                initial = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
                gate.weight.weight_loader(gate.weight, initial)
                gate(hidden_states)
                cache_ptr = gate._weight_fp32.data_ptr()

                updated = initial + 1
                _model_load_weights_direct(gate, [("weight", updated)])

                self.assertEqual(gate._weight_fp32.data_ptr(), cache_ptr)
                torch.testing.assert_close(gate._weight_fp32, updated.float())
                torch.testing.assert_close(
                    gate(hidden_states),
                    F.linear(hidden_states.float(), updated.float()),
                )


if __name__ == "__main__":
    unittest.main()
