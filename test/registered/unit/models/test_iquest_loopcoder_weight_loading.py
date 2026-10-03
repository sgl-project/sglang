"""Regression tests for IQuestLoopCoder checkpoint weight loading."""

import unittest

import torch

from sglang.srt.layers.linear import MergedColumnParallelLinear, QKVParallelLinear
from sglang.srt.models.iquest_loopcoder import (
    IQuestLoopCoderForCausalLM,
    LoopGateProjection,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestIQuestLoopCoderWeightLoading(CustomTestCase):
    def _make_minimal_model(self, named_parameters):
        model = object.__new__(IQuestLoopCoderForCausalLM)
        model.named_parameters = lambda: iter(named_parameters)
        return model

    @staticmethod
    def _fill(param, value):
        with torch.no_grad():
            param.fill_(value)

    def test_canonical_gate_weight_and_bias_load_tp_shards(self):
        for tp_size, tp_rank in ((1, 0), (2, 0), (2, 1)):
            with (
                self.subTest(tp_size=tp_size, tp_rank=tp_rank),
                get_parallel().override(tp_size=tp_size, tp_rank=tp_rank),
            ):
                gates = {}
                named_parameters = []
                weights = []
                expected = {}
                shard_size = 4 // tp_size

                for layer, start in ((0, 20), (79, 2000)):
                    gate = LoopGateProjection(total_num_heads=4, head_dim=2)
                    self._fill(gate.gate_proj.weight, -3)
                    self._fill(gate.gate_proj.bias, -4)
                    gates[layer] = gate

                    weight = torch.arange(
                        start, start + 8, dtype=torch.float32
                    ).reshape(4, 2)
                    bias = torch.arange(start, start + 4, dtype=torch.float32)
                    named_parameters.extend(
                        [
                            (
                                f"model.gate_projections.{layer}.gate_proj.weight",
                                gate.gate_proj.weight,
                            ),
                            (
                                f"model.gate_projections.{layer}.gate_proj.bias",
                                gate.gate_proj.bias,
                            ),
                        ]
                    )
                    weights.extend(
                        [
                            (f"model.gate_projections.{layer}.weight", weight),
                            (f"model.gate_projections.{layer}.bias", bias),
                        ]
                    )
                    expected[layer] = (
                        weight[tp_rank * shard_size : (tp_rank + 1) * shard_size],
                        bias[tp_rank * shard_size : (tp_rank + 1) * shard_size],
                    )

                model = self._make_minimal_model(named_parameters)
                model.load_weights(weights)

                for layer, gate in gates.items():
                    expected_weight, expected_bias = expected[layer]
                    torch.testing.assert_close(
                        gate.gate_proj.weight, expected_weight, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        gate.gate_proj.bias, expected_bias, rtol=0, atol=0
                    )

    def test_tp1_qkv_and_mlp_gate_up_mappings(self):
        with get_parallel().override(tp_size=1, tp_rank=0):
            qkv_proj = QKVParallelLinear(
                hidden_size=2,
                head_size=2,
                total_num_heads=2,
                total_num_kv_heads=2,
                bias=False,
                prefix="model.layers.0.self_attn.qkv_proj",
            )
            gate_up_proj = MergedColumnParallelLinear(
                input_size=2,
                output_sizes=[3, 3],
                bias=False,
                prefix="model.layers.0.mlp.gate_up_proj",
            )
            self._fill(qkv_proj.weight, -5)
            self._fill(gate_up_proj.weight, -6)

            q_weight = torch.arange(0, 8, dtype=torch.float32).reshape(4, 2)
            k_weight = torch.arange(10, 18, dtype=torch.float32).reshape(4, 2)
            v_weight = torch.arange(20, 28, dtype=torch.float32).reshape(4, 2)
            gate_weight = torch.arange(30, 36, dtype=torch.float32).reshape(3, 2)
            up_weight = torch.arange(40, 46, dtype=torch.float32).reshape(3, 2)

            model = self._make_minimal_model(
                [
                    (
                        "model.layers.0.self_attn.qkv_proj.weight",
                        qkv_proj.weight,
                    ),
                    (
                        "model.layers.0.mlp.gate_up_proj.weight",
                        gate_up_proj.weight,
                    ),
                ]
            )
            model.load_weights(
                [
                    ("model.layers.0.self_attn.q_proj.weight", q_weight),
                    ("model.layers.0.self_attn.k_proj.weight", k_weight),
                    ("model.layers.0.self_attn.v_proj.weight", v_weight),
                    ("model.layers.0.mlp.gate_proj.weight", gate_weight),
                    ("model.layers.0.mlp.up_proj.weight", up_weight),
                ]
            )

        torch.testing.assert_close(
            qkv_proj.weight, torch.cat((q_weight, k_weight, v_weight)), rtol=0, atol=0
        )
        torch.testing.assert_close(
            gate_up_proj.weight, torch.cat((gate_weight, up_weight)), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
