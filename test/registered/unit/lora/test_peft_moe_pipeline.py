"""Exercise LoRAConfig, tensor ingestion and adapter normalization on CPU."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def new_adapter(config, **dimensions):
    base_config = SimpleNamespace(
        **{
            "num_hidden_layers": 1,
            "num_experts": 3,
            "hidden_size": 5,
            "moe_intermediate_size": 3,
            **dimensions,
        }
    )
    return LoRAAdapter(
        "cpu-test", LoRAConfig.from_dict(config), base_config, None, None
    )


class TestPEFTMoEPipeline(CustomTestCase):
    def config(self, **overrides):
        return {
            "peft_type": "LORA",
            "r": 2,
            "lora_alpha": 16,
            "target_modules": ["gate_proj", "up_proj", "down_proj"],
            "target_parameters": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
            **overrides,
        }

    def test_both_layouts_complete_pipeline_and_scaling(self):
        e, r, h, i = 3, 2, 5, 3
        prefix = "base_model.model.model.language_model.layers.0.mlp.experts"
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                adapter = new_adapter(self.config())
                saved = {}
                originals = {}
                for projection, inputs, outputs, wrapper in (
                    ("gate_up_proj", h, 2 * i, ".base_layer"),
                    ("down_proj", i, h, ""),
                ):
                    a_dim, b_dim = (outputs, inputs) if legacy else (inputs, outputs)
                    a = (
                        torch.arange(e * r * a_dim, dtype=torch.float64).reshape(
                            e * r, a_dim
                        )
                        / 16
                    )
                    b = (
                        torch.arange(b_dim * r * e, dtype=torch.float64).reshape(
                            b_dim, r * e
                        )
                        / 32
                    )
                    saved[f"{prefix}{wrapper}.lora_A.weight"] = a
                    saved[f"{prefix}{wrapper}.lora_B.weight"] = b
                    originals[projection] = (a, b, inputs, outputs)
                adapter.initialize_weights_from_tensors(saved)
                normalized = adapter.layers[0].weights
                for projection, (
                    saved_a,
                    saved_b,
                    inputs,
                    outputs,
                ) in originals.items():
                    a = normalized[f"{prefix}.{projection}.lora_A.weight"]
                    b = normalized[f"{prefix}.{projection}.lora_B.weight"]
                    self.assertEqual(
                        tuple(a.shape),
                        (e, 2 * r if projection == "gate_up_proj" else r, inputs),
                    )
                    self.assertEqual(tuple(b.shape), (e, outputs, r))
                    expected = torch.zeros(e, outputs, inputs, dtype=torch.float64)
                    for expert in range(e):
                        for output in range(outputs):
                            for input_ in range(inputs):
                                for rank in range(r):
                                    if legacy:
                                        expected[expert, output, input_] += (
                                            saved_a[expert * r + rank, output]
                                            * saved_b[input_, rank * e + expert]
                                        )
                                    else:
                                        expected[expert, output, input_] += (
                                            saved_b[output, rank * e + expert]
                                            * saved_a[expert * r + rank, input_]
                                        )
                    if projection == "gate_up_proj":
                        actual = torch.cat(
                            (
                                torch.bmm(b[:, :i], a[:, :r]),
                                torch.bmm(b[:, i:], a[:, r:]),
                            ),
                            dim=1,
                        )
                    else:
                        actual = torch.bmm(b, a)
                    scale = 16 / r
                    torch.testing.assert_close(
                        actual * adapter.scaling,
                        expected * scale,
                        rtol=1e-12,
                        atol=1e-12,
                    )

    def test_existing_3d_gate_up_is_stacked_once(self):
        prefix = "model.layers.0.mlp.experts.gate_up_proj"
        adapter = new_adapter(self.config())
        a = torch.randn(3, 2, 5)
        b = torch.randn(3, 6, 2)
        adapter.initialize_weights_from_tensors(
            {f"{prefix}.lora_A.weight": a, f"{prefix}.lora_B.weight": b}
        )
        weights = adapter.layers[0].weights
        actual_a = weights[f"{prefix}.lora_A.weight"]
        self.assertEqual(tuple(actual_a.shape), (3, 4, 5))
        torch.testing.assert_close(actual_a[:, :2], a)
        torch.testing.assert_close(actual_a[:, 2:], a)
        self.assertIs(weights[f"{prefix}.lora_B.weight"], b)

    def test_dense_attention_normalization_is_unchanged(self):
        adapter = new_adapter(self.config(target_parameters=None))
        saved = {}
        prefix = "model.layers.0.self_attn"
        originals = {}
        for projection in ("q", "k", "v"):
            a, b = torch.randn(2, 5), torch.randn(4, 2)
            saved[f"{prefix}.{projection}_proj.lora_A.weight"] = a
            saved[f"{prefix}.{projection}_proj.lora_B.weight"] = b
            originals[projection] = (a, b)
        adapter.initialize_weights_from_tensors(saved)
        weights = adapter.layers[0].weights
        a = weights[f"{prefix}.qkv_proj.lora_A.weight"]
        b = weights[f"{prefix}.qkv_proj.lora_B.weight"]
        for index, projection in enumerate(("q", "k", "v")):
            expected_a, expected_b = originals[projection]
            torch.testing.assert_close(a[2 * index : 2 * (index + 1)], expected_a)
            torch.testing.assert_close(b[4 * index : 4 * (index + 1)], expected_b)


if __name__ == "__main__":
    unittest.main()
