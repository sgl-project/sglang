"""CPU checks for the serialized-checkpoint INT64 model reference."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from sglang.test.ascend.static_int8_reference.checkpoint import CheckpointIntegerOracle
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def fixture(directory, strategy):
    prefix = "model.layers.0."
    specifications = (
        ("self_attn.q_proj", 4, 8),
        ("self_attn.k_proj", 2, 8),
        ("self_attn.v_proj", 2, 8),
        ("self_attn.o_proj", 8, 8),
        ("mlp.gate_proj", 6, 8),
        ("mlp.up_proj", 6, 8),
        ("mlp.down_proj", 8, 6),
    )
    serialized = {}
    for index, (name, rows, columns) in enumerate(specifications):
        name = prefix + name
        serialized[name + ".weight"] = (
            torch.arange(rows * columns).reshape(rows, columns) + index - 20
        ).to(torch.int8)
        scale = (
            torch.arange(1, rows + 1).float() * 0.015625
            if strategy == "channel"
            else torch.tensor([0.015625 * (index + 1)])
        )
        serialized[name + ".weight_scale"] = scale.reshape(-1, 1)
        serialized[name + ".input_scale"] = torch.tensor([0.125])
    serialized["model.norm.weight"] = torch.ones(8, dtype=torch.bfloat16)
    serialized["model.embed_tokens.weight"] = torch.ones(10, 8, dtype=torch.bfloat16)
    serialized["lm_head.weight"] = torch.ones(10, 8, dtype=torch.bfloat16)
    save_file(serialized, Path(directory) / "model.safetensors")
    runtime = {}
    groups = {
        "self_attn.qkv_proj": (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        ),
        "self_attn.o_proj": ("self_attn.o_proj",),
        "mlp.gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
        "mlp.down_proj": ("mlp.down_proj",),
    }
    for fused, parts in groups.items():
        names = [prefix + part for part in parts]
        weights = [serialized[name + ".weight"] for name in names]
        output = prefix + fused
        runtime[output + ".weight"] = torch.cat(weights).T.contiguous()
        runtime[output + ".weight_scale"] = torch.cat(
            [
                serialized[name + ".weight_scale"].flatten().expand(weight.shape[0])
                for name, weight in zip(names, weights)
            ]
        )
        runtime[output + ".input_scale"] = torch.tensor([0.125])
        runtime[output + ".npu_static_quant_scale"] = torch.full(
            (weights[0].shape[1],), 0.125
        )
        runtime[output + ".npu_static_quant_offset"] = torch.zeros(
            weights[0].shape[1], dtype=torch.int32
        )
    for name in ("model.norm.weight", "model.embed_tokens.weight", "lm_head.weight"):
        runtime[name] = serialized[name].clone()
    return serialized, runtime


class TestStaticCheckpointReference(unittest.TestCase):
    def test_channel_and_tensor_fused_loader_identity(self):
        for strategy in ("channel", "tensor"):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as path:
                _, runtime = fixture(path, strategy)
                oracle = CheckpointIntegerOracle(path)
                checks = oracle.verify_model(
                    SimpleNamespace(state_dict=lambda: runtime)
                )
                self.assertEqual(len(oracle.logical_names), 7)
                self.assertEqual(len(oracle.groups), 4)
                self.assertEqual(len(checks), 23)
                self.assertTrue(all(row["passed"] for row in checks))

    def test_corrupt_loaded_weight_scale_offset_and_norm_rejected(self):
        for name in (
            "model.layers.0.self_attn.qkv_proj.weight",
            "model.layers.0.self_attn.qkv_proj.weight_scale",
            "model.layers.0.self_attn.qkv_proj.npu_static_quant_offset",
            "model.norm.weight",
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as path:
                _, runtime = fixture(path, "channel")
                runtime[name].flatten()[0] += 1
                oracle = CheckpointIntegerOracle(path)
                with self.assertRaisesRegex(
                    ValueError, "Loaded checkpoint tensor differs"
                ):
                    oracle.verify_model(SimpleNamespace(state_dict=lambda: runtime))

    def test_missing_loaded_nonquantized_tensor_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            _, runtime = fixture(path, "channel")
            del runtime["model.norm.weight"]
            with self.assertRaisesRegex(ValueError, "Missing loaded checkpoint tensor"):
                CheckpointIntegerOracle(path).verify_model(
                    SimpleNamespace(state_dict=lambda: runtime)
                )

    def test_unequal_serialized_fused_activation_scales_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            serialized, runtime = fixture(path, "channel")
            serialized["model.layers.0.self_attn.k_proj.input_scale"] = torch.tensor(
                [0.25]
            )
            save_file(serialized, Path(path) / "model.safetensors")
            with self.assertRaisesRegex(ValueError, "Serialized fused input scales"):
                CheckpointIntegerOracle(path).verify_model(
                    SimpleNamespace(state_dict=lambda: runtime)
                )

    def test_integer_outputs_come_from_serialized_codes(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as path:
                serialized, runtime = fixture(path, "tensor")
                oracle = CheckpointIntegerOracle(path)
                oracle.verify_model(SimpleNamespace(state_dict=lambda: runtime))
                x = torch.linspace(-2, 2, 24).reshape(3, 8).to(dtype)
                bias = torch.linspace(-0.1, 0.1, 8).to(dtype)
                names = [
                    "model.layers.0.self_attn." + part
                    for part in ("q_proj", "k_proj", "v_proj")
                ]
                integer_weight = torch.cat(
                    [serialized[name + ".weight"] for name in names]
                ).to(torch.int64)
                scales = torch.cat(
                    [
                        serialized[name + ".weight_scale"]
                        .flatten()
                        .expand(serialized[name + ".weight"].shape[0])
                        for name in names
                    ]
                )
                codes = torch.round(x.float() / 0.125).clamp(-128, 127).to(torch.int64)
                expected = (
                    (codes @ integer_weight.T).float() * (scales * 0.125) + bias.float()
                ).to(dtype)
                # Mutating the runtime after identity checking must not change
                # the independent expected computation from serialized codes.
                runtime["model.layers.0.self_attn.qkv_proj.weight"].fill_(0)
                actual = oracle.linear("model.layers.0.self_attn.qkv_proj", x, bias)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertEqual(len(oracle.calls), 1)


if __name__ == "__main__":
    unittest.main()
