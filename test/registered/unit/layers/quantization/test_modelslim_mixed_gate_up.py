"""CPU contracts for ModelSlim mixed-precision Qwen3.5 dense projections."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers import linear
from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    _maybe_enable_silu_fp4_quant_fusion,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PREFIX = "model.language_model.layers.0.mlp"


def make_config(gate="FLOAT", up="W4A8_MXFP"):
    return ModelSlimConfig(
        {
            f"{PREFIX}.gate_proj.weight": gate,
            f"{PREFIX}.up_proj.weight": up,
            f"{PREFIX}.down_proj.weight": "W4A8_MXFP",
            "packed_modules_mapping": {"gate_up_proj": ["gate_proj", "up_proj"]},
        }
    )


class TestModelSlimMixedGateUp(unittest.TestCase):
    def setUp(self):
        self.old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        self.group_patch = patch.object(
            linear, "_resolve_linear_group", return_value=None
        )
        self.group_patch.start()

    def tearDown(self):
        self.group_patch.stop()
        torch.set_default_dtype(self.old_dtype)

    def make_mlp(self, config, *, allow=True):
        return Qwen2MoeMLP(
            64,
            64,
            "silu",
            quant_config=config,
            prefix=PREFIX,
            allow_mixed_precision=allow,
        )

    def test_config_requires_complete_different_precision_entries(self):
        prefixes = [f"{PREFIX}.gate_proj", f"{PREFIX}.up_proj"]
        for gate, up, split in (
            ("FLOAT", "W4A8_MXFP", True),
            ("W4A8_MXFP", "FLOAT", True),
            ("FLOAT", "FLOAT", False),
            ("W4A8_MXFP", "W4A8_MXFP", False),
        ):
            with self.subTest(gate=gate, up=up):
                self.assertEqual(
                    make_config(gate, up).should_split_merged_linear(prefixes), split
                )
        config = make_config()
        del config.quant_description[f"{PREFIX}.up_proj.weight"]
        self.assertFalse(config.should_split_merged_linear(prefixes))
        with self.assertRaisesRegex(ValueError, "some but not all shards"):
            self.make_mlp(config)

    def test_split_preserves_weight_formats_and_default_fused_guard(self):
        for gate, up in (("FLOAT", "W4A8_MXFP"), ("W4A8_MXFP", "FLOAT")):
            with self.subTest(gate=gate):
                config = make_config(gate, up)
                mlp = self.make_mlp(config)
                self.assertIsNone(mlp.gate_up_proj)
                for name, scheme in (("gate_proj", gate), ("up_proj", up)):
                    projection = getattr(mlp, name)
                    self.assertEqual(
                        projection.weight.dtype,
                        torch.bfloat16 if scheme == "FLOAT" else torch.uint8,
                    )
                    self.assertEqual(
                        tuple(projection.weight.shape),
                        (64, 64) if scheme == "FLOAT" else (64, 32),
                    )
                    self.assertEqual(
                        hasattr(projection, "weight_scale"), scheme != "FLOAT"
                    )
                with self.assertRaisesRegex(ValueError, "some but not all shards"):
                    self.make_mlp(config, allow=False)
                _maybe_enable_silu_fp4_quant_fusion(mlp)
                self.assertFalse(mlp._enable_silu_fp4_quant_fusion)

    def test_uniform_precision_still_uses_merged_projection(self):
        for scheme in ("FLOAT", "W4A8_MXFP"):
            with self.subTest(scheme=scheme):
                mlp = self.make_mlp(make_config(scheme, scheme))
                self.assertIsInstance(
                    mlp.gate_up_proj, linear.MergedColumnParallelLinear
                )
                self.assertFalse(hasattr(mlp, "gate_proj"))
                self.assertFalse(hasattr(mlp, "up_proj"))

    def test_split_tp_weight_and_scale_partitioning(self):
        for rank in (0, 1):
            with (
                self.subTest(rank=rank),
                patch.object(
                    linear,
                    "_resolve_linear_group",
                    return_value=linear._LogicalGroup(rank, 2),
                ),
            ):
                mlp = Qwen2MoeMLP(
                    128,
                    128,
                    "silu",
                    quant_config=make_config(),
                    prefix=PREFIX,
                    parallel_group="attn_tp",
                    allow_mixed_precision=True,
                )
                for projection, field, shape, axis in (
                    (mlp.gate_proj, "weight", (128, 128), 0),
                    (mlp.up_proj, "weight", (128, 64), 0),
                    (mlp.up_proj, "weight_scale", (128, 4), 0),
                    (mlp.down_proj, "weight", (128, 64), 1),
                    (mlp.down_proj, "weight_scale", (128, 4), 1),
                ):
                    self.assertEqual(projection.parallel_group, "attn_tp")
                    param = getattr(projection, field)
                    source = (
                        torch.arange(shape[0] * shape[1]).reshape(shape) % 251
                    ).to(param.dtype)
                    param.weight_loader(param, source)
                    local_size = source.shape[axis] // 2
                    torch.testing.assert_close(
                        param,
                        source.narrow(axis, rank * local_size, local_size),
                    )

    def test_forward_preserves_gate_up_order_and_activation(self):
        mlp = self.make_mlp(make_config())
        x = torch.randn(3, 64)
        gate, up = torch.randn_like(x), torch.randn_like(x)
        with (
            patch.object(mlp.gate_proj, "forward", return_value=(gate, None)) as g,
            patch.object(mlp.up_proj, "forward", return_value=(up, None)) as u,
            patch.object(
                mlp.down_proj, "forward", side_effect=lambda y: (y, None)
            ) as d,
        ):
            output = mlp(x)
        g.assert_called_once_with(x)
        u.assert_called_once_with(x)
        expected = torch.nn.functional.silu(gate) * up
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(d.call_args.args[0], expected)

    def test_qwen35_loaders_keep_uniform_shard_routing(self):
        for model_class, checkpoint_prefix in (
            (Qwen3_5ForCausalLM, "model.layers.0.mlp"),
            (Qwen3_5ForConditionalGeneration, PREFIX),
        ):
            for scheme in ("FLOAT", "W4A8_MXFP"):
                with self.subTest(model=model_class.__name__, scheme=scheme):
                    model = model_class.__new__(model_class)
                    nn.Module.__init__(model)
                    mlp = self.make_mlp(make_config(scheme, scheme))
                    model.model = nn.Module()
                    model.model.layers = nn.ModuleList([nn.Module()])
                    model.model.layers[0].mlp = mlp
                    model.config = SimpleNamespace(tie_word_embeddings=False)
                    model.pp_group = SimpleNamespace(is_last_rank=True)
                    model.pooler = None
                    merged = mlp.gate_up_proj
                    checkpoint = []
                    for leaf in ("weight", "weight_scale"):
                        if not hasattr(merged, leaf):
                            continue
                        param = getattr(merged, leaf)
                        half_shape = (param.shape[0] // 2, *param.shape[1:])
                        checkpoint.extend(
                            (
                                f"{checkpoint_prefix}.{projection}.{leaf}",
                                torch.full(half_shape, value, dtype=param.dtype),
                            )
                            for projection, value in (("gate_proj", 7), ("up_proj", 11))
                        )
                    loaded = model.load_weights(checkpoint)
                    expected_names = {
                        f"model.layers.0.mlp.gate_up_proj.{leaf}"
                        for leaf in ("weight", "weight_scale")
                        if hasattr(merged, leaf)
                    }
                    self.assertEqual(loaded, expected_names)
                    for leaf in ("weight", "weight_scale"):
                        if hasattr(merged, leaf):
                            gate, up = getattr(merged, leaf).chunk(2, dim=0)
                            torch.testing.assert_close(gate, torch.full_like(gate, 7))
                            torch.testing.assert_close(up, torch.full_like(up, 11))

    def test_qwen35_loaders_preserve_unmerged_checkpoint_keys(self):
        for model_class, checkpoint_prefix in (
            (Qwen3_5ForCausalLM, "model.layers.0.mlp"),
            (Qwen3_5ForConditionalGeneration, PREFIX),
        ):
            with self.subTest(model=model_class.__name__):
                model = model_class.__new__(model_class)
                nn.Module.__init__(model)
                mlp = self.make_mlp(make_config())
                model.model = nn.Module()
                model.model.layers = nn.ModuleList([nn.Module()])
                model.model.layers[0].mlp = mlp
                model.config = SimpleNamespace(tie_word_embeddings=False)
                model.pp_group = SimpleNamespace(is_last_rank=True)
                model.pooler = None
                params = dict(model.named_parameters())
                checkpoint = []
                expected_names = set()
                expected_values = {}
                for leaf in (
                    "gate_proj.weight",
                    "up_proj.weight",
                    "up_proj.weight_scale",
                    "down_proj.weight",
                    "down_proj.weight_scale",
                ):
                    runtime_name = f"model.layers.0.mlp.{leaf}"
                    param = params[runtime_name]
                    value = (torch.arange(param.numel()).reshape(param.shape) % 251).to(
                        param.dtype
                    )
                    checkpoint.append((f"{checkpoint_prefix}.{leaf}", value))
                    expected_names.add(runtime_name)
                    expected_values[runtime_name] = value
                checkpoint.append(
                    ("mtp.layers.0.mlp.gate_proj.weight", torch.ones(64, 64))
                )
                loaded = model.load_weights(checkpoint)
                self.assertEqual(loaded, expected_names)
                for name in expected_names:
                    torch.testing.assert_close(params[name], expected_values[name])


if __name__ == "__main__":
    unittest.main()
