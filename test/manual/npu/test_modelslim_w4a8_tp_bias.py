"""CPU regression for rank-local ModelSlim W4A8 compensation loading."""

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(
    os.environ.get("SGLANG_TEST_SOURCE_ROOT", Path(__file__).resolve().parents[3])
)


def load_methods(path, class_name, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    methods = [
        n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    # Avoid importing NPU initialization; execute the production loading methods.
    for method in methods:
        method.body = [n for n in method.body if not isinstance(n, ast.ImportFrom)]
    node.bases = []
    node.decorator_list = []
    node.body = methods
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[class_name]


def set_attrs(param, attrs):
    for name, value in attrs.items():
        setattr(param, name, value)


class TestW4A8TPBias(unittest.TestCase):
    def setUp(self):
        self.namespace = dict(
            torch=torch,
            FusedMoeWeightScaleSupported=SimpleNamespace(
                CHANNEL=SimpleNamespace(value="channel"),
                GROUP=SimpleNamespace(value="group"),
            ),
            set_weight_attrs=set_attrs,
            _is_cpu=True,
            _maybe_copy_weight_view_before_h2d=lambda value: value,
        )
        self.scheme_type = load_methods(
            "python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_w4a8_int8_moe.py",
            "ModelSlimW4A8Int8MoE",
            {"create_weights"},
            self.namespace,
        )
        self.loader_type = load_methods(
            "python/sglang/srt/layers/moe/fused_moe_triton/layer.py",
            "FusedMoE",
            {"_load_model_weight_or_group_weight_scale", "_load_w2"},
            self.namespace,
        )
        self.kernel_type = load_methods(
            "python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py",
            "NPUW4A8Int8MoEMethod",
            {"_update_bias"},
            self.namespace,
        )

    def make_layer(self, tp):
        layer = torch.nn.Module()
        layer.moe_tp_size = tp
        scheme = self.scheme_type()
        scheme.weight_prefix = "w2"
        scheme.tp_size = 1  # Config instantiation leaves this default unchanged.
        scheme.is_per_channel_weight = True
        scheme.create_weights(
            layer,
            num_experts=1,
            hidden_size=4,
            intermediate_size_per_partition=32 // tp,
        )
        return layer

    def test_tp_shards_reconstruct_unsplit_compensation(self):
        # Distinct partials make copying the full bias on every rank observable.
        checkpoint = torch.arange(1, 65, dtype=torch.float32).reshape(4, 16)
        for tp in (1, 2, 4, 8, 16):
            with self.subTest(tp=tp):
                total = torch.zeros(4)
                for rank in range(tp):
                    layer = self.make_layer(tp)
                    self.assertEqual(layer.w2_scale_bias.shape, (1, 4, 16 // tp))
                    self.assertEqual(layer.w2_scale_bias.quant_method, "group")
                    loader = self.loader_type()
                    loader.moe_tp_size, loader.moe_tp_rank = tp, rank
                    loader.quant_config = None
                    loader.use_padded_loading = False
                    loader.use_presharded_weights = False
                    loader.use_triton_kernels = False
                    loader._load_model_weight_or_group_weight_scale(
                        expert_data=layer.w2_scale_bias.data[0],
                        shard_dim=1,
                        shard_id="w2",
                        loaded_weight=checkpoint,
                    )
                    expected = checkpoint[
                        :, rank * (16 // tp) : (rank + 1) * (16 // tp)
                    ]
                    torch.testing.assert_close(layer.w2_scale_bias[0], expected)
                    self.kernel_type._update_bias(layer, "w2")
                    total += layer.w2_scale_bias[0]
                torch.testing.assert_close(total, checkpoint.sum(dim=1))

    def test_invalid_tp_rejected(self):
        for tp in (3, 32):
            with self.subTest(tp=tp), self.assertRaisesRegex(ValueError, "divide 16"):
                self.make_layer(tp)

    def test_per_channel_weight_scales_stay_replicated(self):
        layer = self.make_layer(16)
        self.assertEqual(layer.w2_weight_scale.quant_method, "channel")
        self.assertEqual(layer.w2_weight_scale.shape, (1, 4, 1))


if __name__ == "__main__":
    unittest.main()
