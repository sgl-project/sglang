import unittest

import torch

from sglang.srt.layers.linear import MergedColumnParallelLinear, QKVParallelLinear
from sglang.srt.layers.parameter import PerTensorScaleParameter
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestModelOptNvfp4(CustomTestCase):
    def _make_layer(self):
        return MergedColumnParallelLinear(
            input_size=16,
            output_sizes=[16, 16],
            bias=False,
            tp_rank=0,
            tp_size=1,
        )

    def _make_qkv_layer(self):
        return QKVParallelLinear(
            hidden_size=16,
            head_size=8,
            total_num_heads=2,
            total_num_kv_heads=2,
            bias=False,
            tp_rank=0,
            tp_size=1,
        )

    def test_fused_scalar_scale_load_fills_all_logical_slots(self):
        layer = self._make_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(2, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        layer.weight_loader_v2(scale, torch.tensor(0.25, dtype=torch.float32))

        torch.testing.assert_close(scale, torch.tensor([0.25, 0.25]))

    def test_fused_scalar_scale_load_rejects_non_scalar(self):
        layer = self._make_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(2, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        with self.assertRaisesRegex(ValueError, "Expected scalar scale"):
            layer.weight_loader_v2(scale, torch.tensor([0.25, 0.5]))

    def test_fused_qkv_scalar_scale_load_fills_all_logical_slots(self):
        layer = self._make_qkv_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(3, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        layer.weight_loader_v2(scale, torch.tensor(0.125, dtype=torch.float32))

        torch.testing.assert_close(scale, torch.tensor([0.125, 0.125, 0.125]))

    def test_explicit_shard_scale_loads_stay_independent(self):
        layer = self._make_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(2, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        layer.weight_loader_v2(scale, torch.tensor(0.25, dtype=torch.float32), 0)
        layer.weight_loader_v2(scale, torch.tensor(0.5, dtype=torch.float32), 1)

        torch.testing.assert_close(scale, torch.tensor([0.25, 0.5]))


class TestModelOptFp4Config(CustomTestCase):
    @staticmethod
    def _flat_config(input_group_size=16):
        return {
            "config_groups": {
                "group_0": {
                    "input_activations": {"group_size": input_group_size},
                    "weights": {"group_size": 16},
                }
            },
            "ignore": ["lm_head"],
            "quant_algo": "NVFP4",
        }

    def test_parses_flat_group_size(self):
        result = ModelOptFp4Config.from_config(self._flat_config())

        self.assertEqual(result.group_size, 16)

    def test_rejects_inconsistent_group_sizes(self):
        across_groups = self._flat_config()
        across_groups["config_groups"]["group_1"] = {"weights": {"group_size": 32}}

        for name, config in (
            ("within_group", self._flat_config(input_group_size=32)),
            ("across_groups", across_groups),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    ValueError, r"Inconsistent group_size values: \[16, 32\]"
                ):
                    ModelOptFp4Config.from_config(config)


if __name__ == "__main__":
    unittest.main()
