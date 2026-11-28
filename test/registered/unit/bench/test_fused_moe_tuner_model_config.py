"""CPU regressions for fused MoE tuner quantization metadata."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _load_common_utils():
    # Import the complete helper, stubbing only optional runtime dependencies.
    triton_utils = "sglang.srt.layers.moe.moe_runner.triton_utils"
    stubs = {
        "torch": SimpleNamespace(dtype=object, bfloat16="bfloat16"),
        "sglang.srt.hf_transformers": SimpleNamespace(get_config=Mock()),
        f"{triton_utils}.fused_moe": SimpleNamespace(get_config_dtype_str=Mock()),
        f"{triton_utils}.fused_moe_triton_config": SimpleNamespace(
            get_config_file_name=Mock()
        ),
        "sglang.srt.utils": SimpleNamespace(
            is_hip=lambda: False, is_gfx95_supported=lambda: False
        ),
    }
    source = (
        Path(__file__).resolve().parents[4]
        / "benchmark/kernels/fused_moe_triton/common_utils.py"
    )
    spec = importlib.util.spec_from_file_location("tuner_common_utils", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class TestFusedMoeTunerModelConfig(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.common_utils = _load_common_utils()

    def _block_shape(self, quantization_config):
        config = SimpleNamespace(
            architectures=["Qwen3MoeForCausalLM"],
            hidden_size=2048,
            moe_intermediate_size=768,
            num_experts=128,
            num_experts_per_tok=8,
        )
        if quantization_config is not None:
            config.quantization_config = quantization_config
        with patch.object(self.common_utils, "get_config", return_value=config):
            return self.common_utils.get_model_config("test-model", tp_size=2)[
                "block_shape"
            ]

    def _compressed_shape(self, weights):
        return self._block_shape({"config_groups": {"group_0": {"weights": weights}}})

    def test_real_nullable_channel_recipe(self):
        # Relevant weight fields from:
        # https://huggingface.co/RedHatAI/Qwen3-30B-A3B-FP8-dynamic/raw/main/config.json
        self.assertIsNone(
            self._compressed_shape(
                {
                    "strategy": "channel",
                    "group_size": None,
                    "block_structure": None,
                    "type": "float",
                    "num_bits": 8,
                    "dynamic": False,
                }
            )
        )

    def test_non_grouped_and_missing_metadata(self):
        for weights in (
            {},
            {"strategy": "channel"},
            {"strategy": "tensor"},
            {"strategy": "channel", "group_size": 128},
            {"strategy": "tensor", "group_size": 128},
        ):
            with self.subTest(weights=weights):
                self.assertIsNone(self._compressed_shape(weights))
        for size in (None, -1, 0, -2, True, False, "128", 128.0):
            with self.subTest(group_size=size):
                self.assertIsNone(self._compressed_shape({"group_size": size}))
        for config in (None, {}, {"config_groups": {}}, {"config_groups": {"g": {}}}):
            with self.subTest(config=config):
                self.assertIsNone(self._block_shape(config))

    def test_positive_groups_without_a_strategy_whitelist(self):
        for strategy in (None, "group", "tensor_group", "future_group"):
            for size in (16, 128):
                with self.subTest(strategy=strategy, group_size=size):
                    self.assertEqual(
                        self._compressed_shape(
                            {"strategy": strategy, "group_size": size}
                        ),
                        [0, size],
                    )
        self.assertEqual(self._compressed_shape({"group_size": 64}), [0, 64])

    def test_block_and_top_level_dimensions(self):
        for shape in ([128, 128], [64, 128], [1, 32]):
            with self.subTest(shape=shape):
                self.assertEqual(self._block_shape({"weight_block_size": shape}), shape)
                self.assertEqual(
                    self._compressed_shape(
                        {
                            "strategy": "block",
                            "block_structure": shape,
                            "group_size": 64,
                        }
                    ),
                    shape,
                )

    def test_first_config_group_keeps_precedence(self):
        for weights, expected in (
            ({"strategy": "channel", "group_size": None}, None),
            ({"strategy": "group", "group_size": 64}, [0, 64]),
            ({"strategy": "block", "block_structure": [64, 128]}, [64, 128]),
        ):
            with self.subTest(weights=weights):
                self.assertEqual(
                    self._block_shape(
                        {
                            "weight_block_size": [128, 128],
                            "config_groups": {
                                "first": {"weights": weights},
                                "second": {"weights": {"group_size": 32}},
                            },
                        }
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
