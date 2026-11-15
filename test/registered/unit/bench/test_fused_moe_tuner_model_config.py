"""CPU regressions for MoE tuner quantization metadata and search filtering."""

import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_TUNER_DIR = Path(__file__).resolve().parents[4] / "benchmark/kernels/fused_moe_triton"


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
    spec = importlib.util.spec_from_file_location(
        "tuner_common_utils", _TUNER_DIR / "common_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


def _filter_search_space(block_shape):
    # Execute the tuner's actual filter without initializing Ray or GPU workers.
    source = _TUNER_DIR / "tuning_fused_moe_triton.py"
    main = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    block_filter = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "block_shape is not None"
    )
    namespace = {
        "block_shape": block_shape,
        "search_space": [{"BLOCK_SIZE_K": size} for size in (64, 128, 256)],
    }
    exec(
        compile(ast.Module(body=[block_filter], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return [config["BLOCK_SIZE_K"] for config in namespace["search_space"]]


class TestFusedMoeTunerModelConfig(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.common_utils = _load_common_utils()

    def _model_config(self, quantization_config):
        # Model dimensions from RedHatAI/Qwen3-30B-A3B-FP8-dynamic.
        config = SimpleNamespace(
            architectures=["Qwen3MoeForCausalLM"],
            hidden_size=2048,
            moe_intermediate_size=768,
            num_experts=128,
            num_experts_per_tok=8,
            num_hidden_layers=48,
        )
        if quantization_config is not None:
            config.quantization_config = quantization_config
        with patch.object(self.common_utils, "get_config", return_value=config):
            return self.common_utils.get_model_config("test-model", tp_size=2)

    def _compressed_config(self, weights):
        return self._model_config({"config_groups": {"group_0": {"weights": weights}}})

    def test_real_nullable_channel_recipe_does_not_filter_blocks(self):
        # Weight recipe copied from:
        # https://huggingface.co/RedHatAI/Qwen3-30B-A3B-FP8-dynamic/raw/main/config.json
        result = self._compressed_config(
            {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": None,
                "num_bits": 8,
                "observer": "mse",
                "observer_kwargs": {},
                "strategy": "channel",
                "symmetric": True,
                "type": "float",
            }
        )
        self.assertEqual(_filter_search_space(result["block_shape"]), [64, 128, 256])
        self.assertIsNone(result["block_shape"])
        self.assertEqual(result["num_experts"], 128)
        self.assertEqual(result["shard_intermediate_size"], 768)

    def test_non_grouped_and_missing_metadata(self):
        for weights in (
            {},
            {"group_size": None},
            {"group_size": -1},
            {"strategy": None, "group_size": -1},
            {"strategy": "channel"},
            {"strategy": "channel", "group_size": -1},
            {"strategy": "tensor"},
            {"strategy": "tensor", "group_size": None},
        ):
            with self.subTest(weights=weights):
                self.assertIsNone(self._compressed_config(weights)["block_shape"])
        for quantization_config in (
            None,
            {},
            {"config_groups": {}},
            {"config_groups": {"g": {}}},
        ):
            with self.subTest(quantization_config=quantization_config):
                self.assertIsNone(
                    self._model_config(quantization_config)["block_shape"]
                )

    def test_grouped_int4_including_legacy_inferred_strategy(self):
        for strategy in ("group", None):
            for group_size in (32, 64, 128):
                weights = {"type": "int", "num_bits": 4, "group_size": group_size}
                if strategy is not None:
                    weights["strategy"] = strategy
                with self.subTest(weights=weights):
                    shape = self._compressed_config(weights)["block_shape"]
                    self.assertEqual(shape, [0, group_size])
                    expected = {32: [], 64: [64], 128: [64, 128]}[group_size]
                    self.assertEqual(_filter_search_space(shape), expected)
        self.assertEqual(
            self._compressed_config({"strategy": None, "group_size": 128})[
                "block_shape"
            ],
            [0, 128],
        )

    def test_block_fp8_dimensions(self):
        for shape in ([128, 128], [64, 128], [1, 32]):
            with self.subTest(shape=shape):
                self.assertEqual(
                    self._model_config({"weight_block_size": shape})["block_shape"],
                    shape,
                )
                self.assertEqual(
                    self._compressed_config(
                        {
                            "strategy": "block",
                            "block_structure": shape,
                            "group_size": None,
                        }
                    )["block_shape"],
                    shape,
                )

    def test_config_groups_keep_existing_precedence_over_top_level_blocks(self):
        for weights, expected in (
            ({"strategy": "channel", "group_size": None}, None),
            ({"strategy": "group", "group_size": 64}, [0, 64]),
            ({"strategy": "block", "block_structure": [64, 128]}, [64, 128]),
        ):
            with self.subTest(weights=weights):
                result = self._model_config(
                    {
                        "weight_block_size": [128, 128],
                        "config_groups": {"g": {"weights": weights}},
                    }
                )
                self.assertEqual(result["block_shape"], expected)

    def test_rejects_invalid_group_sizes(self):
        for strategy in ("group", None):
            for size in (0, -2, True, False, "128", 128.0, -1.0):
                with self.subTest(strategy=strategy, size=size):
                    with self.assertRaisesRegex(ValueError, "group_size"):
                        self._compressed_config(
                            {"strategy": strategy, "group_size": size}
                        )
        for weights in (
            {"strategy": "group"},
            {"strategy": "group", "group_size": None},
            {"strategy": "group", "group_size": -1},
        ):
            with self.subTest(weights=weights):
                with self.assertRaisesRegex(ValueError, "group_size"):
                    self._compressed_config(weights)

    def test_rejects_invalid_block_dimensions(self):
        for shape in (
            None,
            [],
            [128],
            [128, None],
            [0, 128],
            [-1, 128],
            [True, 128],
            [128.0, 128],
            [128, 128, 128],
        ):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "block_structure"):
                    self._compressed_config(
                        {"strategy": "block", "block_structure": shape}
                    )

    def test_rejects_unsupported_strategy_instead_of_assuming_groups(self):
        with self.assertRaisesRegex(ValueError, "strategy"):
            self._compressed_config({"strategy": "unknown", "group_size": 128})


if __name__ == "__main__":
    unittest.main()
