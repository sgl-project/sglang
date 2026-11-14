"""Decoder configs must produce every LoRA projection shape without a GPU."""

import importlib.util
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from transformers import Gemma4Config, Gemma4TextConfig, LlamaConfig, Qwen2Config

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_TUNER_PATH = (
    Path(__file__).resolve().parents[4]
    / "benchmark/kernels/lora_csgmv/tune_lora_csgmv.py"
)


class TestCSGMVTunerDims(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # Only GPU execution imports are replaced. Load the real tuner and use
        # real AutoConfig deserialization from local config files below.
        imports = {
            "triton": {},
            "sglang.kernels.ops.lora.dense.chunked_sgmv_expand": {
                "_chunked_lora_expand_kernel": None
            },
            "sglang.kernels.ops.lora.dense.chunked_sgmv_shrink": {
                "_chunked_lora_shrink_kernel": None
            },
            "sglang.kernels.ops.lora.dense.lora_tuning_config": {
                "DEFAULT_EXPAND_CONFIG": {},
                "DEFAULT_SHRINK_CONFIG": {},
                "get_lora_config_file_name": None,
            },
            "sglang.srt.lora.utils": {"LoRABatchInfo": object},
        }
        stubs = {}
        for name, attrs in imports.items():
            module = types.ModuleType(name)
            module.__dict__.update(attrs)
            stubs[name] = module
        spec = importlib.util.spec_from_file_location("csgmv_tuner_dims", _TUNER_PATH)
        cls.tuner = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", stubs):
            spec.loader.exec_module(cls.tuner)

    def _from_config(self, config):
        with tempfile.TemporaryDirectory() as path:
            config.save_pretrained(path)
            return self.tuner.get_model_dims(Namespace(model=path))

    def test_gemma4_nested_config_keeps_both_attention_layouts(self):
        """The wrapper has no hidden_size; its decoder has two KV/head layouts.

        Dropping either layout leaves some projection shapes untuned, while
        repeated sliding layers must not duplicate the dimension list.
        """
        config = Gemma4Config(
            text_config=Gemma4TextConfig(
                hidden_size=5376,
                intermediate_size=21504,
                num_hidden_layers=6,
                num_attention_heads=32,
                num_key_value_heads=16,
                head_dim=128,
                global_head_dim=256,
                num_global_key_value_heads=8,
                attention_k_eq_v=True,
            )
        )
        self.assertEqual(
            self._from_config(config),
            [
                ("qkv", 5376, 8192, 3, [0, 4096, 6144, 8192]),
                ("o_proj", 4096, 5376, 1, [0, 5376]),
                ("qkv", 5376, 12288, 3, [0, 8192, 10240, 12288]),
                ("o_proj", 8192, 5376, 1, [0, 5376]),
                ("gate_up", 5376, 43008, 2, [0, 21504, 43008]),
                ("down_proj", 21504, 5376, 1, [0, 5376]),
            ],
        )

    def test_flat_configs_preserve_grouped_query_projection_shapes(self):
        # Qwen2 lacks head_dim, so the tuner must derive it from hidden_size.
        for config_class in (LlamaConfig, Qwen2Config):
            with self.subTest(config=config_class.__name__):
                config = config_class(
                    hidden_size=1024,
                    intermediate_size=4096,
                    num_attention_heads=8,
                    num_key_value_heads=2,
                )
                self.assertEqual(
                    self._from_config(config),
                    [
                        ("qkv", 1024, 1536, 3, [0, 1024, 1280, 1536]),
                        ("o_proj", 1024, 1024, 1, [0, 1024]),
                        ("gate_up", 1024, 8192, 2, [0, 4096, 8192]),
                        ("down_proj", 4096, 1024, 1, [0, 1024]),
                    ],
                )

    def test_manual_dimensions_preserve_defaults_and_odd_qkv_width(self):
        for qkv, intermediate, expected in (
            (None, None, (32, 16, 24, 24)),
            (33, 20, (33, 16, 24, 20)),
        ):
            with self.subTest(qkv=qkv, intermediate=intermediate):
                args = Namespace(
                    model=None,
                    hidden_size=8,
                    qkv_output_dim=qkv,
                    intermediate_size=intermediate,
                )
                width, q, qk, mlp = expected
                self.assertEqual(
                    self.tuner.get_model_dims(args),
                    [
                        ("qkv", 8, width, 3, [0, q, qk, width]),
                        ("o_proj", q, 8, 1, [0, 8]),
                        ("gate_up", 8, 2 * mlp, 2, [0, mlp, 2 * mlp]),
                        ("down_proj", mlp, 8, 1, [0, 8]),
                    ],
                )


if __name__ == "__main__":
    unittest.main()
