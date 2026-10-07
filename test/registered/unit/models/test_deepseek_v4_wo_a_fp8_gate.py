"""Unit tests for the DeepSeek-V4 FP8 wo_a GEMM selection."""

import os
import unittest
from unittest.mock import patch

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _config(ignored_layers=None):
    return Fp8Config(
        is_checkpoint_fp8_serialized=True,
        ignored_layers=ignored_layers,
        weight_block_size=[128, 128],
    )


class TestWoAFp8GemmEnabled(CustomTestCase):
    def _enabled(self, ignored_layers=None):
        with patch.object(deepseek_v4, "_FP8_WO_A_GEMM", True):
            return deepseek_v4.wo_a_fp8_gemm_enabled(_config(ignored_layers))

    def test_quantized_wo_a_uses_fp8_gemm(self):
        self.assertTrue(self._enabled())
        self.assertTrue(
            self._enabled(
                [
                    "model.layers.3.self_attn.wq_a",
                    "model.layers.3.mlp",
                    "model.layers.12.self_attn.compressor",
                ]
            )
        )

    def test_excluded_wo_a_keeps_bf16(self):
        for ignored in (
            ["model.layers.0.self_attn.wo_a"],
            ["layers.42.self_attn.wo_a"],
            ["model.layers.7"],
            ["model.layers.7.self_attn"],
            ["model.decoder.self_attn.wo_a"],
            ["wo_a"],
        ):
            with self.subTest(ignored=ignored):
                self.assertFalse(self._enabled(ignored))

    def test_env_excluded_wo_a_keeps_bf16(self):
        with patch.dict(
            os.environ,
            {"SGLANG_FP8_IGNORED_LAYERS": "model.layers.5.self_attn.wo_a"},
        ):
            self.assertFalse(self._enabled())


if __name__ == "__main__":
    unittest.main()
