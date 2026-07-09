"""Tests for _get_and_verify_dtype auto-resolution.

Hunyuan V3 checkpoints (tencent/Hy3, Hy3-FP8, Hy3-preview) ship without a
dtype in config.json, which the generic rule treats as a float32 checkpoint
and downcasts to float16. They are bf16 models, so `hy_v3` must resolve to
bfloat16 (like the existing gemma special case).
"""

import unittest

import torch

from sglang.srt.configs.model_config import _get_and_verify_dtype
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAutoDtypeResolution(CustomTestCase):
    def test_hy_v3_without_config_dtype_resolves_to_bf16(self):
        config = {"model_type": "hy_v3"}
        self.assertEqual(_get_and_verify_dtype(config, "auto"), torch.bfloat16)

    def test_hy_v3_explicit_dtype_is_honored(self):
        config = {"model_type": "hy_v3"}
        self.assertEqual(_get_and_verify_dtype(config, "float16"), torch.float16)

    def test_generic_model_without_config_dtype_resolves_to_fp16(self):
        config = {"model_type": "llama"}
        self.assertEqual(_get_and_verify_dtype(config, "auto"), torch.float16)

    def test_config_dtype_takes_precedence_over_special_case(self):
        config = {"model_type": "hy_v3", "torch_dtype": "bfloat16"}
        self.assertEqual(_get_and_verify_dtype(config, "auto"), torch.bfloat16)

    def test_gemma_without_config_dtype_resolves_to_bf16(self):
        config = {"model_type": "gemma2"}
        self.assertEqual(_get_and_verify_dtype(config, "auto"), torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
