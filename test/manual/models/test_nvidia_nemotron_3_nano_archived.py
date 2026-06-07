"""Archived test classes split out of test/registered/models_e2e/test_nvidia_nemotron_3_nano.py.

Originally registered with `register_cuda_ci(...)`. Moved here as part of
the per-commit pruning effort to keep the code reachable manually.
Run with `python3 test/manual/models/test_nvidia_nemotron_3_nano_archived.py`.
"""

import unittest

from sglang.srt.utils import is_sm80_supported, is_sm90_supported
from sglang.test.kits.lm_eval_kit import LMEvalMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

NEMOTRON_3_NANO_THINKING_ARGS = [
    "--trust-remote-code",
    "--tool-call-parser",
    "qwen3_coder",
    "--reasoning-parser",
    "deepseek-r1",
]


class TestNvidiaNemotron3Nano30BBF16(LMEvalMixin, DefaultServerBase):
    """Test Nemotron-3-Nano-30B BF16 model with lm-eval GSM8K evaluation."""

    model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    model_config_name = "lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml"
    other_args = [
        "--tp-size",
        "2",
    ] + NEMOTRON_3_NANO_THINKING_ARGS


class TestNvidiaNemotron3Nano30BBF16FlashInfer(LMEvalMixin, DefaultServerBase):
    """Test Nemotron-3-Nano-30B BF16 model with lm-eval GSM8K evaluation using flashinfer mamba backend."""

    model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    model_config_name = "lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml"
    other_args = [
        "--tp-size",
        "2",
        "--mamba-backend",
        "flashinfer",
    ] + NEMOTRON_3_NANO_THINKING_ARGS


@unittest.skip("Skip, test pass locally but compiling takes too long in CI")
@unittest.skipIf(
    not (is_sm80_supported() or is_sm90_supported()),
    "NVFP4 Marlin fallback test requires CUDA SM8X/SM9X",
)
class TestNvidiaNemotron3Nano30BNVFP4Marlin(LMEvalMixin, DefaultServerBase):
    """Test Nemotron-3-Nano-30B NVFP4 model with the Marlin path."""

    model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
    model_config_name = "lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4.yaml"
    other_args = [
        "--tp-size",
        "1",
        "--quantization",
        "modelopt_fp4",
        "--fp4-gemm-backend",
        "marlin",
        "--moe-runner-backend",
        "marlin",
    ] + NEMOTRON_3_NANO_THINKING_ARGS


if __name__ == "__main__":
    unittest.main()
