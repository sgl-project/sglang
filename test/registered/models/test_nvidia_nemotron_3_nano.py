import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.gsm8k_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

# NVIDIA Nemotron-3 Nano 30B model tests (CUDA only)
# GSM8K evaluation for BF16 and FP8 variants

register_cuda_ci(est_time=180, suite="stage-b-test-large-2-gpu")


class TestNvidiaNemotron3Nano30BBF16(GSM8KMixin, DefaultServerBase):
    """Test Nemotron-3-Nano-30B BF16 model with GSM8K accuracy evaluation."""

    model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    gsm8k_accuracy_thres = 0.45
    other_args = [
        "--trust-remote-code",
        "--tp-size",
        "2",
        "--tool-call-parser",
        "qwen3_coder",
        "--reasoning-parser",
        "deepseek-r1",
    ]


class TestNvidiaNemotron3Nano30BFP8(GSM8KMixin, DefaultServerBase):
    """Test Nemotron-3-Nano-30B FP8 model with GSM8K accuracy evaluation."""

    model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
    gsm8k_accuracy_thres = 0.30
    other_args = [
        "--trust-remote-code",
        "--tp-size",
        "2",
        "--tool-call-parser",
        "qwen3_coder",
        "--reasoning-parser",
        "deepseek-r1",
    ]


if __name__ == "__main__":
    unittest.main()
