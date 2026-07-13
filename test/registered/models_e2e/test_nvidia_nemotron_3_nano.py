import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.lm_eval_kit import LMEvalMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(
    est_time=190,
    stage="base-b",
    runner_config="2-gpu-large",
)

NEMOTRON_3_NANO_THINKING_ARGS = [
    "--trust-remote-code",
    "--tool-call-parser",
    "qwen3_coder",
    "--reasoning-parser",
    "deepseek-r1",
]


class TestNvidiaNemotron3Nano30BFP8(LMEvalMixin, DefaultServerBase):
    """Test Nemotron-3-Nano-30B FP8 model with lm-eval GSM8K evaluation."""

    model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
    model_config_name = "lm_eval_configs/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8.yaml"
    other_args = [
        "--tp-size",
        "2",
        # FlashInfer trtllm allreduce fusion on flashinfer 0.6.14 degrades
        # gsm8k for this model (strict-match 0.84 -> 0.78); the ground truth
        # was calibrated with fusion disabled. Keep it off until the fusion
        # numerics regression is resolved.
        "--enforce-disable-flashinfer-allreduce-fusion",
    ] + NEMOTRON_3_NANO_THINKING_ARGS


if __name__ == "__main__":
    unittest.main()
