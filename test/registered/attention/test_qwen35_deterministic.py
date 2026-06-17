"""
Usage:
cd test/srt
python3 -m unittest test_qwen35_deterministic.TestQwen35Fa3Deterministic
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_deterministic_utils import (
    COMMON_SERVER_ARGS,
    TestDeterministicBase,
)

register_cuda_ci(est_time=360, stage="extra-b", runner_config="4-gpu-h100")

QWEN35 = "Qwen/Qwen3.5-35B-A3B"


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestQwen35Fa3Deterministic(TestDeterministicBase):
    @classmethod
    def get_model(cls):
        return QWEN35

    @classmethod
    def get_server_args(cls):
        return list(COMMON_SERVER_ARGS) + [
            "--tp",
            "4",
            "--attention-backend",
            "fa3",
            "--skip-server-warmup",
            "--mamba-scheduler-strategy",
            "extra_buffer",
            "--enable-flashinfer-allreduce-fusion",
            "--tokenizer-worker-num",
            "6",
            "--mem-fraction-static",
            "0.8",
        ]


if __name__ == "__main__":
    unittest.main()
