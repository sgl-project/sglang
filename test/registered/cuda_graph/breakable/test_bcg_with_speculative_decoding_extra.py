"""Extra: BCG coexistence with MTP (NEXTN) speculative decoding.

EAGLE3 lives in the sibling file test_bcg_with_speculative_decoding.py.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.pcg_spec_fixture import PCGSpecBase

register_cuda_ci(est_time=240, stage="weekly", runner_config="4-gpu-h100")


class TestBCGWithMTP(PCGSpecBase, unittest.TestCase):
    """BCG (default prefill backend) + MTP (NEXTN) on Qwen3.5-35B-A3B, FP8."""

    model = "Qwen/Qwen3.5-35B-A3B"
    server_args = [
        "--tp",
        "2",
        "--trust-remote-code",
        "--quantization",
        "fp8",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--speculative-algorithm",
        "NEXTN",
        "--reasoning-parser",
        "qwen3",
    ]
    timeout_mult = 3
    max_tokens = 8192
    thinking_mode = "qwen3"
    accuracy_threshold = 0.75


if __name__ == "__main__":
    unittest.main()
