"""Extra: PCG coexistence with non-EAGLE3 speculative decoding variants.

EAGLE3 stays per-commit in the sibling file
test_pcg_with_speculative_decoding.py.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.pcg_spec_fixture import PCGSpecBase

register_cuda_ci(est_time=531, stage="extra-a", runner_config="2-gpu-large")


class TestPCGWithMTP(PCGSpecBase, unittest.TestCase):
    """PCG + MTP (NEXTN) on Qwen3.5-35B-A3B with FP8."""

    model = "Qwen/Qwen3.5-35B-A3B"
    server_args = [
        "--tp",
        "2",
        "--trust-remote-code",
        "--quantization",
        "fp8",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--enable-piecewise-cuda-graph",
        "--speculative-algorithm",
        "NEXTN",
        "--reasoning-parser",
        "qwen3",
    ]
    timeout_mult = 3
    max_tokens = 8192
    thinking_mode = "qwen3"
    accuracy_threshold = 0.75


class TestPCGWithSTANDALONE(PCGSpecBase, unittest.TestCase):
    """PCG + STANDALONE on Llama-3.1-8B-Instruct + Llama-3.2-1B-Instruct."""

    model = "meta-llama/Llama-3.1-8B-Instruct"
    server_args = [
        "--trust-remote-code",
        "--enforce-piecewise-cuda-graph",
        "--mem-fraction-static",
        "0.5",
        "--speculative-algorithm",
        "STANDALONE",
        "--speculative-draft-model-path",
        "meta-llama/Llama-3.2-1B-Instruct",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ]
    accuracy_threshold = 0.50


class TestPCGWithNGRAM(PCGSpecBase, unittest.TestCase):
    """PCG + NGRAM on Qwen2.5-Coder-7B-Instruct."""

    model = "Qwen/Qwen2.5-Coder-7B-Instruct"
    server_args = [
        "--trust-remote-code",
        "--enforce-piecewise-cuda-graph",
        "--speculative-algorithm",
        "NGRAM",
        "--speculative-num-draft-tokens",
        "16",
        "--cuda-graph-max-bs",
        "8",
        "--mem-fraction-static",
        "0.8",
    ]


if __name__ == "__main__":
    unittest.main()
