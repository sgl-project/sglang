"""Test piecewise CUDA graph coexisting with speculative decoding (EAGLE3).

PCG handles prefill/extend path while speculative decoding (EAGLE3) uses
decode CUDA graphs. This test verifies they don't interfere with each
other. MTP / STANDALONE / NGRAM variants moved to the sibling file
test_pcg_with_speculative_decoding_extra.py.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.pcg_spec_fixture import PCGSpecBase

register_cuda_ci(est_time=531, stage="stage-b", runner_config="2-gpu-large")


class TestPCGWithEAGLE3(PCGSpecBase, unittest.TestCase):
    """PCG + EAGLE3 on Qwen3-30B-A3B-Instruct-2507."""

    model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    server_args = [
        "--tp",
        "2",
        "--trust-remote-code",
        "--enforce-piecewise-cuda-graph",
        "--mem-fraction-static",
        "0.6",
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        "lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "4",
        "--speculative-num-draft-tokens",
        "8",
    ]
    timeout_mult = 3
    server_env = {"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"}
    accuracy_threshold = 0.75


if __name__ == "__main__":
    unittest.main()
