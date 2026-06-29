"""Test breakable CUDA graph (BCG) coexisting with EAGLE3 speculative
decoding. Sibling of test_pcg_with_speculative_decoding.py — same
target/draft pair, only flips the prefill backend from tc_piecewise to
breakable. Verifies the draft-side BCG plumbing in PrefillCudaGraphRunner
stays wired (capture_hidden_mode for EAGLE, static_draft_hidden_states
buffer sized from the draft's fc input, EagleDraftInput at capture, and
the load_batch refresh).
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.pcg_spec_fixture import PCGSpecBase

register_cuda_ci(est_time=107, stage="base-b", runner_config="2-gpu-large")


class TestBCGWithEAGLE3(PCGSpecBase, unittest.TestCase):
    """BCG + EAGLE3 on Qwen3-30B-A3B-Instruct-2507."""

    model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    server_args = [
        "--tp",
        "2",
        "--trust-remote-code",
        "--cuda-graph-backend-prefill=breakable",
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
