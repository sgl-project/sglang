"""Test the breakable CUDA graph (BCG) backend coexisting with DFLASH.

Sibling of test_pcg_with_speculative_decoding_dflash.py — same target/draft
pair, only flips the prefill backend from tc_piecewise to breakable. Exercises
the DFlash aux stash protocol under BCG replay: because the captured body skips
forward()'s Python, the runner re-arms the fused aux stash from its static
buffer (at the padded body-output width) before the eager logits tail pops it.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.pcg_spec_fixture import PCGSpecBase
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_TARGET_MODEL_DFLASH,
    CustomTestCase,
)

register_cuda_ci(est_time=531, suite="nightly-1-gpu", nightly=True)


class TestBCGWithDFlash(PCGSpecBase, CustomTestCase):
    """Breakable prefill CUDA graph + DFLASH on Llama-3.1-8B-Instruct."""

    model = DEFAULT_TARGET_MODEL_DFLASH
    server_args = [
        "--trust-remote-code",
        "--attention-backend",
        "flashinfer",
        "--cuda-graph-backend-prefill",
        "breakable",
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        DEFAULT_DRAFT_MODEL_DFLASH,
        "--page-size",
        "1",
        "--max-running-requests",
        "64",
        "--mem-fraction-static",
        "0.7",
        "--cuda-graph-bs-decode",
        *[str(i) for i in range(1, 65)],
    ]
    server_env = {"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"}
    accuracy_threshold = 0.75
    speedup_threshold = 2.8


if __name__ == "__main__":
    unittest.main()
