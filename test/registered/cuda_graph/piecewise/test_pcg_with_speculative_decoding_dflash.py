"""Test piecewise CUDA graph coexisting with speculative decoding (DFLASH).

PCG handles prefill/extend path while DFlash needs target aux hidden states
from prefill to materialize draft KV cache. This verifies PCG captures that
path with the DFlash hidden-state variant enabled.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.pcg_spec_fixture import PCGSpecBase
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_TARGET_MODEL_DFLASH,
    CustomTestCase,
)

register_cuda_ci(est_time=80, stage="base-b", runner_config="1-gpu-small")


class TestPCGWithDFlash(PCGSpecBase, CustomTestCase):
    """PCG + DFLASH on Llama-3.1-8B-Instruct."""

    model = DEFAULT_TARGET_MODEL_DFLASH
    server_args = [
        "--trust-remote-code",
        "--attention-backend",
        "flashinfer",
        "--cuda-graph-backend-prefill",
        "tc_piecewise",
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        DEFAULT_DRAFT_MODEL_DFLASH,
        "--page-size",
        "1",
        "--max-running-requests",
        "64",
        # Keep headroom for the draft KV pool + piecewise cuda graph
        # private pools on 32GB CI cards.
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
