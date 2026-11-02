"""End-to-end test of --cuda-graph-persistence with EAGLE3 speculative decoding.

Same native / SAVE / LOAD sequence and assertions as
test_cuda_graph_persistence.py, on Llama-3.1-8B-Instruct with its EAGLE3 draft:
the archive then holds the three decode-type graph kinds the EAGLE worker
captures (target_verify, draft_decode, draft_extend) instead of the target
decode graphs, and LOAD restores each kind. A small chain (3 steps, top-k 1, 4
draft tokens) keeps the run short; greedy outputs stay deterministic. Needs
the optional dependency `foundry` (sglang[foundry]); skipped otherwise.

Run:  python3 test/registered/cuda_graph/persistence/test_cuda_graph_persistence_spec.py
"""

import unittest

# Module import (not the class): unittest would collect an imported class and
# run the decode test here as well.
import test_cuda_graph_persistence as decode_e2e

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
)

# 1-gpu-large-foundry installs Foundry (scripts/ci/cuda/ci_install_foundry.sh); skips elsewhere.
register_cuda_ci(est_time=400, stage="extra-a", runner_config="1-gpu-large-foundry")


class TestFoundryGraphPersistenceSpec(decode_e2e.TestFoundryGraphPersistence):
    model = DEFAULT_TARGET_MODEL_EAGLE3
    decode_bs = [1, 2, 4, 8]
    decode_kinds = ["target_verify", "draft_decode", "draft_extend"]

    @classmethod
    def setUpClass(cls):
        cls.extra_server_args = [
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            DEFAULT_DRAFT_MODEL_EAGLE3,
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
        ]
        super().setUpClass()


if __name__ == "__main__":
    unittest.main(verbosity=3)
