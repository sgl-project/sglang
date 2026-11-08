"""Nightly end-to-end test of --cuda-graph-persistence with full-backend prefill
graphs on a dense model (Qwen3-1.7B, real weights, FlashInfer). Same native /
SAVE / LOAD sequence and assertions as test_cuda_graph_persistence.py, which
covers the prefill graphs together with EAGLE3 on every commit; this dense case
runs nightly because it needs the weights and adds no graph kind. Needs the
optional dependency sglang[foundry].

Run:  python3 test/registered/e2e/cuda_graph/persistence/test_cuda_graph_persistence_nightly.py
"""

import unittest

# Module import (not the class): unittest would collect an imported class and
# run the per-commit tests here as well.
import test_cuda_graph_persistence as decode_e2e

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=281, stage="nightly", runner_config="1-gpu-large")


class TestFoundryGraphPersistencePrefill(decode_e2e.TestFoundryGraphPersistence):
    model = "Qwen/Qwen3-1.7B"
    # Tokens per bucket. A prefill replays only when bucket padding is at most 2x,
    # so the smallest bucket must cover the 5-12 token prompts.
    prefill_bs = [8, 16, 32, 64]
    # Full-backend prefill needs capture-stable EXTEND metadata, which flashinfer provides.
    attention_backend = "flashinfer"


if __name__ == "__main__":
    unittest.main(verbosity=3)
