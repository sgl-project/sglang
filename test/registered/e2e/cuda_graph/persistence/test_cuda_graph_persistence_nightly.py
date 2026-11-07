"""Nightly end-to-end tests of --cuda-graph-persistence in two niche configurations.

Same native / SAVE / LOAD sequence and assertions as
test_cuda_graph_persistence.py: full-backend prefill graphs plus decode graphs
on Qwen3-1.7B, and EAGLE3 speculative decoding on Qwen3-30B-A3B-FP8. They run
nightly because both configurations are niche and a failure from a Foundry
dependency change is not attributable to a single PR. Needs the optional
dependency sglang[foundry].

EAGLE3: the archive holds the three decode-type graph kinds the EAGLE worker
captures (target_verify, draft_decode, draft_extend) instead of the target
decode graphs, and LOAD restores each kind. A small chain (3 steps, top-k 1, 4
draft tokens) keeps the run short; greedy outputs stay deterministic. Dummy
weights (--load-format dummy, target and draft) keep CI from downloading 31 GB:
the graphs, their restore and the target's outputs are exercised, draft
acceptance is not (a random draft accepts no token).

Run:  python3 test/registered/e2e/cuda_graph/persistence/test_cuda_graph_persistence_nightly.py
"""

import unittest

# Module import (not the class): unittest would collect an imported class and
# run the decode test here as well.
import test_cuda_graph_persistence as decode_e2e

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=521, stage="nightly", runner_config="1-gpu-large")

# Ungated target and draft (the default EAGLE3 pair's Llama target is gated).
_TARGET = "Qwen/Qwen3-30B-A3B-FP8"
_DRAFT = "Tengyunw/qwen3_30b_moe_eagle3"


class TestFoundryGraphPersistencePrefill(decode_e2e.TestFoundryGraphPersistence):
    model = "Qwen/Qwen3-1.7B"
    # Tokens per bucket. A prefill replays only when bucket padding is at most 2x,
    # so the smallest bucket must cover the 5-12 token prompts.
    prefill_bs = [8, 16, 32, 64]
    # Full-backend prefill needs capture-stable EXTEND metadata, which flashinfer provides.
    attention_backend = "flashinfer"


class TestFoundryGraphPersistenceSpec(decode_e2e.TestFoundryGraphPersistence):
    model = _TARGET
    attention_backend = "flashinfer"
    decode_bs = [1, 2, 4, 8]
    decode_kinds = ["target_verify", "draft_decode", "draft_extend"]
    extra_server_args = [
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        _DRAFT,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--load-format",
        "dummy",
    ]


if __name__ == "__main__":
    unittest.main(verbosity=3)
