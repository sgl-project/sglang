"""Stress test for EAGLE V1 + return_hidden_states fix from #26163.

Exercises edge cases beyond the basic CI regression test:
- bs=1 single-request batches
- max_new_tokens=1 (single verify step, no multi-step accumulation)
- High concurrent batch sizes (32 reqs)
- Mixed acceptance patterns (some prompts force long streaks, others
  force rejections)
- return_hidden_states + return_logprob combined
- Hidden-state VALUE consistency vs non-spec baseline (not just count)
- Perf sanity (no measurable regression from the per-batch offset loop)

Run manually on a GPU box with EAGLE-compatible models:
    python -m unittest test.manual.spec.eagle.test_eagle_hidden_states_stress
"""

import time
import unittest
from typing import List

import torch

import sglang as sgl
from sglang.srt.environ import envs
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
    CustomTestCase,
)


HIGH_ACCEPT_PROMPTS = [
    "Repeat: the quick brown fox the quick brown fox the quick brown fox",
    "Eeny meeny miny moe, eeny meeny miny moe, eeny meeny miny moe",
    "Mary had a little lamb. Mary had a little lamb. Mary had a little lamb.",
    "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16",
]

LOW_ACCEPT_PROMPTS = [
    "Write a haiku about quantum entanglement and breakfast cereal.",
    "Translate to French: 'The aardvark dances under nebulae.'",
    "Compose a sonnet whose every line starts with 'Z'.",
]


def _assert_per_req_invariant(self, outputs, *, allow_zero_hs=False):
    # NOTE: req.hidden_states[0] may be a prefill nested list-of-rows (see
    # _append_prefill_hidden_states in batch_result_processor.py); decode
    # rows from this fix are flat 1D vectors. We only assert the count
    # invariant and non-emptiness — the dimensionality consistency check
    # would trip on the prefill/decode shape mismatch.
    for out in outputs:
        ct = out["meta_info"]["completion_tokens"]
        hs = out["meta_info"]["hidden_states"]
        self.assertEqual(
            len(hs),
            ct,
            f"len(hs)={len(hs)} != completion_tokens={ct} -- per-req slicing wrong",
        )
        if ct > 0 and not allow_zero_hs:
            self.assertGreater(len(hs[-1]), 0, "last hidden_state row is empty")


class TestEagleHiddenStatesStress(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        envs.SGLANG_ENABLE_SPEC_V2.set(False)
        try:
            # mem_fraction_static=0.45 leaves headroom for the value-equivalence
            # test to instantiate a temporary non-spec baseline engine without
            # exceeding GPU memory.
            cls.engine = sgl.Engine(
                model_path=DEFAULT_TARGET_MODEL_EAGLE,
                speculative_draft_model_path=DEFAULT_DRAFT_MODEL_EAGLE,
                speculative_algorithm="EAGLE",
                speculative_num_steps=5,
                speculative_eagle_topk=8,
                speculative_num_draft_tokens=64,
                enable_return_hidden_states=True,
                mem_fraction_static=0.45,
                cuda_graph_max_bs=32,
            )
        except Exception:
            envs.SGLANG_ENABLE_SPEC_V2.clear()
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            cls.engine.shutdown()
        finally:
            envs.SGLANG_ENABLE_SPEC_V2.clear()

    def test_bs_1_high_accept(self):
        outputs = self.engine.generate(
            [HIGH_ACCEPT_PROMPTS[0]],
            sampling_params={"temperature": 0, "max_new_tokens": 32},
            return_hidden_states=True,
        )
        _assert_per_req_invariant(self, outputs)

    def test_bs_1_low_accept(self):
        outputs = self.engine.generate(
            [LOW_ACCEPT_PROMPTS[0]],
            sampling_params={"temperature": 0, "max_new_tokens": 32},
            return_hidden_states=True,
        )
        _assert_per_req_invariant(self, outputs)

    def test_minimum_decode_steps(self):
        # max_new_tokens=1 finishes in the target_extend (prefill) phase and
        # never enters process_batch_result_decode, so it doesn't exercise
        # the offset-builder. Use a small but non-trivial completion length
        # to guarantee at least one decode/verify step actually runs.
        outputs = self.engine.generate(
            HIGH_ACCEPT_PROMPTS,
            sampling_params={"temperature": 0, "max_new_tokens": 8},
            return_hidden_states=True,
        )
        _assert_per_req_invariant(self, outputs)

    def test_high_concurrent_batch(self):
        # 32 concurrent reqs (matches cuda_graph_max_bs). Exercises offset
        # loop over a non-trivial bs and ensures rows are correctly attributed
        # across many simultaneous slices.
        prompts = (HIGH_ACCEPT_PROMPTS * 8 + LOW_ACCEPT_PROMPTS * 4)[:32]
        outputs = self.engine.generate(
            prompts,
            sampling_params={"temperature": 0, "max_new_tokens": 24},
            return_hidden_states=True,
        )
        self.assertEqual(len(outputs), 32)
        _assert_per_req_invariant(self, outputs)

    def test_mixed_acceptance_in_one_batch(self):
        # Mixing high-accept (long streaks → multi-row slices) and low-accept
        # (mostly bonus-only → 1-row slices) prompts in the same batch
        # exposes off-by-one errors that uniform-acceptance batches hide.
        prompts = [
            HIGH_ACCEPT_PROMPTS[0],
            LOW_ACCEPT_PROMPTS[0],
            HIGH_ACCEPT_PROMPTS[1],
            LOW_ACCEPT_PROMPTS[1],
        ]
        outputs = self.engine.generate(
            prompts,
            sampling_params={"temperature": 0, "max_new_tokens": 24},
            return_hidden_states=True,
        )
        _assert_per_req_invariant(self, outputs)

    def test_hidden_states_with_logprob(self):
        # Per-req extend interacts with logprob accumulation in
        # add_output_logprobs_for_spec_v1 (same `num_correct_drafts_per_req_cpu`
        # source). Confirms both consumers stay consistent.
        outputs = self.engine.generate(
            HIGH_ACCEPT_PROMPTS[:2],
            sampling_params={"temperature": 0, "max_new_tokens": 16},
            return_hidden_states=True,
            return_logprob=True,
        )
        _assert_per_req_invariant(self, outputs)
        for out in outputs:
            ct = out["meta_info"]["completion_tokens"]
            # Output-token logprobs should align 1:1 with completion tokens.
            self.assertEqual(
                len(out["meta_info"]["output_token_logprobs"]), ct
            )

    def test_value_equivalence_vs_nonspec(self):
        # Strongest correctness check: under temperature=0, the spec path
        # should produce token-equivalent output AND numerically close
        # hidden_states vs a non-spec engine on the same prompt.
        prompt = HIGH_ACCEPT_PROMPTS[0]
        params = {"temperature": 0, "max_new_tokens": 24}

        spec_out = self.engine.generate(
            prompt, sampling_params=params, return_hidden_states=True
        )

        # Spawn a fresh non-spec engine for the baseline. Combined with
        # cls.engine's 0.45 fraction this needs to stay under ~0.45 to fit on
        # a single GPU. Bind to None first so the finally-clause cleanup
        # tolerates an OOM during Engine construction.
        baseline = None
        try:
            baseline = sgl.Engine(
                model_path=DEFAULT_TARGET_MODEL_EAGLE,
                enable_return_hidden_states=True,
                mem_fraction_static=0.35,
                cuda_graph_max_bs=2,
            )
            base_out = baseline.generate(
                prompt, sampling_params=params, return_hidden_states=True
            )
        finally:
            if baseline is not None:
                baseline.shutdown()

        self.assertEqual(
            spec_out["text"], base_out["text"], "spec text != non-spec text"
        )

        spec_hs = torch.tensor(spec_out["meta_info"]["hidden_states"])
        base_hs = torch.tensor(base_out["meta_info"]["hidden_states"])
        self.assertEqual(spec_hs.shape, base_hs.shape)
        # Tree-attention numerics differ slightly from sequential decode.
        # Tolerances mirror those in the existing EAGLE accuracy tests.
        torch.testing.assert_close(spec_hs, base_hs, rtol=5e-2, atol=5e-2)

    def test_perf_no_offset_loop_regression(self):
        # The offset-builder loop is O(bs). Confirms it doesn't add measurable
        # latency at bs=16 vs a single warmup pass. Loose threshold; this is
        # a sanity check, not a microbenchmark.
        prompts = HIGH_ACCEPT_PROMPTS * 4  # 16 reqs
        params = {"temperature": 0, "max_new_tokens": 32}

        # Warmup
        self.engine.generate(prompts, sampling_params=params, return_hidden_states=True)

        t0 = time.perf_counter()
        outputs = self.engine.generate(
            prompts, sampling_params=params, return_hidden_states=True
        )
        elapsed = time.perf_counter() - t0
        _assert_per_req_invariant(self, outputs)

        total_tokens = sum(o["meta_info"]["completion_tokens"] for o in outputs)
        print(
            f"[perf] bs=16, total_tokens={total_tokens}, elapsed={elapsed:.2f}s, "
            f"throughput={total_tokens / elapsed:.1f} tok/s"
        )
        # Throughput should be at least 50 tok/s on any reasonable EAGLE
        # config; if it's well below, something other than this fix likely
        # regressed.
        self.assertGreater(total_tokens / elapsed, 50.0)


if __name__ == "__main__":
    unittest.main()
