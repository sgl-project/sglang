"""Regression test for EAGLE3 speculative decoding + return_hidden_states.

EAGLE3 lands in the same ``is_spec_v1`` branch as plain EAGLE in
``batch_result_processor.process_batch_result_decode``, but returns
*concatenated aux hidden_states* from multiple decoder layers — so
``shape[-1]`` is wider than the model's hidden_dim, not equal to it. Plain
EAGLE returns single-layer hidden_states.

This test pins the per-request invariant ``len(hs) == completion_tokens`` on
EAGLE3's wider aux layout so future refactors of ``_fill_requests`` /
``accept_indices`` slicing don't silently break EAGLE3 (see issue #26163 and
PR #26217 which generalizes the spec-V1 fix to EAGLE3 transitively).

Verbatim-repeat prompts let the drafter land long accept streaks and actually
exercise the multi-row slice that triggered the original bug.
"""

import unittest

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
    CustomTestCase,
)

register_cuda_ci(est_time=120, stage="extra-a", runner_config="1-gpu-large")


class TestEagle3ReturnHiddenStates(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # speculative_eagle_topk=8 (>1) forces spec V1 in speculative_hook.py
        # regardless of SGLANG_ENABLE_SPEC_V2 — no env-var management needed.
        cls.engine = sgl.Engine(
            model_path=DEFAULT_TARGET_MODEL_EAGLE3,
            speculative_draft_model_path=DEFAULT_DRAFT_MODEL_EAGLE3,
            speculative_algorithm="EAGLE3",
            speculative_num_steps=5,
            speculative_eagle_topk=8,
            speculative_num_draft_tokens=64,
            enable_return_hidden_states=True,
            mem_fraction_static=0.7,
            cuda_graph_max_bs=8,
            dtype="float16",
        )

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def test_eagle3_with_return_hidden_states(self):
        prompts = [
            "Repeat: the quick brown fox the quick brown fox the quick brown fox",
            "Eeny meeny miny moe, eeny meeny miny moe, eeny meeny miny moe",
        ]
        outputs = self.engine.generate(
            prompts,
            sampling_params={"temperature": 0, "max_new_tokens": 32},
            return_hidden_states=True,
        )

        for out in outputs:
            hs = out["meta_info"]["hidden_states"]
            ct = out["meta_info"]["completion_tokens"]
            # Per-request invariant: one hidden_state row per emitted token.
            # Before PR #26217 this asserted len(hs) ≈ num_verify_steps
            # (e.g. 8 vs 32) because the spec-V1 path appended hidden_states[i]
            # one row per verify step instead of extending by the per-req slice.
            self.assertEqual(
                len(hs),
                ct,
                f"EAGLE3 hidden_states truncation: len(hs)={len(hs)}, "
                f"completion_tokens={ct} (see issue #26163)",
            )
            # EAGLE3 returns *concatenated* aux hidden_states; just sanity-check
            # the row is non-empty. The exact shape[-1] depends on which layers
            # the EAGLE3 draft was trained against.
            self.assertGreater(
                len(hs[0]),
                0,
                "EAGLE3 hidden_states row is empty; expected concatenated aux dim.",
            )


if __name__ == "__main__":
    unittest.main()
