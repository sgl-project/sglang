"""Regression test for EAGLE V1 speculative decoding + return_hidden_states.

The verify path slices `logits_output.hidden_states` down to accepted
positions via `accept_indices`, producing a flat
[sum(num_accept_per_req), hidden_dim] tensor. The downstream consumer in
`batch_result_processor.process_batch_result_decode` previously did
`hidden_states[i]`, which picks one flat row per step instead of the
per-request slice (see issue #26163). Prompts here repeat verbatim phrases
so the drafter lands long accept streaks and actually exercises the
multi-row slice.

EAGLE3 lands in the same `is_spec_v1` branch and uses the same slicing
arithmetic, though it returns concatenated aux hidden_states with a
different shape[-1] than plain EAGLE — explicit EAGLE3 coverage is a
follow-up. STANDALONE uses CaptureHiddenMode.NULL on target verify, so
its hidden_states is always None and this fix never fires for it.
"""

import unittest

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
    CustomTestCase,
)

register_cuda_ci(est_time=120, stage="extra-a", runner_config="1-gpu-large")


class TestEagleReturnHiddenStates(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # speculative_eagle_topk=8 (>1) forces spec V1 in speculative_hook.py
        # regardless of SGLANG_ENABLE_SPEC_V2 — no env-var management needed.
        cls.engine = sgl.Engine(
            model_path=DEFAULT_TARGET_MODEL_EAGLE,
            speculative_draft_model_path=DEFAULT_DRAFT_MODEL_EAGLE,
            speculative_algorithm="EAGLE",
            speculative_num_steps=5,
            speculative_eagle_topk=8,
            speculative_num_draft_tokens=64,
            enable_return_hidden_states=True,
            mem_fraction_static=0.7,
            cuda_graph_max_bs=8,
        )

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def test_eagle_with_return_hidden_states(self):
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
            self.assertEqual(len(hs), out["meta_info"]["completion_tokens"])
            self.assertGreater(len(hs[0]), 0)


if __name__ == "__main__":
    unittest.main()
