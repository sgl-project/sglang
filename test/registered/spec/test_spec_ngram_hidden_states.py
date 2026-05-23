"""Regression test for NGRAM speculative decoding + return_hidden_states.

The verify path slices `logits_output.hidden_states` down to accepted
positions in `NgramVerifyInput._fill_requests`. The slice was previously
gated by a bare truthiness check that raised on multi-row tensors
(see issue #26131). Prompts here repeat verbatim n-grams so the drafter
lands long accept streaks and actually exercises the multi-row slice.
"""

import unittest

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
)

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")


class TestNgramReturnHiddenStates(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            enable_return_hidden_states=True,
            speculative_algorithm="NGRAM",
            speculative_num_draft_tokens=16,
            mem_fraction_static=0.7,
            cuda_graph_max_bs=8,
        )

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def test_ngram_with_return_hidden_states(self):
        prompts = [
            "Repeat the phrase the quick brown fox five times: the quick brown fox",
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
