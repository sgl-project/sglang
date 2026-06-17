"""Regression test for NGRAM speculative decoding + return_hidden_states (issue #26163).

Complements the EAGLE coverage in test/registered/spec/eagle/. NGRAM is corpus-based
and lands long verbatim n-gram matches, so it lights up high-accept-rate verify
windows that exercise the strided spec-V2 hidden_states slicing.
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
            cuda_graph_max_bs_decode=8,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine is not None:
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
