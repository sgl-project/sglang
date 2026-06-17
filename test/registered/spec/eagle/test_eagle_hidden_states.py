"""Regression test for issue #26163: return_hidden_states under EAGLE spec V2."""

import unittest

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
    CustomTestCase,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-large")


class TestEagleReturnHiddenStates(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = sgl.Engine(
            model_path=DEFAULT_TARGET_MODEL_EAGLE,
            speculative_algorithm="EAGLE",
            speculative_draft_model_path=DEFAULT_DRAFT_MODEL_EAGLE,
            speculative_num_steps=3,
            speculative_eagle_topk=4,
            speculative_num_draft_tokens=8,
            enable_return_hidden_states=True,
            mem_fraction_static=0.7,
            attention_backend="triton",
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine is not None:
            cls.engine.shutdown()

    def test_hidden_states_length_matches_completion(self):
        # Two prompts to exercise cross-request stride aliasing.
        prompts = [
            "Repeat: the quick brown fox the quick brown fox the quick brown fox",
            "Count down from ten: ten nine eight",
        ]
        max_new_tokens = 32
        outputs = self.engine.generate(
            prompts,
            sampling_params={"temperature": 0, "max_new_tokens": max_new_tokens},
            return_hidden_states=True,
        )

        for out in outputs:
            meta = out["meta_info"]
            hs = meta["hidden_states"]
            ct = meta["completion_tokens"]
            self.assertEqual(
                len(hs),
                ct,
                f"len(hidden_states)={len(hs)} but completion_tokens={ct}",
            )
            # hs[0] is the prefill block (List[List[float]]); hs[1:] are decode rows.
            decode_rows = hs[1:]
            self.assertGreater(len(decode_rows), 0)
            hidden_dim = len(decode_rows[0])
            self.assertGreater(hidden_dim, 0)
            for row in decode_rows:
                self.assertIsInstance(row, list)
                self.assertEqual(len(row), hidden_dim)


if __name__ == "__main__":
    unittest.main()
