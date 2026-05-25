"""Regression test for EAGLE V2 + return_hidden_states (issue #26163)."""

import unittest

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
    CustomTestCase,
)

register_cuda_ci(est_time=120, stage="extra-a", runner_config="1-gpu-large")


class TestEagleV2ReturnHiddenStates(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # topk=1 + overlap (default) routes to spec V2;
        # topk>1 forces V1 via arg_groups/speculative_hook.py.
        cls.engine = sgl.Engine(
            model_path=DEFAULT_TARGET_MODEL_EAGLE,
            speculative_draft_model_path=DEFAULT_DRAFT_MODEL_EAGLE,
            speculative_algorithm="EAGLE",
            speculative_num_steps=5,
            speculative_eagle_topk=1,
            enable_return_hidden_states=True,
            mem_fraction_static=0.7,
            cuda_graph_max_bs=8,
        )

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def test_eagle_v2_with_return_hidden_states(self):
        # Repetitive prompts so the drafter lands long accept streaks.
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
