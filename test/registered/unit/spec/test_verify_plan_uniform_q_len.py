"""The uniform_q_len hint for FlashInfer's EAGLE target-verify paged plan is
derived only from EAGLE verify batches with a positive num_tokens_per_req;
every other plan call stays byte-for-byte the previous one. Capture fixes the
kernel's query tile from the hint, so the same value must reach every plan
through a captured verify wrapper."""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.flashinfer_backend import _verify_plan_uniform_q_len
from sglang.srt.speculative.spec_info import SpecInputType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _spec_info(kind=SpecInputType.EAGLE_VERIFY, num_tokens_per_req=4):
    return SimpleNamespace(spec_input_type=kind, num_tokens_per_req=num_tokens_per_req)


class TestVerifyPlanUniformQLen(CustomTestCase):
    def test_only_eagle_verify_batches_are_hinted(self):
        self.assertEqual(
            _verify_plan_uniform_q_len(_spec_info(num_tokens_per_req=4)), 4
        )
        self.assertIsNone(_verify_plan_uniform_q_len(None))
        for kind in (
            SpecInputType.EAGLE_DRAFT,
            SpecInputType.EAGLE_DRAFT_EXTEND,
            SpecInputType.DFLASH_VERIFY,
            SpecInputType.NGRAM_VERIFY,
        ):
            self.assertIsNone(_verify_plan_uniform_q_len(_spec_info(kind=kind)))

    def test_unset_num_tokens_per_req_is_not_a_promise(self):
        # SpecInput defaults num_tokens_per_req to -1; 0 must not be read as a hint either
        for value in (-1, 0):
            self.assertIsNone(
                _verify_plan_uniform_q_len(_spec_info(num_tokens_per_req=value))
            )


if __name__ == "__main__":
    unittest.main()
