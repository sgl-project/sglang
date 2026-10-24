"""The uniform_q_len hint for FlashInfer's EAGLE target-verify paged plan must be
passed only to a FlashInfer whose plan() takes the keyword and only for EAGLE verify
batches; every other plan call stays byte-for-byte the previous one."""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.flashinfer_backend import (
    _prefill_plan_accepts_uniform_q_len,
    _verify_plan_uniform_q_len,
)
from sglang.srt.speculative.spec_info import SpecInputType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _PlanWithHint:
    def plan(self, qo_indptr, kv_indptr, *args, uniform_q_len=None, **kwargs):
        return uniform_q_len


class _PlanWithoutHint:
    def plan(self, qo_indptr, kv_indptr, *args, **kwargs):
        return None


class _PlanKwargsOnly:
    def plan(self, *args, **kwargs):
        return None


def _spec_info(kind=SpecInputType.EAGLE_VERIFY, num_tokens_per_req=4):
    return SimpleNamespace(spec_input_type=kind, num_tokens_per_req=num_tokens_per_req)


class TestVerifyPlanUniformQLen(CustomTestCase):
    def test_capability_check_fails_closed_without_the_keyword(self):
        self.assertTrue(_prefill_plan_accepts_uniform_q_len(_PlanWithHint))
        self.assertFalse(_prefill_plan_accepts_uniform_q_len(_PlanWithoutHint))
        # a **kwargs-only plan() would swallow the keyword silently or raise later
        self.assertFalse(_prefill_plan_accepts_uniform_q_len(_PlanKwargsOnly))

    def test_only_eagle_verify_batches_on_a_capable_flashinfer_are_hinted(self):
        hinted = _verify_plan_uniform_q_len(
            wrapper_paged=_PlanWithHint(), spec_info=_spec_info(num_tokens_per_req=4)
        )
        self.assertEqual(hinted, 4)
        for wrapper in (_PlanWithoutHint(), _PlanKwargsOnly()):
            self.assertIsNone(
                _verify_plan_uniform_q_len(
                    wrapper_paged=wrapper, spec_info=_spec_info()
                )
            )
        self.assertIsNone(
            _verify_plan_uniform_q_len(wrapper_paged=_PlanWithHint(), spec_info=None)
        )
        for kind in (
            SpecInputType.EAGLE_DRAFT,
            SpecInputType.EAGLE_DRAFT_EXTEND,
            SpecInputType.DFLASH_VERIFY,
            SpecInputType.NGRAM_VERIFY,
        ):
            self.assertIsNone(
                _verify_plan_uniform_q_len(
                    wrapper_paged=_PlanWithHint(), spec_info=_spec_info(kind=kind)
                )
            )

    def test_unset_num_tokens_per_req_is_not_a_promise(self):
        # SpecInput defaults num_tokens_per_req to -1; 0 must not be read as a hint either
        for value in (-1, 0):
            self.assertIsNone(
                _verify_plan_uniform_q_len(
                    wrapper_paged=_PlanWithHint(),
                    spec_info=_spec_info(num_tokens_per_req=value),
                )
            )


if __name__ == "__main__":
    unittest.main()
