"""Req owns its finish state: code outside the commit's finish checks changes
it only through the Req finish methods."""

import unittest
from array import array
from pathlib import Path
from unittest.mock import patch

from sglang.srt.managers import schedule_batch
from sglang.srt.managers.req_types import (
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_TOKEN,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test import field_writes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_SRT_DIR = Path(schedule_batch.__file__).resolve().parents[1]

_FINISH_FIELDS = {"to_finish", "finished_reason", "discard_output_reason"}
_OWNER_SCOPES = {
    "Req.__init__",
    "Req.request_finish",
    "Req.request_discard",
    "Req.clear_requested_finish",
    "Req.finish_outside_step",
    # The step commit's finish checks.
    "Req.update_finish_state",
    "Req._check_token_based_finish",
    "Req._check_str_based_finish",
    "Req._check_vocab_boundary_finish",
    "Req._cap_finished_len_at_max_new_tokens",
}


class TestReqFinishStateWriters(CustomTestCase):
    def test_only_finish_methods_write_finish_state(self):
        found = field_writes.scan_tree(
            _SRT_DIR, _FINISH_FIELDS, owner_scopes=_OWNER_SCOPES
        )
        self.assertFalse(
            found,
            "Write the finish state through a Req finish method:\n  "
            + "\n  ".join(found),
        )


def _finish_state(req):
    return {name: getattr(req, name) for name in sorted(_FINISH_FIELDS)}


class TestReqFinishMethods(CustomTestCase):
    def test_each_method_changes_only_its_field(self):
        pending = FINISH_ABORT("pending")
        reason = FINISH_LENGTH(length=3)
        cases = [
            ("request_finish", (reason,), dict(to_finish=reason)),
            ("request_discard", (reason,), dict(discard_output_reason=reason)),
            ("clear_requested_finish", (), dict(to_finish=None)),
            ("finish_outside_step", (reason,), dict(finished_reason=reason)),
        ]
        for method, args, changed in cases:
            with self.subTest(method=method):
                req = Req("r", "", [1, 2], SamplingParams())
                req.to_finish = pending
                expected = {**_finish_state(req), **changed}
                getattr(req, method)(*args)
                self.assertEqual(_finish_state(req), expected)

    def test_requested_finish_applies_at_the_commit(self):
        req = Req("r", "", [1, 2], SamplingParams())
        reason = FINISH_ABORT("aborted")
        req.request_finish(reason)
        self.assertFalse(req.finished())
        req.update_finish_state(0)
        self.assertIs(req.finished_reason, reason)
        self.assertIsNone(req.to_finish)

    def test_abort_resets_start_lens_for_the_one_token_prompt(self):
        req = Req(
            rid="r",
            origin_input_text="",
            origin_input_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
            return_logprob=True,
            return_routed_experts=True,
            routed_experts_start_len=-1,
        )
        req.logprob_start_len = 2
        with patch("sglang.srt.managers.schedule_batch.get_parallel") as parallel:
            parallel().tp_rank = 0
            req.set_finish_with_abort("rejected at intake")
        self.assertEqual(list(req.origin_input_ids), [0])
        self.assertEqual(req.logprob_start_len, -1)
        self.assertEqual(req.routed_experts_start_len, 0)


class _StopTokenizer:
    eos_token_id = 3
    additional_stop_token_ids = {4}


class TestModelEosStop(CustomTestCase):
    def _finished_on(self, token_id, *, stop_on_model_eos, stop_token_ids=None):
        sampling_params = SamplingParams(
            max_new_tokens=8, stop_token_ids=stop_token_ids
        )
        sampling_params.normalize(tokenizer=None)
        req = Req("r", "", array("q", [1]), sampling_params, eos_token_ids={2})
        req.tokenizer = _StopTokenizer()
        req.stop_on_model_eos = stop_on_model_eos
        req.output_ids = array("q", [10, token_id])
        req.update_finish_state(new_accepted_len=1)
        return isinstance(req.finished_reason, FINISH_MATCHED_TOKEN)

    def test_model_eos_finishes_only_when_stop_on_model_eos(self):
        """Each model EOS source (config, tokenizer, additional stops) finishes a
        request by default and none does without stop_on_model_eos; the request's
        own stop_token_ids finish it either way."""
        for token_id in (2, 3, 4):
            with self.subTest(token_id=token_id):
                self.assertTrue(self._finished_on(token_id, stop_on_model_eos=True))
                self.assertFalse(self._finished_on(token_id, stop_on_model_eos=False))
        self.assertTrue(
            self._finished_on(5, stop_on_model_eos=False, stop_token_ids=[5])
        )


if __name__ == "__main__":
    unittest.main()
