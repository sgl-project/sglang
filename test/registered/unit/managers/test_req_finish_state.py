"""Req owns its finish state: code outside the commit's finish checks changes
it only through the Req finish methods."""

import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.managers import schedule_batch
from sglang.srt.managers.req_types import FINISH_ABORT, FINISH_LENGTH
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


if __name__ == "__main__":
    unittest.main()
