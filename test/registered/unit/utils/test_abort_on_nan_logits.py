"""Unit tests for the SGLANG_ABORT_ON_NAN_LOGITS abort path -- CPU, no server.

Covers full-row NaN detection (`detect_full_nan_rows`) and the scheduler-side
abort it drives (`SchedulerBatchResultProcessor._abort_full_nan_logits_reqs`).
"""

import unittest
from http import HTTPStatus
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.utils.async_probe import detect_full_nan_rows, sanitize_nan_logits
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_req(rid: str, finished: bool = False, is_retracted: bool = False):
    return SimpleNamespace(
        rid=rid,
        finished=lambda: finished,
        is_retracted=is_retracted,
        skip_radix_cache_insert=False,
        to_finish=None,
    )


def _abort(reqs, full_nan_rows):
    """Drive the abort helper; it never touches `self`, so pass None."""
    SchedulerBatchResultProcessor._abort_full_nan_logits_reqs(
        None,
        SimpleNamespace(reqs=reqs),
        LogitsProcessorOutput(next_token_logits=None, full_nan_rows=full_nan_rows),
    )


class TestDetectFullNanRows(CustomTestCase):
    def test_returns_none_when_disabled(self):
        logits = torch.full((2, 4), float("nan"))
        with envs.SGLANG_ABORT_ON_NAN_LOGITS.override(False):
            self.assertIsNone(detect_full_nan_rows(logits))

    def test_flags_only_full_rows(self):
        logits = torch.zeros((4, 3))
        logits[1] = float("nan")  # full row -> flagged
        logits[2][0] = float("nan")  # partial row -> not flagged
        with envs.SGLANG_ABORT_ON_NAN_LOGITS.override(True):
            mask = detect_full_nan_rows(logits)
        self.assertEqual(mask.tolist(), [False, True, False, False])

    def test_sanitize_makes_full_nan_row_uniform(self):
        """The motivating failure: sanitization alone leaves a samplable row."""
        logits = torch.full((1, 5), float("nan"))
        with envs.SGLANG_ABORT_ON_NAN_LOGITS.override(True):
            mask = detect_full_nan_rows(logits)
        with envs.SGLANG_SANITIZE_NAN_LOGITS.override(True):
            sanitize_nan_logits(logits)
        # Every logit is now equal, i.e. softmax is uniform over the vocab.
        self.assertEqual(len(torch.unique(logits)), 1)
        self.assertEqual(mask.tolist(), [True])


class TestAbortFullNanLogitsReqs(CustomTestCase):
    def test_no_mask_is_a_noop(self):
        reqs = [_fake_req("a"), _fake_req("b")]
        _abort(reqs, None)
        for req in reqs:
            self.assertIsNone(req.to_finish)
            self.assertFalse(req.skip_radix_cache_insert)

    def test_aborts_only_flagged_reqs(self):
        reqs = [_fake_req("a"), _fake_req("b"), _fake_req("c")]
        _abort(reqs, torch.tensor([False, True, False]))

        self.assertIsNone(reqs[0].to_finish)
        self.assertFalse(reqs[0].skip_radix_cache_insert)
        self.assertIsNone(reqs[2].to_finish)

        # Retriable status, and the poisoned KV is kept out of the radix tree.
        self.assertIsInstance(reqs[1].to_finish, FINISH_ABORT)
        self.assertEqual(reqs[1].to_finish.status_code, HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertTrue(reqs[1].skip_radix_cache_insert)

    def test_skips_finished_and_retracted_reqs(self):
        reqs = [_fake_req("a", finished=True), _fake_req("b", is_retracted=True)]
        _abort(reqs, torch.tensor([True, True]))
        for req in reqs:
            self.assertIsNone(req.to_finish)
            self.assertFalse(req.skip_radix_cache_insert)

    def test_rejects_mask_that_is_not_one_row_per_req(self):
        reqs = [_fake_req("a"), _fake_req("b")]
        with self.assertRaises(AssertionError):
            _abort(reqs, torch.tensor([False, True, False, False]))


if __name__ == "__main__":
    unittest.main()
