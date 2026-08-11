"""Unit tests for dropping stale requests during a weight-update pause.

`GenerateReqInput.start_weight_version` is a caller-declared label saying which
weight version a request starts generating under, and
`PauseGenerationReqInput.abort_below_start_weight_version` drops everything
strictly below a threshold while the engine is paused, before `retract`
re-prefills requests that are going to be discarded anyway.
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import GenerateReqInput, PauseGenerationReqInput
from sglang.srt.managers.scheduler import _is_stale_start_version

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(start_weight_version=None):
    return SimpleNamespace(
        rid="r", start_weight_version=start_weight_version, to_finish=None
    )


class TestStaleStartVersionPredicate(CustomTestCase):
    """Both pause modes route through this, so its truth table is the contract."""

    def test_strictly_below_is_stale(self):
        self.assertTrue(_is_stale_start_version(_req(6), 7))
        self.assertFalse(_is_stale_start_version(_req(7), 7))
        self.assertFalse(_is_stale_start_version(_req(8), 7))

    def test_undeclared_version_is_never_stale(self):
        # An unknown version cannot be evaluated, and guessing is worse than
        # keeping the request.
        self.assertFalse(_is_stale_start_version(_req(None), 7))

    def test_no_threshold_means_nothing_is_stale(self):
        self.assertFalse(_is_stale_start_version(_req(1), None))


class TestBatchChildrenInheritIt(CustomTestCase):
    """A batched request is split by `__getitem__` before it reaches the scheduler.

    A field the split forgets to carry is not an error: the child simply starts
    with None, the sweep can never match it, and the echo goes missing -- so a
    batched caller silently loses the feature the single-request path has."""

    def test_getitem_carries_start_weight_version(self):
        obj = GenerateReqInput(
            text=["a", "b"], sampling_params={"max_new_tokens": 1}, start_weight_version=7
        )
        obj.normalize_batch_and_arguments()
        self.assertEqual([obj[i].start_weight_version for i in range(2)], [7, 7])

    def test_getitem_leaves_it_undeclared_when_absent(self):
        obj = GenerateReqInput(text=["a", "b"], sampling_params={"max_new_tokens": 1})
        obj.normalize_batch_and_arguments()
        self.assertEqual([obj[i].start_weight_version for i in range(2)], [None, None])


class TestPauseThresholdValidation(CustomTestCase):
    def test_modes_that_keep_survivors_accept_threshold(self):
        for mode in ("retract", "in_place"):
            req = PauseGenerationReqInput(
                mode=mode, abort_below_start_weight_version=7
            )
            self.assertEqual(req.abort_below_start_weight_version, 7, mode)

    def test_abort_mode_rejects_threshold(self):
        # It aborts everything already, so a filter there could only mislead.
        with self.assertRaises(ValueError):
            PauseGenerationReqInput(mode="abort", abort_below_start_weight_version=7)

    def test_modes_still_work_without_threshold(self):
        for mode in ("abort", "retract", "in_place"):
            self.assertIsNone(
                PauseGenerationReqInput(mode=mode).abort_below_start_weight_version
            )


if __name__ == "__main__":
    unittest.main()
