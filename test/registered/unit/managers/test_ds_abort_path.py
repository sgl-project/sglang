"""Double Sparsity per-request abort path.

When the DS adapter sanitizes a row it stamps an ``error_class`` into the
request's ``meta_info["double_sparsity"]`` summary. The scheduler's
``_maybe_abort_on_ds_error`` hook must turn that into a finished request *in the
same scheduler step* — driving ``req.set_finish_with_abort(...)`` then
``req.update_finish_state(...)`` (the post-#25725 finisher), NOT the removed
``check_finished`` path. This regression test pins that behaviour without a GPU.
"""

import unittest
from array import array
from unittest import mock

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.sampling.sampling_params import SamplingParams


class _FakeLogitsOutput:
    def __init__(self, per_request_summary):
        self.per_request_summary = per_request_summary
        self.hidden_states = None


def _make_req(rid="ds-abort-0"):
    sp = SamplingParams(max_new_tokens=16)
    sp.normalize(tokenizer=None)
    req = Req(
        rid=rid,
        origin_input_text="hello world",
        origin_input_ids=array("i", [1, 2, 3, 4]),
        sampling_params=sp,
    )
    return req


class TestDSAbortPath(unittest.TestCase):
    def test_ds_error_aborts_in_same_step(self):
        req = _make_req()
        self.assertFalse(req.finished())

        faulted = _FakeLogitsOutput(
            {
                "double_sparsity": [
                    {
                        "error_class": "DSSelectorError",
                        "error_message": "reuse-edge sanitized row 0",
                    }
                ]
            }
        )
        # self is unused by the hook; pass None. Patch the rank-0 error log
        # (set_finish_with_abort touches the TP group only to decide who logs).
        with mock.patch(
            "sglang.srt.managers.schedule_batch.get_tensor_model_parallel_rank",
            return_value=0,
        ):
            aborted = SchedulerBatchResultProcessor._maybe_abort_on_ds_error(
                None, 0, req, faulted
            )

        self.assertTrue(aborted)
        # set_finish_with_abort + update_finish_state ran THIS call:
        self.assertTrue(req.finished())
        self.assertIsNotNone(req.finished_reason)
        # The failure summary is materialised on the request for the response.
        self.assertEqual(
            req.per_request_summary["double_sparsity"]["error_class"],
            "DSSelectorError",
        )

    def test_no_error_does_not_abort(self):
        req = _make_req("ds-ok-0")
        healthy = _FakeLogitsOutput(
            {
                "double_sparsity": [
                    {"selected_tokens": 2048, "total_tokens": 4774, "dense_fallback": 0}
                ]
            }
        )
        aborted = SchedulerBatchResultProcessor._maybe_abort_on_ds_error(
            None, 0, req, healthy
        )
        self.assertFalse(aborted)
        self.assertFalse(req.finished())

    def test_missing_summary_does_not_abort(self):
        req = _make_req("ds-none-0")
        self.assertFalse(
            SchedulerBatchResultProcessor._maybe_abort_on_ds_error(
                None, 0, req, _FakeLogitsOutput(None)
            )
        )
        self.assertFalse(
            SchedulerBatchResultProcessor._maybe_abort_on_ds_error(None, 0, req, None)
        )
        self.assertFalse(req.finished())


if __name__ == "__main__":
    unittest.main()
