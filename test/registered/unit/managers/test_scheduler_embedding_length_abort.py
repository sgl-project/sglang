"""Scheduler containment for over-length embedding requests."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _FakeReq:
    """Stand-in for Req that records whether the abort contract was applied."""

    def __init__(self):
        self.tokenizer = None
        self.logprob_start_len = None
        self.to_finish = None

    def set_finish_with_abort(self, error_msg, **kwargs):
        self.to_finish = error_msg


class TestSchedulerEmbeddingLengthAbort(CustomTestCase):
    def test_over_length_embedding_request_is_aborted_before_admission(self):
        """An embedding request that fails length validation must already carry a
        finish reason when it reaches the queue; otherwise it is admitted at full
        length, no error is ever returned, and the caller blocks until timeout.
        """
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tokenizer = None
        scheduler.max_req_input_len = 8
        scheduler._maybe_namespace_elastic_radix_cache = MagicMock()

        enqueued = []
        scheduler._add_request_to_queue = lambda req: enqueued.append(
            (req, req.to_finish)
        )

        recv_req = MagicMock(mm_inputs=None)
        req = _FakeReq()
        error = (
            "Input length (16 tokens) exceeds the maximum allowed length (8 tokens)."
        )

        with (
            patch("sglang.srt.managers.scheduler.Req", return_value=req),
            patch(
                "sglang.srt.managers.scheduler.get_serving",
                return_value=SimpleNamespace(allow_auto_truncate=False),
            ),
            patch(
                "sglang.srt.managers.scheduler.validate_input_length",
                return_value=error,
            ) as validate_input_length,
        ):
            scheduler.handle_embedding_request(recv_req)

        validate_input_length.assert_called_once_with(req, 8, False)
        self.assertEqual(enqueued, [(req, error)])


if __name__ == "__main__":
    unittest.main()
