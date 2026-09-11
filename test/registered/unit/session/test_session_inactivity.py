"""Session validation should preserve the idle timeout for rejected requests."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.io_struct import SessionParams, TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _request(params):
    return TokenizedGenerateReqInput(
        rid="request-a",
        input_text="",
        input_ids=array("q", [1, 2]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=2),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        session_params=params,
    )


class TestSessionInactivity(CustomTestCase):
    def setUp(self):
        self.clock = self.enterContext(
            patch("sglang.srt.session.session_controller.time.monotonic")
        )

    def test_rejected_request_preserves_timeout(self):
        cases = [
            (False, {"rid": "missing-request"}, False),
            (False, {"rid": "missing-request", "replace": True}, False),
            (True, {"replace": True}, False),
            (True, {"drop_previous_output": True}, False),
            (True, {"offset": 1}, False),
            (True, {}, True),
        ]
        for streaming, params, inflight in cases:
            with self.subTest(streaming=streaming, params=params, inflight=inflight):
                self.clock.return_value = 100
                session = Session(0, "session-a", streaming=streaming, timeout=10)
                session._inflight = inflight
                self.clock.return_value = 109
                with patch(
                    "sglang.srt.managers.schedule_batch.get_parallel",
                    return_value=SimpleNamespace(tp_rank=0),
                ):
                    result = session.create_req(
                        _request(SessionParams(id="session-a", **params)),
                        tokenizer=None,
                        vocab_size=128,
                    )

                self.assertIsInstance(result.to_finish, FINISH_ABORT)
                self.clock.return_value = 111
                self.assertTrue(session.is_timed_out())
                self.assertEqual(session.last_active_time, 100)

    def test_accepted_request_refreshes_timeout(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.clock.return_value = 100
                session = Session(0, "session-a", streaming=streaming, timeout=10)
                self.clock.return_value = 109
                result = session.create_req(
                    _request(SessionParams(id="session-a")),
                    tokenizer=None,
                    vocab_size=128,
                )

                self.assertIsNone(result.to_finish)
                self.clock.return_value = 111
                self.assertFalse(session.is_timed_out())
                self.assertEqual(session.last_active_time, 109)


if __name__ == "__main__":
    unittest.main()
