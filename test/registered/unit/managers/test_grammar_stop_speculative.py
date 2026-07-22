"""Regression tests for grammar termination during speculative decoding."""

import unittest
from array import array

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

EOS_TOKEN_ID = 2
STOP_TOKEN_ID = 17


class _FakeTokenizer:
    eos_token_id = -1
    additional_stop_token_ids = None


class _TerminatedGrammar:
    @staticmethod
    def is_terminated():
        return True


def _make_req(stop_token_ids=None):
    sampling_params = SamplingParams(
        max_new_tokens=1_000,
        stop_token_ids=stop_token_ids,
    )
    sampling_params.normalize(tokenizer=_FakeTokenizer())
    req = Req(
        rid="grammar-stop",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=sampling_params,
        eos_token_ids={EOS_TOKEN_ID},
        vocab_size=100,
    )
    req.tokenizer = _FakeTokenizer()
    req.grammar = _TerminatedGrammar()
    req.output_ids = array("q", [11, 13, STOP_TOKEN_ID, EOS_TOKEN_ID])
    return req


class TestGrammarStopSpeculative(unittest.TestCase):
    def test_requested_stop_token_wins_over_trailing_eos(self):
        req = _make_req(stop_token_ids=[STOP_TOKEN_ID])

        req.update_finish_state(new_accepted_len=4)

        self.assertEqual(req.finished_reason.matched, STOP_TOKEN_ID)
        self.assertEqual(req.finished_len, 3)
        self.assertEqual(list(req.output_ids_through_stop), [11, 13, STOP_TOKEN_ID])

    def test_tokenizer_stop_token_wins_over_trailing_eos(self):
        req = _make_req()
        req.tokenizer.additional_stop_token_ids = {STOP_TOKEN_ID}

        req.update_finish_state(new_accepted_len=4)

        self.assertEqual(req.finished_reason.matched, STOP_TOKEN_ID)
        self.assertEqual(req.finished_len, 3)
        self.assertEqual(list(req.output_ids_through_stop), [11, 13, STOP_TOKEN_ID])


if __name__ == "__main__":
    unittest.main()
