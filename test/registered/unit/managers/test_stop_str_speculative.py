"""Regression: under speculative decoding (multi-token commits) a stop string
committed mid-chunk must still trigger the finish check, else the request
over-generates. Drives the real `Req.update_finish_state`; pure CPU."""

import unittest
from array import array

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

STOP_ID = 1
ID_TO_TEXT = {STOP_ID: "STOP", **{i: chr(ord("a") + i % 26) for i in range(10, 40)}}

# "STOP" (index 3) sits 6 tokens back: outside the old (stop_str_max_len + 1)
# window, inside the one widened by new_accepted_len.
MIDCHUNK = [10, 11, 12, STOP_ID, 20, 21, 22, 23, 24]


class _FakeTokenizer:
    eos_token_id = -1
    additional_stop_token_ids = None

    def decode(self, ids):
        return "".join(ID_TO_TEXT[int(i)] for i in ids)


def _make_req(output_ids, stop):
    sp = SamplingParams(max_new_tokens=1000, stop=stop)
    sp.normalize(tokenizer=None)  # char-based stop_str_max_len
    req = Req(
        rid="t",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=sp,
        eos_token_ids=set(),
        vocab_size=10_000,
    )
    req.tokenizer = _FakeTokenizer()
    req.output_ids = array("q", output_ids)
    return req


class TestStopStrSpeculative(unittest.TestCase):
    def test_stop_str_midchunk_finishes(self):
        req = _make_req(MIDCHUNK, stop=["STOP"])
        req.update_finish_state(new_accepted_len=6)
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_reason.matched, "STOP")

    def test_no_stop_str_does_not_finish(self):
        req = _make_req([10, 11, 12, 20, 21, 22, 23, 24], stop=["STOP"])
        req.update_finish_state(new_accepted_len=6)
        self.assertFalse(req.finished())


if __name__ == "__main__":
    unittest.main()
