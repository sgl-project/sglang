"""Regression test for stop-string detection under speculative decoding.

Speculative decoding commits multiple tokens per step, so a stop string can land
mid-chunk (with more tokens accepted after it in the same step). The finish check
(`Req.update_finish_state` -> `_check_str_based_finish` -> `tail_str`) must widen
its scan window by the accepted-token count, otherwise the fixed tail slides past
the stop string and the request over-generates instead of stopping.

Pure CPU unit test: drives `update_finish_state` directly on a `Req` with a fake
tokenizer; no model / GPU.
"""

import unittest
from array import array

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Token id -> decoded text. Concatenated by the fake tokenizer's decode().
STOP_ID = 1
ID_TO_TEXT = {STOP_ID: "STOP"}
for _i in range(10, 40):
    ID_TO_TEXT[_i] = chr(ord("a") + (_i % 26))


class _FakeTokenizer:
    eos_token_id = -1
    additional_stop_token_ids = None

    def decode(self, ids):
        return "".join(ID_TO_TEXT[int(i)] for i in ids)


def _make_req(output_ids, stop=None, max_new_tokens=1000):
    sp = SamplingParams(max_new_tokens=max_new_tokens, stop=stop)
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


class TestSpecStopStringFinish(unittest.TestCase):
    def test_stop_str_midchunk_finishes(self):
        # `stop_str_max_len` == len("STOP") == 4, so the old fixed window was 5
        # tokens. A 6-token spec chunk with "STOP" at its front leaves "STOP"
        # 6 tokens back -> outside the old window, inside the widened one.
        prefix = [10, 11, 12]
        chunk = [STOP_ID, 20, 21, 22, 23, 24]
        req = _make_req(prefix + chunk, stop=["STOP"])
        req.update_finish_state(new_accepted_len=len(chunk))
        self.assertTrue(
            req.finished(),
            "stop string committed mid-chunk must be detected, not skipped",
        )
        self.assertEqual(req.finished_reason.matched, "STOP")

    def test_no_stop_str_does_not_finish(self):
        # Control: the same chunk shape without the stop string must not finish.
        req = _make_req([10, 11, 12, 20, 21, 22, 23, 24], stop=["STOP"])
        req.update_finish_state(new_accepted_len=6)
        self.assertFalse(req.finished())


if __name__ == "__main__":
    unittest.main()
