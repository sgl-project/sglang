"""The cache key must stop where the client's output stops.

A verify step commits a whole accepted chunk, so output_ids can run past the
stop position. Those tokens never reach the client, so caching them stores a
suffix no continuation can reproduce.
"""

import unittest
from array import array

import sglang.srt.runtime_context as rc
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

EOS_TOKEN_ID = 2
PROMPT_LEN = 6


class _FakeTokenizer:
    eos_token_id = -1
    additional_stop_token_ids = None


def _make_req(max_new_tokens: int) -> Req:
    sampling_params = SamplingParams(max_new_tokens=max_new_tokens)
    sampling_params.normalize(tokenizer=_FakeTokenizer())
    req = Req(
        rid="spec-overshoot",
        origin_input_text="",
        origin_input_ids=array("q", range(PROMPT_LEN)),
        sampling_params=sampling_params,
        eos_token_ids={EOS_TOKEN_ID},
        vocab_size=100,
    )
    req.tokenizer = _FakeTokenizer()
    return req


def _commit(req: Req, tokens: list[int], accepted_len: int) -> None:
    req.output_ids = array("q", tokens)
    req.kv_committed_len = PROMPT_LEN + len(tokens)
    req.update_finish_state(new_accepted_len=accepted_len)


class TestSpecOvershootCacheLen(unittest.TestCase):
    def setUp(self):
        rc.reset_context()
        rc.get_context().set_server_args(ServerArgs(model_path="dummy"))

    def tearDown(self):
        rc.reset_context()

    def test_length_overshoot_is_excluded(self):
        req = _make_req(max_new_tokens=4)

        _commit(req, [11, 12, 13, 14, 15, 16, 17], accepted_len=7)

        self.assertEqual(req.finished_len, 4)
        self.assertEqual(req.effective_kv_committed_len(), PROMPT_LEN + 4)

    def test_eos_overshoot_is_excluded(self):
        req = _make_req(max_new_tokens=100)

        _commit(req, [11, 12, EOS_TOKEN_ID, 14, 15], accepted_len=5)

        self.assertEqual(req.finished_len, 3)
        self.assertEqual(req.effective_kv_committed_len(), PROMPT_LEN + 3)

    def test_exact_stop_keeps_the_whole_commit(self):
        req = _make_req(max_new_tokens=4)

        _commit(req, [11, 12, 13, 14], accepted_len=1)

        self.assertEqual(req.finished_len, 4)
        self.assertEqual(req.effective_kv_committed_len(), PROMPT_LEN + 4)

    def test_unfinished_request_keeps_the_whole_commit(self):
        req = _make_req(max_new_tokens=100)

        _commit(req, [11, 12, 13], accepted_len=3)

        self.assertIsNone(req.finished_len)
        self.assertEqual(req.effective_kv_committed_len(), PROMPT_LEN + 3)


if __name__ == "__main__":
    unittest.main()
