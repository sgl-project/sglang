"""Overlap-scheduler penalizer history regression test (#41124).

With the overlap scheduler on (the default), the penalizer used to be fed at
prepare_for_decode time, where the previous step's token is still in flight —
so token i was penalized against [prompt[-1]] + output[:i-1] instead of
output[:i]. The penalizer is now fed when the token resolves, which the
non-overlap path has always done, so both schedules must produce the same
penalized distribution.
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=170, suite="stage-b-test-small-1-gpu")


class TestOverlapPenaltyHistory(CustomTestCase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST

    def _run_penalized_greedy(self, overlap: bool):
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[] if overlap else ["--disable-overlap-schedule"],
        )
        try:
            response = requests.post(
                self.base_url + "/generate",
                json={
                    # Eliezer-shaped repetition so the penalty actually bites.
                    "text": "Write the word banana over and over: banana",
                    "sampling_params": {
                        "max_new_tokens": 32,
                        "temperature": 0,
                        "repetition_penalty": 1.3,
                    },
                    "return_logprob": True,
                    "return_text_in_logprobs": True,
                    "logprob_start_len": 0,
                },
            )
            self.assertEqual(response.status_code, 200)
            meta = response.json()["meta_info"]
            return [t[0] for t in meta["output_token_logprobs"]]
        finally:
            kill_process_tree(process.pid)

    def test_overlap_penalizer_history_matches_nonoverlap(self):
        on = self._run_penalized_greedy(overlap=True)
        off = self._run_penalized_greedy(overlap=False)
        self.assertEqual(len(on), len(off))
        diffs = [abs(a - b) for a, b in zip(on, off)]
        # Matching histories differ by fp16 scheduling noise (~0.014 max in the
        # issue's measurement); the one-step-stale history diverges by ~0.5+.
        self.assertLess(max(diffs), 0.05)


if __name__ == "__main__":
    unittest.main()
