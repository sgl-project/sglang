"""A retracted request must not ask its re-prefill for input logprobs it already has.

Its re-prefill covers prompt + generated tokens; the extra input-logprob rows were computed and
then discarded, and a backend that does not serve input logprobs rejects the batch.
"""

import unittest

from sglang.srt.managers.schedule_batch import compute_extend_logprob_start_len
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

PROMPT_LEN = 10
FILL_LEN = 15  # prompt + 5 generated tokens re-prefilled after retraction


def _start(*, logprob_start_len: int, finalized: bool) -> int:
    return compute_extend_logprob_start_len(
        logprob_start_len=logprob_start_len,
        prefix_len=0,
        extend_len=FILL_LEN,
        full_untruncated_fill_len=FILL_LEN,
        input_logprobs_finalized=finalized,
    )


class TestRetractInputLogprobs(CustomTestCase):
    def test_retracted_request_asks_for_no_input_logprobs(self):
        for logprob_start_len in (0, PROMPT_LEN, -1):
            self.assertEqual(
                _start(logprob_start_len=logprob_start_len, finalized=True), FILL_LEN
            )

    def test_first_prefill_unchanged(self):
        self.assertEqual(
            _start(logprob_start_len=PROMPT_LEN, finalized=False), PROMPT_LEN
        )
        self.assertEqual(_start(logprob_start_len=0, finalized=False), 0)


if __name__ == "__main__":
    unittest.main()
