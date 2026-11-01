"""A retracted request must not ask its re-prefill for input logprobs it already has.

Retraction re-prefills prompt + generated tokens, while logprob_start_len still
points at the prompt end, so the generated tokens were requested again as input
(prompt) logprobs: full-vocab logits that the result processor then discarded, and
a fatal request on backends that cannot serve input logprobs.
"""

import unittest
from array import array

from sglang.srt.managers.schedule_batch import (
    Req,
    compute_extend_input_logprob_token_ids,
    compute_extend_logprob_start_len,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PROMPT_LEN = 10
NUM_OUTPUT_TOKENS = 5


def _logprob_req(*, logprob_start_len: int) -> Req:
    sampling_params = SamplingParams(max_new_tokens=32)
    sampling_params.normalize(None)
    req = Req(
        rid="req",
        origin_input_text="",
        origin_input_ids=array("q", range(PROMPT_LEN)),
        sampling_params=sampling_params,
        return_logprob=True,
        vocab_size=128,
    )
    req.logprob_start_len = logprob_start_len
    return req


def _input_logprob_rows_on_reprefill(req: Req) -> tuple[int, list[int]]:
    """Input-logprob rows the re-prefill of a retracted req asks for, and the
    target ids it adds to the batch, at the longest prefix the radix cache may
    hand it. The batch's flat target-id array stays aligned only if the two
    agree."""
    req.refresh_fill_ids()
    fill_len = len(req.full_untruncated_fill_ids)
    prefix_len = req._compute_max_prefix_len(fill_len)
    extend_len = fill_len - prefix_len
    start = compute_extend_logprob_start_len(
        logprob_start_len=req.logprob_start_len,
        prefix_len=prefix_len,
        extend_len=extend_len,
        full_untruncated_fill_len=fill_len,
        input_logprobs_finalized=req.logprob.input_token_logprobs_val is not None,
    )
    token_ids = compute_extend_input_logprob_token_ids(
        origin_input_ids=req.origin_input_ids,
        logprob_start_len=req.logprob_start_len,
        prefix_len=prefix_len,
        extend_end=fill_len,
        num_input_logprobs=extend_len - start,
    )
    return extend_len - start, token_ids


def _finish_first_prefill_and_retract(req: Req) -> None:
    # Input logprobs finalized (empty for output-only), tokens decoded, retracted.
    req.logprob.input_token_logprobs_val = []
    req.output_ids.extend(range(NUM_OUTPUT_TOKENS))
    req.reset_for_retract()


class TestRetractInputLogprobs(CustomTestCase):
    def test_retracted_decode_asks_for_no_input_logprobs(self):
        # Output-only logprobs: intake sets the start at the prompt end.
        req = _logprob_req(logprob_start_len=PROMPT_LEN)
        _finish_first_prefill_and_retract(req)

        self.assertEqual(_input_logprob_rows_on_reprefill(req), (0, []))

    def test_retracted_prompt_logprob_req_adds_no_target_ids(self):
        # Its prompt logprobs were returned on the first prefill. Target ids for
        # rows nobody computes would shift every later request's ids in the batch.
        req = _logprob_req(logprob_start_len=0)
        _finish_first_prefill_and_retract(req)

        self.assertEqual(_input_logprob_rows_on_reprefill(req), (0, []))

    def test_retracted_before_prompt_logprobs_finalized_still_asks(self):
        # A chunked prefill retracted mid-prompt has not finalized its prompt
        # logprobs, so the re-prefill must compute all of them: one target per
        # row, the next prompt token, and 0 past the prompt end.
        req = _logprob_req(logprob_start_len=0)
        req.reset_for_retract()

        self.assertEqual(
            _input_logprob_rows_on_reprefill(req),
            (PROMPT_LEN, list(range(1, PROMPT_LEN)) + [0]),
        )


if __name__ == "__main__":
    unittest.main()
