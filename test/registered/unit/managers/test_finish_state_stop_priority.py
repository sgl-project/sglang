"""Regression tests for stop conditions at the max-new-tokens boundary."""

import unittest
from array import array

from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    FINISH_MATCHED_STR,
    FINISH_MATCHED_TOKEN,
    FINISHED_MATCHED_REGEX,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

STOP_ID = 1
EOS_ID = 2
VOCAB_SIZE = 100


class _FakeTokenizer:
    eos_token_id = EOS_ID
    additional_stop_token_ids = None

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))

    def decode(self, token_ids):
        return "".join(
            {
                STOP_ID: "STOP",
                EOS_ID: "<EOS>",
                10: " blue",
                11: " sky",
                12: ".",
                60: "a",
                61: ".",
                99: " extra",
            }.get(int(token_id), "?")
            for token_id in token_ids
        )


class _TerminatedGrammar:
    @staticmethod
    def is_terminated():
        return True


def _make_req(
    output_ids,
    *,
    max_new_tokens,
    grammar=None,
    vocab_size=VOCAB_SIZE,
    **sampling_kwargs,
):
    tokenizer = _FakeTokenizer()
    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        **sampling_kwargs,
    )
    sampling_params.normalize(tokenizer=tokenizer)
    req = Req(
        rid="finish-priority",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=sampling_params,
        eos_token_ids=frozenset(),
        vocab_size=vocab_size,
    )
    req.tokenizer = tokenizer
    req.grammar = grammar
    req.output_ids = array("q", output_ids)
    return req


class TestFinishStateStopPriority(CustomTestCase):
    def test_eos_at_max_token_boundary_finishes_as_stop(self):
        for output_ids in ([10, EOS_ID], [10, 11, EOS_ID]):
            with self.subTest(max_new_tokens=len(output_ids)):
                req = _make_req(output_ids, max_new_tokens=len(output_ids))

                req.update_finish_state()

                self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
                self.assertEqual(req.finished_reason.matched, EOS_ID)
                self.assertEqual(req.finished_len, len(output_ids))

    def test_explicit_stop_token_at_boundary_finishes_as_stop(self):
        req = _make_req([10, 11], max_new_tokens=2, stop_token_ids=[11])

        req.update_finish_state()

        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_reason.matched, 11)
        self.assertEqual(req.finished_len, 2)

    def test_stop_string_at_boundary_finishes_as_stop(self):
        req = _make_req([10, STOP_ID], max_new_tokens=2, stop=["STOP"])

        req.update_finish_state(new_accepted_len=2)

        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_STR)
        self.assertEqual(req.finished_reason.matched, "STOP")
        self.assertEqual(req.finished_len, 2)

    def test_stop_regex_at_boundary_finishes_as_stop(self):
        req = _make_req([60, 61], max_new_tokens=2, stop_regex=[r"\.$"])

        req.update_finish_state(new_accepted_len=2)

        self.assertIsInstance(req.finished_reason, FINISHED_MATCHED_REGEX)
        self.assertEqual(req.finished_reason.matched, r"\.$")
        self.assertEqual(req.finished_len, 2)

    def test_stop_string_wins_over_eos_at_boundary(self):
        req = _make_req([10, STOP_ID, EOS_ID], max_new_tokens=3, stop=["STOP"])

        req.update_finish_state(new_accepted_len=3)

        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_STR)
        self.assertEqual(req.finished_reason.matched, "STOP")
        self.assertEqual(req.finished_len, 2)
        self.assertEqual(list(req.output_ids), [10, STOP_ID, EOS_ID])
        self.assertEqual(list(req.output_ids_through_stop), [10, STOP_ID])

    def test_speculative_eos_within_budget_finishes_as_stop(self):
        req = _make_req([10, EOS_ID, 99, 99], max_new_tokens=3)

        req.update_finish_state(new_accepted_len=3)

        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_reason.matched, EOS_ID)
        self.assertEqual(req.finished_len, 2)
        self.assertEqual(list(req.output_ids), [10, EOS_ID, 99, 99])
        self.assertEqual(list(req.output_ids_through_stop), [10, EOS_ID])

    def test_speculative_eos_beyond_budget_finishes_as_length(self):
        req = _make_req([10, 11, 12, EOS_ID], max_new_tokens=3)

        req.update_finish_state(new_accepted_len=3)

        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertEqual(req.finished_len, 3)
        self.assertEqual(list(req.output_ids), [10, 11, 12, EOS_ID])
        self.assertEqual(list(req.output_ids_through_stop), [10, 11, 12])

    def test_speculative_stop_string_beyond_budget_finishes_as_length(self):
        req = _make_req([10, 11, 12, STOP_ID], max_new_tokens=3, stop=["STOP"])

        req.update_finish_state(new_accepted_len=3)

        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertEqual(req.finished_len, 3)

    def test_speculative_stop_string_within_budget_finishes_as_stop(self):
        req = _make_req([10, STOP_ID, 99, 99], max_new_tokens=3, stop=["STOP"])

        req.update_finish_state(new_accepted_len=3)

        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_STR)
        self.assertEqual(req.finished_reason.matched, "STOP")
        self.assertEqual(req.finished_len, 2)
        self.assertEqual(list(req.output_ids_through_stop), [10, STOP_ID])

    def test_grammar_at_boundary_finishes_as_stop(self):
        req = _make_req([10, 11], max_new_tokens=2, grammar=_TerminatedGrammar())

        req.update_finish_state()

        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_reason.matched, 11)

    def test_grammar_beyond_budget_finishes_as_length(self):
        req = _make_req([10, 11, 12], max_new_tokens=2, grammar=_TerminatedGrammar())

        req.update_finish_state(new_accepted_len=2)

        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertEqual(req.finished_len, 2)

    def test_ignore_eos_preserves_length_finish(self):
        req = _make_req([10, EOS_ID], max_new_tokens=2, ignore_eos=True)

        req.update_finish_state()

        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertEqual(req.finished_len, 2)

    def test_non_stop_token_at_boundary_finishes_as_length(self):
        req = _make_req([10, 11], max_new_tokens=2)

        req.update_finish_state()

        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertEqual(req.finished_len, 2)

    def test_zero_token_budget_finishes_as_length(self):
        req = _make_req([], max_new_tokens=0)

        req.update_finish_state()

        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertEqual(req.finished_len, 0)

    def test_invalid_token_at_boundary_preserves_length_finish(self):
        req = _make_req([10, VOCAB_SIZE], max_new_tokens=2)

        req.update_finish_state()

        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertEqual(req.finished_len, 2)
        self.assertEqual(list(req.output_ids), [10, VOCAB_SIZE])

    def test_invalid_token_before_boundary_preserves_nan_finish(self):
        req = _make_req([10, VOCAB_SIZE], max_new_tokens=3)

        req.update_finish_state()

        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_STR)
        self.assertEqual(req.finished_reason.matched, "NaN happened")
        self.assertEqual(req.finished_len, 2)


if __name__ == "__main__":
    unittest.main()
