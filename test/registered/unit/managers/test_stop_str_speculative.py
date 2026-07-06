"""Regression for stop-string / stop-regex finishing under speculative decoding
(multi-token commits): a stop committed mid-chunk must (1) trigger the finish
check and (2) set finished_len so the emitted output is trimmed at the stop, not
leaking tokens accepted after it. Drives the real `Req.update_finish_state` with
a fake tokenizer; pure CPU. Each test guards a distinct branch of
`_locate_str_stop_finished_len` / `_check_str_based_finish`."""

import unittest
from array import array

from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    FINISH_MATCHED_STR,
    FINISH_MATCHED_TOKEN,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Token id -> decoded text; decode() concatenates. Distinct symbols so no
# accidental cross-matches (10-39 are lowercase letters).
STOP_ID = 1
EOS_ID = 2
ID_TO_TEXT = {
    STOP_ID: "STOP",
    EOS_ID: "",
    **{i: chr(ord("a") + i % 26) for i in range(10, 40)},
    60: "a",
    61: ".",
    62: "b",
    63: ".",
    70: "X",
    71: "Y",
}

# "STOP" (index 3) sits 6 tokens back: outside the old (stop_str_max_len + 1)
# window, inside the one widened by new_accepted_len.
MIDCHUNK = [10, 11, 12, STOP_ID, 20, 21, 22, 23, 24]


class _FakeTokenizer:
    eos_token_id = -1
    additional_stop_token_ids = None

    def decode(self, ids):
        return "".join(ID_TO_TEXT[int(i)] for i in ids)


class _MockTokenizerForNormalize:
    """Mock tokenizer for normalize() - returns char-count as token list."""

    def encode(self, s, add_special_tokens=False):
        return list(range(len(s)))  # One "token" per character


def _make_req(
    output_ids,
    stop=None,
    stop_regex=None,
    eos_token_ids=frozenset(),
    max_new_tokens=1000,
):
    sp = SamplingParams(max_new_tokens=max_new_tokens, stop=stop, stop_regex=stop_regex)
    sp.normalize(tokenizer=_MockTokenizerForNormalize())  # char-based stop_str_max_len
    req = Req(
        rid="t",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=sp,
        eos_token_ids=eos_token_ids,
        vocab_size=10_000,
    )
    req.tokenizer = _FakeTokenizer()
    req.output_ids = array("q", output_ids)
    return req


class TestStopStrSpeculative(unittest.TestCase):
    def test_no_stop_does_not_finish(self):
        req = _make_req([10, 11, 12, 20, 21, 22, 23, 24], stop=["STOP"])
        req.update_finish_state(new_accepted_len=6)
        self.assertFalse(req.finished())

    # --- _locate_str_stop_finished_len: loop match / fallback / empty loop / span ---
    def test_stop_str_midchunk(self):
        # Loop match: "STOP" (index 3) finishes; finished_len lands just past it,
        # so the 5 trailing tokens are dropped.
        req = _make_req(MIDCHUNK, stop=["STOP"])
        req.update_finish_state(new_accepted_len=6)
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_reason.matched, "STOP")
        self.assertEqual(req.finished_len, 4)

    def test_stop_str_wins_over_eos_in_same_step(self):
        # A step accepting both the stop string and EOS must finish as
        # FINISH_MATCHED_STR with finished_len past the stop string, not as
        # FINISH_MATCHED_TOKEN trimming only the EOS token (which leaks "STOP").
        req = _make_req(
            [10, 11, STOP_ID, EOS_ID], stop=["STOP"], eos_token_ids={EOS_ID}
        )
        req.update_finish_state(new_accepted_len=4)
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_reason.matched, "STOP")
        self.assertEqual(req.finished_len, 3)

    def test_stop_str_at_chunk_end_uses_full_len(self):
        # Stop is the last token -> loop never matches before the full window ->
        # fallback returns len(output_ids).
        req = _make_req([10, 11, 12, 20, 21, STOP_ID], stop=["STOP"])
        req.update_finish_state(new_accepted_len=6)
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_len, 6)

    def test_stop_str_non_spec_single_token(self):
        # new_accepted_len == 1: locate's range is empty -> fallback (non-spec
        # path preserved, no extra decode).
        req = _make_req([10, 11, STOP_ID], stop=["STOP"])
        req.update_finish_state(new_accepted_len=1)
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_len, 3)

    def test_stop_str_spanning_two_tokens(self):
        # "XY" completes only once both tokens (70, 71) are decoded -> finished_len
        # covers both; trailing tokens dropped.
        req = _make_req([10, 11, 70, 71, 20, 21], stop=["XY"])
        req.update_finish_state(new_accepted_len=6)
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_len, 4)

    # --- regex matched() branch ---
    def test_stop_regex_midchunk(self):
        req = _make_req([10, 11, 70, 71, 20, 21], stop_regex=[r"XY"])
        req.update_finish_state(new_accepted_len=6)
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_reason.matched, r"XY")
        self.assertEqual(req.finished_len, 4)

    def test_stop_regex_end_anchored_trims_at_first_match(self):
        # Documents current behavior: text "a.b." matches `\.$` at the chunk end,
        # but locate scans growing prefixes and stops at the FIRST period ("a."),
        # so finished_len == 2 and "b." is dropped.
        req = _make_req([60, 61, 62, 63], stop_regex=[r"\.$"])
        req.update_finish_state(new_accepted_len=4)
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_reason.matched, r"\.$")
        self.assertEqual(req.finished_len, 2)

    # --- decoded_text-only branch ---
    def test_stop_str_only_in_decoded_text_sets_no_finished_len(self):
        # Documents current behavior: when the stop is only in decoded_text (not
        # the tail), the request finishes but finished_len is left unset.
        req = _make_req([10, 11, 12], stop=["STOP"])  # no STOP in output tokens
        req.decoded_text = "earlier STOP text"
        req.update_finish_state(new_accepted_len=3)
        self.assertTrue(req.finished())
        self.assertIsNone(req.finished_len)


class TestFinishLengthVsStopSpeculative(unittest.TestCase):
    """A stop token/string inside a multi-token accepted run that crosses
    max_new_tokens must be honored (correct finish_reason + trim at the stop),
    not masked by the length limit. Guards the ordering/clamping between the
    length check and the stop checks in `update_finish_state`.

    Setup for all: max_new_tokens=10, output_ids has 11 tokens (an accepted run
    of 3 crossed the budget), new_accepted_len=3 -> the run is indices 8,9,10.
    """

    def test_eos_within_budget_beats_length(self):
        # EOS at index 8 (finished_len 9 <= 10): the stop token wins over length,
        # and the token leaked after it (index 9) is trimmed.
        req = _make_req(
            [10, 11, 12, 13, 14, 15, 16, 17, EOS_ID, 20, 21],
            eos_token_ids={EOS_ID},
            max_new_tokens=10,
        )
        req.update_finish_state(new_accepted_len=3)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_reason.matched, EOS_ID)
        self.assertEqual(req.finished_len, 9)

    def test_eos_at_budget_boundary_is_stop_not_length(self):
        # EOS at index 9 -> finished_len 10 == the length cap. finished_len is the
        # same either way, but the reason must be 'stop', not 'length'.
        req = _make_req(
            [10, 11, 12, 13, 14, 15, 16, 17, 20, EOS_ID, 21],
            eos_token_ids={EOS_ID},
            max_new_tokens=10,
        )
        req.update_finish_state(new_accepted_len=3)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_len, 10)

    def test_eos_past_budget_is_length_limited(self):
        # EOS at index 10, past the budget: no stop within budget, so the request
        # is length-limited and trimmed to max_new_tokens (no leak).
        req = _make_req(
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 20, EOS_ID],
            eos_token_ids={EOS_ID},
            max_new_tokens=10,
        )
        req.update_finish_state(new_accepted_len=3)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertEqual(req.finished_len, 10)

    def test_stop_str_within_budget_beats_length(self):
        # A stop string inside the over-budget run is honored, not masked by the
        # length cap. Stop lands within budget so its reason wins.
        req = _make_req(
            [10, 11, 12, 13, 14, 15, 16, 17, STOP_ID, 20, 21],
            stop=["STOP"],
            max_new_tokens=10,
        )
        req.update_finish_state(new_accepted_len=3)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_STR)
        self.assertEqual(req.finished_reason.matched, "STOP")
        self.assertIsNotNone(req.finished_len)
        self.assertLessEqual(req.finished_len, 10)

    def test_no_stop_over_budget_is_length(self):
        # Baseline (unchanged): no stop in the run -> FINISH_LENGTH at the cap.
        req = _make_req(list(range(10, 21)), max_new_tokens=10)
        req.update_finish_state(new_accepted_len=3)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertEqual(req.finished_len, 10)


if __name__ == "__main__":
    unittest.main()
