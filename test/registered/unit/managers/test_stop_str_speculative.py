"""Regression for stop-string / stop-regex finishing under speculative decoding
(multi-token commits): a stop committed mid-chunk must (1) trigger the finish
check and (2) set finished_len so the emitted output is trimmed at the stop, not
leaking tokens accepted after it. Drives the real `Req.update_finish_state` with
a fake tokenizer; pure CPU. Each test guards a distinct branch of
`_locate_str_stop_finished_len` / `_check_str_based_finish`."""

import unittest
from array import array
from unittest.mock import patch

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

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


def _make_req(output_ids, stop=None, stop_regex=None, eos_token_ids=frozenset()):
    sp = SamplingParams(max_new_tokens=1000, stop=stop, stop_regex=stop_regex)
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


class TestStopStrSpeculative(CustomTestCase):
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
        req = _make_req([10, 11, STOP_ID], stop=["OP", "ST", "STOP"])
        with patch.object(
            req.tokenizer, "decode", wraps=req.tokenizer.decode
        ) as decode:
            req.update_finish_state()
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_reason.matched, "OP")
        self.assertEqual(req.finished_len, 3)
        self.assertEqual(decode.call_count, 1)

    def test_stop_str_spanning_two_tokens(self):
        # "XY" completes only once both tokens (70, 71) are decoded -> finished_len
        # covers both; trailing tokens dropped.
        req = _make_req([10, 11, 70, 71, 20, 21], stop=["XY"])
        req.update_finish_state(new_accepted_len=6)
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_len, 4)

    def test_multiple_stop_strs_match_sequential_token_boundary(self):
        # A later-listed stop can complete before the first-listed stop within
        # one accepted chunk. Selection must match one-token-at-a-time decoding.
        cases = [
            ([], [10, STOP_ID, 70, 71], ["XY", "STOP"], "STOP", 2, "k"),
            ([10, 70], [71, STOP_ID, 11], ["STOP", "XY"], "XY", 3, "k"),
            ([], [10, STOP_ID, 70, 71], ["STOPXY", "STOP"], "STOP", 2, "k"),
        ]
        manager = DetokenizerManager.__new__(DetokenizerManager)
        for prefix, chunk, stops, matched, finished_len, text in cases:
            for stop_order in (stops, stops[::-1]):
                with self.subTest(prefix=prefix, chunk=chunk, stops=stop_order):
                    req = _make_req(prefix + chunk, stop=stop_order)
                    req.update_finish_state(new_accepted_len=len(chunk))
                    sequential = _make_req(prefix, stop=stop_order)
                    for token in chunk:
                        sequential.output_ids.append(token)
                        sequential.update_finish_state()
                        if sequential.finished():
                            break

                    self.assertTrue(req.finished())
                    self.assertEqual(req.finished_reason.matched, matched)
                    self.assertEqual(req.finished_len, finished_len)
                    self.assertEqual(
                        req.finished_reason.to_json(),
                        sequential.finished_reason.to_json(),
                    )
                    self.assertEqual(req.finished_len, sequential.finished_len)
                    self.assertEqual(
                        req.output_ids_through_stop, sequential.output_ids_through_stop
                    )
                    self.assertEqual(list(req.output_ids), prefix + chunk)
                    for no_stop_trim in (False, True):
                        self.assertEqual(
                            manager.trim_matched_stop(
                                req.tokenizer.decode(req.output_ids_through_stop),
                                req.finished_reason.to_json(),
                                no_stop_trim,
                            ),
                            text + matched if no_stop_trim else text,
                        )

    def test_stop_strs_completing_in_same_token_keep_list_priority(self):
        # Both strings appear in the STOP token. Character position must not
        # override list priority when the first matching token is the same.
        for stop_order in (["OP", "ST"], ["ST", "OP"]):
            for chunk in ([STOP_ID], [STOP_ID, 70, 71]):
                with self.subTest(stops=stop_order, chunk=chunk):
                    req = _make_req([10] + chunk, stop=stop_order)
                    req.update_finish_state(new_accepted_len=len(chunk))
                    self.assertEqual(req.finished_reason.matched, stop_order[0])
                    self.assertEqual(req.finished_len, 2)
                    self.assertEqual(list(req.output_ids_through_stop), [10, STOP_ID])

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

    def test_decoded_text_fallback_keeps_existing_list_priority(self):
        for stops, output_ids, matched, finished_len in [
            (["XY", "STOP"], [10, STOP_ID, 11], "XY", None),
            (["STOP", "XY"], [10, STOP_ID, 11], "STOP", 2),
            (["XY", "fallback", "STOP"], [10, STOP_ID, 70, 71], "STOP", 2),
        ]:
            with self.subTest(stops=stops):
                req = _make_req(output_ids, stop=stops)
                req.decoded_text = "earlier XY fallback text"
                req.update_finish_state(new_accepted_len=len(output_ids))
                self.assertEqual(req.finished_reason.matched, matched)
                self.assertEqual(req.finished_len, finished_len)


if __name__ == "__main__":
    unittest.main()
