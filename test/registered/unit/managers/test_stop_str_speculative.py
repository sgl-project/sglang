"""Regression for stop-string / stop-regex finishing under speculative decoding
(multi-token commits): a stop committed mid-chunk must (1) trigger the finish
check and (2) set finished_len so the emitted output is trimmed at the stop, not
leaking tokens accepted after it. Drives the real `Req.update_finish_state` with
a fake tokenizer; pure CPU. Each test guards a distinct branch of
`_locate_str_stop_finished_len` / `_check_str_based_finish`."""

import unittest
from array import array

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

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


AAA_CANONICAL_ID = 80
A_ID = 32
FILLER_ID = 1000
EURO = "€"
EURO_CANONICAL_ID = 200
EURO_BYTES = list(EURO.encode("utf-8"))
# Every tokenizer id used below stays inside this vocab.
VOCAB_SIZE = 10_000


class _AaaTokenizer:
    """Canonical encode("AAA") is one token; generated [A, A, A] decodes to AAA."""

    eos_token_id = -1
    additional_stop_token_ids = None

    def encode(self, s, add_special_tokens=False):
        if s == "AAA":
            return [AAA_CANONICAL_ID]
        raise AssertionError(f"unexpected encode({s!r})")

    def decode(self, ids):
        pieces = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id == AAA_CANONICAL_ID:
                pieces.append("AAA")
            elif token_id == A_ID:
                pieces.append("A")
            else:
                pieces.append("x")
        return "".join(pieces)


class _EuroByteFallbackTokenizer:
    """Canonical encode("€") is one token; generated UTF-8 bytes decode
    to U+FFFD until the full 3-byte sequence is present."""

    eos_token_id = -1
    additional_stop_token_ids = None

    def encode(self, s, add_special_tokens=False):
        if s == EURO:
            return [EURO_CANONICAL_ID]
        raise AssertionError(f"unexpected encode({s!r})")

    def decode(self, ids):
        pieces = []
        buf = bytearray()

        def flush_bytes():
            if buf:
                pieces.append(bytes(buf).decode("utf-8", errors="replace"))
                buf.clear()

        for token_id in ids:
            token_id = int(token_id)
            if token_id == EURO_CANONICAL_ID:
                flush_bytes()
                pieces.append(EURO)
            elif 0 <= token_id <= 255:
                buf.append(token_id)
            else:
                flush_bytes()
                pieces.append("x")
        flush_bytes()
        return "".join(pieces)


class _DecodeCountingTokenizer:
    eos_token_id = -1
    additional_stop_token_ids = None

    def __init__(self):
        self.decode_calls = 0

    def encode(self, s, add_special_tokens=False):
        return [1]

    def decode(self, ids):
        self.decode_calls += 1
        return "x" * len(list(ids))


def _make_req(output_ids, stop=None, stop_regex=None, eos_token_ids=frozenset()):
    return _make_req_with_tokenizer(
        _FakeTokenizer(),
        output_ids,
        stop=stop,
        stop_regex=stop_regex,
        eos_token_ids=eos_token_ids,
        normalize_tokenizer=_MockTokenizerForNormalize(),
    )


def _make_req_with_tokenizer(
    tokenizer,
    output_ids,
    stop=None,
    stop_regex=None,
    eos_token_ids=frozenset(),
    normalize_tokenizer=None,
):
    sp = SamplingParams(max_new_tokens=1000, stop=stop, stop_regex=stop_regex)
    sp.normalize(tokenizer=normalize_tokenizer or tokenizer)
    req = Req(
        rid="t",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=sp,
        eos_token_ids=eos_token_ids,
        vocab_size=VOCAB_SIZE,
    )
    req.tokenizer = tokenizer
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

    def test_stop_regex_end_anchored_midchunk(self):
        """An end-anchored stop_regex that matches only after a token accepted
        mid-step ("a." in "a.b") must still finish the request."""
        req = _make_req([60, 61, 62], stop_regex=[r"[.!?]\s*$"])
        req.update_finish_state(new_accepted_len=3)
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_reason.matched, r"[.!?]\s*$")
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

    def test_noncanonical_aaa_sequential_accepts_one_finishes_at_first_match(self):
        """encode("AAA") is one token; generated [A,A,A] must still stop.

        One accepted token per step with a canonical window of 1 only
        searches two tokens and sees "AA".
        """
        tokenizer = _AaaTokenizer()
        req = _make_req_with_tokenizer(tokenizer, [FILLER_ID, FILLER_ID], stop=["AAA"])

        for _ in range(2):
            req.output_ids.append(A_ID)
            req.update_finish_state(new_accepted_len=1)
            self.assertFalse(req.finished())

        req.output_ids.append(A_ID)
        req.update_finish_state(new_accepted_len=1)
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_reason.matched, "AAA")
        self.assertEqual(req.finished_len, 5)

    def test_noncanonical_aaa_multi_accept_trims_at_first_match(self):
        """Stop "AAA" spans a prior [A,A] accept and a later [A, filler] accept."""
        tokenizer = _AaaTokenizer()
        req = _make_req_with_tokenizer(tokenizer, [FILLER_ID], stop=["AAA"])

        req.output_ids.extend([A_ID, A_ID])
        req.update_finish_state(new_accepted_len=2)
        self.assertFalse(req.finished())

        req.output_ids.extend([A_ID, FILLER_ID])
        req.update_finish_state(new_accepted_len=2)
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_reason.matched, "AAA")
        self.assertEqual(req.finished_len, 4)
        self.assertEqual(
            list(req.output_ids_through_stop), [FILLER_ID, A_ID, A_ID, A_ID]
        )

    def test_unicode_byte_fallback_sequential_accepts_one(self):
        """Partial UTF-8 bytes decode to U+FFFD until the full sequence exists."""
        tokenizer = _EuroByteFallbackTokenizer()
        self.assertEqual(tokenizer.decode([EURO_BYTES[0]]), "\ufffd")
        self.assertEqual(tokenizer.decode(EURO_BYTES[:2]), "\ufffd")
        self.assertEqual(tokenizer.decode(EURO_BYTES), EURO)
        self.assertEqual(
            tokenizer.encode(EURO, add_special_tokens=False), [EURO_CANONICAL_ID]
        )

        req = _make_req_with_tokenizer(tokenizer, [FILLER_ID], stop=[EURO])

        for byte_id in EURO_BYTES[:-1]:
            req.output_ids.append(byte_id)
            req.update_finish_state(new_accepted_len=1)
            self.assertFalse(req.finished())

        req.output_ids.append(EURO_BYTES[-1])
        req.update_finish_state(new_accepted_len=1)
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_reason.matched, EURO)
        self.assertEqual(req.finished_len, 4)

    def test_regex_only_lookback_has_no_string_window(self):
        req = _make_req([10, 11, 12, 13], stop_regex=[r"[a-z]{3}"])
        self.assertEqual(req.sampling_params.stop_str_max_len, 0)
        self.assertEqual(
            req._stop_match_tail_len(1), req.sampling_params.stop_regex_max_len + 1
        )

    def test_no_stop_update_does_not_call_decode(self):
        tokenizer = _DecodeCountingTokenizer()
        req = _make_req_with_tokenizer(tokenizer, [10, 11, 12])
        req.update_finish_state(new_accepted_len=1)
        self.assertFalse(req.finished())
        self.assertEqual(tokenizer.decode_calls, 0)


if __name__ == "__main__":
    unittest.main()
