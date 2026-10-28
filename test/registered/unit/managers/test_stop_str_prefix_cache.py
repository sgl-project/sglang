"""CPU regressions for streaming stop-prefix checks and tail decode reuse."""

import random
import unittest
from array import array
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.managers import schedule_batch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _CountingTokenizer:
    eos_token_id = -1
    additional_stop_token_ids = None

    def __init__(self, replacements=None):
        self.calls = 0
        self.replacements = replacements or {}

    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))

    def decode(self, ids):
        self.calls += 1
        return "".join(self.replacements.get(int(i), chr(int(i))) for i in ids)


def _make_req(text, *, stops=("XY",), stream=True, regex=None):
    tokenizer = _CountingTokenizer()
    params = SamplingParams(max_new_tokens=1000, stop=list(stops), stop_regex=regex)
    params.normalize(tokenizer=tokenizer)
    req = Req(
        rid="stop-prefix-test",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=params,
        eos_token_ids=set(),
        vocab_size=0x110000,
        stream=stream,
    )
    req.tokenizer = tokenizer
    req.output_ids = array("q", map(ord, text))
    return req


class TestStopStringPrefixCache(unittest.TestCase):
    def setUp(self):
        enabled = patch.object(schedule_batch, "_ENABLE_STOP_STR_OPTIMIZATION", True)
        enabled.start()
        self.addCleanup(enabled.stop)

    def test_streaming_finish_and_prefix_share_decode(self):
        req = _make_req("ordinary.")
        self.assertFalse(req._check_str_based_finish())
        self.assertFalse(req.check_match_stop_str_prefix())
        self.assertEqual(req.tokenizer.calls, 1)

    def test_nonstream_does_not_cache(self):
        req = _make_req("ordinary.", stream=False)
        self.assertEqual(req.tail_str(), "ry.")
        self.assertEqual(req.tail_str(), "ry.")
        self.assertEqual(req.tokenizer.calls, 2)
        self.assertFalse(hasattr(req, "_stop_tail_cache"))

    def test_regex_does_not_cache_whole_output(self):
        req = _make_req("ordinary.", regex=["x.*Y"])
        self.assertEqual(req.tail_str(), "ordinary.")
        self.assertEqual(req.tail_str(), "ordinary.")
        self.assertEqual(req.tokenizer.calls, 2)
        self.assertFalse(hasattr(req, "_stop_tail_cache"))

    def test_no_stops_does_not_decode(self):
        req = _make_req("ordinary.", stops=())
        self.assertEqual(req.tail_str(), "")
        self.assertFalse(req.check_match_stop_str_prefix())
        self.assertEqual(req.tokenizer.calls, 0)

    def test_empty_output(self):
        req = _make_req("")
        self.assertEqual(req.tail_str(), "")
        self.assertFalse(req.check_match_stop_str_prefix())

    def test_changed_content_with_same_length_and_last_id(self):
        req = _make_req("abc")
        self.assertEqual(req.tail_str(), "abc")
        req.output_ids[0] = ord("z")
        self.assertEqual(req.tail_str(), "zbc")
        self.assertEqual(req.tokenizer.calls, 2)

    def test_append_retract_and_regrow(self):
        req = _make_req("abc")
        self.assertEqual(req.tail_str(), "abc")
        req.output_ids.append(ord("d"))
        self.assertEqual(req.tail_str(), "bcd")
        del req.output_ids[-2:]
        self.assertEqual(req.tail_str(), "ab")
        req.output_ids.extend(map(ord, "ed"))
        self.assertEqual(req.tail_str(), "bed")

    def test_changed_tokenizer(self):
        req = _make_req("abc")
        self.assertEqual(req.tail_str(), "abc")
        req.tokenizer = _CountingTokenizer({ord("a"): "z"})
        self.assertEqual(req.tail_str(), "zbc")
        self.assertEqual(req.tokenizer.calls, 1)

    def test_changed_window_and_multi_token_accept(self):
        req = _make_req("abcdef")
        self.assertEqual(req.tail_str(), "def")
        self.assertEqual(req.tail_str(new_accepted_len=4), "abcdef")
        self.assertEqual(req.tail_str(), "def")
        req.sampling_params.stop_str_max_len = 4
        self.assertEqual(req.tail_str(), "bcdef")

    def test_stream_toggle_preserves_values(self):
        req = _make_req("abc")
        self.assertEqual(req.tail_str(), "abc")
        req.stream = False
        req.output_ids[0] = ord("z")
        self.assertEqual(req.tail_str(), "zbc")
        req.stream = True
        self.assertEqual(req.tail_str(), "zbc")

    def test_full_matches_precede_last_character_guard(self):
        # The last character is absent from the stop, but a complete match
        # earlier in the decoded tail must still hold the stream.
        for text, stops in [
            ("before<STOP>after.", ["<STOP>"]),
            ("beforeé💥after.", ["é💥"]),
        ]:
            with self.subTest(text=text):
                req = _make_req(text, stops=stops)
                req.sampling_params.stop_str_max_len = len(text)
                self.assertTrue(req.check_match_stop_str_prefix())

    def test_partial_unicode_overlapping_and_empty_stops(self):
        for text, stops, expected in [
            ("hello X", ["XY"], True),
            ("hello .", ["XY"], False),
            ("hello你", ["你好"], True),
            ("hello💥a", ["💥abc"], True),
            ("e\u0301", ["\u0301x"], True),
            ("a\n", ["\n\n"], True),
            ("ababa", ["aba", "ab"], True),
            ("abc", [""], False),
            ("", ["XY"], False),
        ]:
            with self.subTest(text=text, stops=stops):
                req = _make_req(text, stops=stops)
                req.sampling_params.stop_str_max_len = len(text)
                self.assertEqual(req.check_match_stop_str_prefix(), expected)

    def test_random_unicode_matches_prefix_definition(self):
        rng = random.Random(28)
        alphabet = "abé💥\u0301<>_\n。"
        req = _make_req("")
        for _ in range(2000):
            text = "".join(rng.choices(alphabet, k=rng.randrange(30)))
            stops = [
                "".join(rng.choices(alphabet, k=rng.randrange(10)))
                for _ in range(rng.randrange(5))
            ]
            req.output_ids = array("q", map(ord, text))
            req.sampling_params.stop_strs = stops
            req.sampling_params.stop_str_max_len = len(text)
            expected = (
                any(
                    stop in text
                    or any(text.endswith(stop[:n]) for n in range(1, len(stop) + 1))
                    for stop in stops
                    if stop
                )
                if text
                else False
            )
            self.assertEqual(req.check_match_stop_str_prefix(), expected)


class TestStopStringOptimizationOptIn(unittest.TestCase):
    def test_environment_flag_defaults_off(self):
        self.assertFalse(envs.SGLANG_ENABLE_STOP_STR_OPTIMIZATION.default)
        with envs.SGLANG_ENABLE_STOP_STR_OPTIMIZATION.override(True):
            self.assertTrue(envs.SGLANG_ENABLE_STOP_STR_OPTIMIZATION.get())
        with envs.SGLANG_ENABLE_STOP_STR_OPTIMIZATION.override(False):
            self.assertFalse(envs.SGLANG_ENABLE_STOP_STR_OPTIMIZATION.get())

    def test_disabled_path_does_not_cache_or_prune(self):
        class StopWithoutMembershipCheck(str):
            def __contains__(self, value):
                raise AssertionError("Disabled path must not use the prefix guard")

        with patch.object(schedule_batch, "_ENABLE_STOP_STR_OPTIMIZATION", False):
            req = _make_req("ordinary.")
            req.sampling_params.stop_strs = [StopWithoutMembershipCheck("XY")]
            self.assertFalse(req._check_str_based_finish())
            self.assertFalse(req.check_match_stop_str_prefix())
            self.assertEqual(req.tokenizer.calls, 2)
            self.assertFalse(hasattr(req, "_stop_tail_cache"))


if __name__ == "__main__":
    unittest.main()
