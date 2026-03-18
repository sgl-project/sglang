"""Unit tests for srt/sampling/sampling_params.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")

import unittest
from unittest.mock import MagicMock

from sglang.srt.sampling.sampling_params import (
    MAX_LEN,
    TOP_K_ALL,
    SamplingParams,
    _max_length_from_subpattern,
    get_max_seq_length,
)


# ---------------------------------------------------------------------------
# SamplingParams.__init__ — implicit conversions
# ---------------------------------------------------------------------------
class TestSamplingParamsInit(unittest.TestCase):
    """Test the two implicit conversions that happen in __init__."""

    def test_zero_temperature_becomes_greedy(self):
        """temperature=0 should trigger greedy mode: top_k=1, temperature=1.0."""
        sp = SamplingParams(temperature=0.0)
        self.assertEqual(sp.top_k, 1)
        self.assertEqual(sp.temperature, 1.0)

    def test_near_zero_temperature_becomes_greedy(self):
        """temperature just below 1e-6 should also trigger greedy."""
        sp = SamplingParams(temperature=1e-7)
        self.assertEqual(sp.top_k, 1)
        self.assertEqual(sp.temperature, 1.0)

    def test_temperature_at_eps_boundary_not_greedy(self):
        """temperature exactly at 1e-6 should NOT trigger greedy (< not <=)."""
        sp = SamplingParams(temperature=1e-6)
        self.assertEqual(sp.temperature, 1e-6)
        # top_k should remain at TOP_K_ALL (from -1 default)
        self.assertEqual(sp.top_k, TOP_K_ALL)

    def test_negative_temperature_not_modified(self):
        """Negative temperature is invalid but __init__ doesn't reject it —
        that's verify()'s job. Confirm __init__ leaves it unchanged."""
        sp = SamplingParams(temperature=-1.0)
        self.assertEqual(sp.temperature, -1.0)

    def test_top_k_minus_one_becomes_top_k_all(self):
        """top_k=-1 means 'whole vocabulary', converted to TOP_K_ALL."""
        sp = SamplingParams(top_k=-1)
        self.assertEqual(sp.top_k, TOP_K_ALL)

    def test_positive_top_k_preserved(self):
        """An explicit positive top_k should be kept as-is."""
        sp = SamplingParams(top_k=50)
        self.assertEqual(sp.top_k, 50)

    def test_stop_token_ids_stored_as_set(self):
        """stop_token_ids list should be converted to a set for O(1) lookup."""
        sp = SamplingParams(stop_token_ids=[1, 2, 3])
        self.assertIsInstance(sp.stop_token_ids, set)
        self.assertEqual(sp.stop_token_ids, {1, 2, 3})

    def test_stop_token_ids_none_stays_none(self):
        sp = SamplingParams(stop_token_ids=None)
        self.assertIsNone(sp.stop_token_ids)


# ---------------------------------------------------------------------------
# SamplingParams.verify — parameter validation
# ---------------------------------------------------------------------------
class TestSamplingParamsVerify(unittest.TestCase):
    """Test that verify() rejects invalid values and accepts valid ones."""

    VOCAB_SIZE = 32000

    def _make(self, **kwargs):
        """Helper: create SamplingParams with safe defaults, override with kwargs."""
        defaults = dict(temperature=1.0, top_p=1.0, top_k=10, min_p=0.0)
        defaults.update(kwargs)
        return SamplingParams(**defaults)

    # --- happy path ---
    def test_valid_params_pass(self):
        sp = self._make()
        sp.verify(self.VOCAB_SIZE)  # should not raise

    # --- temperature ---
    def test_negative_temperature_raises(self):
        sp = self._make(temperature=-0.5)
        with self.assertRaises(ValueError):
            sp.verify(self.VOCAB_SIZE)

    # --- top_p ---
    def test_top_p_zero_raises(self):
        """top_p=0 is not in (0, 1]."""
        sp = self._make(top_p=0.0)
        with self.assertRaises(ValueError):
            sp.verify(self.VOCAB_SIZE)

    def test_top_p_above_one_raises(self):
        sp = self._make(top_p=1.1)
        with self.assertRaises(ValueError):
            sp.verify(self.VOCAB_SIZE)

    def test_top_p_exactly_one_is_valid(self):
        sp = self._make(top_p=1.0)
        sp.verify(self.VOCAB_SIZE)  # should not raise

    def test_top_p_small_positive_is_valid(self):
        sp = self._make(top_p=0.01)
        sp.verify(self.VOCAB_SIZE)

    # --- min_p ---
    def test_min_p_negative_raises(self):
        sp = self._make(min_p=-0.1)
        with self.assertRaises(ValueError):
            sp.verify(self.VOCAB_SIZE)

    def test_min_p_above_one_raises(self):
        sp = self._make(min_p=1.01)
        with self.assertRaises(ValueError):
            sp.verify(self.VOCAB_SIZE)

    def test_min_p_boundaries_valid(self):
        """Both 0.0 and 1.0 are valid."""
        self._make(min_p=0.0).verify(self.VOCAB_SIZE)
        self._make(min_p=1.0).verify(self.VOCAB_SIZE)

    # --- top_k ---
    def test_top_k_zero_raises(self):
        """top_k=0 is invalid (must be >=1 or -1 for all)."""
        sp = self._make()
        sp.top_k = 0  # bypass __init__ conversion
        with self.assertRaises(ValueError):
            sp.verify(self.VOCAB_SIZE)

    # --- frequency_penalty ---
    def test_frequency_penalty_below_minus_two_raises(self):
        sp = self._make(frequency_penalty=-2.1)
        with self.assertRaises(ValueError):
            sp.verify(self.VOCAB_SIZE)

    def test_frequency_penalty_above_two_raises(self):
        sp = self._make(frequency_penalty=2.1)
        with self.assertRaises(ValueError):
            sp.verify(self.VOCAB_SIZE)

    def test_frequency_penalty_boundaries_valid(self):
        self._make(frequency_penalty=-2.0).verify(self.VOCAB_SIZE)
        self._make(frequency_penalty=2.0).verify(self.VOCAB_SIZE)

    # --- presence_penalty ---
    def test_presence_penalty_out_of_range_raises(self):
        sp = self._make(presence_penalty=2.5)
        with self.assertRaises(ValueError):
            sp.verify(self.VOCAB_SIZE)

    # --- repetition_penalty ---
    def test_repetition_penalty_negative_raises(self):
        sp = self._make(repetition_penalty=-0.1)
        with self.assertRaises(ValueError):
            sp.verify(self.VOCAB_SIZE)

    def test_repetition_penalty_above_two_raises(self):
        sp = self._make(repetition_penalty=2.1)
        with self.assertRaises(ValueError):
            sp.verify(self.VOCAB_SIZE)

    def test_repetition_penalty_boundaries_valid(self):
        self._make(repetition_penalty=0.0).verify(self.VOCAB_SIZE)
        self._make(repetition_penalty=2.0).verify(self.VOCAB_SIZE)

    # --- min_new_tokens / max_new_tokens ---
    def test_negative_min_new_tokens_raises(self):
        sp = self._make(min_new_tokens=-1)
        with self.assertRaises(ValueError):
            sp.verify(self.VOCAB_SIZE)

    def test_negative_max_new_tokens_raises(self):
        sp = self._make(max_new_tokens=-1)
        with self.assertRaises(ValueError):
            sp.verify(self.VOCAB_SIZE)

    def test_min_exceeds_max_new_tokens_raises(self):
        sp = self._make(min_new_tokens=100, max_new_tokens=50)
        with self.assertRaises(ValueError):
            sp.verify(self.VOCAB_SIZE)

    def test_min_equals_max_new_tokens_valid(self):
        sp = self._make(min_new_tokens=10, max_new_tokens=10)
        sp.verify(self.VOCAB_SIZE)

    # --- logit_bias ---
    def test_logit_bias_token_exceeds_vocab_raises(self):
        sp = self._make(logit_bias={"99999": 1.0})
        with self.assertRaises(ValueError):
            sp.verify(self.VOCAB_SIZE)

    def test_logit_bias_negative_token_raises(self):
        sp = self._make(logit_bias={"-1": 1.0})
        with self.assertRaises(ValueError):
            sp.verify(self.VOCAB_SIZE)

    def test_logit_bias_valid_tokens(self):
        sp = self._make(logit_bias={"0": 1.0, "31999": -0.5})
        sp.verify(self.VOCAB_SIZE)

    # --- grammar mutual exclusion ---
    def test_multiple_grammars_raises(self):
        sp = self._make(json_schema='{"type":"object"}', regex="abc")
        with self.assertRaises(ValueError):
            sp.verify(self.VOCAB_SIZE)

    def test_single_grammar_valid(self):
        sp = self._make(json_schema='{"type":"object"}')
        sp.verify(self.VOCAB_SIZE)

    def test_all_three_grammars_set_raises(self):
        sp = self._make(json_schema='{}', regex="a", ebnf="rule")
        with self.assertRaises(ValueError):
            sp.verify(self.VOCAB_SIZE)


# ---------------------------------------------------------------------------
# SamplingParams.normalize — stop string processing
# ---------------------------------------------------------------------------
class TestSamplingParamsNormalize(unittest.TestCase):
    """Test that normalize() correctly processes stop strings and regex."""

    def test_none_stop_strs_becomes_empty_list(self):
        sp = SamplingParams(stop=None)
        sp.normalize(tokenizer=None)
        self.assertEqual(sp.stop_strs, [])
        self.assertEqual(sp.stop_str_max_len, 0)

    def test_string_stop_str_wrapped_in_list(self):
        sp = SamplingParams(stop="<|end|>")
        sp.normalize(tokenizer=None)
        self.assertEqual(sp.stop_strs, ["<|end|>"])

    def test_list_stop_strs_unchanged(self):
        sp = SamplingParams(stop=["stop1", "stop2"])
        sp.normalize(tokenizer=None)
        self.assertEqual(sp.stop_strs, ["stop1", "stop2"])

    def test_stop_str_max_len_without_tokenizer(self):
        """Without a tokenizer, max_len is the raw string character count."""
        sp = SamplingParams(stop=["ab", "cdef"])
        sp.normalize(tokenizer=None)
        self.assertEqual(sp.stop_str_max_len, 4)  # len("cdef")

    def test_stop_str_max_len_with_tokenizer(self):
        """With a tokenizer, max_len counts encoded token IDs."""
        tokenizer = MagicMock()
        # "hello" encodes to 2 tokens, "world!!" to 3 tokens
        tokenizer.encode.side_effect = lambda s, add_special_tokens=False: {
            "hello": [101, 102],
            "world!!": [201, 202, 203],
        }[s]
        sp = SamplingParams(stop=["hello", "world!!"])
        sp.normalize(tokenizer=tokenizer)
        self.assertEqual(sp.stop_str_max_len, 3)

    def test_none_stop_regex_becomes_empty_list(self):
        sp = SamplingParams(stop_regex=None)
        sp.normalize(tokenizer=None)
        self.assertEqual(sp.stop_regex_strs, [])
        self.assertEqual(sp.stop_regex_max_len, 0)

    def test_string_stop_regex_wrapped_in_list(self):
        sp = SamplingParams(stop_regex=r"\d+")
        sp.normalize(tokenizer=None)
        self.assertEqual(sp.stop_regex_strs, [r"\d+"])

    def test_stop_regex_max_len_computed(self):
        """Bounded regex should compute a finite max length."""
        sp = SamplingParams(stop_regex=r"[a-z]{3}")
        sp.normalize(tokenizer=None)
        self.assertEqual(sp.stop_regex_max_len, 3)


# ---------------------------------------------------------------------------
# get_max_seq_length / _max_length_from_subpattern — regex analysis
# ---------------------------------------------------------------------------
class TestRegexMaxLength(unittest.TestCase):
    """Test the recursive regex AST length calculator."""

    def test_literal_string(self):
        """'abc' → 3 literals."""
        self.assertEqual(get_max_seq_length("abc"), 3)

    def test_character_class(self):
        """'[a-z]' → 1 (one character from set)."""
        self.assertEqual(get_max_seq_length("[a-z]"), 1)

    def test_dot_any(self):
        """'.' matches any one character → 1."""
        self.assertEqual(get_max_seq_length("."), 1)

    def test_unbounded_star(self):
        """'a*' → unbounded repeat → MAX_LEN."""
        result = get_max_seq_length("a*")
        self.assertEqual(result, MAX_LEN)

    def test_unbounded_plus(self):
        """'a+' → at least 1, unbounded → MAX_LEN."""
        result = get_max_seq_length("a+")
        self.assertEqual(result, MAX_LEN)

    def test_bounded_repeat(self):
        """'a{5}' → exactly 5 × 1 = 5."""
        self.assertEqual(get_max_seq_length("a{5}"), 5)

    def test_bounded_range_repeat(self):
        """'a{2,4}' → upper bound is 4 × 1 = 4."""
        self.assertEqual(get_max_seq_length("a{2,4}"), 4)

    def test_branch_takes_max(self):
        """'abc|de' → max(3, 2) = 3."""
        self.assertEqual(get_max_seq_length("abc|de"), 3)

    def test_subpattern_group(self):
        """'(abc)' → 3 from the group content."""
        self.assertEqual(get_max_seq_length("(abc)"), 3)

    def test_zero_width_assertions_ignored(self):
        """'^abc$' → anchors contribute 0, result = 3."""
        self.assertEqual(get_max_seq_length("^abc$"), 3)

    def test_complex_pattern(self):
        """'(foo|bar)\\d{2}' → max(3,3) + 2 = 5."""
        self.assertEqual(get_max_seq_length(r"(foo|bar)\d{2}"), 5)

    def test_nested_groups(self):
        """'((ab))' → nested subpatterns still = 2."""
        self.assertEqual(get_max_seq_length("((ab))"), 2)

    def test_question_mark_optional(self):
        """'a?' → {0,1} repeat → 1 × 1 = 1."""
        self.assertEqual(get_max_seq_length("a?"), 1)

    def test_mixed_unbounded_and_bounded(self):
        """'ab+c{3}' → 1 + MAX_LEN + 3. Since MAX_LEN dominates,
        the total includes it."""
        result = get_max_seq_length("ab+c{3}")
        self.assertGreaterEqual(result, MAX_LEN)

    def test_empty_regex(self):
        """Empty regex matches empty string → length 0."""
        self.assertEqual(get_max_seq_length(""), 0)

    def test_lookahead_triggers_unhandled_token(self):
        """Lookahead (?=...) produces an ASSERT token which is not handled —
        the else branch should log a warning and add MAX_LEN."""
        result = get_max_seq_length("(?=a)b")
        # ASSERT (lookahead) → MAX_LEN, LITERAL 'b' → 1
        self.assertGreaterEqual(result, MAX_LEN)

    def test_lookbehind_triggers_unhandled_token(self):
        """Lookbehind (?<=...) also produces an unhandled ASSERT token."""
        result = get_max_seq_length("(?<=x)y")
        self.assertGreaterEqual(result, MAX_LEN)


if __name__ == "__main__":
    unittest.main()
