"""Unit tests for srt/sampling/custom_logit_processor.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")

import json
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.sampling.custom_logit_processor import (
    CustomLogitProcessor,
    DeepseekOCRNoRepeatNGramLogitProcessor,
    DeepSeekR1ThinkingBudgetLogitProcessor,
    DisallowedTokensLogitsProcessor,
    Qwen3ThinkingBudgetLogitProcessor,
    _cache_from_str,
)
from sglang.test.test_utils import CustomTestCase


# Helper: mock a Req object (used by ThinkingBudget and NGram processors)
def _make_req(origin_input_ids=None, output_ids=None):
    req = MagicMock()
    req.origin_input_ids = origin_input_ids or []
    req.output_ids = output_ids or []
    return req


# Serialization round-trip
class TestCustomLogitProcessorSerialization(CustomTestCase):

    def test_to_str_produces_valid_json(self):
        """Test that to_str() produces valid JSON with a 'callable' field."""
        s = DisallowedTokensLogitsProcessor.to_str()
        data = json.loads(s)
        self.assertIn("callable", data)
        self.assertIsInstance(data["callable"], str)

    def test_round_trip_serialization(self):
        """Test serialize then deserialize produces a usable processor."""
        s = DisallowedTokensLogitsProcessor.to_str()
        processor = CustomLogitProcessor.from_str(s)
        self.assertIsInstance(processor, DisallowedTokensLogitsProcessor)

    def test_from_str_is_cached(self):
        """Test that from_str uses LRU cache for repeated calls."""
        _cache_from_str.cache_clear()
        s = DisallowedTokensLogitsProcessor.to_str()
        cls1 = _cache_from_str(s)
        cls2 = _cache_from_str(s)
        self.assertIs(cls1, cls2)


# DisallowedTokensLogitsProcessor
class TestDisallowedTokensLogitsProcessor(CustomTestCase):
    def setUp(self):
        self.processor = DisallowedTokensLogitsProcessor()

    def test_disallowed_tokens_set_to_neg_inf(self):
        """Test that disallowed token positions are set to -inf for all batch items."""
        logits = torch.zeros(2, 10)
        params = [{"token_ids": [2, 5]}, {"token_ids": [2, 5]}]
        result = self.processor(logits, params)
        self.assertTrue(torch.isinf(result[0, 2]) and result[0, 2] < 0)
        self.assertTrue(torch.isinf(result[0, 5]) and result[0, 5] < 0)
        self.assertTrue(torch.isinf(result[1, 2]) and result[1, 2] < 0)

    def test_allowed_tokens_unchanged(self):
        """Test that non-disallowed tokens keep their original logit values."""
        logits = torch.ones(1, 10)
        params = [{"token_ids": [3]}]
        result = self.processor(logits, params)
        self.assertEqual(result[0, 0].item(), 1.0)
        self.assertEqual(result[0, 4].item(), 1.0)
        self.assertTrue(torch.isinf(result[0, 3]) and result[0, 3] < 0)

    def test_mismatched_params_raises(self):
        """Test that mismatched token_ids across batch items raises AssertionError."""
        logits = torch.zeros(2, 10)
        params = [{"token_ids": [1, 2]}, {"token_ids": [3, 4]}]
        with self.assertRaises(AssertionError):
            self.processor(logits, params)


# ThinkingBudgetLogitProcessor (using Qwen3 variant)
class TestThinkingBudgetLogitProcessor(CustomTestCase):
    """Test thinking budget enforcement using Qwen3 token IDs.

    Qwen3 tokens:
        THINKING_START = 151667
        THINKING_END   = 151668
        NEW_LINE       = 198
    """

    START = Qwen3ThinkingBudgetLogitProcessor.THINKING_START_TOKEN_ID
    END = Qwen3ThinkingBudgetLogitProcessor.THINKING_END_TOKEN_ID
    NL = Qwen3ThinkingBudgetLogitProcessor.NEW_LINE_TOKEN_ID
    VOCAB = 200000

    def setUp(self):
        self.processor = Qwen3ThinkingBudgetLogitProcessor()

    def _logits(self, batch_size=1):
        return torch.zeros(batch_size, self.VOCAB)

    def test_budget_not_exceeded_no_change(self):
        """Test no modification when thinking tokens are within budget."""
        req = _make_req(
            origin_input_ids=[self.START],
            output_ids=[100, 101],  # 2 tokens after start
        )
        params = [{"thinking_budget": 10, "__req__": req}]
        logits = self._logits()
        result = self.processor(logits, params)
        self.assertEqual(result[0, 0].item(), 0.0)  # unchanged

    def test_budget_exceeded_forces_newline_first(self):
        """Test forcing newline when budget exceeded and last token is not newline."""
        req = _make_req(
            origin_input_ids=[self.START],
            output_ids=[100] * 5,  # 5 tokens, budget=5 → exceeded
        )
        params = [{"thinking_budget": 5, "__req__": req}]
        logits = self._logits()
        result = self.processor(logits, params)
        # newline should be the only non-neg-inf token
        self.assertEqual(result[0, self.NL].item(), 0.0)
        self.assertTrue(torch.isinf(result[0, 0]) and result[0, 0] < 0)

    def test_budget_exceeded_with_newline_forces_end_token(self):
        """Test forcing end token when budget exceeded and last token is newline."""
        req = _make_req(
            origin_input_ids=[self.START],
            output_ids=[100] * 5 + [self.NL],  # 6 tokens, last is newline
        )
        params = [{"thinking_budget": 5, "__req__": req}]
        logits = self._logits()
        result = self.processor(logits, params)
        self.assertEqual(result[0, self.END].item(), 0.0)
        self.assertTrue(torch.isinf(result[0, 0]) and result[0, 0] < 0)

    def test_skips_when_not_in_thinking(self):
        """Test skip when THINKING_START is absent (no thinking phase)."""
        req = _make_req(origin_input_ids=[100, 101], output_ids=[102])
        params = [{"thinking_budget": 0, "__req__": req}]
        logits = self._logits()
        original = logits.clone()
        result = self.processor(logits, params)
        self.assertTrue(torch.equal(result, original))

    def test_skips_when_thinking_already_ended(self):
        """Test skip when THINKING_END already appeared."""
        req = _make_req(
            origin_input_ids=[self.START],
            output_ids=[100, self.END, 200],
        )
        params = [{"thinking_budget": 0, "__req__": req}]
        logits = self._logits()
        original = logits.clone()
        result = self.processor(logits, params)
        self.assertTrue(torch.equal(result, original))

    def test_skips_when_budget_is_none(self):
        """Test that thinking_budget=None is ignored even during thinking phase."""
        req = _make_req(origin_input_ids=[self.START], output_ids=[100] * 10)
        params = [{"thinking_budget": None, "__req__": req}]
        logits = self._logits()
        original = logits.clone()
        result = self.processor(logits, params)
        self.assertTrue(torch.equal(result, original))

    def test_skips_when_budget_is_negative(self):
        """Test that negative thinking_budget is treated as disabled (no enforcement)."""
        req = _make_req(origin_input_ids=[self.START], output_ids=[100] * 10)
        params = [{"thinking_budget": -1, "__req__": req}]
        logits = self._logits()
        original = logits.clone()
        result = self.processor(logits, params)
        self.assertTrue(torch.equal(result, original))

    def test_none_params_returns_unchanged(self):
        """Test that passing None as param list returns logits unchanged."""
        logits = self._logits()
        original = logits.clone()
        result = self.processor(logits, None)
        self.assertTrue(torch.equal(result, original))

    def test_empty_params_returns_unchanged(self):
        """Test that passing empty param list returns logits unchanged."""
        logits = self._logits()
        original = logits.clone()
        result = self.processor(logits, [])
        self.assertTrue(torch.equal(result, original))

    def test_budget_zero_forces_immediate_end(self):
        """Test that budget=0 forces thinking to end immediately."""
        req = _make_req(
            origin_input_ids=[self.START],
            output_ids=[100],  # 1 token after start > budget=0
        )
        params = [{"thinking_budget": 0, "__req__": req}]
        logits = self._logits()
        result = self.processor(logits, params)
        # Should force newline since last token (100) is not newline
        self.assertEqual(result[0, self.NL].item(), 0.0)

    def test_none_param_dict_in_list_skipped(self):
        """Test that None entry in param list is skipped gracefully."""
        req = _make_req(
            origin_input_ids=[self.START],
            output_ids=[100] * 10,
        )
        params = [None, {"thinking_budget": 0, "__req__": req}]
        logits = self._logits(batch_size=2)
        result = self.processor(logits, params)
        # Batch 0 (None param) should be unchanged
        self.assertEqual(result[0, 0].item(), 0.0)
        # Batch 1 should have been modified (budget exceeded)
        self.assertEqual(result[1, self.NL].item(), 0.0)
        self.assertTrue(torch.isinf(result[1, 0]) and result[1, 0] < 0)

    def test_multiple_thinking_start_counts_from_first(self):
        """Test that budget counts from the first THINKING_START occurrence."""
        req = _make_req(
            origin_input_ids=[self.START, 100, 101],
            output_ids=[self.START, 200, 201],  # second START in output
        )
        # cur_ids = [START, 100, 101, START, 200, 201]
        # First START at index 0, tokens_after_start = 5
        # Budget=10 → 5 < 10 → no modification
        params = [{"thinking_budget": 10, "__req__": req}]
        logits = self._logits()
        original = logits.clone()
        result = self.processor(logits, params)
        self.assertTrue(torch.equal(result, original))

    def test_deepseek_r1_variant_forces_end(self):
        """Test DeepSeekR1 variant with its own token IDs."""
        proc = DeepSeekR1ThinkingBudgetLogitProcessor()
        START = proc.THINKING_START_TOKEN_ID  # 128798
        NL = proc.NEW_LINE_TOKEN_ID  # 201
        VOCAB = 200000

        req = _make_req(origin_input_ids=[START], output_ids=[100] * 5)
        params = [{"thinking_budget": 5, "__req__": req}]
        logits = torch.zeros(1, VOCAB)
        result = proc(logits, params)
        # Budget exceeded, last token (100) is not newline → force newline
        self.assertEqual(result[0, NL].item(), 0.0)
        self.assertTrue(torch.isinf(result[0, 0]) and result[0, 0] < 0)


# DeepseekOCRNoRepeatNGramLogitProcessor
class TestDeepseekOCRNoRepeatNGramLogitProcessor(CustomTestCase):
    VOCAB = 100

    def setUp(self):
        self.processor = DeepseekOCRNoRepeatNGramLogitProcessor()

    def _logits(self, batch_size=1):
        return torch.zeros(batch_size, self.VOCAB)

    def test_bans_repeated_bigrams(self):
        """Test banning token that completes a repeated bigram."""
        req = _make_req(origin_input_ids=[1, 2, 3, 1, 2])
        params = [
            {
                "__req__": req,
                "ngram_size": 2,
                "window_size": 100,
            }
        ]
        logits = self._logits()
        result = self.processor(logits, params)
        self.assertTrue(torch.isinf(result[0, 3]) and result[0, 3] < 0)

    def test_non_repeated_tokens_unchanged(self):
        """Test that tokens not completing a repeated ngram are unchanged."""
        req = _make_req(origin_input_ids=[1, 2, 3, 1, 2])
        params = [{"__req__": req, "ngram_size": 2, "window_size": 100}]
        logits = self._logits()
        result = self.processor(logits, params)
        # Token 1 is not banned (prefix (2) was followed by 3, not 1)
        self.assertEqual(result[0, 1].item(), 0.0)

    def test_window_size_limits_search(self):
        """Test that ngrams outside the window are not considered."""
        # Sequence: [1,2,3,...,1,2] but window only covers the last 3 tokens
        req = _make_req(origin_input_ids=[1, 2, 3, 4, 5, 1, 2])
        params = [{"__req__": req, "ngram_size": 2, "window_size": 3}]
        logits = self._logits()
        result = self.processor(logits, params)
        # Window covers [5, 1, 2]. The bigram (1,2) from index 0-1 is outside.
        # Within window: bigrams are (5,1), (1,2). Current prefix is (2).
        # No bigram starting with prefix (2) in window → nothing banned.
        self.assertEqual(result[0, 3].item(), 0.0)

    def test_whitelist_protects_tokens(self):
        """Test that whitelisted tokens are not banned despite repeated ngrams."""
        req = _make_req(origin_input_ids=[1, 2, 3, 1, 2])
        params = [
            {
                "__req__": req,
                "ngram_size": 2,
                "window_size": 100,
                "whitelist_token_ids": [3],
            }
        ]
        logits = self._logits()
        result = self.processor(logits, params)
        # Token 3 would be banned but is whitelisted
        self.assertEqual(result[0, 3].item(), 0.0)

    def test_ngram_size_zero_skips(self):
        """ngram_size=0 is invalid and should be skipped (no modification)."""
        req = _make_req(origin_input_ids=[1, 2, 1, 2])
        params = [{"__req__": req, "ngram_size": 0, "window_size": 100}]
        logits = self._logits()
        original = logits.clone()
        result = self.processor(logits, params)
        self.assertTrue(torch.equal(result, original))

    def test_window_size_zero_skips(self):
        """Test that window_size=0 disables ngram checking (no modification)."""
        req = _make_req(origin_input_ids=[1, 2, 1, 2])
        params = [{"__req__": req, "ngram_size": 2, "window_size": 0}]
        logits = self._logits()
        original = logits.clone()
        result = self.processor(logits, params)
        self.assertTrue(torch.equal(result, original))

    def test_empty_params_returns_unchanged(self):
        """Test that None param list returns logits unchanged (early return)."""
        logits = self._logits()
        original = logits.clone()
        result = self.processor(logits, None)
        self.assertTrue(torch.equal(result, original))

    def test_short_sequence_skips(self):
        """Sequence shorter than ngram_size should be skipped."""
        req = _make_req(origin_input_ids=[1])
        params = [{"__req__": req, "ngram_size": 3, "window_size": 100}]
        logits = self._logits()
        original = logits.clone()
        result = self.processor(logits, params)
        self.assertTrue(torch.equal(result, original))

    def test_unigram_mode(self):
        """ngram_size=1 bans any token already seen in the window."""
        req = _make_req(origin_input_ids=[5, 10, 15])
        params = [{"__req__": req, "ngram_size": 1, "window_size": 100}]
        logits = self._logits()
        result = self.processor(logits, params)
        # All tokens in [5, 10, 15] should be banned
        self.assertTrue(torch.isinf(result[0, 5]) and result[0, 5] < 0)
        self.assertTrue(torch.isinf(result[0, 10]) and result[0, 10] < 0)
        self.assertTrue(torch.isinf(result[0, 15]) and result[0, 15] < 0)
        # Other tokens should be fine
        self.assertEqual(result[0, 0].item(), 0.0)

    def test_none_req_skips(self):
        """If __req__ is missing, the batch item should be skipped."""
        params = [{"ngram_size": 2, "window_size": 100}]
        logits = self._logits()
        original = logits.clone()
        result = self.processor(logits, params)
        self.assertTrue(torch.equal(result, original))

    def test_invalid_ngram_size_type_skips(self):
        """Non-numeric ngram_size should be handled gracefully."""
        req = _make_req(origin_input_ids=[1, 2, 1, 2])
        params = [{"__req__": req, "ngram_size": "invalid", "window_size": 100}]
        logits = self._logits()
        original = logits.clone()
        result = self.processor(logits, params)
        self.assertTrue(torch.equal(result, original))

    def test_falsy_params_in_list_skipped(self):
        """A falsy entry (None, {}, 0) in param list should be skipped."""
        req = _make_req(origin_input_ids=[1, 2, 1, 2])
        params = [None, {"__req__": req, "ngram_size": 2, "window_size": 100}]
        logits = self._logits(batch_size=2)
        result = self.processor(logits, params)
        # Batch 0 (None) unchanged
        self.assertEqual(result[0, 0].item(), 0.0)
        # Batch 1 has ban applied
        self.assertTrue(torch.isinf(result[1, 1]) and result[1, 1] < 0)

    def test_search_end_leq_search_start_skips(self):
        """Test skip when window is too small for the ngram_size."""
        # sequence length=4, ngram_size=3, window_size=2
        # search_start = max(0, 4-2) = 2
        # search_end = 4 - 3 + 1 = 2
        # search_end (2) <= search_start (2) → skip
        req = _make_req(origin_input_ids=[1, 2, 3, 4])
        params = [{"__req__": req, "ngram_size": 3, "window_size": 2}]
        logits = self._logits()
        original = logits.clone()
        result = self.processor(logits, params)
        self.assertTrue(torch.equal(result, original))

    def test_invalid_whitelist_type_handled(self):
        """Test graceful handling of non-iterable whitelist_token_ids."""
        req = _make_req(origin_input_ids=[1, 2, 1, 2])
        params = [
            {
                "__req__": req,
                "ngram_size": 2,
                "window_size": 100,
                "whitelist_token_ids": 999,  # int, not iterable
            }
        ]
        logits = self._logits()
        result = self.processor(logits, params)
        # Should still ban token 1 (whitelist parse fails, falls back to empty set)
        self.assertTrue(torch.isinf(result[0, 1]) and result[0, 1] < 0)

    def test_batch_processing(self):
        """Test that multiple batch items are processed independently."""
        req1 = _make_req(
            origin_input_ids=[1, 2, 1, 2]
        )  # will ban token 2 (bigram repeat)
        req2 = _make_req(origin_input_ids=[3, 4, 5])  # no repeat
        params = [
            {"__req__": req1, "ngram_size": 2, "window_size": 100},
            {"__req__": req2, "ngram_size": 2, "window_size": 100},
        ]
        logits = self._logits(batch_size=2)
        result = self.processor(logits, params)
        # Batch 0: bigram (1,2) appeared, prefix is (2) → ban token that followed (2) = 1
        # Also (2,1) appeared, prefix is (2) → already covered
        # Actually: sequence is [1,2,1,2], prefix is last (ngram_size-1)=1 token = (2)
        # Scanning: index 0: (1,2) prefix=(1); index 1: (2,1) prefix=(2)→bans 1; index 2: (1,2) prefix=(1)
        # So prefix (2) appeared at index 1, followed by token 1. Ban token 1.
        self.assertTrue(torch.isinf(result[0, 1]) and result[0, 1] < 0)
        # Batch 1: prefix is (5), no matching prefix in window → no bans
        self.assertEqual(result[1, 3].item(), 0.0)
        self.assertEqual(result[1, 4].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
