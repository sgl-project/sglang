"""Unit tests for srt/sampling/custom_logit_processor.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=8, suite="stage-b-test-cpu-intel")

import unittest
from array import array
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.parser.inkling_tokenizer import (
    CONTENT_THINKING,
    END_MESSAGE,
    INKLING_SPECIAL_TOKEN_IDS,
)
from sglang.srt.sampling.custom_logit_processor import (
    NGRAM_OUTPUT_ONLY_KEY,
    CustomLogitProcessor,
    DeepseekOCRNoRepeatNGramLogitProcessor,
    DeepSeekR1ThinkingBudgetLogitProcessor,
    DisallowedTokensLogitsProcessor,
    InklingThinkingBudgetLogitProcessor,
    Qwen3ThinkingBudgetLogitProcessor,
)
from sglang.srt.sampling.sampling_batch_info import (
    ProcessorEntry,
    SamplingBatchInfo,
)
from sglang.srt.session.uocr_doc_reuse import UocrReqState
from sglang.test.test_utils import CustomTestCase


# Helper: mock a Req object (used by ThinkingBudget and NGram processors)
def _make_req(origin_input_ids=None, output_ids=None):
    req = MagicMock()
    req.origin_input_ids = array("q", origin_input_ids or [])
    req.output_ids = array("q", output_ids or [])
    # A real Req declares these as None; a MagicMock would hand out a Mock.
    req.page_guard_state = None
    req.uocr_doc = None
    return req


class TestApplyCustomLogitProcessor(CustomTestCase):
    def test_repeats_request_params_for_each_token(self):
        batch_size = 3
        num_tokens = 2
        params = [{"value": 1.0}, {"value": 2.0}, {"value": 3.0}]

        def processor(logits, custom_params):
            self.assertEqual(
                custom_params, [params[0], params[0], params[2], params[2]]
            )
            for row, param in zip(logits, custom_params, strict=True):
                row.fill_(param["value"])
            return logits

        sampling_info = SamplingBatchInfo(
            temperatures=torch.ones(batch_size, 1),
            top_ps=torch.ones(batch_size),
            top_ks=torch.zeros(batch_size, dtype=torch.int32),
            min_ps=torch.zeros(batch_size),
            is_all_greedy=False,
            is_any_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=4,
            has_custom_logit_processor=True,
            custom_params=params,
            custom_logit_processor={
                0: ProcessorEntry(
                    processor=processor, rows=[0, 2], indices=torch.tensor([0, 2])
                )
            },
            device="cpu",
        )
        logits = torch.zeros(batch_size * num_tokens, 4)

        apply_custom_logit_processor(
            logits, sampling_info, num_tokens_in_batch=num_tokens
        )

        expected = torch.tensor(
            [[1.0] * 4, [1.0] * 4, [0.0] * 4, [0.0] * 4, [3.0] * 4, [3.0] * 4]
        )
        self.assertTrue(torch.equal(logits, expected))


class TestCustomLogitProcessorRows(CustomTestCase):
    """The recorded row list must select the same rows as the device indices.

    Rows and indices are two representations of one selection, rebuilt at three
    separate sites. Every other case here constructs the entry by hand, so only
    a filtered or merged batch exercises both representations at once -- drift
    between them would process the wrong rows with nothing turning red.
    """

    KEY = 0

    @staticmethod
    def _mark_row(logits, custom_params):
        logits[:, 1] = 1.0
        return logits

    def _info(self, batch_size, rows):
        return SamplingBatchInfo(
            temperatures=torch.ones(batch_size, 1),
            top_ps=torch.ones(batch_size),
            top_ks=torch.zeros(batch_size, dtype=torch.int32),
            min_ps=torch.zeros(batch_size),
            is_all_greedy=False,
            is_any_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=4,
            penalizer_orchestrator=MagicMock(is_required=False),
            has_custom_logit_processor=True,
            custom_params=[{"row": i} for i in range(batch_size)],
            custom_logit_processor={
                self.KEY: ProcessorEntry(
                    processor=self._mark_row,
                    rows=list(rows),
                    indices=torch.tensor(rows, dtype=torch.long),
                )
            },
            device="cpu",
        )

    def test_rows_track_the_mask_and_both_paths_agree(self):
        # Rows [0, 2] of 3; keep [0, 2] remaps them to [0, 1] of 2.
        info = self._info(3, [0, 2])
        info.filter_batch([0, 2], torch.tensor([0, 2]))
        # Merge a 2-row batch whose only opted-in row is old index 1.
        info.merge_batch(self._info(2, [1]))

        entry = info.custom_logit_processor[self.KEY]
        self.assertEqual(entry.rows, entry.indices.tolist())

        expected = torch.zeros(4, 4)
        expected[[0, 1, 3], 1] = 1.0
        logits = torch.zeros(4, 4)
        apply_custom_logit_processor(logits, info)
        self.assertTrue(torch.equal(logits, expected))

    def test_a_reordering_filter_keeps_each_row_on_its_own_params(self):
        """Retraction filters the running batch in keep-preference order, not
        ascending. The rows path then fed every row another request's params,
        so the page guard forced or allowed EOS on the wrong document."""

        def write_row_id(logits, custom_params):
            for i, params in enumerate(custom_params):
                logits[i, 0] = params["row"]
            return logits

        for opted_in in ([0, 1, 2], [0, 2]):
            with self.subTest(opted_in=opted_in):
                info = self._info(3, opted_in)
                entry = info.custom_logit_processor[self.KEY]
                info.custom_logit_processor = {
                    self.KEY: ProcessorEntry(
                        processor=write_row_id, rows=entry.rows, indices=entry.indices
                    )
                }
                # New row j holds old request keep[j].
                keep = [2, 0, 1]
                info.filter_batch(keep, torch.tensor(keep))
                logits = torch.full((3, 4), -1.0)
                apply_custom_logit_processor(logits, info)
                want = [float(old) if old in opted_in else -1.0 for old in keep]
                self.assertEqual(logits[:, 0].tolist(), want)


# Serialization round-trip
class TestCustomLogitProcessorSerialization(CustomTestCase):
    def test_round_trip_serialization(self):
        """Test serialize then deserialize produces a usable processor."""
        s = DisallowedTokensLogitsProcessor.to_str()
        processor = CustomLogitProcessor.from_str(s)
        self.assertIsInstance(processor, DisallowedTokensLogitsProcessor)


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

    def test_multiple_thinking_start_counts_from_most_recent(self):
        """Test that budget counts from the most recent THINKING_START occurrence."""
        req = _make_req(
            origin_input_ids=[self.START, 100, 101],
            output_ids=[self.START, 200, 201],  # second START in output
        )
        # cur_ids = [START, 100, 101, START, 200, 201]
        # Most recent START at index 3, tokens_after_start = 2
        # Budget=4 → 2 < 4 → no modification (index 0 would have given 5)
        params = [{"thinking_budget": 4, "__req__": req}]
        logits = self._logits()
        original = logits.clone()
        result = self.processor(logits, params)
        self.assertTrue(torch.equal(result, original))

    def test_inkling_end_token_in_prompt_does_not_disable_budget(self):
        proc = InklingThinkingBudgetLogitProcessor()
        start, end = proc.THINKING_START_TOKEN_ID, proc.THINKING_END_TOKEN_ID
        self.assertEqual(start, INKLING_SPECIAL_TOKEN_IDS[CONTENT_THINKING])
        self.assertEqual(end, INKLING_SPECIAL_TOKEN_IDS[END_MESSAGE])

        req = _make_req(
            origin_input_ids=[100, end, 101, end],
            output_ids=[start] + [100] * 5,
        )
        params = [{"thinking_budget": 5, "__req__": req}]
        for forced_token in (proc.NEW_LINE_TOKEN_ID, end):
            result = proc(torch.zeros(1, end + 1), params)
            self.assertEqual(
                torch.isfinite(result[0]).nonzero().flatten().tolist(), [forced_token]
            )
            req.output_ids.append(forced_token)

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
        req = _make_req(output_ids=[1, 2, 3, 1, 2])
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
        req = _make_req(output_ids=[1, 2, 3, 1, 2])
        params = [{"__req__": req, "ngram_size": 2, "window_size": 100}]
        logits = self._logits()
        result = self.processor(logits, params)
        # Token 1 is not banned (prefix (2) was followed by 3, not 1)
        self.assertEqual(result[0, 1].item(), 0.0)

    def test_window_size_limits_search(self):
        """Test that ngrams outside the window are not considered."""
        # Sequence: [1,2,3,...,1,2] but window only covers the last 3 tokens
        req = _make_req(output_ids=[1, 2, 3, 4, 5, 1, 2])
        params = [{"__req__": req, "ngram_size": 2, "window_size": 3}]
        logits = self._logits()
        result = self.processor(logits, params)
        # Window covers [5, 1, 2]. The bigram (1,2) from index 0-1 is outside.
        # Within window: bigrams are (5,1), (1,2). Current prefix is (2).
        # No bigram starting with prefix (2) in window → nothing banned.
        self.assertEqual(result[0, 3].item(), 0.0)

    def test_whitelist_protects_tokens(self):
        """Test that whitelisted tokens are not banned despite repeated ngrams."""
        req = _make_req(output_ids=[1, 2, 3, 1, 2])
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
        req = _make_req(output_ids=[1, 2, 1, 2])
        params = [{"__req__": req, "ngram_size": 0, "window_size": 100}]
        logits = self._logits()
        original = logits.clone()
        result = self.processor(logits, params)
        self.assertTrue(torch.equal(result, original))

    def test_window_size_zero_skips(self):
        """Test that window_size=0 disables ngram checking (no modification)."""
        req = _make_req(output_ids=[1, 2, 1, 2])
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
        req = _make_req(output_ids=[1])
        params = [{"__req__": req, "ngram_size": 3, "window_size": 100}]
        logits = self._logits()
        original = logits.clone()
        result = self.processor(logits, params)
        self.assertTrue(torch.equal(result, original))

    def test_unigram_mode(self):
        """ngram_size=1 bans any token already seen in the window."""
        req = _make_req(output_ids=[5, 10, 15])
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
        req = _make_req(output_ids=[1, 2, 1, 2])
        params = [{"__req__": req, "ngram_size": "invalid", "window_size": 100}]
        logits = self._logits()
        original = logits.clone()
        result = self.processor(logits, params)
        self.assertTrue(torch.equal(result, original))

    def test_falsy_params_in_list_skipped(self):
        """A falsy entry (None, {}, 0) in param list should be skipped."""
        req = _make_req(output_ids=[1, 2, 1, 2])
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
        req = _make_req(output_ids=[1, 2, 3, 4])
        params = [{"__req__": req, "ngram_size": 3, "window_size": 2}]
        logits = self._logits()
        original = logits.clone()
        result = self.processor(logits, params)
        self.assertTrue(torch.equal(result, original))

    def test_invalid_whitelist_type_handled(self):
        """Test graceful handling of non-iterable whitelist_token_ids."""
        req = _make_req(output_ids=[1, 2, 1, 2])
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
        req1 = _make_req(output_ids=[1, 2, 1, 2])  # will ban token 2 (bigram repeat)
        req2 = _make_req(output_ids=[3, 4, 5])  # no repeat
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

    def test_output_only_domain_is_opt_in(self):
        """The n-gram search domain is the generated tokens only on request.

        vLLM passes the processor generated tokens only (both the v0
        `logits_processors` API and the v1 adapter), which Unlimited-OCR v2
        serving asks for; DeepSeek-OCR and Unlimited-OCR v1 callers keep the
        prompt+output domain this processor always searched.
        """
        # Prompt ends (2, 9); the output's last token is (2), so the prompt's
        # bigram (2, 9) bans 9 under the prompt+output domain.
        req = _make_req(origin_input_ids=[1, 2, 9], output_ids=[2])
        params = [{"__req__": req, "ngram_size": 2, "window_size": 100}]

        result = self.processor(self._logits(), params)
        self.assertTrue(torch.isinf(result[0, 9]) and result[0, 9] < 0)

        params[0][NGRAM_OUTPUT_ONLY_KEY] = True
        result = self.processor(self._logits(), params)
        self.assertEqual(result[0, 9].item(), 0.0)


class TestPageLengthCap(CustomTestCase):
    """The page guard holds a window to one output page per prompt page and
    cuts a diverging page instead of spending every later page's budget."""

    VOCAB, PAGE, EOS, IMG = 100, 7, 1, 50

    def _row(
        self,
        prompt_pages,
        output_ids,
        cap=4,
        context_pages=0,
        req=None,
        stop_only=False,
    ):
        if req is None:
            prompt = []
            for _ in range(prompt_pages):
                prompt += [self.PAGE, self.IMG, self.IMG]
            req = _make_req(origin_input_ids=prompt, output_ids=[])
            req.uocr_doc = UocrReqState(context_pages=context_pages)
        req.output_ids = output_ids
        params = [
            {
                "__req__": req,
                "max_page_tokens": cap,
                "page_token_id": self.PAGE,
                "eos_token_id": self.EOS,
                "page_stop_only": stop_only,
            }
        ]
        row = DeepseekOCRNoRepeatNGramLogitProcessor()(
            torch.zeros(1, self.VOCAB), params
        )[0]
        return row, req

    def _forced(self, prompt_pages, output_ids, cap=4, context_pages=0):
        row, _ = self._row(prompt_pages, output_ids, cap, context_pages)
        finite = torch.isfinite(row).nonzero().flatten().tolist()
        return finite[0] if len(finite) == 1 else None

    def _banned(self, row):
        return {i for i in (self.PAGE, self.EOS) if row[i] == -float("inf")}

    def test_long_page_turns_to_next_page_then_stops_on_last(self):
        P = self.PAGE
        self.assertEqual(self._forced(3, [P, 10, 11, 12, 13]), P)
        self.assertEqual(self._forced(3, [P, 10, P, 11, P, 12, 13, 14, 15]), self.EOS)

    def test_page_under_the_cap_is_untouched(self):
        P = self.PAGE
        self.assertIsNone(self._forced(3, [P, 10, 11, 12, 13, P, 20, 21, 22]))
        self.assertIsNone(self._forced(3, [P, 10, 11, 12, 13], cap=0))

    def test_stopping_a_page_short_is_not_allowed(self):
        """A window that merges two pages into one body emits one marker too
        few and then stops, which drops a page from the document."""
        P = self.PAGE
        row, _ = self._row(3, [P, 10, P, 11])
        self.assertEqual(self._banned(row), {self.EOS})
        # The cap is off: the gate still holds, it just cannot turn a long page.
        row, _ = self._row(3, [P, 10, P, 11], cap=0)
        self.assertEqual(self._banned(row), {self.EOS})

    def test_a_prompt_without_pages_is_left_alone(self):
        """Plain text or an untagged image has no page to count; banning EOS
        there would never let the request stop."""
        req = _make_req(origin_input_ids=[10, 11], output_ids=[])
        row, _ = self._row(0, [10, 11, 12, 13, 14], req=req)
        self.assertEqual(self._banned(row), set())

    def test_a_window_cannot_open_more_pages_than_the_prompt_asks_for(self):
        """Divergence also comes out as a run of tiny pages, which the length cap
        never sees: a 10-page window wrote 87 pages of a few tokens each."""
        P = self.PAGE
        row, req = self._row(2, [P, 10, P, 11])
        self.assertEqual(self._banned(row), {P})
        # Everything else is still on the table, including EOS.
        self.assertTrue(torch.isfinite(row[self.EOS]) and torch.isfinite(row[10]))
        # One page still to open: the marker stays available.
        row, _ = self._row(3, [P, 10, P, 11])
        self.assertTrue(torch.isfinite(row[P]))

    def test_a_marker_past_the_last_page_stops_the_window(self):
        """The mask cannot catch a marker that is already in output_ids: a
        reuse window is pre-filled with the previous window's text."""
        P = self.PAGE
        self.assertEqual(self._forced(2, [P, 10, P, 11, P]), self.EOS)

    def test_the_marker_count_survives_across_steps(self):
        """The count is kept per request, so a 100k-token output is not rescanned
        every step; the tail that arrived since the last call is enough."""
        P = self.PAGE
        out = [P, 10]
        row, req = self._row(2, out)
        self.assertTrue(torch.isfinite(row[P]))
        out = out + [11, P]
        row, _ = self._row(2, out, req=req)
        self.assertEqual(self._banned(row), {P})
        st = req.page_guard_state
        self.assertEqual((st.seen, st.marks, st.last), (4, 2, 3))

    def test_a_reuse_window_counts_only_the_pages_it_owns(self):
        """A reused window's prompt opens with two context pages whose text is
        spliced in rather than decoded (see session/uocr_doc_reuse.py). Counting
        them would turn the page instead of stopping on the window's last."""
        P = self.PAGE
        spliced = [30, 31, P]  # the previous window's text tail and its marker
        # Prompt = block 0 + image tail + one owned page: one page to emit.
        self.assertEqual(
            self._forced(3, spliced + [40, 41, 42, 43], context_pages=2), self.EOS
        )
        # Counting the context pages turns the page and writes one page too many.
        self.assertEqual(self._forced(3, spliced + [40, 41, 42, 43]), P)
        # Two owned pages: the first long page turns, the last one stops.
        self.assertEqual(
            self._forced(4, spliced + [40, 41, 42, 43], context_pages=2), P
        )
        self.assertEqual(
            self._forced(4, spliced + [40, P, 50, 51, 52, 53], context_pages=2),
            self.EOS,
        )

    def test_a_window_without_instruction_ends_on_the_extra_marker(self):
        """Dropping the instruction made a reuse window's last page loop to the
        token cap: <PAGE> was masked there and the model never chose EOS."""
        P = self.PAGE
        spliced = [30, 31, P]  # previous window's text tail + the forced marker
        # Prompt = <PAGE> image tail + one owned page; the forced marker opens it.
        _, req = self._row(2, [], cap=0, context_pages=1)
        req.uocr_doc.end_on_marker = True
        row, _ = self._row(2, spliced + [40, 41], cap=0, req=req)
        # The last owned page may turn: neither <PAGE> nor EOS is masked.
        self.assertEqual(self._banned(row), set())
        # That marker is past the window's pages, so the next step stops.
        row, _ = self._row(2, spliced + [40, 41, P, 42], cap=0, req=req)
        finite = torch.isfinite(row).nonzero().flatten().tolist()
        self.assertEqual(finite, [self.EOS])

    def test_stop_only_guard_ends_a_reuse_window_and_nothing_else(self):
        """--uocr-max-page-tokens 0 turned the guard off, so a reuse
        window without an instruction never stopped."""
        P = self.PAGE
        spliced = [30, 31, P]
        _, req = self._row(2, [], cap=0, context_pages=1, stop_only=True)
        req.uocr_doc.end_on_marker = True
        # Owed pages do not ban EOS and a long page is not cut: no threshold.
        row, _ = self._row(2, spliced + [40] * 50, cap=0, req=req, stop_only=True)
        self.assertEqual(self._banned(row), set())
        row, _ = self._row(2, spliced + [40, P, 41], cap=0, req=req, stop_only=True)
        finite = torch.isfinite(row).nonzero().flatten().tolist()
        self.assertEqual(finite, [self.EOS])
        # A window that fell back to its full prompt is left alone.
        _, plain = self._row(2, [], cap=0, stop_only=True)
        plain.uocr_doc.end_on_marker = False
        row, _ = self._row(2, [P, 40, P, 41, P, 42], cap=0, req=plain, stop_only=True)
        self.assertEqual(self._banned(row), set())

    def test_a_lookahead_page_ends_the_window_on_its_marker(self):
        """Window pages 2 + 1 lookahead: the lookahead page must not be
        transcribed, and the last own page may still turn (it is the boundary
        the cross-page markers live on)."""
        P = self.PAGE
        prompt = [P, self.IMG, self.IMG] * 3
        req = _make_req(origin_input_ids=prompt, output_ids=[])

        def row(out):
            req.output_ids = out
            params = [
                {
                    "__req__": req,
                    "max_page_tokens": 0,
                    "page_stop_only": True,
                    "page_token_id": P,
                    "eos_token_id": self.EOS,
                    "uocr_doc_lookahead": 1,
                }
            ]
            return DeepseekOCRNoRepeatNGramLogitProcessor()(
                torch.zeros(1, self.VOCAB), params
            )[0]

        self.assertEqual(self._banned(row([P, 40, P, 41])), set())
        finite = torch.isfinite(row([P, 40, P, 41, P, 42])).nonzero().flatten().tolist()
        self.assertEqual(finite, [self.EOS])


class TestNGramProcessorBatchEdits(CustomTestCase):
    """A batch mixing forced, page-banned and n-gram-banned rows.

    Regression: the edits were written row by row through Python indices; on
    CUDA every write blocked the host on the forward that produced the logits.
    """

    PAGE, EOS, VOCAB = 7, 1, 64

    def _batch(self):
        """Two forced rows, two rows with a page and an n-gram ban, one skipped."""
        P = self.PAGE

        def req(output_ids):
            r = _make_req(origin_input_ids=[P, 50, P, 50], output_ids=output_ids)
            r.uocr_doc = UocrReqState(context_pages=0)
            return r

        common = {
            "page_token_id": P,
            "eos_token_id": self.EOS,
            "ngram_size": 2,
            "window_size": 100,
            NGRAM_OUTPUT_ONLY_KEY: True,
        }
        return [
            # Page cap reached on the last page: forced to EOS.
            {"__req__": req([P, 3, P, 4, 4, 4]), "max_page_tokens": 3, **common},
            # Repeating tail: forced to EOS.
            {
                "__req__": req([P, 2, 5, 2, 5, 2, 5]),
                "max_page_tokens": 0,
                "rep_max_pattern_size": 2,
                "rep_min_count": 3,
                **common,
            },
            # A page short: EOS banned, plus the n-gram ban of 3 after 2.
            {"__req__": req([P, 2, 3, 2]), "max_page_tokens": 0, **common},
            None,
            # Last page: <PAGE> banned, plus the n-gram ban of 9 after 8.
            {"__req__": req([P, 8, 9, P, 8]), "max_page_tokens": 0, **common},
        ]

    def _expected(self):
        """What the per-row edits leave on all-zero logits."""
        out = torch.zeros(5, self.VOCAB)
        for row in (0, 1):
            out[row] = -float("inf")
            out[row, self.EOS] = 0.0
        out[2, [self.EOS, 3]] = -float("inf")
        out[4, [self.PAGE, 9]] = -float("inf")
        return out

    def test_every_row_gets_its_own_edits(self):
        result = DeepseekOCRNoRepeatNGramLogitProcessor()(
            torch.zeros(5, self.VOCAB), self._batch()
        )
        self.assertTrue(torch.equal(result, self._expected()))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_edits_match_and_do_not_sync(self):
        logits = torch.zeros(5, self.VOCAB, device="cuda")
        torch.cuda.synchronize()
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            result = DeepseekOCRNoRepeatNGramLogitProcessor()(logits, self._batch())
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        self.assertTrue(torch.equal(result.cpu(), self._expected()))


if __name__ == "__main__":
    unittest.main()
