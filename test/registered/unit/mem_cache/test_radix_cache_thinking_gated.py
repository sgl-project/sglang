"""Regression tests for gated thinking-token stripping (issue #22373, T2 fix).

The T2 fix gates the thinking-token strip behind a per-parser flag
``strip_thinking_from_cache``.  For most parsers (e.g. deepseek-r1) the
flag is True: thinking tokens are stripped from the cache (same behaviour
as the original fix).  For parsers like ``minimax-append-think``, the
flag is False: thinking tokens are part of visible output and the full
sequence is cached.

These tests are CPU-only, no model or GPU required.
Run with:
    python -m pytest test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")

import unittest

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

# ---------------------------------------------------------------------------
# Mock infrastructure (mirrors test_radix_cache_thinking.py)
# ---------------------------------------------------------------------------


class _MockReqToTokenPool:
    """Minimal mock matching the req_to_token_pool interface used by
    cache_finished_req: ``pool.req_to_token[idx, :n]`` returns a tensor."""

    def __init__(self, max_reqs: int = 4, max_seq_len: int = 256):
        self.req_to_token = torch.zeros((max_reqs, max_seq_len), dtype=torch.int64)


class _MockAllocator:
    """Mock KV pool allocator that records every free() call."""

    def __init__(self):
        self.freed: list[torch.Tensor] = []
        self.device = torch.device("cpu")

    def alloc(self, n: int) -> torch.Tensor:
        return torch.arange(n, dtype=torch.int64)

    def free(self, indices: torch.Tensor):
        if isinstance(indices, torch.Tensor) and indices.numel() > 0:
            self.freed.append(indices.clone())

    def available_size(self) -> int:
        return 999_999


class _MockReq:
    """Simulated Req with all attributes accessed by cache_finished_req.

    Extends the original mock with ``strip_thinking_from_cache`` — the new
    per-request gating flag that controls whether thinking tokens are
    stripped from the radix cache.
    """

    def __init__(self):
        self._kv_committed_len: int = 0
        self.kv_committed_freed: bool = False
        self.origin_input_ids: list[int] = []
        self.output_ids: list[int] = []
        self.req_pool_idx: int = 0
        self.extra_key = None
        self.last_node = None
        self.cache_protected_len: int = 0
        self.priority: int = 0

        # Reasoning attributes
        self.require_reasoning: bool = False
        self.reasoning_tokens: int = 0
        self._is_reasoning_over: bool = False

        # NEW: Parser-gated stripping flag.
        # True  → strip thinking from cache (deepseek-r1, qwen3, etc.)
        # False → keep thinking in cache   (minimax-append-think)
        # Default is True (safe default: unknown parsers strip).
        self.strip_thinking_from_cache: bool = True

    def pop_committed_kv_cache(self) -> int:
        assert not self.kv_committed_freed
        self.kv_committed_freed = True
        return self._kv_committed_len


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROMPT_TOKENS = [10, 20, 30]
THINKING_TOKENS = [50, 100, 101, 51]  # 4 thinking tokens
ANSWER_TOKENS = [200, 201]
OUTPUT_WITH_THINKING = THINKING_TOKENS + ANSWER_TOKENS  # 6 tokens


def _make_cache() -> tuple[RadixCache, _MockReqToTokenPool, _MockAllocator]:
    """Create a RadixCache wired to mock pool + allocator."""
    allocator = _MockAllocator()
    pool = _MockReqToTokenPool()
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
    )
    cache = RadixCache(params)
    return cache, pool, allocator


def _prepare_req(
    cache: RadixCache,
    pool: _MockReqToTokenPool,
    prompt: list[int],
    output: list[int],
    *,
    reasoning_tokens: int = 0,
    require_reasoning: bool = False,
    strip_thinking_from_cache: bool = True,
    req_pool_idx: int = 0,
    kv_base: int = 100,
) -> _MockReq:
    """Build a _MockReq, write KV indices into the pool, and return it."""
    req = _MockReq()
    req.origin_input_ids = list(prompt)
    req.output_ids = list(output)
    req.require_reasoning = require_reasoning
    req.reasoning_tokens = reasoning_tokens
    req.strip_thinking_from_cache = strip_thinking_from_cache
    req.req_pool_idx = req_pool_idx

    total_len = len(prompt) + len(output)
    req._kv_committed_len = total_len

    kv_indices = torch.arange(kv_base, kv_base + total_len, dtype=torch.int64)
    pool.req_to_token[req_pool_idx, :total_len] = kv_indices

    req.last_node = cache.root_node
    return req


def _all_freed(allocator: _MockAllocator) -> set[int]:
    """Flatten every free() call into a single set of ints."""
    result: set[int] = set()
    for t in allocator.freed:
        result.update(t.tolist())
    return result


# ===================================================================
# Test Suite 1: Gating — strip_thinking_from_cache=True (deepseek-r1)
# ===================================================================


class TestGatedStripTrue(unittest.TestCase):
    """When strip_thinking_from_cache=True AND reasoning_tokens > 0,
    thinking tokens ARE stripped — only prompt prefix is cached.
    This is the deepseek-r1 / qwen3 / default behavior.

    Status: Should PASS both before and after the fix (existing behavior).
    """

    def test_deepseek_style_strips_thinking(self):
        """Parser=deepseek-r1 (strip=True): only prompt cached."""
        cache, pool, alloc = _make_cache()
        req = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=OUTPUT_WITH_THINKING,
            reasoning_tokens=len(THINKING_TOKENS),
            strip_thinking_from_cache=True,  # deepseek-r1 behavior
        )

        cache.cache_finished_req(req, is_insert=True)

        # Tree should contain exactly the prompt prefix
        self.assertEqual(cache.total_size(), len(PROMPT_TOKENS))

        # Thinking tokens not reachable
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(PROMPT_TOKENS + THINKING_TOKENS))
        )
        self.assertEqual(len(match.device_indices), len(PROMPT_TOKENS))

    def test_deepseek_style_frees_output_kv(self):
        """Parser=deepseek-r1 (strip=True): output KV indices are freed."""
        cache, pool, alloc = _make_cache()
        req = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=OUTPUT_WITH_THINKING,
            reasoning_tokens=len(THINKING_TOKENS),
            strip_thinking_from_cache=True,
            kv_base=100,
        )

        cache.cache_finished_req(req, is_insert=True)

        # KV indices for output tokens (103..108) should be freed
        freed = _all_freed(alloc)
        prompt_len = len(PROMPT_TOKENS)
        total_len = prompt_len + len(OUTPUT_WITH_THINKING)
        expected_freed = set(range(100 + prompt_len, 100 + total_len))
        self.assertTrue(
            expected_freed.issubset(freed),
            f"Output KV indices {expected_freed} should be freed, got {freed}",
        )


# ===================================================================
# Test Suite 2: Gating — strip_thinking_from_cache=False (minimax)
# ===================================================================


class TestGatedStripFalse(unittest.TestCase):
    """When strip_thinking_from_cache=False AND reasoning_tokens > 0,
    thinking tokens are NOT stripped — the full sequence is cached.
    This is the minimax-append-think behavior.

    Status: FAIL before fix, PASS after fix.
    Before fix: cache_finished_req ignores the flag and strips anyway.
    """

    def test_minimax_keeps_thinking_in_cache(self):
        """Parser=minimax-append-think (strip=False): full output cached.

        This is the core bug fix test. MiniMax includes thinking tokens
        in its output, so they must remain in the cache for multi-turn
        prefix matching to work.
        """
        cache, pool, alloc = _make_cache()
        all_tokens = PROMPT_TOKENS + OUTPUT_WITH_THINKING  # 9 tokens

        req = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=OUTPUT_WITH_THINKING,
            reasoning_tokens=len(THINKING_TOKENS),
            strip_thinking_from_cache=False,  # minimax-append-think
        )

        cache.cache_finished_req(req, is_insert=True)

        # Full sequence should be cached
        self.assertEqual(
            cache.total_size(),
            len(all_tokens),
            "minimax-append-think: full sequence (prompt+thinking+answer) "
            "must be cached, not stripped",
        )

        # Full match against the complete sequence
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(all_tokens)))
        self.assertEqual(len(match.device_indices), len(all_tokens))

    def test_minimax_kv_indices_are_complete(self):
        """Parser=minimax-append-think: KV indices in tree span full sequence."""
        cache, pool, alloc = _make_cache()
        all_tokens = PROMPT_TOKENS + OUTPUT_WITH_THINKING

        req = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=OUTPUT_WITH_THINKING,
            reasoning_tokens=len(THINKING_TOKENS),
            strip_thinking_from_cache=False,
            kv_base=100,
        )

        cache.cache_finished_req(req, is_insert=True)

        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(all_tokens)))
        expected_kv = torch.arange(100, 100 + len(all_tokens), dtype=torch.int64)
        torch.testing.assert_close(match.device_indices, expected_kv)

    def test_minimax_no_output_kv_freed(self):
        """Parser=minimax-append-think: NO output KV indices should be freed
        (they're all inserted into the tree)."""
        cache, pool, alloc = _make_cache()

        req = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=OUTPUT_WITH_THINKING,
            reasoning_tokens=len(THINKING_TOKENS),
            strip_thinking_from_cache=False,
            kv_base=100,
        )

        cache.cache_finished_req(req, is_insert=True)

        # The freed set should NOT contain any of the output KV indices
        freed = _all_freed(alloc)
        prompt_len = len(PROMPT_TOKENS)
        total_len = prompt_len + len(OUTPUT_WITH_THINKING)
        output_kv = set(range(100 + prompt_len, 100 + total_len))
        wrongly_freed = freed & output_kv
        self.assertEqual(
            len(wrongly_freed),
            0,
            f"minimax-append-think: output KV indices should NOT be freed, "
            f"but these were: {wrongly_freed}",
        )


# ===================================================================
# Test Suite 3: MiniMax multi-turn prefix match
# ===================================================================


class TestMiniMaxMultiTurn(unittest.TestCase):
    """MiniMax M2.5 multi-turn: Turn 2's prompt includes thinking tokens
    from Turn 1 (they're part of the assistant message). The cache must
    find a prefix match including those thinking tokens.

    Status: FAIL before fix, PASS after fix.
    """

    def test_multiturn_thinking_prefix_match(self):
        """Turn 1 caches [prompt + thinking + answer].
        Turn 2 queries [prompt + thinking + answer + new_user_tokens].
        Prefix match should cover the full Turn 1 output.

        This directly tests the scenario described in issue #22373:
        MiniMax appends thinking to the conversation, so Turn 2 sees
        thinking tokens and expects them in the cache.
        """
        cache, pool, alloc = _make_cache()

        # Turn 1: minimax-append-think request
        req1 = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=OUTPUT_WITH_THINKING,
            reasoning_tokens=len(THINKING_TOKENS),
            strip_thinking_from_cache=False,  # minimax-append-think
        )
        cache.cache_finished_req(req1, is_insert=True)

        # Verify full Turn 1 is cached
        turn1_all = PROMPT_TOKENS + OUTPUT_WITH_THINKING
        self.assertEqual(cache.total_size(), len(turn1_all))

        # Turn 2: new request extends the conversation
        new_user_tokens = [300, 301, 302]
        turn2_key = turn1_all + new_user_tokens
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(turn2_key)))

        # Should match the full Turn 1 sequence (9 tokens)
        self.assertEqual(
            len(match.device_indices),
            len(turn1_all),
            "Turn 2 should hit the full Turn 1 prefix including thinking",
        )

    def test_multiturn_thinking_stripped_no_match(self):
        """Contrast: deepseek-r1 strips thinking → Turn 2 with thinking
        tokens in key only matches the prompt prefix.

        This shows the difference between the two parser behaviors.
        """
        cache, pool, alloc = _make_cache()

        # Turn 1: deepseek-r1 request (strips thinking)
        req1 = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=OUTPUT_WITH_THINKING,
            reasoning_tokens=len(THINKING_TOKENS),
            strip_thinking_from_cache=True,  # deepseek-r1
        )
        cache.cache_finished_req(req1, is_insert=True)

        # Only prompt cached
        self.assertEqual(cache.total_size(), len(PROMPT_TOKENS))

        # Turn 2 query with thinking → only prompt matches
        turn2_key = PROMPT_TOKENS + OUTPUT_WITH_THINKING + [300]
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(turn2_key)))
        self.assertEqual(
            len(match.device_indices),
            len(PROMPT_TOKENS),
            "deepseek-r1: Turn 2 sees only prompt prefix (thinking stripped)",
        )


# ===================================================================
# Test Suite 4: Default safety — unknown parsers strip by default
# ===================================================================


class TestDefaultSafety(unittest.TestCase):
    """The strip_thinking_from_cache flag defaults to True, so any
    new/unknown parser gets the safe stripping behavior.

    Status: Should PASS both before and after fix.
    """

    def test_default_flag_is_true(self):
        """_MockReq.strip_thinking_from_cache defaults to True."""
        req = _MockReq()
        self.assertTrue(
            req.strip_thinking_from_cache,
            "Default must be True (strip) — safe for unknown parsers",
        )

    def test_unknown_parser_strips_thinking(self):
        """A request with default strip_thinking_from_cache=True and
        reasoning_tokens > 0 must strip thinking (safe default)."""
        cache, pool, alloc = _make_cache()
        req = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=OUTPUT_WITH_THINKING,
            reasoning_tokens=len(THINKING_TOKENS),
            # strip_thinking_from_cache not set → defaults to True
        )

        cache.cache_finished_req(req, is_insert=True)

        self.assertEqual(
            cache.total_size(),
            len(PROMPT_TOKENS),
            "Unknown/default parser: must strip thinking (safe default)",
        )


# ===================================================================
# Test Suite 5: Edge cases
# ===================================================================


class TestEdgeCases(unittest.TestCase):

    def test_zero_reasoning_tokens_no_strip_regardless_of_flag(self):
        """reasoning_tokens=0 → no stripping, even if strip flag is True.

        Backward compatibility: non-reasoning requests always cache
        the full sequence regardless of the flag setting.

        Status: Should PASS both before and after fix.
        """
        cache, pool, alloc = _make_cache()
        all_tokens = PROMPT_TOKENS + OUTPUT_WITH_THINKING

        req = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=OUTPUT_WITH_THINKING,
            reasoning_tokens=0,  # no reasoning
            strip_thinking_from_cache=True,  # flag is True but irrelevant
        )

        cache.cache_finished_req(req, is_insert=True)

        self.assertEqual(
            cache.total_size(),
            len(all_tokens),
            "reasoning_tokens=0 → all tokens cached, strip flag irrelevant",
        )

    def test_minimax_plain_parser_strips(self):
        """Parser=minimax (NOT minimax-append-think) uses Qwen3Detector
        which separates thinking from content. It SHOULD strip.

        minimax → Qwen3Detector → strip_thinking_from_cache=True
        minimax-append-think → MiniMaxAppendThinkDetector → False

        Status: Should PASS both before and after fix.
        """
        cache, pool, alloc = _make_cache()
        req = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=OUTPUT_WITH_THINKING,
            reasoning_tokens=len(THINKING_TOKENS),
            strip_thinking_from_cache=True,  # minimax (plain) → strip
        )

        cache.cache_finished_req(req, is_insert=True)

        self.assertEqual(
            cache.total_size(),
            len(PROMPT_TOKENS),
            "minimax (plain) parser: must strip thinking like deepseek-r1",
        )

    def test_strip_flag_respected_per_request(self):
        """Two sequential requests with different strip flags.
        Verifies the flag is checked per-request, not cached globally.

        Scenario: mixed deployment where parser might change between
        requests (unusual but possible with dynamic routing).

        Status: FAIL before fix (second req stripped), PASS after fix.
        """
        cache, pool, alloc = _make_cache()

        # Request 1: strip=True (deepseek-r1 style)
        req1 = _prepare_req(
            cache,
            pool,
            prompt=[10, 20],
            output=[50, 51, 200],  # 2 thinking + 1 answer
            reasoning_tokens=2,
            strip_thinking_from_cache=True,
            req_pool_idx=0,
            kv_base=100,
        )
        cache.cache_finished_req(req1, is_insert=True)

        # Only prompt cached for req1
        self.assertEqual(cache.total_size(), 2)

        # Request 2: strip=False (minimax-append-think style), different prompt
        req2 = _prepare_req(
            cache,
            pool,
            prompt=[30, 40],
            output=[60, 61, 201],  # 2 thinking + 1 answer
            reasoning_tokens=2,
            strip_thinking_from_cache=False,
            req_pool_idx=1,
            kv_base=200,
        )
        cache.cache_finished_req(req2, is_insert=True)

        # req2's full sequence should be cached: prompt(2) + output(3) = 5
        # Total: req1 prompt(2) + req2 full(5) = 7
        self.assertEqual(
            cache.total_size(),
            7,
            "Per-request flag: req1 stripped (2 tokens), req2 kept (5 tokens)",
        )

    def test_minimax_all_output_is_thinking(self):
        """minimax-append-think: model outputs only thinking (no answer yet).
        With strip=False, all tokens including thinking should be cached.

        Status: FAIL before fix, PASS after fix.
        """
        cache, pool, alloc = _make_cache()
        thinking_only = [50, 100, 101, 51]
        all_tokens = PROMPT_TOKENS + thinking_only

        req = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=thinking_only,
            reasoning_tokens=len(thinking_only),
            strip_thinking_from_cache=False,  # minimax-append-think
        )

        cache.cache_finished_req(req, is_insert=True)

        self.assertEqual(
            cache.total_size(),
            len(all_tokens),
            "minimax-append-think: thinking-only output must be fully cached",
        )

    def test_zero_reasoning_minimax_caches_all(self):
        """minimax-append-think with reasoning_tokens=0 (non-reasoning request).
        Everything cached — same as any non-reasoning request.

        Status: Should PASS both before and after fix.
        """
        cache, pool, alloc = _make_cache()
        all_tokens = PROMPT_TOKENS + ANSWER_TOKENS

        req = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=ANSWER_TOKENS,
            reasoning_tokens=0,
            strip_thinking_from_cache=False,  # irrelevant when reasoning_tokens=0
        )

        cache.cache_finished_req(req, is_insert=True)

        self.assertEqual(cache.total_size(), len(all_tokens))


# ===================================================================
# Test Suite 6: is_insert=False path with strip=False
# ===================================================================


class TestInsertFalseWithStripFalse(unittest.TestCase):
    """When is_insert=False, nothing is inserted regardless of strip flag."""

    def test_is_insert_false_ignores_strip_flag(self):
        """is_insert=False → empty tree, KV slots freed, strip flag irrelevant.

        Status: Should PASS both before and after fix.
        """
        cache, pool, alloc = _make_cache()
        req = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=OUTPUT_WITH_THINKING,
            reasoning_tokens=len(THINKING_TOKENS),
            strip_thinking_from_cache=False,  # would keep thinking if inserted
        )

        cache.cache_finished_req(req, is_insert=False)

        self.assertEqual(cache.total_size(), 0, "is_insert=False → empty tree")

        freed = _all_freed(alloc)
        self.assertGreater(len(freed), 0, "Some KV slots must be freed")


# ===================================================================
# Test Suite 7: Detector class-level attribute verification
# ===================================================================


class TestDetectorClassAttributes(unittest.TestCase):
    """Verify the strip_thinking_from_cache class attribute on detectors
    and the ReasoningParser → detector → flag propagation chain."""

    def test_base_detector_strip_is_true(self):
        """BaseReasoningFormatDetector.strip_thinking_from_cache is True."""
        from sglang.srt.parser.reasoning_parser import BaseReasoningFormatDetector

        self.assertTrue(BaseReasoningFormatDetector.strip_thinking_from_cache)

    def test_minimax_append_think_detector_strip_is_false(self):
        """MiniMaxAppendThinkDetector.strip_thinking_from_cache is False."""
        from sglang.srt.parser.reasoning_parser import MiniMaxAppendThinkDetector

        self.assertFalse(MiniMaxAppendThinkDetector.strip_thinking_from_cache)

    def test_minimax_parser_detector_strips(self):
        """minimax → Qwen3Detector → inherits strip=True from base."""
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        parser = ReasoningParser(model_type="minimax", stream_reasoning=False)
        self.assertTrue(parser.detector.strip_thinking_from_cache)

    def test_minimax_append_think_parser_detector_retains(self):
        """minimax-append-think → MiniMaxAppendThinkDetector → strip=False."""
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        parser = ReasoningParser(
            model_type="minimax-append-think", stream_reasoning=False
        )
        self.assertFalse(parser.detector.strip_thinking_from_cache)

    def test_deepseek_parser_detector_strips(self):
        """deepseek-r1 → DeepSeekR1Detector → inherits strip=True."""
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        parser = ReasoningParser(model_type="deepseek-r1", stream_reasoning=False)
        self.assertTrue(parser.detector.strip_thinking_from_cache)


if __name__ == "__main__":
    unittest.main()
