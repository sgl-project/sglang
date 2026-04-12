"""Regression tests for SGLang issue #22373.

Problem: cache_finished_req() in radix_cache.py used to insert thinking
tokens (<think>...</think>) AND subsequent answer tokens into the radix
tree.  In multi-turn reasoning, Turn 2's prompt is [prompt + answer]
(no thinking tokens), so the cached [prompt + thinking + answer] sequence
creates unreachable dead entries that waste KV cache memory.  Worse,
answer tokens after thinking have incorrect RoPE positions and cannot be
reused.

Fix: When req.reasoning_tokens > 0, cache_finished_req truncates
token_ids and kv_indices to the prompt prefix (origin_input_ids) only,
preventing BOTH thinking and answer tokens from being cached.

These tests are CPU-only, no model or GPU required.
Run with:
    python -m pytest test/registered/unit/mem_cache/test_radix_cache_thinking.py -v
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
# Mock infrastructure
# ---------------------------------------------------------------------------


class _MockReqToTokenPool:
    """Minimal mock matching the req_to_token_pool interface used by
    cache_finished_req: ``pool.req_to_token[idx, :n]`` returns a tensor."""

    def __init__(self, max_reqs: int = 4, max_seq_len: int = 128):
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
    """Simulated Req with all attributes accessed by cache_finished_req."""

    def __init__(self):
        self._kv_committed_len: int = 0
        self.kv_committed_freed: bool = False
        self.origin_input_ids: list[int] = []
        self.output_ids: list[int] = []
        self.req_pool_idx: int = 0
        self.extra_key = None
        self.last_node = None  # set to cache.root_node before use
        self.cache_protected_len: int = 0
        self.priority: int = 0

        # Reasoning attributes (mirroring Req in schedule_batch.py)
        self.require_reasoning: bool = False
        self.reasoning_tokens: int = 0
        self._is_reasoning_over: bool = False
        self.strip_thinking_from_cache: bool = True

    def pop_committed_kv_cache(self) -> int:
        assert not self.kv_committed_freed
        self.kv_committed_freed = True
        return self._kv_committed_len


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Arbitrary token IDs — not tied to any real tokenizer.
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
    req_pool_idx: int = 0,
    kv_base: int = 100,
) -> _MockReq:
    """Build a _MockReq, write KV indices into the pool, and return it.

    KV indices are ``[kv_base, kv_base+1, ..., kv_base+total_len-1]`` so
    each position has a unique, traceable value.
    """
    req = _MockReq()
    req.origin_input_ids = list(prompt)
    req.output_ids = list(output)
    req.require_reasoning = require_reasoning
    req.reasoning_tokens = reasoning_tokens
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
# Test Suite 1: Thinking tokens stripped — only prompt prefix cached
# ===================================================================


class TestThinkingStrip(unittest.TestCase):
    """When reasoning_tokens > 0, cache_finished_req must cache ONLY the
    prompt prefix (origin_input_ids).  Both thinking and answer tokens
    are excluded because answer tokens have wrong RoPE positions."""

    def test_thinking_tokens_stripped_from_cache(self):
        """With reasoning_tokens > 0, only the prompt prefix is in the tree."""
        cache, pool, alloc = _make_cache()
        req = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=OUTPUT_WITH_THINKING,
            reasoning_tokens=len(THINKING_TOKENS),
        )

        cache.cache_finished_req(req, is_insert=True)

        # Tree should contain exactly the prompt prefix
        self.assertEqual(cache.total_size(), len(PROMPT_TOKENS))

        # Match against prompt → full hit
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(PROMPT_TOKENS)))
        self.assertEqual(len(match.device_indices), len(PROMPT_TOKENS))

        # Match against prompt + thinking → still only prompt hit
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(PROMPT_TOKENS + THINKING_TOKENS))
        )
        self.assertEqual(len(match.device_indices), len(PROMPT_TOKENS))

        # Match against prompt + answer → still only prompt hit
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(PROMPT_TOKENS + ANSWER_TOKENS))
        )
        self.assertEqual(len(match.device_indices), len(PROMPT_TOKENS))

    def test_cached_kv_indices_are_prompt_only(self):
        """KV indices in the tree must correspond to prompt positions only."""
        cache, pool, alloc = _make_cache()
        req = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=OUTPUT_WITH_THINKING,
            reasoning_tokens=len(THINKING_TOKENS),
            kv_base=100,
        )

        cache.cache_finished_req(req, is_insert=True)

        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(PROMPT_TOKENS)))
        # Prompt positions 0,1,2 → KV indices 100,101,102
        expected_kv = torch.tensor([100, 101, 102], dtype=torch.int64)
        torch.testing.assert_close(match.device_indices, expected_kv)


# ===================================================================
# Test Suite 2: Multi-turn prefix matching
# ===================================================================


class TestMultiTurnPrefixMatch(unittest.TestCase):
    """After stripping, Turn 2 should hit the cached prompt prefix."""

    def test_multiturn_prefix_match_after_fix(self):
        """Turn 1 caches only prompt.  Turn 2's match_prefix with
        [prompt + answer + new_user_tokens] hits exactly the prompt."""
        cache, pool, alloc = _make_cache()

        # Turn 1: reasoning request
        req1 = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,  # [10, 20, 30]
            output=OUTPUT_WITH_THINKING,  # [50,100,101,51, 200,201]
            reasoning_tokens=len(THINKING_TOKENS),
        )
        cache.cache_finished_req(req1, is_insert=True)

        # Only prompt prefix should be cached
        self.assertEqual(cache.total_size(), len(PROMPT_TOKENS))

        # Turn 2: new request shares prompt prefix + answer + new user tokens
        turn2_tokens = PROMPT_TOKENS + ANSWER_TOKENS + [40, 50]
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(turn2_tokens)))
        # Should match exactly the 3-token prompt prefix
        self.assertEqual(len(match.device_indices), len(PROMPT_TOKENS))

    def test_thinking_in_query_no_spurious_match(self):
        """Thinking tokens in a lookup key don't match anything beyond prompt."""
        cache, pool, alloc = _make_cache()
        req1 = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=OUTPUT_WITH_THINKING,
            reasoning_tokens=len(THINKING_TOKENS),
        )
        cache.cache_finished_req(req1, is_insert=True)

        # Wrong query that includes thinking tokens
        wrong_query = PROMPT_TOKENS + OUTPUT_WITH_THINKING + [40, 50]
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(wrong_query)))
        self.assertEqual(
            len(match.device_indices),
            len(PROMPT_TOKENS),
            "Thinking tokens in query should not extend match past prompt",
        )


# ===================================================================
# Test Suite 3: Backward compatibility (no reasoning)
# ===================================================================


class TestBackwardCompat(unittest.TestCase):
    """Non-reasoning requests must cache ALL tokens unchanged."""

    def test_no_reasoning_tokens_caches_all(self):
        """reasoning_tokens=0 → every token is cached."""
        cache, pool, alloc = _make_cache()
        all_tokens = PROMPT_TOKENS + OUTPUT_WITH_THINKING  # 9 tokens

        req = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=OUTPUT_WITH_THINKING,
            reasoning_tokens=0,
        )

        cache.cache_finished_req(req, is_insert=True)

        self.assertEqual(cache.total_size(), len(all_tokens))

        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(all_tokens)))
        self.assertEqual(len(match.device_indices), len(all_tokens))

    def test_zero_reasoning_tokens_with_require_reasoning_false(self):
        """Non-reasoning request: reasoning_tokens=0, require_reasoning=False.
        All tokens should be cached."""
        cache, pool, alloc = _make_cache()
        prompt = [10, 20, 30, 40]
        output = [50, 60]
        all_tokens = prompt + output

        req = _prepare_req(
            cache,
            pool,
            prompt=prompt,
            output=output,
            reasoning_tokens=0,
            require_reasoning=False,
        )

        cache.cache_finished_req(req, is_insert=True)

        self.assertEqual(cache.total_size(), len(all_tokens))

        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(all_tokens)))
        self.assertEqual(len(match.device_indices), len(all_tokens))


# ===================================================================
# Test Suite 4: Edge cases
# ===================================================================


class TestEdgeCases(unittest.TestCase):

    def test_all_output_is_thinking(self):
        """Model stops mid-thinking: reasoning_tokens == len(output_ids).
        Only prompt prefix should be cached."""
        cache, pool, alloc = _make_cache()
        thinking = [50, 100, 101, 102]

        req = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=thinking,
            reasoning_tokens=len(thinking),
        )

        cache.cache_finished_req(req, is_insert=True)

        self.assertEqual(cache.total_size(), len(PROMPT_TOKENS))

        # Thinking tokens not reachable
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(PROMPT_TOKENS + thinking))
        )
        self.assertEqual(len(match.device_indices), len(PROMPT_TOKENS))

    def test_empty_output(self):
        """No output tokens, reasoning_tokens=0 — only prompt cached."""
        cache, pool, alloc = _make_cache()

        req = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=[],
            reasoning_tokens=0,
        )

        cache.cache_finished_req(req, is_insert=True)

        self.assertEqual(cache.total_size(), len(PROMPT_TOKENS))

        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(PROMPT_TOKENS)))
        self.assertEqual(len(match.device_indices), len(PROMPT_TOKENS))

    def test_long_thinking_short_answer(self):
        """Very long thinking block, single answer token.
        Only prompt prefix cached."""
        cache, pool, alloc = _make_cache()
        long_think = [50] + list(range(1000, 1050)) + [51]
        short_answer = [200]

        req = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=long_think + short_answer,
            reasoning_tokens=len(long_think),
        )

        cache.cache_finished_req(req, is_insert=True)

        # Only prompt prefix cached (answer also excluded due to RoPE mismatch)
        self.assertEqual(cache.total_size(), len(PROMPT_TOKENS))

        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(PROMPT_TOKENS)))
        self.assertEqual(len(match.device_indices), len(PROMPT_TOKENS))


# ===================================================================
# Test Suite 5: Cache size metrics
# ===================================================================


class TestCacheSize(unittest.TestCase):
    """total_size() must reflect only cached tokens."""

    def test_cache_size_prompt_only_with_thinking(self):
        """With reasoning, cache size = prompt length only."""
        cache, pool, alloc = _make_cache()
        req = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=OUTPUT_WITH_THINKING,
            reasoning_tokens=len(THINKING_TOKENS),
        )

        cache.cache_finished_req(req, is_insert=True)

        self.assertEqual(
            cache.total_size(),
            len(PROMPT_TOKENS),
            "Cache size should equal prompt length when reasoning active",
        )

    def test_cache_size_all_tokens_without_reasoning(self):
        """Without reasoning, cache size = all tokens."""
        cache, pool, alloc = _make_cache()
        req = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=OUTPUT_WITH_THINKING,
            reasoning_tokens=0,
        )

        cache.cache_finished_req(req, is_insert=True)

        expected = len(PROMPT_TOKENS) + len(OUTPUT_WITH_THINKING)
        self.assertEqual(cache.total_size(), expected)


# ===================================================================
# Test Suite 6: is_insert=False path
# ===================================================================


class TestInsertFalse(unittest.TestCase):
    """When is_insert=False, nothing is inserted — only KV slots freed."""

    def test_is_insert_false_frees_kv_and_empty_tree(self):
        """is_insert=False → empty tree, KV slots freed."""
        cache, pool, alloc = _make_cache()
        req = _prepare_req(
            cache,
            pool,
            prompt=PROMPT_TOKENS,
            output=OUTPUT_WITH_THINKING,
            reasoning_tokens=len(THINKING_TOKENS),
        )

        cache.cache_finished_req(req, is_insert=False)

        self.assertEqual(cache.total_size(), 0, "is_insert=False → empty tree")

        # KV slots should be freed (all non-protected)
        freed = _all_freed(alloc)
        self.assertGreater(len(freed), 0, "Some KV slots must be freed")


if __name__ == "__main__":
    unittest.main()
