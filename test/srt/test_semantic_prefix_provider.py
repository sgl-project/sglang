"""Tests for SemanticPrefixProvider integration in RadixCache.

All tests run without a GPU — they use :meth:`RadixCache.create_simulated`
which requires no physical memory pools.
"""

from __future__ import annotations

import unittest
from typing import Optional
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.semantic_prefix import (
    SemanticPrefixProvider,
    SemanticPrefixResult,
)


# ──────────────────────────────────────────────────────────────
# Helper provider implementations
# ──────────────────────────────────────────────────────────────


class _NullProvider(SemanticPrefixProvider):
    """Provider that never returns a donor."""

    def on_prefix_miss(self, rid: str, token_ids: list[int]):
        return None

    def on_request_cached(self, rid: str, token_ids: list[int]) -> None:
        pass


class _FixedDonorProvider(SemanticPrefixProvider):
    """Provider that always returns a fixed donor sequence."""

    def __init__(self, donor_ids: list[int], source_id: str = "") -> None:
        self.donor_ids = donor_ids
        self.source_id = source_id
        self.miss_calls: list[tuple] = []
        self.cached_calls: list[tuple] = []

    def on_prefix_miss(
        self, rid: str, token_ids: list[int]
    ) -> Optional[SemanticPrefixResult]:
        self.miss_calls.append((rid, token_ids))
        return SemanticPrefixResult(
            alternate_token_ids=self.donor_ids,
            num_cached_tokens=len(self.donor_ids),
            source_id=self.source_id,
        )

    def on_request_cached(self, rid: str, token_ids: list[int]) -> None:
        self.cached_calls.append((rid, token_ids))


class _RaisingProvider(SemanticPrefixProvider):
    """Provider that always raises (used to test exception propagation)."""

    def on_prefix_miss(self, rid: str, token_ids: list[int]):
        raise RuntimeError("deliberate test error")

    def on_request_cached(self, rid: str, token_ids: list[int]) -> None:
        pass


# ──────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────


def _make_cache() -> RadixCache:
    return RadixCache.create_simulated()


def _insert(cache: RadixCache, token_ids: list[int]) -> None:
    cache.insert(
        InsertParams(
            key=RadixKey(token_ids),
            value=torch.tensor(token_ids, dtype=torch.int64),
        )
    )


def _mock_req(rid: str = "r1") -> MagicMock:
    req = MagicMock()
    req.rid = rid
    return req


def _make_req_for_finished(
    cache: RadixCache, rid: str, token_ids: list[int]
) -> MagicMock:
    """Return a Req-like mock suitable for cache_finished_req."""
    req = MagicMock()
    req.rid = rid
    req.origin_input_ids = token_ids
    req.output_ids = []
    req.extra_key = None
    req.req_pool_idx = 0
    req.last_node = cache.root_node  # avoids dec_lock_ref tree traversal
    req.cache_protected_len = 0
    req.priority = 0
    req.pop_committed_kv_cache.return_value = len(token_ids)
    return req


def _attach_pool(cache: RadixCache, num_tokens: int) -> None:
    """Wire minimal mock memory pools onto a simulated cache."""
    kv_pool = torch.arange(num_tokens * 2, dtype=torch.int64).reshape(2, num_tokens)
    req_to_token = MagicMock()
    req_to_token.req_to_token = kv_pool
    cache.req_to_token_pool = req_to_token
    allocator = MagicMock()
    allocator.free = MagicMock()
    cache.token_to_kv_pool_allocator = allocator


# ──────────────────────────────────────────────────────────────
# Test suites
# ──────────────────────────────────────────────────────────────


class TestSetSemanticProvider(unittest.TestCase):
    """Tests for set_semantic_provider()."""

    def test_stores_provider(self):
        cache = _make_cache()
        p = _NullProvider()
        cache.set_semantic_provider(p)
        self.assertIs(cache._semantic_provider, p)

    def test_on_init_called_once(self):
        cache = _make_cache()
        p = MagicMock(spec=SemanticPrefixProvider)
        cache.set_semantic_provider(p)
        p.on_init.assert_called_once_with()

    def test_set_none_clears_provider(self):
        cache = _make_cache()
        p = _NullProvider()
        cache.set_semantic_provider(p)
        cache.set_semantic_provider(None)
        self.assertIsNone(cache._semantic_provider)

    def test_set_none_does_not_call_on_init(self):
        """Passing None should not raise and should not call on_init."""
        cache = _make_cache()
        # Should not raise — no on_init to call
        cache.set_semantic_provider(None)

    def test_replace_provider(self):
        """Replacing a provider calls on_init on the new one only."""
        cache = _make_cache()
        p1 = MagicMock(spec=SemanticPrefixProvider)
        p2 = MagicMock(spec=SemanticPrefixProvider)
        cache.set_semantic_provider(p1)
        cache.set_semantic_provider(p2)
        p1.on_init.assert_called_once()
        p2.on_init.assert_called_once()
        self.assertIs(cache._semantic_provider, p2)

    def test_provider_none_by_default(self):
        cache = _make_cache()
        self.assertIsNone(cache._semantic_provider)


class TestMatchPrefixNoProvider(unittest.TestCase):
    """Baseline: no semantic provider — exact-match behaviour unchanged."""

    def test_miss_returns_empty(self):
        cache = _make_cache()
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3])))
        self.assertEqual(len(result.device_indices), 0)

    def test_hit_returns_indices(self):
        cache = _make_cache()
        ids = [10, 20, 30]
        _insert(cache, ids)
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(ids)))
        self.assertEqual(len(result.device_indices), len(ids))

    def test_partial_hit(self):
        cache = _make_cache()
        _insert(cache, [1, 2, 3, 4, 5])
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3])))
        self.assertGreater(len(result.device_indices), 0)


class TestMatchPrefixSemanticFallback(unittest.TestCase):
    """Tests for the semantic fallback path in match_prefix."""

    # ── provider not triggered on exact hit ─────────────────────────────────

    def test_provider_not_called_on_exact_hit(self):
        ids = [10, 20, 30, 40, 50]
        cache = _make_cache()
        _insert(cache, ids)
        p = MagicMock(spec=SemanticPrefixProvider)
        cache.set_semantic_provider(p)

        req = _mock_req()
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(ids), req=req))

        self.assertGreater(len(result.device_indices), 0)
        p.on_prefix_miss.assert_not_called()

    # ── provider not triggered without params.req ────────────────────────────

    def test_provider_not_called_without_req(self):
        cache = _make_cache()
        p = _FixedDonorProvider(donor_ids=[1, 2, 3])
        cache.set_semantic_provider(p)

        # params.req is None (default) → fallback must not activate
        cache.match_prefix(MatchPrefixParams(key=RadixKey([99, 88, 77])))

        self.assertEqual(len(p.miss_calls), 0)

    # ── provider called on exact miss with req ───────────────────────────────

    def test_provider_called_on_miss(self):
        cache = _make_cache()
        p = _NullProvider()
        p.on_prefix_miss = MagicMock(return_value=None)
        cache.set_semantic_provider(p)

        req = _mock_req(rid="req-1")
        query = [5, 6, 7]
        cache.match_prefix(MatchPrefixParams(key=RadixKey(query), req=req))

        p.on_prefix_miss.assert_called_once_with(rid="req-1", token_ids=query)

    def test_provider_receives_original_token_ids(self):
        """Provider gets the original (pre-page-alignment) token IDs."""
        received: list[list[int]] = []

        class _Recorder(SemanticPrefixProvider):
            def on_prefix_miss(self, rid, token_ids):
                received.append(list(token_ids))
                return None

            def on_request_cached(self, rid, token_ids):
                pass

        cache = _make_cache()
        cache.set_semantic_provider(_Recorder())
        query = [7, 8, 9, 11]
        cache.match_prefix(MatchPrefixParams(key=RadixKey(query), req=_mock_req()))
        self.assertEqual(received, [query])

    # ── provider returns None → cold prefill ────────────────────────────────

    def test_provider_returns_none_result_stays_empty(self):
        cache = _make_cache()
        cache.set_semantic_provider(_NullProvider())
        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 3]), req=_mock_req())
        )
        self.assertEqual(len(result.device_indices), 0)

    # ── provider returns alternate donor IDs ────────────────────────────────

    def test_alternate_tokens_cached_gives_hit(self):
        donor_ids = [10, 20, 30, 40, 50]
        cache = _make_cache()
        _insert(cache, donor_ids)

        p = _FixedDonorProvider(donor_ids=donor_ids, source_id="doc-42")
        cache.set_semantic_provider(p)

        req = _mock_req(rid="req-2")
        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey([99, 88, 77]), req=req)
        )

        self.assertGreater(len(result.device_indices), 0)
        self.assertEqual(len(p.miss_calls), 1)
        self.assertEqual(p.miss_calls[0][0], "req-2")

    def test_alternate_tokens_not_cached_stays_empty(self):
        cache = _make_cache()  # empty — donor IDs not inserted
        p = _FixedDonorProvider(donor_ids=[55, 66, 77])
        cache.set_semantic_provider(p)

        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 3]), req=_mock_req())
        )
        self.assertEqual(len(result.device_indices), 0)

    def test_extra_key_preserved_in_alternate_lookup(self):
        """extra_key from the original query is used when looking up alternate tokens."""
        donor_ids = [10, 20, 30]
        extra_key = "lora-7"
        cache = _make_cache()
        cache.insert(
            InsertParams(
                key=RadixKey(donor_ids, extra_key=extra_key),
                value=torch.tensor(donor_ids, dtype=torch.int64),
            )
        )

        p = _FixedDonorProvider(donor_ids=donor_ids)
        cache.set_semantic_provider(p)

        req = _mock_req()
        result = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey([99, 88], extra_key=extra_key), req=req
            )
        )
        self.assertGreater(len(result.device_indices), 0)

    # ── exception propagation ────────────────────────────────────────────────

    def test_provider_exception_propagates(self):
        """Exceptions from the provider are not silently swallowed."""
        cache = _make_cache()
        cache.set_semantic_provider(_RaisingProvider())

        with self.assertRaises(RuntimeError, msg="deliberate test error"):
            cache.match_prefix(
                MatchPrefixParams(key=RadixKey([1, 2, 3]), req=_mock_req())
            )

    # ── logging ─────────────────────────────────────────────────────────────

    def test_source_id_logged_on_semantic_hit(self):
        """A non-empty source_id triggers a DEBUG log entry."""
        donor_ids = [5, 6, 7, 8]
        cache = _make_cache()
        _insert(cache, donor_ids)

        p = _FixedDonorProvider(donor_ids=donor_ids, source_id="my-donor")
        cache.set_semantic_provider(p)

        req = _mock_req()
        with self.assertLogs(
            "sglang.srt.mem_cache.radix_cache", level="DEBUG"
        ) as cm:
            cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3]), req=req))

        self.assertTrue(
            any("my-donor" in line for line in cm.output),
            f"'my-donor' not found in log output: {cm.output}",
        )

    def test_no_log_when_source_id_empty(self):
        """An empty source_id must not produce a log entry at DEBUG level."""
        donor_ids = [5, 6, 7, 8]
        cache = _make_cache()
        _insert(cache, donor_ids)

        p = _FixedDonorProvider(donor_ids=donor_ids, source_id="")  # empty
        cache.set_semantic_provider(p)

        req = _mock_req()
        import logging

        with self.assertLogs(
            "sglang.srt.mem_cache.radix_cache", level="DEBUG"
        ) as cm:
            # Force at least one log entry so assertLogs doesn't fail on empty
            logging.getLogger("sglang.srt.mem_cache.radix_cache").debug("sentinel")
            cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3]), req=req))

        semantic_logs = [
            line
            for line in cm.output
            if "Semantic KV hit" in line
        ]
        self.assertEqual(len(semantic_logs), 0)


class TestMatchPrefixExact(unittest.TestCase):
    """_match_prefix_exact never invokes the semantic provider."""

    def test_exact_hit(self):
        cache = _make_cache()
        ids = [1, 2, 3]
        _insert(cache, ids)
        result = cache._match_prefix_exact(MatchPrefixParams(key=RadixKey(ids)))
        self.assertEqual(len(result.device_indices), len(ids))

    def test_exact_miss_returns_empty(self):
        cache = _make_cache()
        result = cache._match_prefix_exact(
            MatchPrefixParams(key=RadixKey([99, 88, 77]))
        )
        self.assertEqual(len(result.device_indices), 0)

    def test_does_not_call_provider(self):
        cache = _make_cache()
        p = MagicMock(spec=SemanticPrefixProvider)
        cache.set_semantic_provider(p)

        cache._match_prefix_exact(
            MatchPrefixParams(key=RadixKey([1, 2, 3]), req=_mock_req())
        )
        p.on_prefix_miss.assert_not_called()


class TestOnRequestCachedHook(unittest.TestCase):
    """cache_finished_req must notify the provider after a successful insert."""

    def _make_cache_with_pool(
        self, provider: SemanticPrefixProvider, num_tokens: int = 10
    ) -> RadixCache:
        cache = _make_cache()
        cache.set_semantic_provider(provider)
        _attach_pool(cache, num_tokens)
        return cache

    def test_called_on_insert(self):
        p = MagicMock(spec=SemanticPrefixProvider)
        p.on_init = MagicMock()
        p.on_request_cached = MagicMock()
        cache = self._make_cache_with_pool(p, num_tokens=8)

        token_ids = [1, 2, 3, 4]
        req = _make_req_for_finished(cache, "req-a", token_ids)

        cache.cache_finished_req(req, is_insert=True)

        p.on_request_cached.assert_called_once()
        kwargs = p.on_request_cached.call_args.kwargs
        self.assertEqual(kwargs["rid"], "req-a")
        self.assertEqual(kwargs["token_ids"], token_ids)

    def test_not_called_when_no_insert(self):
        p = MagicMock(spec=SemanticPrefixProvider)
        p.on_init = MagicMock()
        p.on_request_cached = MagicMock()
        cache = self._make_cache_with_pool(p, num_tokens=8)

        token_ids = [5, 6, 7, 8]
        req = _make_req_for_finished(cache, "req-b", token_ids)

        cache.cache_finished_req(req, is_insert=False)

        p.on_request_cached.assert_not_called()

    def test_not_called_without_provider(self):
        """No provider → cache_finished_req must not raise."""
        cache = _make_cache()
        _attach_pool(cache, num_tokens=8)

        token_ids = [1, 2, 3]
        req = _make_req_for_finished(cache, "req-c", token_ids)

        # Must not raise AttributeError
        cache.cache_finished_req(req, is_insert=True)

    def test_token_ids_match_committed_range(self):
        """on_request_cached receives exactly the committed token IDs."""
        p = MagicMock(spec=SemanticPrefixProvider)
        p.on_init = MagicMock()
        p.on_request_cached = MagicMock()
        cache = self._make_cache_with_pool(p, num_tokens=10)

        full_ids = list(range(6))
        committed_len = 4  # only first 4 tokens are "committed"
        req = _make_req_for_finished(cache, "req-d", full_ids[:committed_len])
        req.pop_committed_kv_cache.return_value = committed_len

        cache.cache_finished_req(req, is_insert=True)

        kwargs = p.on_request_cached.call_args.kwargs
        self.assertEqual(kwargs["token_ids"], full_ids[:committed_len])


class TestMultipleRequests(unittest.TestCase):
    """Semantic provider is called independently for each request."""

    def test_each_miss_calls_provider(self):
        cache = _make_cache()
        p = _NullProvider()
        p.on_prefix_miss = MagicMock(return_value=None)
        cache.set_semantic_provider(p)

        for i in range(3):
            req = _mock_req(rid=f"req-{i}")
            cache.match_prefix(MatchPrefixParams(key=RadixKey([i * 10]), req=req))

        self.assertEqual(p.on_prefix_miss.call_count, 3)

    def test_second_provider_replaces_first(self):
        cache = _make_cache()
        p1 = MagicMock(spec=SemanticPrefixProvider)
        p2 = MagicMock(spec=SemanticPrefixProvider)
        p2.on_prefix_miss = MagicMock(return_value=None)
        cache.set_semantic_provider(p1)
        cache.set_semantic_provider(p2)

        req = _mock_req()
        cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3]), req=req))

        p1.on_prefix_miss.assert_not_called()
        p2.on_prefix_miss.assert_called_once()


if __name__ == "__main__":
    unittest.main()
