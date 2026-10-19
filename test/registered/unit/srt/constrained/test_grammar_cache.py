"""Unit tests for the bounded (LRU) compiled-grammar cache in BaseGrammarBackend.

Target path upstream: test/registered/unit/srt/constrained/test_grammar_cache.py
(mirrors python/sglang/srt/constrained/base_grammar_backend.py)
"""

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that may pull in sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import unittest
from unittest import mock

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
)
from sglang.test.test_utils import CustomTestCase


class _DummyGrammar(BaseGrammarObject):
    """Minimal grammar object: copy() returns a distinct instance like real backends."""

    def __init__(self, tag: str):
        super().__init__()
        self.tag = tag
        self.copies = 0

    def copy(self):
        self.copies += 1
        return _DummyGrammar(self.tag)

    def maybe_init_reasoning(self, require_reasoning: bool):
        pass


class _StubBackend(BaseGrammarBackend):
    """Backend whose dispatch is synchronous and counts compilations."""

    def __init__(self, cache_max_entries=None):
        super().__init__(cache_max_entries=cache_max_entries)
        self.compiled = []

    def dispatch_json(self, key_string: str) -> BaseGrammarObject:
        self.compiled.append(key_string)
        return _DummyGrammar(key_string)


def _fill(backend: _StubBackend, n: int, prefix: str = "s"):
    for i in range(n):
        key = ("json", f"{prefix}{i}")
        backend.set_cache(key, _DummyGrammar(key[1]))


class TestGrammarCacheBound(CustomTestCase):
    def test_default_bound_from_class_constant(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_GRAMMAR_CACHE_MAX_ENTRIES", None)
            b = _StubBackend()
        self.assertEqual(
            b.cache_max_entries, BaseGrammarBackend.DEFAULT_CACHE_MAX_ENTRIES
        )

    def test_env_var_overrides_bound(self):
        with mock.patch.dict(os.environ, {"SGLANG_GRAMMAR_CACHE_MAX_ENTRIES": "3"}):
            b = _StubBackend()
        self.assertEqual(b.cache_max_entries, 3)

    def test_constructor_argument_wins_over_env(self):
        with mock.patch.dict(os.environ, {"SGLANG_GRAMMAR_CACHE_MAX_ENTRIES": "3"}):
            b = _StubBackend(cache_max_entries=7)
        self.assertEqual(b.cache_max_entries, 7)

    def test_evicts_oldest_when_full(self):
        b = _StubBackend(cache_max_entries=3)
        _fill(b, 5)
        self.assertEqual(len(b.cache), 3)
        self.assertEqual([k[1] for k in b.cache], ["s2", "s3", "s4"])

    def test_hit_refreshes_recency(self):
        b = _StubBackend(cache_max_entries=3)
        _fill(b, 3)  # s0 s1 s2
        value, cached = b.get_cached_or_future_value(
            ("json", "s0"), require_reasoning=False
        )
        self.assertTrue(cached)
        self.assertIsInstance(value, _DummyGrammar)
        b.set_cache(("json", "s3"), _DummyGrammar("s3"))  # should evict s1, not s0
        self.assertEqual([k[1] for k in b.cache], ["s2", "s0", "s3"])

    def test_hit_returns_copy_not_cached_instance(self):
        b = _StubBackend(cache_max_entries=2)
        original = _DummyGrammar("x")
        b.set_cache(("json", "x"), original)
        value, cached = b.get_cached_or_future_value(
            ("json", "x"), require_reasoning=False
        )
        self.assertTrue(cached)
        self.assertIsNot(value, original)
        self.assertEqual(original.copies, 1)

    def test_eviction_does_not_affect_in_flight_copy(self):
        b = _StubBackend(cache_max_entries=1)
        b.set_cache(("json", "a"), _DummyGrammar("a"))
        in_flight, _ = b.get_cached_or_future_value(
            ("json", "a"), require_reasoning=False
        )
        b.set_cache(("json", "b"), _DummyGrammar("b"))  # evicts "a"
        self.assertNotIn(("json", "a"), b.cache)
        self.assertEqual(in_flight.tag, "a")  # request still holds a usable grammar

    def test_miss_after_eviction_recompiles(self):
        b = _StubBackend(cache_max_entries=1)
        fut, cached = b.get_cached_or_future_value(
            ("json", "a"), require_reasoning=False
        )
        self.assertFalse(cached)
        b.set_cache(("json", "a"), fut.result())
        b.set_cache(("json", "b"), _DummyGrammar("b"))
        fut2, cached2 = b.get_cached_or_future_value(
            ("json", "a"), require_reasoning=False
        )
        self.assertFalse(cached2)
        fut2.result()
        self.assertEqual(b.compiled, ["a", "a"])

    def test_zero_or_negative_disables_bound(self):
        for v in (0, -1):
            b = _StubBackend(cache_max_entries=v)
            _fill(b, 10)
            self.assertEqual(len(b.cache), 10)

    def test_reset_clears(self):
        b = _StubBackend(cache_max_entries=4)
        _fill(b, 4)
        b.reset()
        self.assertEqual(len(b.cache), 0)

    def test_same_key_reinsert_keeps_single_entry(self):
        b = _StubBackend(cache_max_entries=2)
        for _ in range(5):
            b.set_cache(("json", "same"), _DummyGrammar("same"))
        self.assertEqual(len(b.cache), 1)


if __name__ == "__main__":
    unittest.main()
