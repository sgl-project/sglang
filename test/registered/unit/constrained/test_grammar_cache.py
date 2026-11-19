"""Unit tests for the bounded (LRU) compiled-grammar cache in BaseGrammarBackend.

Target path upstream: test/registered/unit/constrained/test_grammar_cache.py
(mirrors python/sglang/srt/constrained/base_grammar_backend.py)
"""

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that may pull in sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from typing import Optional
from unittest import mock

from xgrammar import TokenizerInfo

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
    InvalidGrammarObject,
)
from sglang.srt.constrained.reasoner_grammar_backend import (
    ReasonerGrammarBackend,
    ReasonerGrammarObject,
)
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.test.test_utils import CustomTestCase

NOMINAL = BaseGrammarBackend.NOMINAL_GRAMMAR_BYTES


def _kb(key_string: str, key_type: str = "json") -> int:
    """Bytes the cache charges for the key text itself."""
    return BaseGrammarBackend.key_bytes((key_type, key_string))


# Every entry in the LRU tests costs 1 byte plus its ("json", "sN") key.
UNIT = 1 + _kb("s0")


class _DummyGrammar(BaseGrammarObject):
    """Minimal grammar object: copy() returns a distinct instance like real backends.

    ``size`` is what memory_size_bytes() reports; None mimics a backend without
    a size API (outlines, llguidance).
    """

    def __init__(self, tag: str, size: Optional[int] = None):
        super().__init__()
        self.tag = tag
        self.size = size
        self.copies = 0

    def copy(self):
        self.copies += 1
        return _DummyGrammar(self.tag, self.size)

    def memory_size_bytes(self):
        return self.size

    def maybe_init_reasoning(self, require_reasoning: bool):
        pass


class _StubBackend(BaseGrammarBackend):
    """Backend whose dispatch is synchronous and counts compilations."""

    def __init__(self, cache_limit_bytes=None):
        super().__init__(cache_limit_bytes=cache_limit_bytes)
        self.compiled = []

    def dispatch_json(self, key_string: str) -> BaseGrammarObject:
        self.compiled.append(key_string)
        return _DummyGrammar(key_string)


def _fill(backend: _StubBackend, n: int, prefix: str = "s", size=None):
    for i in range(n):
        key = ("json", f"{prefix}{i}")
        backend.set_cache(key, _DummyGrammar(key[1], size))


def _keys(backend: BaseGrammarBackend):
    return [k[1] for k in backend.cache]


class TestGrammarCacheLRU(CustomTestCase):
    """LRU behaviour under the byte budget; entries here cost UNIT bytes
    each (1 byte plus the key), so the budget doubles as an entry count."""

    def test_default_budget(self):
        b = _StubBackend()
        self.assertEqual(
            b.cache_limit_bytes, BaseGrammarBackend.DEFAULT_CACHE_LIMIT_BYTES
        )
        self.assertEqual(b.cache_size_bytes, 0)
        self.assertFalse(hasattr(b, "cache_max_entries"))

    def test_constructor_argument_overrides_budget(self):
        b = _StubBackend(cache_limit_bytes=7)
        self.assertEqual(b.cache_limit_bytes, 7)

    def test_evicts_oldest_when_full(self):
        b = _StubBackend(cache_limit_bytes=3 * UNIT)
        _fill(b, 5, size=1)
        self.assertEqual(len(b.cache), 3)
        self.assertEqual(_keys(b), ["s2", "s3", "s4"])

    def test_hit_refreshes_recency(self):
        b = _StubBackend(cache_limit_bytes=3 * UNIT)
        _fill(b, 3, size=1)  # s0 s1 s2
        value, cached = b.get_cached_or_future_value(
            ("json", "s0"), require_reasoning=False
        )
        self.assertTrue(cached)
        self.assertIsInstance(value, _DummyGrammar)
        b.set_cache(("json", "s3"), _DummyGrammar("s3", 1))  # evicts s1, not s0
        self.assertEqual(_keys(b), ["s2", "s0", "s3"])

    def test_hit_returns_copy_not_cached_instance(self):
        b = _StubBackend(cache_limit_bytes=2 * UNIT)
        original = _DummyGrammar("x", 1)
        b.set_cache(("json", "x"), original)
        value, cached = b.get_cached_or_future_value(
            ("json", "x"), require_reasoning=False
        )
        self.assertTrue(cached)
        self.assertIsNot(value, original)
        self.assertEqual(original.copies, 1)

    def test_eviction_does_not_affect_in_flight_copy(self):
        b = _StubBackend(cache_limit_bytes=UNIT)
        b.set_cache(("json", "a"), _DummyGrammar("a", 1))
        in_flight, _ = b.get_cached_or_future_value(
            ("json", "a"), require_reasoning=False
        )
        b.set_cache(("json", "b"), _DummyGrammar("b", 1))  # evicts "a"
        self.assertNotIn(("json", "a"), b.cache)
        self.assertEqual(in_flight.tag, "a")  # request still holds a usable grammar

    def test_miss_after_eviction_recompiles(self):
        # dispatch_json yields size-less grammars, charged NOMINAL each
        b = _StubBackend(cache_limit_bytes=NOMINAL)
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

    def test_zero_or_negative_disables_the_bound(self):
        for v in (0, -1):
            b = _StubBackend(cache_limit_bytes=v)
            _fill(b, 10, size=10 * NOMINAL)
            self.assertEqual(len(b.cache), 10)

    def test_no_entry_count_bound(self):
        """Only memory limits the cache: many small grammars all stay."""
        b = _StubBackend(cache_limit_bytes=1 << 30)
        _fill(b, 1000, size=1)
        self.assertEqual(len(b.cache), 1000)

    def test_reset_clears(self):
        b = _StubBackend(cache_limit_bytes=4 * UNIT)
        _fill(b, 4, size=1)
        b.reset()
        self.assertEqual(len(b.cache), 0)

    def test_same_key_reinsert_keeps_single_entry(self):
        b = _StubBackend(cache_limit_bytes=2 * UNIT)
        for _ in range(5):
            b.set_cache(("json", "same"), _DummyGrammar("same", 1))
        self.assertEqual(len(b.cache), 1)


class TestGrammarCacheByteBudget(CustomTestCase):
    """The byte budget is the only bound (DEFAULT_CACHE_LIMIT_BYTES)."""

    def test_evicts_oldest_past_byte_budget(self):
        b = _StubBackend(cache_limit_bytes=3 * 40 + 2 * _kb("s0"))
        _fill(b, 3, size=40)  # three entries exceed the budget -> s0 goes
        self.assertEqual(_keys(b), ["s1", "s2"])
        self.assertEqual(b.cache_size_bytes, 2 * (40 + _kb("s0")))

    def test_tracks_bytes_of_the_cached_grammars(self):
        b = _StubBackend(cache_limit_bytes=0)
        b.set_cache(("json", "a"), _DummyGrammar("a", 7))
        b.set_cache(("regex", "b"), _DummyGrammar("b", 11))
        self.assertEqual(b.cache_size_bytes, 7 + _kb("a") + 11 + _kb("b", "regex"))

    def test_key_text_is_charged(self):
        """The cache retains the schema text as the key; a 60 KB schema
        costs 60 KB even if the backend reports a tiny grammar."""
        b = _StubBackend(cache_limit_bytes=0)
        big_key = "x" * 60_000
        b.set_cache(("json", big_key), _DummyGrammar("g", 1))
        self.assertEqual(b.cache_size_bytes, 1 + len("json") + 60_000)

    def test_unknown_size_is_charged_nominally(self):
        b = _StubBackend(cache_limit_bytes=0)
        b.set_cache(("json", "none"), _DummyGrammar("none", None))
        b.set_cache(("json", "bad"), _DummyGrammar("bad", -5))
        b.set_cache(("json", "mock"), mock.MagicMock(spec=BaseGrammarObject))
        self.assertEqual(
            b.cache_size_bytes,
            3 * NOMINAL + _kb("none") + _kb("bad") + _kb("mock"),
        )

    def test_invalid_grammar_is_charged_key_and_message(self):
        b = _StubBackend(cache_limit_bytes=0)
        b.set_cache(("json", "bad"), InvalidGrammarObject("boom"))
        self.assertEqual(b.cache_size_bytes, _kb("bad") + len("boom"))

    def test_distinct_invalid_grammars_do_not_accumulate(self):
        """The scheduler caches failed compilations; a stream of distinct
        invalid schemas must be evicted like any other entry."""
        b = _StubBackend(cache_limit_bytes=500)
        for i in range(200):
            b.set_cache(("json", f"broken-{i:04d}"), InvalidGrammarObject("bad"))
        self.assertLess(len(b.cache), 200)
        self.assertLessEqual(b.cache_size_bytes, 500)
        self.assertEqual(_keys(b)[-1], "broken-0199")  # newest kept

    def test_single_oversized_entry_is_kept(self):
        """A grammar larger than the whole budget must still be cached, or a
        client repeating it would recompile on every request."""
        b = _StubBackend(cache_limit_bytes=100)
        b.set_cache(("json", "small"), _DummyGrammar("small", 10))
        b.set_cache(("json", "huge"), _DummyGrammar("huge", 1000))
        self.assertEqual(_keys(b), ["huge"])
        self.assertEqual(b.cache_size_bytes, 1000 + _kb("huge"))
        _, cached = b.get_cached_or_future_value(
            ("json", "huge"), require_reasoning=False
        )
        self.assertTrue(cached)

    def test_replacing_a_key_adjusts_the_charge(self):
        b = _StubBackend(cache_limit_bytes=0)
        b.set_cache(("json", "k"), _DummyGrammar("k", 50))
        b.set_cache(("json", "k"), _DummyGrammar("k", 20))
        self.assertEqual(len(b.cache), 1)
        self.assertEqual(b.cache_size_bytes, 20 + _kb("k"))

    def test_discard_cache_releases_the_charge(self):
        b = _StubBackend(cache_limit_bytes=0)
        _fill(b, 2, size=30)
        b.discard_cache(("json", "s0"))
        b.discard_cache(("json", "missing"))  # no-op
        self.assertEqual(_keys(b), ["s1"])
        self.assertEqual(b.cache_size_bytes, 30 + _kb("s1"))

    def test_reset_zeroes_the_charge(self):
        b = _StubBackend(cache_limit_bytes=0)
        _fill(b, 3, size=30)
        b.reset()
        self.assertEqual(len(b.cache), 0)
        self.assertEqual(b.cache_size_bytes, 0)
        _fill(b, 1, size=5)  # usable again after reset
        self.assertEqual(b.cache_size_bytes, 5 + _kb("s0"))


def _tiny_xgrammar_backend(**kwargs) -> XGrammarGrammarBackend:
    vocab = [chr(i) for i in range(128)] + ["<eos>"]
    info = TokenizerInfo(vocab, stop_token_ids=[128])
    tokenizer = SimpleNamespace(init_xgrammar=lambda: (info, None))
    return XGrammarGrammarBackend(tokenizer, len(vocab), **kwargs)


class TestXGrammarCompilerCacheBudget(CustomTestCase):
    """xgrammar's GrammarCompiler keeps its own cache of the same compiled
    grammars; it must share the budget or evicting from ours frees nothing."""

    SCHEMA = '{"type": "object", "properties": {"a": {"type": "integer"}}}'

    def test_compiler_gets_the_same_budget(self):
        b = _tiny_xgrammar_backend(cache_limit_bytes=12345)
        self.assertEqual(b.cache_limit_bytes, 12345)
        self.assertEqual(b.grammar_compiler.cache_limit_bytes, 12345)

    def test_default_budget_reaches_the_compiler(self):
        b = _tiny_xgrammar_backend()
        self.assertEqual(
            b.grammar_compiler.cache_limit_bytes,
            BaseGrammarBackend.DEFAULT_CACHE_LIMIT_BYTES,
        )

    def test_disabled_budget_means_unlimited_compiler_cache(self):
        b = _tiny_xgrammar_backend(cache_limit_bytes=0)
        self.assertEqual(b.grammar_compiler.cache_limit_bytes, -1)

    def test_compiled_grammar_reports_its_size(self):
        b = _tiny_xgrammar_backend(cache_limit_bytes=1 << 20)
        grammar = b.dispatch_json(self.SCHEMA)
        self.assertNotIsInstance(grammar, InvalidGrammarObject)
        size = grammar.memory_size_bytes()
        self.assertIsInstance(size, int)
        self.assertGreater(size, 0)
        b.set_cache(("json", self.SCHEMA), grammar)
        self.assertEqual(b.cache_size_bytes, size + _kb(self.SCHEMA))
        self.assertGreater(b.grammar_compiler.get_cache_size_bytes(), 0)

    def test_reset_clears_both_caches(self):
        b = _tiny_xgrammar_backend(cache_limit_bytes=1 << 20)
        b.set_cache(("json", self.SCHEMA), b.dispatch_json(self.SCHEMA))
        b.reset()
        self.assertEqual(len(b.cache), 0)
        self.assertEqual(b.cache_size_bytes, 0)
        self.assertEqual(b.grammar_compiler.get_cache_size_bytes(), 0)


class _DummyTokenizer:
    def __init__(self, token_map):
        self._token_map = token_map

    def encode(self, text, add_special_tokens=False):
        return list(self._token_map.get(text, []))

    def decode(self, token_ids, **kwargs):
        return ""


def _reasoner_parts():
    detector = SimpleNamespace(
        think_start_token="<think>",
        think_end_token="</think>",
        think_excluded_tokens=["<tool_call>", "</tool_call>"],
        get_think_end_token_ids=lambda tokenizer: tokenizer.encode(
            "</think>", add_special_tokens=False
        ),
        get_answer_frame=lambda: None,
    )
    tokenizer = _DummyTokenizer(
        {"<think>": [1], "</think>": [2], "<tool_call>": [3], "</tool_call>": [4]}
    )
    return SimpleNamespace(detector=detector), tokenizer


class TestReasonerGrammarBackendCache(CustomTestCase):
    def test_reset_reaches_the_wrapped_backend(self):
        """/flush_cache calls reset() on the outer backend only; with a
        reasoning parser the xgrammar compiler cache lives in the inner one."""
        parser, tokenizer = _reasoner_parts()
        inner = _StubBackend(cache_limit_bytes=0)
        inner.set_cache(("json", "inner"), _DummyGrammar("inner", 10))
        reasoner = ReasonerGrammarBackend(inner, parser, tokenizer)
        reasoner.set_cache(("json", "outer"), _DummyGrammar("outer", 10))
        reasoner.reset()
        self.assertEqual(len(reasoner.cache), 0)
        self.assertEqual(reasoner.cache_size_bytes, 0)
        self.assertEqual(len(inner.cache), 0)
        self.assertEqual(inner.cache_size_bytes, 0)

    def test_reset_calls_inner_reset_even_for_mock_backend(self):
        parser, tokenizer = _reasoner_parts()
        inner = mock.MagicMock(spec=BaseGrammarBackend)
        reasoner = ReasonerGrammarBackend(inner, parser, tokenizer)
        reasoner.reset()
        inner.reset.assert_called_once_with()

    def test_takes_budget_from_wrapped_backend_by_default(self):
        parser, tokenizer = _reasoner_parts()
        inner = _StubBackend(cache_limit_bytes=777)
        reasoner = ReasonerGrammarBackend(inner, parser, tokenizer)
        self.assertEqual(reasoner.cache_limit_bytes, 777)

    def test_explicit_budget_wins(self):
        parser, tokenizer = _reasoner_parts()
        inner = _StubBackend(cache_limit_bytes=777)
        reasoner = ReasonerGrammarBackend(
            inner, parser, tokenizer, cache_limit_bytes=555
        )
        self.assertEqual(reasoner.cache_limit_bytes, 555)

    def test_wrapper_object_reports_inner_size(self):
        wrapped = ReasonerGrammarObject(_DummyGrammar("g", 42), think_end_ids=[2])
        self.assertEqual(wrapped.memory_size_bytes(), 42)
        self.assertEqual(
            ReasonerGrammarObject(None, think_end_ids=[2]).memory_size_bytes(), 0
        )

    def test_byte_budget_applies_to_wrapped_grammars(self):
        parser, tokenizer = _reasoner_parts()
        inner = _StubBackend(cache_limit_bytes=3 * 40 + 2 * _kb("s0"))
        reasoner = ReasonerGrammarBackend(inner, parser, tokenizer)
        for i in range(3):
            reasoner.set_cache(
                ("json", f"s{i}"),
                ReasonerGrammarObject(_DummyGrammar(f"s{i}", 40), think_end_ids=[2]),
            )
        self.assertEqual(_keys(reasoner), ["s1", "s2"])
        self.assertEqual(reasoner.cache_size_bytes, 2 * (40 + _kb("s0")))


if __name__ == "__main__":
    unittest.main()
