"""
Unit tests for sglang.srt.constrained.base_grammar_backend.

Test Coverage:
- GrammarStats: default values, mutable default isolation
- BaseGrammarObject: default behavior
- InvalidGrammarObject: error message
- BaseGrammarBackend: caching, dispatch routing, unsupported fallback,
  thread pool execution, cache hit/miss
- create_grammar_backend: factory routing, "none" backend, invalid name,
  custom registry, reasoner wrapping
- register_grammar_backend: registration and lookup

Usage:
    python -m pytest test_base_grammar_backend.py -v
"""

import unittest
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

from sglang.srt.constrained.base_grammar_backend import (
    GRAMMAR_BACKEND_REGISTRY,
    BaseGrammarBackend,
    BaseGrammarObject,
    GrammarStats,
    InvalidGrammarObject,
    create_grammar_backend,
    register_grammar_backend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(2.0, "stage-a-test-cpu")


class TestGrammarStats(unittest.TestCase):
    """Test GrammarStats dataclass."""

    def test_defaults(self):
        stats = GrammarStats()
        self.assertIsNone(stats.compilation_time)
        self.assertIsNone(stats.schema_count)
        self.assertIsNone(stats.ebnf_size)
        self.assertFalse(stats.is_cache_hit)
        self.assertFalse(stats.is_grammar_aborted)
        self.assertEqual(stats.tree_traversal_time, [])
        self.assertIsNone(stats.dispatch_type)
        self.assertEqual(stats.num_timeout, 0)

    def test_tree_traversal_time_mutable_default(self):
        """Ensure each instance gets its own list."""
        s1 = GrammarStats()
        s2 = GrammarStats()
        s1.tree_traversal_time.append(0.1)
        self.assertEqual(len(s2.tree_traversal_time), 0)


class TestBaseGrammarObject(unittest.TestCase):
    """Test BaseGrammarObject base class."""

    def test_is_terminated_default(self):
        obj = BaseGrammarObject()
        self.assertFalse(obj.is_terminated())

    def test_maybe_init_reasoning_noop(self):
        obj = BaseGrammarObject()
        obj.maybe_init_reasoning(True)  # Should not raise


class TestInvalidGrammarObject(unittest.TestCase):
    """Test InvalidGrammarObject."""

    def test_default_error_message(self):
        obj = InvalidGrammarObject()
        self.assertEqual(obj.error_message, "Unknown grammar error")

    def test_custom_error_message(self):
        obj = InvalidGrammarObject("Regex compilation failed")
        self.assertEqual(obj.error_message, "Regex compilation failed")


class TestBaseGrammarBackend(unittest.TestCase):
    """Test BaseGrammarBackend caching and dispatch."""

    def setUp(self):
        self.backend = BaseGrammarBackend()

    def tearDown(self):
        self.backend.executor.shutdown(wait=True)

    def test_set_and_get_cache(self):
        obj = BaseGrammarObject()
        key = ("json", '{"type": "object"}')
        self.backend.set_cache(key, obj)
        self.assertIn(key, self.backend.cache)
        self.assertIs(self.backend.cache[key], obj)

    def test_reset_clears_cache(self):
        self.backend.set_cache(("json", "schema"), BaseGrammarObject())
        self.backend.reset()
        self.assertEqual(len(self.backend.cache), 0)

    def test_cache_hit_returns_copy(self):
        """Cache hit should return a copy of the cached object."""
        mock_copy = BaseGrammarObject()
        obj = MagicMock(spec=BaseGrammarObject)
        obj.copy.return_value = mock_copy

        key = ("json", "schema")
        self.backend.set_cache(key, obj)
        result, cache_hit = self.backend.get_cached_or_future_value(key, False)

        self.assertTrue(cache_hit)
        obj.copy.assert_called_once()
        self.assertIs(result, mock_copy)

    def test_cache_hit_inits_reasoning(self):
        obj = MagicMock(spec=BaseGrammarObject)
        copied = MagicMock(spec=BaseGrammarObject)
        obj.copy.return_value = copied

        key = ("json", "schema")
        self.backend.set_cache(key, obj)
        self.backend.get_cached_or_future_value(key, True)
        copied.maybe_init_reasoning.assert_called_once_with(True)

    def test_cache_miss_returns_future(self):
        key = ("json", "schema")
        result, cache_hit = self.backend.get_cached_or_future_value(key, False)
        self.assertFalse(cache_hit)
        self.assertIsInstance(result, Future)
        # The future should complete (dispatch_json returns InvalidGrammarObject)
        value = result.result(timeout=5)
        self.assertIsInstance(value, InvalidGrammarObject)

    def test_all_dispatch_methods_unsupported(self):
        """All dispatch methods on base class return InvalidGrammarObject."""
        cases = [
            ("dispatch_json", ("schema",)),
            ("dispatch_regex", ("[a-z]+",)),
            ("dispatch_ebnf", ("root ::= 'hello'",)),
            ("dispatch_structural_tag", ("{}",)),
        ]
        for method_name, args in cases:
            with self.subTest(method=method_name):
                result = getattr(self.backend, method_name)(*args)
                self.assertIsInstance(result, InvalidGrammarObject)

    def test_dispatch_fallback_raises(self):
        with self.assertRaises(ValueError):
            self.backend.dispatch_fallback("unknown", "value")

    def test_init_value_dispatch_routes_all_types(self):
        """_init_value_dispatch routes all grammar types to their dispatch methods."""
        cases = [
            ("json", "schema"),
            ("regex", "[a-z]+"),
            ("ebnf", "root ::= 'x'"),
            ("structural_tag", "{}"),
        ]
        for grammar_type, value in cases:
            with self.subTest(grammar_type=grammar_type):
                result = self.backend._init_value_dispatch((grammar_type, value), False)
                self.assertIsInstance(result, InvalidGrammarObject)

    def test_init_value_dispatch_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            self.backend._init_value_dispatch(("unknown_type", "value"), False)

    def test_init_value_dispatch_sets_compilation_time(self):
        """When grammar has stats, compilation_time should be set."""
        mock_grammar = MagicMock(spec=BaseGrammarObject)
        mock_grammar.grammar_stats = GrammarStats()
        self.backend.dispatch_json = MagicMock(return_value=mock_grammar)

        result = self.backend._init_value_dispatch(("json", "schema"), False)
        self.assertIsNotNone(result.grammar_stats.compilation_time)
        self.assertGreater(result.grammar_stats.compilation_time, 0)

    def test_init_value_dispatch_no_stats(self):
        """When grammar has no stats, should not crash."""
        mock_grammar = MagicMock(spec=BaseGrammarObject)
        mock_grammar.grammar_stats = None
        self.backend.dispatch_json = MagicMock(return_value=mock_grammar)
        # Should not raise
        self.backend._init_value_dispatch(("json", "schema"), False)

    def test_reset_then_miss(self):
        """After reset, previously cached keys should be misses."""
        key = ("json", "schema")
        obj = MagicMock(spec=BaseGrammarObject)
        obj.copy.return_value = obj
        self.backend.set_cache(key, obj)

        _, hit = self.backend.get_cached_or_future_value(key, False)
        self.assertTrue(hit)

        self.backend.reset()
        result, hit = self.backend.get_cached_or_future_value(key, False)
        self.assertFalse(hit)
        self.assertIsInstance(result, Future)

    def test_dispatch_fallback_error_message_content(self):
        """dispatch_fallback error should include the key type and value."""
        with self.assertRaises(ValueError) as ctx:
            self.backend.dispatch_fallback("custom_type", "custom_value")
        self.assertIn("custom_type", str(ctx.exception))
        self.assertIn("custom_value", str(ctx.exception))

    def test_init_value_dispatch_none_grammar(self):
        """When dispatch returns None, should not crash on stats check."""
        self.backend.dispatch_json = MagicMock(return_value=None)
        result = self.backend._init_value_dispatch(("json", "schema"), False)
        self.assertIsNone(result)

    def test_cache_miss_duplicate_key_submits_separate_futures(self):
        """Two cache misses for the same key each get their own Future.

        The backend does not deduplicate in-flight compilations — that is
        handled at the GrammarManager level via grammar_queue. Each call
        to get_cached_or_future_value with an uncached key submits a new
        task to the executor."""
        key = ("json", "schema")
        result1, hit1 = self.backend.get_cached_or_future_value(key, False)
        result2, hit2 = self.backend.get_cached_or_future_value(key, False)

        self.assertFalse(hit1)
        self.assertFalse(hit2)
        self.assertIsInstance(result1, Future)
        self.assertIsInstance(result2, Future)
        # They are independent futures, not shared
        self.assertIsNot(result1, result2)

        # Both should complete successfully
        self.assertIsInstance(result1.result(timeout=5), InvalidGrammarObject)
        self.assertIsInstance(result2.result(timeout=5), InvalidGrammarObject)


class TestRegisterGrammarBackend(unittest.TestCase):
    """Test grammar backend registry."""

    def setUp(self):
        self._saved = dict(GRAMMAR_BACKEND_REGISTRY)

    def tearDown(self):
        GRAMMAR_BACKEND_REGISTRY.clear()
        GRAMMAR_BACKEND_REGISTRY.update(self._saved)

    def test_register_and_use(self):
        mock_init = MagicMock(return_value="custom_backend")
        register_grammar_backend("my_backend", mock_init)
        self.assertIn("my_backend", GRAMMAR_BACKEND_REGISTRY)

    def test_overwrite_registration(self):
        register_grammar_backend("dup", lambda *a: "first")
        register_grammar_backend("dup", lambda *a: "second")
        self.assertEqual(
            GRAMMAR_BACKEND_REGISTRY["dup"](None, None, None, None), "second"
        )


class TestCreateGrammarBackend(unittest.TestCase):
    """Test create_grammar_backend factory function."""

    def setUp(self):
        self._saved = dict(GRAMMAR_BACKEND_REGISTRY)

    def tearDown(self):
        GRAMMAR_BACKEND_REGISTRY.clear()
        GRAMMAR_BACKEND_REGISTRY.update(self._saved)

    def _make_server_args(self, backend="none", reasoning_parser=None):
        args = MagicMock()
        args.grammar_backend = backend
        args.reasoning_parser = reasoning_parser
        args.constrained_json_whitespace_pattern = None
        args.constrained_json_disable_any_whitespace = False
        return args

    def test_none_backend_returns_none(self):
        args = self._make_server_args("none")
        result = create_grammar_backend(args, None, 32000)
        self.assertIsNone(result)

    def test_invalid_backend_raises(self):
        args = self._make_server_args("nonexistent_backend")
        with self.assertRaises(ValueError):
            create_grammar_backend(args, None, 32000)

    def test_custom_registered_backend(self):
        mock_backend = MagicMock()
        register_grammar_backend("test_custom", lambda *a: mock_backend)
        args = self._make_server_args("test_custom")
        result = create_grammar_backend(args, "tok", 32000, {1, 2})
        self.assertIs(result, mock_backend)

    def test_custom_backend_receives_args(self):
        received = {}

        def init_fn(server_args, tokenizer, vocab_size, eos_token_ids):
            received["server_args"] = server_args
            received["tokenizer"] = tokenizer
            received["vocab_size"] = vocab_size
            received["eos_token_ids"] = eos_token_ids
            return MagicMock()

        register_grammar_backend("capture", init_fn)
        args = self._make_server_args("capture")
        create_grammar_backend(args, "my_tok", 50000, {0, 2})
        self.assertEqual(received["tokenizer"], "my_tok")
        self.assertEqual(received["vocab_size"], 50000)
        self.assertEqual(received["eos_token_ids"], {0, 2})

    def test_custom_backend_skips_reasoner_wrapping(self):
        """Custom registered backends return directly, bypassing reasoner wrapping."""
        mock_inner = MagicMock(spec=BaseGrammarBackend)
        register_grammar_backend("inner_r", lambda *a: mock_inner)

        args = self._make_server_args("inner_r", reasoning_parser="deepseek")
        tokenizer = MagicMock()
        tokenizer.think_end_id = 42

        result = create_grammar_backend(args, tokenizer, 32000)
        # Custom backends return early, no reasoner wrapping applied
        self.assertIs(result, mock_inner)

    @patch("sglang.srt.constrained.outlines_backend.OutlinesGrammarBackend")
    def test_outlines_backend(self, mock_outlines_cls):
        mock_backend = MagicMock(spec=BaseGrammarBackend)
        mock_outlines_cls.return_value = mock_backend
        args = self._make_server_args("outlines")
        args.constrained_json_whitespace_pattern = r"\s*"

        result = create_grammar_backend(args, "tok", 32000)
        mock_outlines_cls.assert_called_once_with("tok", whitespace_pattern=r"\s*")
        self.assertIs(result, mock_backend)

    @patch("sglang.srt.constrained.xgrammar_backend.XGrammarGrammarBackend")
    def test_xgrammar_backend(self, mock_xgrammar_cls):
        mock_backend = MagicMock(spec=BaseGrammarBackend)
        mock_xgrammar_cls.return_value = mock_backend
        args = self._make_server_args("xgrammar")
        args.constrained_json_disable_any_whitespace = True

        result = create_grammar_backend(args, "tok", 32000, {1, 2})
        mock_xgrammar_cls.assert_called_once_with(
            "tok", vocab_size=32000, model_eos_token_ids=[1, 2], any_whitespace=False
        )
        self.assertIs(result, mock_backend)

    @patch("sglang.srt.constrained.xgrammar_backend.XGrammarGrammarBackend")
    def test_xgrammar_unsupported_tokenizer_falls_back_to_none(self, mock_xgrammar_cls):
        from sglang.srt.constrained.xgrammar_backend import TokenizerNotSupportedError

        mock_xgrammar_cls.side_effect = TokenizerNotSupportedError(
            "unsupported tokenizer"
        )
        args = self._make_server_args("xgrammar")

        result = create_grammar_backend(args, "tok", 32000, {1})
        self.assertIsNone(result)
        self.assertEqual(args.grammar_backend, "none")

    @patch("sglang.srt.constrained.llguidance_backend.GuidanceBackend")
    def test_llguidance_backend(self, mock_guidance_cls):
        mock_backend = MagicMock(spec=BaseGrammarBackend)
        mock_guidance_cls.return_value = mock_backend
        args = self._make_server_args("llguidance")
        args.constrained_json_disable_any_whitespace = False
        args.constrained_json_whitespace_pattern = r"\s+"

        result = create_grammar_backend(args, "tok", 32000)
        mock_guidance_cls.assert_called_once_with(
            tokenizer="tok", any_whitespace=True, whitespace_pattern=r"\s+"
        )
        self.assertIs(result, mock_backend)

    @patch("sglang.srt.constrained.outlines_backend.OutlinesGrammarBackend")
    def test_reasoner_wrapping_on_builtin_backend(self, mock_outlines_cls):
        """Non-custom backends get wrapped with ReasonerGrammarBackend."""
        from sglang.srt.constrained.reasoner_grammar_backend import (
            ReasonerGrammarBackend,
        )

        mock_backend = MagicMock(spec=BaseGrammarBackend)
        mock_outlines_cls.return_value = mock_backend
        args = self._make_server_args("outlines", reasoning_parser="deepseek")
        tokenizer = MagicMock()
        tokenizer.think_end_id = 42

        result = create_grammar_backend(args, tokenizer, 32000)
        self.assertIsInstance(result, ReasonerGrammarBackend)
        self.assertEqual(result.think_end_id, 42)
        self.assertIs(result.grammar_backend, mock_backend)

    @patch("sglang.srt.constrained.outlines_backend.OutlinesGrammarBackend")
    def test_no_reasoner_wrapping_without_think_end_id(self, mock_outlines_cls):
        """Without think_end_id on tokenizer, no reasoner wrapping."""
        mock_backend = MagicMock(spec=BaseGrammarBackend)
        mock_outlines_cls.return_value = mock_backend
        args = self._make_server_args("outlines", reasoning_parser="deepseek")
        tokenizer = MagicMock(spec=[])  # No think_end_id attribute

        result = create_grammar_backend(args, tokenizer, 32000)
        self.assertIs(result, mock_backend)

    @patch("sglang.srt.constrained.outlines_backend.OutlinesGrammarBackend")
    def test_no_reasoner_wrapping_without_reasoning_parser(self, mock_outlines_cls):
        """Without reasoning_parser, no reasoner wrapping even with think_end_id."""
        mock_backend = MagicMock(spec=BaseGrammarBackend)
        mock_outlines_cls.return_value = mock_backend
        args = self._make_server_args("outlines", reasoning_parser=None)
        tokenizer = MagicMock()
        tokenizer.think_end_id = 42

        result = create_grammar_backend(args, tokenizer, 32000)
        self.assertIs(result, mock_backend)

    @patch("sglang.srt.constrained.xgrammar_backend.XGrammarGrammarBackend")
    def test_xgrammar_eos_none(self, mock_xgrammar_cls):
        """eos_token_ids=None should pass None, not an empty list."""
        mock_xgrammar_cls.return_value = MagicMock(spec=BaseGrammarBackend)
        args = self._make_server_args("xgrammar")

        create_grammar_backend(args, "tok", 32000, None)
        _, kwargs = mock_xgrammar_cls.call_args
        self.assertIsNone(kwargs["model_eos_token_ids"])


if __name__ == "__main__":
    unittest.main()
