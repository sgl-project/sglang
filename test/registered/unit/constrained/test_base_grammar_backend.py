"""
Unit tests for sglang.srt.constrained.base_grammar_backend.

Test Coverage:
- GrammarStats: default values, custom values
- BaseGrammarObject: finished property, default behavior, copy
- InvalidGrammarObject: error message, repr, inheritance
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
from unittest.mock import MagicMock

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

register_cpu_ci(2.0, "stage-a-cpu-only")


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

    def test_custom_values(self):
        stats = GrammarStats(
            compilation_time=1.5,
            schema_count=3,
            ebnf_size=100,
            is_cache_hit=True,
            dispatch_type="json",
            num_timeout=2,
        )
        self.assertEqual(stats.compilation_time, 1.5)
        self.assertEqual(stats.schema_count, 3)
        self.assertTrue(stats.is_cache_hit)
        self.assertEqual(stats.dispatch_type, "json")

    def test_tree_traversal_time_mutable_default(self):
        """Ensure each instance gets its own list."""
        s1 = GrammarStats()
        s2 = GrammarStats()
        s1.tree_traversal_time.append(0.1)
        self.assertEqual(len(s2.tree_traversal_time), 0)


class TestBaseGrammarObject(unittest.TestCase):
    """Test BaseGrammarObject base class."""

    def test_initial_state(self):
        obj = BaseGrammarObject()
        self.assertFalse(obj.finished)
        self.assertIsNone(obj.grammar_stats)
        self.assertIsNone(obj.current_token)

    def test_finished_property(self):
        obj = BaseGrammarObject()
        obj.finished = True
        self.assertTrue(obj.finished)
        obj.finished = False
        self.assertFalse(obj.finished)

    def test_is_terminated_default(self):
        obj = BaseGrammarObject()
        self.assertFalse(obj.is_terminated())

    def test_copy_returns_self(self):
        obj = BaseGrammarObject()
        self.assertIs(obj.copy(), obj)

    def test_maybe_init_reasoning_noop(self):
        obj = BaseGrammarObject()
        obj.maybe_init_reasoning(True)  # Should not raise

    def test_abstract_methods_raise(self):
        obj = BaseGrammarObject()
        with self.assertRaises(NotImplementedError):
            obj.accept_token(0)
        with self.assertRaises(NotImplementedError):
            obj.rollback(1)
        with self.assertRaises(NotImplementedError):
            obj.allocate_vocab_mask(100, 1, "cpu")
        with self.assertRaises(NotImplementedError):
            obj.fill_vocab_mask(None, 0)
        with self.assertRaises(NotImplementedError):
            obj.try_jump_forward(None)
        with self.assertRaises(NotImplementedError):
            obj.jump_forward_str_state(None)
        with self.assertRaises(NotImplementedError):
            obj.jump_and_retokenize([], [], 0)

    def test_static_methods_raise(self):
        with self.assertRaises(NotImplementedError):
            BaseGrammarObject.move_vocab_mask(None, "cpu")
        with self.assertRaises(NotImplementedError):
            BaseGrammarObject.apply_vocab_mask(None, None)


class TestInvalidGrammarObject(unittest.TestCase):
    """Test InvalidGrammarObject."""

    def test_default_error_message(self):
        obj = InvalidGrammarObject()
        self.assertEqual(obj.error_message, "Unknown grammar error")

    def test_custom_error_message(self):
        obj = InvalidGrammarObject("Regex compilation failed")
        self.assertEqual(obj.error_message, "Regex compilation failed")

    def test_repr(self):
        obj = InvalidGrammarObject("test error")
        self.assertEqual(repr(obj), "InvalidGrammarObject(error_message='test error')")

    def test_is_base_grammar_object(self):
        obj = InvalidGrammarObject()
        self.assertIsInstance(obj, BaseGrammarObject)

    def test_inherits_finished_property(self):
        obj = InvalidGrammarObject("err")
        self.assertFalse(obj.finished)
        obj.finished = True
        self.assertTrue(obj.finished)


class TestBaseGrammarBackend(unittest.TestCase):
    """Test BaseGrammarBackend caching and dispatch."""

    def setUp(self):
        self.backend = BaseGrammarBackend()

    def tearDown(self):
        self.backend.executor.shutdown(wait=False)

    def test_initial_cache_empty(self):
        self.assertEqual(len(self.backend.cache), 0)

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

    def test_dispatch_json_unsupported(self):
        result = self.backend.dispatch_json("schema")
        self.assertIsInstance(result, InvalidGrammarObject)

    def test_dispatch_regex_unsupported(self):
        result = self.backend.dispatch_regex("[a-z]+")
        self.assertIsInstance(result, InvalidGrammarObject)

    def test_dispatch_ebnf_unsupported(self):
        result = self.backend.dispatch_ebnf("root ::= 'hello'")
        self.assertIsInstance(result, InvalidGrammarObject)

    def test_dispatch_structural_tag_unsupported(self):
        result = self.backend.dispatch_structural_tag("{}")
        self.assertIsInstance(result, InvalidGrammarObject)

    def test_dispatch_fallback_raises(self):
        with self.assertRaises(ValueError):
            self.backend.dispatch_fallback("unknown", "value")

    def test_init_value_dispatch_routes_json(self):
        result = self.backend._init_value_dispatch(("json", "schema"), False)
        self.assertIsInstance(result, InvalidGrammarObject)

    def test_init_value_dispatch_routes_regex(self):
        result = self.backend._init_value_dispatch(("regex", "[a-z]+"), False)
        self.assertIsInstance(result, InvalidGrammarObject)

    def test_init_value_dispatch_routes_ebnf(self):
        result = self.backend._init_value_dispatch(("ebnf", "root ::= 'x'"), False)
        self.assertIsInstance(result, InvalidGrammarObject)

    def test_init_value_dispatch_routes_structural_tag(self):
        result = self.backend._init_value_dispatch(("structural_tag", "{}"), False)
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

    def test_multiple_cache_keys(self):
        obj1 = BaseGrammarObject()
        obj2 = BaseGrammarObject()
        self.backend.set_cache(("json", "s1"), obj1)
        self.backend.set_cache(("regex", "r1"), obj2)
        self.assertEqual(len(self.backend.cache), 2)


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


if __name__ == "__main__":
    unittest.main()
