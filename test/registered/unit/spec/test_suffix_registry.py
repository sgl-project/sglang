"""Unit tests for the SUFFIX speculative-decoding plugin registration.

CPU-only: exercises the plugin wiring (registration, duck-typed predicates,
arg validation, lazy factory) without loading a model or arctic_inference.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.speculative import sgl_suffix_plugin
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import (
    _REGISTRY,
    CustomSpecAlgo,
    _assert_custom_spec_algo_conforms,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _SuffixRegistered(CustomTestCase):
    """Re-register SUFFIX into an isolated registry so tests don't leak."""

    def setUp(self):
        self._snapshot = _REGISTRY.copy()
        _REGISTRY.clear()
        SpeculativeAlgorithm.register(
            "SUFFIX",
            supports_overlap=True,
            validate_server_args=sgl_suffix_plugin._validate_suffix_args,
            spec_class=sgl_suffix_plugin._SuffixLike,
        )(sgl_suffix_plugin._suffix_factory)
        self.algo = SpeculativeAlgorithm.from_string("SUFFIX")

    def tearDown(self):
        _REGISTRY.clear()
        _REGISTRY.update(self._snapshot)


class TestSuffixRegistration(_SuffixRegistered):
    def test_from_string_returns_custom_spec(self):
        self.assertIsInstance(self.algo, CustomSpecAlgo)
        self.assertEqual(self.algo.name, "SUFFIX")

    def test_case_insensitive(self):
        self.assertIs(
            SpeculativeAlgorithm.from_string("suffix"),
            SpeculativeAlgorithm.from_string("SUFFIX"),
        )

    def test_dispatches_like_ngram(self):
        # SUFFIX reuses NGRAM's verify / KV-cache dispatch.
        self.assertTrue(self.algo.is_ngram())
        self.assertTrue(self.algo.is_suffix())

    def test_other_predicates_false(self):
        self.assertFalse(self.algo.is_eagle())
        self.assertFalse(self.algo.is_eagle3())
        self.assertFalse(self.algo.is_standalone())
        self.assertFalse(self.algo.is_none())
        self.assertTrue(self.algo.is_speculative())

    def test_supports_overlap_spec_v2(self):
        # The worker rides NGRAM's spec-v2-native machinery, so overlap
        # scheduling is supported (and supports_spec_v2 follows it).
        self.assertTrue(self.algo.supports_overlap)
        self.assertTrue(self.algo.supports_spec_v2())

    def test_not_equal_to_builtin(self):
        self.assertNotEqual(self.algo, SpeculativeAlgorithm.NGRAM)
        self.assertNotEqual(self.algo, SpeculativeAlgorithm.EAGLE)


class TestSuffixConformance(_SuffixRegistered):
    def test_spec_class_conforms(self):
        # _SuffixLike must satisfy the enum's is_*/supports_* duck-type contract.
        _assert_custom_spec_algo_conforms(sgl_suffix_plugin._SuffixLike)

    def test_provides_create_future_map(self):
        # The scheduler calls create_future_map() unconditionally; CustomSpecAlgo's
        # base does not define it, so _SuffixLike must (regression guard).
        self.assertTrue(callable(getattr(self.algo, "create_future_map", None)))


class TestSuffixFactory(_SuffixRegistered):
    def test_factory_returns_suffix_worker(self):
        # Lazy: importing SuffixWorker does not require arctic_inference (that is
        # imported only when the worker is constructed).
        from sglang.srt.speculative.suffix_worker import SuffixWorker

        for disable_overlap in (True, False):
            server_args = SimpleNamespace(disable_overlap_schedule=disable_overlap)
            self.assertIs(self.algo.create_worker(server_args), SuffixWorker)


class TestSuffixArgValidation(_SuffixRegistered):
    _FIELDS = (
        "speculative_suffix_max_tree_depth",
        "speculative_suffix_max_cached_requests",
        "speculative_suffix_max_spec_factor",
        "speculative_suffix_min_token_prob",
    )

    def test_validator_passes_with_all_fields(self):
        sa = SimpleNamespace(**{f: 1 for f in self._FIELDS})
        self.algo.validate_server_args(sa)  # does not raise

    def test_validator_raises_when_field_missing(self):
        sa = SimpleNamespace(**{f: 1 for f in self._FIELDS[1:]})  # drop the first
        with self.assertRaisesRegex(ValueError, "speculative_suffix_max_tree_depth"):
            self.algo.validate_server_args(sa)


if __name__ == "__main__":
    unittest.main(verbosity=3)
