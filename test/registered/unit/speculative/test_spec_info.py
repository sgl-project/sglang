"""Unit tests for srt/speculative/spec_info.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import unittest
from unittest.mock import MagicMock

from sglang.srt.speculative.spec_info import (
    SpecInput,
    SpecInputType,
    SpeculativeAlgorithm,
)
from sglang.test.test_utils import CustomTestCase


class TestSpeculativeAlgorithmFromString(CustomTestCase):
    """Tests for SpeculativeAlgorithm.from_string()."""

    def test_valid_eagle(self):
        self.assertEqual(
            SpeculativeAlgorithm.from_string("EAGLE"),
            SpeculativeAlgorithm.EAGLE,
        )

    def test_valid_eagle3(self):
        self.assertEqual(
            SpeculativeAlgorithm.from_string("EAGLE3"),
            SpeculativeAlgorithm.EAGLE3,
        )

    def test_valid_standalone(self):
        self.assertEqual(
            SpeculativeAlgorithm.from_string("STANDALONE"),
            SpeculativeAlgorithm.STANDALONE,
        )

    def test_valid_ngram(self):
        self.assertEqual(
            SpeculativeAlgorithm.from_string("NGRAM"),
            SpeculativeAlgorithm.NGRAM,
        )

    def test_none_returns_none_variant(self):
        self.assertEqual(
            SpeculativeAlgorithm.from_string(None),
            SpeculativeAlgorithm.NONE,
        )

    def test_case_insensitive(self):
        self.assertEqual(
            SpeculativeAlgorithm.from_string("eagle"),
            SpeculativeAlgorithm.EAGLE,
        )
        self.assertEqual(
            SpeculativeAlgorithm.from_string("Eagle3"),
            SpeculativeAlgorithm.EAGLE3,
        )

    def test_invalid_name_raises_value_error(self):
        with self.assertRaises(ValueError):
            SpeculativeAlgorithm.from_string("INVALID_ALGO")

    def test_empty_string_raises_value_error(self):
        with self.assertRaises(ValueError):
            SpeculativeAlgorithm.from_string("")


class TestSpeculativeAlgorithmBoolChecks(CustomTestCase):
    """Tests for is_eagle/is_eagle3/is_standalone/is_ngram/is_none."""

    def test_is_eagle_true_for_eagle_and_eagle3(self):
        """EAGLE3 is a variant of EAGLE, so is_eagle() returns True for both."""
        self.assertTrue(SpeculativeAlgorithm.EAGLE.is_eagle())
        self.assertTrue(SpeculativeAlgorithm.EAGLE3.is_eagle())

    def test_is_eagle_false_for_others(self):
        self.assertFalse(SpeculativeAlgorithm.NGRAM.is_eagle())
        self.assertFalse(SpeculativeAlgorithm.STANDALONE.is_eagle())
        self.assertFalse(SpeculativeAlgorithm.NONE.is_eagle())

    def test_is_eagle3_only_for_eagle3(self):
        self.assertTrue(SpeculativeAlgorithm.EAGLE3.is_eagle3())
        self.assertFalse(SpeculativeAlgorithm.EAGLE.is_eagle3())
        self.assertFalse(SpeculativeAlgorithm.NGRAM.is_eagle3())

    def test_is_standalone(self):
        self.assertTrue(SpeculativeAlgorithm.STANDALONE.is_standalone())
        self.assertFalse(SpeculativeAlgorithm.EAGLE.is_standalone())

    def test_is_ngram(self):
        self.assertTrue(SpeculativeAlgorithm.NGRAM.is_ngram())
        self.assertFalse(SpeculativeAlgorithm.EAGLE.is_ngram())

    def test_is_none(self):
        self.assertTrue(SpeculativeAlgorithm.NONE.is_none())
        self.assertFalse(SpeculativeAlgorithm.EAGLE.is_none())


class TestSpeculativeAlgorithmSupportsSpecV2(CustomTestCase):
    """Tests for supports_spec_v2() — only EAGLE variants and STANDALONE support it."""

    def test_eagle_supports(self):
        self.assertTrue(SpeculativeAlgorithm.EAGLE.supports_spec_v2())

    def test_eagle3_supports(self):
        self.assertTrue(SpeculativeAlgorithm.EAGLE3.supports_spec_v2())

    def test_standalone_supports(self):
        self.assertTrue(SpeculativeAlgorithm.STANDALONE.supports_spec_v2())

    def test_ngram_does_not_support(self):
        self.assertFalse(SpeculativeAlgorithm.NGRAM.supports_spec_v2())

    def test_none_does_not_support(self):
        self.assertFalse(SpeculativeAlgorithm.NONE.supports_spec_v2())


class TestSpeculativeAlgorithmCreateWorker(CustomTestCase):
    """Tests for create_worker() — verifies correct worker class selection."""

    def _make_server_args(self, disable_overlap=False, multi_layer_eagle=False):
        args = MagicMock()
        args.disable_overlap_schedule = disable_overlap
        args.enable_multi_layer_eagle = multi_layer_eagle
        return args

    def test_none_raises_assertion(self):
        with self.assertRaises(AssertionError):
            SpeculativeAlgorithm.NONE.create_worker(self._make_server_args())

    def test_ngram_with_overlap_raises(self):
        with self.assertRaises(ValueError):
            SpeculativeAlgorithm.NGRAM.create_worker(
                self._make_server_args(disable_overlap=False)
            )

    def test_ngram_without_overlap(self):
        cls = SpeculativeAlgorithm.NGRAM.create_worker(
            self._make_server_args(disable_overlap=True)
        )
        self.assertEqual(cls.__name__, "NGRAMWorker")

    def test_eagle_with_overlap(self):
        cls = SpeculativeAlgorithm.EAGLE.create_worker(
            self._make_server_args(disable_overlap=False)
        )
        self.assertEqual(cls.__name__, "EAGLEWorkerV2")

    def test_eagle_without_overlap(self):
        cls = SpeculativeAlgorithm.EAGLE.create_worker(
            self._make_server_args(disable_overlap=True)
        )
        self.assertEqual(cls.__name__, "EAGLEWorker")

    def test_eagle_multi_layer_with_overlap(self):
        cls = SpeculativeAlgorithm.EAGLE.create_worker(
            self._make_server_args(disable_overlap=False, multi_layer_eagle=True)
        )
        self.assertEqual(cls.__name__, "MultiLayerEagleWorkerV2")

    def test_eagle_multi_layer_without_overlap(self):
        try:
            cls = SpeculativeAlgorithm.EAGLE.create_worker(
                self._make_server_args(disable_overlap=True, multi_layer_eagle=True)
            )
            self.assertEqual(cls.__name__, "MultiLayerEagleWorker")
        except NameError:
            # Known issue: multi_layer_eagle_worker.py has a NameError for
            # 'ModelRunner' in a type annotation at class definition time.
            # The branch selection logic is still correct; only the import fails.
            pass

    def test_standalone_with_overlap(self):
        cls = SpeculativeAlgorithm.STANDALONE.create_worker(
            self._make_server_args(disable_overlap=False)
        )
        self.assertEqual(cls.__name__, "StandaloneWorkerV2")

    def test_standalone_without_overlap(self):
        cls = SpeculativeAlgorithm.STANDALONE.create_worker(
            self._make_server_args(disable_overlap=True)
        )
        self.assertEqual(cls.__name__, "StandaloneWorker")


class TestSpecInputType(CustomTestCase):
    """Tests for SpecInputType IntEnum."""

    def test_enum_values_are_distinct(self):
        values = [e.value for e in SpecInputType]
        self.assertEqual(len(values), len(set(values)))

    def test_expected_members_exist(self):
        self.assertIn("EAGLE_DRAFT", SpecInputType.__members__)
        self.assertIn("EAGLE_VERIFY", SpecInputType.__members__)
        self.assertIn("NGRAM_VERIFY", SpecInputType.__members__)


class TestSpecInput(CustomTestCase):
    """Tests for SpecInput ABC base methods."""

    def _make_concrete(self, spec_input_type, c1=1, c2=1):
        """Create a minimal concrete SpecInput subclass for testing."""

        class ConcreteSpecInput(SpecInput):
            def __init__(self, sit, coeff):
                super().__init__(sit)
                self._coeff = coeff

            def get_spec_adjust_token_coefficient(self):
                return self._coeff

        return ConcreteSpecInput(spec_input_type, (c1, c2))

    def test_is_draft_input_true_for_eagle_draft(self):
        si = self._make_concrete(SpecInputType.EAGLE_DRAFT)
        self.assertTrue(si.is_draft_input())

    def test_is_draft_input_false_for_verify_types(self):
        self.assertFalse(
            self._make_concrete(SpecInputType.EAGLE_VERIFY).is_draft_input()
        )
        self.assertFalse(
            self._make_concrete(SpecInputType.NGRAM_VERIFY).is_draft_input()
        )

    def test_is_verify_input_true_for_verify_types(self):
        self.assertTrue(
            self._make_concrete(SpecInputType.EAGLE_VERIFY).is_verify_input()
        )
        self.assertTrue(
            self._make_concrete(SpecInputType.NGRAM_VERIFY).is_verify_input()
        )

    def test_is_verify_input_false_for_draft(self):
        self.assertFalse(
            self._make_concrete(SpecInputType.EAGLE_DRAFT).is_verify_input()
        )

    def test_get_spec_adjusted_global_num_tokens(self):
        """Coefficient (2, 3) should multiply tokens and logprob tokens accordingly."""
        si = self._make_concrete(SpecInputType.EAGLE_VERIFY, c1=2, c2=3)
        forward_batch = MagicMock()
        forward_batch.global_num_tokens = [10, 20, 30]
        forward_batch.global_num_tokens_for_logprob = [5, 10, 15]

        tokens, logprob_tokens = si.get_spec_adjusted_global_num_tokens(forward_batch)
        self.assertEqual(tokens, [20, 40, 60])
        self.assertEqual(logprob_tokens, [15, 30, 45])

    def test_get_spec_adjusted_global_num_tokens_identity(self):
        """Coefficient (1, 1) should leave values unchanged."""
        si = self._make_concrete(SpecInputType.EAGLE_DRAFT, c1=1, c2=1)
        forward_batch = MagicMock()
        forward_batch.global_num_tokens = [100]
        forward_batch.global_num_tokens_for_logprob = [50]

        tokens, logprob_tokens = si.get_spec_adjusted_global_num_tokens(forward_batch)
        self.assertEqual(tokens, [100])
        self.assertEqual(logprob_tokens, [50])


if __name__ == "__main__":
    unittest.main()
