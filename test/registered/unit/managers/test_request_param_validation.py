"""Tests for request parameter validation guards.

Covers the bounds introduced for request-level DoS hardening:
- ``n`` (parallel sample num) is capped before expansion.
- ``top_logprobs_num`` is bounded by the vocabulary size.
- ``input_ids`` must lie in [0, vocab_size).
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.io_struct import (
    MAX_PARALLEL_SAMPLE_NUM,
    GenerateReqInput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_tokenizer_manager(vocab_size: int = 100) -> TokenizerManager:
    """A bare TokenizerManager instance with a stubbed model config.

    Only the fields used by the validation helpers under test are populated.
    """
    tm = object.__new__(TokenizerManager)
    tm.model_config = SimpleNamespace(vocab_size=vocab_size)
    return tm


class TestParallelSampleNumBound(unittest.TestCase):
    def test_huge_n_is_rejected_before_expansion(self):
        req = GenerateReqInput(text="hi", sampling_params={"n": 5_000_000})
        with self.assertRaisesRegex(
            ValueError,
            f"n \\(parallel sample num\\) must be in \\[1, {MAX_PARALLEL_SAMPLE_NUM}\\]",
        ):
            # Raises in _handle_parallel_sampling, i.e. before any per-sample
            # state is materialized.
            req.normalize_batch_and_arguments()

    def test_non_positive_n_is_rejected(self):
        for bad_n in (0, -1):
            req = GenerateReqInput(text="hi", sampling_params={"n": bad_n})
            with self.assertRaisesRegex(ValueError, "parallel sample num"):
                req.normalize_batch_and_arguments()

    def test_non_int_n_is_rejected(self):
        req = GenerateReqInput(text="hi", sampling_params={"n": "1024"})
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            req.normalize_batch_and_arguments()

    def test_legitimate_n_still_works(self):
        req = GenerateReqInput(text="hi", sampling_params={"n": 4, "max_new_tokens": 1})
        req.normalize_batch_and_arguments()
        self.assertEqual(req.parallel_sample_num, 4)


class TestTopLogprobsNumValidation(unittest.TestCase):
    def setUp(self):
        self.tm = _make_tokenizer_manager(vocab_size=100)

    def test_over_vocab_top_logprobs_num_rejected(self):
        req = GenerateReqInput(text="hi", return_logprob=True, top_logprobs_num=10**9)
        with self.assertRaisesRegex(
            ValueError, r"top_logprobs_num must be in \[0, 100\]"
        ):
            self.tm._validate_top_logprobs_num(req)

    def test_negative_top_logprobs_num_rejected(self):
        req = GenerateReqInput(text="hi", return_logprob=True, top_logprobs_num=-1)
        with self.assertRaisesRegex(
            ValueError, r"top_logprobs_num must be in \[0, 100\]"
        ):
            self.tm._validate_top_logprobs_num(req)

    def test_top_logprobs_num_list_validated_elementwise(self):
        req = GenerateReqInput(
            text="hi", return_logprob=True, top_logprobs_num=[1, 2, 10**9]
        )
        with self.assertRaisesRegex(
            ValueError, r"top_logprobs_num must be in \[0, 100\]"
        ):
            self.tm._validate_top_logprobs_num(req)

    def test_boolean_top_logprobs_num_rejected(self):
        # bool is an int subclass; JSON true/false must not pass as 1/0.
        for bad in (True, False, [1, True]):
            req = GenerateReqInput(text="hi", return_logprob=True, top_logprobs_num=bad)
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                self.tm._validate_top_logprobs_num(req)

    def test_valid_top_logprobs_num_accepted(self):
        req = GenerateReqInput(text="hi", return_logprob=True, top_logprobs_num=10)
        # Should not raise.
        self.tm._validate_top_logprobs_num(req)


class TestInputIdsInVocabValidation(unittest.TestCase):
    def setUp(self):
        self.tm = _make_tokenizer_manager(vocab_size=100)

    def test_negative_input_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, r"outside the vocab range"):
            self.tm._validate_input_ids_in_vocab([-1], vocab_size=100)

    def test_over_vocab_input_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, r"outside the vocab range"):
            self.tm._validate_input_ids_in_vocab([10**9], vocab_size=100)

    def test_nested_batch_input_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, r"outside the vocab range"):
            self.tm._validate_input_ids_in_vocab([[1, 2], [3, -5]], vocab_size=100)

    def test_boolean_input_ids_rejected(self):
        # bool is an int subclass; JSON true/false must not pass as 1/0.
        with self.assertRaisesRegex(ValueError, r"outside the vocab range"):
            self.tm._validate_input_ids_in_vocab([1, True], vocab_size=100)
        with self.assertRaisesRegex(ValueError, r"outside the vocab range"):
            self.tm._validate_input_ids_in_vocab([[False, 2]], vocab_size=100)

    def test_valid_input_ids_accepted(self):
        # Should not raise.
        self.tm._validate_input_ids_in_vocab([0, 50, 99], vocab_size=100)
        self.tm._validate_input_ids_in_vocab([[1, 2], [3, 4]], vocab_size=100)

    def test_empty_input_ids_accepted(self):
        # Should not raise.
        self.tm._validate_input_ids_in_vocab(None, vocab_size=100)
        self.tm._validate_input_ids_in_vocab([], vocab_size=100)


if __name__ == "__main__":
    unittest.main()
