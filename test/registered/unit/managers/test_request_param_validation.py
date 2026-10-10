"""Tests for request parameter validation guards.

Covers the bounds introduced for request-level DoS hardening:
- ``n`` (parallel sample num) is capped before expansion.
- ``top_logprobs_num`` is bounded by the vocabulary size.
- ``input_ids`` must lie in [0, vocab_size).
"""

import asyncio
import unittest
from types import SimpleNamespace

from sglang.srt.managers.io_struct import (
    MAX_PARALLEL_SAMPLE_NUM,
    EmbeddingReqInput,
    GenerateReqInput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_tokenizer_manager(vocab_size: int = 100) -> TokenizerManager:
    """A bare TokenizerManager instance with a stubbed model config.

    Only the fields used by the validation helpers under test are populated.
    """
    tm = object.__new__(TokenizerManager)
    tm.model_config = SimpleNamespace(vocab_size=vocab_size)
    tm.context_len = 128
    tm.num_reserved_tokens = 0
    tm.allow_auto_truncate = False
    tm.validate_total_tokens = False
    tm.is_generation = True
    return tm


class TestParallelSampleNumBound(CustomTestCase):
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
        for n in ("1024", 1.5, True, False):
            with self.subTest(n=n):
                req = GenerateReqInput(text="hi", sampling_params={"n": n})
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    req.normalize_batch_and_arguments()

    def test_beam_search_does_not_hide_invalid_n(self):
        """Beam normalization must not discard invalid caller-supplied n."""
        for n in (0, -1, True, "2", MAX_PARALLEL_SAMPLE_NUM + 1):
            with self.subTest(n=n):
                req = GenerateReqInput(
                    text="hi", sampling_params={"n": n, "beam_width": 4}
                )
                with self.assertRaisesRegex(ValueError, "parallel sample num"):
                    req.normalize_batch_and_arguments()

    def test_beam_search_does_not_expand_return_sequences(self):
        req = GenerateReqInput(text="hi", sampling_params={"n": 2, "beam_width": 4})
        req.normalize_batch_and_arguments()
        self.assertTrue(req.is_single)
        self.assertEqual(req.parallel_sample_num, 1)
        self.assertEqual(req.sampling_params["n"], 2)

    def test_parallel_sampling_limit_is_inclusive(self):
        req = GenerateReqInput(
            text="hi", sampling_params={"n": MAX_PARALLEL_SAMPLE_NUM}
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.parallel_sample_num, MAX_PARALLEL_SAMPLE_NUM)
        self.assertEqual(len(req.rid), MAX_PARALLEL_SAMPLE_NUM)
        req = GenerateReqInput(
            text="hi", sampling_params={"n": MAX_PARALLEL_SAMPLE_NUM + 1}
        )
        with self.assertRaisesRegex(ValueError, "parallel sample num"):
            req.normalize_batch_and_arguments()


class TestTopLogprobsNumValidation(CustomTestCase):
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

    def test_normalized_batch_rejects_invalid_top_logprobs(self):
        """Per-request validation must remain wired after batch normalization."""
        req = GenerateReqInput(text=["first", "second"], top_logprobs_num=[0, 101])
        req.normalize_batch_and_arguments()
        with get_context().override_server_args():
            self.tm._validate_one_request(req[0], [1])
            with self.assertRaisesRegex(ValueError, "top_logprobs_num"):
                self.tm._validate_one_request(req[1], [1])

    def test_valid_top_logprobs_num_accepted(self):
        for value in (0, 10, 100, None):
            with self.subTest(value=value):
                req = GenerateReqInput(
                    text="hi", return_logprob=True, top_logprobs_num=value
                )
                self.tm._validate_top_logprobs_num(req)


class TestInputIdsInVocabValidation(CustomTestCase):
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

    def test_non_integer_input_ids_rejected(self):
        for bad in (1.5, "1", None):
            with self.subTest(token_id=bad):
                with self.assertRaisesRegex(ValueError, "outside the vocab range"):
                    self.tm._validate_input_ids_in_vocab([bad], vocab_size=100)

    def test_raw_ids_rejected_before_tokenization(self):
        """Generation and embedding must reject IDs before model processing."""
        for request_type in (GenerateReqInput, EmbeddingReqInput):
            with self.subTest(request_type=request_type):
                req = request_type(input_ids=[-1], sampling_params={})
                with self.assertRaisesRegex(ValueError, "outside the vocab range"):
                    asyncio.run(self.tm._tokenize_one_request(req))

    def test_batch_raw_ids_rejected_before_tokenization(self):
        """The pre-tokenized batch path must apply the same raw-ID guard."""
        req = GenerateReqInput(input_ids=[[-1], [1]], sampling_params={})
        req.normalize_batch_and_arguments()
        with self.assertRaisesRegex(ValueError, "outside the vocab range"):
            asyncio.run(self.tm._batch_tokenize_and_process(req.batch_size, req))

    def test_multimodal_padding_ids_not_rejected_as_user_input(self):
        """Processors can replace valid input IDs with out-of-vocab padding."""
        req = GenerateReqInput(input_ids=[1], sampling_params={})
        with get_context().override_server_args():
            self.tm._validate_one_request(req, [1, 1_000_000])

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
