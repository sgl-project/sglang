"""Regression tests for the requests the one-batch benchmark builds."""

import unittest

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.benchmark.one_batch import (
    BenchArgs,
    prepare_inputs_for_correctness_test,
    prepare_synthetic_inputs_for_latency_test,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _Tokenizer:
    def encode(self, text):
        return list(range(1, len(text.split()) + 1))


class TestOneBatchRequestTokenBudget(CustomTestCase):
    """Benchmark requests must carry the run's output length as an integer
    max_new_tokens, so the length check the scheduler runs on a Req works."""

    def test_latency_requests_carry_the_swept_output_len(self):
        (req,) = prepare_synthetic_inputs_for_latency_test(
            batch_size=1, input_len=4, output_len=2
        )
        self.assertEqual(req.sampling_params.max_new_tokens, 2)
        self.assertFalse(req.next_output_finishes_by_length())
        req.output_ids.append(7)
        self.assertTrue(req.next_output_finishes_by_length())

    def test_latency_requests_default_to_an_integer_budget(self):
        (req,) = prepare_synthetic_inputs_for_latency_test(batch_size=1, input_len=4)
        self.assertFalse(req.next_output_finishes_by_length())

    def test_correctness_requests_carry_the_run_output_len(self):
        bench_args = BenchArgs(output_len=(3,), cut_len=1)
        _, reqs = prepare_inputs_for_correctness_test(bench_args, _Tokenizer(), [])
        self.assertEqual({req.sampling_params.max_new_tokens for req in reqs}, {3})


if __name__ == "__main__":
    unittest.main()
