"""
python3 -m unittest test_chunked_prefill.TestChunkedPrefill.test_mixed_chunked_prefill_without_radix_cache
"""

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase, run_mmlu_test, run_mulit_request_test

register_cuda_ci(est_time=312, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=312, suite="stage-b-test-small-1-gpu-amd")


class TestChunkedPrefill(CustomTestCase):
    def test_chunked_prefill(self):
        run_mmlu_test(disable_radix_cache=False, enable_mixed_chunk=False)

    def test_mixed_chunked_prefill(self):
        run_mmlu_test(disable_radix_cache=False, enable_mixed_chunk=True)

    def test_chunked_prefill_without_radix_cache(self):
        run_mmlu_test(disable_radix_cache=True, enable_mixed_chunk=False)

    def test_mixed_chunked_prefill_without_radix_cache(self):
        run_mmlu_test(disable_radix_cache=True, enable_mixed_chunk=True)

    def test_mixed_chunked_prefill_multi_requests(self):
        run_mulit_request_test(
            enable_mixed_chunk=True,
            chunked_prefill_size=2048,
        )


if __name__ == "__main__":
    unittest.main()
