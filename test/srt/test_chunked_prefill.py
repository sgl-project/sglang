"""
python3 -m unittest test_chunked_prefill.TestChunkedPrefill.test_mixed_chunked_prefill_without_radix_cache
"""

import unittest

from sglang.test.test_utils import CustomTestCase, run_mmlu_test, run_mulit_request_test


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
