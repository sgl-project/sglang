import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase, run_mmlu_test

register_cuda_ci(est_time=360, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=312, suite="stage-b-test-1-gpu-small-amd")


class TestMixedChunkedPrefill(CustomTestCase):
    def test_mixed_chunked_prefill(self):
        run_mmlu_test(disable_radix_cache=False, enable_mixed_chunk=True)

    def test_mixed_chunked_prefill_without_radix_cache(self):
        run_mmlu_test(disable_radix_cache=True, enable_mixed_chunk=True)


if __name__ == "__main__":
    unittest.main()
