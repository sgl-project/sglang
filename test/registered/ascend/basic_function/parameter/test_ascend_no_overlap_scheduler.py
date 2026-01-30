import unittest

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase, run_mmlu_test

register_npu_ci(
    est_time=400,
    suite="nightly-1-npu-a3",
    nightly=True,
    disabled="run failed",
)


class TestOverlapSchedule(CustomTestCase):
    """Testcase: Verify that the model can successfully process inference requests and achieve an accuracy of ≥ 0.65 when the overlap scheduler is disabled,
    covering all combination scenarios of radix cache (enabled/disabled) and chunked prefill (enabled/disabled).

    [Test Category] Parameter
    [Test Target] --disable-radix-cache;--disable-overlap
    """

    def test_no_radix_attention_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=True,
            chunked_prefill_size=128,
            disable_overlap=True,
        )

    def test_no_radix_attention_no_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=True, chunked_prefill_size=-1, disable_overlap=True
        )

    def test_radix_attention_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=False,
            chunked_prefill_size=128,
            disable_overlap=True,
        )

    def test_radix_attention_no_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=False, chunked_prefill_size=-1, disable_overlap=True
        )


if __name__ == "__main__":
    unittest.main()
