import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    run_bench_offline_throughput,
    run_bench_one_batch,
    write_github_step_summary,
)

register_cuda_ci(est_time=120, suite="stage-b-test-large-1-gpu-performance")


class TestBenchOneBatch1GPU(CustomTestCase):

    def test_bs1_small(self):
        _, output_throughput, _ = run_bench_one_batch(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST, ["--cuda-graph-max-bs", "2"]
        )
        self.assertGreater(output_throughput, 50)

    def test_bs1_default(self):
        output_throughput = run_bench_offline_throughput(
            DEFAULT_MODEL_NAME_FOR_TEST, ["--cuda-graph-max-bs", "2"]
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs1_default (llama-3.1-8b)\n"
                f"output_throughput: {output_throughput:.2f} token/s\n"
            )
            self.assertGreater(output_throughput, 135)


if __name__ == "__main__":
    unittest.main()
