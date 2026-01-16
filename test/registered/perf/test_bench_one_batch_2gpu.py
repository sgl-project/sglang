import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_MOE_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    is_in_ci,
    run_bench_offline_throughput,
    write_github_step_summary,
)

register_cuda_ci(est_time=180, suite="stage-b-test-large-2-gpu-performance")


class TestBenchOneBatch2GPU(CustomTestCase):

    def test_moe_tp2_bs1(self):
        output_throughput = run_bench_offline_throughput(
            DEFAULT_MOE_MODEL_NAME_FOR_TEST, ["--tp", "2", "--cuda-graph-max-bs", "2"]
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_moe_tp2_bs1 (Mixtral-8x7B)\n"
                f"output_throughput: {output_throughput:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(output_throughput, 85)
            else:
                self.assertGreater(output_throughput, 125)

    def test_torch_compile_tp2_bs1(self):
        output_throughput = run_bench_offline_throughput(
            DEFAULT_MODEL_NAME_FOR_TEST,
            ["--tp", "2", "--enable-torch-compile", "--cuda-graph-max-bs", "2"],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_torch_compile_tp2_bs1 (Mixtral-8x7B)\n"
                f"output_throughput: {output_throughput:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(output_throughput, 200)
            else:
                self.assertGreater(output_throughput, 220)


if __name__ == "__main__":
    unittest.main()
