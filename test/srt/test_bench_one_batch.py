import unittest

from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_MOE_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    run_bench_offline_throughput,
    run_bench_one_batch,
    write_github_step_summary,
)

# We use `run_bench_offline_throughput`` instead of `run_bench_one_batch` for most cases
# because `run_bench_offline_throughput`` has overlap scheduler.


class TestBenchOneBatch(CustomTestCase):

    def test_bs1_small(self):
        output_throughput = run_bench_one_batch(
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

    def test_moe_tp2_bs1(self):
        output_throughput = run_bench_offline_throughput(
            DEFAULT_MOE_MODEL_NAME_FOR_TEST, ["--tp", "2", "--cuda-graph-max-bs", "2"]
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_moe_tp2_bs1 (Mixtral-8x7B)\n"
                f"output_throughput: {output_throughput:.2f} token/s\n"
            )
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
            self.assertGreater(output_throughput, 220)


if __name__ == "__main__":
    unittest.main()
