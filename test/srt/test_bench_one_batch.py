import unittest

from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_MOE_MODEL_NAME_FOR_TEST,
    is_in_ci,
    run_bench_one_batch,
    write_github_step_summary,
)


class TestBenchOneBatch(unittest.TestCase):
    def test_default(self):
        output_throughput = run_bench_one_batch(DEFAULT_MODEL_NAME_FOR_TEST, [])

        if is_in_ci():
            write_github_step_summary(
                f"### test_default\n"
                f"output_throughput : {output_throughput:.2f} token/s\n"
            )
            self.assertGreater(output_throughput, 140)

    def test_moe_tp2(self):
        output_throughput = run_bench_one_batch(
            DEFAULT_MOE_MODEL_NAME_FOR_TEST, ["--tp", "2"]
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_moe_tp2\n"
                f"output_throughput : {output_throughput:.2f} token/s\n"
            )
            self.assertGreater(output_throughput, 125)

    def test_torch_compile_tp2(self):
        output_throughput = run_bench_one_batch(
            DEFAULT_MODEL_NAME_FOR_TEST, ["--tp", "2", "--enable-torch-compile"]
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_torch_compile_tp2\n"
                f"output_throughput : {output_throughput:.2f} token/s\n"
            )
            self.assertGreater(output_throughput, 250)


if __name__ == "__main__":
    unittest.main()
