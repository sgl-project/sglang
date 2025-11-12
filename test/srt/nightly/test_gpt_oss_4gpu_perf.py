import unittest

from nightly_utils import NightlyBenchmarkRunner

from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

PROFILE_DIR = "performance_profiles_gpt_oss_4gpu"


class TestNightlyGptOss4GpuPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = [
            (
                "openai/gpt-oss-120b",
                [
                    "--tp",
                    "4",
                    "--cuda-graph-max-bs",
                    "200",
                    "--mem-fraction-static",
                    "0.93",
                ],
            ),
        ]
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 1, 8, 16, 64]
        cls.input_lens = (4096,)
        cls.output_lens = (512,)
        cls.runner = NightlyBenchmarkRunner(PROFILE_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_profile_directory()

    def test_bench_one_batch(self):
        all_model_succeed = True

        for model_path, other_args in self.models:
            with self.subTest(model=model_path):
                results, success = self.runner.run_benchmark_for_model(
                    model_path=model_path,
                    batch_sizes=self.batch_sizes,
                    input_lens=self.input_lens,
                    output_lens=self.output_lens,
                    other_args=other_args,
                )

                if not success:
                    all_model_succeed = False

                self.runner.add_report(results)

        self.runner.write_final_report()

        if not all_model_succeed:
            raise AssertionError("Some models failed the perf tests.")


if __name__ == "__main__":
    unittest.main()
