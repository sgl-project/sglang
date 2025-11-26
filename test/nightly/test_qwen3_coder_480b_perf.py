import unittest

from nightly_utils import NightlyBenchmarkRunner

from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

QWEN3_CODER_480B_MODEL_PATH = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
PROFILE_DIR = "performance_profiles_qwen3_coder_480b"


class TestNightlyQwen3Coder480BPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_CODER_480B_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        # Qwen3-Coder-480B is a 480B MoE model with 35B active params
        cls.other_args = [
            "--tp",
            "8",
            "--ep",
            "8",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
        ]

        cls.runner = NightlyBenchmarkRunner(PROFILE_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_profile_directory()

    def test_bench_one_batch(self):
        results, success = self.runner.run_benchmark_for_model(
            model_path=self.model,
            batch_sizes=self.batch_sizes,
            input_lens=self.input_lens,
            output_lens=self.output_lens,
            other_args=self.other_args,
        )

        self.runner.add_report(results)
        self.runner.write_final_report()

        if not success:
            raise AssertionError(
                f"Benchmark failed for {self.model}. Check the logs for details."
            )


if __name__ == "__main__":
    unittest.main()
