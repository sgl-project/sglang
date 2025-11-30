import unittest

from nightly_utils import NightlyBenchmarkRunner

from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

KIMI_K2_THINKING_MODEL_PATH = "moonshotai/Kimi-K2-Thinking"
PROFILE_DIR = "performance_profiles_kimi_k2_thinking"


class TestNightlyKimiK2ThinkingPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = KIMI_K2_THINKING_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        # Kimi-K2-Thinking requires specific launch arguments
        cls.other_args = [
            "--tp",
            "8",
            "--trust-remote-code",
            "--tool-call-parser",
            "kimi_k2",
            "--reasoning-parser",
            "kimi_k2",
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
            extra_bench_args=["--trust-remote-code"],
        )

        self.runner.add_report(results)
        self.runner.write_final_report()

        if not success:
            raise AssertionError(
                f"Benchmark failed for {self.model}. Check the logs for details."
            )


if __name__ == "__main__":
    unittest.main()
