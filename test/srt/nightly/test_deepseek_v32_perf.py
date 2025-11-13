import unittest

from nightly_utils import NightlyBenchmarkRunner

from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2-Exp"
PROFILE_DIR = "performance_profiles_deepseek_v32"


class TestNightlyDeepseekV32Performance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V32_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 1, 8, 16, 64]
        # Test both 4k and 32k input lengths
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096,32768"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        # Use default configuration (removed dp attention)
        cls.other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
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

        self.runner.write_final_report()

        if not success:
            raise AssertionError(f"Benchmark failed for {self.model}")


if __name__ == "__main__":
    unittest.main()
