import unittest

from nightly_utils import NightlyBenchmarkRunner

from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

MINIMAX_M2_MODEL_PATH = "MiniMaxAI/MiniMax-M2"
PROFILE_DIR = "performance_profiles_minimax_m2"


class TestNightlyMiniMaxM2Performance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MINIMAX_M2_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        # MiniMax-M2 is a 230B MoE model with 10B active params
        cls.other_args = [
            "--tp",
            "8",
            "--ep",
            "8",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
            "--trust-remote-code",
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
