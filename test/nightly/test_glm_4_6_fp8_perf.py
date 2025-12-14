import unittest

from nightly_utils import NightlyBenchmarkRunner

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

register_cuda_ci(est_time=900, suite="nightly-4-gpu-b200", nightly=True)


GLM_4_6_FP8_MODEL_PATH = "zai-org/GLM-4.6-FP8"
PROFILE_DIR = "performance_profiles_glm_4_6_fp8"


class TestNightlyGLM46FP8Performance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GLM_4_6_FP8_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        # TODO: Currently for h200, we are using 8 gpu runner. We may need h200 4 gpu runner in the future.
        cls.other_args = [
            "--tp",
            "4",
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
        )

        self.runner.add_report(results)
        self.runner.write_final_report()

        if not success:
            raise AssertionError(
                f"Benchmark failed for {self.model}. Check the logs for details."
            )


if __name__ == "__main__":
    unittest.main()
