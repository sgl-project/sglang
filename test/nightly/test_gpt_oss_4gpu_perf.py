import unittest

from nightly_utils import NightlyBenchmarkRunner

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

register_cuda_ci(est_time=600, suite="nightly-4-gpu-b200", nightly=True)

PROFILE_DIR = "performance_profiles_gpt_oss_4gpu"
# GPT-OSS uses the gpt-oss parser for tool calling
TOOL_CALL_PARSER = "gpt-oss"


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
        all_perf_succeed = True
        all_tool_call_succeed = True

        for model_path, other_args in self.models:
            with self.subTest(model=model_path):
                # Run combined perf + tool call benchmark (single server launch)
                perf_results, tool_results, perf_ok, tool_ok = (
                    self.runner.run_perf_and_tool_call_benchmark(
                        model_path=model_path,
                        batch_sizes=self.batch_sizes,
                        input_lens=self.input_lens,
                        output_lens=self.output_lens,
                        other_args=other_args,
                        tool_call_parser=TOOL_CALL_PARSER,
                    )
                )

                if not perf_ok:
                    all_perf_succeed = False
                if not tool_ok:
                    all_tool_call_succeed = False

                self.runner.add_report(perf_results)
                if tool_results:
                    self.runner.add_tool_call_report(tool_results)

        self.runner.write_final_report()

        # Fail at end if ANY test failed
        if not all_perf_succeed or not all_tool_call_succeed:
            raise AssertionError("Some benchmarks failed (perf or tool call).")


if __name__ == "__main__":
    unittest.main()
