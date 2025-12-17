import unittest

from nightly_utils import NightlyBenchmarkRunner

from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

DEEPSEEK_V31_MODEL_PATH = "deepseek-ai/DeepSeek-V3.1"
PROFILE_DIR = "performance_profiles_deepseek_v31"
# DeepSeek V3.1 uses the deepseekv31 parser for tool calling
TOOL_CALL_PARSER = "deepseekv31"


class TestNightlyDeepseekV31Performance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V31_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        # Define variant configurations
        cls.variants = [
            {
                "name": "basic",
                "other_args": [
                    "--trust-remote-code",
                    "--tp",
                    "8",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                ],
            },
            {
                "name": "mtp",
                "other_args": [
                    "--trust-remote-code",
                    "--tp",
                    "8",
                    "--speculative-algorithm",
                    "EAGLE",
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--mem-frac",
                    "0.7",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                ],
            },
        ]

        cls.runner = NightlyBenchmarkRunner(PROFILE_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_profile_directory()

    def test_bench_one_batch(self):
        failed_perf_variants = []
        failed_tool_call_variants = []

        try:
            for variant_config in self.variants:
                with self.subTest(variant=variant_config["name"]):
                    # Run combined perf + tool call benchmark (single server launch)
                    perf_results, tool_results, perf_ok, tool_ok = (
                        self.runner.run_perf_and_tool_call_benchmark(
                            model_path=self.model,
                            batch_sizes=self.batch_sizes,
                            input_lens=self.input_lens,
                            output_lens=self.output_lens,
                            other_args=variant_config["other_args"],
                            variant=variant_config["name"],
                            tool_call_parser=TOOL_CALL_PARSER,
                        )
                    )

                    if not perf_ok:
                        failed_perf_variants.append(variant_config["name"])
                    if not tool_ok:
                        failed_tool_call_variants.append(variant_config["name"])

                    self.runner.add_report(perf_results)
                    if tool_results:
                        self.runner.add_tool_call_report(tool_results)
        finally:
            self.runner.write_final_report()

        # Fail at end if ANY test failed
        if failed_perf_variants or failed_tool_call_variants:
            msg = f"Benchmark failed for {self.model}."
            if failed_perf_variants:
                msg += f" Perf failed variants: {', '.join(failed_perf_variants)}."
            if failed_tool_call_variants:
                msg += f" Tool call failed variants: {', '.join(failed_tool_call_variants)}."
            raise AssertionError(msg)


if __name__ == "__main__":
    unittest.main()
