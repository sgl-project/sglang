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
                    "--dp",
                    "8",
                    "--enable-dp-attention",
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
                    "--dp",
                    "8",
                    "--enable-dp-attention",
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
            {
                "name": "nsa",
                "other_args": [
                    "--trust-remote-code",
                    "--tp",
                    "8",
                    "--dp",
                    "8",
                    "--enable-dp-attention",
                    "--attention-backend",
                    "nsa",
                    "--nsa-prefill-backend",
                    "flashmla_sparse",
                    "--nsa-decode-backend",
                    "flashmla_kv",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                ],
            },
            {
                "name": "pure_tp",
                "other_args": [
                    "--trust-remote-code",
                    "--tp",
                    "8",
                    "--attention-backend",
                    "nsa",
                    "--nsa-prefill-backend",
                    "flashmla_sparse",
                    "--nsa-decode-backend",
                    "flashmla_kv",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                ],
            },
        ]

        cls.runner = NightlyBenchmarkRunner(PROFILE_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_profile_directory()

    def test_bench_one_batch(self):
        failed_variants = []

        try:
            for variant_config in self.variants:
                with self.subTest(variant=variant_config["name"]):
                    results, success = self.runner.run_benchmark_for_model(
                        model_path=self.model,
                        batch_sizes=self.batch_sizes,
                        input_lens=self.input_lens,
                        output_lens=self.output_lens,
                        other_args=variant_config["other_args"],
                        variant=variant_config["name"],
                    )

                    if not success:
                        failed_variants.append(variant_config["name"])

                    self.runner.add_report(results)
        finally:
            self.runner.write_final_report()

        if failed_variants:
            raise AssertionError(
                f"Benchmark failed for {self.model} with the following variants: "
                f"{', '.join(failed_variants)}"
            )


if __name__ == "__main__":
    unittest.main()
