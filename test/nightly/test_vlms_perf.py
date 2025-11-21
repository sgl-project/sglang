import os
import unittest
import warnings

from nightly_utils import NightlyBenchmarkRunner

from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    ModelLaunchSettings,
    _parse_int_list_env,
    parse_models,
)

PROFILE_DIR = "performance_profiles_vlms"

MODEL_DEFAULTS = [
    # Keep conservative defaults. Can be overridden by env NIGHTLY_VLM_MODELS
    ModelLaunchSettings(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        extra_args=["--mem-fraction-static=0.7"],
    ),
    ModelLaunchSettings(
        "google/gemma-3-27b-it",
    ),
    ModelLaunchSettings("Qwen/Qwen3-VL-30B-A3B-Instruct", extra_args=["--tp=2"]),
    # "OpenGVLab/InternVL2_5-2B",
    # buggy in official transformers impl
    # "openbmb/MiniCPM-V-2_6",
]


class TestNightlyVLMModelsPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )

        nightly_vlm_models_str = os.environ.get("NIGHTLY_VLM_MODELS")
        if nightly_vlm_models_str:
            cls.models = []
            model_paths = parse_models(nightly_vlm_models_str)
            for model_path in model_paths:
                cls.models.append(ModelLaunchSettings(model_path))
        else:
            cls.models = MODEL_DEFAULTS

        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.batch_sizes = _parse_int_list_env("NIGHTLY_VLM_BATCH_SIZES", "1,1,2,8,16")
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_VLM_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_VLM_OUTPUT_LENS", "512"))
        cls.runner = NightlyBenchmarkRunner(PROFILE_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_profile_directory()

    def test_bench_one_batch(self):
        all_model_succeed = True

        for model_setup in self.models:
            with self.subTest(model=model_setup.model_path):
                # VLMs need additional benchmark args for dataset and trust-remote-code
                extra_bench_args = [
                    "--trust-remote-code",
                    "--dataset-name=mmmu",
                ]

                results, success = self.runner.run_benchmark_for_model(
                    model_path=model_setup.model_path,
                    batch_sizes=self.batch_sizes,
                    input_lens=self.input_lens,
                    output_lens=self.output_lens,
                    other_args=model_setup.extra_args,
                    extra_bench_args=extra_bench_args,
                )

                if not success:
                    all_model_succeed = False

                self.runner.add_report(results)

        self.runner.write_final_report()

        if not all_model_succeed:
            raise AssertionError("Some models failed the perf tests.")


if __name__ == "__main__":
    unittest.main()
