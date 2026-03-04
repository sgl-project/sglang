import unittest

from nightly_utils import NightlyBenchmarkRunner

from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    ModelLaunchSettings,
    _parse_int_list_env,
    parse_models,
)

PROFILE_DIR = "performance_profiles_text_models"


class TestNightlyTextModelsPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = []
        # TODO: replace with DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1 or other model lists
        for model_path in parse_models("meta-llama/Llama-3.1-8B-Instruct"):
            cls.models.append(ModelLaunchSettings(model_path, tp_size=1))
        for model_path in parse_models("Qwen/Qwen2-57B-A14B-Instruct"):
            cls.models.append(ModelLaunchSettings(model_path, tp_size=2))
        # (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1), False, False),
        # (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2), False, True),
        # (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1), True, False),
        # (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2), True, True),
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))
        cls.runner = NightlyBenchmarkRunner(PROFILE_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_profile_directory()

    def test_bench_one_batch(self):
        all_model_succeed = True

        for model_setup in self.models:
            with self.subTest(model=model_setup.model_path):
                results, success = self.runner.run_benchmark_for_model(
                    model_path=model_setup.model_path,
                    batch_sizes=self.batch_sizes,
                    input_lens=self.input_lens,
                    output_lens=self.output_lens,
                    other_args=model_setup.extra_args,
                )

                if not success:
                    all_model_succeed = False

                self.runner.add_report(results)

        self.runner.write_final_report()

        if not all_model_succeed:
            raise AssertionError("Some models failed the perf tests.")


if __name__ == "__main__":
    unittest.main()
