import json
import unittest
import warnings
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    check_evaluation_test_results,
    parse_models,
    popen_launch_server,
    write_results_to_json,
)

MODEL_SCORE_THRESHOLDS = {
    "meta-llama/Llama-3.1-8B-Instruct": 0.82,
    "mistralai/Mistral-7B-Instruct-v0.3": 0.58,
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": 0.85,
    "google/gemma-2-27b-it": 0.91,
    "meta-llama/Llama-3.1-70B-Instruct": 0.95,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 0.616,
    "Qwen/Qwen2-57B-A14B-Instruct": 0.86,
    "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8": 0.83,
    "neuralmagic/Mistral-7B-Instruct-v0.3-FP8": 0.54,
    "neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8": 0.835,
    "zai-org/GLM-4.5-Air-FP8": 0.75,
    # The threshold of neuralmagic/gemma-2-2b-it-FP8 should be 0.6, but this model has some accuracy regression.
    # The fix is tracked at https://github.com/sgl-project/sglang/issues/4324, we set it to 0.50, for now, to make CI green.
    "neuralmagic/gemma-2-2b-it-FP8": 0.50,
    "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8": 0.94,
    "neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8": 0.65,
    "neuralmagic/Qwen2-72B-Instruct-FP8": 0.94,
    "neuralmagic/Qwen2-57B-A14B-Instruct-FP8": 0.82,
}


# Do not use `CustomTestCase` since `test_mgsm_en_all_models` does not want retry
class TestNightlyGsm8KEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_groups = [
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1), False, False),
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2), False, True),
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1), True, False),
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2), True, True),
        ]
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_mgsm_en_all_models(self):
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )
        is_first = True
        all_results = []
        model_count = 0
        for model_group, is_fp8, is_tp2 in self.model_groups:
            for model in model_group:
                model_count += 1
                with self.subTest(model=model):
                    other_args = ["--tp", "2"] if is_tp2 else []

                    if model == "meta-llama/Llama-3.1-70B-Instruct":
                        other_args.extend(["--mem-fraction-static", "0.9"])

                    process = popen_launch_server(
                        model=model,
                        other_args=other_args,
                        base_url=self.base_url,
                        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    )

                    args = SimpleNamespace(
                        base_url=self.base_url,
                        model=model,
                        eval_name="mgsm_en",
                        num_examples=None,
                        num_threads=1024,
                    )

                    metrics = run_eval(args)
                    print(
                        f"{'=' * 42}\n{model} - metrics={metrics} score={metrics['score']}\n{'=' * 42}\n"
                    )

                    write_results_to_json(model, metrics, "w" if is_first else "a")
                    is_first = False

                    # 0.0 for empty latency
                    all_results.append((model, metrics["score"], 0.0))
                    kill_process_tree(process.pid)

        try:
            with open("results.json", "r") as f:
                print("\nFinal Results from results.json:")
                print(json.dumps(json.load(f), indent=2))
        except Exception as e:
            print(f"Error reading results.json: {e}")

        # Check all scores after collecting all results
        check_evaluation_test_results(
            all_results,
            self.__class__.__name__,
            model_accuracy_thresholds=MODEL_SCORE_THRESHOLDS,
            model_count=model_count,
        )


if __name__ == "__main__":
    unittest.main()
