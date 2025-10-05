import json
import os
import unittest
import warnings
from datetime import datetime
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
    ModelDeploySetup,
    is_in_ci,
    parse_models,
    popen_launch_server,
    write_github_step_summary,
    write_results_to_json,
)

MODEL_SCORE_THRESHOLDS = {
    "meta-llama/Llama-3.1-8B-Instruct": 0.82,
    "mistralai/Mistral-7B-Instruct-v0.3": 0.58,
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": 0.85,
    "meta-llama/Llama-3.1-70B-Instruct": 0.95,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 0.64,
    "Qwen/Qwen2-57B-A14B-Instruct": 0.86,
    "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8": 0.83,
    "neuralmagic/Mistral-7B-Instruct-v0.3-FP8": 0.54,
    "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8": 0.94,
    "neuralmagic/Qwen2-72B-Instruct-FP8": 0.94,
    "neuralmagic/Qwen2-57B-A14B-Instruct-FP8": 0.86,
    "neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8": 0.65,
    "google/gemma-2-27b-it": 0.91,
    "neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8": 0.84,
}

NO_MOE_PADDING_MODELS = {"neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8"}
DISABLE_HF_XET_MODELS = {
    "Qwen/Qwen2-57B-A14B-Instruct",
    "neuralmagic/Qwen2-57B-A14B-Instruct-FP8",
}
TRITON_MOE_MODELS = {
    "neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8",
    "neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.3",
}


def check_model_scores(results):
    failed_models = []
    summary = " | model | score | threshold |\n"
    summary += "| ----- | ----- | --------- |\n"

    for model, score in results:
        threshold = MODEL_SCORE_THRESHOLDS.get(model)
        if threshold is None:
            print(f"Warning: No threshold defined for model {model}")
            continue

        if score < threshold:
            failed_models.append(
                f"\nScore Check Failed: {model}\n"
                f"Model {model} score ({score:.4f}) is below threshold ({threshold:.4f})"
            )

        line = f"| {model} | {score} | {threshold} |\n"
        summary += line

    print(summary)

    if is_in_ci():
        write_github_step_summary(f"### TestNightlyGsm8KEval\n{summary}")

    if failed_models:
        raise AssertionError("\n".join(failed_models))


# Do not use `CustomTestCase` since `test_mgsm_en_all_models` does not want retry
class TestNightlyGsm8KEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = []
        cls.base_url = DEFAULT_URL_FOR_TEST
        extra_args = ["--log-level-http", "warning", "--trust-remote-code"]

        def create_model_setup(model_path, tp_size):
            env = {
                "SGLANG_MOE_PADDING": (
                    "0" if model_path in NO_MOE_PADDING_MODELS else "1"
                ),
                "HF_HUB_DISABLE_XET": (
                    "1" if model_path in DISABLE_HF_XET_MODELS else "0"
                ),
                "SGLANG_USE_AITER": "0" if model_path in TRITON_MOE_MODELS else "1",
            }
            cls.models.append(
                ModelDeploySetup(
                    model_path, tp_size=tp_size, extra_args=extra_args, env=env
                )
            )

        models_tp1 = parse_models(
            DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1
        ) + parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1)
        for model_path in models_tp1:
            create_model_setup(model_path, 1)

        models_tp2 = parse_models(
            DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2
        ) + parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2)
        for model_path in models_tp2:
            create_model_setup(model_path, 2)

    def test_mgsm_en_all_models(self):
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )
        is_first = True
        all_results = []

        for model_setup in self.models:
            with self.subTest(model=model_setup.model_path):
                process = popen_launch_server(
                    model=model_setup.model_path,
                    base_url=self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=model_setup.extra_args,
                    env=model_setup.env,
                )

                args = SimpleNamespace(
                    base_url=self.base_url,
                    model=model_setup.model_path,
                    eval_name="mgsm_en",
                    num_examples=None,
                    num_threads=1024,
                )
                # Allow retries, so flaky errors are avoided.
                threshold = MODEL_SCORE_THRESHOLDS.get(model_setup.model_path)
                for attempt in range(3):
                    try:
                        metrics = run_eval(args)
                        score = metrics["score"]
                        if score >= threshold:
                            break
                    except Exception as e:
                        print(f"Attempt {attempt + 1} failed with error: {e}")
                print(
                    f"{'=' * 42}\n{model_setup.model_path} - metrics={metrics} score={metrics['score']}\n{'=' * 42}\n"
                )

                write_results_to_json(
                    model_setup.model_path, metrics, "w" if is_first else "a"
                )
                is_first = False

                all_results.append((model_setup.model_path, metrics["score"]))
                kill_process_tree(process.pid)

        try:
            with open("results.json", "r") as f:
                print("\nFinal Results from results.json:")
                print(json.dumps(json.load(f), indent=2))
        except Exception as e:
            print(f"Error reading results.json: {e}")

        # Check all scores after collecting all results
        check_model_scores(all_results)


if __name__ == "__main__":
    unittest.main()
