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
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

MODEL_SCORE_THRESHOLDS = {
    "meta-llama/Llama-3.1-8B-Instruct": 0.82,
    "mistralai/Mistral-7B-Instruct-v0.3": 0.56,
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": 0.85,
    "meta-llama/Llama-3.1-70B-Instruct": 0.95,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 0.64,
    "Qwen/Qwen2-57B-A14B-Instruct": 0.86,
    "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8": 0.81,
    "neuralmagic/Mistral-7B-Instruct-v0.3-FP8": 0.54,
    "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8": 0.94,
    "neuralmagic/Qwen2-72B-Instruct-FP8": 0.94,
    "neuralmagic/Qwen2-57B-A14B-Instruct-FP8": 0.82,
}

# Models currently failing on AMD MI300x.
failing_models = {
    "google/gemma-2-27b-it",
    "neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8",
    "neuralmagic/gemma-2-2b-it-FP8",
    "neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8",
}


def remove_failing_models(model_str):
    models = model_str.split(",")
    filtered = [m for m in models if m not in failing_models]
    return ",".join(filtered)


DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1 = remove_failing_models(
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1
)
DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2 = remove_failing_models(
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2
)
DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1 = remove_failing_models(
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1
)
DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2 = remove_failing_models(
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2
)


def parse_models(model_string):
    return [model.strip() for model in model_string.split(",") if model.strip()]


def popen_launch_server_wrapper(base_url, model, is_tp2):
    other_args = ["--log-level-http", "warning", "--trust-remote-code"]
    if is_tp2:
        other_args.extend(["--tp", "2"])

    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )
    return process


def write_results_to_json(model, metrics, mode="a"):
    result = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "metrics": metrics,
        "score": metrics["score"],
    }

    existing_results = []
    if mode == "a" and os.path.exists("results.json"):
        try:
            with open("results.json", "r") as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            existing_results = []

    if isinstance(existing_results, list):
        existing_results.append(result)
    else:
        existing_results = [result]

    with open("results.json", "w") as f:
        json.dump(existing_results, f, indent=2)


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

        for model_group, is_fp8, is_tp2 in self.model_groups:
            for model in model_group:
                with self.subTest(model=model):
                    process = popen_launch_server_wrapper(self.base_url, model, is_tp2)

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

                    all_results.append((model, metrics["score"]))
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
