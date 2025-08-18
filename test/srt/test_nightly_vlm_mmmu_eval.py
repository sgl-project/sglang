import json
import os
import unittest
import warnings
from datetime import datetime
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    parse_models,
    popen_launch_server,
    write_github_step_summary,
)

MODEL_SCORE_THRESHOLDS = {
    # Conservative thresholds on 100 MMMU samples
    "Qwen/Qwen2-VL-7B-Instruct": 0.30,
    "Qwen/Qwen2.5-VL-7B-Instruct": 0.32,
    "OpenGVLab/InternVL2_5-2B": 0.22,
    "google/gemma-3-4b-it": 0.18,
}


DEFAULT_VLM_MODELS = ",".join(
    [
        "Qwen/Qwen2-VL-7B-Instruct",
        # "Qwen/Qwen2.5-VL-7B-Instruct",
        # "OpenGVLab/InternVL2_5-2B",
        # "google/gemma-3-4b-it",
    ]
)


def _extra_args_for_model(model: str):
    args = ["--trust-remote-code", "--cuda-graph-max-bs", "4"]
    if model.startswith("google/gemma-3"):
        args += ["--enable-multimodal"]
    return args


def popen_launch_server_wrapper(base_url, model):
    other_args = _extra_args_for_model(model)
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
    if mode == "a" and os.path.exists("results_vlm_mmmu.json"):
        try:
            with open("results_vlm_mmmu.json", "r") as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            existing_results = []

    if isinstance(existing_results, list):
        existing_results.append(result)
    else:
        existing_results = [result]

    with open("results_vlm_mmmu.json", "w") as f:
        json.dump(existing_results, f, indent=2)


def check_model_scores(results):
    failed_models = []
    summary = " | model | score | threshold | status |\n"
    summary += "| ----- | ----- | --------- | ------ |\n"

    for model, score in results:
        threshold = MODEL_SCORE_THRESHOLDS.get(model)
        if threshold is None:
            print(f"Warning: No threshold defined for model {model}")
            continue

        is_success = score >= threshold
        status_emoji = "✅" if is_success else "❌"

        if not is_success:
            failed_models.append(
                f"\nScore Check Failed: {model}\n"
                f"Model {model} score ({score:.4f}) is below threshold ({threshold:.4f})"
            )

        line = f"| {model} | {score} | {threshold} | {status_emoji} |\n"
        summary += line

    print(summary)

    if is_in_ci():
        write_github_step_summary(f"### TestNightlyVLMMmmuEval\n{summary}")

    if failed_models:
        raise AssertionError("\n".join(failed_models))


class TestNightlyVLMMmmuEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Allow overriding models via env, else use defaults
        cls.models = parse_models(
            os.environ.get("NIGHTLY_VLM_MODELS", DEFAULT_VLM_MODELS)
        )
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_mmmu_vlm_models(self):
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )
        is_first = True
        all_results = []

        for model in self.models:
            with self.subTest(model=model):
                process = popen_launch_server_wrapper(self.base_url, model)

                try:
                    args = SimpleNamespace(
                        base_url=self.base_url,
                        model=model,
                        eval_name="mmmu",
                        num_examples=100,
                        num_threads=512,
                        max_tokens=30,
                    )

                    metrics = run_eval(args)
                    print(
                        f"{'=' * 42}\n{model} - metrics={metrics} score={metrics['score']}\n{'=' * 42}\n"
                    )

                    write_results_to_json(model, metrics, "w" if is_first else "a")
                    is_first = False

                    all_results.append((model, metrics["score"]))
                finally:
                    kill_process_tree(process.pid)

        try:
            with open("results_vlm_mmmu.json", "r") as f:
                print("\nFinal Results from results_vlm_mmmu.json:")
                print(json.dumps(json.load(f), indent=2))
        except Exception as e:
            print(f"Error reading results_vlm_mmmu.json: {e}")

        check_model_scores(all_results)


if __name__ == "__main__":
    unittest.main()
