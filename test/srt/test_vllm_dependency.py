import json
import os
import unittest
import warnings
from datetime import datetime
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_QUANT_TP1,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
    write_results_to_json,
)

MODEL_SCORE_THRESHOLDS = {
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4": 0.825,
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4": 0.825,
    "hugging-quants/Mixtral-8x7B-Instruct-v0.1-AWQ-INT4": 0.62,
}


def parse_models(model_string):
    return [model.strip() for model in model_string.split(",") if model.strip()]


def popen_launch_server_wrapper(base_url, model, is_fp8, is_tp2):
    other_args = ["--log-level-http", "warning", "--trust-remote-code"]
    if is_fp8:
        if "Llama-3" in model or "gemma-2" in model:
            other_args.extend(["--kv-cache-dtype", "fp8_e5m2"])
        elif "Qwen2-72B-Instruct-FP8" in model:
            other_args.extend(["--quantization", "fp8"])
        elif "neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8" in model:
            other_args.extend([])
        else:
            other_args.extend(["--quantization", "fp8", "--kv-cache-dtype", "fp8_e5m2"])
    if is_tp2:
        other_args.extend(["--tp", "2"])
    if "DeepSeek" in model:
        other_args.extend(["--mem-frac", "0.85"])

    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )
    return process


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
        write_github_step_summary(
            f"### TestNightlyGsm8KEval for vLLM awq, gptq, gguf\n{summary}"
        )

    if failed_models:
        raise AssertionError("\n".join(failed_models))


class TestNightlyGsm8KEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_groups = [
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_QUANT_TP1), False, False),
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
                    process = popen_launch_server_wrapper(
                        self.base_url, model, is_fp8, is_tp2
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
