"""Helpers shared by XPU nightly model tests.

The summary writer mirrors `python/sglang/test/ascend/test_ascend_utils.py`
so XPU and Ascend nightly runs render the same Markdown table in
`$GITHUB_STEP_SUMMARY`.
"""

import json
import os
import sys

from sglang.srt.environ import envs
from sglang.test.test_utils import is_in_ci, write_github_step_summary

HEADER = """
| Model | Server | Client | Prompts | Output Throughput | Expected Output Throughput | Accuracy | Expected Accuracy | Status |
| ----- | ------ | ------ | ------- | ----------------- | -------------------------- | -------- | ----------------- | ------ |
"""

_HEADER_WRITTEN = False


def _write_header_once():
    global _HEADER_WRITTEN
    if not _HEADER_WRITTEN:
        write_github_step_summary(HEADER)
        _HEADER_WRITTEN = True


def write_results_to_github_step_summary(results: dict):
    if not is_in_ci():
        return

    _write_header_once()

    def fmt(metrics, key, precision):
        v = metrics.get(key, "-")
        return f"{v:.{precision}f}" if isinstance(v, (int, float)) else v

    summary = ""
    for model, metrics in results.items():
        accuracy = fmt(metrics, "accuracy", 4)
        accuracy_threshold = metrics.get("accuracy_threshold", "N/A")
        output_throughput = fmt(metrics, "output_throughput", 2)
        output_throughput_threshold = metrics.get("output_throughput_threshold", "N/A")
        server = metrics.get("server", "N/A")
        client = metrics.get("client", "N/A")
        num_prompts = metrics.get("num_prompts", "N/A")
        error = metrics.get("error", "")
        status = "PASS" if error == "" else f"FAIL: {error}"
        summary += (
            f"| {model} | {server} | {client} | {num_prompts} "
            f"| {output_throughput} | {output_throughput_threshold} "
            f"| {accuracy} | {accuracy_threshold} | {status} |\n"
        )
    write_github_step_summary(summary)
    _append_metric_records(results)


def _append_metric_records(results: dict) -> None:
    """Append one JSON record per model to `SGLANG_TEST_METRICS_FILE`, if set.

    Consumed by the nightly XPU dashboard step in xpu-ci-job-monitor.yml to
    render per-model ref/actual/status/duration tables. Errors are swallowed
    so a broken write never turns a passing test red.
    """
    path = envs.SGLANG_TEST_METRICS_FILE.get()
    if not path:
        return
    # sys.argv[0] is the test script path when a unittest file is run via
    # `python3 test_foo.py`; renderer groups rich records to the file they came
    # from so file-level fallback rows don't double-count them.
    test_file = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else ""
    try:
        with open(path, "a") as f:
            for model, metrics in results.items():
                record = {
                    "kind": "model",
                    "test_file": test_file,
                    "model": model,
                    "accuracy": metrics.get("accuracy"),
                    "accuracy_threshold": metrics.get("accuracy_threshold"),
                    "output_throughput": metrics.get("output_throughput"),
                    "output_throughput_threshold": metrics.get(
                        "output_throughput_threshold"
                    ),
                    "latency": metrics.get("latency"),
                    "num_prompts": metrics.get("num_prompts"),
                    "num_threads": metrics.get("num_threads"),
                    "max_tokens": metrics.get("max_tokens"),
                    "error": metrics.get("error", ""),
                    "status": "pass" if not metrics.get("error") else "fail",
                }
                f.write(json.dumps(record) + "\n")
    except OSError:
        pass
