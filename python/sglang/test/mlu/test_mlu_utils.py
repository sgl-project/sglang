import os

from sglang.test.test_utils import write_github_step_summary

MODEL_WEIGHTS_DIR = os.environ.get("MLU_MODEL_WEIGHTS_DIR", "/data/models")

QWEN3_8B_WEIGHTS_PATH = os.environ.get(
    "MLU_QWEN3_8B_MODEL_PATH", os.path.join(MODEL_WEIGHTS_DIR, "Qwen3-8B")
)

_HEADER = """
| Model | Server | Client | Output Throughput | Expected Output Throughput | Latency | Accuracy | Expected Accuracy | Status |
| ----- | ------ | ------ | ----------------- | -------------------------- | ------- | -------- | ----------------- | ------ |
"""


def write_results_to_github_step_summary(results: dict):
    if not os.getenv("SGLANG_IS_IN_CI"):
        return

    _write_github_step_summary_once(_HEADER)

    def get_float(metrics, item, precision):
        value = metrics.get(item, "-")
        return f"{value:.{precision}f}" if isinstance(value, (int, float)) else value

    summary = ""
    for model, metrics in results.items():
        model = model.replace(MODEL_WEIGHTS_DIR, "")
        output_throughput = get_float(metrics, "output_throughput", 2)
        output_throughput_threshold = metrics.get("output_throughput_threshold", "N/A")
        accuracy = get_float(metrics, "accuracy", 4)
        accuracy_threshold = metrics.get("accuracy_threshold", "N/A")
        latency = get_float(metrics, "latency", 4)
        server = metrics.get("server", "N/A")
        client = metrics.get("client", "N/A")
        error = metrics.get("error", "")
        status = "PASS" if error == "" else f"FAIL {error}"
        summary += f"| {model} | {server} | {client} | {output_throughput} | {output_throughput_threshold} | {latency} | {accuracy} | {accuracy_threshold} | {status} |\n"
    write_github_step_summary(summary)


def _write_github_step_summary_once(summary: str):
    if getattr(_write_github_step_summary_once, "has_written", False):
        return
    _write_github_step_summary_once.has_written = True
    write_github_step_summary(summary)
