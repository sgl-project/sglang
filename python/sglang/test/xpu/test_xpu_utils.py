"""Helpers shared by XPU nightly model tests.

The summary writer mirrors `python/sglang/test/ascend/test_ascend_utils.py`
so XPU and Ascend nightly runs render the same Markdown table in
`$GITHUB_STEP_SUMMARY`.
"""

from sglang.test.test_utils import is_in_ci, write_github_step_summary

HEADER = """
| Model | Server | Client | Output Throughput | Expected Output Throughput | Accuracy | Expected Accuracy | Status |
| ----- | ------ | ------ | ----------------- | -------------------------- | -------- | ----------------- | ------ |
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
        error = metrics.get("error", "")
        status = "PASS" if error == "" else f"FAIL: {error}"
        summary += (
            f"| {model} | {server} | {client} | {output_throughput} "
            f"| {output_throughput_threshold} | {accuracy} "
            f"| {accuracy_threshold} | {status} |\n"
        )
    write_github_step_summary(summary)
