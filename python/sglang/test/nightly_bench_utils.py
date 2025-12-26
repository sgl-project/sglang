import json
import logging
import os
from typing import List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# TODO:
# There is huge redundancy between BenchmarkResult and BenchOneCaseResult, and redundancy between to_markdown_row, generate_markdown_report, get_report_summary.
# We should refactor them to reduce the code duplication.
# 1. Delete the BenchmarkResult use BenchOneCaseResult directly.
# 2. Merge all related markdown rendering functions into BenchOneCaseResult


class BenchmarkResult(BaseModel):
    """Pydantic model for benchmark results table data, for a single isl and osl"""

    model_path: str
    run_name: str
    batch_size: int
    input_len: int
    output_len: int
    latency: float
    input_throughput: float
    output_throughput: float
    overall_throughput: float
    last_ttft: float
    last_gen_throughput: float
    acc_length: Optional[float] = None
    profile_link_extend: Optional[str] = None
    profile_link_decode: Optional[str] = None

    @staticmethod
    def help_str() -> str:
        return f"""
Note: To view the traces through perfetto-ui, please:
    1. open with Google Chrome
    2. allow popup
"""

    def to_markdown_row(
        self, trace_dir, base_url: str = "", relay_base: str = ""
    ) -> str:
        """Convert this benchmark result to a markdown table row."""

        hourly_cost_per_gpu = 2  # $2/hour for one H100
        hourly_cost = hourly_cost_per_gpu * 1  # Assuming tp_size = 1 for simplicity
        input_util = 0.7
        accept_length = round(self.acc_length, 2) if self.acc_length > 0 else "n/a"
        itl = 1 / (self.output_throughput / self.batch_size) * 1000
        input_cost = 1e6 / (self.input_throughput * input_util) / 3600 * hourly_cost
        output_cost = 1e6 / self.output_throughput / 3600 * hourly_cost

        def get_perfetto_relay_link_from_trace_file(trace_file: str):
            from urllib.parse import quote

            rel_path = os.path.relpath(trace_file, trace_dir)
            raw_file_link = f"{base_url}/{rel_path}"
            relay_link = (
                f"{relay_base}?src={quote(raw_file_link, safe='')}"
                if relay_base
                else raw_file_link
            )
            return relay_link

        # Handle profile links
        profile_link = "NA | NA"
        if self.profile_link_extend or self.profile_link_decode:
            # Create a combined link or use the first available one
            trace_files = [self.profile_link_extend, self.profile_link_decode]
            if any(trace_file is None for trace_file in trace_files):
                logger.error("Some trace files are None", f"{trace_files=}")
            trace_files_relay_links = [
                (
                    f"[trace]({get_perfetto_relay_link_from_trace_file(trace_file)})"
                    if trace_file
                    else "N/A"
                )
                for trace_file in trace_files
            ]

            profile_link = " | ".join(trace_files_relay_links)

        # Build the row
        return f"| {self.batch_size} | {self.input_len} | {self.latency:.2f} | {self.input_throughput:.2f} | {self.output_throughput:.2f} | {accept_length} | {itl:.2f} | {input_cost:.2f} | {output_cost:.2f} | {profile_link} |\n"


def generate_markdown_report(trace_dir, results: List[BenchmarkResult]) -> str:
    """Generate a markdown report from a list of BenchmarkResult object from a single run."""
    # Build model header with run_name if it's not "default"
    model_header = results[0].model_path
    if results[0].run_name and results[0].run_name != "default":
        model_header += f" ({results[0].run_name})"

    # Include GPU config in model header if available
    gpu_config = os.getenv("GPU_CONFIG", "")
    if gpu_config:
        model_header += f" [{gpu_config}]"

    summary = f"### {model_header}\n"

    summary += "| batch size | input len | latency (s) | input throughput (tok/s)  | output throughput (tok/s) | acc length | ITL (ms) | input cost ($/1M) | output cost ($/1M) | profile (extend) | profile (decode)|\n"
    summary += "| ---------- | --------- | ----------- | ------------------------- | ------------------------- | ---------- | -------- | ----------------- | ------------------ | ---------------- | --------------- |\n"

    # all results should share the same isl & osl
    for result in results:
        base_url = os.getenv("TRACE_BASE_URL", "").rstrip("/")
        relay_base = os.getenv(
            "PERFETTO_RELAY_URL",
            "",
        ).rstrip("/")
        summary += result.to_markdown_row(trace_dir, base_url, relay_base)

    return summary


def save_results_as_pydantic_models(
    results: List, pydantic_result_filename: str, model_path: str
):
    """Save benchmark results as JSON using Pydantic models."""
    json_results = []

    for res in results:
        profile_link_extend = None
        profile_link_decode = None

        if res.profile_link:
            for file in os.listdir(res.profile_link):
                if file.endswith(".trace.json.gz") or file.endswith(".trace.json"):
                    if "extend" in file.lower() or "prefill" in file.lower():
                        profile_link_extend = os.path.join(res.profile_link, file)
                    elif "decode" in file.lower():
                        profile_link_decode = os.path.join(res.profile_link, file)

        benchmark_result = BenchmarkResult(
            model_path=model_path,
            run_name=res.run_name,
            batch_size=res.batch_size,
            input_len=res.input_len,
            output_len=res.output_len,
            latency=res.latency,
            input_throughput=res.input_throughput,
            output_throughput=res.output_throughput,
            overall_throughput=res.overall_throughput,
            last_gen_throughput=res.last_gen_throughput,
            last_ttft=res.last_ttft,
            acc_length=res.acc_length,
            profile_link_extend=profile_link_extend,
            profile_link_decode=profile_link_decode,
        )
        json_results.append(benchmark_result.model_dump())

    with open(pydantic_result_filename, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
