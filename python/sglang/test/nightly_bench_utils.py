import json
import os
from typing import List, Optional

from pydantic import BaseModel

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
    server_args: Optional[List[str]] = None

    def to_markdown_row(self) -> str:
        """Convert this benchmark result to a markdown table row."""

        hourly_cost_per_gpu = 2  # $2/hour for one H100
        hourly_cost = hourly_cost_per_gpu * 1  # Assuming tp_size = 1 for simplicity
        input_util = 0.7
        accept_length = round(self.acc_length, 2) if self.acc_length > 0 else "n/a"
        itl = 1 / (self.output_throughput / self.batch_size) * 1000
        input_cost = 1e6 / (self.input_throughput * input_util) / 3600 * hourly_cost
        output_cost = 1e6 / self.output_throughput / 3600 * hourly_cost

        return f"| {self.batch_size} | {self.input_len} | {self.latency:.2f} | {self.input_throughput:.2f} | {self.output_throughput:.2f} | {accept_length} | {itl:.2f} | {input_cost:.2f} | {output_cost:.2f} |\n"


def generate_markdown_report(
    results: List[BenchmarkResult], variant: Optional[str] = None
) -> str:
    """Generate a markdown report from a list of BenchmarkResult object from a single run."""
    # Build model header with run_name if it's not "default"
    model_header = results[0].model_path
    if results[0].run_name and results[0].run_name != "default":
        model_header += f" ({results[0].run_name})"

    # Include GPU config in model header if available
    gpu_config = os.getenv("GPU_CONFIG", "")
    if gpu_config:
        model_header += f" [{gpu_config}]"

    if variant:
        model_header += f" ({variant})"

    summary = f"### {model_header}\n"

    summary += "| batch size | input len | latency (s) | input throughput (tok/s)  | output throughput (tok/s) | acc length | ITL (ms) | input cost ($/1M) | output cost ($/1M) |\n"
    summary += "| ---------- | --------- | ----------- | ------------------------- | ------------------------- | ---------- | -------- | ----------------- | ------------------ |\n"

    # all results should share the same isl & osl
    for result in results:
        summary += result.to_markdown_row()

    return summary


def generate_simple_markdown_report(
    results: List[BenchmarkResult], default_gpu_config: str = ""
) -> str:
    """Generate a markdown report without the H100-priced cost columns.

    Drops the leading result when it is a warmup run, which the caller requests
    by repeating the first batch size.
    """
    model_header = results[0].model_path
    if results[0].run_name and results[0].run_name != "default":
        model_header += f" ({results[0].run_name})"

    gpu_config = os.getenv("GPU_CONFIG", default_gpu_config)
    if gpu_config:
        model_header += f" [{gpu_config}]"

    summary = f"### {model_header}\n"
    summary += "| batch size | input len | latency (s) | input throughput (tok/s) | output throughput (tok/s) | ITL (ms) |\n"
    summary += "| ---------- | --------- | ----------- | ------------------------ | ------------------------- | -------- |\n"

    report_results = (
        results[1:]
        if len(results) > 1 and results[0].batch_size == results[1].batch_size
        else results
    )

    for result in report_results:
        itl = (
            1 / (result.output_throughput / result.batch_size) * 1000
            if result.output_throughput > 0
            else 0
        )
        summary += (
            f"| {result.batch_size} | {result.input_len} | {result.latency:.2f} | "
            f"{result.input_throughput:.2f} | {result.output_throughput:.2f} | {itl:.2f} |\n"
        )

    return summary


def save_results_as_pydantic_models(
    results: List,
    pydantic_result_filename: str,
    model_path: str,
    server_args: Optional[List[str]] = None,
):
    """Save benchmark results as JSON using Pydantic models."""
    json_results = []

    for res in results:
        profile_link_extend = None
        profile_link_decode = None

        if res.profile_link:
            # Collect all trace files, preferring TP-0 to match upload behavior
            # (only TP-0 traces are published to avoid duplicates)
            extend_files = []
            decode_files = []
            for file in os.listdir(res.profile_link):
                if file.endswith(".trace.json.gz") or file.endswith(".trace.json"):
                    if "extend" in file.lower() or "prefill" in file.lower():
                        extend_files.append(file)
                    elif "decode" in file.lower():
                        decode_files.append(file)

            # Sort to prefer TP-0 files (TP-0 < TP-1 < TP-2... alphabetically)
            extend_files.sort()
            decode_files.sort()

            if extend_files:
                profile_link_extend = os.path.join(res.profile_link, extend_files[0])
            if decode_files:
                profile_link_decode = os.path.join(res.profile_link, decode_files[0])

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
            server_args=server_args,
        )
        json_results.append(benchmark_result.model_dump())

    with open(pydantic_result_filename, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
