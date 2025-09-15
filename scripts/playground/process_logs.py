#!/usr/bin/env python3
"""
Script to process benchmark logs and extract benchmark results with their corresponding "Finish i=" lines.
"""

import argparse
import re
from typing import List, Optional, Dict, Any


def parse_benchmark_result(
    finish_line: str, benchmark_lines: List[str]
) -> Dict[str, Any]:
    """
    Parse a benchmark result to extract key metrics.

    Args:
        finish_line: The "Finish i=" line
        benchmark_lines: The benchmark result section lines

    Returns:
        Dictionary with parsed metrics
    """
    result = {}

    # Parse finish line: "Finish i=25: batch_size=8, steps=5, topk=2, num_draft_tokens=6, speed=107.51 token/s, step_time=31.88 ms"
    finish_match = re.search(
        r"Finish i=(\d+): batch_size=(\d+), steps=(\d+), topk=(\d+), num_draft_tokens=(\d+), speed=([\d.]+) token/s, step_time=([\d.]+) ms",
        finish_line,
    )
    if finish_match:
        result["i"] = int(finish_match.group(1))
        result["batch_size"] = int(finish_match.group(2))
        result["steps"] = int(finish_match.group(3))
        result["topk"] = int(finish_match.group(4))
        result["num_draft_tokens"] = int(finish_match.group(5))
        result["speed_token_per_s"] = float(finish_match.group(6))
        result["step_time_ms"] = float(finish_match.group(7))

    # Parse benchmark section
    for line in benchmark_lines:
        line = line.strip()
        if line.startswith("Backend:"):
            result["backend"] = line.split(":")[1].strip()
        elif line.startswith("Benchmark duration (s):"):
            result["benchmark_duration_s"] = float(line.split(":")[1].strip())
        elif line.startswith("Output token throughput (tok/s):"):
            result["output_throughput_tok_s"] = float(line.split(":")[1].strip())
        elif line.startswith("Accept length:"):
            result["accept_length"] = float(line.split(":")[1].strip())
        elif line.startswith("Mean E2E Latency (ms):"):
            result["mean_e2e_latency_ms"] = float(line.split(":")[1].strip())
        elif line.startswith("Mean TTFT (ms):"):
            result["mean_ttft_ms"] = float(line.split(":")[1].strip())
        elif line.startswith("Mean ITL (ms):"):
            result["mean_itl_ms"] = float(line.split(":")[1].strip())
        elif line.startswith("Median ITL (ms):"):
            result["median_itl_ms"] = float(line.split(":")[1].strip())
        elif line.startswith("P99 ITL (ms):"):
            result["p99_itl_ms"] = float(line.split(":")[1].strip())

    return result


def process_logs(log_file_path: str, output_file_path: Optional[str] = None) -> None:
    """
    Process the logs file to extract benchmark results.

    Args:
        log_file_path: Path to the input log file
        output_file_path: Optional path to output file. If None, prints to stdout.
    """

    with open(log_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Strip line numbers and clean lines (if they exist)
    cleaned_lines = []
    for line in lines:
        # Remove line numbers at the beginning (format: " 15386|") if they exist
        cleaned_line = re.sub(r"^\s*\d+\|", "", line)
        cleaned_lines.append(cleaned_line)

    results = []
    parsed_results = []
    i = 0

    while i < len(cleaned_lines):
        line = cleaned_lines[i].strip()

        # Look for "Finish i=" line first
        if line.startswith("Finish i="):
            finish_line = line
            # Look for the benchmark section in the previous lines (it comes before the Finish line)
            benchmark_section = None
            # Look backwards for the end of benchmark section first
            for j in range(1, min(50, i + 1)):  # Look back up to 50 lines
                prev_line = cleaned_lines[i - j].strip()
                if prev_line == "==================================================":
                    # Found benchmark section end, now collect backwards to find the start
                    benchmark_section = []
                    k = i - j

                    # Collect lines backwards until we find the start marker
                    while k >= 0:
                        current_line = cleaned_lines[k].strip()
                        benchmark_section.insert(0, current_line)  # Insert at beginning

                        if (
                            current_line
                            == "============ Serving Benchmark Result ============"
                        ):
                            break
                        k -= 1

                    break

            if benchmark_section:
                # Create the result with "Finish i=" line first, then benchmark section
                result = [finish_line] + benchmark_section
                results.append(result)

                # Parse the result for TSV output
                parsed_result = parse_benchmark_result(finish_line, benchmark_section)
                parsed_results.append(parsed_result)

        i += 1

    # Output results
    if output_file_path:
        # Write full results to main output file
        with open(output_file_path, "w", encoding="utf-8") as f:
            for idx, result in enumerate(results):
                if idx > 0:
                    f.write("\n\n")  # Add spacing between results
                f.write("\n".join(result))
                f.write("\n")

        # Write TSV summary to summary file
        summary_file_path = (
            output_file_path.replace(".txt", "_summary.tsv")
            if output_file_path.endswith(".txt")
            else f"summary_{output_file_path}.tsv"
        )
        with open(summary_file_path, "w", encoding="utf-8") as f:
            # Write TSV header
            headers = [
                "i",
                "batch_size",
                "steps",
                "topk",
                "num_draft_tokens",
                "backend",
                "benchmark_duration_s",
                "speed_token_per_s",
                "step_time_ms",
                "output_throughput_tok_s",
                "accept_length",
                "mean_e2e_latency_ms",
                "mean_ttft_ms",
                "mean_itl_ms",
                "median_itl_ms",
                "p99_itl_ms",
            ]
            f.write("\t".join(headers) + "\n")

            # Write data rows
            for parsed_result in parsed_results:
                row = []
                for header in headers:
                    value = parsed_result.get(header, "")
                    row.append(str(value))
                f.write("\t".join(row) + "\n")

        print(f"Results written to {output_file_path}")
        print(f"TSV summary written to {summary_file_path}")
    else:
        # Print to stdout
        for idx, result in enumerate(results):
            if idx > 0:
                print("\n")  # Add spacing between results
            print("\n".join(result))


def main():
    parser = argparse.ArgumentParser(
        description="Process benchmark logs and extract results with their Finish i= lines"
    )
    parser.add_argument("log_file", help="Path to the input log file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (optional, prints to stdout if not provided)",
    )

    args = parser.parse_args()

    try:
        process_logs(args.log_file, args.output)
        if args.output:
            print(f"Results written to {args.output}")
    except FileNotFoundError:
        print(f"Error: File '{args.log_file}' not found")
    except Exception as e:
        print(f"Error processing logs: {e}")


if __name__ == "__main__":
    main()
