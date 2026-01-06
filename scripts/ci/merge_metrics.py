#!/usr/bin/env python3
"""
Merge multiple test metrics JSONL files into a single output file.

This script searches for files matching ${base_path}.*.jsonl pattern,
parses and validates each line as JSON, and merges them into a single
output file. Invalid lines are skipped with warnings to stderr.

Usage:
    python3 merge_metrics.py <base_path> <output_file>

Example:
    python3 merge_metrics.py /tmp/test_metrics /tmp/merged_metrics.jsonl

This will search for /tmp/test_metrics.*.jsonl files and merge them.
"""

import glob
import json
import sys
from pathlib import Path


def merge_metrics(base_path: str, output_file: str) -> int:
    """
    Merge multiple metrics JSONL files into a single output.

    Args:
        base_path: Base path pattern for input files (without .*.jsonl suffix)
        output_file: Path to the output merged JSONL file

    Returns:
        0 on success, 1 on error
    """
    # Search for all matching JSONL files
    pattern = f"{base_path}.*.jsonl"
    input_files = sorted(glob.glob(pattern))

    if not input_files:
        # No input files found - create empty output file
        print(
            f"No input files found matching pattern: {pattern}",
            file=sys.stderr,
        )
        print(f"Creating empty output file: {output_file}", file=sys.stderr)
        Path(output_file).touch()
        return 0

    print(
        f"Found {len(input_files)} metrics file(s) to merge",
        file=sys.stderr,
    )

    total_lines = 0
    valid_lines = 0
    invalid_lines = 0

    try:
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as out_f:
            for input_file in input_files:
                print(f"Processing: {input_file}", file=sys.stderr)
                try:
                    with open(input_file, "r", encoding="utf-8") as in_f:
                        for line_num, line in enumerate(in_f, 1):
                            total_lines += 1
                            line = line.strip()
                            if not line:
                                # Skip empty lines
                                continue

                            try:
                                # Validate JSON by parsing
                                json.loads(line)
                                # Write valid line to output
                                out_f.write(line + "\n")
                                valid_lines += 1
                            except json.JSONDecodeError as e:
                                invalid_lines += 1
                                print(
                                    f"Warning: Invalid JSON in {input_file}:{line_num}: {e}",
                                    file=sys.stderr,
                                )
                                print(f"  Skipping line: {line[:100]}", file=sys.stderr)
                except IOError as e:
                    print(
                        f"Warning: Failed to read {input_file}: {e}",
                        file=sys.stderr,
                    )
                    continue

        print(
            f"Merge complete: {valid_lines} valid lines written to {output_file}",
            file=sys.stderr,
        )
        if invalid_lines > 0:
            print(
                f"Warning: Skipped {invalid_lines} invalid line(s)",
                file=sys.stderr,
            )

        return 0

    except IOError as e:
        print(f"Error: Failed to write output file {output_file}: {e}", file=sys.stderr)
        return 1


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python3 merge_metrics.py <base_path> <output_file>",
            file=sys.stderr,
        )
        print(
            "\nExample: python3 merge_metrics.py /tmp/test_metrics /tmp/merged.jsonl",
            file=sys.stderr,
        )
        return 1

    base_path = sys.argv[1]
    output_file = sys.argv[2]

    return merge_metrics(base_path, output_file)


if __name__ == "__main__":
    sys.exit(main())
