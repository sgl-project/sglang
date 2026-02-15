#!/usr/bin/env python3
"""Merge per-partition metrics into a consolidated metrics file.

This script reads all per-partition metric JSON files and consolidates them
into a single JSON file with run-level metadata.

Usage:
    python3 scripts/ci/merge_metrics.py \
        --input-dir metrics/ \
        --output consolidated-metrics-12345678.json \
        --run-id 12345678 \
        --commit-sha abc123def456
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone


def find_partition_files(input_dir: str) -> list[str]:
    """Find all partition metric files in the input directory."""
    pattern = os.path.join(input_dir, "**/metrics-*.json")
    return glob.glob(pattern, recursive=True)


def load_partition_metrics(filepath: str) -> dict | None:
    """Load a partition metrics file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Failed to load {filepath}: {e}")
        return None


def merge_metrics(
    input_dir: str,
    output_file: str,
    run_id: str,
    commit_sha: str,
    branch: str | None = None,
) -> bool:
    """Merge all partition metrics into a consolidated file."""
    run_date = datetime.now(timezone.utc).isoformat()

    # Find all partition files
    partition_files = find_partition_files(input_dir)
    print(f"Found {len(partition_files)} partition file(s)")

    all_results = []
    if not partition_files:
        print("No partition metrics files found")
    else:
        # Load all partition files
        for filepath in sorted(partition_files):
            print(f"  Reading: {filepath}")
            metrics = load_partition_metrics(filepath)
            if metrics and "results" in metrics:
                all_results.extend(metrics["results"])
        print(f"Total results collected: {len(all_results)}")

    # Create consolidated structure
    consolidated = {
        "run_id": run_id,
        "run_date": run_date,
        "commit_sha": commit_sha,
        "branch": branch,
        "results": all_results,
    }

    # Ensure output directory exists and write output
    try:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(consolidated, f, indent=2)

        if not partition_files:
            print(f"Created empty consolidated file: {output_file}")
        else:
            print(f"Saved consolidated metrics to: {output_file}")
        return True
    except OSError as e:
        print(f"Error writing consolidated file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-partition metrics into consolidated file"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing partition metric files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output file path for consolidated metrics JSON",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="GitHub Actions run ID",
    )
    parser.add_argument(
        "--commit-sha",
        required=True,
        help="Git commit SHA",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Git branch name (optional)",
    )

    args = parser.parse_args()

    success = merge_metrics(
        input_dir=args.input_dir,
        output_file=args.output,
        run_id=args.run_id,
        commit_sha=args.commit_sha,
        branch=args.branch,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
