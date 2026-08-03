#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path


RESULT_NAME = re.compile(r"(?P<backend>[^/]+)-isl(?P<input_length>\d+)\.jsonl$")
SUMMARY_FIELDS = (
    "backend",
    "input_length",
    "completed",
    "duration",
    "input_throughput",
    "total_throughput",
    "mean_e2e_latency_ms",
    "p95_e2e_latency_ms",
    "input_throughput_vs_dep",
    "source",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge DEP/DWDP serving JSONL results into one summary."
    )
    parser.add_argument(
        "result_dirs",
        nargs="+",
        type=Path,
        help="Directories containing <backend>-isl<input_length>.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write summary.json and summary.csv here (defaults to the first input).",
    )
    return parser.parse_args()


def read_last_result(path: Path) -> dict:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty benchmark result: {path}")
    return json.loads(lines[-1])


def collect_results(result_dirs: list[Path]) -> list[dict]:
    by_case = {}
    for result_dir in result_dirs:
        for path in sorted(result_dir.glob("*-isl*.jsonl")):
            match = RESULT_NAME.fullmatch(path.name)
            if match is None:
                continue
            result = read_last_result(path)
            row = {
                "backend": match.group("backend"),
                "input_length": int(match.group("input_length")),
                "completed": result["completed"],
                "duration": result["duration"],
                "input_throughput": result["input_throughput"],
                "total_throughput": result["total_throughput"],
                "mean_e2e_latency_ms": result["mean_e2e_latency_ms"],
                "p95_e2e_latency_ms": result["p95_e2e_latency_ms"],
                "source": str(path),
            }
            by_case[(row["backend"], row["input_length"])] = row

    rows = sorted(by_case.values(), key=lambda row: (row["input_length"], row["backend"]))
    dep_throughput = {
        row["input_length"]: row["input_throughput"]
        for row in rows
        if row["backend"] == "dep"
    }
    for row in rows:
        baseline = dep_throughput.get(row["input_length"])
        row["input_throughput_vs_dep"] = (
            row["input_throughput"] / baseline if baseline else None
        )
    return rows


def write_summary(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = collect_results(args.result_dirs)
    if not rows:
        raise SystemExit("no benchmark JSONL results found")
    output_dir = args.output_dir or args.result_dirs[0]
    write_summary(rows, output_dir)
    for row in rows:
        speedup = row["input_throughput_vs_dep"]
        speedup_text = f", vs_dep={speedup:.3f}x" if speedup is not None else ""
        print(
            f"{row['backend']} ISL={row['input_length']}: "
            f"{row['input_throughput']:.2f} input tok/s{speedup_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
