#!/usr/bin/env python3
"""Summarize Ascend NPU profiler CSV exports into a markdown report."""

import argparse
import csv
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def _pick_name(row: dict[str, str]) -> str:
    for key in ("Op Name", "Operator Name", "Name", "Kernel Name", "op_name"):
        if key in row and row[key]:
            return row[key]
    return next(iter(row.values()), "unknown")


def _pick_time(row: dict[str, str]) -> float:
    for key in (
        "Total Time (us)",
        "Total Time (ms)",
        "Duration (us)",
        "Duration (ms)",
        "Device Duration (us)",
        "Device Duration (ms)",
    ):
        if key in row and row[key]:
            val = float(str(row[key]).replace(",", ""))
            if "ms" in key.lower():
                return val
            return val / 1000.0
    return 0.0


def _top_rows(rows: list[dict[str, str]], top: int) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for row in rows:
        t = _pick_time(row)
        if t <= 0:
            continue
        ranked.append((_pick_name(row), t))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top]


def _render_table(title: str, ranked: list[tuple[str, float]]) -> str:
    if not ranked:
        return f"## {title}\n\n_No data found._\n"
    total = sum(t for _, t in ranked) or 1.0
    lines = [
        f"## {title}",
        "",
        "| Rank | Name | Time (ms) | Share (%) |",
        "| ---: | --- | ---: | ---: |",
    ]
    for i, (name, ms) in enumerate(ranked, start=1):
        share = 100.0 * ms / total
        lines.append(f"| {i} | {name} | {ms:.3f} | {share:.1f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace-dir", required=True, help="ASCEND_PROFILER_OUTPUT directory")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--markdown-out", required=True)
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    op_rows = _read_csv(trace_dir / "operator_details.csv")
    kernel_rows = _read_csv(trace_dir / "kernel_details.csv")

    op_ranked = _top_rows(op_rows, args.top)
    kernel_ranked = _top_rows(kernel_rows, min(args.top, 10))

    parts = [
        "# SGLang NPU Trace Summary",
        "",
        f"Trace directory: `{trace_dir}`",
        "",
        _render_table("Top Operators", op_ranked),
        _render_table("Top Kernels", kernel_ranked),
    ]

    out = Path(args.markdown_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
