"""Render the records of the MI35x fixed-sequence benchmark as tables.

A record is one server cell of one recipe, as ``fixed_seq_common`` writes it:
its GSM8K gate and the ``bench_serving`` result of each (ISL, OSL, concurrency)
point it ran. This module only reads records, so it stays standard library and
runs on a bare runner:

    python3 fixed_seq_report.py --markdown report.md --csv points.csv rec1.json ...
"""

import argparse
import csv
import json
import sys
from pathlib import Path

CSV_FIELDS = [
    "recipe",
    "model",
    "model_revision",
    "cell",
    "gpus",
    "isl",
    "osl",
    "conc",
    "completed",
    "num_prompts",
    "total_tok_s_per_gpu",
    "output_tok_s_per_gpu",
    "total_tok_s",
    "output_tok_s",
    "request_s",
    "interactivity_tok_s_user",
    "median_ttft_ms",
    "p99_ttft_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "median_itl_ms",
    "median_e2e_latency_ms",
    "accept_length",
    "error",
]


def _workload(isl: int, osl: int) -> str:
    def short(n: int) -> str:
        return f"{n // 1024}K" if n % 1024 == 0 else f"{n / 1000:g}K"

    return f"{short(isl)}/{osl if osl < 1000 else short(osl)}"


def point_row(record: dict, point: dict) -> dict:
    """One CSV row for ``point`` of ``record``; metrics are None when it failed."""
    result = point.get("result") or {}
    gpus = record["gpus"]

    def per_gpu(key):
        value = result.get(key)
        return value / gpus if isinstance(value, (int, float)) else None

    tpot = result.get("median_tpot_ms")
    return {
        "recipe": record["recipe"],
        "model": record["model"],
        "model_revision": record.get("model_revision") or "",
        "cell": record["cell"],
        "gpus": gpus,
        "isl": point["isl"],
        "osl": point["osl"],
        "conc": point["conc"],
        "completed": result.get("completed"),
        "num_prompts": point["num_prompts"],
        "total_tok_s_per_gpu": per_gpu("total_throughput"),
        "output_tok_s_per_gpu": per_gpu("output_throughput"),
        "total_tok_s": result.get("total_throughput"),
        "output_tok_s": result.get("output_throughput"),
        "request_s": result.get("request_throughput"),
        "interactivity_tok_s_user": 1000 / tpot if tpot else None,
        "median_ttft_ms": result.get("median_ttft_ms"),
        "p99_ttft_ms": result.get("p99_ttft_ms"),
        "median_tpot_ms": tpot,
        "p99_tpot_ms": result.get("p99_tpot_ms"),
        "median_itl_ms": result.get("median_itl_ms"),
        "median_e2e_latency_ms": result.get("median_e2e_latency_ms"),
        "accept_length": result.get("accept_length"),
        "error": point.get("error") or "",
    }


def _num(value, fmt: str, scale: float = 1) -> str:
    if not isinstance(value, (int, float)):
        return "–"
    return format(value * scale, fmt)


POINT_HEADER = (
    "| conc | server | total tok/s/GPU | output tok/s/GPU | tok/s/user "
    "| TTFT p50 / p99 (s) | TPOT p50 / p99 (ms) | E2E p50 (s) | accept len "
    "| done |\n"
    "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
)


def _point_line(row: dict) -> str:
    if row["error"]:
        return f"| {row['conc']} | {row['cell']} | ❌ {row['error']} | | | | | | | |\n"
    return (
        f"| {row['conc']} | {row['cell']} "
        f"| {_num(row['total_tok_s_per_gpu'], ',.0f')} "
        f"| {_num(row['output_tok_s_per_gpu'], ',.0f')} "
        f"| {_num(row['interactivity_tok_s_user'], '.1f')} "
        f"| {_num(row['median_ttft_ms'], '.2f', 1e-3)} / "
        f"{_num(row['p99_ttft_ms'], '.2f', 1e-3)} "
        f"| {_num(row['median_tpot_ms'], '.1f')} / {_num(row['p99_tpot_ms'], '.1f')} "
        f"| {_num(row['median_e2e_latency_ms'], '.1f', 1e-3)} "
        f"| {_num(row['accept_length'], '.2f') if row['accept_length'] else '–'} "
        f"| {_done(row)} |\n"
    )


def _done(row: dict) -> str:
    completed = row["completed"]
    if completed is None:
        return f"–/{row['num_prompts']}"
    if completed < row["num_prompts"]:
        return f"⚠️ {completed}/{row['num_prompts']}"
    return f"{completed}/{row['num_prompts']}"


def _server_line(record: dict) -> str:
    gsm8k = record.get("gsm8k") or {}
    if record.get("error"):
        status = f"❌ {record['error']}"
    else:
        mark = "✅" if gsm8k.get("passed") else "❌"
        status = f"{mark} {_num(gsm8k.get('accuracy'), '.3f')}"
    return (
        f"| {record['cell']} | {status} | {_num(gsm8k.get('threshold'), '.2f')} "
        f"| `{record.get('image') or '?'}` "
        f"| `{(record.get('sglang_commit') or '?')[:10]}` |\n"
    )


def render_markdown(records: list) -> str:
    """Per model: the GSM8K gate of each server, then one table per workload
    with every concurrency, whichever server cell measured it."""
    if not records:
        return "No fixed-sequence results.\n"
    out = [
        "tok/s/user = 1000 / median TPOT. Per GPU = divided by the server's TP size.\n"
    ]
    by_recipe: dict = {}
    for record in records:
        by_recipe.setdefault(record["recipe"], []).append(record)
    for recipe, recipe_records in sorted(by_recipe.items()):
        head = recipe_records[0]
        revision = next(
            (r["model_revision"] for r in recipe_records if r.get("model_revision")),
            "?",
        )
        out.append(
            f"\n## {head['title']}\n\n`{head['model']}` @ `{revision[:10]}`\n\n"
            "| server | GSM8K | threshold | image | sglang |\n"
            "| --- | ---: | ---: | --- | --- |\n"
        )
        for record in sorted(recipe_records, key=lambda r: r["cell"]):
            out.append(_server_line(record))
        rows = [
            point_row(record, point)
            for record in recipe_records
            for point in record.get("points", [])
        ]
        workloads = sorted({(row["isl"], row["osl"]) for row in rows})
        for isl, osl in workloads:
            out.append(f"\n### {_workload(isl, osl)} (ISL {isl}, OSL {osl})\n\n")
            out.append(POINT_HEADER)
            for row in sorted(
                (r for r in rows if (r["isl"], r["osl"]) == (isl, osl)),
                key=lambda r: (r["conc"], r["cell"]),
            ):
                out.append(_point_line(row))
    return "".join(out)


def write_csv(records: list, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            for point in record.get("points", []):
                writer.writerow(point_row(record, point))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", nargs="*", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    records = [json.loads(path.read_text()) for path in args.records]
    markdown = render_markdown(records)
    if args.markdown:
        args.markdown.write_text(markdown)
    else:
        sys.stdout.write(markdown)
    if args.csv:
        write_csv(records, args.csv)


if __name__ == "__main__":
    main()
