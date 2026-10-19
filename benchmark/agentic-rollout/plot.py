# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "prometheus-client"]
# ///
"""Plot one or more synthetic-session result directories; no running server needed."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from metrics import counter_rate, metric_values


def read_rows(path):
    with path.open() as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def percentile(values, q):
    values = sorted(v for v in values if v is not None)
    return values[max(0, math.ceil(q * len(values)) - 1)] if values else None


def cache_hit(rows, source):
    values = [
        (
            r["meta_info"].get("cached_tokens")
            if source == "total"
            else (r["meta_info"].get("cached_tokens_details") or {}).get(source)
        )
        for r in rows
    ]
    total = sum(r["meta_info"]["prompt_tokens"] for r in rows)
    return sum(values) / total if total and all(v is not None for v in values) else None


def summarize(rows):
    return {
        "requests": len(rows),
        "ttft_p50_s": percentile([r["ttft_s"] for r in rows], 0.5),
        "ttft_p95_s": percentile([r["ttft_s"] for r in rows], 0.95),
        "avg_token_time_s": (
            sum(r["avg_token_time_s"] for r in rows) / len(rows)
            if rows and all(r["avg_token_time_s"] is not None for r in rows)
            else None
        ),
        "context_tokens": (
            sum(r["context_tokens"] for r in rows) / len(rows) if rows else None
        ),
        **{
            f"{source}_hit": cache_hit(rows, source)
            for source in ("total", "device", "host")
        },
    }


def export_csv(path, rows):
    if not rows:
        return
    with path.open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(directory, window):
    manifest = json.loads((directory / "manifest.json").read_text())
    all_rows = list(read_rows(directory / "requests.jsonl"))
    rows = [r for r in all_rows if "error" not in r]
    start, finish = manifest["started_at"], manifest["finished_at"]
    windows, turns, rates = defaultdict(list), defaultdict(list), defaultdict(int)
    for row in rows:
        windows[int((row["first_token_at"] - start) // window)].append(row)
        turns[row["turn"]].append(row)
        # Attribute received token increments to their actual arrival windows.
        previous_count = 0
        for offset, count in row["events"]:
            rates[int((row["submitted_at"] + offset - start) // window)] += (
                count - previous_count
            )
            previous_count = count
    by_time = []
    for index in range(int((finish - start) // window) + 1):
        duration = min(window, finish - start - index * window)
        if duration <= 0:
            continue
        by_time.append(
            {
                "elapsed_s": index * window,
                **summarize(windows[index]),
                "output_tokens_s": rates[index] / duration,
            }
        )
    by_turn = [{"turn": t, **summarize(group)} for t, group in sorted(turns.items())]
    export_csv(directory / "windows.csv", by_time)
    export_csv(directory / "turns.csv", by_turn)
    summary = {
        "status": manifest["status"],
        "failed_requests": len(all_rows) - len(rows),
        **summarize(rows),
        "output_tokens_s": sum(rates.values()) / (finish - start),
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return manifest, by_time, by_turn


def plot(directories, output, window):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 2, figsize=(15, 15), constrained_layout=True)
    axes = axes.ravel()
    titles = [
        "TTFT (seconds)",
        "Received output tokens / second",
        "Average time per output token (s)",
        "Token-weighted cache hit",
        "Running / queued requests (per exporter)",
        "Cache occupancy (per exporter)",
        "Cache transfer rates (tokens/s)",
        "Mean context tokens",
    ]
    turn_figure, turn_axes = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True
    )
    configs = {}
    common = Path(os.path.commonpath([p.resolve().parent for p in directories]))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for run_index, directory in enumerate(directories):
        color = colors[run_index % len(colors)]
        manifest, by_time, by_turn = analyze(directory, window)
        label = str(directory.resolve().relative_to(common)) + (
            " [FAILED]" if manifest["status"] != "completed" else ""
        )
        configs[label] = {
            "arguments": manifest["arguments"],
            "server_info": manifest["server_info"],
        }
        x = [r["elapsed_s"] for r in by_time]
        for axis, key, suffix in [
            (0, "ttft_p50_s", "p50"),
            (0, "ttft_p95_s", "p95"),
            (1, "output_tokens_s", ""),
            (2, "avg_token_time_s", ""),
            (3, "total_hit", "total"),
            (3, "device_hit", "GPU"),
            (3, "host_hit", "CPU"),
            (7, "context_tokens", ""),
        ]:
            values = [r[key] for r in by_time]
            if any(v is not None for v in values):
                axes[axis].plot(
                    x,
                    values,
                    color=color,
                    linestyle={"p50": "--", "GPU": "--", "CPU": ":"}.get(suffix, "-"),
                    marker=".",
                    label=f"{label} {suffix}",
                )
        for axis, key in [(0, "ttft_p95_s"), (1, "context_tokens")]:
            turn_axes[axis].plot(
                [r["turn"] for r in by_turn],
                [r[key] for r in by_turn],
                color=color,
                marker=".",
                label=label,
            )
        series = defaultdict(list)
        previous = {}
        for row in read_rows(directory / "metrics.jsonl"):
            url = row["url"]
            if "error" in row:
                previous.pop(url, None)
                continue
            current = metric_values(row["text"])
            elapsed = row["timestamp"] - manifest["started_at"]
            for name in (
                "sglang:num_running_reqs",
                "sglang:num_queue_reqs",
                "sglang:token_usage",
            ):
                values = [
                    value for (metric, _), value in current.items() if metric == name
                ]
                if values:
                    value = max(values) if "usage" in name else sum(values)
                    series[(5 if "usage" in name else 4, url, name)].append(
                        (elapsed, value)
                    )
            host = []
            for (name, labels), used in current.items():
                if name == "sglang:hicache_host_used_tokens":
                    total = current.get(("sglang:hicache_host_total_tokens", labels))
                    if total is not None and total > 0:
                        host.append(used / total)
            if host:
                series[(5, url, "host_usage")].append((elapsed, max(host)))
            if url in previous:
                stamp, before = previous[url]
                names = {
                    name
                    for name, _ in current
                    if name.endswith("_total")
                    and any(word in name for word in ("restore", "backup", "load"))
                    and "token" in name
                }
                for name in sorted(names):
                    value = counter_rate(
                        before, current, name, row["timestamp"] - stamp
                    )
                    series[(6, url, name)].append((elapsed, value))
            previous[url] = row["timestamp"], current
        urls = {
            url: index for index, url in enumerate(sorted({key[1] for key in series}))
        }
        for (axis, url, name), points in series.items():
            if any(v is not None for _, v in points):
                axes[axis].plot(
                    [t for t, _ in points],
                    [v for _, v in points],
                    color=color,
                    linestyle=(
                        "--"
                        if name in ("sglang:num_queue_reqs", "host_usage")
                        or "backup" in name
                        else "-"
                    ),
                    marker=".",
                    label=f"{label} exporter {urls[url]} {name.removeprefix('sglang:')}",
                )
    for axis, title in zip(axes, titles):
        axis.set(title=title, xlabel="Elapsed seconds")
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(fontsize=6)
        else:
            axis.text(0.5, 0.5, "Unavailable", ha="center", transform=axis.transAxes)
    for axis, title in zip(turn_axes, ("TTFT p95 (seconds)", "Mean context tokens")):
        axis.set(title=title, xlabel="Turn")
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(fontsize=7)
    figure.savefig(output, dpi=160)
    turn_figure.savefig(output.with_name(output.stem + "-turns.png"), dpi=160)
    output.with_suffix(".configs.json").write_text(json.dumps(configs, indent=2) + "\n")
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directories", nargs="+", type=Path, help="Run result directories to compare"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synthetic-sessions.png"),
        help="PNG path; also writes a turn plot and configuration sidecar",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=30,
        help="Seconds per time window for request statistics",
    )
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be positive")
    plot(args.directories, args.output, args.window)
