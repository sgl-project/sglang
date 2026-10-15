# /// script
# requires-python = ">=3.11"
# dependencies = ["prometheus-client"]
# ///
"""Build an offline Rollout Simulator and Explorer HTML from a recording."""

import argparse
import json
import math
import statistics
from pathlib import Path

from metrics import (
    counter_rate,
)
from metrics import gauge as metric_gauge
from metrics import (
    hit_percentages,
    metric_values,
    occupancy,
    values,
)


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Interrupted writes remain visible as recording errors.
                rows.append({"error": "Incomplete JSONL record"})
    return rows


def panels_for(reqs, scrapes, origin, duration):
    series = []
    parsed = []
    for row in sorted(scrapes, key=lambda r: r["timestamp"]):
        try:
            samples = metric_values(row.get("text", "")) if not row.get("error") else {}
        except ValueError:
            samples = {}
        parsed.append((row["timestamp"] - origin, samples))

    def gauge(name, mult=1):
        return [
            [t, None if (v := metric_gauge(ss, name)) is None else v * mult]
            for t, ss in parsed
        ]

    def rate(name):
        return [
            [t1, counter_rate(s0, s1, "sglang:" + name, t1 - t0)]
            for (t0, s0), (t1, s1) in zip(parsed, parsed[1:])
        ]

    cache_hits = [[], [], []]
    for (t0, s0), (t1, s1) in zip(parsed, parsed[1:]):
        for points, value in zip(
            cache_hits, hit_percentages(s0, s1) if t1 > t0 else [None] * 3
        ):
            points.append([t1, value])
    l1_occupied, l1_capacity, l2_occupied, l2_capacity = [], [], [], []
    l1_usage, l2_usage = [], []
    for t, samples in parsed:
        active, evictable, capacity, usage1 = occupancy(
            samples, ("kv_used_tokens", "kv_evictable_tokens", "max_total_num_tokens")
        )
        host, host_capacity, usage2 = occupancy(
            samples, ("hicache_host_used_tokens", "hicache_host_total_tokens")
        )
        l1_occupied.append([t, active + evictable if active is not None else None])
        l1_capacity.append([t, metric_gauge(samples, "max_total_num_tokens")])
        l2_occupied.append([t, metric_gauge(samples, "hicache_host_used_tokens")])
        l2_capacity.append([t, metric_gauge(samples, "hicache_host_total_tokens")])
        l1_usage.append([t, usage1])
        l2_usage.append([t, usage2])

    bins = [(i, min(i + 2, duration)) for i in range(0, math.ceil(duration), 2)]

    def distribution(observations):
        buckets = [[] for _ in bins]
        for elapsed, value in observations:
            if 0 <= elapsed < duration:
                buckets[int(elapsed // 2)].append(value)
        groups = [
            ((a + b) / 2, sorted(values)) for (a, b), values in zip(bins, buckets)
        ]
        return [
            {
                "name": name,
                "points": [
                    [
                        t,
                        (
                            (
                                statistics.mean(v)
                                if q is None
                                else v[math.ceil(q * len(v)) - 1]
                            )
                            if v
                            else None
                        ),
                    ]
                    for t, v in groups
                ],
            }
            for name, q in (("avg", None), ("p50", 0.5), ("p90", 0.9), ("p99", 0.99))
        ]

    def dist(key, scale=1):
        return distribution(
            [
                (
                    r.get(
                        "completed_at",
                        r.get(
                            "failed_at",
                            r.get("first_token_at", r.get("submitted_at", origin)),
                        ),
                    )
                    - origin,
                    r[key] * scale,
                )
                for r in reqs
                if r.get(key) is not None
            ]
        )

    def add(title, unit, source, items):
        series.append({"title": title, "unit": unit, "source": source, "series": items})

    def one(name, points, **style):
        return {"name": name, "points": points, **style}

    add(
        "QPS",
        "req/s",
        "Successful client completions / 2-second window",
        [
            one(
                "requests",
                [
                    [
                        (a + b) / 2,
                        sum(
                            a <= r.get("completed_at", float("inf")) - origin < b
                            and not r.get("error")
                            for r in reqs
                        )
                        / (b - a),
                    ]
                    for a, b in bins
                ],
            )
        ],
    )
    add(
        "Running And Queued Requests",
        "requests",
        "Server num_running_reqs / num_queue_reqs gauges",
        [
            one("running", gauge("num_running_reqs")),
            one("queued", gauge("num_queue_reqs")),
        ],
    )
    add(
        "Input Length",
        "tokens",
        "Client context, grouped by completion/failure time",
        dist("context_tokens"),
    )
    reqs = [
        {**r, "output_tokens": (r.get("meta_info") or {}).get("completion_tokens")}
        for r in reqs
    ]
    add(
        "Output Length",
        "tokens",
        "Observed output token counts; includes partial requests",
        dist("output_tokens"),
    )
    add(
        "TTFT",
        "ms",
        "Client submit to first output; includes partial requests with measured TTFT",
        dist("ttft_s", 1000),
    )
    # Report the distribution of observed per-token stream intervals, not GPU timing.
    itl = []
    for r in reqs:
        for (t0, n0), (t1, n1) in zip(r.get("events", []), r.get("events", [])[1:]):
            if n1 > n0:
                itl.append(
                    (r["submitted_at"] + t1 - origin, (t1 - t0) / (n1 - n0) * 1000)
                )
    add(
        "ITL",
        "ms",
        "Client stream intervals / received token increments; not kernel timing",
        distribution(itl),
    )
    add(
        "Prompt / Input Throughput",
        "tokens/s",
        "Server counter deltas / scrape interval; cached sums all tiers",
        [
            one("prompt", rate("prompt_tokens_total")),
            one("cached", rate("cached_tokens_total")),
        ],
    )
    add(
        "Output Throughput",
        "tokens/s",
        "Server generation counter deltas / scrape interval",
        [one("total", rate("generation_tokens_total"))],
    )
    add(
        "Cache Hit Rate",
        "%",
        "Counter deltas from prefill_effective_tokens_total; each tier / all prompt tokens. Total includes storage hits.",
        [
            one("L1 GPU", cache_hits[0], color=0),
            one("L2 CPU", cache_hits[1], color=1),
            one("Total", cache_hits[2], color=2, dashed=True),
        ],
    )
    add(
        "E2E Request Latency",
        "ms",
        "Client submit to completed stream",
        dist("latency_s", 1000),
    )
    add(
        "KV Usage",
        "%",
        "L1: (active + evictable) / GPU capacity. L2: host used / host capacity.",
        [one("L1 GPU", l1_usage, color=0), one("L2 CPU", l2_usage, color=1)],
    )
    add(
        "KV Tokens",
        "tokens",
        "L1 occupied = active + evictable GPU tokens. L2 occupied = host used tokens. Dashed lines show capacity.",
        [
            one("L1 occupied", l1_occupied, color=0),
            one("L1 capacity", l1_capacity, color=0, dashed=True),
            one("L2 occupied", l2_occupied, color=1),
            one("L2 capacity", l2_capacity, color=1, dashed=True),
        ],
    )
    add(
        "Mamba Usage",
        "%",
        "Maximum server mamba_usage across reported ranks",
        [
            one(
                "usage",
                [
                    [
                        t,
                        (
                            None
                            if not (v := list(values(ss, "mamba_usage").values()))
                            else max(v) * 100
                        ),
                    ]
                    for t, ss in parsed
                ],
            )
        ],
    )
    add(
        "Mamba Tokens",
        "tokens",
        "Server Mamba token gauges",
        [
            one("used", gauge("mamba_used_tokens")),
            one("available", gauge("mamba_available_tokens")),
            one("evictable", gauge("mamba_evictable_tokens")),
        ],
    )
    add(
        "Spec Acceptance Length",
        "tokens",
        "Unweighted mean spec_accept_length across reported ranks",
        [
            one(
                "accepted",
                [
                    [
                        t,
                        (
                            statistics.mean(v)
                            if (v := list(values(ss, "spec_accept_length").values()))
                            else None
                        ),
                    ]
                    for t, ss in parsed
                ],
            )
        ],
    )
    return series


def load_run(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    requests = read_rows(directory / "requests.jsonl")
    scrapes = read_rows(directory / "metrics.jsonl")
    sessions = read_rows(directory / "sessions.jsonl")
    origin = manifest["started_at"]
    stamps = [origin] + [
        r[k]
        for r in requests + scrapes + sessions
        for k in ("completed_at", "failed_at", "submitted_at", "timestamp")
        if isinstance(r.get(k), (int, float))
    ]
    stamps.extend(
        r["submitted_at"] + event[0]
        for r in requests
        if "submitted_at" in r
        for event in r.get("events", [])
    )
    duration = max(0.001, manifest.get("finished_at", max(stamps)) - origin)
    info = manifest.get("server_info", {})
    args = manifest.get("arguments", {})
    server = info.get("server_args", info)
    rows = []
    for r in requests:
        meta = r.get("meta_info") or {}
        rank = meta.get("dp_rank")
        if rank is None:
            rank = r.get("rank")
        if rank is None and info.get("dp_size") == 1:
            rank = 0
        phases, unavailable = [], []
        for kind, first, last in (
            ("Tool call", "tool_started_at", "tool_completed_at"),
            ("Wait", "client_wait_started_at", "submitted_at"),
            (
                "Sampling",
                "submitted_at",
                "completed_at" if "completed_at" in r else "failed_at",
            ),
        ):
            if kind == "Tool call" and r.get("turn") == 0:
                continue
            if (
                r.get(first) is not None
                and r.get(last) is not None
                and r[last] >= r[first]
            ):
                phases.append(
                    {"type": kind, "start": r[first] - origin, "end": r[last] - origin}
                )
            else:
                unavailable.append(kind)
        rows.append(
            {
                "conversation": r.get("conversation", "unknown"),
                "turn": r.get("turn", "?"),
                "worker": rank,
                "phases": phases,
                "unavailable": unavailable,
                "error": r.get("error"),
                "ttft": r.get("ttft_s"),
                "latency": r.get("latency_s"),
                "wait": r.get("client_wait_s"),
                "context": r.get("context_tokens"),
                "output": meta.get("completion_tokens"),
                "hits": meta.get("cached_tokens_details") or {},
            }
        )
    exporters = {}
    for scrape in scrapes:
        if "timestamp" in scrape:
            exporters.setdefault(scrape.get("url", "unknown"), []).append(scrape)
    if not exporters:
        exporters["No metrics recorded"] = []
    summary = {
        "model": server.get("model_path", args.get("tokenizer", "Unknown model")),
        "duration": duration,
        "requests": len(requests),
        "conversations": len({r["conversation"] for r in rows}),
        "turns": args.get("turns", "?"),
        "concurrency": args.get("concurrency", "?"),
        "page_size": server.get("page_size", "?"),
        "status": (
            manifest.get("status", "incomplete")
            if manifest.get("finished_at")
            else "incomplete"
        ),
        "errors": sum(bool(r.get("error")) for r in requests + sessions + scrapes),
        "error": manifest.get("error"),
    }
    return {
        "summary": summary,
        "rows": rows,
        "exporters": {
            url: panels_for(requests, records, origin, duration)
            for url, records in exporters.items()
        },
    }


def build(directory, output):
    data = load_run(directory)
    assets = Path(__file__).parent
    encoded = json.dumps(data, allow_nan=False).replace("<", r"\u003c")
    html = (assets / "viewer.html").read_text()
    html = html.replace("__CSS__", (assets / "viewer.css").read_text())
    html = html.replace("__JS__", (assets / "viewer.js").read_text())
    html = html.replace("__REAL_DATA__", encoded)
    output.write_text(html)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Recorded run directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rollout.html"),
        help="Self-contained HTML output",
    )
    args = parser.parse_args()
    build(args.directory, args.output)
    print(args.output)
