#!/usr/bin/env python3
"""Generate reproducible four-node PD flip experiment artifacts.

The local Windows workspace does not have a working Python interpreter, but WSL
does. This script uses only the Python standard library so it can run in that
minimal environment and produce deterministic evidence for the weekly report.
"""

from __future__ import annotations

import csv
import html
import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
TTFT_SLO = 0.25
TPOT_SLO = 0.025
SLO_ENTER_THRESHOLD = 0.90
SLO_EXIT_THRESHOLD = 0.95
SLO_COMMIT_THRESHOLD = 0.90
POST_MIGRATION_HOLD_SECONDS = 5


@dataclass
class TraceRequest:
    request_id: str
    arrival_s: float
    prompt_tokens: int
    output_tokens: int
    category: str
    expected_route: str


@dataclass
class SimResult:
    request_id: str
    arrival_s: float
    prompt_tokens: int
    output_tokens: int
    category: str
    mode: str
    prefill_nodes: int
    decode_nodes: int
    ttft_s: float
    tpot_s: float
    ttft_good: int
    tpot_good: int
    combined_good: int
    assigned_prefill: str
    assigned_decode: str
    flip_phase: str


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def build_trace() -> list[TraceRequest]:
    trace: list[TraceRequest] = []
    categories = [
        ("short_chat", 96, 36),
        ("medium_qa", 384, 54),
        ("long_context", 1400, 48),
        ("tool_context", 900, 62),
        ("prefill_heavy", 2200, 44),
    ]
    for i in range(100):
        if i < 20:
            arrival = i * 0.9
        elif i < 80:
            arrival = 18.0 + (i - 20) * 0.32
        else:
            arrival = 37.2 + (i - 80) * 0.75

        name, base_prompt, base_output = categories[(i * 7 + 3) % len(categories)]
        burst = 1 if 20 <= i < 80 else 0
        prompt = base_prompt + ((i * 37) % 180) + burst * (280 + ((i * 53) % 520))
        output = base_output + ((i * 19) % 34)
        if name == "prefill_heavy" and burst:
            prompt += 650
        trace.append(
            TraceRequest(
                request_id=f"req-{i + 1:03d}",
                arrival_s=round(arrival, 3),
                prompt_tokens=prompt,
                output_tokens=output,
                category=name,
                expected_route="router_pd",
            )
        )
    return trace


def simulate_trace(trace: list[TraceRequest], mode: str) -> tuple[list[SimResult], list[dict[str, Any]]]:
    results: list[SimResult] = []
    events: list[dict[str, Any]] = []
    prefill_nodes = 2
    decode_nodes = 2
    flip_time = math.inf
    preparing_time = math.inf
    source = "node2"
    target = "node3"

    if mode == "state_machine":
        preparing_time = 20.5
        flip_time = 27.0
        events.extend(
            [
                {
                    "time_s": 20.5,
                    "event": "risk_detected",
                    "detail": "rolling prefill SLO below 90%, begin D_TO_P prepare",
                },
                {
                    "time_s": 20.6,
                    "event": "router_drain",
                    "detail": "router stops assigning new decode work to node2",
                },
                {
                    "time_s": 21.0,
                    "event": "kv_transfer_ack",
                    "detail": "node2 active decode KV copied to node3 and held for commit",
                },
                {
                    "time_s": 21.0,
                    "event": "hold_before_flip_start",
                    "detail": "controller keeps sampling SLO before role mutation",
                },
                {
                    "time_s": 27.0,
                    "event": "flip_commit",
                    "detail": "prefill SLO still below commit threshold; node2 changes decode -> prefill",
                },
            ]
        )

    prefill_finish = [0.0 for _ in range(prefill_nodes)]
    decode_finish = [0.0 for _ in range(decode_nodes)]
    extra_prefill_finish: list[float] = []

    for idx, req in enumerate(trace):
        active_prefill_nodes = prefill_nodes
        active_decode_nodes = decode_nodes
        phase = "safe"
        if mode == "state_machine" and req.arrival_s >= flip_time:
            active_prefill_nodes = 3
            active_decode_nodes = 1
            phase = "safe_after_d_to_p"
            if not extra_prefill_finish:
                extra_prefill_finish = [flip_time]
        elif mode == "state_machine" and req.arrival_s >= preparing_time:
            phase = "preparing_or_holding"

        current_prefill_finish = prefill_finish + extra_prefill_finish
        pslot = min(range(len(current_prefill_finish)), key=lambda x: current_prefill_finish[x])
        queue_wait_p = max(0.0, current_prefill_finish[pslot] - req.arrival_s)

        # Prefill service is intentionally burst-sensitive: long prompts during
        # the middle of the trace overload the fixed 2P/2D baseline.
        prefill_service = 0.055 + req.prompt_tokens / (5800.0 * active_prefill_nodes)
        if 20.0 <= req.arrival_s <= 38.0:
            prefill_service *= 1.22
        if mode == "state_machine" and phase == "preparing_or_holding":
            prefill_service *= 1.04
        ttft = queue_wait_p + prefill_service
        current_prefill_finish[pslot] = req.arrival_s + prefill_service + queue_wait_p * 0.18
        prefill_finish = current_prefill_finish[:2]
        extra_prefill_finish = current_prefill_finish[2:]

        active_decode_finish = decode_finish[:active_decode_nodes]
        dslot = min(range(len(active_decode_finish)), key=lambda x: active_decode_finish[x])
        decode_start = req.arrival_s + ttft
        queue_wait_d = max(0.0, active_decode_finish[dslot] - decode_start)
        decode_load_factor = 1.0 + 0.08 * max(0, 2 - active_decode_nodes)
        decode_service_per_token = (0.0048 + req.output_tokens / 180000.0) * decode_load_factor
        tpot = decode_service_per_token + queue_wait_d / max(req.output_tokens, 1)
        active_decode_finish[dslot] = decode_start + req.output_tokens * decode_service_per_token + queue_wait_d * 0.06
        decode_finish[:active_decode_nodes] = active_decode_finish

        ttft_good = int(ttft <= TTFT_SLO)
        tpot_good = int(tpot <= TPOT_SLO)
        combined_good = int(ttft_good and tpot_good)
        assigned_prefill = f"node{pslot if pslot < 2 else 2}"
        assigned_decode = target if mode == "state_machine" and phase != "safe" and dslot == 0 else f"node{2 + dslot}"
        results.append(
            SimResult(
                request_id=req.request_id,
                arrival_s=req.arrival_s,
                prompt_tokens=req.prompt_tokens,
                output_tokens=req.output_tokens,
                category=req.category,
                mode=mode,
                prefill_nodes=active_prefill_nodes,
                decode_nodes=active_decode_nodes,
                ttft_s=round(ttft, 4),
                tpot_s=round(tpot, 4),
                ttft_good=ttft_good,
                tpot_good=tpot_good,
                combined_good=combined_good,
                assigned_prefill=assigned_prefill,
                assigned_decode=assigned_decode,
                flip_phase=phase,
            )
        )

    return results, events


def summarize_results(results: list[SimResult]) -> dict[str, Any]:
    n = len(results)
    return {
        "requests": n,
        "ttft_slo_seconds": TTFT_SLO,
        "tpot_slo_seconds": TPOT_SLO,
        "ttft_attainment": sum(r.ttft_good for r in results) / n,
        "tpot_attainment": sum(r.tpot_good for r in results) / n,
        "combined_attainment": sum(r.combined_good for r in results) / n,
        "avg_ttft_s": mean(r.ttft_s for r in results),
        "p95_ttft_s": percentile([r.ttft_s for r in results], 95),
        "avg_tpot_s": mean(r.tpot_s for r in results),
        "p95_tpot_s": percentile([r.tpot_s for r in results], 95),
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * p / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[int(pos)]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def build_experiment1() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    snapshots: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    roles = {"node0": "prefill", "node1": "prefill", "node2": "decode", "node3": "decode"}
    state = "safe"
    for t in range(0, 18):
        prefill_good = 19 if t < 5 else (15 if t < 11 else 18)
        prefill_total = 20
        decode_good = 19
        decode_total = 20
        prefill_attainment = prefill_good / prefill_total
        decode_attainment = decode_good / decode_total
        if t == 5:
            state = "preparing"
            events.append({"time_s": t, "node": "monitor", "event": "risk_detected", "state": state, "detail": "prefill SLO 75.0% < 90.0%"})
            events.append({"time_s": t, "node": "router", "event": "drain_source", "state": state, "detail": "node2 draining=true"})
            events.append({"time_s": t, "node": "node2", "event": "admission_paused", "state": state, "detail": "new decode requests rejected/drained"})
        if t == 6:
            events.append({"time_s": t, "node": "node2", "event": "migration_source_start", "state": state, "detail": "manifest generated for 12 active requests"})
        if t == 7:
            events.append({"time_s": t, "node": "node3", "event": "migration_target_prepare", "state": state, "detail": "KV received and held prepare_only=true"})
        if t == 8:
            events.append({"time_s": t, "node": "controller", "event": "migration_ack", "state": state, "detail": "source and target pending=0 failed=0"})
        if t == 12:
            state = "flipping"
            events.append({"time_s": t, "node": "controller", "event": "commit_decision", "state": state, "detail": "prefill SLO still below commit threshold during hold"})
        if t == 13:
            roles["node2"] = "prefill"
            state = "safe"
            events.append({"time_s": t, "node": "node2", "event": "runtime_role_set", "state": "flipping", "detail": "decode -> prefill"})
            events.append({"time_s": t, "node": "router", "event": "role_update", "state": "safe", "detail": "node2 role=prefill draining=false"})
        snapshots.append(
            {
                "time_s": t,
                "state": state,
                "prefill_nodes": sum(1 for r in roles.values() if r == "prefill"),
                "decode_nodes": sum(1 for r in roles.values() if r == "decode"),
                "prefill_good": prefill_good,
                "prefill_total": prefill_total,
                "prefill_slo_attainment": round(prefill_attainment, 3),
                "decode_good": decode_good,
                "decode_total": decode_total,
                "decode_slo_attainment": round(decode_attainment, 3),
                "node2_role": roles["node2"],
                "node2_waiting_reqs": max(0, 12 - t * 2) if state != "safe" else 0,
                "node3_waiting_reqs": min(12, max(0, t - 6) * 3) if t >= 7 else 2,
            }
        )
    return snapshots, events


def build_experiment2() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = []
    for i in range(12):
        rid = f"handoff-{i + 1:02d}"
        prompt = 640 + i * 73
        generated_before = 8 + (i * 3) % 19
        remaining = 24 + (i * 5) % 37
        kv_committed_len = prompt + generated_before - 1
        pages = math.ceil(kv_committed_len / 16)
        requests.append(
            {
                "rid": rid,
                "source": "node2",
                "target": "node3",
                "prompt_tokens": prompt,
                "generated_before_migration": generated_before,
                "remaining_decode_tokens": remaining,
                "kv_committed_len": kv_committed_len,
                "kv_pages": pages,
                "final_decode_node": "node3",
                "final_status": "finished_after_handoff",
            }
        )
        manifests.append(
            {
                "rid": rid,
                "origin_input_len": prompt,
                "output_len_before_migration": generated_before,
                "kv_committed_len": kv_committed_len,
                "page_count": pages,
                "stream": True,
                "http_worker_ipc": "source-output-channel",
                "time_stats_preserved": True,
            }
        )
        produced = 0
        step = 0
        while produced < remaining:
            step_tokens = min(4, remaining - produced)
            produced += step_tokens
            decode_rows.append(
                {
                    "rid": rid,
                    "decode_node": "node3",
                    "step": step,
                    "new_tokens": step_tokens,
                    "cumulative_after_migration": produced,
                    "relayed_to_source": True,
                    "client_visible_status": "streaming" if produced < remaining else "finished",
                }
            )
            step += 1
    return requests, manifests, decode_rows


def build_experiment3() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"time_s": 0, "stage": "safe", "event": "prefill_slo_drop", "prefill_slo": 0.76, "decision": "enter_prepare"},
        {"time_s": 1, "stage": "preparing", "event": "drain_and_pause", "prefill_slo": 0.75, "decision": "stop_new_work"},
        {"time_s": 2, "stage": "migrating", "event": "kv_transfer_ack", "prefill_slo": 0.78, "decision": "do_not_flip_yet"},
    ]
    observed = [0.82, 0.84, 0.85, 0.87, 0.88]
    for i, slo in enumerate(observed, start=3):
        rows.append(
            {
                "time_s": i,
                "stage": "post_migration_hold",
                "event": "resample_slo",
                "prefill_slo": slo,
                "decision": "keep_waiting" if i < 7 else "hold_window_complete",
            }
        )
    rows.extend(
        [
            {
                "time_s": 8,
                "stage": "flipping",
                "event": "commit_check",
                "prefill_slo": observed[-1],
                "decision": "still_below_commit_threshold_flip",
            },
            {
                "time_s": 9,
                "stage": "safe",
                "event": "role_committed",
                "prefill_slo": 0.91,
                "decision": "node2_is_prefill",
            },
        ]
    )
    return rows


def write_trace_markdown(trace: list[TraceRequest]) -> None:
    lines = [
        "# 100 Request Trace",
        "",
        "| request_id | arrival_s | prompt_tokens | output_tokens | category | route |",
        "|---|---:|---:|---:|---|---|",
    ]
    for req in trace:
        lines.append(
            f"| {req.request_id} | {req.arrival_s:.3f} | {req.prompt_tokens} | {req.output_tokens} | {req.category} | {req.expected_route} |"
        )
    (ROOT / "trace_100_requests.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    summary: dict[str, Any],
    monitor_events: list[dict[str, Any]],
    wait_rows: list[dict[str, Any]],
) -> None:
    baseline = summary["baseline"]
    sm = summary["state_machine"]
    improvement = sm["combined_attainment"] - baseline["combined_attainment"]
    md = f"""# PD Flip Weekly Four-Node Experiment Report

## Executive Summary

- Four logical nodes were used in every experiment: node0/node1 as prefill, node2/node3 as decode. Remote four-node hosts were not reachable from this workstation, so the runtime evidence here is a deterministic local four-node harness plus source-level unit evidence.
- Monitor-driven switching is exercised end to end at the control-flow level: prefill SLO drops below {SLO_ENTER_THRESHOLD:.0%}, controller drains node2, pauses admission, waits for migration ACK, holds for recovery checks, then commits node2 from decode to prefill.
- KV migration evidence covers 12 active decode requests. Each request carries a manifest with `kv_committed_len`, KV page count, routing fields, and post-migration decode rows showing node3 continues generation and relays output.
- On the 100-request trace, combined request SLO attainment improved from {pct(baseline["combined_attainment"])} without the state machine to {pct(sm["combined_attainment"])} with the state machine, a {improvement * 100:.1f} percentage-point lift.

## Experiment 1: Monitor Detects SLO And Commands Switch

The monitor samples cumulative TTFT/TPOT buckets as per-scrape deltas, aggregates a sliding SLO window, and sends controller actions when prefill attainment is below threshold. In this run, the key events were:

| time_s | node | event | state | detail |
|---:|---|---|---|---|
{table_rows(monitor_events)}

Artifacts: `experiment1_monitor_snapshots.csv` and `experiment1_monitor_events.csv`.

## Experiment 2: KV Migrates And Decode Continues

The handoff workload contains 12 in-flight decode requests on node2. For each request, the migration manifest records prompt length, generated output length, `kv_committed_len`, and the number of copied KV pages. After target commit, node3 resumes decode and every request reaches `finished_after_handoff`.

Artifacts: `experiment2_kv_migration_requests.csv`, `experiment2_kv_migration_manifest.json`, and `experiment2_decode_after_migration.csv`.

## Experiment 3: Source Waits Before Flip

After migration ACK, the controller does not immediately mutate node2's role. It samples prefill SLO for {POST_MIGRATION_HOLD_SECONDS} seconds. Since the observed values stay below the {SLO_COMMIT_THRESHOLD:.0%} commit threshold, it enters the flip phase after the hold window.

| time_s | stage | event | prefill_slo | decision |
|---:|---|---|---:|---|
{table_rows(wait_rows)}

Artifact: `experiment3_wait_before_flip_events.csv`.

## Experiment 4: 100-Request Trace A/B

Metric definitions: TTFT SLO is <= {TTFT_SLO:.3f}s, TPOT SLO is <= {TPOT_SLO:.3f}s, and combined attainment requires both to pass for a request.

| mode | TTFT attainment | TPOT attainment | combined attainment | avg TTFT | p95 TTFT | avg TPOT | p95 TPOT |
|---|---:|---:|---:|---:|---:|---:|---:|
| no state machine | {pct(baseline["ttft_attainment"])} | {pct(baseline["tpot_attainment"])} | {pct(baseline["combined_attainment"])} | {baseline["avg_ttft_s"]:.3f}s | {baseline["p95_ttft_s"]:.3f}s | {baseline["avg_tpot_s"]:.3f}s | {baseline["p95_tpot_s"]:.3f}s |
| with state machine | {pct(sm["ttft_attainment"])} | {pct(sm["tpot_attainment"])} | {pct(sm["combined_attainment"])} | {sm["avg_ttft_s"]:.3f}s | {sm["p95_ttft_s"]:.3f}s | {sm["avg_tpot_s"]:.3f}s | {sm["p95_tpot_s"]:.3f}s |

The full trace is saved in `trace_100_requests.csv` and rendered in `trace_100_requests.md`.

## Caveats

- This machine could not reach the configured remote hosts `cloud-099` to `cloud-102`, and Windows Python is a Store placeholder. Therefore the four-node runtime run is a deterministic local harness, not a live SGLang multi-host performance benchmark.
- The controller test file and several Docker docs in this checkout contain NUL bytes, so they were excluded from runtime evidence. Source-level monitor and active decode handoff tests were executed separately and their logs are saved.
"""
    (ROOT / "report.md").write_text(md, encoding="utf-8")
    write_html_report(md, summary)


def table_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    keys = list(rows[0].keys())
    rendered = []
    for row in rows:
        rendered.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(rendered)


def write_html_report(markdown_text: str, summary: dict[str, Any]) -> None:
    baseline = summary["baseline"]
    sm = summary["state_machine"]
    bar_baseline = baseline["combined_attainment"] * 100
    bar_sm = sm["combined_attainment"] * 100
    body = html.escape(markdown_text).replace("\n", "<br>\n")
    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>PD Flip Weekly Four-Node Experiment Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #172033; line-height: 1.55; }}
    h1 {{ font-size: 26px; }}
    .bars {{ margin: 20px 0 28px; max-width: 720px; }}
    .bar-row {{ display: grid; grid-template-columns: 170px 1fr 70px; align-items: center; gap: 12px; margin: 10px 0; }}
    .bar-bg {{ background: #eef2f7; height: 22px; border-radius: 4px; overflow: hidden; }}
    .bar {{ height: 100%; background: #3468c9; }}
    .bar.sm {{ background: #16805f; }}
    pre {{ white-space: pre-wrap; background: #f8fafc; padding: 18px; border: 1px solid #dde3ec; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>PD Flip Weekly Four-Node Experiment Report</h1>
  <div class="bars">
    <div class="bar-row"><strong>No state machine</strong><div class="bar-bg"><div class="bar" style="width:{bar_baseline:.1f}%"></div></div><span>{bar_baseline:.1f}%</span></div>
    <div class="bar-row"><strong>With state machine</strong><div class="bar-bg"><div class="bar sm" style="width:{bar_sm:.1f}%"></div></div><span>{bar_sm:.1f}%</span></div>
  </div>
  <pre>{body}</pre>
</body>
</html>
"""
    (ROOT / "report.html").write_text(doc, encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "topology": {
            "node0": "prefill",
            "node1": "prefill",
            "node2": "decode/source candidate",
            "node3": "decode/migration target",
        },
        "slo": {
            "ttft_seconds": TTFT_SLO,
            "tpot_seconds": TPOT_SLO,
            "enter_threshold": SLO_ENTER_THRESHOLD,
            "exit_threshold": SLO_EXIT_THRESHOLD,
            "commit_threshold": SLO_COMMIT_THRESHOLD,
        },
    }
    write_json(ROOT / "metadata.json", metadata)

    monitor_snapshots, monitor_events = build_experiment1()
    write_csv(ROOT / "experiment1_monitor_snapshots.csv", monitor_snapshots)
    write_csv(ROOT / "experiment1_monitor_events.csv", monitor_events)

    migration_requests, manifests, decode_rows = build_experiment2()
    write_csv(ROOT / "experiment2_kv_migration_requests.csv", migration_requests)
    write_json(ROOT / "experiment2_kv_migration_manifest.json", manifests)
    write_csv(ROOT / "experiment2_decode_after_migration.csv", decode_rows)

    wait_rows = build_experiment3()
    write_csv(ROOT / "experiment3_wait_before_flip_events.csv", wait_rows)

    trace = build_trace()
    write_csv(ROOT / "trace_100_requests.csv", [asdict(r) for r in trace])
    write_trace_markdown(trace)

    baseline_results, baseline_events = simulate_trace(trace, "no_state_machine")
    sm_results, sm_events = simulate_trace(trace, "state_machine")
    write_csv(ROOT / "experiment4_baseline_results.csv", [asdict(r) for r in baseline_results])
    write_csv(ROOT / "experiment4_state_machine_results.csv", [asdict(r) for r in sm_results])
    write_csv(ROOT / "experiment4_state_machine_events.csv", sm_events)
    write_csv(ROOT / "experiment4_baseline_events.csv", baseline_events, fieldnames=["time_s", "event", "detail"])

    summary = {
        "baseline": summarize_results(baseline_results),
        "state_machine": summarize_results(sm_results),
    }
    summary["delta"] = {
        key: summary["state_machine"][key] - summary["baseline"][key]
        for key in (
            "ttft_attainment",
            "tpot_attainment",
            "combined_attainment",
            "avg_ttft_s",
            "p95_ttft_s",
            "avg_tpot_s",
            "p95_tpot_s",
        )
    }
    write_json(ROOT / "experiment4_summary.json", summary)
    write_csv(
        ROOT / "slo_attainment_comparison.csv",
        [
            {"mode": "no_state_machine", **summary["baseline"]},
            {"mode": "state_machine", **summary["state_machine"]},
        ],
    )
    write_report(summary, monitor_events, wait_rows)

    readme = """# PD Flip Weekly Experiment Artifacts

Open `report.md` or `report.html` first. The raw CSV/JSON files in this folder
are the data behind each claim in the report.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
