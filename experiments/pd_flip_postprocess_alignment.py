#!/usr/bin/env python3
"""Create alignment helper files for a PD flip experiment suite."""

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_csv(path: Path) -> Iterable[Dict[str, str]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def parse_runner_wall(line: str) -> Optional[float]:
    match = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", line)
    if not match:
        return None
    dt = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    return dt.replace(tzinfo=timezone.utc).timestamp()


def find_force_wall(suite: Path, experiment: str) -> Optional[float]:
    log_path = suite / "suite_runner.log"
    if not log_path.exists():
        return None
    needle = "forcing two-phase D->P for %s" % experiment
    last_seen = None
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if needle in line:
                last_seen = parse_runner_wall(line)
    return last_seen


def as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def add_event(
    rows: List[Dict[str, Any]],
    *,
    experiment: str,
    source: str,
    event: str,
    t_rel_s: Optional[float],
    wall_s: Optional[float],
    node: str = "",
    request_id: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    rows.append(
        {
            "experiment": experiment,
            "source": source,
            "event": event,
            "t_rel_s": "" if t_rel_s is None else "%.6f" % t_rel_s,
            "wall_s": "" if wall_s is None else "%.6f" % wall_s,
            "node": node,
            "request_id": request_id,
            "details_json": json.dumps(details or {}, sort_keys=True),
        }
    )


def build_aligned_timeline(suite: Path, experiment: str, mode: str) -> List[Dict[str, Any]]:
    exp_dir = suite / experiment
    summary = read_json(exp_dir / mode / "summary.json")
    run_wall = float(summary["run_started_wall"])
    rows: List[Dict[str, Any]] = []

    add_event(
        rows,
        experiment=experiment,
        source="request_runner",
        event="run_start",
        t_rel_s=0.0,
        wall_s=run_wall,
        details={"mode": mode, "trace_requests": summary.get("trace_requests")},
    )

    metrics_path = exp_dir / mode / "request_metrics.csv"
    if not metrics_path.exists():
        metrics_path = exp_dir / mode / "request_metrics.jsonl"
    if metrics_path.suffix == ".csv":
        metric_rows = list(iter_csv(metrics_path))
    else:
        metric_rows = []
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        metric_rows.append(json.loads(line))

    for row in metric_rows:
        request_id = str(row.get("request_id") or "")
        start_wall = as_float(row.get("start_wall"))
        end_wall = as_float(row.get("end_wall"))
        ttft = as_float(row.get("ttft_s"))
        status = row.get("status")
        add_event(
            rows,
            experiment=experiment,
            source="request",
            event="request_start",
            t_rel_s=None if start_wall is None else start_wall - run_wall,
            wall_s=start_wall,
            request_id=request_id,
            details={
                "prompt_kind": row.get("prompt_kind"),
                "ttft_slo_s": row.get("ttft_slo_s"),
                "tpot_slo_s": row.get("tpot_slo_s"),
            },
        )
        if start_wall is not None and ttft is not None:
            add_event(
                rows,
                experiment=experiment,
                source="request",
                event="first_token",
                t_rel_s=start_wall + ttft - run_wall,
                wall_s=start_wall + ttft,
                request_id=request_id,
                details={"ttft_s": ttft, "ttft_met": row.get("ttft_met")},
            )
        add_event(
            rows,
            experiment=experiment,
            source="request",
            event="request_end",
            t_rel_s=None if end_wall is None else end_wall - run_wall,
            wall_s=end_wall,
            request_id=request_id,
            details={
                "status": status,
                "latency_s": row.get("latency_s"),
                "ttft_s": row.get("ttft_s"),
                "avg_tpot_s": row.get("avg_tpot_s"),
                "all_met": row.get("all_met"),
            },
        )

    controller_wall = find_force_wall(suite, experiment)
    cumulative = 0.0
    for row in iter_csv(exp_dir / "migration_link" / "controller_actions.csv") or []:
        elapsed = as_float(row.get("elapsed_seconds")) or 0.0
        cumulative += elapsed
        wall = None if controller_wall is None else controller_wall + cumulative
        add_event(
            rows,
            experiment=experiment,
            source="controller",
            event=str(row.get("step") or "controller_action"),
            t_rel_s=None if wall is None else wall - run_wall,
            wall_s=wall,
            node=str(row.get("target") or ""),
            details={
                "action_index": row.get("action_index"),
                "success": row.get("success"),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "url": row.get("url"),
            },
        )

    for row in iter_csv(exp_dir / "migration_link" / "migration_status_samples.csv") or []:
        wall = as_float(row.get("ts_wall"))
        details = {
            "role": row.get("role"),
            "state": row.get("state"),
            "session_id": row.get("session_id"),
            "pending_reqs": row.get("pending_reqs"),
            "transferred_reqs": row.get("transferred_reqs"),
            "failed_reqs": row.get("failed_reqs"),
            "held_reqs": row.get("held_reqs"),
        }
        timing_debug = row.get("timing_debug")
        if timing_debug:
            details["timing_debug"] = timing_debug
        add_event(
            rows,
            experiment=experiment,
            source="worker_status",
            event="migration_status_sample",
            t_rel_s=None if wall is None else wall - run_wall,
            wall_s=wall,
            node=str(row.get("node") or ""),
            details=details,
        )

    rows.sort(key=lambda item: (item["t_rel_s"] == "", float(item["t_rel_s"] or 0.0), item["source"], item["event"]))
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "experiment",
        "source",
        "event",
        "t_rel_s",
        "wall_s",
        "node",
        "request_id",
        "details_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_notes(suite: Path, experiment: str, mode: str) -> None:
    notes = """# Alignment Notes

This suite includes raw files plus derived alignment helpers.

- `aligned_timeline.csv`: merged request, controller, and worker-status events for the state-machine 200-request run.
- Time zero is the replay runner `run_started_wall` from `{experiment}/{mode}/summary.json`.
- Request `first_token` wall time is reconstructed as `start_wall + ttft_s`.
- Controller action wall time is anchored to the suite log line `forcing two-phase D->P`; per-action times are cumulative `elapsed_seconds`.
- Worker status rows use sampler `ts_wall`; sampler interval was 0.2s, so status-stage boundary precision is about one polling interval.
- Worker `timing_debug` values are preserved as raw JSON in `details_json`. Many of those values are worker-local monotonic times, so use them as intra-worker relative timing unless wall-clock fields are added later.

Primary A/B result:

- Baseline no-state-machine: see `09_ab_200_baseline_no_state_machine/baseline/summary.json`.
- State-machine two-phase: see `{experiment}/{mode}/summary.json`.

Important caveat: runs 01-08 and the first 10 run have request/controller raw data, but their sampler was started with an old node spec and did not collect worker status. The rerun of `{experiment}` fixed this and is the aligned chain-measurement source.
""".format(experiment=experiment, mode=mode)
    (suite / "alignment_notes.md").write_text(notes, encoding="utf-8")


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("usage: pd_flip_postprocess_alignment.py <suite-dir>", file=sys.stderr)
        return 2
    suite = Path(argv[1])
    experiment = "10_ab_200_state_machine_two_phase"
    mode = "state_machine"
    rows = build_aligned_timeline(suite, experiment, mode)
    write_csv(suite / "aligned_timeline.csv", rows)
    write_notes(suite, experiment, mode)
    print(json.dumps({"aligned_rows": len(rows), "suite": str(suite)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
