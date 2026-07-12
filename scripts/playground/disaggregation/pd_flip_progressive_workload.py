#!/usr/bin/env python3
"""Generate auditable progressive PD-flip traffic and request ledgers.

The selected mode controls only prompt warm-up/input strategy. Actual HiCache
mode acceptance must come from worker ``request_measurements.stitch_mode``.
"""

import argparse
import json
import os
import random
import secrets
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class PromptPlan:
    warmup_prompts: Tuple[str, ...]
    active_prompt: str
    trial_nonce: Optional[str] = None


def build_prompt_plan(mode: str, seed: int) -> PromptPlan:
    rng = random.Random(seed)
    shared = " ".join(["pd-flip-shared-prefix"] * 512)
    unique = " ".join("u%08x" % rng.getrandbits(32) for _ in range(256))
    if mode == "full":
        active = shared + " active-long " + unique
        return PromptPlan((active,), active)
    if mode == "partial":
        warm = " ".join(shared.split()[:256])
        return PromptPlan((warm,), warm + " cold-suffix " + unique)
    if mode == "zero":
        # Zero-hit trials must not accidentally reuse a deterministic prefix across runs.
        nonce = secrets.token_hex(16)
        cold = " ".join("z" + nonce + secrets.token_hex(8) for _ in range(768))
        return PromptPlan((), cold, nonce)
    raise ValueError("unknown mode: %s" % mode)


class OpenAIClient:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: float):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _request(self, payload: JsonDict) -> urllib.request.Request:
        return urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )

    def complete(self, payload: JsonDict) -> Tuple[int, str]:
        with urllib.request.urlopen(
            self._request(payload), timeout=self.timeout_seconds
        ) as response:
            raw = response.read().decode("utf-8", errors="replace")
            body = json.loads(raw)
            text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            return int(response.status), str(text or "")

    def stream(self, payload: JsonDict) -> Tuple[int, List[JsonDict], List[float]]:
        sequences: List[JsonDict] = []
        arrivals: List[float] = []
        saw_done = False
        saw_finish = False
        with urllib.request.urlopen(
            self._request(payload), timeout=self.timeout_seconds
        ) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    saw_done = True
                    continue
                chunk = json.loads(data)
                choice = chunk.get("choices", [{}])[0]
                if choice.get("finish_reason") is not None:
                    saw_finish = True
                text = choice.get("delta", {}).get("content")
                if text is None:
                    continue
                sequences.append(
                    {"chunk_ordinal": len(sequences), "text": str(text)}
                )
                arrivals.append(time.monotonic())
            if not sequences or not saw_finish or not saw_done:
                raise RuntimeError(
                    "incomplete SSE stream: content=%s finish_reason=%s done=%s"
                    % (bool(sequences), saw_finish, saw_done)
                )
            return int(response.status), sequences, arrivals


def percentile95(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def write_jsonl(path: Path, rows: Sequence[JsonDict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")


def request_payload(
    model: str, prompt: str, max_tokens: int, stream: bool, rid: str
) -> JsonDict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
        "rid": rid,
    }


def fetch_json(url: str, api_key: str, timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        headers={"Authorization": "Bearer " + api_key},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def flatten_status(payload: Any) -> List[JsonDict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        fanout = payload.get("workers") or payload.get("results")
        if isinstance(fanout, list):
            return [item for item in fanout if isinstance(item, dict)]
        return [payload]
    return []


def atomic_write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def wait_for(predicate, timeout: float, interval: float, description: str):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    raise TimeoutError("timed out waiting for %s (last=%r)" % (description, last))


EXPECTED_STITCH_MODE = {
    "full": "full_prefix_stitch",
    "partial": "partial_prefix_stitch",
    "zero": "source_decode_full_fallback",
}


def validate_target_measurement(
    source_payload: Any, target_payload: Any, rid: str, mode: str
) -> JsonDict:
    source_rows = [row.get("status", row) for row in flatten_status(source_payload)]
    target_rows = [row.get("status", row) for row in flatten_status(target_payload)]
    source_sessions = {
        row.get("session_id") for row in source_rows if row.get("session_id")
    }
    target_sessions = {
        row.get("session_id") for row in target_rows if row.get("session_id")
    }
    shared_sessions = source_sessions & target_sessions
    if not shared_sessions:
        raise RuntimeError("source and target do not report the same migration session")
    matches = []
    for row in target_rows:
        if row.get("session_id") not in shared_sessions:
            continue
        for measurement in row.get("request_measurements", []):
            if measurement.get("rid") == rid:
                matches.append(measurement)
    if len(matches) != 1:
        raise RuntimeError("expected exactly one target measurement for rid %s" % rid)
    measurement = matches[0]
    if measurement.get("stitch_mode") != EXPECTED_STITCH_MODE[mode]:
        raise RuntimeError("target stitch_mode does not match requested trial")
    if measurement.get("final_owner") != "target":
        raise RuntimeError("target did not become final owner")
    if measurement.get("output_boundary") is None:
        raise RuntimeError("target measurement is missing output_boundary")
    return measurement


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_prompt_plan(args.mode, args.seed)
    api_key = os.environ.get(args.admin_api_key_env, "")
    if not api_key:
        raise ValueError(
            "admin API key environment variable %s is empty"
            % args.admin_api_key_env
        )
    client = OpenAIClient(args.base_url, api_key, args.timeout_seconds)
    started = time.monotonic()
    metrics: List[JsonDict] = []
    errors: List[JsonDict] = []
    lock = threading.Lock()
    active_request_id = str(uuid.uuid4())

    def record_error(request_id: str, kind: str, exc: Exception) -> None:
        with lock:
            errors.append(
                {
                    "request_id": request_id,
                    "prompt_kind": kind,
                    "ts_wall": time.time(),
                    "error": repr(exc),
                }
            )

    def run_nonstream(kind: str, prompt: str, max_tokens: int) -> None:
        request_id = str(uuid.uuid4())
        begin = time.monotonic()
        try:
            status, output_text = client.complete(
                request_payload(args.model, prompt, max_tokens, False, request_id)
            )
            end = time.monotonic()
            elapsed = end - begin
            row = {
                "request_id": request_id,
                "prompt_kind": kind,
                "mode": args.mode,
                "decision_path": args.decision_path,
                "arrival_offset_s": begin - started,
                "start_monotonic": begin,
                "end_monotonic": end,
                "migration_phase": "workload",
                "active_during_migration": kind == "pressure_short",
                "status": status,
                "ttft_slo_s": args.ttft_slo_seconds,
                "ttft_s": elapsed,
                "ttft_met": elapsed <= args.ttft_slo_seconds,
                "tpot_slo_s": args.tpot_slo_seconds,
                "avg_tpot_s": None,
                "p95_tpot_s": None,
                "tpot_avg_met": None,
                "output_chunks": [{"chunk_ordinal": 0, "text": output_text}],
                "output_text": output_text,
            }
            with lock:
                metrics.append(row)
        except Exception as exc:
            record_error(request_id, kind, exc)

    for prompt in plan.warmup_prompts:
        run_nonstream("warmup", prompt, args.warmup_max_tokens)

    def run_active() -> None:
        request_id = active_request_id
        begin = time.monotonic()
        try:
            status, sequences, arrivals = client.stream(
                request_payload(
                    args.model,
                    plan.active_prompt,
                    args.long_max_tokens,
                    True,
                    request_id,
                )
            )
            end = time.monotonic()
            ttft = arrivals[0] - begin if arrivals else None
            gaps = [right - left for left, right in zip(arrivals, arrivals[1:])]
            avg_tpot = statistics.mean(gaps) if gaps else None
            row = {
                "request_id": request_id,
                "prompt_kind": "active_long",
                "mode": args.mode,
                "decision_path": args.decision_path,
                "arrival_offset_s": begin - started,
                "start_monotonic": begin,
                "end_monotonic": end,
                "migration_phase": "workload",
                "active_during_migration": True,
                "status": status,
                "ttft_slo_s": args.ttft_slo_seconds,
                "ttft_s": ttft,
                "ttft_met": ttft is not None and ttft <= args.ttft_slo_seconds,
                "tpot_slo_s": args.tpot_slo_seconds,
                "avg_tpot_s": avg_tpot,
                "p95_tpot_s": percentile95(gaps),
                "tpot_avg_met": avg_tpot is not None and avg_tpot <= args.tpot_slo_seconds,
                "output_chunks": sequences,
                "output_text": "".join(item["text"] for item in sequences),
            }
            with lock:
                metrics.append(row)
        except Exception as exc:
            record_error(request_id, "active_long", exc)

    active = threading.Thread(target=run_active, name="pd-flip-active-request")
    active.start()

    def active_is_running():
        payload = fetch_json(
            args.source_url.rstrip("/") + "/pd_flip/runtime_role/status",
            api_key,
            args.timeout_seconds,
        )
        for row in flatten_status(payload):
            status = row.get("status", row)
            for request in status.get("running_requests", []):
                if request.get("rid") == active_request_id:
                    return payload
        return None

    try:
        ready_snapshot = wait_for(
            active_is_running,
            args.coordination_timeout_seconds,
            args.poll_interval_seconds,
            "active rid %s on source" % active_request_id,
        )
        atomic_write_json(
            Path(args.ready_marker),
            {"rid": active_request_id, "source_snapshot": ready_snapshot},
        )
        wait_for(
            lambda: Path(args.controller_done_marker).exists(),
            args.coordination_timeout_seconds,
            args.poll_interval_seconds,
            "controller done marker",
        )
    except Exception as exc:
        record_error(active_request_id, "coordination", exc)
    pressure_count = args.pressure_requests
    if pressure_count is None:
        pressure_count = 8 if args.decision_path == "recovery" else 80
    for index in range(pressure_count):
        run_nonstream(
            "pressure_short",
            "short prefill pressure %d %s" % (index, plan.active_prompt[:128]),
            args.pressure_max_tokens,
        )
        if args.pressure_interval_seconds > 0:
            time.sleep(args.pressure_interval_seconds)
    active.join(timeout=args.timeout_seconds + 30)
    if active.is_alive():
        errors.append(
            {
                "request_id": "active-thread",
                "prompt_kind": "active_long",
                "ts_wall": time.time(),
                "error": "active request did not finish before join deadline",
            }
        )

    migration_measurement = None
    try:
        source_status = fetch_json(
            args.source_url.rstrip("/") + "/pd_flip/migration/status",
            api_key,
            args.timeout_seconds,
        )
        target_status = fetch_json(
            args.target_url.rstrip("/") + "/pd_flip/migration/status",
            api_key,
            args.timeout_seconds,
        )
        migration_measurement = validate_target_measurement(
            source_status, target_status, active_request_id, args.mode
        )
        control_id = str(uuid.uuid4())
        _, control_text = client.complete(
            request_payload(
                args.model,
                plan.active_prompt,
                args.long_max_tokens,
                False,
                control_id,
            )
        )
        active_row = next(
            row
            for row in metrics
            if row.get("request_id") == active_request_id
        )
        if control_text != active_row.get("output_text"):
            raise RuntimeError("active output differs from post-migration control")
        active_row["control_request_id"] = control_id
        active_row["control_exact_match"] = True
        active_row["migration_measurement"] = migration_measurement
    except Exception as exc:
        record_error(active_request_id, "post_migration_validation", exc)

    metrics.sort(key=lambda row: float(row["arrival_offset_s"]))
    write_jsonl(output_dir / "request_metrics.jsonl", metrics)
    write_jsonl(output_dir / "errors.jsonl", errors)
    config = {
        "base_url": args.base_url,
        "source_url": args.source_url,
        "target_url": args.target_url,
        "model": args.model,
        "mode": args.mode,
        "decision_path": args.decision_path,
        "seed": args.seed,
        "zero_trial_nonce": plan.trial_nonce,
        "active_request_id": active_request_id,
        "pressure_requests": pressure_count,
        "pressure_interval_seconds": args.pressure_interval_seconds,
        "long_max_tokens": args.long_max_tokens,
        "note": "mode is an input strategy; accept actual stitch_mode from worker request_measurements",
    }
    (output_dir / "workload_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--target-url", required=True)
    parser.add_argument("--admin-api-key-env", default="ADMIN_API_KEY")
    parser.add_argument("--ready-marker", required=True)
    parser.add_argument("--controller-done-marker", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("full", "partial", "zero"), required=True)
    parser.add_argument(
        "--decision-path", choices=("recovery", "commit"), required=True
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pressure-requests", type=int, default=None)
    parser.add_argument("--pressure-interval-seconds", type=float, default=0.25)
    parser.add_argument("--pressure-max-tokens", type=int, default=1)
    parser.add_argument("--warmup-max-tokens", type=int, default=1)
    parser.add_argument("--long-max-tokens", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--coordination-timeout-seconds", type=float, default=300)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.1)
    parser.add_argument("--ttft-slo-seconds", type=float, default=0.2)
    parser.add_argument("--tpot-slo-seconds", type=float, default=0.02)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
