#!/usr/bin/env python3
"""Generate and replay a request-level PD flip SLO trace.

The replay path intentionally depends only on the Python standard library and
keeps Python 3.6 compatibility for older controller hosts.
"""

import argparse
import csv
import json
import math
import random
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


JsonDict = Dict[str, Any]


class PromptProfile:
    def __init__(
        self,
        name: str,
        count_weight: int,
        max_tokens: int,
        ttft_slo_s: float,
        tpot_slo_s: float,
        repeat_blocks: int,
        answer_style: str,
    ) -> None:
        self.name = name
        self.count_weight = count_weight
        self.max_tokens = max_tokens
        self.ttft_slo_s = ttft_slo_s
        self.tpot_slo_s = tpot_slo_s
        self.repeat_blocks = repeat_blocks
        self.answer_style = answer_style


PROFILES: Sequence[PromptProfile] = (
    PromptProfile(
        name="short",
        count_weight=12,
        max_tokens=96,
        ttft_slo_s=3.0,
        tpot_slo_s=0.025,
        repeat_blocks=1,
        answer_style="Return exactly six terse bullets.",
    ),
    PromptProfile(
        name="medium",
        count_weight=6,
        max_tokens=256,
        ttft_slo_s=5.0,
        tpot_slo_s=0.030,
        repeat_blocks=8,
        answer_style="Return a structured checklist with concise rationale.",
    ),
    PromptProfile(
        name="long",
        count_weight=2,
        max_tokens=768,
        ttft_slo_s=8.0,
        tpot_slo_s=0.035,
        repeat_blocks=32,
        answer_style="Return a detailed runbook and keep going until the budget is used.",
    ),
)


def build_trace(
    *,
    num_requests: int,
    interval_seconds: float,
    model: str,
    seed: int,
    temperature: float = 0.0,
    stream: bool = True,
    short_chars: Optional[int] = None,
    long_chars: Optional[int] = None,
    short_count: Optional[int] = None,
    long_count: Optional[int] = None,
) -> List[JsonDict]:
    if num_requests <= 0:
        raise ValueError("num_requests must be positive")
    if interval_seconds < 0:
        raise ValueError("interval_seconds must be non-negative")

    rng = random.Random(seed)
    profile_by_name = {profile.name: profile for profile in PROFILES}
    explicit_char_mix = any(
        value is not None
        for value in (short_chars, long_chars, short_count, long_count)
    )
    target_chars_by_kind: Dict[str, Optional[int]] = {}
    if explicit_char_mix:
        if None in (short_chars, long_chars, short_count, long_count):
            raise ValueError(
                "short_chars, long_chars, short_count, and long_count must be provided together"
            )
        if short_count < 0 or long_count < 0:
            raise ValueError("short_count and long_count must be non-negative")
        if short_count + long_count != num_requests:
            raise ValueError("short_count + long_count must equal num_requests")
        if short_chars <= 0 or long_chars <= 0:
            raise ValueError("short_chars and long_chars must be positive")
        profile_names = ["short"] * int(short_count) + ["long"] * int(long_count)
        target_chars_by_kind = {"short": int(short_chars), "long": int(long_chars)}
        rng.shuffle(profile_names)
    else:
        profile_names = _weighted_profile_names(num_requests, rng)
    records: List[JsonDict] = []

    for index, profile_name in enumerate(profile_names):
        profile = profile_by_name[profile_name]
        request_id = f"trace-{index:04d}"
        prompt = _make_prompt(
            profile,
            index,
            rng,
            target_chars=target_chars_by_kind.get(profile_name),
        )
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": profile.max_tokens,
            "temperature": temperature,
            "stream": stream,
            "custom_params": {
                "trace_request_id": request_id,
                "pd_flip_slo": {
                    "ttft_seconds": profile.ttft_slo_s,
                    "tpot_seconds": profile.tpot_slo_s,
                },
            },
        }
        records.append(
            {
                "request_id": request_id,
                "arrival_offset_s": round(index * interval_seconds, 6),
                "prompt_kind": profile.name,
                "prompt_chars": len(prompt),
                "prompt_words": len(prompt.split()),
                "max_tokens": profile.max_tokens,
                "stream": stream,
                "ttft_slo_s": profile.ttft_slo_s,
                "tpot_slo_s": profile.tpot_slo_s,
                "body": body,
            }
        )

    return records


def _weighted_profile_names(num_requests: int, rng: random.Random) -> List[str]:
    names: List[str] = []
    total_weight = sum(profile.count_weight for profile in PROFILES)
    full_cycles, remainder = divmod(num_requests, total_weight)
    for _ in range(full_cycles):
        for profile in PROFILES:
            names.extend([profile.name] * profile.count_weight)
    if remainder:
        weighted = [
            profile.name
            for profile in PROFILES
            for _ in range(profile.count_weight)
        ]
        names.extend(weighted[:remainder])
    rng.shuffle(names)
    return names


def _make_prompt(
    profile: PromptProfile,
    index: int,
    rng: random.Random,
    target_chars: Optional[int] = None,
) -> str:
    topics = [
        "prefill pressure",
        "decode latency",
        "KV migration",
        "router draining",
        "request admission",
        "SLO accounting",
        "rollback checks",
        "observability",
    ]
    rng.shuffle(topics)
    block = (
        "Context block {block}: The serving cluster is running disaggregated "
        "prefill and decode workers. Inspect {topic_a}, {topic_b}, and "
        "{topic_c}. Mention how TTFT and TPOT should be interpreted for this "
        "request, and include one operational risk plus one mitigation."
    )
    blocks = [
        block.format(
            block=block_index,
            topic_a=topics[block_index % len(topics)],
            topic_b=topics[(block_index + 2) % len(topics)],
            topic_c=topics[(block_index + 5) % len(topics)],
        )
        for block_index in range(profile.repeat_blocks)
    ]
    prompt = (
        f"Request {index} is a {profile.name} workload sample for a PD runtime "
        f"role flip experiment. {profile.answer_style}\n\n"
        + "\n".join(blocks)
    )
    return _fit_prompt_to_chars(prompt, target_chars=target_chars)


def _fit_prompt_to_chars(prompt: str, target_chars: Optional[int]) -> str:
    if target_chars is None:
        return prompt
    if target_chars <= 0:
        raise ValueError("target_chars must be positive")
    if len(prompt) >= target_chars:
        return prompt[:target_chars]
    filler = (
        "\n\nDeterministic padding for KV migration trace. "
        "This sentence keeps prompt length stable while preserving ASCII text. "
    )
    pieces = [prompt]
    remaining = target_chars - len(prompt)
    while remaining > 0:
        chunk = filler[:remaining]
        pieces.append(chunk)
        remaining -= len(chunk)
    return "".join(pieces)


def write_trace(trace: Sequence[JsonDict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "trace_requests.jsonl"
    csv_path = output_dir / "trace_requests.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in trace:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    fields = [
        "request_id",
        "arrival_offset_s",
        "prompt_kind",
        "prompt_chars",
        "prompt_words",
        "max_tokens",
        "stream",
        "ttft_slo_s",
        "tpot_slo_s",
        "body",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in trace:
            row = {key: record.get(key) for key in fields}
            row["body"] = json.dumps(record["body"], sort_keys=True)
            writer.writerow(row)


def load_trace(path: Path) -> List[JsonDict]:
    records: List[JsonDict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"{path} contains no trace records")
    return records


def compute_metrics(
    record: JsonDict,
    *,
    scheduled_monotonic: float,
    start_monotonic: float,
    first_token_monotonic: Optional[float],
    token_monotonic_times: Sequence[float],
    end_monotonic: float,
    status: str,
    error: Optional[str],
) -> JsonDict:
    intervals = [
        max(0.0, token_monotonic_times[index] - token_monotonic_times[index - 1])
        for index in range(1, len(token_monotonic_times))
    ]
    ttft_s = (
        max(0.0, first_token_monotonic - start_monotonic)
        if first_token_monotonic is not None
        else None
    )
    avg_tpot_s = sum(intervals) / len(intervals) if intervals else None
    p50_tpot_s = _percentile_nearest_rank(intervals, 50) if intervals else None
    p95_tpot_s = _percentile_nearest_rank(intervals, 95) if intervals else None
    max_tpot_s = max(intervals) if intervals else None

    ttft_slo_s = float(record["ttft_slo_s"])
    tpot_slo_s = float(record["tpot_slo_s"])
    good_tpot_intervals = sum(1 for value in intervals if value <= tpot_slo_s)
    total_tpot_intervals = len(intervals)
    ttft_met = ttft_s is not None and ttft_s <= ttft_slo_s
    tpot_avg_met = avg_tpot_s is not None and avg_tpot_s <= tpot_slo_s
    tpot_p95_met = p95_tpot_s is not None and p95_tpot_s <= tpot_slo_s

    return {
        "request_id": record["request_id"],
        "prompt_kind": record.get("prompt_kind"),
        "arrival_offset_s": record.get("arrival_offset_s"),
        "scheduled_monotonic": scheduled_monotonic,
        "start_monotonic": start_monotonic,
        "first_token_monotonic": first_token_monotonic,
        "end_monotonic": end_monotonic,
        "queue_delay_s": max(0.0, start_monotonic - scheduled_monotonic),
        "latency_s": max(0.0, end_monotonic - start_monotonic),
        "ttft_slo_s": ttft_slo_s,
        "tpot_slo_s": tpot_slo_s,
        "ttft_s": ttft_s,
        "avg_tpot_s": avg_tpot_s,
        "p50_tpot_s": p50_tpot_s,
        "p95_tpot_s": p95_tpot_s,
        "max_tpot_s": max_tpot_s,
        "completion_chunks": len(token_monotonic_times),
        "good_tpot_intervals": good_tpot_intervals,
        "total_tpot_intervals": total_tpot_intervals,
        "tpot_interval_attainment": (
            good_tpot_intervals / total_tpot_intervals
            if total_tpot_intervals
            else None
        ),
        "ttft_met": ttft_met,
        "tpot_avg_met": tpot_avg_met,
        "tpot_p95_met": tpot_p95_met,
        "all_met": ttft_met and tpot_avg_met,
        "status": status,
        "error": error,
    }


def _percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    rank = max(1, math.ceil((percentile / 100.0) * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def replay_trace(
    *,
    trace: Sequence[JsonDict],
    router_url: str,
    mode: str,
    output_dir: Path,
    ledger_path: Optional[Path],
    timeout_seconds: float,
    max_workers: int,
    api_key: Optional[str],
) -> JsonDict:
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    if ledger_path is not None:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text("", encoding="utf-8")

    ledger_lock = threading.Lock()
    run_started_monotonic = time.monotonic()
    run_started_wall = time.time()
    futures = []
    max_workers = max(1, max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for record in trace:
            scheduled = run_started_monotonic + float(record["arrival_offset_s"])
            sleep_s = scheduled - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            futures.append(
                executor.submit(
                    _send_one_request,
                    record,
                    router_url,
                    scheduled,
                    timeout_seconds,
                    ledger_path,
                    ledger_lock,
                    api_key,
                    mode,
                )
            )

        results = [future.result() for future in as_completed(futures)]

    results.sort(key=lambda item: item["metrics"]["request_id"])
    metrics = [item["metrics"] for item in results]
    responses = [item["response"] for item in results if item.get("response")]
    errors = [item["error_record"] for item in results if item.get("error_record")]
    interval_rows = [
        row
        for item in results
        for row in item.get("tpot_intervals", [])
    ]

    _write_mode_outputs(mode_dir, metrics, responses, errors, interval_rows)
    summary = summarize_metrics(metrics)
    summary.update(
        {
            "mode": mode,
            "router_url": router_url,
            "run_started_wall": run_started_wall,
            "run_elapsed_s": time.monotonic() - run_started_monotonic,
            "trace_requests": len(trace),
            "ledger_path": str(ledger_path) if ledger_path is not None else None,
        }
    )
    with (mode_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    _write_summary_csv(mode_dir / "slo_summary.csv", [summary])
    return summary


def _send_one_request(
    record: JsonDict,
    router_url: str,
    scheduled_monotonic: float,
    timeout_seconds: float,
    ledger_path: Optional[Path],
    ledger_lock: threading.Lock,
    api_key: Optional[str],
    mode: str,
) -> JsonDict:
    start_monotonic = time.monotonic()
    start_wall = time.time()
    first_token_monotonic: Optional[float] = None
    token_times: List[float] = []
    intervals: List[float] = []
    content_parts: List[str] = []
    finish_reason: Optional[str] = None
    upstream_request_id: Optional[str] = None
    status = "completed"
    error_text: Optional[str] = None

    try:
        request = _build_http_request(
            router_url.rstrip("/") + "/v1/chat/completions",
            record["body"],
            api_key,
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if bool(record.get("body", {}).get("stream", True)):
                for data in _iter_sse_data(response):
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    upstream_request_id = chunk.get("id") or upstream_request_id
                    choice = (chunk.get("choices") or [{}])[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    token_text = _extract_stream_text(choice)
                    if not token_text:
                        continue
                    now = time.monotonic()
                    if first_token_monotonic is None:
                        first_token_monotonic = now
                    elif token_times:
                        intervals.append(max(0.0, now - token_times[-1]))
                    token_times.append(now)
                    content_parts.append(token_text)
                    _append_ledger(
                        ledger_path,
                        ledger_lock,
                        record,
                        mode,
                        "running",
                        start_monotonic,
                        first_token_monotonic,
                        token_times,
                        intervals,
                        None,
                    )
            else:
                raw = response.read().decode("utf-8", errors="replace")
                chunk = json.loads(raw)
                upstream_request_id = chunk.get("id") or upstream_request_id
                choice = (chunk.get("choices") or [{}])[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                token_text = _extract_non_stream_text(choice)
                now = time.monotonic()
                if token_text:
                    first_token_monotonic = now
                    token_times.append(now)
                    content_parts.append(token_text)
                    _append_ledger(
                        ledger_path,
                        ledger_lock,
                        record,
                        mode,
                        "running",
                        start_monotonic,
                        first_token_monotonic,
                        token_times,
                        intervals,
                        None,
                    )
    except urllib.error.HTTPError as exc:
        status = "http_error"
        body = exc.read().decode("utf-8", errors="replace")
        error_text = f"HTTP {exc.code}: {body}"
    except Exception as exc:
        status = "error"
        error_text = f"{type(exc).__name__}: {exc}"
        error_text += "\n" + traceback.format_exc()

    end_monotonic = time.monotonic()
    _append_ledger(
        ledger_path,
        ledger_lock,
        record,
        mode,
        status,
        start_monotonic,
        first_token_monotonic,
        token_times,
        intervals,
        error_text,
    )
    metrics = compute_metrics(
        record,
        scheduled_monotonic=scheduled_monotonic,
        start_monotonic=start_monotonic,
        first_token_monotonic=first_token_monotonic,
        token_monotonic_times=token_times,
        end_monotonic=end_monotonic,
        status=status,
        error=error_text,
    )
    metrics["start_wall"] = start_wall
    metrics["end_wall"] = time.time()
    metrics["upstream_request_id"] = upstream_request_id
    metrics["finish_reason"] = finish_reason

    response_record = {
        "request_id": record["request_id"],
        "upstream_request_id": upstream_request_id,
        "status": status,
        "finish_reason": finish_reason,
        "content": "".join(content_parts),
    }
    error_record = (
        {
            "request_id": record["request_id"],
            "status": status,
            "error": error_text,
        }
        if error_text
        else None
    )
    interval_rows = [
        {
            "request_id": record["request_id"],
            "interval_index": index,
            "interval_s": value,
            "tpot_slo_s": float(record["tpot_slo_s"]),
            "met": value <= float(record["tpot_slo_s"]),
        }
        for index, value in enumerate(intervals, 1)
    ]
    return {
        "metrics": metrics,
        "response": response_record,
        "error_record": error_record,
        "tpot_intervals": interval_rows,
    }


def _build_http_request(url: str, body: JsonDict, api_key: Optional[str]) -> urllib.request.Request:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    return request


def _iter_sse_data(response: Any) -> Iterable[str]:
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        yield line[5:].strip()


def _extract_stream_text(choice: JsonDict) -> str:
    delta = choice.get("delta")
    if isinstance(delta, dict):
        parts = [
            delta.get("content"),
            delta.get("reasoning_content"),
        ]
        return "".join(part for part in parts if isinstance(part, str))
    text = choice.get("text")
    return text if isinstance(text, str) else ""


def _extract_non_stream_text(choice: JsonDict) -> str:
    message = choice.get("message")
    if isinstance(message, dict):
        parts = [
            message.get("reasoning_content"),
            message.get("content"),
        ]
        return "".join(part for part in parts if isinstance(part, str))
    return _extract_stream_text(choice)


def _parse_bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("expected true or false, got %s" % value)


def _append_ledger(
    ledger_path: Optional[Path],
    ledger_lock: threading.Lock,
    record: JsonDict,
    mode: str,
    status: str,
    start_monotonic: float,
    first_token_monotonic: Optional[float],
    token_times: Sequence[float],
    intervals: Sequence[float],
    error: Optional[str],
) -> None:
    if ledger_path is None:
        return
    ttft_s = (
        first_token_monotonic - start_monotonic
        if first_token_monotonic is not None
        else None
    )
    tpot_slo_s = float(record["tpot_slo_s"])
    ledger = {
        "request_id": record["request_id"],
        "mode": mode,
        "event_time": time.monotonic(),
        "start_time": start_monotonic,
        "first_token_time": first_token_monotonic,
        "last_token_time": token_times[-1] if token_times else None,
        "completion_tokens": len(token_times),
        "ttft_slo_seconds": float(record["ttft_slo_s"]),
        "tpot_slo_seconds": tpot_slo_s,
        "ttft_seconds": ttft_s,
        "ttft_met": (
            ttft_s is not None and ttft_s <= float(record["ttft_slo_s"])
        ),
        "good_tpot_intervals": sum(1 for value in intervals if value <= tpot_slo_s),
        "total_tpot_intervals": len(intervals),
        "status": status,
    }
    if error is not None:
        ledger["error"] = error
    with ledger_lock:
        with ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ledger, sort_keys=True) + "\n")


def _write_mode_outputs(
    mode_dir: Path,
    metrics: Sequence[JsonDict],
    responses: Sequence[JsonDict],
    errors: Sequence[JsonDict],
    interval_rows: Sequence[JsonDict],
) -> None:
    _write_jsonl(mode_dir / "request_metrics.jsonl", metrics)
    _write_jsonl(mode_dir / "responses.jsonl", responses)
    _write_jsonl(mode_dir / "errors.jsonl", errors)

    _write_csv(
        mode_dir / "ttft.csv",
        metrics,
        [
            "request_id",
            "prompt_kind",
            "arrival_offset_s",
            "ttft_slo_s",
            "ttft_s",
            "ttft_met",
            "status",
            "error",
        ],
    )
    _write_csv(
        mode_dir / "tpot.csv",
        metrics,
        [
            "request_id",
            "prompt_kind",
            "arrival_offset_s",
            "tpot_slo_s",
            "avg_tpot_s",
            "p50_tpot_s",
            "p95_tpot_s",
            "max_tpot_s",
            "good_tpot_intervals",
            "total_tpot_intervals",
            "tpot_interval_attainment",
            "tpot_avg_met",
            "tpot_p95_met",
            "status",
            "error",
        ],
    )
    _write_csv(
        mode_dir / "tpot_tokens.csv",
        interval_rows,
        ["request_id", "interval_index", "interval_s", "tpot_slo_s", "met"],
    )
    _write_csv(
        mode_dir / "slo_attainment.csv",
        metrics,
        [
            "request_id",
            "prompt_kind",
            "arrival_offset_s",
            "status",
            "ttft_slo_s",
            "ttft_s",
            "ttft_met",
            "tpot_slo_s",
            "avg_tpot_s",
            "p95_tpot_s",
            "tpot_avg_met",
            "tpot_p95_met",
            "good_tpot_intervals",
            "total_tpot_intervals",
            "tpot_interval_attainment",
            "all_met",
            "error",
        ],
    )


def summarize_metrics(metrics: Sequence[JsonDict]) -> JsonDict:
    request_count = len(metrics)
    completed = [m for m in metrics if m.get("status") == "completed"]
    ttft_met_count = sum(1 for m in metrics if m.get("ttft_met"))
    tpot_avg_met_count = sum(1 for m in metrics if m.get("tpot_avg_met"))
    tpot_p95_met_count = sum(1 for m in metrics if m.get("tpot_p95_met"))
    all_met_count = sum(1 for m in metrics if m.get("all_met"))
    tpot_good = sum(int(m.get("good_tpot_intervals") or 0) for m in metrics)
    tpot_total = sum(int(m.get("total_tpot_intervals") or 0) for m in metrics)
    return {
        "request_count": request_count,
        "completed_count": len(completed),
        "error_count": request_count - len(completed),
        "ttft_met_count": ttft_met_count,
        "ttft_attainment": _rate(ttft_met_count, request_count),
        "tpot_avg_met_count": tpot_avg_met_count,
        "tpot_avg_attainment": _rate(tpot_avg_met_count, request_count),
        "tpot_p95_met_count": tpot_p95_met_count,
        "tpot_p95_attainment": _rate(tpot_p95_met_count, request_count),
        "tpot_good_intervals": tpot_good,
        "tpot_total_intervals": tpot_total,
        "tpot_interval_attainment": _rate(tpot_good, tpot_total),
        "all_met_count": all_met_count,
        "all_attainment": _rate(all_met_count, request_count),
    }


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def _write_jsonl(path: Path, rows: Sequence[JsonDict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[JsonDict], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_summary_csv(path: Path, summaries: Sequence[JsonDict]) -> None:
    fields = [
        "mode",
        "request_count",
        "completed_count",
        "error_count",
        "ttft_met_count",
        "ttft_attainment",
        "tpot_avg_met_count",
        "tpot_avg_attainment",
        "tpot_p95_met_count",
        "tpot_p95_attainment",
        "tpot_good_intervals",
        "tpot_total_intervals",
        "tpot_interval_attainment",
        "all_met_count",
        "all_attainment",
        "run_elapsed_s",
        "ledger_path",
    ]
    _write_csv(path, summaries, fields)


def combine_summaries(output_dir: Path, modes: Sequence[str]) -> None:
    summaries = []
    for mode in modes:
        path = output_dir / mode / "summary.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            summaries.append(json.load(f))
    _write_summary_csv(output_dir / "slo_summary.csv", summaries)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    generate = subparsers.add_parser("generate")
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--model", required=True)
    generate.add_argument("--num-requests", type=int, default=200)
    generate.add_argument("--interval-seconds", type=float, default=1.0)
    generate.add_argument("--seed", type=int, default=20260707)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--stream", type=_parse_bool, default=True)
    generate.add_argument("--short-chars", type=int, default=None)
    generate.add_argument("--long-chars", type=int, default=None)
    generate.add_argument("--short-count", type=int, default=None)
    generate.add_argument("--long-count", type=int, default=None)

    replay = subparsers.add_parser("replay")
    replay.add_argument("--trace-jsonl", required=True)
    replay.add_argument("--router-url", required=True)
    replay.add_argument("--mode", required=True)
    replay.add_argument("--output-dir", required=True)
    replay.add_argument("--ledger-path", default=None)
    replay.add_argument("--timeout-seconds", type=float, default=900.0)
    replay.add_argument("--max-workers", type=int, default=256)
    replay.add_argument("--api-key", default=None)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output-dir", required=True)
    summarize.add_argument("--modes", nargs="+", default=["baseline", "state_machine"])

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    if args.command == "generate":
        output_dir = Path(args.output_dir)
        trace = build_trace(
            num_requests=args.num_requests,
            interval_seconds=args.interval_seconds,
            model=args.model,
            seed=args.seed,
            temperature=args.temperature,
            stream=args.stream,
            short_chars=args.short_chars,
            long_chars=args.long_chars,
            short_count=args.short_count,
            long_count=args.long_count,
        )
        write_trace(trace, output_dir)
        print(json.dumps({"trace_requests": len(trace), "output_dir": str(output_dir)}))
        return 0

    if args.command == "replay":
        summary = replay_trace(
            trace=load_trace(Path(args.trace_jsonl)),
            router_url=args.router_url,
            mode=args.mode,
            output_dir=Path(args.output_dir),
            ledger_path=Path(args.ledger_path) if args.ledger_path else None,
            timeout_seconds=args.timeout_seconds,
            max_workers=args.max_workers,
            api_key=args.api_key,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["error_count"] == 0 else 1

    if args.command == "summarize":
        combine_summaries(Path(args.output_dir), args.modes)
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
