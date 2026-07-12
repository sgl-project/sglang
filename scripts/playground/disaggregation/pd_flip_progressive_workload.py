#!/usr/bin/env python3
"""Generate auditable progressive PD-flip traffic and request ledgers.

The selected mode controls only prompt warm-up/input strategy. Actual HiCache
mode acceptance must come from worker ``request_measurements.stitch_mode``.
"""

import argparse
import json
import random
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
        cold = " ".join("z%08x" % rng.getrandbits(32) for _ in range(768))
        return PromptPlan((), cold)
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
        with urllib.request.urlopen(
            self._request(payload), timeout=self.timeout_seconds
        ) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                chunk = json.loads(data)
                text = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if text is None:
                    continue
                sequences.append({"seq": len(sequences), "text": str(text)})
                arrivals.append(time.monotonic())
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


def request_payload(model: str, prompt: str, max_tokens: int, stream: bool) -> JsonDict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
    }


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_prompt_plan(args.mode, args.seed)
    client = OpenAIClient(args.base_url, args.api_key, args.timeout_seconds)
    started = time.monotonic()
    metrics: List[JsonDict] = []
    errors: List[JsonDict] = []
    lock = threading.Lock()

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
                request_payload(args.model, prompt, max_tokens, False)
            )
            elapsed = time.monotonic() - begin
            row = {
                "request_id": request_id,
                "prompt_kind": kind,
                "mode": args.mode,
                "decision_path": args.decision_path,
                "arrival_offset_s": begin - started,
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
                "output_sequences": [{"seq": 0, "text": output_text}],
                "output_text": output_text,
            }
            with lock:
                metrics.append(row)
        except Exception as exc:
            record_error(request_id, kind, exc)

    for prompt in plan.warmup_prompts:
        run_nonstream("warmup", prompt, args.warmup_max_tokens)

    def run_active() -> None:
        request_id = str(uuid.uuid4())
        begin = time.monotonic()
        try:
            status, sequences, arrivals = client.stream(
                request_payload(args.model, plan.active_prompt, args.long_max_tokens, True)
            )
            ttft = arrivals[0] - begin if arrivals else None
            gaps = [right - left for left, right in zip(arrivals, arrivals[1:])]
            avg_tpot = statistics.mean(gaps) if gaps else None
            row = {
                "request_id": request_id,
                "prompt_kind": "active_long",
                "mode": args.mode,
                "decision_path": args.decision_path,
                "arrival_offset_s": begin - started,
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
                "output_sequences": sequences,
                "output_text": "".join(item["text"] for item in sequences),
            }
            with lock:
                metrics.append(row)
        except Exception as exc:
            record_error(request_id, "active_long", exc)

    active = threading.Thread(target=run_active, name="pd-flip-active-request")
    active.start()
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

    metrics.sort(key=lambda row: float(row["arrival_offset_s"]))
    write_jsonl(output_dir / "request_metrics.jsonl", metrics)
    write_jsonl(output_dir / "errors.jsonl", errors)
    config = {
        "base_url": args.base_url,
        "model": args.model,
        "mode": args.mode,
        "decision_path": args.decision_path,
        "seed": args.seed,
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
    parser.add_argument("--api-key", required=True)
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
    parser.add_argument("--ttft-slo-seconds", type=float, default=0.2)
    parser.add_argument("--tpot-slo-seconds", type=float, default=0.02)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
