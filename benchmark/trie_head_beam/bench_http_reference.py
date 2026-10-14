#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import math
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
from common import require_single_visible_gpu
from transformers import AutoTokenizer


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 3),
        "p50": round(percentile(values, 0.50), 3),
        "p90": round(percentile(values, 0.90), 3),
        "p95": round(percentile(values, 0.95), 3),
        "p99": round(percentile(values, 0.99), 3),
        "max": round(max(values), 3),
    }


def get_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def post_json(
    url: str, payload: dict[str, Any], timeout: float = 300.0
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


class ConstraintValidator:
    def __init__(
        self,
        valid_keys_path: Path,
        *,
        token_start: int,
        codebook_size: int,
        sid_length: int,
    ) -> None:
        self.valid_keys = np.load(valid_keys_path, mmap_mode="r")
        self.token_start = token_start
        self.codebook_size = codebook_size
        self.sid_length = sid_length

    def validate(self, rows: list[list[int]], expected: int) -> dict[str, int]:
        wrong_result_count = int(len(rows) != expected)
        length_invalid = sum(len(row) != self.sid_length for row in rows)
        complete_rows = [row for row in rows if len(row) == self.sid_length]
        if not complete_rows:
            return {
                "result_count": len(rows),
                "invalid_sid_count": length_invalid,
                "duplicate_sid_count": 0,
                "wrong_result_count": wrong_result_count,
            }

        tokens = np.asarray(complete_rows, dtype=np.int64)
        offsets = self.token_start + (
            np.arange(self.sid_length, dtype=np.int64) * self.codebook_size
        )
        codes = tokens - offsets
        in_range = np.all(
            (codes >= 0) & (codes < self.codebook_size),
            axis=1,
        )
        keys = np.zeros(codes.shape[0], dtype=np.uint64)
        for column in codes.T:
            keys = keys * self.codebook_size + column.astype(np.uint64)
        positions = np.searchsorted(self.valid_keys, keys)
        bounded = positions < self.valid_keys.size
        valid = np.zeros(keys.size, dtype=np.bool_)
        valid[bounded] = self.valid_keys[positions[bounded]] == keys[bounded]
        valid &= in_range
        duplicates = int(keys.size - np.unique(keys).size)
        return {
            "result_count": len(rows),
            "invalid_sid_count": length_invalid + int((~valid).sum()),
            "duplicate_sid_count": duplicates,
            "wrong_result_count": wrong_result_count,
        }


class GpuMonitor:
    def __init__(self, physical_gpu: int) -> None:
        self.physical_gpu = physical_gpu
        self.samples: list[int] = []
        self._error: Exception | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)
            if self._thread.is_alive():
                raise RuntimeError("GPU memory monitor did not stop")
        if self._error is not None:
            raise RuntimeError("GPU memory monitoring failed") from self._error

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={self.physical_gpu}",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                self.samples.append(int(output.strip().splitlines()[0]))
            except (OSError, subprocess.SubprocessError, IndexError, ValueError) as exc:
                self._error = exc
                return
            self._stop.wait(0.1)


def load_input_rows(
    prompts_path: Path, model_dir: Path, needed: int
) -> list[dict[str, Any]]:
    source_rows = json.loads(prompts_path.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("prompt source must be a non-empty JSON list")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
    )
    rows = []
    for index in range(needed):
        source = source_rows[index % len(source_rows)]
        prompt = source["prompt"]
        input_ids = tokenizer.encode(prompt)
        rows.append(
            {
                "request_index": index,
                "source_dataset_index": source.get("dataset_index"),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "input_ids": input_ids,
            }
        )
    return rows


def extract_beams(response: dict[str, Any]) -> list[list[int]]:
    beams = response.get("meta_info", {}).get("beam_results")
    if not isinstance(beams, list):
        raise ValueError("Generation response must contain meta_info.beam_results")
    return [
        [int(token) for token in row.get("output_ids", ())]
        for row in beams
        if isinstance(row, dict)
    ]


async def one_request(
    base_url: str,
    row: dict[str, Any],
    beams: int,
    timeout: float,
    request_prefix: str,
) -> tuple[dict[str, Any], float]:
    payload = {
        "rid": f"{request_prefix}-{row['request_index']}",
        "input_ids": row["input_ids"],
        "sampling_params": {
            "beam_width": beams,
            "max_new_tokens": 3,
            "n": beams,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        "stream": False,
    }
    started = time.perf_counter()
    response = await asyncio.to_thread(
        post_json,
        f"{base_url}/generate",
        payload,
        timeout,
    )
    return response, (time.perf_counter() - started) * 1000.0


async def run(args: argparse.Namespace) -> None:
    validator = ConstraintValidator(
        args.valid_sid_keys,
        token_start=args.token_start,
        codebook_size=args.codebook_size,
        sid_length=args.sid_length,
    )
    input_rows = load_input_rows(
        args.prompts,
        args.model_dir,
        args.requests + args.warmup,
    )
    timed_rows = input_rows[: args.requests]
    warmup_rows = input_rows[args.requests :]

    build = get_json(f"{args.base_url}/get_server_info")
    status_before = build
    for offset in range(0, len(warmup_rows), args.concurrency):
        await asyncio.gather(
            *(
                one_request(
                    args.base_url,
                    row,
                    args.beams,
                    args.timeout,
                    f"{args.label}-warmup",
                )
                for row in warmup_rows[offset : offset + args.concurrency]
            )
        )

    monitor = GpuMonitor(args.physical_gpu)
    monitor.start()
    latencies: list[float] = []
    invalid_sid_count = 0
    duplicate_sid_count = 0
    wrong_result_count = 0
    error_count = 0
    output_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for offset in range(0, len(timed_rows), args.concurrency):
        wave = timed_rows[offset : offset + args.concurrency]
        results = await asyncio.gather(
            *(
                one_request(
                    args.base_url,
                    row,
                    args.beams,
                    args.timeout,
                    args.label,
                )
                for row in wave
            ),
            return_exceptions=True,
        )
        for row, result in zip(wave, results, strict=True):
            if isinstance(result, Exception):
                error_count += 1
                output_rows.append(
                    {
                        "request_index": row["request_index"],
                        "source_dataset_index": row["source_dataset_index"],
                        "prompt_sha256": row["prompt_sha256"],
                        "input_ids": row["input_ids"],
                        "error": f"{type(result).__name__}: {result}",
                    }
                )
                continue
            response, latency_ms = result
            beams = extract_beams(response)
            latencies.append(latency_ms)
            output_rows.append(
                {
                    "request_index": row["request_index"],
                    "source_dataset_index": row["source_dataset_index"],
                    "prompt_sha256": row["prompt_sha256"],
                    "input_ids": row["input_ids"],
                    "latency_ms": round(latency_ms, 3),
                    "sid_tokens": beams,
                }
            )
        print(f"{len(output_rows)}/{args.requests} complete", flush=True)
    wall_seconds = time.perf_counter() - started
    monitor.stop()
    status_after = get_json(f"{args.base_url}/get_server_info")

    for row in output_rows:
        if "error" in row:
            continue
        validity = validator.validate(row["sid_tokens"], args.beams)
        row.update(validity)
        invalid_sid_count += validity["invalid_sid_count"]
        duplicate_sid_count += validity["duplicate_sid_count"]
        wrong_result_count += validity["wrong_result_count"]

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.raw_output, "wt", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    successful = len(latencies)
    summary = {
        "label": args.label,
        "engine": args.engine_label,
        "requests": args.requests,
        "successful_requests": successful,
        "warmup_requests": args.warmup,
        "concurrency": args.concurrency,
        "beams": args.beams,
        "wall_seconds": round(wall_seconds, 3),
        "completion_qps": round(successful / wall_seconds, 4),
        "latency_ms": latency_summary(latencies) if latencies else None,
        "error_count": error_count,
        "invalid_sid_count": invalid_sid_count,
        "duplicate_sid_count": duplicate_sid_count,
        "requests_with_wrong_result_count": wrong_result_count,
        "gpu_memory_used_mib": {
            "max_observed": max(monitor.samples) if monitor.samples else None,
            "samples": len(monitor.samples),
        },
        "inputs": {
            "prompt_source": str(args.prompts),
            "model_dir": str(args.model_dir),
            "constraint_source": "external valid-SID index",
            "valid_sid_keys": str(args.valid_sid_keys),
            "token_start": args.token_start,
            "codebook_size": args.codebook_size,
            "sid_length": args.sid_length,
            "ordered_prompt_count": len(timed_rows),
            "minimum_input_tokens": min(len(row["input_ids"]) for row in timed_rows),
            "maximum_input_tokens": max(len(row["input_ids"]) for row in timed_rows),
        },
        "server_build": build,
        "server_status_before": status_before,
        "server_status_after": status_after,
        "raw_output": str(args.raw_output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark an existing HTTP reference implementation."
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--engine-label", default="external-http-reference")
    parser.add_argument("--base-url", default="http://127.0.0.1:9455")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--valid-sid-keys", type=Path, required=True)
    parser.add_argument("--token-start", type=int, default=151669)
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument("--sid-length", type=int, default=3)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--beams", type=int, default=2000)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    require_single_visible_gpu(args.physical_gpu)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
