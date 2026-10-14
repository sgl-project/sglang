#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from common import require_single_visible_gpu


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
        offsets = self.token_start + np.arange(self.sid_length) * self.codebook_size
        codes = tokens - offsets[None, :]
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
        return {
            "result_count": len(rows),
            "invalid_sid_count": length_invalid + int((~valid).sum()),
            "duplicate_sid_count": int(keys.size - np.unique(keys).size),
            "wrong_result_count": wrong_result_count,
        }


class GpuMonitor:
    def __init__(self, physical_gpu: int) -> None:
        self.physical_gpu = physical_gpu
        self.samples: list[int] = []
        self._error: Exception | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=6)
        if self._thread.is_alive():
            raise RuntimeError("GPU memory monitor did not stop")
        if self._error is not None:
            raise RuntimeError("GPU memory monitoring failed") from self._error

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.physical_gpu}",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.samples.append(int(result.stdout.strip().splitlines()[0]))
            except (OSError, subprocess.SubprocessError, IndexError, ValueError) as exc:
                self._error = exc
                return
            self._stop.wait(0.1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_input_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append(
                {
                    "request_index": len(rows),
                    "source_dataset_index": row.get("source_dataset_index"),
                    "prompt_sha256": row.get("prompt_sha256"),
                    "input_ids": [int(token) for token in row["input_ids"]],
                }
            )
            if len(rows) == limit:
                break
    if len(rows) != limit:
        raise ValueError(f"Expected {limit} inputs in {path}, found {len(rows)}")
    return rows


def extract_sid_tokens(raw: object) -> list[list[int]]:
    if not isinstance(raw, dict):
        raise TypeError("Generation response must be an object")
    beams = raw.get("meta_info", {}).get("beam_results")
    if not isinstance(beams, list):
        raise ValueError("Generation response must contain meta_info.beam_results")
    return [
        [
            int(token)
            for token in (
                beam.get("output_ids")
                or beam.get("meta_info", {}).get("output_ids")
                or []
            )
        ]
        for beam in beams
    ]


def latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(percent: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * percent / 100.0
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return {
        "mean": round(statistics.fmean(values), 3),
        "p50": round(percentile(50), 3),
        "p95": round(percentile(95), 3),
        "p99": round(percentile(99), 3),
        "max": round(max(values), 3),
    }


def run(args: argparse.Namespace) -> None:
    from sglang import Engine

    validator = ConstraintValidator(
        args.valid_sid_keys,
        token_start=args.token_start,
        codebook_size=args.codebook_size,
        sid_length=args.sid_length,
    )
    correctness_rows = load_input_rows(
        args.correctness_inputs, args.correctness_requests
    )
    performance_rows = load_input_rows(
        args.performance_inputs, args.performance_requests
    )
    graph_buckets = [args.beams * concurrency for concurrency in args.graph_concurrency]
    graph_kwargs = (
        {"disable_cuda_graph": True}
        if args.disable_cuda_graph
        else {
            "cuda_graph_max_bs_decode": max(graph_buckets),
            "cuda_graph_bs_decode": graph_buckets,
            "disable_prefill_cuda_graph": True,
        }
    )
    engine_kwargs = {
        "model_path": str(args.model),
        "tokenizer_path": str(args.model),
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "mem_fraction_static": args.mem_fraction_static,
        "max_running_requests": args.max_running_requests,
        "context_length": args.context_length,
        "tp_size": 1,
        "dp_size": 1,
        "attention_backend": "flashinfer",
        "schedule_policy": "fcfs",
        "random_seed": 0,
        **graph_kwargs,
    }
    print(json.dumps({"engine_kwargs": engine_kwargs}, indent=2), flush=True)
    startup_started = time.perf_counter()
    engine = Engine(**engine_kwargs)
    engine_init_seconds = time.perf_counter() - startup_started
    print(f"Engine initialized in {engine_init_seconds:.3f}s", flush=True)
    sampling_params = {
        "beam_width": args.beams,
        "max_new_tokens": args.sid_length,
        "n": args.beams,
    }

    async def generate(row: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            raw = await engine.async_generate(
                input_ids=row["input_ids"],
                sampling_params=sampling_params,
            )
            return {
                **row,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "sid_tokens": extract_sid_tokens(raw),
            }
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            return {
                **row,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }

    async def execute(
        rows: list[dict[str, Any]], concurrency: int
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(concurrency)

        async def limited(row: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await generate(row)

        return await asyncio.gather(*(limited(row) for row in rows))

    async def warmup(rows: list[dict[str, Any]], concurrency: int) -> None:
        results = await execute(rows, concurrency)
        failures = [row for row in results if "error" in row]
        if failures:
            raise RuntimeError(f"Warmup failed: {failures[0]['error']}")

    def run_point(
        label: str,
        rows: list[dict[str, Any]],
        concurrency: int,
        warmup_count: int,
    ) -> dict[str, Any]:
        warmup_offset = len(rows) % 198
        warmup_rows = [
            dict(rows[(warmup_offset + index) % len(rows)])
            for index in range(warmup_count)
        ]
        engine.loop.run_until_complete(warmup(warmup_rows, concurrency))
        engine.freeze_gc()

        monitor = GpuMonitor(args.physical_gpu)
        monitor.start()
        started = time.perf_counter()
        output_rows = engine.loop.run_until_complete(execute(rows, concurrency))
        wall_seconds = time.perf_counter() - started
        monitor.stop()

        invalid_sid_count = 0
        duplicate_sid_count = 0
        wrong_result_count = 0
        successful_latencies = []
        for row in output_rows:
            if "error" in row:
                continue
            validity = validator.validate(row["sid_tokens"], args.beams)
            row.update(validity)
            invalid_sid_count += validity["invalid_sid_count"]
            duplicate_sid_count += validity["duplicate_sid_count"]
            wrong_result_count += validity["wrong_result_count"]
            successful_latencies.append(float(row["latency_ms"]))

        raw_path = args.output_dir / f"{label}.jsonl.gz"
        with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")

        successful = len(successful_latencies)
        summary = {
            "label": label,
            "requests": len(rows),
            "successful_requests": successful,
            "concurrency": concurrency,
            "beams": args.beams,
            "wall_seconds": round(wall_seconds, 3),
            "completion_qps": round(successful / wall_seconds, 4),
            "latency_ms": (
                latency_summary(successful_latencies) if successful_latencies else None
            ),
            "error_count": len(rows) - successful,
            "invalid_sid_count": invalid_sid_count,
            "duplicate_sid_count": duplicate_sid_count,
            "requests_with_wrong_result_count": wrong_result_count,
            "gpu_memory_used_mib": {
                "max_observed": max(monitor.samples) if monitor.samples else None,
                "samples": len(monitor.samples),
            },
            "raw_output": str(raw_path),
        }
        (args.output_dir / f"{label}.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2), flush=True)
        return summary

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        run_point(
            args.correctness_label,
            correctness_rows,
            concurrency=1,
            warmup_count=args.correctness_warmup,
        )
    ]
    for concurrency in args.concurrency:
        summaries.append(
            run_point(
                f"{args.performance_label}-c{concurrency}-n{len(performance_rows)}",
                performance_rows,
                concurrency=concurrency,
                warmup_count=args.performance_warmup,
            )
        )

    engine.shutdown()
    compact = {
        "pid": os.getpid(),
        "engine_init_seconds": round(engine_init_seconds, 3),
        "engine_kwargs": engine_kwargs,
        "source": {
            "model": str(args.model),
            "valid_sid_keys": str(args.valid_sid_keys),
            "correctness_inputs": str(args.correctness_inputs),
            "correctness_inputs_sha256": sha256(args.correctness_inputs),
            "performance_inputs": str(args.performance_inputs),
            "performance_inputs_sha256": sha256(args.performance_inputs),
        },
        "summaries": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(compact, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark one trie-head beam width across concurrency points."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--valid-sid-keys", type=Path, required=True)
    parser.add_argument("--correctness-inputs", type=Path, required=True)
    parser.add_argument("--performance-inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--correctness-label", default="trie-head-c1-correctness-n200")
    parser.add_argument("--performance-label", default="trie-head")
    parser.add_argument("--correctness-requests", type=int, default=200)
    parser.add_argument("--performance-requests", type=int, default=1000)
    parser.add_argument("--correctness-warmup", type=int, default=50)
    parser.add_argument("--performance-warmup", type=int, default=50)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 10])
    parser.add_argument(
        "--graph-concurrency", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10]
    )
    parser.add_argument("--beams", type=int, default=2000)
    parser.add_argument("--max-running-requests", type=int, default=20020)
    parser.add_argument("--context-length", type=int, default=10000)
    parser.add_argument("--mem-fraction-static", type=float, default=0.55)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--token-start", type=int, default=151669)
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument("--sid-length", type=int, default=3)
    args = parser.parse_args()
    require_single_visible_gpu(args.physical_gpu)
    run(args)


if __name__ == "__main__":
    main()
