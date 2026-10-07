"""Replay a timestamped variable-length trace through SGLang Simulator."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from infercast_trace import file_identity, load_trace_requests
from sglang_simulator.dataset import GenericRequest, SimpleDataset
from sglang_simulator.simulation.benchmark import BenchmarkConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sim-config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--warmup-trace",
        type=Path,
        help="Optional trace replayed before measurement to prime the radix cache.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-total-tokens", type=int, default=262144)
    parser.add_argument("--max-prefill-tokens", type=int, default=196608)
    parser.add_argument("--chunked-prefill-size", type=int, default=196608)
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--num-continuous-decode-steps", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument(
        "--bootstrap-visible-accelerator",
        action="store_true",
        help=(
            "Keep the allocated accelerator visible for framework import-time "
            "architecture checks; simulation still uses the CPU dummy engine."
        ),
    )
    parser.add_argument("--enable-radix-cache", action="store_true")
    parser.add_argument("--enable-hierarchical-cache", action="store_true")
    parser.add_argument("--hicache-ratio", type=float, default=2.0)
    return parser.parse_args()


def snapshot_outputs(output_dir: Path, phase: str) -> dict[str, str]:
    phase_dir = output_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name in ("metrics.json", "request.jsonl", "iteration.jsonl"):
        source = output_dir / name
        if source.exists():
            destination = phase_dir / name
            shutil.copy2(source, destination)
            artifacts[name] = str(destination)
    return artifacts


def load_trace(path: Path) -> SimpleDataset:
    requests = [
        GenericRequest(
            token_ids=request.token_ids,
            input_length=request.input_length,
            output_length=request.output_length,
            custom_params={
                "created_time": request.timestamp_ms / 1000.0,
                "trace": request.metadata,
            },
        )
        for request in load_trace_requests(path)
    ]
    return SimpleDataset(reqs=requests)


def main() -> None:
    args = parse_args()
    sim_config = args.sim_config.resolve()
    model_path = args.model_path.resolve()
    trace_path = args.trace.resolve()
    warmup_trace_path = (
        args.warmup_trace.resolve() if args.warmup_trace is not None else None
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.bootstrap_visible_accelerator:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
    os.environ["SGLANG_SIMULATOR_OUTPUT_MODE"] = "OFFLINE"
    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(sim_config)
    os.environ["SGLANG_SIMULATOR_OUTPUT_DIR"] = str(output_dir)

    repository_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repository_root))

    from benchmark.simulator.bench_runner import SGLangBenchmarkRunner
    from sglang.srt.server_args import ServerArgs

    dataset = load_trace(trace_path)
    warmup_dataset = (
        load_trace(warmup_trace_path) if warmup_trace_path is not None else None
    )
    if args.page_size < 1:
        raise ValueError("page size must be positive")
    server_args = {
        "model_path": str(model_path),
        "load_format": "dummy",
        "device": "cpu",
        "skip_tokenizer_init": True,
        "max_total_tokens": args.max_total_tokens,
        "max_prefill_tokens": args.max_prefill_tokens,
        "chunked_prefill_size": args.chunked_prefill_size,
        "mem_fraction_static": args.mem_fraction_static,
        "num_continuous_decode_steps": args.num_continuous_decode_steps,
        "page_size": args.page_size,
        "disable_radix_cache": not args.enable_radix_cache,
    }
    if args.enable_hierarchical_cache:
        server_args.update(
            enable_hierarchical_cache=True,
            hicache_ratio=args.hicache_ratio,
            hicache_storage_backend="file",
            hicache_storage_prefetch_policy="wait_complete",
        )
    runner = SGLangBenchmarkRunner(server_args=ServerArgs(**server_args))
    try:
        warmup = None
        if warmup_dataset is not None:
            warmup_metrics = runner.benchmark(
                BenchmarkConfig(
                    request_rate=float("inf"), ignore_request_timestamp=False
                ),
                dataset=warmup_dataset,
            )
            if warmup_metrics is None:
                raise RuntimeError("SGLang Simulator did not produce warmup metrics")
            warmup = {
                "trace": file_identity(warmup_trace_path),
                "request_count": len(warmup_dataset),
                "metrics": warmup_metrics,
                "artifacts": snapshot_outputs(output_dir, "warmup"),
            }

        metrics = runner.benchmark(
            BenchmarkConfig(request_rate=float("inf"), ignore_request_timestamp=False),
            dataset=dataset,
        )
        if metrics is None:
            raise RuntimeError("SGLang Simulator did not produce metrics")
        result = {
            "method": "agentx_open_loop_diagnostic",
            "trace": file_identity(trace_path),
            "warmup": warmup,
            "request_count": len(dataset),
            "page_size": args.page_size,
            "cache": {
                "radix": args.enable_radix_cache,
                "hierarchical": args.enable_hierarchical_cache,
                "hicache_ratio": (
                    args.hicache_ratio if args.enable_hierarchical_cache else None
                ),
            },
            "metrics": metrics,
            "artifacts": snapshot_outputs(output_dir, "target"),
        }
    finally:
        runner.shutdown()

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
