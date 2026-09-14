"""Run a CPU-only SGLang Simulator benchmark with an InferCast config."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from sglang_simulator.dataset import GenericRequest, SimpleDataset
from sglang_simulator.simulation.benchmark import BenchmarkConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sim-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input-length", type=int, default=1024)
    parser.add_argument("--output-length", type=int, default=4)
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--max-total-tokens", type=int, default=65536)
    parser.add_argument("--max-prefill-tokens", type=int)
    parser.add_argument("--max-running-requests", type=int)
    parser.add_argument("--chunked-prefill-size", type=int, default=-1)
    parser.add_argument("--mem-fraction-static", type=float)
    parser.add_argument("--num-continuous-decode-steps", type=int, default=1)
    args = parser.parse_args()
    for name in (
        "input_length",
        "output_length",
        "num_requests",
        "max_total_tokens",
        "num_continuous_decode_steps",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be non-negative")
    if args.max_running_requests is not None and args.max_running_requests < 1:
        parser.error("--max-running-requests must be positive")
    if args.request_rate <= 0:
        parser.error("--request-rate must be positive")
    return args


def make_dataset(
    args: argparse.Namespace,
    *,
    count: int,
    token_offset: int = 0,
) -> SimpleDataset:
    interval = 0.0 if args.request_rate == float("inf") else 1.0 / args.request_rate
    return SimpleDataset(
        reqs=[
            GenericRequest(
                token_ids=[1000 + token_offset + index] * args.input_length,
                input_length=args.input_length,
                output_length=args.output_length,
                custom_params={"created_time": index * interval},
            )
            for index in range(count)
        ]
    )


def main() -> None:
    args = parse_args()
    sim_config = args.sim_config.resolve()
    model_path = args.model_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
    os.environ["SGLANG_SIMULATOR_OUTPUT_MODE"] = "OFFLINE"
    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(sim_config)
    os.environ["SGLANG_SIMULATOR_OUTPUT_DIR"] = str(output_dir)

    repository_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repository_root))

    from benchmark.simulator.bench_runner import SGLangBenchmarkRunner
    from sglang.srt.server_args import ServerArgs

    server_args = {
        "model_path": str(model_path),
        "load_format": "dummy",
        "device": "cpu",
        "skip_tokenizer_init": True,
        "max_total_tokens": args.max_total_tokens,
        "chunked_prefill_size": args.chunked_prefill_size,
        "page_size": 256,
        "disable_radix_cache": True,
        "num_continuous_decode_steps": args.num_continuous_decode_steps,
    }
    if args.max_prefill_tokens is not None:
        if args.max_prefill_tokens < 1:
            raise ValueError("--max-prefill-tokens must be positive")
        server_args["max_prefill_tokens"] = args.max_prefill_tokens
    if args.mem_fraction_static is not None:
        server_args["mem_fraction_static"] = args.mem_fraction_static
    if args.max_running_requests is not None:
        server_args["max_running_requests"] = args.max_running_requests

    runner = SGLangBenchmarkRunner(server_args=ServerArgs(**server_args))
    try:
        benchmark_config = BenchmarkConfig(
            request_rate=args.request_rate,
            ignore_request_timestamp=False,
        )
        if args.warmup_requests:
            runner.benchmark(
                benchmark_config,
                dataset=make_dataset(
                    args,
                    count=args.warmup_requests,
                    token_offset=args.num_requests,
                ),
            )
            runner.flush_cache()
        metrics = runner.benchmark(
            benchmark_config,
            dataset=make_dataset(args, count=args.num_requests),
        )
        if metrics is None:
            raise RuntimeError("SGLang Simulator did not produce metrics")
        result = {
            "model_path": str(model_path),
            "sim_config": str(sim_config),
            "workload": {
                "input_length": args.input_length,
                "output_length": args.output_length,
                "num_requests": args.num_requests,
                "warmup_requests": args.warmup_requests,
                "max_running_requests": args.max_running_requests,
                "request_rate": (
                    None if math.isinf(args.request_rate) else args.request_rate
                ),
            },
            "metrics": metrics,
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
