import argparse
import json
from typing import List

import polars as pl

from sglang.srt.debug_utils.schedule_simulator.request import SimRequest


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Schedule Simulator for analyzing request scheduling across GPUs"
    )

    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--input", type=str, help="Path to request_logger JSON file"
    )
    data_group.add_argument(
        "--synthetic", action="store_true", help="Use synthetic data generation"
    )

    parser.add_argument("--synth-num-requests", type=int, default=1000)
    parser.add_argument("--synth-input-len", type=int, default=1024)
    parser.add_argument("--synth-output-len", type=int, default=256)
    parser.add_argument("--synth-range-ratio", type=float, default=1.0)
    parser.add_argument("--synth-seed", type=int, default=None)

    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument(
        "--router", type=str, choices=["random", "round_robin"], default="round_robin"
    )
    parser.add_argument("--scheduler", type=str, choices=["fifo"], default="fifo")
    parser.add_argument("--max-total-tokens", type=int, default=100000)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--log-level", type=int, choices=[0, 1, 2], default=0)

    return parser


def _load_requests(args: argparse.Namespace) -> List[SimRequest]:
    from sglang.srt.debug_utils.schedule_simulator import (
        generate_random_requests,
        load_from_request_logger,
    )

    if args.input:
        requests = load_from_request_logger(args.input)
        print(f"Loaded {len(requests)} requests from {args.input}")
    else:
        requests = generate_random_requests(
            num_requests=args.synth_num_requests,
            input_len=args.synth_input_len,
            output_len=args.synth_output_len,
            range_ratio=args.synth_range_ratio,
            seed=args.synth_seed,
        )
        print(
            f"Generated {len(requests)} synthetic requests "
            f"(synth_input_len={args.synth_input_len}, "
            f"synth_output_len={args.synth_output_len}, "
            f"synth_range_ratio={args.synth_range_ratio})"
        )
    return requests


def _create_router(name: str):
    from sglang.srt.debug_utils.schedule_simulator import RandomRouter, RoundRobinRouter

    return RandomRouter() if name == "random" else RoundRobinRouter()


def _create_scheduler(name: str, max_total_tokens: int):
    from sglang.srt.debug_utils.schedule_simulator import FIFOScheduler

    if name == "fifo":
        return FIFOScheduler(max_total_tokens=max_total_tokens)
    raise ValueError(f"Unknown scheduler: {name}")


def main(args: argparse.Namespace) -> pl.DataFrame:
    from sglang.srt.debug_utils.schedule_simulator import (
        AttentionBalancednessRecorder,
        BatchSizeBalancednessRecorder,
        Simulator,
    )

    requests = _load_requests(args)
    router = _create_router(args.router)
    scheduler = _create_scheduler(args.scheduler, args.max_total_tokens)

    sim = Simulator(
        num_gpus=args.num_gpus,
        router=router,
        scheduler=scheduler,
        recorders=[BatchSizeBalancednessRecorder(), AttentionBalancednessRecorder()],
        log_level=args.log_level,
    )

    print(
        f"Running simulation with {args.num_gpus} GPUs, router={args.router}, scheduler={args.scheduler}"
    )
    result = sim.run(requests)

    df = pl.DataFrame(
        [
            {
                "step": r.step,
                "gpu_id": r.gpu_id,
                "running_count": r.running_count,
                "pending_count": r.pending_count,
                "total_seq_len": r.total_seq_len,
                "running_req_ids": r.running_req_ids,
                "pending_req_ids": r.pending_req_ids,
            }
            for r in result.step_records
        ]
    )

    print("\n=== Summary ===")
    for key, value in result.summary.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.summary, f, indent=2)
        print(f"\nSummary saved to {args.output}")

    return df
