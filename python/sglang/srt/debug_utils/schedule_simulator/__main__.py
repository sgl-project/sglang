import argparse
import json
from typing import Optional

import polars as pl

from sglang.srt.debug_utils.schedule_simulator import (
    AttentionBalancednessRecorder,
    BatchSizeBalancednessRecorder,
    FIFOScheduler,
    RandomRouter,
    RoundRobinRouter,
    Simulator,
    generate_random_requests,
    load_from_request_logger,
)


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Schedule Simulator for analyzing request scheduling across GPUs"
    )

    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--input",
        type=str,
        help="Path to request_logger JSON file",
    )
    data_group.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data generation",
    )

    parser.add_argument(
        "--num-requests",
        type=int,
        default=1000,
        help="Number of synthetic requests to generate (default: 1000)",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=1024,
        help="Target input length for synthetic data (default: 1024)",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=256,
        help="Target output length for synthetic data (default: 256)",
    )
    parser.add_argument(
        "--range-ratio",
        type=float,
        default=1.0,
        help="Range ratio for synthetic data, e.g., 0.5 means [len*0.5, len] (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for synthetic data (default: None)",
    )

    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs to simulate (default: 8)",
    )
    parser.add_argument(
        "--router",
        type=str,
        choices=["random", "round_robin"],
        default="round_robin",
        help="Router policy (default: round_robin)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["fifo"],
        default="fifo",
        help="Scheduler policy (default: fifo)",
    )
    parser.add_argument(
        "--max-running",
        type=int,
        default=256,
        help="Max running requests per GPU for FIFO scheduler (default: 256)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (default: stdout)",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Log level: 0=none, 1=counts per GPU, 2=request IDs (default: 0)",
    )

    return parser


def main(args: argparse.Namespace) -> pl.DataFrame:
    if args.input:
        requests = load_from_request_logger(args.input)
        print(f"Loaded {len(requests)} requests from {args.input}")
    else:
        requests = generate_random_requests(
            num_requests=args.num_requests,
            input_len=args.input_len,
            output_len=args.output_len,
            range_ratio=args.range_ratio,
            seed=args.seed,
        )
        print(f"Generated {len(requests)} synthetic requests (input_len={args.input_len}, output_len={args.output_len}, range_ratio={args.range_ratio})")

    if args.router == "random":
        router = RandomRouter()
    else:
        router = RoundRobinRouter()

    if args.scheduler == "fifo":
        scheduler = FIFOScheduler(max_running_requests=args.max_running)
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    recorders = [
        BatchSizeBalancednessRecorder(),
        AttentionBalancednessRecorder(),
    ]

    sim = Simulator(
        num_gpus=args.num_gpus,
        router=router,
        scheduler=scheduler,
        recorders=recorders,
        log_level=args.log_level,
    )

    print(f"Running simulation with {args.num_gpus} GPUs, router={args.router}, scheduler={args.scheduler}")
    result = sim.run(requests)

    df = pl.DataFrame([
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
    ])

    print("\n=== Summary ===")
    for key, value in result.summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.summary, f, indent=2)
        print(f"\nSummary saved to {args.output}")

    return df


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    df = main(args)
    print(f"\nDataFrame shape: {df.shape}")
    print(df.head(20))
