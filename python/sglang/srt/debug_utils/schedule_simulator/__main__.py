import argparse
import json
from typing import List

from sglang.srt.debug_utils.schedule_simulator import (
    AttentionBalancednessRecorder,
    BatchSizeBalancednessRecorder,
    FIFOScheduler,
    RandomRouter,
    RoundRobinRouter,
    Simulator,
    load_from_request_logger,
)


def main():
    parser = argparse.ArgumentParser(
        description="Schedule Simulator for analyzing request scheduling across GPUs"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to request_logger JSON file",
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

    args = parser.parse_args()

    requests = load_from_request_logger(args.input)
    print(f"Loaded {len(requests)} requests from {args.input}")

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
    )

    print(f"Running simulation with {args.num_gpus} GPUs, router={args.router}, scheduler={args.scheduler}")
    summary = sim.run(requests)

    print("\n=== Results ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

