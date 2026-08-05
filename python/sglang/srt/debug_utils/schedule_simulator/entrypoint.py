import argparse
import json
import random
from typing import List

from sglang.srt.debug_utils.schedule_simulator.data_source.data_loader import (
    load_from_request_logger,
)
from sglang.srt.debug_utils.schedule_simulator.data_source.data_synthesis import (
    generate_gsp_requests,
    generate_random_requests,
)
from sglang.srt.debug_utils.schedule_simulator.metrics import (
    AttentionComputeBalancednessRecorder,
    AvgBatchSizeRecorder,
    BatchSizeBalancednessRecorder,
)
from sglang.srt.debug_utils.schedule_simulator.request import SimRequest
from sglang.srt.debug_utils.schedule_simulator.routers import (
    RandomRouter,
    RoundRobinRouter,
    StickyRouter,
)
from sglang.srt.debug_utils.schedule_simulator.schedulers import FIFOScheduler
from sglang.srt.debug_utils.schedule_simulator.simulator import (
    SimulationResult,
    Simulator,
)


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
    data_group.add_argument(
        "--synth-gsp",
        action="store_true",
        help="Use generated-shared-prefix (GSP) data generation",
    )

    # Shared synthetic arguments
    parser.add_argument("--synth-seed", type=int, default=None)

    # Random dataset arguments (aligned with bench_serving.py --random-* options)
    parser.add_argument("--synth-random-num-requests", type=int, default=1000)
    parser.add_argument("--synth-random-input-len", type=int, default=1024)
    parser.add_argument("--synth-random-output-len", type=int, default=256)
    parser.add_argument("--synth-random-range-ratio", type=float, default=0.0)

    # GSP dataset arguments (aligned with bench_serving.py --gsp-* options)
    parser.add_argument("--synth-gsp-num-groups", type=int, default=64)
    parser.add_argument("--synth-gsp-prompts-per-group", type=int, default=16)
    parser.add_argument("--synth-gsp-system-prompt-len", type=int, default=2048)
    parser.add_argument("--synth-gsp-question-len", type=int, default=128)
    parser.add_argument("--synth-gsp-output-len", type=int, default=256)
    parser.add_argument("--synth-gsp-range-ratio", type=float, default=1.0)

    parser.add_argument("--num-gpus-per-engine", type=int, default=8)
    parser.add_argument("--num-engines", type=int, default=1)
    parser.add_argument(
        "--router",
        type=str,
        choices=["random", "round_robin", "sticky"],
        default="round_robin",
    )
    parser.add_argument("--scheduler", type=str, choices=["fifo"], default="fifo")
    parser.add_argument("--max-total-tokens", type=int, default=100000)
    parser.add_argument(
        "--stop-criteria",
        type=str,
        choices=["all_done", "exist_no_pending"],
        default="all_done",
        help="all_done: run until all requests complete; exist_no_pending: stop when any GPU has no pending requests",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--log-level", type=int, choices=[0, 1, 2], default=0)

    return parser


def _load_requests(args: argparse.Namespace) -> List[SimRequest]:
    if args.input:
        requests = load_from_request_logger(args.input)
        print(f"Loaded {len(requests)} requests from {args.input}")
    elif args.synth_gsp:
        requests = generate_gsp_requests(
            num_groups=args.synth_gsp_num_groups,
            prompts_per_group=args.synth_gsp_prompts_per_group,
            system_prompt_len=args.synth_gsp_system_prompt_len,
            question_len=args.synth_gsp_question_len,
            output_len=args.synth_gsp_output_len,
            range_ratio=args.synth_gsp_range_ratio,
            seed=args.synth_seed,
        )
    else:
        requests = generate_random_requests(
            num_requests=args.synth_random_num_requests,
            input_len=args.synth_random_input_len,
            output_len=args.synth_random_output_len,
            range_ratio=args.synth_random_range_ratio,
            seed=args.synth_seed,
        )
    return requests


def _create_router(name: str, total_gpus: int):
    if name == "random":
        return RandomRouter(total_gpus)
    if name == "round_robin":
        return RoundRobinRouter(total_gpus)
    if name == "sticky":
        return StickyRouter(total_gpus)
    raise ValueError(f"Unknown router: {name}")


def _create_scheduler(name: str):
    if name == "fifo":
        return FIFOScheduler()
    raise ValueError(f"Unknown scheduler: {name}")


def main(args: argparse.Namespace) -> SimulationResult:
    if args.synth_seed is not None:
        random.seed(args.synth_seed)
    requests = _load_requests(args)
    total_gpus = args.num_gpus_per_engine * args.num_engines
    router = _create_router(args.router, total_gpus)
    scheduler = _create_scheduler(args.scheduler)

    sim = Simulator(
        num_gpus_per_engine=args.num_gpus_per_engine,
        router=router,
        scheduler=scheduler,
        recorders=[
            BatchSizeBalancednessRecorder(),
            AttentionComputeBalancednessRecorder(),
            AvgBatchSizeRecorder(),
        ],
        log_level=args.log_level,
        max_total_tokens=args.max_total_tokens,
        stop_criteria=args.stop_criteria,
        max_steps=args.max_steps,
    )

    print(
        f"Running simulation with {args.num_gpus_per_engine} GPUs/engine x {args.num_engines} engines, router={args.router}, scheduler={args.scheduler}"
    )
    result = sim.run(requests)

    print("\n=== Summary ===")
    for key, value in result.summary.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.summary, f, indent=2)
        print(f"\nSummary saved to {args.output}")

    return result
