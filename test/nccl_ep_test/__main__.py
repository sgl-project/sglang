"""Two-GPU reproduction entrypoint; run with torchrun --module nccl_ep_test."""

import argparse
import json
from pathlib import Path

from .environment import binding_check, prepare_jit, report, snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "environment",
            "eager",
            "capture",
            "dynamic",
            "runner",
            "cleanup",
            "zero_length",
            "duplicates",
            "benchmark",
            "summarize",
        ),
    )
    parser.add_argument(
        "--implementation", choices=("native", "sglang"), default="sglang"
    )
    parser.add_argument("--buckets", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--reports", type=Path, nargs=2, metavar=("RANK0", "RANK1"))
    args = parser.parse_args()
    if args.mode == "environment":
        return report(
            "environment",
            lambda: dict(snapshot(), bindings=binding_check(), jit=prepare_jit()),
        )
    if args.mode == "summarize":
        from .benchmark import summarize_pair

        if args.reports is None:
            parser.error("summarize requires --reports RANK0 RANK1")
        print(
            json.dumps(
                summarize_pair([json.loads(p.read_text()) for p in args.reports]),
                indent=2,
            )
        )
        return 0
    if (
        min(args.buckets) < 1
        or max(args.buckets) > 1024
        or min(args.replays, args.layers, args.generations) < 1
    ):
        parser.error("Use buckets in [1, 1024] and positive replays/layers/generations")
    options = dict(
        buckets=args.buckets,
        replays=args.replays,
        layers=args.layers,
        generations=args.generations,
        identity=args.identity,
    )
    if args.mode in ("zero_length", "duplicates"):
        minimum = 8 if args.mode == "zero_length" else 16
        if args.implementation != "native" or max(args.buckets) < minimum:
            parser.error(
                f"{args.mode} requires native execution and capacity >= {minimum}"
            )
    if (
        args.mode in ("runner", "cleanup", "benchmark")
        and args.implementation != "sglang"
    ):
        parser.error(f"{args.mode} requires --implementation sglang")
    if args.mode == "benchmark":
        from .benchmark import exercise_benchmark

        if args.identity or args.layers != 2:
            parser.error("The matched benchmark uses two weighted expert layers")
        return report(
            "benchmark",
            lambda: exercise_benchmark(
                buckets=args.buckets,
                samples=args.samples,
                warmups=args.warmups,
                rounds=args.rounds,
            ),
        )
    suffix = "_identity" if args.identity else ""
    if args.implementation == "native":
        from .native import exercise

        return report(
            f"native_{args.mode}{suffix}", lambda: exercise(mode=args.mode, **options)
        )
    if args.mode == "eager":
        from .dispatcher import exercise_eager

        operation = lambda: exercise_eager(**options)
    elif args.mode in ("runner", "cleanup"):
        from .sglang_graph import exercise_cleanup, exercise_runner

        exercise = exercise_runner if args.mode == "runner" else exercise_cleanup
        operation = lambda: exercise(**options)
    else:
        from .sglang_graph import exercise_graph

        operation = lambda: exercise_graph(mode=args.mode, **options)
    return report(f"sglang_{args.mode}{suffix}", operation)


if __name__ == "__main__":
    raise SystemExit(main())
