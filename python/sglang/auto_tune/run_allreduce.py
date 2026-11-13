"""All-reduce tuner. It does not start a server and does not load a model.

One process cannot all-reduce with itself, so torchrun starts one process per
GPU. ``--sizes`` is the message sizes in bytes. Each is timed as a bf16 tensor.

    torchrun --nproc_per_node=8 -m sglang.auto_tune.run_allreduce \
        --tp 8 --sizes 16384 32768 --graph 0 --output-dir /tmp/ar-tune

``--graph 0`` (default) times eager calls and writes
``allreduce_configs_eager.json`` for ``SGLANG_TUNED_EAGER_CONFIG``.
``--graph 1`` times calls captured into a CUDA graph and writes
``allreduce_configs_graph.json`` for ``SGLANG_TUNED_GRAPH_CONFIG``.
"""

from __future__ import annotations

import argparse
import os
import sys

from sglang.auto_tune.allreduce_tuner import (
    _SYNC_EVERY,
    QR_QUANT_REGIMES,
    print_allreduce_summary,
    run_allreduce_tuning,
)


_STACK_DIR = os.environ.get("ALLREDUCE_TUNE_STACK_DIR", "/tmp/ar-tune-stacks")
_STACK_PERIOD_S = 20


def _bool01(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected 0 or 1")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Time every all-reduce kernel per message size, eagerly "
        "or inside a CUDA graph, and write the fastest per size.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  torchrun --nproc_per_node=8 -m sglang.auto_tune.run_allreduce \\
      --selection-config /tmp/ar-tune/selection_config.json \\
      --tp 8 --graph 0 --median 1 -n 200 \\
      --output-dir /tmp/ar-tune \\
      --sizes 16384 65536 131072

  torchrun --nproc_per_node=8 -m sglang.auto_tune.run_allreduce \\
      --selection-config /tmp/ar-tune/selection_config.json \\
      --tp 8 --graph 1 --output-dir /tmp/ar-tune \\
      --sizes 16384 65536
""",
    )
    parser.add_argument("--tp", "--tp-size", dest="tp_size", type=int, required=True)
    parser.add_argument(
        "--graph",
        type=_bool01,
        default=False,
        help=(
            "0 (default): eager calls, for SGLANG_TUNED_EAGER_CONFIG. "
            "1: calls captured into a CUDA graph inside each communicator's "
            "capture context and replayed, for SGLANG_TUNED_GRAPH_CONFIG."
        ),
    )
    parser.add_argument(
        "--ingraph-calls",
        type=int,
        default=10,
        help=(
            "Graph mode: all-reduces captured back to back in one graph. "
            "The time is per call. Default 10."
        ),
    )
    parser.add_argument(
        "-n",
        "--num-iters",
        dest="num_iters",
        type=int,
        default=100,
        help=(
            "Timed calls per kernel and size in eager mode, timed graph "
            "replays in graph mode. Warm-up is extra. Default 100."
        ),
    )
    parser.add_argument(
        "--median",
        type=_bool01,
        default=False,
        help=(
            "1: winner from the median of per-call (eager) or per-replay "
            "(graph) times. 0 (default): mean of one timed loop."
        ),
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        required=True,
        help="Message sizes in bytes. Each is timed as a bf16 tensor.",
    )
    parser.add_argument(
        "--selection-config",
        default=None,
        help=(
            "JSON selection from generate_selection_config. The tuner opens "
            "exactly these backends. Without it, the selection is built from "
            "--qr-quant and the cap flags, and includes FlyDSL INT4."
        ),
    )
    parser.add_argument(
        "--qr-quant",
        default="INT4",
        choices=list(QR_QUANT_REGIMES),
        help=(
            "HIP QuickReduce codec when no --selection-config is given. "
            "One codec; default INT4. Ignored when --selection-config is set."
        ),
    )
    parser.add_argument(
        "--qr-cast-bf16-to-fp16",
        type=int,
        choices=(0, 1),
        default=1,
        help=(
            "ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16 for HIP QuickReduce. 1 (the "
            "server default) casts inside the timed kernel. Ignored when "
            "--selection-config is set."
        ),
    )
    parser.add_argument(
        "--ignore-acr-max-size",
        type=_bool01,
        default=False,
        help=(
            "1: time AITER past AITER_CUSTOM_AR_MAX_SIZE, up to half its "
            "registered pool. The server runs such a row the same way. "
            "Ignored when --selection-config is set."
        ),
    )
    parser.add_argument(
        "--ignore-scr-max-size",
        type=_bool01,
        default=False,
        help=(
            "1: time SGLang's custom all-reduce past its max size, up to "
            "its buffer. The server runs such a row the same way. "
            "Ignored when --selection-config is set."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="allreduce_tune_out",
        help="Rank 0 writes allreduce_configs_<eager|graph>.json here.",
    )
    parser.add_argument(
        "--sync-every",
        type=int,
        default=_SYNC_EVERY,
        help=(
            "Barrier across ranks every this many timed calls (eager) or "
            f"replays (graph). Default {_SYNC_EVERY}. 0 never barriers inside "
            "the loop, which is what hangs: the kernels spin-wait on their "
            "peers, so the ranks must not drift far apart."
        ),
    )
    parser.add_argument(
        "--timeout",
        dest="stack_period_s",
        type=float,
        default=_STACK_PERIOD_S,
        help=(
            "Seconds of no progress before every rank's stack is dumped, "
            f"repeating. Default {_STACK_PERIOD_S}. 0 turns the dumps off. "
            "Raise it when a size legitimately takes longer than this, so "
            "the dumps do not bury the results."
        ),
    )
    return parser


def _enable_stack_dumps(rank: int, period_s: float = _STACK_PERIOD_S) -> None:
    """Print every rank's stack into this terminal while the run is hung.

    A rank inside torch.cuda.synchronize is waiting on a kernel. A rank inside
    broadcast_object_list or barrier is waiting on a peer. ``period_s`` is how
    long a rank may make no progress before it dumps; 0 turns the dumps off.
    """
    import faulthandler
    import threading

    if period_s <= 0:
        return

    os.makedirs(_STACK_DIR, exist_ok=True)
    stream = open(os.path.join(_STACK_DIR, f"rank{rank}.txt"), "w")
    faulthandler.enable(file=stream, all_threads=True)
    faulthandler.dump_traceback_later(period_s, repeat=True, file=stream)
    if rank != 0:
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    def _report() -> None:
        import time

        time.sleep(period_s + 5)
        while True:
            parts = ["\n===== where each rank is ====="]
            for other in range(world_size):
                try:
                    with open(os.path.join(_STACK_DIR, f"rank{other}.txt")) as fh:
                        lines = fh.readlines()
                except OSError:
                    lines = []
                tail = "".join(lines[-8:]).rstrip() or "(no dump yet)"
                parts.append(f"--- rank {other} ---\n{tail}")
            print("\n".join(parts), file=sys.stderr, flush=True)
            time.sleep(period_s)

    threading.Thread(target=_report, name="stack-report", daemon=True).start()


def _init_distributed(stack_period_s: float = _STACK_PERIOD_S) -> None:
    import torch
    import torch.distributed as dist

    if "LOCAL_RANK" not in os.environ:
        sys.exit(
            "All-reduce tuning needs one process per GPU:\n"
            "torchrun --nproc_per_node=8 -m sglang.auto_tune.run_allreduce "
            "--tp 8 --sizes <bytes>..."
        )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    _enable_stack_dumps(
        int(os.environ.get("RANK", os.environ["LOCAL_RANK"])), stack_period_s
    )
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def main(argv: list[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    if args.num_iters < 1:
        sys.exit("-n must be at least 1")
    if args.ingraph_calls < 1:
        sys.exit("--ingraph-calls must be at least 1")
    if args.sync_every < 0:
        sys.exit("--sync-every must be 0 or more")
    if any(size <= 0 or size % 2 for size in args.sizes):
        sys.exit("--sizes must be positive multiples of 2 bytes (bf16 tensors)")
    _init_distributed(args.stack_period_s)

    import torch.distributed as dist

    rank = dist.get_rank()
    try:
        if args.tp_size != dist.get_world_size():
            if rank == 0:
                print(
                    f"--tp {args.tp_size} does not match the torchrun world "
                    f"size {dist.get_world_size()}. Use "
                    f"--nproc_per_node={args.tp_size}.",
                    file=sys.stderr,
                )
            sys.exit(1)

        results = run_allreduce_tuning(
            args.sizes,
            tp_size=args.tp_size,
            mode="graph" if args.graph else "eager",
            output_dir=args.output_dir if rank == 0 else None,
            verbose=(rank == 0),
            qr_quant=args.qr_quant,
            num_iters=args.num_iters,
            ingraph_calls=args.ingraph_calls,
            ignore_acr_max_size=args.ignore_acr_max_size,
            ignore_scr_max_size=args.ignore_scr_max_size,
            qr_cast_bf16_to_fp16=bool(args.qr_cast_bf16_to_fp16),
            use_median=args.median,
            sync_every=args.sync_every,
            selection_config=args.selection_config,
        )
        if rank == 0:
            print_allreduce_summary(results)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
