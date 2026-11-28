"""All-reduce tuner. This does not start a server and does not tune MoE.

One process cannot all-reduce with itself. torchrun starts one process per
GPU and sets RANK, LOCAL_RANK, and WORLD_SIZE. QuickReduce is timed at INT4
unless --qr-quant selects other codecs.

    torchrun --nproc_per_node=8 -m sglang.auto_tune.run_allreduce \
        --model-path amd/Qwen3.8-2.4T-A95B-Quark-MXFP4 \
        --tp 8 --qr-quant INT4 --use-aiter-allreduce \
        --qr-cast-bf16-to-fp16 1 --output-dir /tmp/ar-tune

--model-path is a Hugging Face id or a local directory that contains
config.json. The run reads that config and allocates random tensors. It does
not load the weights. Rank 0 writes allreduce_configs.json.

The server uses that file when SGLANG_ALLREDUCE_TUNE_CONFIG points at it.
"""

from __future__ import annotations

import argparse
import os
import sys

from sglang.auto_tune.allreduce_tuner import (
    QR_QUANT_REGIMES,
    print_allreduce_summary,
    run_allreduce_tuning,
)
from sglang.auto_tune.utils import get_model_config


def _optional_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune all-reduce backends, including QuickReduce INT4."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Hugging Face id or a local directory with config.json.",
    )
    parser.add_argument("--tp", "--tp-size", dest="tp_size", type=int, required=True)
    parser.add_argument("--ep", "--ep-size", dest="ep_size", type=int, default=1)
    parser.add_argument(
        "--qr-quant",
        nargs="+",
        default=["INT4"],
        choices=list(QR_QUANT_REGIMES),
        help="QuickReduce codecs to time. Default is INT4.",
    )
    parser.add_argument(
        "--use-aiter-allreduce",
        action="store_true",
        help=(
            "Accepted for older commands. SGLang custom all-reduce, AITER, "
            "and custom all-reduce v2 are all timed; this flag does not "
            "turn the others off."
        ),
    )
    parser.add_argument(
        "--use-car-v2",
        action="store_true",
        help=(
            "Accepted for older commands. Custom all-reduce v2 is always "
            "timed next to SGLang's custom all-reduce and AITER."
        ),
    )
    parser.add_argument(
        "--ignore-acr-max-size",
        action="store_true",
        help=(
            "Ignore AITER_CUSTOM_AR_MAX_SIZE and time AITER up to its "
            "registered pool. SGLang custom all-reduce and v2 keep their "
            "own cutoffs. The server keeps its own cutoff."
        ),
    )
    parser.add_argument(
        "--qr-cast-bf16-to-fp16",
        type=int,
        choices=(0, 1),
        default=1,
        help=(
            "Value of ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16, set before "
            "QuickReduce is built. 1, the server default, casts bf16 to fp16 "
            "inside the timed QuickReduce kernel. 0 times the bf16 kernels."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="allreduce_tune_out",
        help="Rank 0 writes allreduce_configs.json here.",
    )
    parser.add_argument(
        "-n",
        "--num-iters",
        dest="num_iters",
        type=int,
        default=100,
        help=(
            "Timed calls per kernel and size. Warmup calls are extra and "
            "are not counted. Default is 100."
        ),
    )
    parser.add_argument(
        "--median",
        nargs="?",
        const=True,
        default=False,
        type=_optional_bool,
        help=(
            "Pick the winner from the median of the timed calls. "
            "--median and --median true turn it on. Without it, the "
            "winner is the mean of one timed loop."
        ),
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Message sizes in bytes, replacing the sizes derived from the "
            "model. A bf16 tensor of shape [num_tokens, hidden] is "
            "num_tokens * hidden * 2 bytes."
        ),
    )
    parser.add_argument(
        "--qr-ignore-min-size",
        action="store_true",
        help=(
            "Time QuickReduce below the codec's minimum message size. "
            "The server keeps the minimum."
        ),
    )
    parser.add_argument(
        "--ignore-scr-size-bounds",
        dest="ignore_scr_size_bounds",
        action="store_true",
        help=(
            "Do not apply the published QuickReduce or custom all-reduce size "
            "caps. A size is skipped only when the kernel rejects the tensor, "
            "and the rejection is printed. The server keeps its own bounds."
        ),
    )
    return parser


_STACK_DIR = os.environ.get("ALLREDUCE_TUNE_STACK_DIR", "/tmp/ar-tune-stacks")
_STACK_PERIOD_S = 20


def _enable_stack_dumps(rank: int) -> None:
    """Print every rank's stack into this terminal while the run is hung.

    Each rank records its own stack. Rank 0 reprints the latest dump from
    every rank on stderr, which torchrun shows in the same shell. A rank
    inside torch.cuda.synchronize is waiting on a kernel. A rank inside
    broadcast_object_list or barrier is waiting on a peer.
    """
    import faulthandler
    import threading

    os.makedirs(_STACK_DIR, exist_ok=True)
    path = os.path.join(_STACK_DIR, f"rank{rank}.txt")
    stream = open(path, "w")
    faulthandler.enable(file=stream, all_threads=True)
    faulthandler.dump_traceback_later(_STACK_PERIOD_S, repeat=True, file=stream)
    stream.flush()

    if rank != 0:
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    def _report() -> None:
        import time

        time.sleep(_STACK_PERIOD_S + 5)
        while True:
            parts = ["\n===== where each rank is ====="]
            for other in range(world_size):
                other_path = os.path.join(_STACK_DIR, f"rank{other}.txt")
                try:
                    with open(other_path) as handle:
                        lines = handle.readlines()
                except OSError:
                    lines = []
                tail = "".join(lines[-8:]).rstrip() or "(no dump yet)"
                parts.append(f"--- rank {other} ---\n{tail}")
            print("\n".join(parts), file=sys.stderr, flush=True)
            time.sleep(_STACK_PERIOD_S)

    threading.Thread(target=_report, name="stack-report", daemon=True).start()


def _init_distributed() -> None:
    import torch
    import torch.distributed as dist

    if "LOCAL_RANK" not in os.environ:
        sys.exit(
            "All-reduce tuning needs one process per GPU.\n"
            "torchrun --nproc_per_node=8 -m sglang.auto_tune.run_allreduce "
            "--model-path <model> --tp 8 --qr-quant INT4"
        )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    _enable_stack_dumps(int(os.environ.get("RANK", os.environ["LOCAL_RANK"])))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def main(argv: list[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    if args.num_iters < 1:
        sys.exit("-n must be at least 1")
    _init_distributed()

    import torch.distributed as dist

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    try:
        if args.tp_size != world_size:
            if rank == 0:
                print(
                    f"--tp {args.tp_size} does not match the torchrun world "
                    f"size {world_size}. Use --nproc_per_node={args.tp_size}.",
                    file=sys.stderr,
                )
            sys.exit(1)

        model_config = get_model_config(args.model_path, args.tp_size, args.ep_size)
        results = run_allreduce_tuning(
            model_config,
            tp_size=args.tp_size,
            ep_size=args.ep_size,
            output_dir=args.output_dir if rank == 0 else None,
            verbose=(rank == 0),
            qr_quants=args.qr_quant,
            num_iters=args.num_iters,
            sizes=args.sizes,
            qr_ignore_min_size=args.qr_ignore_min_size or args.ignore_scr_size_bounds,
            qr_ignore_max_size=args.ignore_scr_size_bounds,
            use_aiter_allreduce=args.use_aiter_allreduce,
            qr_cast_bf16_to_fp16=bool(args.qr_cast_bf16_to_fp16),
            use_car_v2=args.use_car_v2,
            ignore_acr_max_size=args.ignore_acr_max_size,
            use_median=bool(args.median),
        )
        if rank == 0:
            print_allreduce_summary(results)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
