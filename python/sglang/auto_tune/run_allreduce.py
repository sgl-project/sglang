"""All-reduce tuner. This does not start a server and does not tune MoE.

One process cannot all-reduce with itself. torchrun starts one process per
GPU and sets RANK, LOCAL_RANK, and WORLD_SIZE. QuickReduce is timed at INT4
unless --qr-quant selects other codecs.

    torchrun --nproc_per_node=8 -m sglang.auto_tune.run_allreduce \
        --model-path amd/Qwen3.8-2.4T-A95B-Quark-MXFP4 \
        --tp 8 --qr-quant INT4 --output-dir /tmp/ar-tune

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
        "--output-dir",
        default="allreduce_tune_out",
        help="Rank 0 writes allreduce_configs.json here.",
    )
    parser.add_argument("--num-iters", type=int, default=100)
    return parser


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
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def main(argv: list[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
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
        )
        if rank == 0:
            print_allreduce_summary(results)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
