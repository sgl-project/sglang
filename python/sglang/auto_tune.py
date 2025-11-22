"""
CLI entry point for auto-tuning SGLang kernels.

Usage:
    python3 -m sglang.auto_tune --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 --tp 8
"""

import argparse
from typing import Optional, Sequence

from python.sglang.tune.tune_fused_moe_triton import tune_fused_moe_triton


def build_arg_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser for auto tuning."""

    parser = argparse.ArgumentParser(
        description="Auto-tune SGLang backend kernels for a given model."
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507"
    )
    parser.add_argument("--tp-size", "--tp", type=int, default=1)
    parser.add_argument("--ep-size", "--ep", type=int, default=1)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "fp8_w8a8", "int8_w8a16", "int8_w8a8"],
        default="auto",
    )
    parser.add_argument(
        "--per-channel-quant",
        action="store_true",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--disable-shared-experts-fusion", action="store_true")
    parser.add_argument(
        "--num-iters",
        type=int,
        default=10,
        help="Number of iterations per config during tuning",
    )
    return parser
    

def run_auto_tune(args: argparse.Namespace) -> None:
    config_path, best_configs = tune_fused_moe_triton(
        model=args.model,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        dtype=args.dtype,
        per_channel_quant=args.per_channel_quant,
        batch_size=args.batch_size,
        seed=args.seed,
        disable_shared_experts_fusion=args.disable_shared_experts_fusion,
        num_iters=args.num_iters,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_auto_tune(args)


if __name__ == "__main__":
    main()
