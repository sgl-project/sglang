"""
CLI entry point for auto-tuning SGLang kernels.

Usage:
    python3 -m sglang.auto_tune --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 --tp 8
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from python.sglang.tune.tune_fused_moe_triton import tune_fused_moe_triton
from python.sglang.tune.utils import (
    get_config_filename,
    save_configs,
)


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
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output file path for tuned configs; default matches the runtime lookup "
            "location under fused_moe_triton/configs/triton_<ver>/..."
        ),
    )
    return parser
    

def run_auto_tune(args: argparse.Namespace) -> None:
    best_configs = tune_fused_moe_triton(
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
    # Derive default output path to match runtime lookup
    if args.output:
        output_path = Path(args.output)
    else:
        # Recompute filename here so tune_fused_moe_triton stays pure
        from python.sglang.tune.utils import get_model_config
        import triton

        model_config = get_model_config(
            args.model,
            args.tp_size,
            args.ep_size,
            args.disable_shared_experts_fusion,
        )

        filename = get_config_filename(
            model_config["num_experts"],
            model_config["shard_intermediate_size"],
            model_config["hidden_size"],
            model_config["topk"],
            model_config["dtype"],
            args.dtype == "fp8_w8a8",
            args.dtype == "int8_w8a8",
            args.dtype == "int8_w8a16",
            args.per_channel_quant,
            model_config["block_shape"],
        )
        triton_version_dir = f"triton_{triton.__version__.replace('.', '_')}"
        output_path = (
            Path(__file__).resolve().parent.parent
            / "srt"
            / "layers"
            / "moe"
            / "fused_moe_triton"
            / "configs"
            / triton_version_dir
            / filename
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_configs(best_configs, str(output_path))
    print(f"Saved tuning results to {output_path}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_auto_tune(args)


if __name__ == "__main__":
    main()
