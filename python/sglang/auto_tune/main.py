import argparse
import sys
import time
from typing import Optional


def create_parser():
    parser = argparse.ArgumentParser(
        description="SGLang Auto Tuner - Tune kernel configurations for a model."
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--tp", "--tp-size", dest="tp_size", type=int, default=1, help="TP size")
    parser.add_argument("--ep", "--ep-size", dest="ep_size", type=int, default=1, help="EP size")
    return parser


def auto_tune(args: argparse.Namespace, extra_argv: Optional[list] = None):
    parser = create_parser()
    args = parser.parse_args(extra_argv)

    print("=" * 60)
    print("SGLang Auto Tuner")
    print("=" * 60)
    print(f"\nModel: {args.model_path}")
    print(f"TP: {args.tp_size}, EP: {args.ep_size}")

    from sglang.auto_tune.utils import get_model_config
    from sglang.auto_tune.moe_tuner import run_moe_tuning

    print("\n[Step 1/3] Extracting model configuration...")
    model_config = get_model_config(args.model_path, args.tp_size, args.ep_size)
    print(f"  Architecture: {model_config['architecture']}")

    print("\n[Step 2/3] Tuning kernels...")
    moe_start = time.perf_counter()
    best_configs = run_moe_tuning(
        model_config, tp_size=args.tp_size, ep_size=args.ep_size,
        verbose=True,
    )
    moe_end = time.perf_counter()
    print(f"\nMoE tuning took {moe_end - moe_start:.2f}s")

    print("\n[Step 3/3] Summary")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Architecture: {model_config['architecture']}")
    print(f"MoE configs: {len(best_configs)} batch sizes tuned")
    print("=" * 60)


def main():
    parser = create_parser()
    args = parser.parse_args()
    auto_tune(args, None)


if __name__ == "__main__":
    main()
