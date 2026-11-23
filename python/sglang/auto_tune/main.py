import argparse
import sys
import time
from typing import Optional

from sglang.auto_tune.utils import get_model_config
from sglang.auto_tune.moe_tuner import run_moe_tuning
from sglang.auto_tune.allreduce_tuner import run_allreduce_tuning, print_allreduce_summary


def create_parser():
    parser = argparse.ArgumentParser(
        description="SGLang Auto Tuner - Tune kernel configurations for a model."
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--tp", "--tp-size", dest="tp_size", type=int, default=1, help="TP size")
    parser.add_argument("--ep", "--ep-size", dest="ep_size", type=int, default=1, help="EP size")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save configs")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None, help="Specific batch sizes")
    parser.add_argument("--skip-moe", action="store_true", help="Skip MoE tuning")
    parser.add_argument("--allreduce", action="store_true", help="Enable allreduce algorithm tuning")
    return parser


def auto_tune(args: argparse.Namespace, extra_argv: Optional[list] = None):
    parser = create_parser()
    args = parser.parse_args(extra_argv)

    print("=" * 60)
    print("SGLang Auto Tuner")
    print("=" * 60)
    print(f"\nModel: {args.model_path}")
    print(f"TP: {args.tp_size}, EP: {args.ep_size}")

    print("\n[Step 1/3] Extracting model configuration...")
    try:
        model_config = get_model_config(args.model_path, args.tp_size, args.ep_size)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"  Architecture: {model_config['architecture']}")
    print(f"  MoE: {model_config.get('is_moe', False)}")

    if not model_config.get("is_moe", False):
        print("\n  This model does not have MoE layers (dense model).")
        print("  Auto-tuner currently supports MoE kernel tuning and allreduce tuning.")
        print("  Skipping MoE kernel tuning.")
        best_configs = {}
    else:
        print("\n[Step 2/3] Tuning kernels...")
        moe_start = time.perf_counter()
        best_configs = run_moe_tuning(
            model_config, tp_size=args.tp_size, ep_size=args.ep_size,
            batch_sizes=args.batch_sizes, output_dir=args.output_dir, verbose=True,
        )
        moe_end = time.perf_counter()
        print(f"\nMoE tuning took {moe_end - moe_start:.2f}s")

    # Allreduce tuning
    if args.allreduce:
        print("\n--- Allreduce Algorithm Tuning ---")
        allreduce_start = time.perf_counter()
        allreduce_configs = run_allreduce_tuning(
            model_config,
            tp_size=args.tp_size,
            ep_size=args.ep_size,
            output_dir=args.output_dir,
            verbose=True,
        )
        allreduce_end = time.perf_counter()
        print(f"\nAllreduce tuning took {allreduce_end - allreduce_start:.2f}s")
    else:
        allreduce_configs = {}

    print("\n[Step 3/3] Summary")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Architecture: {model_config['architecture']}")
    print(f"Experts: {model_config['num_experts']}, TopK: {model_config['topk']}")
    print(f"MoE configs: {len(best_configs)} batch sizes tuned")
    print(f"Allreduce configs: {len(allreduce_configs)} sizes profiled")
    if args.output_dir:
        print(f"Configs saved to: {args.output_dir}")
    print("=" * 60)


def main():
    parser = create_parser()
    args = parser.parse_args()
    auto_tune(args, None)


if __name__ == "__main__":
    main()