"""Collect per-layer per-expert routing stats from Qwen3-30B-A3B.

Generates synthetic stats (Zipf-distributed) when model is not available,
or collects real stats from model forward passes with ShareGPT prompts.

Usage:
  # Synthetic (no model needed):
  PYTHONPATH=python python3 scripts/heter_moe_collect_routing.py --synthetic

  # Real (requires model download first):
  PYTHONPATH=python python3 scripts/heter_moe_collect_routing.py \
      --model-path Qwen/Qwen3-30B-A3B
"""

import argparse
import json
import os

import numpy as np

NUM_EXPERTS = 128
NUM_LAYERS = 48
TOP_K = 8
OUT_DIR = "/data/heter-moe/routing_stats"


def generate_synthetic_routing(batch_size, phase, seed=42):
    """Generate Zipf-distributed routing stats mimicking real MoE imbalance."""
    rng = np.random.default_rng(seed + batch_size + (0 if phase == "prefill" else 1))

    result = {}
    for layer_idx in range(NUM_LAYERS):
        # Zipf popularity: some experts get many more tokens
        zipf_weights = 1.0 / np.arange(1, NUM_EXPERTS + 1) ** 1.1
        zipf_weights /= zipf_weights.sum()

        # Shuffle so hot experts aren't always the same across layers
        layer_perm = rng.permutation(NUM_EXPERTS)
        shuffled_weights = zipf_weights[layer_perm]

        total_tokens = batch_size * TOP_K
        if phase == "decode":
            total_tokens = batch_size * TOP_K

        counts = rng.multinomial(total_tokens, shuffled_weights)
        result[f"transformer_block_{layer_idx}"] = counts.tolist()

    return result


def save_stats(data, batch_size, phase, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fname = f"batch{batch_size}_{phase}.json"
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic Zipf-distributed stats",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="HuggingFace model path for real collection",
    )
    args = parser.parse_args()

    batch_sizes = [2**i for i in range(11)]  # 1, 2, 4, ..., 1024

    if args.synthetic or args.model_path is None:
        print("Generating synthetic routing stats (Zipf distribution)...")
        for bs in batch_sizes:
            for phase in ["prefill", "decode"]:
                data = generate_synthetic_routing(bs, phase)
                path = save_stats(data, bs, phase, OUT_DIR)
                total = sum(sum(v) for v in data.values())
                print(f"  batch={bs:>5} {phase:>7}: {path} (total_tokens={total})")
        print(f"\nAll stats saved to {OUT_DIR}/")
    else:
        print(f"Real routing collection from {args.model_path} not yet implemented.")
        print("Requires: model download, ShareGPT dataset, forward pass hooks.")
        print("Use --synthetic for now. Real collection will be added in step 5-6.")


if __name__ == "__main__":
    main()
