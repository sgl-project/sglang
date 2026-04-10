"""Collect per-layer per-expert routing stats from Qwen3-30B-A3B.

Modes:
  --synthetic: Zipf-distributed fake stats (no GPU needed)
  --model-path: Real stats from model forward passes with ShareGPT

Usage:
  PYTHONPATH=python python3 scripts/heter_moe_collect_routing.py --synthetic

  CUDA_VISIBLE_DEVICES=4,5 python3 scripts/heter_moe_collect_routing.py \
      --model-path /data/heter-moe/models/qwen3-30b-a3b-bf16 \
      --sharegpt-path /data/heter-moe/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

NUM_EXPERTS = 128
NUM_LAYERS = 48
TOP_K = 8
OUT_DIR = "/data/heter-moe/routing_stats"
REAL_OUT_DIR = "/data/heter-moe/routing_stats/real"


def generate_synthetic_routing(batch_size, phase, seed=42):
    rng = np.random.default_rng(seed + batch_size + (0 if phase == "prefill" else 1))
    result = {}
    for layer_idx in range(NUM_LAYERS):
        zipf_weights = 1.0 / np.arange(1, NUM_EXPERTS + 1) ** 1.1
        zipf_weights /= zipf_weights.sum()
        layer_perm = rng.permutation(NUM_EXPERTS)
        shuffled_weights = zipf_weights[layer_perm]
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


def load_sharegpt(path, max_samples=2048):
    with open(path) as f:
        data = json.load(f)
    prompts = []
    for conv in data:
        turns = conv.get("conversations", [])
        for turn in turns:
            if turn.get("from") == "human" and turn.get("value", "").strip():
                prompts.append(turn["value"].strip())
                if len(prompts) >= max_samples:
                    return prompts
    return prompts


def collect_real_routing(model_path, sharegpt_path, batch_sizes):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_path} (BF16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"Loading ShareGPT from {sharegpt_path}...")
    prompts = load_sharegpt(sharegpt_path, max_samples=max(batch_sizes) * 2)
    print(f"  Loaded {len(prompts)} prompts")

    # Register hooks on all MoE gate layers to capture routing decisions
    routing_captures = {}

    def make_gate_hook(layer_idx):
        def hook_fn(module, input, output):
            with torch.no_grad():
                # HF Qwen3MoE gate returns (logits, weights, selected_experts)
                # or just a tensor — handle both
                if isinstance(output, tuple):
                    selected_experts = output[2]  # [batch*seq, top_k]
                else:
                    _, selected_experts = torch.topk(output, TOP_K, dim=-1)
                counts = torch.zeros(
                    NUM_EXPERTS, dtype=torch.long, device=selected_experts.device
                )
                counts.scatter_add_(
                    0,
                    selected_experts.reshape(-1),
                    torch.ones(
                        selected_experts.numel(),
                        dtype=torch.long,
                        device=selected_experts.device,
                    ),
                )
                routing_captures[layer_idx] = counts.cpu().numpy().tolist()

        return hook_fn

    hooks = []
    moe_layer_idx = 0
    for name, module in model.named_modules():
        if name.endswith(".mlp.gate"):
            hooks.append(module.register_forward_hook(make_gate_hook(moe_layer_idx)))
            moe_layer_idx += 1
    print(f"  Registered hooks on {moe_layer_idx} MoE gate layers")

    if moe_layer_idx == 0:
        for name, module in model.named_modules():
            if "gate" in name and isinstance(module, torch.nn.Linear):
                if module.out_features == NUM_EXPERTS:
                    hooks.append(
                        module.register_forward_hook(make_gate_hook(moe_layer_idx))
                    )
                    moe_layer_idx += 1
        print(f"  (fallback) Registered hooks on {moe_layer_idx} gate layers")

    results = {}
    for bs in batch_sizes:
        batch_prompts = prompts[:bs]
        if len(batch_prompts) < bs:
            batch_prompts = batch_prompts * (bs // len(batch_prompts) + 1)
            batch_prompts = batch_prompts[:bs]

        # Prefill: tokenize and run forward
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        routing_captures.clear()
        print(
            f"\n  batch={bs}: running prefill ({inputs['input_ids'].shape})...",
            end="",
            flush=True,
        )
        with torch.no_grad():
            model(**inputs)
        print(f" captured {len(routing_captures)} layers")

        prefill_data = {}
        for layer_idx in range(moe_layer_idx):
            key = f"transformer_block_{layer_idx}"
            if layer_idx in routing_captures:
                prefill_data[key] = routing_captures[layer_idx]
            else:
                prefill_data[key] = [0] * NUM_EXPERTS
        path = save_stats(prefill_data, bs, "prefill", REAL_OUT_DIR)
        print(f"    saved: {path}")
        results[(bs, "prefill")] = prefill_data

        # Decode: generate 1 new token per sequence
        routing_captures.clear()
        print(f"  batch={bs}: running decode...", end="", flush=True)
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=False,
            )
        print(f" captured {len(routing_captures)} layers")

        decode_data = {}
        for layer_idx in range(moe_layer_idx):
            key = f"transformer_block_{layer_idx}"
            if layer_idx in routing_captures:
                decode_data[key] = routing_captures[layer_idx]
            else:
                decode_data[key] = [0] * NUM_EXPERTS
        path = save_stats(decode_data, bs, "decode", REAL_OUT_DIR)
        print(f"    saved: {path}")
        results[(bs, "decode")] = decode_data

    for h in hooks:
        h.remove()

    return results


def print_imbalance_summary(results):
    print("\n" + "=" * 70)
    print(
        f"{'Batch':>6} {'Phase':>8} {'MaxLoad':>8} {'MinLoad':>8} {'Ratio':>8} {'Gini':>8}"
    )
    print("-" * 70)
    for (bs, phase), data in sorted(results.items()):
        all_counts = []
        for key, counts in data.items():
            all_counts.extend(counts)
        arr = np.array(all_counts).reshape(-1, NUM_EXPERTS)
        avg_per_layer = arr.mean(axis=0)
        max_load = avg_per_layer.max()
        min_load = avg_per_layer.min()
        ratio = max_load / max(min_load, 1)
        # Gini coefficient
        sorted_loads = np.sort(avg_per_layer)
        n = len(sorted_loads)
        cum = np.cumsum(sorted_loads)
        gini = (2.0 * np.sum((np.arange(1, n + 1) * sorted_loads)) / (n * cum[-1])) - (
            n + 1
        ) / n
        print(
            f"{bs:>6} {phase:>8} {max_load:>8.1f} {min_load:>8.1f} {ratio:>8.1f} {gini:>8.3f}"
        )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument(
        "--sharegpt-path",
        type=str,
        default="/data/heter-moe/datasets/ShareGPT_V3_unfiltered_cleaned_split.json",
    )
    args = parser.parse_args()

    batch_sizes = [2**i for i in range(11)]

    if args.synthetic:
        print("Generating synthetic routing stats (Zipf distribution)...")
        for bs in batch_sizes:
            for phase in ["prefill", "decode"]:
                data = generate_synthetic_routing(bs, phase)
                path = save_stats(data, bs, phase, OUT_DIR)
                total = sum(sum(v) for v in data.values())
                print(f"  batch={bs:>5} {phase:>7}: {path} (total_tokens={total})")
        print(f"\nSynthetic stats saved to {OUT_DIR}/")

    elif args.model_path:
        if not os.path.exists(args.sharegpt_path):
            print(f"ShareGPT not found at {args.sharegpt_path}")
            print("Downloading ShareGPT_V3...")
            os.makedirs(os.path.dirname(args.sharegpt_path), exist_ok=True)
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
                filename="ShareGPT_V3_unfiltered_cleaned_split.json",
                repo_type="dataset",
                local_dir=os.path.dirname(args.sharegpt_path),
            )
            print(f"Downloaded to {args.sharegpt_path}")

        results = collect_real_routing(args.model_path, args.sharegpt_path, batch_sizes)
        print_imbalance_summary(results)
        print(f"\nReal stats saved to {REAL_OUT_DIR}/")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
