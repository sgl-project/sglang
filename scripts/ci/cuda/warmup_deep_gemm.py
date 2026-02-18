"""
Warmup DeepGEMM JIT compilation for specified models.

Takes an explicit list of model:tp pairs. Skips models not in HF cache
and deduplicates models sharing the same architecture.

Usage:
    python3 scripts/ci/cuda/warmup_deep_gemm.py \
        deepseek-ai/DeepSeek-V3-0324:8 \
        deepseek-ai/DeepSeek-V3.2-Exp:8
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def model_in_hf_cache(model_name):
    """Check if a model exists in the HF cache."""
    cache_dir = os.environ.get(
        "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    )
    hub_dir = os.path.join(cache_dir, "hub")
    safe_name = "models--" + model_name.replace("/", "--")
    snapshots_dir = os.path.join(hub_dir, safe_name, "snapshots")
    return os.path.isdir(snapshots_dir) and bool(os.listdir(snapshots_dir))


def get_config_json(model_name):
    """Load config.json for a cached model."""
    cache_dir = os.environ.get(
        "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    )
    hub_dir = os.path.join(cache_dir, "hub")
    safe_name = "models--" + model_name.replace("/", "--")
    snapshots_dir = os.path.join(hub_dir, safe_name, "snapshots")

    if not os.path.isdir(snapshots_dir):
        return None

    snapshots = sorted(
        Path(snapshots_dir).iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for snapshot in snapshots:
        config_path = snapshot / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
    return None


def get_architecture_key(config, tp):
    """Key for dedup: models with same key share DeepGEMM kernels."""
    if config is None:
        return None
    num_experts = (
        config.get("num_local_experts")
        or config.get("n_routed_experts")
        or config.get("num_experts")
        or 0
    )
    hidden_size = config.get("hidden_size", 0)
    intermediate_size = config.get("intermediate_size") or config.get(
        "moe_intermediate_size", 0
    )
    return (num_experts, hidden_size, intermediate_size, tp)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: warmup_deep_gemm.py model1:tp1 [model2:tp2 ...]")
        print("Example: warmup_deep_gemm.py deepseek-ai/DeepSeek-V3-0324:8")
        sys.exit(0)

    # Parse model:tp pairs
    model_tp_pairs = []
    for arg in sys.argv[1:]:
        if ":" not in arg:
            print(f"Error: expected model:tp format, got '{arg}'")
            sys.exit(1)
        model, tp_str = arg.rsplit(":", 1)
        model_tp_pairs.append((model, int(tp_str)))

    print(f"=== DeepGEMM Warmup ({len(model_tp_pairs)} model(s) requested) ===\n")

    # Filter to models in HF cache
    cached = []
    for model, tp in model_tp_pairs:
        if model_in_hf_cache(model):
            cached.append((model, tp))
            print(f"  FOUND  {model} (tp={tp})")
        else:
            print(f"  SKIP   {model} (tp={tp}): not in HF cache")

    if not cached:
        print("\nNo requested models found in HF cache. Done.")
        return

    # Deduplicate by architecture
    arch_groups = {}
    for model, tp in cached:
        config = get_config_json(model)
        key = get_architecture_key(config, tp)
        if key not in arch_groups:
            arch_groups[key] = []
        arch_groups[key].append((model, tp))

    to_warmup = []
    print(
        f"\nDedup: {len(cached)} model(s) -> {len(arch_groups)} unique architecture(s)"
    )
    for key, group in arch_groups.items():
        rep = group[0]
        to_warmup.append(rep)
        if len(group) > 1:
            others = ", ".join(m for m, _ in group[1:])
            print(f"  {rep[0]} tp={rep[1]} (also covers: {others})")
        else:
            print(f"  {rep[0]} tp={rep[1]}")

    # Run compile_deep_gemm for each unique architecture
    for i, (model, tp) in enumerate(to_warmup, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(to_warmup)}] Warming up: {model} (tp={tp})")
        print(f"{'=' * 60}")
        cmd = [
            sys.executable,
            "-m",
            "sglang.compile_deep_gemm",
            "--model",
            model,
            "--tp",
            str(tp),
            "--trust-remote-code",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Warning: failed for {model} (exit code {result.returncode})")
            print("Continuing with remaining models...")
        else:
            print(f"Successfully warmed up: {model}")

    print("\nDeepGEMM warmup complete.")


if __name__ == "__main__":
    main()
