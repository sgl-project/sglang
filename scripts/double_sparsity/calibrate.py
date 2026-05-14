"""Generate a Double Sparsity calibration JSON from a HuggingFace model.

For each transformer layer, accumulate per-channel statistics on the K
projection output across a small calibration set, then emit the top-S
channel indices per (layer, KV head).

Usage:
  python scripts/double_sparsity/calibrate.py \
      --model meta-llama/Meta-Llama-3.1-8B-Instruct \
      --output ./calib_8b.json \
      --heavy-channels 32 \
      --n-samples 64 --seq-len 4096

  # Synthetic mode (no dataset; useful for smoke tests):
  python scripts/double_sparsity/calibrate.py \
      --model meta-llama/Meta-Llama-3.1-8B-Instruct \
      --output ./calib_8b_synth.json \
      --synthetic --n-samples 8 --seq-len 1024

The output schema matches the parser at
`python/sglang/srt/mem_cache/sparsity/algorithms/double_sparsity_config.py`
(schema_version=1, indexing="global_kv_head_id", channel_type="k").

This script does NOT depend on the SGLang server. It loads the HF model
directly with `AutoModelForCausalLM`, hooks `k_proj` outputs, runs a few
forward passes, and writes the JSON. Calibration is single-replica
(TP=1); per-rank slicing happens at server startup.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _load_calibration_prompts(args, tokenizer) -> List[torch.Tensor]:
    """Return a list of input_ids tensors for `args.n_samples` prompts."""
    if args.synthetic:
        torch.manual_seed(args.seed)
        vocab = tokenizer.vocab_size
        return [
            torch.randint(0, vocab, (1, args.seq_len), dtype=torch.long)
            for _ in range(args.n_samples)
        ]
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise SystemExit(
                "datasets package required for HuggingFace calibration. "
                "Install with `pip install datasets`, or pass --synthetic / "
                "--prompts-file."
            ) from e
        ds = load_dataset(args.dataset, args.dataset_subset, split="train")
        text_field = args.dataset_text_field
        texts = [row[text_field] for row in ds if row[text_field].strip()]

    out = []
    for txt in texts:
        if len(out) >= args.n_samples:
            break
        toks = tokenizer(
            txt, return_tensors="pt", truncation=True, max_length=args.seq_len
        )
        if toks.input_ids.shape[1] >= max(args.min_prompt_len, 64):
            out.append(toks.input_ids)
    if len(out) < args.n_samples:
        logger.warning(
            "only %d prompts >= min length found (asked for %d); proceeding",
            len(out),
            args.n_samples,
        )
    return out


def _find_k_proj_modules(model):
    """Return list of (layer_id, k_proj module) pairs in layer order.

    Tested on Llama-style models where each transformer block has
    `self_attn.k_proj`. Errors if no matching modules are found.
    """
    pairs = []
    for name, module in model.named_modules():
        if name.endswith(".self_attn.k_proj"):
            # Extract layer id from a name like "model.layers.31.self_attn.k_proj"
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        layer_id = int(parts[i + 1])
                        pairs.append((layer_id, module))
                        break
                    except ValueError:
                        pass
    if not pairs:
        raise RuntimeError(
            "no `*.self_attn.k_proj` modules found; "
            "this script targets Llama-style models. Adapt for other archs."
        )
    pairs.sort(key=lambda x: x[0])
    return pairs


def calibrate(args) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading tokenizer + model %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)

    # Multi-GPU loading for >24-32GB models (e.g., Llama-3.1-70B in bf16
    # cannot land on a single H200 once forward activations + KV are added).
    # When --device-map none (the historical default), keep the previous
    # single-replica behavior unchanged. When set (e.g., "auto"), let
    # Accelerate place layers across visible GPUs and skip the explicit .to().
    use_device_map = args.device_map and args.device_map != "none"
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    if use_device_map:
        load_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    # Disable HF KV cache during calibration forwards. Otherwise HF allocates
    # full prompt KV at every forward, which is exactly the memory pressure
    # device_map="auto" is trying to avoid. We only need K projections via
    # forward hooks; no autoregressive generation.
    model.config.use_cache = False
    if not use_device_map:
        model = model.to(args.device)
    model.eval()

    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    num_layers = cfg.num_hidden_layers
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    if not (0 < args.heavy_channels <= head_dim):
        raise ValueError(
            f"heavy_channels={args.heavy_channels} must be in (0, {head_dim}]"
        )

    pairs = _find_k_proj_modules(model)
    if len(pairs) != num_layers:
        logger.warning(
            "found %d k_proj modules but cfg.num_hidden_layers=%d",
            len(pairs),
            num_layers,
        )

    # Per-layer accumulator. Lazily allocated on the same device as each
    # hooked k_proj output's first activation — so Accelerate-placed layers
    # don't trigger cross-device adds. Final aggregation in `topk` step
    # moves everything to CPU.
    accum: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {layer_id: 0 for layer_id, _ in pairs}

    def make_hook(layer_id: int):
        def _hook(_module, _inp, out):
            # out shape: [..., num_kv_heads * head_dim]; reshape to [..., kv_heads, head_dim]
            x = out.detach()
            shape = (-1, num_kv_heads, head_dim)
            x = x.reshape(shape).to(torch.float64)
            stat = x.abs().mean(dim=0)
            if layer_id not in accum:
                accum[layer_id] = torch.zeros_like(stat)
            accum[layer_id] += stat
            counts[layer_id] += 1

        return _hook

    handles = [m.register_forward_hook(make_hook(lid)) for lid, m in pairs]
    # Where to put input_ids. With device_map="auto", embeddings normally live
    # on cuda:0; otherwise the user's explicit --device.
    input_device = (
        next(model.parameters()).device if use_device_map else args.device
    )
    try:
        prompts = _load_calibration_prompts(args, tokenizer)
        logger.info(
            "running %d calibration forwards (seq_len cap=%d, device_map=%s)",
            len(prompts),
            args.seq_len,
            args.device_map if use_device_map else "(single-replica)",
        )
        with torch.inference_mode():
            for i, ids in enumerate(prompts):
                t0 = time.time()
                model(ids.to(input_device), use_cache=False)
                if i % max(1, len(prompts) // 8) == 0:
                    logger.info(
                        "calibration step %d/%d (%.1fs)",
                        i + 1,
                        len(prompts),
                        time.time() - t0,
                    )
    finally:
        for h in handles:
            h.remove()

    # Per-layer top-S channel indices per kv_head, by accumulated abs_mean.
    # Move accumulators to CPU before topk to avoid any cross-device ops.
    channels = {}
    for layer_id in sorted(accum.keys()):
        importance = accum[layer_id].cpu()  # [num_kv_heads, head_dim]
        topk = importance.topk(args.heavy_channels, dim=1).indices.to(torch.int32)
        channels[str(layer_id)] = topk.tolist()

    return {
        "schema_version": SCHEMA_VERSION,
        "model_arch": cfg.architectures[0] if cfg.architectures else "unknown",
        "model_name_or_path": args.model,
        "head_dim": head_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "heavy_channels": args.heavy_channels,
        "channel_type": "k",
        "indexing": "global_kv_head_id",
        "calibration": {
            "dataset": (
                "synthetic" if args.synthetic else (args.prompts_file or args.dataset)
            ),
            "n_samples": args.n_samples,
            "seq_len": args.seq_len,
            "scoring": "abs_mean",
            "tp_size_at_calibration": 1,
            "git_sha": _git_sha(),
        },
        "channels": channels,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--heavy-channels", type=int, default=32)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--min-prompt-len", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--device-map",
        type=str,
        default="none",
        help=(
            "HuggingFace device_map. 'none' (default) preserves single-GPU "
            "behavior with --device. Use 'auto' for multi-GPU loading "
            "(required for 70B+ models in bf16 that exceed one H200's HBM)."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Run calibration on uniformly random token ids. No dataset needed; "
        "produces a usable but lower-quality calibration. Useful for smoke "
        "tests / e2e fixtures without HF dataset access.",
    )
    p.add_argument(
        "--dataset",
        default="wikitext",
        help="HF dataset name (default: wikitext); ignored under --synthetic.",
    )
    p.add_argument(
        "--dataset-subset",
        default="wikitext-2-raw-v1",
        help="HF dataset subset (default: wikitext-2-raw-v1).",
    )
    p.add_argument(
        "--dataset-text-field",
        default="text",
        help="Which field in each row carries the prompt text.",
    )
    p.add_argument(
        "--prompts-file",
        default=None,
        help="Optional newline-delimited prompts file; overrides --dataset.",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blob = calibrate(args)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(blob, f)
    logger.info(
        "wrote calibration: %s (layers=%d kv_heads=%d S=%d)",
        out_path,
        blob["num_layers"],
        blob["num_kv_heads"],
        blob["heavy_channels"],
    )


if __name__ == "__main__":
    main()
