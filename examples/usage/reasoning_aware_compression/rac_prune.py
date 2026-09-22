#!/usr/bin/env python3
"""
Phase II of Reasoning-Aware Compression (RAC): one-shot prune a reasoning model
against the on-policy chain-of-thought calibration set built by
`rac_collect_traces.py`.

RAC (https://arxiv.org/abs/2509.12464, ICLR 2026) does not change the pruning
solver. Its whole contribution is which activations the solver reconstructs:
prompt tokens *plus* the model's own decode tokens (paper Eq. 7) instead of
prompt tokens alone. So this script is deliberately thin -- it hands the RAC
calibration set to `llm-compressor`'s SparseGPT or Wanda implementation and
saves the result as a checkpoint SGLang can serve.

Requires `llm-compressor`, which is NOT an SGLang dependency:

    pip install "llmcompressor>=0.12.0"

Example (reproduces the paper's 50%-sparsity math setting):

    python rac_prune.py \
        --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --calibration ./rac_traces_math/traces.jsonl \
        --sparsity 0.5 \
        --output-dir ./rac_pruned
"""

import argparse
import json
import os
import random
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

INSTALL_HINT = (
    "This script needs llm-compressor, which SGLang does not depend on.\n"
    "Install it with:\n\n"
    '    pip install "llmcompressor>=0.12.0"\n'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-shot prune a reasoning model on RAC calibration traces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model-path", required=True, help="Dense model to prune.")
    parser.add_argument(
        "--calibration",
        required=True,
        help="traces.jsonl produced by rac_collect_traces.py.",
    )
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--method",
        choices=["sparsegpt", "wanda"],
        default="sparsegpt",
        help="Layer-wise solver. The paper's headline results use SparseGPT.",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.5,
        help="Layer-wise sparsity. The paper sweeps 0.2-0.5.",
    )
    parser.add_argument(
        "--mask-structure",
        default="0:0",
        help="'0:0' is unstructured (the paper's setting). '2:4' gives a "
        "semi-structured mask.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="Calibration sequences longer than this are truncated.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Cap on calibration sequences used. Default: use all of them.",
    )
    parser.add_argument(
        "--pipeline",
        default="sequential",
        help="llm-compressor calibration pipeline. 'sequential' keeps only one "
        "decoder layer's Hessians resident, which is what fits on one GPU.",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_calibration_sequences(
    *, path: str, max_seq_length: int, num_samples: Optional[int], seed: int
) -> List[List[int]]:
    """Read RAC traces back as token id sequences."""
    sequences = []
    with open(path, "r", encoding="utf-8") as trace_file:
        for line in trace_file:
            input_ids = json.loads(line)["input_ids"]
            if input_ids:
                sequences.append(input_ids[:max_seq_length])

    if not sequences:
        raise ValueError(f"No calibration sequences found in {path}")

    if num_samples is not None and num_samples < len(sequences):
        random.Random(seed).shuffle(sequences)
        sequences = sequences[:num_samples]

    return sequences


def build_calibration_dataloader(sequences: List[List[int]]) -> DataLoader:
    """Wrap token id sequences as batches llm-compressor can calibrate on.

    Batch size is 1 on purpose. Batching sequences of different lengths would
    require padding, and pad-token activations would enter the layer-wise
    Hessian as if they were real ones -- exactly the calibration contamination
    RAC is about avoiding.
    """

    def collate(batch: List[List[int]]) -> dict:
        input_ids = torch.tensor(batch[0], dtype=torch.long).unsqueeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    return DataLoader(sequences, batch_size=1, shuffle=False, collate_fn=collate)


def build_recipe(*, method: str, sparsity: float, mask_structure: str):
    """Instantiate the layer-wise solver. RAC leaves this untouched."""
    try:
        from llmcompressor.modifiers.pruning import (
            SparseGPTModifier,
            WandaPruningModifier,
        )
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc

    modifier_cls = SparseGPTModifier if method == "sparsegpt" else WandaPruningModifier
    return modifier_cls(
        sparsity=sparsity,
        mask_structure=mask_structure,
        targets=["Linear"],
        ignore=["re:.*lm_head"],
    )


def measure_sparsity(model) -> float:
    """Fraction of zeros across the pruned Linear weights."""
    num_zeros = 0
    num_weights = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or "lm_head" in name:
            continue
        weight = module.weight
        num_zeros += int((weight == 0).sum().item())
        num_weights += weight.numel()

    return num_zeros / num_weights if num_weights else 0.0


def main() -> None:
    args = parse_args()

    try:
        from llmcompressor import oneshot
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc

    from transformers import AutoModelForCausalLM, AutoTokenizer

    sequences = load_calibration_sequences(
        path=args.calibration,
        max_seq_length=args.max_seq_length,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    num_calibration_tokens = sum(len(sequence) for sequence in sequences)
    print(
        f"[rac] calibrating on {len(sequences)} sequences "
        f"({num_calibration_tokens} tokens) from {args.calibration}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=getattr(torch, args.dtype),
        device_map=args.device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(
        f"[rac] pruning to {args.sparsity:.0%} sparsity "
        f"with {args.method} (mask_structure={args.mask_structure})"
    )
    model = oneshot(
        model=model,
        processor=tokenizer,
        dataset=build_calibration_dataloader(sequences),
        recipe=build_recipe(
            method=args.method,
            sparsity=args.sparsity,
            mask_structure=args.mask_structure,
        ),
        pipeline=args.pipeline,
        output_dir=args.output_dir,
    )

    realized = measure_sparsity(model)
    print(f"\n[rac] realized sparsity: {realized:.2%} (target {args.sparsity:.2%})")
    print(f"[rac] checkpoint written to {os.path.abspath(args.output_dir)}")
    print("\nServe it:")
    print(f"  python -m sglang.launch_server --model-path {args.output_dir}")
    print("\nOr score it against the dense model:")
    print(
        f"  python rac_serve_and_eval.py --model-path {args.output_dir} "
        "--num-problems 100"
    )


if __name__ == "__main__":
    main()
