#!/usr/bin/env python3
"""
SINQ Comprehensive Quantization Evaluation

Tests all SINQ quantization configurations and generates a heatmap
showing Jaccard similarity (attention pattern preservation) across:
- nbits: 2, 3, 4, 5, 6, 8
- tiling_mode: 1D, 2D
- group_size: 64, 128
- method: sinq, asinq

Usage:
    python sinq_heatmap_eval.py --model Qwen/Qwen3-1.7B
    python sinq_heatmap_eval.py --model Qwen/Qwen3-1.7B --quick  # Fewer prompts
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback

import torch
import numpy as np

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Model loading
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class QuantConfig:
    """Quantization configuration."""
    nbits: int
    tiling_mode: str
    group_size: int
    method: str

    @property
    def name(self) -> str:
        return f"{self.method}_{self.nbits}b_g{self.group_size}_{self.tiling_mode}"


# All configurations to test
NBITS_OPTIONS = [2, 3, 4, 5, 6, 8]
TILING_OPTIONS = ["1D", "2D"]
GROUP_SIZE_OPTIONS = [64, 128]
METHOD_OPTIONS = ["sinq", "asinq"]

# Test prompts
EVAL_PROMPTS = [
    "What are the three primary colors?",
    "Explain gravity in one sentence.",
    "What is the capital of France?",
    "Name three planets in our solar system.",
    "What does DNA stand for?",
    "Write a haiku about the moon.",
    "What is 15 + 27?",
    "Who wrote Romeo and Juliet?",
    "What is photosynthesis?",
    "Name the four seasons.",
]

QUICK_PROMPTS = [
    "What is 2 + 2?",
    "Name a color.",
    "What is the sun?",
]


# ============================================================================
# EVALUATION
# ============================================================================

@dataclass
class ConfigResult:
    """Result for a single configuration."""
    config: QuantConfig
    mean_jaccard: float
    std_jaccard: float
    min_jaccard: float
    max_jaccard: float
    compression_ratio: float
    bf16_memory_mb: float
    quant_memory_mb: float
    error: Optional[str] = None
    per_prompt_jaccard: List[float] = field(default_factory=list)
    duration_seconds: float = 0


def compute_jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def get_top_tokens(logits: torch.Tensor, k: int = 10) -> set:
    """Get top-k token indices from logits."""
    probs = torch.softmax(logits, dim=-1)
    top_indices = torch.topk(probs, k).indices
    return set(top_indices.cpu().numpy().tolist())


def get_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0


def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class SINQEvaluator:
    """Evaluates SINQ quantization configurations."""

    def __init__(self, model_name: str, device: str = "cuda:0"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.bf16_model = None
        self.bf16_memory = 0

    def load_bf16_baseline(self):
        """Load the BF16 baseline model."""
        print(f"\n{'='*60}")
        print(f"Loading BF16 baseline: {self.model_name}")
        print(f"{'='*60}")

        clear_gpu_memory()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.bf16_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.bf16_model.eval()

        self.bf16_memory = get_memory_mb()
        print(f"BF16 model memory: {self.bf16_memory:.1f} MB")

    def get_bf16_outputs(self, prompts: List[str], max_tokens: int = 64) -> Dict[str, Tuple[str, List[set]]]:
        """Generate outputs and collect top-k tokens from BF16 model."""
        results = {}

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_len = inputs.input_ids.shape[1]

            # Generate with output scores
            with torch.no_grad():
                outputs = self.bf16_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode response
            response = self.tokenizer.decode(
                outputs.sequences[0][input_len:],
                skip_special_tokens=True
            )

            # Collect top-k tokens for each generated position
            topk_per_step = []
            for score in outputs.scores:
                topk_per_step.append(get_top_tokens(score[0], k=10))

            results[prompt] = (response, topk_per_step)

        return results

    def evaluate_config(
        self,
        config: QuantConfig,
        prompts: List[str],
        bf16_outputs: Dict[str, Tuple[str, List[set]]],
        max_tokens: int = 64,
    ) -> ConfigResult:
        """Evaluate a single quantization configuration."""
        print(f"\n{'-'*60}")
        print(f"Testing: {config.name}")
        print(f"  nbits={config.nbits}, group_size={config.group_size}, "
              f"tiling={config.tiling_mode}, method={config.method}")
        print(f"{'-'*60}")

        start_time = time.time()

        try:
            # Import SINQ
            from sinq.patch_model import AutoSINQHFModel
            from sinq.sinqlinear import sinq_base_quant_config

            # Clear memory
            clear_gpu_memory()

            # Load fresh model for quantization
            model_to_quant = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(self.device)

            # Create quant config
            quant_config = sinq_base_quant_config(
                nbits=config.nbits,
                group_size=config.group_size,
                tiling_mode=config.tiling_mode,
                method=config.method,
                axis=1,
            )

            # Quantize
            print(f"  Quantizing with {config.method}...")
            AutoSINQHFModel.quantize_model(
                model_to_quant,
                self.tokenizer,
                quant_config=quant_config,
                compute_dtype=torch.bfloat16,
                device=self.device,
            )
            model_to_quant.eval()

            quant_memory = get_memory_mb()
            compression_ratio = self.bf16_memory / quant_memory if quant_memory > 0 else 0
            print(f"  Quantized memory: {quant_memory:.1f} MB (compression: {compression_ratio:.2f}x)")

            # Evaluate on prompts
            jaccard_scores = []

            for prompt in prompts:
                bf16_response, bf16_topk = bf16_outputs[prompt]

                # Generate with quantized model
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                input_len = inputs.input_ids.shape[1]

                with torch.no_grad():
                    outputs = model_to_quant.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                # Collect top-k tokens
                quant_topk = []
                for score in outputs.scores:
                    quant_topk.append(get_top_tokens(score[0], k=10))

                # Compute per-step Jaccard
                min_len = min(len(bf16_topk), len(quant_topk))
                if min_len > 0:
                    step_jaccards = [
                        compute_jaccard_similarity(bf16_topk[i], quant_topk[i])
                        for i in range(min_len)
                    ]
                    prompt_jaccard = np.mean(step_jaccards)
                else:
                    prompt_jaccard = 0.0

                jaccard_scores.append(prompt_jaccard)
                print(f"  Prompt '{prompt[:30]}...': Jaccard={prompt_jaccard:.3f}")

            # Cleanup
            del model_to_quant
            clear_gpu_memory()

            duration = time.time() - start_time

            return ConfigResult(
                config=config,
                mean_jaccard=float(np.mean(jaccard_scores)),
                std_jaccard=float(np.std(jaccard_scores)),
                min_jaccard=float(np.min(jaccard_scores)),
                max_jaccard=float(np.max(jaccard_scores)),
                compression_ratio=compression_ratio,
                bf16_memory_mb=self.bf16_memory,
                quant_memory_mb=quant_memory,
                per_prompt_jaccard=jaccard_scores,
                duration_seconds=duration,
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            clear_gpu_memory()

            return ConfigResult(
                config=config,
                mean_jaccard=0.0,
                std_jaccard=0.0,
                min_jaccard=0.0,
                max_jaccard=0.0,
                compression_ratio=0.0,
                bf16_memory_mb=self.bf16_memory,
                quant_memory_mb=0.0,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_heatmaps(results: List[ConfigResult], output_dir: Path):
    """Create heatmap visualizations of the results."""

    # Organize data by method
    for method in METHOD_OPTIONS:
        method_results = [r for r in results if r.config.method == method and r.error is None]
        if not method_results:
            continue

        # Create heatmap data for each tiling mode
        for tiling in TILING_OPTIONS:
            tiling_results = [r for r in method_results if r.config.tiling_mode == tiling]
            if not tiling_results:
                continue

            # Build matrix: rows = nbits, cols = group_size
            nbits_list = sorted(set(r.config.nbits for r in tiling_results))
            group_list = sorted(set(r.config.group_size for r in tiling_results))

            jaccard_matrix = np.zeros((len(nbits_list), len(group_list)))
            compression_matrix = np.zeros((len(nbits_list), len(group_list)))

            for r in tiling_results:
                i = nbits_list.index(r.config.nbits)
                j = group_list.index(r.config.group_size)
                jaccard_matrix[i, j] = r.mean_jaccard
                compression_matrix[i, j] = r.compression_ratio

            # Create figure with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Jaccard heatmap
            sns.heatmap(
                jaccard_matrix,
                ax=axes[0],
                xticklabels=[f"g{g}" for g in group_list],
                yticklabels=[f"{n}b" for n in nbits_list],
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Jaccard Similarity'}
            )
            axes[0].set_xlabel('Group Size')
            axes[0].set_ylabel('Bits')
            axes[0].set_title(f'{method.upper()} {tiling} - Jaccard Similarity\n(Higher = Better Quality)')

            # Compression heatmap
            sns.heatmap(
                compression_matrix,
                ax=axes[1],
                xticklabels=[f"g{g}" for g in group_list],
                yticklabels=[f"{n}b" for n in nbits_list],
                annot=True,
                fmt='.2f',
                cmap='Blues',
                cbar_kws={'label': 'Compression Ratio'}
            )
            axes[1].set_xlabel('Group Size')
            axes[1].set_ylabel('Bits')
            axes[1].set_title(f'{method.upper()} {tiling} - Compression Ratio\n(Higher = Smaller Model)')

            plt.tight_layout()

            filename = output_dir / f"heatmap_{method}_{tiling}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {filename}")

    # Create combined summary heatmap
    create_summary_heatmap(results, output_dir)


def create_summary_heatmap(results: List[ConfigResult], output_dir: Path):
    """Create a summary heatmap showing all configurations."""

    # Filter out errors
    valid_results = [r for r in results if r.error is None]
    if not valid_results:
        print("No valid results for summary heatmap")
        return

    # Create labels and data
    configs = []
    jaccards = []
    compressions = []

    for r in sorted(valid_results, key=lambda x: (-x.config.nbits, x.config.group_size)):
        label = f"{r.config.method[:1].upper()}-{r.config.nbits}b-g{r.config.group_size}-{r.config.tiling_mode}"
        configs.append(label)
        jaccards.append(r.mean_jaccard)
        compressions.append(r.compression_ratio)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, jaccards, width, label='Jaccard Similarity', color='green', alpha=0.7)

    # Secondary axis for compression
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, compressions, width, label='Compression Ratio', color='blue', alpha=0.7)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Jaccard Similarity', color='green')
    ax2.set_ylabel('Compression Ratio', color='blue')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax2.set_ylim(0, max(compressions) * 1.2)

    # Add quality threshold lines
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (0.8)')
    ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Acceptable (0.6)')

    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('SINQ Quantization: Quality vs Compression Trade-off')
    plt.tight_layout()

    filename = output_dir / "summary_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def create_pareto_plot(results: List[ConfigResult], output_dir: Path):
    """Create Pareto frontier plot (quality vs compression)."""

    valid_results = [r for r in results if r.error is None]
    if not valid_results:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color by method
    colors = {'sinq': 'blue', 'asinq': 'red'}
    markers = {'1D': 'o', '2D': 's'}

    for r in valid_results:
        color = colors[r.config.method]
        marker = markers[r.config.tiling_mode]
        label = f"{r.config.method}-{r.config.tiling_mode}"

        ax.scatter(
            r.compression_ratio,
            r.mean_jaccard,
            c=color,
            marker=marker,
            s=100 + r.config.nbits * 20,  # Size by nbits
            alpha=0.7,
            label=label if label not in [t.get_text() for t in ax.get_legend_handles_labels()[1]] else "",
        )

        # Annotate with config
        ax.annotate(
            f"{r.config.nbits}b-g{r.config.group_size}",
            (r.compression_ratio, r.mean_jaccard),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xlabel('Compression Ratio (higher = smaller model)')
    ax.set_ylabel('Jaccard Similarity (higher = better quality)')
    ax.set_title('SINQ Quantization Pareto Frontier')

    # Add quality threshold lines
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.3, label='Good Quality')
    ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.3, label='Acceptable')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Poor')

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = output_dir / "pareto_frontier.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SINQ Comprehensive Evaluation")
    parser.add_argument("--model", "-m", default="Qwen/Qwen3-1.7B", help="Model to evaluate")
    parser.add_argument("--output", "-o", default="sinq_eval_results", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer prompts")
    parser.add_argument("--nbits", type=int, nargs="+", help="Specific nbits to test")
    parser.add_argument("--methods", nargs="+", choices=["sinq", "asinq"], help="Specific methods")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens to generate")

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select prompts
    prompts = QUICK_PROMPTS if args.quick else EVAL_PROMPTS

    # Select configurations
    nbits_to_test = args.nbits or NBITS_OPTIONS
    methods_to_test = args.methods or METHOD_OPTIONS

    configs = []
    for nbits in nbits_to_test:
        for tiling in TILING_OPTIONS:
            for group_size in GROUP_SIZE_OPTIONS:
                for method in methods_to_test:
                    configs.append(QuantConfig(
                        nbits=nbits,
                        tiling_mode=tiling,
                        group_size=group_size,
                        method=method,
                    ))

    print(f"\n{'='*60}")
    print(f"SINQ COMPREHENSIVE EVALUATION")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Configurations to test: {len(configs)}")
    print(f"Prompts: {len(prompts)}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Initialize evaluator
    evaluator = SINQEvaluator(args.model)
    evaluator.load_bf16_baseline()

    # Get BF16 baseline outputs
    print("\nGenerating BF16 baseline outputs...")
    bf16_outputs = evaluator.get_bf16_outputs(prompts, args.max_tokens)

    # Evaluate all configurations
    results = []
    start_time = time.time()

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] ", end="")
        result = evaluator.evaluate_config(config, prompts, bf16_outputs, args.max_tokens)
        results.append(result)

        # Save intermediate results
        if (i + 1) % 5 == 0:
            save_results(results, output_dir)

    total_time = time.time() - start_time

    # Save final results
    save_results(results, output_dir)

    # Generate visualizations
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    create_heatmaps(results, output_dir)
    create_pareto_plot(results, output_dir)

    # Print summary
    print_summary(results, total_time)

    return 0


def save_results(results: List[ConfigResult], output_dir: Path):
    """Save results to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "config": asdict(r.config),
                "mean_jaccard": r.mean_jaccard,
                "std_jaccard": r.std_jaccard,
                "min_jaccard": r.min_jaccard,
                "max_jaccard": r.max_jaccard,
                "compression_ratio": r.compression_ratio,
                "bf16_memory_mb": r.bf16_memory_mb,
                "quant_memory_mb": r.quant_memory_mb,
                "error": r.error,
                "per_prompt_jaccard": r.per_prompt_jaccard,
                "duration_seconds": r.duration_seconds,
            }
            for r in results
        ]
    }

    filename = output_dir / "results.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def print_summary(results: List[ConfigResult], total_time: float):
    """Print evaluation summary."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)

    valid = [r for r in results if r.error is None]
    errors = [r for r in results if r.error is not None]

    print(f"\nTotal configurations: {len(results)}")
    print(f"Successful: {len(valid)}")
    print(f"Failed: {len(errors)}")
    print(f"Total time: {total_time/60:.1f} minutes")

    if errors:
        print("\nFailed configurations:")
        for r in errors:
            print(f"  - {r.config.name}: {r.error[:50]}...")

    if valid:
        # Sort by Jaccard
        sorted_by_quality = sorted(valid, key=lambda x: -x.mean_jaccard)

        print("\n" + "-"*70)
        print("TOP 10 BY QUALITY (Jaccard Similarity)")
        print("-"*70)
        print(f"{'Config':<35} {'Jaccard':>10} {'Compression':>12} {'Memory MB':>10}")
        print("-"*70)
        for r in sorted_by_quality[:10]:
            print(f"{r.config.name:<35} {r.mean_jaccard:>10.3f} {r.compression_ratio:>12.2f}x {r.quant_memory_mb:>10.0f}")

        # Sort by compression
        sorted_by_compression = sorted(valid, key=lambda x: -x.compression_ratio)

        print("\n" + "-"*70)
        print("TOP 10 BY COMPRESSION")
        print("-"*70)
        print(f"{'Config':<35} {'Compression':>12} {'Jaccard':>10} {'Memory MB':>10}")
        print("-"*70)
        for r in sorted_by_compression[:10]:
            print(f"{r.config.name:<35} {r.compression_ratio:>12.2f}x {r.mean_jaccard:>10.3f} {r.quant_memory_mb:>10.0f}")

        # Best trade-off (highest Jaccard with compression > 2x)
        good_compression = [r for r in valid if r.compression_ratio >= 2.0]
        if good_compression:
            best_tradeoff = max(good_compression, key=lambda x: x.mean_jaccard)
            print(f"\n{'='*70}")
            print(f"RECOMMENDED CONFIG (Best quality with >2x compression):")
            print(f"  {best_tradeoff.config.name}")
            print(f"  Jaccard: {best_tradeoff.mean_jaccard:.3f}")
            print(f"  Compression: {best_tradeoff.compression_ratio:.2f}x")
            print(f"  Memory: {best_tradeoff.quant_memory_mb:.0f} MB")
            print(f"{'='*70}")


if __name__ == "__main__":
    exit(main())
