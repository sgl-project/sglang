#!/usr/bin/env python3
"""
SINQ Comprehensive Quantization Evaluation (Schema v1)

Tests all SINQ quantization configurations and generates:
- Heatmaps showing Jaccard similarity across configurations
- Schema v1 comparison reports with full metrics:
  * Jaccard similarity (set overlap)
  * Weighted Jaccard (score-weighted overlap)
  * Rank correlation (Spearman/Kendall)
  * Mass retained (baseline attention preserved)
  * KL divergence (distribution shift)

Configurations tested:
- nbits: 2, 3, 4, 5, 6, 8
- tiling_mode: 1D, 2D
- group_size: 64, 128
- method: sinq, asinq

Usage:
    python sinq_heatmap_eval.py --model Qwen/Qwen3-1.7B
    python sinq_heatmap_eval.py --model Qwen/Qwen3-1.7B --quick  # Fewer prompts
    python sinq_heatmap_eval.py --model Qwen/Qwen3-1.7B --schema-v1  # Full schema output
"""

import argparse
import gc
import json
import sys
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Visualization
import matplotlib
import numpy as np
import torch
from scipy import stats

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Model loading
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import manifest module for reproducibility layer
try:
    # Try to import from attention_explorer schemas
    sys.path.insert(
        0, str(Path(__file__).parent.parent / "examples" / "attention_explorer")
    )
    from schemas.manifest import ExperimentManifest, RunType

    HAS_MANIFEST = True
except ImportError:
    HAS_MANIFEST = False


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
# MANIFOLD ZONE CLASSIFICATION
# ============================================================================

# Manifold zones based on attention distance distribution
ManifoldZone = str  # Type alias: 'syntax_floor' | 'semantic_bridge' | 'long_range' | 'structure_ripple' | 'diffuse' | 'unknown'

MANIFOLD_ZONES = [
    "syntax_floor",
    "semantic_bridge",
    "long_range",
    "structure_ripple",
    "diffuse",
    "unknown",
]


@dataclass
class AttentionFingerprint:
    """Fingerprint computed from attention positions and scores."""

    local_mass: float  # Attention on nearby tokens (distance 0-32)
    mid_mass: float  # Attention on mid-range tokens (distance 33-256)
    long_mass: float  # Attention on distant tokens (distance 257+)
    entropy: float  # Distribution entropy
    histogram: List[float] = field(default_factory=list)  # Distance histogram


def compute_fingerprint(
    positions: List[int],
    scores: List[float],
    current_pos: int,
) -> AttentionFingerprint:
    """
    Compute attention fingerprint from positions and scores.

    Distance bands:
    - Local: 0-32 tokens (syntax/immediate context)
    - Mid: 33-256 tokens (semantic/paragraph level)
    - Long: 257+ tokens (document-level/cross-context)
    """
    if not positions or not scores:
        return AttentionFingerprint(
            local_mass=0.0,
            mid_mass=0.0,
            long_mass=0.0,
            entropy=0.0,
            histogram=[0.0] * 16,
        )

    # Compute distance histogram (16 bins, log-scale)
    histogram = [0.0] * 16
    local_mass = 0.0
    mid_mass = 0.0
    long_mass = 0.0
    total_mass = sum(scores)

    if total_mass == 0:
        return AttentionFingerprint(
            local_mass=0.0,
            mid_mass=0.0,
            long_mass=0.0,
            entropy=0.0,
            histogram=histogram,
        )

    for pos, score in zip(positions, scores):
        distance = max(1, current_pos - pos)

        # Distance bands
        if distance <= 32:
            local_mass += score
        elif distance <= 256:
            mid_mass += score
        else:
            long_mass += score

        # Log-scale histogram bin
        bin_idx = min(15, int(np.log2(distance)))
        histogram[bin_idx] += score

    # Normalize
    local_mass /= total_mass
    mid_mass /= total_mass
    long_mass /= total_mass
    histogram = [h / total_mass for h in histogram]

    # Compute entropy
    entropy = 0.0
    for score in scores:
        if score > 0:
            p = score / total_mass
            entropy -= p * np.log2(p + 1e-10)

    return AttentionFingerprint(
        local_mass=local_mass,
        mid_mass=mid_mass,
        long_mass=long_mass,
        entropy=entropy,
        histogram=histogram,
    )


def classify_manifold_zone(fp: AttentionFingerprint) -> ManifoldZone:
    """
    Classify attention fingerprint into a manifold zone.

    Zones:
    - syntax_floor: High local attention (formatting, JSON, code syntax)
    - semantic_bridge: High mid-range attention (retrieval, semantic connections)
    - long_range: High long-range attention (document planning, cross-context)
    - structure_ripple: Periodic patterns (code blocks, lists)
    - diffuse: Distributed attention (creative, exploratory)
    """
    if fp.local_mass > 0.5:
        return "syntax_floor"
    if fp.mid_mass > 0.4:
        return "semantic_bridge"
    if fp.long_mass > 0.35:
        return "long_range"
    if fp.entropy < 2.0 and fp.local_mass > 0.3:
        return "structure_ripple"
    if fp.entropy > 3.5:
        return "diffuse"
    return "diffuse"  # Default


def compute_zone_from_topk(
    positions: List[int],
    scores: List[float],
    current_pos: int,
) -> ManifoldZone:
    """Compute manifold zone from top-k attention positions."""
    fp = compute_fingerprint(positions, scores, current_pos)
    return classify_manifold_zone(fp)


# ============================================================================
# METRICS COMPUTATION
# ============================================================================


@dataclass
class StepMetrics:
    """Metrics for a single generation step."""

    jaccard: float
    weighted_jaccard: float
    spearman: float
    kendall: float
    mass_retained: float
    kl_divergence: float


@dataclass
class PromptMetrics:
    """Aggregated metrics for a single prompt."""

    prompt_id: str
    prompt_text: str
    jaccard: float
    weighted_jaccard: float
    spearman: float
    kendall: float
    mass_retained: float
    kl_divergence: float
    output_match: bool
    # Manifold zone tracking
    baseline_zone: ManifoldZone = "unknown"
    candidate_zone: ManifoldZone = "unknown"
    zone_drift: bool = False
    per_step_jaccard: List[float] = field(default_factory=list)


@dataclass
class ConfigResult:
    """Result for a single configuration."""

    config: QuantConfig
    mean_jaccard: float
    std_jaccard: float
    min_jaccard: float
    max_jaccard: float
    # New metrics
    mean_weighted_jaccard: float = 0.0
    mean_spearman: float = 0.0
    mean_kendall: float = 0.0
    mean_mass_retained: float = 0.0
    mean_kl_divergence: float = 0.0
    output_match_rate: float = 0.0
    # Manifold analysis
    zone_drift_rate: float = 0.0
    zone_transition_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Existing fields
    compression_ratio: float = 0.0
    bf16_memory_mb: float = 0.0
    quant_memory_mb: float = 0.0
    error: Optional[str] = None
    per_prompt_jaccard: List[float] = field(default_factory=list)
    per_prompt_metrics: List[PromptMetrics] = field(default_factory=list)
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


def compute_weighted_jaccard(
    positions1: List[int],
    scores1: List[float],
    positions2: List[int],
    scores2: List[float],
) -> float:
    """
    Compute weighted Jaccard using scores as weights.

    For shared positions: weight = min(score1, score2)
    For union: weight = max of available scores
    """
    if not positions1 and not positions2:
        return 1.0

    pos_to_score1 = dict(zip(positions1, scores1))
    pos_to_score2 = dict(zip(positions2, scores2))

    all_positions = set(positions1) | set(positions2)
    if not all_positions:
        return 1.0

    intersection_weight = 0.0
    union_weight = 0.0

    for pos in all_positions:
        s1 = pos_to_score1.get(pos, 0.0)
        s2 = pos_to_score2.get(pos, 0.0)

        if pos in pos_to_score1 and pos in pos_to_score2:
            intersection_weight += min(s1, s2)
        union_weight += max(s1, s2)

    return intersection_weight / union_weight if union_weight > 0 else 0.0


def compute_rank_correlation(
    positions1: List[int],
    positions2: List[int],
) -> Tuple[float, float]:
    """
    Compute Spearman and Kendall rank correlation for overlapping positions.

    Returns (spearman_rho, kendall_tau)
    """
    common = set(positions1) & set(positions2)
    if len(common) < 2:
        return 0.0, 0.0

    # Get ranks in each list for common positions
    rank1 = {pos: i for i, pos in enumerate(positions1)}
    rank2 = {pos: i for i, pos in enumerate(positions2)}

    common_list = list(common)
    ranks1 = [rank1[pos] for pos in common_list]
    ranks2 = [rank2[pos] for pos in common_list]

    try:
        spearman, _ = stats.spearmanr(ranks1, ranks2)
        kendall, _ = stats.kendalltau(ranks1, ranks2)
        return (
            float(spearman) if not np.isnan(spearman) else 0.0,
            float(kendall) if not np.isnan(kendall) else 0.0,
        )
    except Exception:
        return 0.0, 0.0


def compute_mass_retained(
    baseline_positions: List[int],
    baseline_scores: List[float],
    candidate_positions: List[int],
) -> float:
    """
    Compute fraction of baseline attention mass preserved in candidate top-k.
    """
    if not baseline_positions or not baseline_scores:
        return 1.0

    candidate_set = set(candidate_positions)
    total_mass = sum(baseline_scores)
    retained_mass = sum(
        score
        for pos, score in zip(baseline_positions, baseline_scores)
        if pos in candidate_set
    )

    return retained_mass / total_mass if total_mass > 0 else 0.0


def compute_kl_divergence(
    baseline_scores: List[float],
    candidate_scores: List[float],
    epsilon: float = 1e-10,
) -> float:
    """
    Compute KL divergence between truncated distributions.
    """
    if not baseline_scores or not candidate_scores:
        return 0.0

    # Normalize
    p = np.array(baseline_scores) + epsilon
    q = np.array(candidate_scores) + epsilon
    p = p / p.sum()
    q = q / q.sum()

    # Pad to same length
    max_len = max(len(p), len(q))
    p = np.pad(p, (0, max_len - len(p)), constant_values=epsilon)
    q = np.pad(q, (0, max_len - len(q)), constant_values=epsilon)

    # Renormalize
    p = p / p.sum()
    q = q / q.sum()

    return float(stats.entropy(p, q))


def compute_step_metrics(
    baseline_positions: List[int],
    baseline_scores: List[float],
    candidate_positions: List[int],
    candidate_scores: List[float],
) -> StepMetrics:
    """Compute all metrics for a single generation step."""
    jaccard = compute_jaccard_similarity(
        set(baseline_positions), set(candidate_positions)
    )
    weighted_jaccard = compute_weighted_jaccard(
        baseline_positions,
        baseline_scores,
        candidate_positions,
        candidate_scores,
    )
    spearman, kendall = compute_rank_correlation(
        baseline_positions, candidate_positions
    )
    mass_retained = compute_mass_retained(
        baseline_positions, baseline_scores, candidate_positions
    )
    kl_div = compute_kl_divergence(baseline_scores, candidate_scores)

    return StepMetrics(
        jaccard=jaccard,
        weighted_jaccard=weighted_jaccard,
        spearman=spearman,
        kendall=kendall,
        mass_retained=mass_retained,
        kl_divergence=kl_div,
    )


def get_top_tokens_with_scores(
    logits: torch.Tensor, k: int = 10
) -> Tuple[List[int], List[float]]:
    """Get top-k token indices and their probability scores from logits."""
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k)
    return (
        top_indices.cpu().numpy().tolist(),
        top_probs.cpu().numpy().tolist(),
    )


def get_top_tokens(logits: torch.Tensor, k: int = 10) -> set:
    """Get top-k token indices from logits (legacy compatibility)."""
    positions, _ = get_top_tokens_with_scores(logits, k)
    return set(positions)


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

    def get_bf16_outputs(
        self, prompts: List[str], max_tokens: int = 64, top_k: int = 10
    ) -> Dict[str, Tuple[str, List[Tuple[List[int], List[float]]]]]:
        """
        Generate outputs and collect top-k tokens with scores from BF16 model.

        Returns dict mapping prompt -> (response, list of (positions, scores) per step)
        """
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
                outputs.sequences[0][input_len:], skip_special_tokens=True
            )

            # Collect top-k tokens WITH SCORES for each generated position
            topk_per_step = []
            for score in outputs.scores:
                positions, scores = get_top_tokens_with_scores(score[0], k=top_k)
                topk_per_step.append((positions, scores))

            results[prompt] = (response, topk_per_step)

        return results

    def evaluate_config(
        self,
        config: QuantConfig,
        prompts: List[str],
        bf16_outputs: Dict[str, Tuple[str, List[Tuple[List[int], List[float]]]]],
        max_tokens: int = 64,
        top_k: int = 10,
    ) -> ConfigResult:
        """Evaluate a single quantization configuration with full metrics."""
        print(f"\n{'-'*60}")
        print(f"Testing: {config.name}")
        print(
            f"  nbits={config.nbits}, group_size={config.group_size}, "
            f"tiling={config.tiling_mode}, method={config.method}"
        )
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
            compression_ratio = (
                self.bf16_memory / quant_memory if quant_memory > 0 else 0
            )
            print(
                f"  Quantized memory: {quant_memory:.1f} MB (compression: {compression_ratio:.2f}x)"
            )

            # Evaluate on prompts - collect ALL metrics
            prompt_metrics_list = []

            for idx, prompt in enumerate(prompts):
                bf16_response, bf16_topk_steps = bf16_outputs[prompt]

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

                # Decode quantized response
                quant_response = self.tokenizer.decode(
                    outputs.sequences[0][input_len:], skip_special_tokens=True
                )

                # Collect top-k tokens WITH SCORES
                quant_topk_steps = []
                for score in outputs.scores:
                    positions, scores = get_top_tokens_with_scores(score[0], k=top_k)
                    quant_topk_steps.append((positions, scores))

                # Compute per-step metrics
                min_len = min(len(bf16_topk_steps), len(quant_topk_steps))
                step_metrics = []

                if min_len > 0:
                    for i in range(min_len):
                        bf16_positions, bf16_scores = bf16_topk_steps[i]
                        quant_positions, quant_scores = quant_topk_steps[i]

                        step_m = compute_step_metrics(
                            bf16_positions,
                            bf16_scores,
                            quant_positions,
                            quant_scores,
                        )
                        step_metrics.append(step_m)

                # Compute manifold zones from aggregated attention
                # We use the middle step as representative (or all steps averaged)
                baseline_zone = "unknown"
                candidate_zone = "unknown"

                if min_len > 0:
                    # Aggregate all positions/scores across steps for zone classification
                    # Use mid-point step for zone classification
                    mid_step = min_len // 2
                    bf16_positions, bf16_scores = bf16_topk_steps[mid_step]
                    quant_positions, quant_scores = quant_topk_steps[mid_step]

                    # Current position is approximately input_len + step
                    current_pos = input_len + mid_step

                    baseline_zone = compute_zone_from_topk(
                        bf16_positions, bf16_scores, current_pos
                    )
                    candidate_zone = compute_zone_from_topk(
                        quant_positions, quant_scores, current_pos
                    )

                zone_drift = baseline_zone != candidate_zone

                # Aggregate step metrics
                if step_metrics:
                    jaccards = [m.jaccard for m in step_metrics]
                    weighted_jaccards = [m.weighted_jaccard for m in step_metrics]
                    spearmans = [m.spearman for m in step_metrics]
                    kendalls = [m.kendall for m in step_metrics]
                    mass_retaineds = [m.mass_retained for m in step_metrics]
                    kl_divs = [m.kl_divergence for m in step_metrics]

                    pm = PromptMetrics(
                        prompt_id=f"prompt_{idx:03d}",
                        prompt_text=prompt[:100],
                        jaccard=float(np.mean(jaccards)),
                        weighted_jaccard=float(np.mean(weighted_jaccards)),
                        spearman=float(np.mean(spearmans)),
                        kendall=float(np.mean(kendalls)),
                        mass_retained=float(np.mean(mass_retaineds)),
                        kl_divergence=float(np.mean(kl_divs)),
                        output_match=(bf16_response.strip() == quant_response.strip()),
                        baseline_zone=baseline_zone,
                        candidate_zone=candidate_zone,
                        zone_drift=zone_drift,
                        per_step_jaccard=jaccards,
                    )
                else:
                    pm = PromptMetrics(
                        prompt_id=f"prompt_{idx:03d}",
                        prompt_text=prompt[:100],
                        jaccard=0.0,
                        weighted_jaccard=0.0,
                        spearman=0.0,
                        kendall=0.0,
                        mass_retained=0.0,
                        kl_divergence=0.0,
                        output_match=False,
                        baseline_zone=baseline_zone,
                        candidate_zone=candidate_zone,
                        zone_drift=zone_drift,
                    )

                prompt_metrics_list.append(pm)
                drift_marker = " [DRIFT]" if zone_drift else ""
                print(
                    f"  Prompt '{prompt[:30]}...': J={pm.jaccard:.3f} wJ={pm.weighted_jaccard:.3f} "
                    f"ρ={pm.spearman:.3f} zone={baseline_zone}->{candidate_zone}{drift_marker}"
                )

            # Cleanup
            del model_to_quant
            clear_gpu_memory()

            duration = time.time() - start_time

            # Aggregate across all prompts
            all_jaccards = [pm.jaccard for pm in prompt_metrics_list]
            all_weighted = [pm.weighted_jaccard for pm in prompt_metrics_list]
            all_spearman = [pm.spearman for pm in prompt_metrics_list]
            all_kendall = [pm.kendall for pm in prompt_metrics_list]
            all_mass = [pm.mass_retained for pm in prompt_metrics_list]
            all_kl = [pm.kl_divergence for pm in prompt_metrics_list]
            all_matches = [pm.output_match for pm in prompt_metrics_list]

            # Compute zone drift rate and transition matrix
            drift_count = sum(1 for pm in prompt_metrics_list if pm.zone_drift)
            zone_drift_rate = (
                drift_count / len(prompt_metrics_list) if prompt_metrics_list else 0.0
            )

            # Build zone transition matrix
            zone_transition_matrix: Dict[str, Dict[str, int]] = defaultdict(
                lambda: defaultdict(int)
            )
            for pm in prompt_metrics_list:
                zone_transition_matrix[pm.baseline_zone][pm.candidate_zone] += 1
            # Convert to regular dict for serialization
            zone_transition_dict = {
                k: dict(v) for k, v in zone_transition_matrix.items()
            }

            return ConfigResult(
                config=config,
                mean_jaccard=float(np.mean(all_jaccards)),
                std_jaccard=float(np.std(all_jaccards)),
                min_jaccard=float(np.min(all_jaccards)),
                max_jaccard=float(np.max(all_jaccards)),
                mean_weighted_jaccard=float(np.mean(all_weighted)),
                mean_spearman=float(np.mean(all_spearman)),
                mean_kendall=float(np.mean(all_kendall)),
                mean_mass_retained=float(np.mean(all_mass)),
                mean_kl_divergence=float(np.mean(all_kl)),
                output_match_rate=(
                    sum(all_matches) / len(all_matches) if all_matches else 0.0
                ),
                zone_drift_rate=zone_drift_rate,
                zone_transition_matrix=zone_transition_dict,
                compression_ratio=compression_ratio,
                bf16_memory_mb=self.bf16_memory,
                quant_memory_mb=quant_memory,
                per_prompt_jaccard=all_jaccards,
                per_prompt_metrics=prompt_metrics_list,
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
        method_results = [
            r for r in results if r.config.method == method and r.error is None
        ]
        if not method_results:
            continue

        # Create heatmap data for each tiling mode
        for tiling in TILING_OPTIONS:
            tiling_results = [
                r for r in method_results if r.config.tiling_mode == tiling
            ]
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
                fmt=".2f",
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                cbar_kws={"label": "Jaccard Similarity"},
            )
            axes[0].set_xlabel("Group Size")
            axes[0].set_ylabel("Bits")
            axes[0].set_title(
                f"{method.upper()} {tiling} - Jaccard Similarity\n(Higher = Better Quality)"
            )

            # Compression heatmap
            sns.heatmap(
                compression_matrix,
                ax=axes[1],
                xticklabels=[f"g{g}" for g in group_list],
                yticklabels=[f"{n}b" for n in nbits_list],
                annot=True,
                fmt=".2f",
                cmap="Blues",
                cbar_kws={"label": "Compression Ratio"},
            )
            axes[1].set_xlabel("Group Size")
            axes[1].set_ylabel("Bits")
            axes[1].set_title(
                f"{method.upper()} {tiling} - Compression Ratio\n(Higher = Smaller Model)"
            )

            plt.tight_layout()

            filename = output_dir / f"heatmap_{method}_{tiling}.png"
            plt.savefig(filename, dpi=150, bbox_inches="tight")
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

    for r in sorted(
        valid_results, key=lambda x: (-x.config.nbits, x.config.group_size)
    ):
        label = f"{r.config.method[:1].upper()}-{r.config.nbits}b-g{r.config.group_size}-{r.config.tiling_mode}"
        configs.append(label)
        jaccards.append(r.mean_jaccard)
        compressions.append(r.compression_ratio)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        jaccards,
        width,
        label="Jaccard Similarity",
        color="green",
        alpha=0.7,
    )

    # Secondary axis for compression
    ax2 = ax.twinx()
    bars2 = ax2.bar(
        x + width / 2,
        compressions,
        width,
        label="Compression Ratio",
        color="blue",
        alpha=0.7,
    )

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Jaccard Similarity", color="green")
    ax2.set_ylabel("Compression Ratio", color="blue")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax2.set_ylim(0, max(compressions) * 1.2)

    # Add quality threshold lines
    ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="Good (0.8)")
    ax.axhline(
        y=0.6, color="orange", linestyle="--", alpha=0.5, label="Acceptable (0.6)"
    )

    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("SINQ Quantization: Quality vs Compression Trade-off")
    plt.tight_layout()

    filename = output_dir / "summary_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def create_pareto_plot(results: List[ConfigResult], output_dir: Path):
    """Create Pareto frontier plot (quality vs compression)."""

    valid_results = [r for r in results if r.error is None]
    if not valid_results:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color by method
    colors = {"sinq": "blue", "asinq": "red"}
    markers = {"1D": "o", "2D": "s"}

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
            label=(
                label
                if label
                not in [t.get_text() for t in ax.get_legend_handles_labels()[1]]
                else ""
            ),
        )

        # Annotate with config
        ax.annotate(
            f"{r.config.nbits}b-g{r.config.group_size}",
            (r.compression_ratio, r.mean_jaccard),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xlabel("Compression Ratio (higher = smaller model)")
    ax.set_ylabel("Jaccard Similarity (higher = better quality)")
    ax.set_title("SINQ Quantization Pareto Frontier")

    # Add quality threshold lines
    ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.3, label="Good Quality")
    ax.axhline(y=0.6, color="orange", linestyle="--", alpha=0.3, label="Acceptable")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3, label="Poor")

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = output_dir / "pareto_frontier.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="SINQ Comprehensive Evaluation")
    parser.add_argument(
        "--model", "-m", default="Qwen/Qwen3-1.7B", help="Model to evaluate"
    )
    parser.add_argument(
        "--output", "-o", default="sinq_eval_results", help="Output directory"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode with fewer prompts"
    )
    parser.add_argument("--nbits", type=int, nargs="+", help="Specific nbits to test")
    parser.add_argument(
        "--methods", nargs="+", choices=["sinq", "asinq"], help="Specific methods"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=64, help="Max tokens to generate"
    )
    parser.add_argument(
        "--schema-v1",
        action="store_true",
        help="Export in Quantization Comparison Schema v1 format",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

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
                    configs.append(
                        QuantConfig(
                            nbits=nbits,
                            tiling_mode=tiling,
                            group_size=group_size,
                            method=method,
                        )
                    )

    # Create experiment manifest for reproducibility
    manifest = None
    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if HAS_MANIFEST:
        manifest = ExperimentManifest.create(
            run_type=RunType.EVALUATION,
            run_id=run_id,
            model_id=args.model,
            capture_git=True,
            capture_hardware=True,
        )
        manifest.set_decoding(
            max_tokens=args.max_tokens,
            temperature=0.0,  # Greedy for reproducibility
            seed=args.seed,
        )
        manifest.set_parameters(
            {
                "quick_mode": args.quick,
                "prompt_count": len(prompts),
                "config_count": len(configs),
                "nbits_tested": nbits_to_test,
                "methods_tested": methods_to_test,
                "tiling_modes": TILING_OPTIONS,
                "group_sizes": GROUP_SIZE_OPTIONS,
            }
        )
        print(f"Manifest created for run: {run_id}")

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
        result = evaluator.evaluate_config(
            config, prompts, bf16_outputs, args.max_tokens
        )
        results.append(result)

        # Save intermediate results
        if (i + 1) % 5 == 0:
            save_results(results, output_dir)

    total_time = time.time() - start_time

    # Save final results
    save_results(results, output_dir)

    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    create_heatmaps(results, output_dir)
    create_pareto_plot(results, output_dir)

    # Print summary
    print_summary(results, total_time)

    # Export Schema v1 if requested
    if args.schema_v1:
        export_all_schema_v1(results, args.model, prompts, output_dir)

    # Finalize and save experiment manifest
    if HAS_MANIFEST and manifest:
        # Compute aggregate statistics across all configs
        mean_jaccards = [r.mean_jaccard for r in results if r.mean_jaccard is not None]
        best_config = max(
            results, key=lambda r: r.mean_jaccard if r.mean_jaccard else 0
        )

        manifest.set_statistics(
            {
                "total_configs_tested": len(configs),
                "successful_configs": len(
                    [r for r in results if r.mean_jaccard is not None]
                ),
                "total_duration_seconds": total_time,
                "overall_mean_jaccard": (
                    np.mean(mean_jaccards) if mean_jaccards else None
                ),
                "overall_std_jaccard": np.std(mean_jaccards) if mean_jaccards else None,
                "best_config": (
                    {
                        "name": best_config.config.name,
                        "mean_jaccard": best_config.mean_jaccard,
                        "zone_drift_rate": best_config.zone_drift_rate,
                        "compression_ratio": best_config.compression_ratio,
                    }
                    if best_config.mean_jaccard
                    else None
                ),
            }
        )

        # Add artifact references
        manifest.add_artifact("results_json", str(output_dir / "results.json"))
        manifest.add_artifact(
            "heatmap_jaccard", str(output_dir / "heatmap_jaccard.png")
        )
        manifest.add_artifact(
            "heatmap_zone_drift", str(output_dir / "heatmap_zone_drift.png")
        )
        manifest.add_artifact(
            "pareto_plot", str(output_dir / "pareto_compression_vs_quality.png")
        )

        # Finalize and save
        manifest.finalize()
        manifest_path = output_dir / "manifest.json"
        manifest.save(str(manifest_path))
        print(f"\nManifest saved to: {manifest_path}")

    return 0


def save_results(results: List[ConfigResult], output_dir: Path):
    """Save results to JSON with all metrics."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "config": asdict(r.config),
                # Jaccard metrics
                "mean_jaccard": r.mean_jaccard,
                "std_jaccard": r.std_jaccard,
                "min_jaccard": r.min_jaccard,
                "max_jaccard": r.max_jaccard,
                # Additional metrics
                "mean_weighted_jaccard": r.mean_weighted_jaccard,
                "mean_spearman": r.mean_spearman,
                "mean_kendall": r.mean_kendall,
                "mean_mass_retained": r.mean_mass_retained,
                "mean_kl_divergence": r.mean_kl_divergence,
                "output_match_rate": r.output_match_rate,
                # Manifold analysis
                "zone_drift_rate": r.zone_drift_rate,
                "zone_transition_matrix": r.zone_transition_matrix,
                # Compression
                "compression_ratio": r.compression_ratio,
                "bf16_memory_mb": r.bf16_memory_mb,
                "quant_memory_mb": r.quant_memory_mb,
                # Status
                "error": r.error,
                "duration_seconds": r.duration_seconds,
                # Per-prompt data
                "per_prompt_jaccard": r.per_prompt_jaccard,
                "per_prompt_metrics": [
                    {
                        "prompt_id": pm.prompt_id,
                        "prompt_text": pm.prompt_text,
                        "jaccard": pm.jaccard,
                        "weighted_jaccard": pm.weighted_jaccard,
                        "spearman": pm.spearman,
                        "kendall": pm.kendall,
                        "mass_retained": pm.mass_retained,
                        "kl_divergence": pm.kl_divergence,
                        "output_match": pm.output_match,
                        "baseline_zone": pm.baseline_zone,
                        "candidate_zone": pm.candidate_zone,
                        "zone_drift": pm.zone_drift,
                        "per_step_jaccard": pm.per_step_jaccard,
                    }
                    for pm in r.per_prompt_metrics
                ],
            }
            for r in results
        ],
    }

    filename = output_dir / "results.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def print_summary(results: List[ConfigResult], total_time: float):
    """Print evaluation summary with all metrics."""
    print("\n" + "=" * 90)
    print("EVALUATION SUMMARY")
    print("=" * 90)

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

        print("\n" + "-" * 90)
        print("TOP 10 BY QUALITY (Jaccard Similarity)")
        print("-" * 90)
        header = f"{'Config':<30} {'Jaccard':>8} {'wJacc':>7} {'Spear':>7} {'Mass':>6} {'KL':>7} {'Match':>6} {'Compr':>7}"
        print(header)
        print("-" * 90)
        for r in sorted_by_quality[:10]:
            print(
                f"{r.config.name:<30} {r.mean_jaccard:>8.3f} {r.mean_weighted_jaccard:>7.3f} "
                f"{r.mean_spearman:>7.3f} {r.mean_mass_retained:>6.3f} {r.mean_kl_divergence:>7.3f} "
                f"{r.output_match_rate:>6.1%} {r.compression_ratio:>6.2f}x"
            )

        # Sort by compression
        sorted_by_compression = sorted(valid, key=lambda x: -x.compression_ratio)

        print("\n" + "-" * 90)
        print("TOP 10 BY COMPRESSION")
        print("-" * 90)
        print(header)
        print("-" * 90)
        for r in sorted_by_compression[:10]:
            print(
                f"{r.config.name:<30} {r.mean_jaccard:>8.3f} {r.mean_weighted_jaccard:>7.3f} "
                f"{r.mean_spearman:>7.3f} {r.mean_mass_retained:>6.3f} {r.mean_kl_divergence:>7.3f} "
                f"{r.output_match_rate:>6.1%} {r.compression_ratio:>6.2f}x"
            )

        # Quality tier distribution
        excellent = [r for r in valid if r.mean_jaccard >= 0.8]
        good = [r for r in valid if 0.6 <= r.mean_jaccard < 0.8]
        acceptable = [r for r in valid if 0.4 <= r.mean_jaccard < 0.6]
        degraded = [r for r in valid if 0.2 <= r.mean_jaccard < 0.4]
        failed = [r for r in valid if r.mean_jaccard < 0.2]

        print("\n" + "-" * 90)
        print("QUALITY TIER DISTRIBUTION")
        print("-" * 90)
        print(f"  Excellent (≥80%): {len(excellent):>3} configs")
        print(f"  Good (≥60%):      {len(good):>3} configs")
        print(f"  Acceptable (≥40%): {len(acceptable):>3} configs")
        print(f"  Degraded (≥20%):  {len(degraded):>3} configs")
        print(f"  Failed (<20%):    {len(failed):>3} configs")

        # Manifold zone drift analysis
        print("\n" + "-" * 90)
        print("MANIFOLD ZONE DRIFT ANALYSIS")
        print("-" * 90)
        drift_rates = [r.zone_drift_rate for r in valid]
        avg_drift = np.mean(drift_rates) if drift_rates else 0.0
        min_drift = np.min(drift_rates) if drift_rates else 0.0
        max_drift = np.max(drift_rates) if drift_rates else 0.0
        no_drift_configs = [r for r in valid if r.zone_drift_rate == 0.0]
        high_drift_configs = [r for r in valid if r.zone_drift_rate > 0.5]

        print(f"  Average zone drift rate: {avg_drift:.1%}")
        print(f"  Min drift:               {min_drift:.1%}")
        print(f"  Max drift:               {max_drift:.1%}")
        print(f"  Configs with no drift:   {len(no_drift_configs)}")
        print(f"  Configs with >50% drift: {len(high_drift_configs)}")

        # Top 5 by lowest zone drift (stable manifold behavior)
        sorted_by_drift = sorted(valid, key=lambda x: x.zone_drift_rate)
        if sorted_by_drift:
            print("\n  Top 5 most stable (lowest zone drift):")
            for r in sorted_by_drift[:5]:
                print(
                    f"    {r.config.name:<30} drift={r.zone_drift_rate:.1%} jaccard={r.mean_jaccard:.3f}"
                )

        # Best trade-off (highest Jaccard with compression > 2x)
        good_compression = [r for r in valid if r.compression_ratio >= 2.0]
        if good_compression:
            best_tradeoff = max(good_compression, key=lambda x: x.mean_jaccard)
            print(f"\n{'='*90}")
            print(f"RECOMMENDED CONFIG (Best quality with >2x compression):")
            print(f"  Config:          {best_tradeoff.config.name}")
            print(f"  Jaccard:         {best_tradeoff.mean_jaccard:.3f}")
            print(f"  Weighted Jaccard: {best_tradeoff.mean_weighted_jaccard:.3f}")
            print(f"  Spearman rho:    {best_tradeoff.mean_spearman:.3f}")
            print(f"  Mass Retained:   {best_tradeoff.mean_mass_retained:.3f}")
            print(f"  KL Divergence:   {best_tradeoff.mean_kl_divergence:.3f}")
            print(f"  Output Match:    {best_tradeoff.output_match_rate:.1%}")
            print(f"  Zone Drift:      {best_tradeoff.zone_drift_rate:.1%}")
            print(f"  Compression:     {best_tradeoff.compression_ratio:.2f}x")
            print(f"  Memory:          {best_tradeoff.quant_memory_mb:.0f} MB")
            print(f"{'='*90}")


def classify_quality_tier(mean_jaccard: float) -> str:
    """Classify quality tier based on Jaccard similarity."""
    if mean_jaccard >= 0.8:
        return "excellent"
    if mean_jaccard >= 0.6:
        return "good"
    if mean_jaccard >= 0.4:
        return "acceptable"
    if mean_jaccard >= 0.2:
        return "degraded"
    return "failed"


def export_schema_v1(
    result: ConfigResult,
    model_name: str,
    prompts: List[str],
    output_dir: Path,
):
    """
    Export a single configuration result in Quantization Comparison Schema v1 format.

    Each config gets its own comparison report against the BF16 baseline.
    """
    comparison_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    # Compute summary statistics
    jaccards = (
        result.per_prompt_jaccard
        if result.per_prompt_jaccard
        else [result.mean_jaccard]
    )
    jaccards_sorted = sorted(jaccards)

    def percentile(arr, p):
        idx = int(len(arr) * p)
        idx = min(idx, len(arr) - 1)
        return arr[idx]

    # Build per-prompt results
    per_prompt_results = []
    for pm in result.per_prompt_metrics:
        per_prompt_results.append(
            {
                "prompt_id": pm.prompt_id,
                "prompt_text": pm.prompt_text,
                "prompt_pack": "custom",
                "jaccard": pm.jaccard,
                "weighted_jaccard": pm.weighted_jaccard,
                "rank_correlation": {
                    "spearman": pm.spearman,
                    "kendall": pm.kendall,
                },
                "mass_retained": pm.mass_retained,
                "kl_divergence": pm.kl_divergence,
                "output_match": pm.output_match,
                "manifold_zone": {
                    "baseline": pm.baseline_zone,
                    "candidate": pm.candidate_zone,
                    "drift": pm.zone_drift,
                },
                "per_step_jaccard": pm.per_step_jaccard,
            }
        )

    report = {
        "schema_version": "1.0.0",
        "comparison_id": comparison_id,
        "timestamp": timestamp,
        "baseline": {
            "model_id": model_name,
            "dtype": "bf16",
            "memory_mb": result.bf16_memory_mb,
        },
        "candidate": {
            "model_id": model_name,
            "dtype": f"int{result.config.nbits}" if result.config.nbits <= 8 else "fp8",
            "quantization": {
                "method": result.config.method,
                "nbits": result.config.nbits,
                "group_size": result.config.group_size,
                "tiling_mode": result.config.tiling_mode,
            },
            "memory_mb": result.quant_memory_mb,
        },
        "evaluation": {
            "prompts": {
                "source": "inline",
                "count": len(prompts),
            },
            "decoding": {
                "max_tokens": 64,
                "temperature": 0.0,
            },
            "attention_capture": {
                "top_k": 10,
                "aggregation": "mean",
            },
        },
        "results": {
            "summary": {
                "jaccard": {
                    "mean": result.mean_jaccard,
                    "std": result.std_jaccard,
                    "min": result.min_jaccard,
                    "max": result.max_jaccard,
                    "median": percentile(jaccards_sorted, 0.5),
                    "p5": percentile(jaccards_sorted, 0.05),
                    "p95": percentile(jaccards_sorted, 0.95),
                },
                "weighted_jaccard": {
                    "mean": result.mean_weighted_jaccard,
                },
                "rank_correlation": {
                    "spearman_mean": result.mean_spearman,
                    "kendall_mean": result.mean_kendall,
                },
                "mass_retained": {
                    "mean": result.mean_mass_retained,
                },
                "kl_divergence": {
                    "mean": result.mean_kl_divergence,
                },
                "output_agreement": {
                    "exact_match_rate": result.output_match_rate,
                },
                "manifold_analysis": {
                    "zone_drift_rate": result.zone_drift_rate,
                    "zone_transition_matrix": result.zone_transition_matrix,
                    "zones": MANIFOLD_ZONES,
                },
                "compression_ratio": result.compression_ratio,
                "quality_tier": classify_quality_tier(result.mean_jaccard),
            },
            "per_prompt": per_prompt_results,
        },
        "metadata": {
            "purpose": "SINQ quantization evaluation",
            "tags": ["sinq", result.config.method, f"{result.config.nbits}bit"],
        },
    }

    # Save to file
    safe_name = result.config.name.replace("/", "_")
    filename = output_dir / f"schema_v1_{safe_name}.json"
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)

    return filename


def export_all_schema_v1(
    results: List[ConfigResult],
    model_name: str,
    prompts: List[str],
    output_dir: Path,
):
    """Export all results in Schema v1 format."""
    print("\n" + "=" * 60)
    print("Exporting Schema v1 reports...")
    print("=" * 60)

    schema_dir = output_dir / "schema_v1"
    schema_dir.mkdir(exist_ok=True)

    exported = []
    for r in results:
        if r.error is None:
            filename = export_schema_v1(r, model_name, prompts, schema_dir)
            exported.append(filename)
            print(f"  Exported: {filename.name}")

    # Create index file
    index = {
        "schema_version": "1.0.0",
        "generated": datetime.now().isoformat(),
        "model": model_name,
        "total_configs": len(results),
        "successful_configs": len(exported),
        "reports": [str(f.name) for f in exported],
    }

    index_file = schema_dir / "index.json"
    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nExported {len(exported)} Schema v1 reports to {schema_dir}/")
    return exported


if __name__ == "__main__":
    exit(main())
