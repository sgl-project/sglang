"""
Quantization Comparison Schema v1 - Python Implementation

Provides:
- Schema validation
- Builder classes for creating comparison reports
- Metrics computation helpers
- Quality tier classification
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# Import manifest module for comprehensive experiment tracking
try:
    from .manifest import get_git_info, get_gpu_info

    HAS_MANIFEST = True
except ImportError:
    HAS_MANIFEST = False


# =============================================================================
# Enums
# =============================================================================


class QuantizationMethod(str, Enum):
    NONE = "none"
    SINQ = "sinq"
    ASINQ = "asinq"
    AWQ = "awq"
    GPTQ = "gptq"
    SQUEEZELLM = "squeezellm"
    FP8 = "fp8"
    MARLIN = "marlin"


class TilingMode(str, Enum):
    ONE_D = "1D"
    TWO_D = "2D"


class ManifoldZone(str, Enum):
    SYNTAX_FLOOR = "syntax_floor"
    SEMANTIC_BRIDGE = "semantic_bridge"
    STRUCTURE_RIPPLE = "structure_ripple"
    LONG_RANGE = "long_range"
    DIFFUSE = "diffuse"
    UNKNOWN = "unknown"


class QualityTier(str, Enum):
    EXCELLENT = "excellent"  # Jaccard >= 0.8
    GOOD = "good"  # Jaccard >= 0.6
    ACCEPTABLE = "acceptable"  # Jaccard >= 0.4
    DEGRADED = "degraded"  # Jaccard >= 0.2
    FAILED = "failed"  # Jaccard < 0.2


class PromptPack(str, Enum):
    JSON_REPAIR = "json_repair"
    COREFERENCE = "coreference"
    COUNTING_TABLES = "counting_tables"
    CODE_EDITING = "code_editing"
    REASONING = "reasoning"
    ADVERSARIAL = "adversarial"
    NATURAL = "natural"
    CUSTOM = "custom"


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_jaccard(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
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
    Compute weighted Jaccard using attention scores as weights.

    For shared positions, weight = min(score1, score2)
    For union positions, weight = max of available scores
    """
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

    spearman, _ = stats.spearmanr(ranks1, ranks2)
    kendall, _ = stats.kendalltau(ranks1, ranks2)

    return (
        float(spearman) if not np.isnan(spearman) else 0.0,
        float(kendall) if not np.isnan(kendall) else 0.0,
    )


def compute_mass_retained(
    baseline_positions: List[int],
    baseline_scores: List[float],
    candidate_positions: List[int],
) -> float:
    """
    Compute fraction of baseline attention mass that falls within candidate top-k.

    This measures "how much of what the baseline model attended to is preserved".
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
    Compute KL divergence between truncated attention distributions.

    Both lists are normalized to sum to 1, with epsilon smoothing.
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

    # Renormalize after padding
    p = p / p.sum()
    q = q / q.sum()

    return float(stats.entropy(p, q))


def compute_bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    statistic: str = "mean",
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        values: List of observed values
        confidence: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap resamples
        statistic: Statistic to compute ("mean", "median", "std")

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if not values or len(values) < 2:
        return (float("nan"), float("nan"))

    values_arr = np.array(values)
    n = len(values_arr)

    # Compute bootstrap distribution
    bootstrap_stats = []
    rng = np.random.default_rng()

    for _ in range(n_bootstrap):
        # Resample with replacement
        resample = values_arr[rng.integers(0, n, size=n)]

        if statistic == "mean":
            bootstrap_stats.append(np.mean(resample))
        elif statistic == "median":
            bootstrap_stats.append(np.median(resample))
        elif statistic == "std":
            bootstrap_stats.append(np.std(resample))
        else:
            bootstrap_stats.append(np.mean(resample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Compute percentile-based confidence interval
    alpha = 1 - confidence
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return (lower, upper)


def classify_quality_tier(mean_jaccard: float) -> QualityTier:
    """Classify quality tier based on mean Jaccard similarity."""
    if mean_jaccard >= 0.8:
        return QualityTier.EXCELLENT
    elif mean_jaccard >= 0.6:
        return QualityTier.GOOD
    elif mean_jaccard >= 0.4:
        return QualityTier.ACCEPTABLE
    elif mean_jaccard >= 0.2:
        return QualityTier.DEGRADED
    else:
        return QualityTier.FAILED


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class QuantizationConfig:
    method: QuantizationMethod = QuantizationMethod.NONE
    nbits: Optional[int] = None
    group_size: Optional[int] = None
    tiling_mode: Optional[TilingMode] = None
    symmetric: bool = True
    calibration_dataset: Optional[str] = None
    calibration_samples: Optional[int] = None
    kernel_path: Optional[str] = None

    def to_dict(self) -> Dict:
        d = {}
        if self.method != QuantizationMethod.NONE:
            d["method"] = self.method.value
        if self.nbits is not None:
            d["nbits"] = self.nbits
        if self.group_size is not None:
            d["group_size"] = self.group_size
        if self.tiling_mode is not None:
            d["tiling_mode"] = self.tiling_mode.value
        d["symmetric"] = self.symmetric
        if self.calibration_dataset:
            d["calibration"] = {
                "dataset": self.calibration_dataset,
                "num_samples": self.calibration_samples,
            }
        if self.kernel_path:
            d["kernel_path"] = self.kernel_path
        return d


@dataclass
class ModelConfig:
    model_id: str
    dtype: str
    revision: Optional[str] = None
    quantization: Optional[QuantizationConfig] = None
    memory_mb: Optional[float] = None

    def to_dict(self) -> Dict:
        d = {
            "model_id": self.model_id,
            "dtype": self.dtype,
        }
        if self.revision:
            d["revision"] = self.revision
        if self.quantization:
            d["quantization"] = self.quantization.to_dict()
        if self.memory_mb is not None:
            d["memory_mb"] = self.memory_mb
        return d


@dataclass
class DecodingConfig:
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    seed: Optional[int] = None
    repetition_penalty: float = 1.0

    def to_dict(self) -> Dict:
        d = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.seed is not None:
            d["seed"] = self.seed
        return d


@dataclass
class AttentionCaptureConfig:
    top_k: int = 64
    layers: Optional[List[int]] = None
    heads: Optional[List[int]] = None
    aggregation: str = "mean"
    sink_filter_enabled: bool = True
    sink_filter_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3])

    def to_dict(self) -> Dict:
        d = {
            "top_k": self.top_k,
            "aggregation": self.aggregation,
            "sink_filter": {
                "enabled": self.sink_filter_enabled,
                "indices": self.sink_filter_indices,
            },
        }
        if self.layers:
            d["layers"] = self.layers
        if self.heads:
            d["heads"] = self.heads
        return d


@dataclass
class PromptResult:
    prompt_id: str
    jaccard: float
    prompt_text: Optional[str] = None
    prompt_pack: Optional[PromptPack] = None
    weighted_jaccard: Optional[float] = None
    spearman: Optional[float] = None
    kendall: Optional[float] = None
    mass_retained: Optional[float] = None
    kl_divergence: Optional[float] = None
    output_match: Optional[bool] = None
    edit_distance: Optional[int] = None
    baseline_zone: Optional[ManifoldZone] = None
    candidate_zone: Optional[ManifoldZone] = None
    per_step_jaccard: Optional[List[float]] = None
    per_layer_jaccard: Optional[List[float]] = None

    @property
    def zone_drift(self) -> bool:
        if self.baseline_zone and self.candidate_zone:
            return self.baseline_zone != self.candidate_zone
        return False

    def to_dict(self) -> Dict:
        d = {
            "prompt_id": self.prompt_id,
            "jaccard": self.jaccard,
        }
        if self.prompt_text:
            d["prompt_text"] = self.prompt_text[:100]
        if self.prompt_pack:
            d["prompt_pack"] = self.prompt_pack.value
        if self.weighted_jaccard is not None:
            d["weighted_jaccard"] = self.weighted_jaccard
        if self.spearman is not None or self.kendall is not None:
            d["rank_correlation"] = {}
            if self.spearman is not None:
                d["rank_correlation"]["spearman"] = self.spearman
            if self.kendall is not None:
                d["rank_correlation"]["kendall"] = self.kendall
        if self.mass_retained is not None:
            d["mass_retained"] = self.mass_retained
        if self.kl_divergence is not None:
            d["kl_divergence"] = self.kl_divergence
        if self.output_match is not None:
            d["output_match"] = self.output_match
        if self.edit_distance is not None:
            d["edit_distance"] = self.edit_distance
        if self.baseline_zone:
            d["baseline_zone"] = self.baseline_zone.value
        if self.candidate_zone:
            d["candidate_zone"] = self.candidate_zone.value
        if self.baseline_zone and self.candidate_zone:
            d["zone_drift"] = self.zone_drift
        if self.per_step_jaccard:
            d["per_step_jaccard"] = self.per_step_jaccard
        if self.per_layer_jaccard:
            d["per_layer_jaccard"] = self.per_layer_jaccard
        return d


@dataclass
class HardwareConfig:
    gpu_type: Optional[str] = None
    gpu_count: int = 1
    gpu_memory_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None
    attention_backend: Optional[str] = None
    tensor_parallel: int = 1
    pipeline_parallel: int = 1

    def to_dict(self) -> Dict:
        d = {}
        if self.gpu_type:
            d["gpu_type"] = self.gpu_type
        d["gpu_count"] = self.gpu_count
        if self.gpu_memory_gb:
            d["gpu_memory_gb"] = self.gpu_memory_gb
        if self.cuda_version:
            d["cuda_version"] = self.cuda_version
        if self.driver_version:
            d["driver_version"] = self.driver_version
        if self.attention_backend:
            d["attention_backend"] = self.attention_backend
        d["tensor_parallel"] = self.tensor_parallel
        d["pipeline_parallel"] = self.pipeline_parallel
        return d

    @classmethod
    def auto_detect(
        cls,
        attention_backend: Optional[str] = None,
        tensor_parallel: int = 1,
        pipeline_parallel: int = 1,
    ) -> "HardwareConfig":
        """
        Auto-detect hardware configuration from the system.

        Args:
            attention_backend: Override attention backend name
            tensor_parallel: Tensor parallel degree
            pipeline_parallel: Pipeline parallel degree

        Returns:
            HardwareConfig with detected values
        """
        config = cls(
            attention_backend=attention_backend,
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
        )

        if HAS_MANIFEST:
            gpus = get_gpu_info()
            if gpus:
                config.gpu_count = len(gpus)
                config.gpu_type = gpus[0].get("name")
                memory_mb = gpus[0].get("memory_mb", 0)
                if memory_mb:
                    config.gpu_memory_gb = round(memory_mb / 1024, 1)
                config.cuda_version = gpus[0].get("cuda_version")
                config.driver_version = gpus[0].get("driver_version")

        return config


# =============================================================================
# Comparison Report Builder
# =============================================================================


class ComparisonReportBuilder:
    """
    Builder for creating Quantization Comparison Schema v1 reports.

    Usage:
        builder = ComparisonReportBuilder()
        builder.set_baseline(ModelConfig(...))
        builder.set_candidate(ModelConfig(...))
        builder.set_decoding(DecodingConfig(...))
        builder.set_attention_capture(AttentionCaptureConfig(...))

        for prompt, baseline_data, candidate_data in evaluation_results:
            builder.add_prompt_result(prompt, baseline_data, candidate_data)

        report = builder.build()
        builder.save("comparison_report.json")
    """

    SCHEMA_VERSION = "1.0.0"

    def __init__(self, auto_detect_hardware: bool = False, capture_git: bool = False):
        """
        Initialize the comparison report builder.

        Args:
            auto_detect_hardware: If True, auto-detect GPU/hardware info
            capture_git: If True, capture git SHA/branch info
        """
        import time

        self.comparison_id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        self._start_time = time.time()
        self.baseline: Optional[ModelConfig] = None
        self.candidate: Optional[ModelConfig] = None
        self.decoding: Optional[DecodingConfig] = None
        self.attention_capture: Optional[AttentionCaptureConfig] = None
        self.hardware: Optional[HardwareConfig] = None
        self.prompt_source: str = "inline"
        self.prompt_path: Optional[str] = None
        self.harness_packs: Optional[List[PromptPack]] = None
        self.harness_duration: Optional[int] = None
        self.harness_seed: Optional[int] = None
        self.fingerprint_enabled: bool = True
        self.fingerprint_location: str = "server"
        self.privacy_masking_enabled: bool = False
        self.prompt_results: List[PromptResult] = []
        self.performance_baseline_tps: Optional[float] = None
        self.performance_candidate_tps: Optional[float] = None
        self.performance_baseline_p50: Optional[float] = None
        self.performance_candidate_p50: Optional[float] = None
        self.performance_baseline_p99: Optional[float] = None
        self.performance_candidate_p99: Optional[float] = None
        self.metadata: Dict[str, Any] = {}
        self.git_info: Optional[Dict[str, Any]] = None

        if auto_detect_hardware:
            self.hardware = HardwareConfig.auto_detect()

        if capture_git and HAS_MANIFEST:
            self.git_info = {"server": get_git_info()}

    def set_baseline(self, config: ModelConfig) -> "ComparisonReportBuilder":
        self.baseline = config
        return self

    def set_candidate(self, config: ModelConfig) -> "ComparisonReportBuilder":
        self.candidate = config
        return self

    def set_decoding(self, config: DecodingConfig) -> "ComparisonReportBuilder":
        self.decoding = config
        return self

    def set_attention_capture(
        self, config: AttentionCaptureConfig
    ) -> "ComparisonReportBuilder":
        self.attention_capture = config
        return self

    def set_hardware(self, config: HardwareConfig) -> "ComparisonReportBuilder":
        self.hardware = config
        return self

    def set_prompt_source(
        self,
        source: str,
        path: Optional[str] = None,
        packs: Optional[List[PromptPack]] = None,
        duration: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> "ComparisonReportBuilder":
        self.prompt_source = source
        self.prompt_path = path
        self.harness_packs = packs
        self.harness_duration = duration
        self.harness_seed = seed
        return self

    def set_performance(
        self,
        baseline_tps: Optional[float] = None,
        candidate_tps: Optional[float] = None,
        baseline_p50: Optional[float] = None,
        candidate_p50: Optional[float] = None,
        baseline_p99: Optional[float] = None,
        candidate_p99: Optional[float] = None,
    ) -> "ComparisonReportBuilder":
        self.performance_baseline_tps = baseline_tps
        self.performance_candidate_tps = candidate_tps
        self.performance_baseline_p50 = baseline_p50
        self.performance_candidate_p50 = candidate_p50
        self.performance_baseline_p99 = baseline_p99
        self.performance_candidate_p99 = candidate_p99
        return self

    def set_metadata(self, **kwargs) -> "ComparisonReportBuilder":
        self.metadata.update(kwargs)
        return self

    def add_prompt_result(self, result: PromptResult) -> "ComparisonReportBuilder":
        self.prompt_results.append(result)
        return self

    def add_prompt_from_attention_data(
        self,
        prompt_id: str,
        baseline_positions: List[int],
        baseline_scores: List[float],
        candidate_positions: List[int],
        candidate_scores: List[float],
        prompt_text: Optional[str] = None,
        prompt_pack: Optional[PromptPack] = None,
        baseline_output: Optional[str] = None,
        candidate_output: Optional[str] = None,
        baseline_zone: Optional[ManifoldZone] = None,
        candidate_zone: Optional[ManifoldZone] = None,
    ) -> "ComparisonReportBuilder":
        """
        Compute all metrics from raw attention data and add result.
        """
        # Apply sink filter if configured
        if self.attention_capture and self.attention_capture.sink_filter_enabled:
            sink_indices = set(self.attention_capture.sink_filter_indices)
            filtered_baseline = [
                (p, s)
                for p, s in zip(baseline_positions, baseline_scores)
                if p not in sink_indices
            ]
            filtered_candidate = [
                (p, s)
                for p, s in zip(candidate_positions, candidate_scores)
                if p not in sink_indices
            ]
            if filtered_baseline:
                baseline_positions, baseline_scores = zip(*filtered_baseline)
                baseline_positions = list(baseline_positions)
                baseline_scores = list(baseline_scores)
            if filtered_candidate:
                candidate_positions, candidate_scores = zip(*filtered_candidate)
                candidate_positions = list(candidate_positions)
                candidate_scores = list(candidate_scores)

        # Compute metrics
        jaccard = compute_jaccard(set(baseline_positions), set(candidate_positions))
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

        # Output comparison
        output_match = None
        edit_distance = None
        if baseline_output is not None and candidate_output is not None:
            output_match = baseline_output == candidate_output
            # Simple edit distance (Levenshtein would be better but this is quick)
            edit_distance = abs(len(baseline_output) - len(candidate_output))

        result = PromptResult(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            prompt_pack=prompt_pack,
            jaccard=jaccard,
            weighted_jaccard=weighted_jaccard,
            spearman=spearman,
            kendall=kendall,
            mass_retained=mass_retained,
            kl_divergence=kl_div,
            output_match=output_match,
            edit_distance=edit_distance,
            baseline_zone=baseline_zone,
            candidate_zone=candidate_zone,
        )

        self.prompt_results.append(result)
        return self

    def _compute_summary_metrics(self, compute_ci: bool = True) -> Dict:
        """
        Compute aggregate metrics from all prompt results.

        Args:
            compute_ci: Whether to compute bootstrap confidence intervals
                        (can be slow for large sample sizes)
        """
        if not self.prompt_results:
            return {}

        jaccards = [r.jaccard for r in self.prompt_results]

        # Compute bootstrap 95% confidence interval for mean Jaccard
        ci_lower, ci_upper = (float("nan"), float("nan"))
        if compute_ci and len(jaccards) >= 3:
            ci_lower, ci_upper = compute_bootstrap_ci(jaccards, confidence=0.95)

        summary = {
            "jaccard": {
                "mean": float(np.mean(jaccards)),
                "std": float(np.std(jaccards)),
                "min": float(np.min(jaccards)),
                "max": float(np.max(jaccards)),
                "median": float(np.median(jaccards)),
                "p5": float(np.percentile(jaccards, 5)),
                "p95": float(np.percentile(jaccards, 95)),
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "sample_size": len(jaccards),
            }
        }

        # Weighted Jaccard
        wj = [
            r.weighted_jaccard
            for r in self.prompt_results
            if r.weighted_jaccard is not None
        ]
        if wj:
            summary["weighted_jaccard"] = {
                "mean": float(np.mean(wj)),
                "std": float(np.std(wj)),
            }

        # Rank correlation
        spearman = [r.spearman for r in self.prompt_results if r.spearman is not None]
        kendall = [r.kendall for r in self.prompt_results if r.kendall is not None]
        if spearman or kendall:
            summary["rank_correlation"] = {}
            if spearman:
                summary["rank_correlation"]["spearman_mean"] = float(np.mean(spearman))
            if kendall:
                summary["rank_correlation"]["kendall_mean"] = float(np.mean(kendall))

        # Mass retained
        mr = [
            r.mass_retained for r in self.prompt_results if r.mass_retained is not None
        ]
        if mr:
            summary["mass_retained"] = {
                "mean": float(np.mean(mr)),
                "std": float(np.std(mr)),
            }

        # KL divergence
        kl = [
            r.kl_divergence for r in self.prompt_results if r.kl_divergence is not None
        ]
        if kl:
            summary["kl_divergence"] = {
                "mean": float(np.mean(kl)),
                "std": float(np.std(kl)),
            }

        # Output agreement
        matches = [
            r.output_match for r in self.prompt_results if r.output_match is not None
        ]
        if matches:
            summary["output_agreement"] = {
                "exact_match_rate": sum(matches) / len(matches),
            }

        # Compression ratio
        if self.baseline and self.candidate:
            if self.baseline.memory_mb and self.candidate.memory_mb:
                summary["compression_ratio"] = (
                    self.baseline.memory_mb / self.candidate.memory_mb
                )

        # Quality tier
        summary["quality_tier"] = classify_quality_tier(
            summary["jaccard"]["mean"]
        ).value

        return summary

    def _compute_manifold_analysis(self, compute_ci: bool = True) -> Optional[Dict]:
        """
        Compute manifold/zone drift analysis.

        Args:
            compute_ci: Whether to compute bootstrap confidence intervals
        """
        results_with_zones = [
            r for r in self.prompt_results if r.baseline_zone and r.candidate_zone
        ]

        if not results_with_zones:
            return None

        # Zone drift rate with confidence interval
        # Convert bool to float for bootstrap
        drifts = [float(r.zone_drift) for r in results_with_zones]
        zone_drift_rate = sum(drifts) / len(drifts)

        # Compute CI for drift rate (treating as proportion)
        drift_ci_lower, drift_ci_upper = (float("nan"), float("nan"))
        if compute_ci and len(drifts) >= 3:
            drift_ci_lower, drift_ci_upper = compute_bootstrap_ci(
                drifts, confidence=0.95
            )

        # Zone transition matrix
        transition_matrix: Dict[str, Dict[str, int]] = {}
        for r in results_with_zones:
            bz = r.baseline_zone.value
            cz = r.candidate_zone.value
            if bz not in transition_matrix:
                transition_matrix[bz] = {}
            transition_matrix[bz][cz] = transition_matrix[bz].get(cz, 0) + 1

        # Drift by pack
        drift_by_pack: Dict[str, float] = {}
        for pack in PromptPack:
            pack_results = [r for r in results_with_zones if r.prompt_pack == pack]
            if pack_results:
                drift_by_pack[pack.value] = sum(
                    float(r.zone_drift) for r in pack_results
                ) / len(pack_results)

        return {
            "zone_drift_rate": zone_drift_rate,
            "zone_drift_ci_95_lower": drift_ci_lower,
            "zone_drift_ci_95_upper": drift_ci_upper,
            "zone_transition_matrix": transition_matrix,
            "drift_by_pack": drift_by_pack if drift_by_pack else None,
            "sample_size": len(results_with_zones),
        }

    def _build_performance_metrics(self) -> Optional[Dict]:
        """Build performance metrics section."""
        if not any(
            [
                self.performance_baseline_tps,
                self.performance_candidate_tps,
            ]
        ):
            return None

        perf = {}
        if self.performance_baseline_tps:
            perf["baseline_throughput_tps"] = self.performance_baseline_tps
        if self.performance_candidate_tps:
            perf["candidate_throughput_tps"] = self.performance_candidate_tps
        if self.performance_baseline_tps and self.performance_candidate_tps:
            perf["throughput_ratio"] = (
                self.performance_candidate_tps / self.performance_baseline_tps
            )
        if self.performance_baseline_p50:
            perf["baseline_latency_p50_ms"] = self.performance_baseline_p50
        if self.performance_candidate_p50:
            perf["candidate_latency_p50_ms"] = self.performance_candidate_p50
        if self.performance_baseline_p99:
            perf["baseline_latency_p99_ms"] = self.performance_baseline_p99
        if self.performance_candidate_p99:
            perf["candidate_latency_p99_ms"] = self.performance_candidate_p99

        return perf

    def build(self) -> Dict:
        """Build the complete comparison report."""
        if not self.baseline or not self.candidate:
            raise ValueError("Both baseline and candidate must be set")
        if not self.decoding:
            raise ValueError("Decoding config must be set")
        if not self.attention_capture:
            self.attention_capture = AttentionCaptureConfig()

        # Build evaluation config
        evaluation = {
            "prompts": {
                "source": self.prompt_source,
                "count": len(self.prompt_results),
            },
            "decoding": self.decoding.to_dict(),
            "attention_capture": self.attention_capture.to_dict(),
            "fingerprint": {
                "enabled": self.fingerprint_enabled,
                "compute_location": self.fingerprint_location,
                "include_manifold_zone": True,
            },
            "privacy_masking": {
                "enabled": self.privacy_masking_enabled,
            },
        }

        if self.prompt_path:
            evaluation["prompts"]["path"] = self.prompt_path
        if self.harness_packs:
            evaluation["prompts"]["harness_config"] = {
                "packs": [p.value for p in self.harness_packs],
            }
            if self.harness_duration:
                evaluation["prompts"]["harness_config"][
                    "duration_minutes"
                ] = self.harness_duration

        # Build results
        results = {
            "summary": self._compute_summary_metrics(),
            "per_prompt": [r.to_dict() for r in self.prompt_results],
        }

        manifold = self._compute_manifold_analysis()
        if manifold:
            results["manifold_analysis"] = manifold

        perf = self._build_performance_metrics()
        if perf:
            results["performance"] = perf

        # Build complete report
        report = {
            "schema_version": self.SCHEMA_VERSION,
            "comparison_id": self.comparison_id,
            "timestamp": self.timestamp,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "evaluation": evaluation,
            "results": results,
        }

        if self.hardware:
            report["hardware"] = self.hardware.to_dict()

        if self.metadata:
            report["metadata"] = self.metadata

        return report

    def save(self, path: str) -> None:
        """Build and save report to JSON file."""
        report = self.build()
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    @classmethod
    def load(cls, path: str) -> Dict:
        """Load a comparison report from JSON file."""
        with open(path) as f:
            return json.load(f)


# =============================================================================
# Schema Validation
# =============================================================================


def validate_report(report: Dict) -> Tuple[bool, List[str]]:
    """
    Validate a comparison report against the schema.

    Returns (is_valid, list_of_errors)
    """
    errors = []

    # Required top-level fields
    required = [
        "schema_version",
        "comparison_id",
        "timestamp",
        "baseline",
        "candidate",
        "evaluation",
        "results",
    ]
    for field in required:
        if field not in report:
            errors.append(f"Missing required field: {field}")

    if errors:
        return False, errors

    # Schema version check
    if report["schema_version"] != "1.0.0":
        errors.append(f"Unsupported schema version: {report['schema_version']}")

    # Baseline/candidate validation
    for config_name in ["baseline", "candidate"]:
        config = report.get(config_name, {})
        if "model_id" not in config:
            errors.append(f"{config_name} missing model_id")
        if "dtype" not in config:
            errors.append(f"{config_name} missing dtype")

    # Evaluation validation
    eval_config = report.get("evaluation", {})
    if "decoding" not in eval_config:
        errors.append("evaluation missing decoding config")
    elif "max_tokens" not in eval_config.get("decoding", {}):
        errors.append("decoding missing max_tokens")

    if "attention_capture" not in eval_config:
        errors.append("evaluation missing attention_capture config")
    elif "top_k" not in eval_config.get("attention_capture", {}):
        errors.append("attention_capture missing top_k")

    # Results validation
    results = report.get("results", {})
    if "summary" not in results:
        errors.append("results missing summary")
    elif "jaccard" not in results.get("summary", {}):
        errors.append("summary missing jaccard metrics")

    if "per_prompt" not in results:
        errors.append("results missing per_prompt array")
    else:
        for i, pr in enumerate(results.get("per_prompt", [])):
            if "prompt_id" not in pr:
                errors.append(f"per_prompt[{i}] missing prompt_id")
            if "jaccard" not in pr:
                errors.append(f"per_prompt[{i}] missing jaccard")

    return len(errors) == 0, errors


# =============================================================================
# Blessed Config Registry
# =============================================================================


@dataclass
class BlessedQuantConfig:
    """A quantization configuration that has been validated and approved for routing."""

    config_id: str
    model_id: str
    quantization: QuantizationConfig
    quality_tier: QualityTier
    mean_jaccard: float
    zone_drift_rate: float
    compression_ratio: float
    approved_for_packs: List[PromptPack]
    excluded_packs: List[PromptPack]
    comparison_id: str  # Reference to the comparison report
    approved_timestamp: str
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "config_id": self.config_id,
            "model_id": self.model_id,
            "quantization": self.quantization.to_dict(),
            "quality_tier": self.quality_tier.value,
            "mean_jaccard": self.mean_jaccard,
            "zone_drift_rate": self.zone_drift_rate,
            "compression_ratio": self.compression_ratio,
            "approved_for_packs": [p.value for p in self.approved_for_packs],
            "excluded_packs": [p.value for p in self.excluded_packs],
            "comparison_id": self.comparison_id,
            "approved_timestamp": self.approved_timestamp,
            "notes": self.notes,
        }


class BlessedConfigRegistry:
    """
    Registry of approved quantization configurations for routing.

    The router can query this to decide which quant config to use
    for a given model and prompt type.
    """

    def __init__(self, path: Optional[str] = None):
        self.configs: Dict[str, BlessedQuantConfig] = {}
        self.path = path
        if path and Path(path).exists():
            self.load(path)

    def add(self, config: BlessedQuantConfig) -> None:
        self.configs[config.config_id] = config

    def get(self, config_id: str) -> Optional[BlessedQuantConfig]:
        return self.configs.get(config_id)

    def get_for_model(self, model_id: str) -> List[BlessedQuantConfig]:
        return [c for c in self.configs.values() if c.model_id == model_id]

    def get_best_for_pack(
        self,
        model_id: str,
        pack: PromptPack,
        min_quality: QualityTier = QualityTier.ACCEPTABLE,
    ) -> Optional[BlessedQuantConfig]:
        """
        Get the best quantization config for a model and prompt pack.

        "Best" = highest compression ratio that meets quality threshold
        and is approved for the given pack.
        """
        candidates = [
            c
            for c in self.get_for_model(model_id)
            if pack in c.approved_for_packs
            and pack not in c.excluded_packs
            and self._quality_meets_threshold(c.quality_tier, min_quality)
        ]

        if not candidates:
            return None

        # Sort by compression ratio (descending) then by quality (descending)
        candidates.sort(
            key=lambda c: (c.compression_ratio, c.mean_jaccard),
            reverse=True,
        )
        return candidates[0]

    def _quality_meets_threshold(
        self, tier: QualityTier, min_tier: QualityTier
    ) -> bool:
        order = [
            QualityTier.FAILED,
            QualityTier.DEGRADED,
            QualityTier.ACCEPTABLE,
            QualityTier.GOOD,
            QualityTier.EXCELLENT,
        ]
        return order.index(tier) >= order.index(min_tier)

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.path
        if not path:
            raise ValueError("No path specified")
        with open(path, "w") as f:
            json.dump(
                {"configs": [c.to_dict() for c in self.configs.values()]},
                f,
                indent=2,
            )

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        # Note: Full deserialization would require more work
        # This is a simplified version


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Build a comparison report
    builder = ComparisonReportBuilder()

    # Set model configs
    builder.set_baseline(
        ModelConfig(
            model_id="Qwen/Qwen3-1.7B",
            dtype="bf16",
            memory_mb=3282.0,
        )
    )

    builder.set_candidate(
        ModelConfig(
            model_id="Qwen/Qwen3-1.7B",
            dtype="int4",
            quantization=QuantizationConfig(
                method=QuantizationMethod.ASINQ,
                nbits=5,
                group_size=64,
                tiling_mode=TilingMode.TWO_D,
                calibration_dataset="wikitext",
                calibration_samples=128,
            ),
            memory_mb=2100.0,
        )
    )

    # Set evaluation config
    builder.set_decoding(
        DecodingConfig(
            max_tokens=50,
            temperature=0.0,
            seed=42,
        )
    )

    builder.set_attention_capture(
        AttentionCaptureConfig(
            top_k=64,
            sink_filter_enabled=True,
            sink_filter_indices=[0, 1, 2, 3],
        )
    )

    builder.set_hardware(
        HardwareConfig(
            gpu_type="NVIDIA RTX 4090",
            gpu_count=1,
            attention_backend="flashinfer",
        )
    )

    # Add example prompt results
    builder.add_prompt_result(
        PromptResult(
            prompt_id="test-001",
            prompt_text="What is the capital of France?",
            prompt_pack=PromptPack.NATURAL,
            jaccard=0.75,
            weighted_jaccard=0.72,
            spearman=0.68,
            kendall=0.62,
            mass_retained=0.81,
            kl_divergence=0.15,
            output_match=True,
            baseline_zone=ManifoldZone.SEMANTIC_BRIDGE,
            candidate_zone=ManifoldZone.SEMANTIC_BRIDGE,
        )
    )

    builder.add_prompt_result(
        PromptResult(
            prompt_id="test-002",
            prompt_text="Fix this JSON: {name: 'test'}",
            prompt_pack=PromptPack.JSON_REPAIR,
            jaccard=0.82,
            weighted_jaccard=0.79,
            spearman=0.75,
            kendall=0.71,
            mass_retained=0.88,
            kl_divergence=0.08,
            output_match=True,
            baseline_zone=ManifoldZone.SYNTAX_FLOOR,
            candidate_zone=ManifoldZone.SYNTAX_FLOOR,
        )
    )

    # Build and print
    report = builder.build()
    print(json.dumps(report, indent=2))

    # Validate
    is_valid, errors = validate_report(report)
    print(f"\nValidation: {'PASSED' if is_valid else 'FAILED'}")
    if errors:
        for e in errors:
            print(f"  - {e}")
