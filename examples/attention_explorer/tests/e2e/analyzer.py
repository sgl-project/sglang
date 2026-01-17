"""
Analysis module for attention exploration data.

Provides pattern detection, manifold classification, and
insight extraction for understanding attention behavior.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

from collector import CollectionRun, Fingerprint, MoERouting, TraceData
from scenarios import ExpectedManifold

logger = logging.getLogger(__name__)


# PCA loadings for fingerprint interpretation (from types.ts)
PCA_LOADINGS = [
    {
        "name": "PC1: Local vs Long-Range",
        "variance_explained": 0.35,
        "loadings": [
            0.6,
            0.1,
            -0.6,
            -0.2,
            0.4,
            0.3,
            0.1,
            0.0,
            -0.1,
            -0.2,
            -0.3,
            -0.3,
            -0.2,
            -0.15,
            -0.1,
            -0.05,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    },
    {
        "name": "PC2: Focused vs Diffuse",
        "variance_explained": 0.22,
        "loadings": [
            0.2,
            0.2,
            0.2,
            -0.7,
            0.3,
            0.2,
            0.1,
            0.0,
            -0.1,
            -0.15,
            -0.2,
            -0.2,
            -0.15,
            -0.1,
            -0.05,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    },
    {
        "name": "PC3: Semantic Bridge",
        "variance_explained": 0.15,
        "loadings": [
            -0.1,
            0.6,
            -0.1,
            0.1,
            -0.2,
            0.1,
            0.3,
            0.3,
            0.3,
            0.2,
            0.1,
            0.0,
            -0.1,
            -0.2,
            -0.25,
            -0.2,
            -0.1,
            0.0,
            0.0,
            0.0,
        ],
    },
    {
        "name": "PC4: Structure Ripple",
        "variance_explained": 0.10,
        "loadings": [
            0.0,
            0.0,
            0.0,
            0.3,
            0.4,
            -0.3,
            0.3,
            -0.3,
            0.3,
            -0.25,
            0.2,
            -0.2,
            0.15,
            -0.15,
            0.1,
            -0.1,
            0.05,
            -0.05,
            0.0,
            0.0,
        ],
    },
]


@dataclass
class ManifoldClassification:
    """Classification of a trace into manifold zones."""

    primary_zone: str
    confidence: float
    zone_scores: Dict[str, float]
    pca_scores: List[float]
    interpretation: str


@dataclass
class PatternDetection:
    """Detected pattern in attention data."""

    pattern_type: str
    strength: float
    evidence: List[str]
    step_indices: List[int]


@dataclass
class MoEInsight:
    """Insight from MoE routing analysis."""

    insight_type: str
    description: str
    expert_stats: Dict[int, float]
    layer_stats: Dict[int, Dict[str, float]]


@dataclass
class TraceAnalysis:
    """Complete analysis of a single trace."""

    scenario_name: str
    expected_manifold: str
    actual_classification: ManifoldClassification
    match_expected: bool
    patterns_detected: List[PatternDetection]
    moe_insights: List[MoEInsight]
    summary_stats: Dict[str, Any]
    recommendations: List[str]


@dataclass
class RunAnalysis:
    """Complete analysis of a collection run."""

    run_id: str
    trace_analyses: List[TraceAnalysis]
    manifold_accuracy: Dict[str, float]  # Expected -> accuracy rate
    pattern_summary: Dict[str, int]  # Pattern type -> count
    moe_global_insights: List[MoEInsight]
    key_findings: List[str]
    sinq_recommendations: List[str]


class AttentionAnalyzer:
    """
    Analyzes attention patterns and extracts insights.
    """

    def __init__(self):
        self.pca_loadings = PCA_LOADINGS

    def analyze_trace(self, trace: TraceData) -> TraceAnalysis:
        """Perform complete analysis of a single trace."""
        # Classify into manifold zone
        classification = self._classify_manifold(trace)

        # Detect patterns
        patterns = self._detect_patterns(trace)

        # Analyze MoE routing
        moe_insights = self._analyze_moe(trace) if trace.moe_routing else []

        # Compute summary statistics
        stats = self._compute_summary_stats(trace)

        # Check if classification matches expected
        match_expected = classification.primary_zone == trace.expected_manifold

        # Generate recommendations
        recommendations = self._generate_recommendations(
            trace, classification, patterns, moe_insights
        )

        return TraceAnalysis(
            scenario_name=trace.scenario_name,
            expected_manifold=trace.expected_manifold,
            actual_classification=classification,
            match_expected=match_expected,
            patterns_detected=patterns,
            moe_insights=moe_insights,
            summary_stats=stats,
            recommendations=recommendations,
        )

    def analyze_run(self, run: CollectionRun) -> RunAnalysis:
        """Perform complete analysis of a collection run."""
        trace_analyses = [self.analyze_trace(t) for t in run.traces]

        # Compute manifold accuracy
        manifold_accuracy = self._compute_manifold_accuracy(trace_analyses)

        # Aggregate patterns
        pattern_summary = defaultdict(int)
        for analysis in trace_analyses:
            for pattern in analysis.patterns_detected:
                pattern_summary[pattern.pattern_type] += 1

        # Global MoE insights
        all_moe_data = []
        for trace in run.traces:
            all_moe_data.extend(trace.moe_routing)
        moe_global = self._analyze_moe_global(all_moe_data) if all_moe_data else []

        # Key findings
        key_findings = self._extract_key_findings(
            trace_analyses, manifold_accuracy, pattern_summary
        )

        # Sinq recommendations
        sinq_recommendations = self._generate_sinq_recommendations(
            trace_analyses, manifold_accuracy, pattern_summary
        )

        return RunAnalysis(
            run_id=run.run_id,
            trace_analyses=trace_analyses,
            manifold_accuracy=dict(manifold_accuracy),
            pattern_summary=dict(pattern_summary),
            moe_global_insights=moe_global,
            key_findings=key_findings,
            sinq_recommendations=sinq_recommendations,
        )

    def _classify_manifold(self, trace: TraceData) -> ManifoldClassification:
        """Classify trace into manifold zone based on attention patterns."""
        if not trace.fingerprints:
            return ManifoldClassification(
                primary_zone="unknown",
                confidence=0.0,
                zone_scores={},
                pca_scores=[],
                interpretation="No fingerprint data available",
            )

        # Aggregate fingerprints
        avg_fp = self._average_fingerprints(trace.fingerprints)

        # Project to PCA space
        pca_scores = self._project_to_pca(avg_fp)

        # Score each zone
        zone_scores = {
            ExpectedManifold.SYNTAX_FLOOR.value: self._score_syntax_floor(
                avg_fp, pca_scores
            ),
            ExpectedManifold.SEMANTIC_BRIDGE.value: self._score_semantic_bridge(
                avg_fp, pca_scores
            ),
            ExpectedManifold.LONG_RANGE.value: self._score_long_range(
                avg_fp, pca_scores
            ),
            ExpectedManifold.STRUCTURE_RIPPLE.value: self._score_structure_ripple(
                avg_fp, pca_scores
            ),
            ExpectedManifold.DIFFUSE.value: self._score_diffuse(avg_fp, pca_scores),
        }

        # Find primary zone
        primary_zone = max(zone_scores, key=zone_scores.get)
        confidence = zone_scores[primary_zone]

        # Generate interpretation
        interpretation = self._interpret_classification(
            primary_zone, avg_fp, pca_scores, zone_scores
        )

        return ManifoldClassification(
            primary_zone=primary_zone,
            confidence=confidence,
            zone_scores=zone_scores,
            pca_scores=pca_scores,
            interpretation=interpretation,
        )

    def _average_fingerprints(self, fps: List[Fingerprint]) -> Fingerprint:
        """Compute average fingerprint."""
        n = len(fps)
        if n == 0:
            return Fingerprint(0, 0, 0, 0, [0] * 16)

        return Fingerprint(
            local_mass=sum(fp.local_mass for fp in fps) / n,
            mid_mass=sum(fp.mid_mass for fp in fps) / n,
            long_mass=sum(fp.long_mass for fp in fps) / n,
            entropy=sum(fp.entropy for fp in fps) / n,
            histogram=[sum(fp.histogram[i] for fp in fps) / n for i in range(16)],
        )

    def _project_to_pca(self, fp: Fingerprint) -> List[float]:
        """Project fingerprint to PCA space."""
        vec = fp.to_vector()
        scores = []

        for pc in self.pca_loadings:
            loadings = pc["loadings"]
            score = sum(v * l for v, l in zip(vec, loadings))
            scores.append(score)

        return scores

    def _score_syntax_floor(self, fp: Fingerprint, pca: List[float]) -> float:
        """Score for syntax floor zone (local attention)."""
        # High local mass, low entropy, positive PC1
        score = fp.local_mass * 0.4
        score += (1 - fp.entropy) * 0.3
        score += max(0, pca[0]) * 0.3 if pca else 0
        return min(1.0, score)

    def _score_semantic_bridge(self, fp: Fingerprint, pca: List[float]) -> float:
        """Score for semantic bridge zone (mid-range attention)."""
        # High mid mass, moderate entropy, positive PC3
        score = fp.mid_mass * 0.4
        score += (1 - abs(fp.entropy - 0.5)) * 0.3  # Moderate entropy
        score += max(0, pca[2]) * 0.3 if len(pca) > 2 else 0
        return min(1.0, score)

    def _score_long_range(self, fp: Fingerprint, pca: List[float]) -> float:
        """Score for long range zone (document-level attention)."""
        # High long mass, negative PC1
        score = fp.long_mass * 0.5
        score += max(0, -pca[0]) * 0.3 if pca else 0
        score += (1 - fp.local_mass) * 0.2
        return min(1.0, score)

    def _score_structure_ripple(self, fp: Fingerprint, pca: List[float]) -> float:
        """Score for structure ripple zone (periodic patterns)."""
        # Alternating histogram pattern, positive PC4
        histogram_variance = self._histogram_periodicity(fp.histogram)
        score = histogram_variance * 0.4
        score += max(0, pca[3]) * 0.3 if len(pca) > 3 else 0
        score += (1 - fp.entropy) * 0.3  # Structured = low entropy
        return min(1.0, score)

    def _score_diffuse(self, fp: Fingerprint, pca: List[float]) -> float:
        """Score for diffuse zone (exploratory attention)."""
        # High entropy, spread across ranges
        score = fp.entropy * 0.5
        score += max(0, -pca[1]) * 0.3 if len(pca) > 1 else 0  # Negative PC2 = diffuse
        # Balanced distribution
        balance = 1 - abs(fp.local_mass - fp.mid_mass) - abs(fp.mid_mass - fp.long_mass)
        score += max(0, balance) * 0.2
        return min(1.0, score)

    def _histogram_periodicity(self, histogram: List[float]) -> float:
        """Measure periodicity in histogram (alternating pattern)."""
        if len(histogram) < 4:
            return 0.0

        # Check for alternating high/low pattern
        diffs = [histogram[i + 1] - histogram[i] for i in range(len(histogram) - 1)]
        sign_changes = sum(
            1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0
        )

        return sign_changes / (len(diffs) - 1) if len(diffs) > 1 else 0

    def _interpret_classification(
        self, zone: str, fp: Fingerprint, pca: List[float], scores: Dict[str, float]
    ) -> str:
        """Generate human-readable interpretation."""
        interpretations = {
            ExpectedManifold.SYNTAX_FLOOR.value: (
                f"Local syntax processing (local_mass={fp.local_mass:.2f}, "
                f"entropy={fp.entropy:.2f}). Attention focuses on nearby tokens "
                f"for grammar and completion."
            ),
            ExpectedManifold.SEMANTIC_BRIDGE.value: (
                f"Mid-range semantic retrieval (mid_mass={fp.mid_mass:.2f}). "
                f"Attention bridges across sentences for meaning and context."
            ),
            ExpectedManifold.LONG_RANGE.value: (
                f"Long-range information retrieval (long_mass={fp.long_mass:.2f}). "
                f"Attention reaches back to document-level context."
            ),
            ExpectedManifold.STRUCTURE_RIPPLE.value: (
                f"Structural pattern generation (entropy={fp.entropy:.2f}). "
                f"Periodic attention for code, lists, or formatted output."
            ),
            ExpectedManifold.DIFFUSE.value: (
                f"Exploratory attention (entropy={fp.entropy:.2f}). "
                f"Broad, uncertain attention for creative or abstract content."
            ),
        }

        base = interpretations.get(zone, "Unknown zone")

        # Add confidence context
        if scores[zone] < 0.3:
            base += " (Low confidence - borderline classification)"
        elif scores[zone] > 0.7:
            base += " (High confidence)"

        return base

    def _detect_patterns(self, trace: TraceData) -> List[PatternDetection]:
        """Detect specific patterns in attention data."""
        patterns = []

        if not trace.attention_steps:
            return patterns

        # 1. Sink token pattern (high attention to token 0)
        sink_steps = []
        for i, step in enumerate(trace.attention_steps):
            first_token = next((t for t in step.top_k_tokens if t.position == 0), None)
            if first_token and first_token.weight > 0.3:
                sink_steps.append(i)

        if len(sink_steps) > len(trace.attention_steps) * 0.3:
            patterns.append(
                PatternDetection(
                    pattern_type="sink_token",
                    strength=len(sink_steps) / len(trace.attention_steps),
                    evidence=[f"High attention to token 0 in {len(sink_steps)} steps"],
                    step_indices=sink_steps,
                )
            )

        # 2. Local copy pattern (attending to recent similar tokens)
        # 3. Induction heads (looking for pattern repetition)
        # 4. Previous token pattern

        prev_token_steps = []
        for i, step in enumerate(trace.attention_steps):
            prev_tokens = [t for t in step.top_k_tokens if t.offset == 1]
            if prev_tokens and sum(t.weight for t in prev_tokens) > 0.4:
                prev_token_steps.append(i)

        if len(prev_token_steps) > len(trace.attention_steps) * 0.5:
            patterns.append(
                PatternDetection(
                    pattern_type="previous_token",
                    strength=len(prev_token_steps) / len(trace.attention_steps),
                    evidence=[
                        f"Strong previous-token attention in {len(prev_token_steps)} steps"
                    ],
                    step_indices=prev_token_steps,
                )
            )

        # 5. Entropy transitions (sudden changes)
        if len(trace.fingerprints) > 5:
            entropies = [fp.entropy for fp in trace.fingerprints]
            transition_steps = []

            for i in range(1, len(entropies)):
                if abs(entropies[i] - entropies[i - 1]) > 0.3:
                    transition_steps.append(i)

            if transition_steps:
                patterns.append(
                    PatternDetection(
                        pattern_type="entropy_transition",
                        strength=len(transition_steps) / len(entropies),
                        evidence=[
                            f"Entropy transitions at steps: {transition_steps[:5]}"
                        ],
                        step_indices=transition_steps,
                    )
                )

        return patterns

    def _analyze_moe(self, trace: TraceData) -> List[MoEInsight]:
        """Analyze MoE routing patterns for a trace."""
        insights = []

        if not trace.moe_routing:
            return insights

        # Expert usage distribution
        expert_counts = defaultdict(int)
        expert_weights = defaultdict(float)

        for routing in trace.moe_routing:
            for eid, weight in zip(routing.expert_ids, routing.expert_weights):
                expert_counts[eid] += 1
                expert_weights[eid] += weight

        # Find dominant experts
        total = sum(expert_counts.values())
        expert_usage = {eid: count / total for eid, count in expert_counts.items()}

        dominant = [(eid, usage) for eid, usage in expert_usage.items() if usage > 0.15]

        if dominant:
            insights.append(
                MoEInsight(
                    insight_type="dominant_experts",
                    description=f"Experts {[e[0] for e in dominant]} dominate routing",
                    expert_stats=expert_usage,
                    layer_stats={},
                )
            )

        # Expert diversity
        n_experts = len(expert_counts)
        entropy = -sum(p * math.log2(p) for p in expert_usage.values() if p > 0)
        max_entropy = math.log2(n_experts) if n_experts > 1 else 1
        diversity = entropy / max_entropy if max_entropy > 0 else 0

        insights.append(
            MoEInsight(
                insight_type="expert_diversity",
                description=f"Expert diversity: {diversity:.2f} (0=concentrated, 1=uniform)",
                expert_stats={"diversity": diversity, "n_experts_used": n_experts},
                layer_stats={},
            )
        )

        return insights

    def _analyze_moe_global(self, all_moe: List[MoERouting]) -> List[MoEInsight]:
        """Global MoE analysis across all traces."""
        insights = []

        # Layer-wise analysis
        by_layer = defaultdict(list)
        for routing in all_moe:
            by_layer[routing.layer_idx].append(routing)

        layer_stats = {}
        for layer, routings in by_layer.items():
            expert_counts = defaultdict(int)
            for r in routings:
                for eid in r.expert_ids:
                    expert_counts[eid] += 1

            total = sum(expert_counts.values())
            layer_stats[layer] = {
                "n_routings": len(routings),
                "top_experts": sorted(expert_counts.items(), key=lambda x: -x[1])[:3],
                "expert_diversity": len(expert_counts) / total if total > 0 else 0,
            }

        if layer_stats:
            insights.append(
                MoEInsight(
                    insight_type="layer_analysis",
                    description="Per-layer expert usage patterns",
                    expert_stats={},
                    layer_stats=layer_stats,
                )
            )

        return insights

    def _compute_summary_stats(self, trace: TraceData) -> Dict[str, Any]:
        """Compute summary statistics for a trace."""
        stats = {
            "n_steps": len(trace.attention_steps),
            "n_fingerprints": len(trace.fingerprints),
            "completion_tokens": trace.completion_tokens,
            "tokens_per_second": trace.tokens_per_second,
        }

        if trace.fingerprints:
            avg_fp = self._average_fingerprints(trace.fingerprints)
            stats["avg_local_mass"] = avg_fp.local_mass
            stats["avg_mid_mass"] = avg_fp.mid_mass
            stats["avg_long_mass"] = avg_fp.long_mass
            stats["avg_entropy"] = avg_fp.entropy

        if trace.moe_routing:
            stats["n_moe_routings"] = len(trace.moe_routing)

        return stats

    def _generate_recommendations(
        self,
        trace: TraceData,
        classification: ManifoldClassification,
        patterns: List[PatternDetection],
        moe_insights: List[MoEInsight],
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recs = []

        # Classification mismatch
        if classification.primary_zone != trace.expected_manifold:
            recs.append(
                f"Classification mismatch: expected {trace.expected_manifold}, "
                f"got {classification.primary_zone}. Consider adjusting scenario "
                f"or refining classification thresholds."
            )

        # Low confidence
        if classification.confidence < 0.3:
            recs.append(
                f"Low classification confidence ({classification.confidence:.2f}). "
                f"Trace may be in transition between zones."
            )

        # Sink token pattern
        sink_pattern = next(
            (p for p in patterns if p.pattern_type == "sink_token"), None
        )
        if sink_pattern and sink_pattern.strength > 0.5:
            recs.append(
                "Strong sink token pattern detected. Consider investigating "
                "attention head specialization."
            )

        return recs

    def _compute_manifold_accuracy(
        self, analyses: List[TraceAnalysis]
    ) -> Dict[str, float]:
        """Compute accuracy of manifold classification vs expected."""
        by_expected = defaultdict(list)
        for analysis in analyses:
            by_expected[analysis.expected_manifold].append(analysis.match_expected)

        return {
            zone: sum(matches) / len(matches) if matches else 0
            for zone, matches in by_expected.items()
        }

    def _extract_key_findings(
        self,
        analyses: List[TraceAnalysis],
        accuracy: Dict[str, float],
        patterns: Dict[str, int],
    ) -> List[str]:
        """Extract key findings from run analysis."""
        findings = []

        # Overall accuracy
        total_match = sum(1 for a in analyses if a.match_expected)
        findings.append(
            f"Manifold classification accuracy: {total_match}/{len(analyses)} "
            f"({100 * total_match / len(analyses):.1f}%)"
        )

        # Best/worst zones
        if accuracy:
            best_zone = max(accuracy, key=accuracy.get)
            worst_zone = min(accuracy, key=accuracy.get)
            findings.append(
                f"Best classified zone: {best_zone} ({accuracy[best_zone]:.0%})"
            )
            if accuracy[worst_zone] < accuracy[best_zone]:
                findings.append(
                    f"Worst classified zone: {worst_zone} ({accuracy[worst_zone]:.0%})"
                )

        # Common patterns
        if patterns:
            common = sorted(patterns.items(), key=lambda x: -x[1])[:3]
            findings.append(
                f"Most common patterns: {', '.join(f'{p[0]}({p[1]})' for p in common)}"
            )

        return findings

    def _generate_sinq_recommendations(
        self,
        analyses: List[TraceAnalysis],
        accuracy: Dict[str, float],
        patterns: Dict[str, int],
    ) -> List[str]:
        """Generate recommendations for sinq implementation."""
        recs = []

        # Zone detection suggestions
        for zone, acc in accuracy.items():
            if acc < 0.7:
                recs.append(
                    f"Zone '{zone}' has low detection accuracy ({acc:.0%}). "
                    f"Consider additional features or refined thresholds for sinq."
                )

        # Pattern-based suggestions
        if patterns.get("sink_token", 0) > len(analyses) * 0.3:
            recs.append(
                "Sink token pattern is common. Sinq should track BOS attention "
                "as a signal for uncertainty or exploration."
            )

        if patterns.get("entropy_transition", 0) > len(analyses) * 0.2:
            recs.append(
                "Entropy transitions are frequent. Sinq should detect phase "
                "changes in attention behavior for routing decisions."
            )

        # General suggestions
        recs.append(
            "Consider implementing real-time manifold tracking in sinq for "
            "dynamic expert routing."
        )
        recs.append(
            "Fingerprint PCA projections could enable fast zone classification "
            "for sinq query routing."
        )

        return recs
