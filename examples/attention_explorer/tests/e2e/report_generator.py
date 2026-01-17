"""
Report generation for attention exploration analysis.

Generates comprehensive reports including:
- Manifold classification accuracy
- Pattern detection summaries
- MoE routing insights
- Sinq implementation recommendations
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from analyzer import RunAnalysis, TraceAnalysis
from collector import CollectionRun

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates analysis reports in multiple formats."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
        self,
        run: CollectionRun,
        analysis: RunAnalysis,
    ):
        """Generate all report formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON report (machine-readable)
        self._generate_json_report(run, analysis, timestamp)

        # Markdown report (human-readable)
        self._generate_markdown_report(run, analysis, timestamp)

        # Summary CSV (for spreadsheet analysis)
        self._generate_csv_summary(run, analysis, timestamp)

        # Sinq insights document
        self._generate_sinq_insights(analysis, timestamp)

        logger.info(f"Reports generated in {self.output_dir}")

    def _generate_json_report(
        self,
        run: CollectionRun,
        analysis: RunAnalysis,
        timestamp: str,
    ):
        """Generate full JSON report."""
        report = {
            "metadata": {
                "timestamp": timestamp,
                "run_id": run.run_id,
                "model": run.model_name,
                "server": run.server_url,
                "start_time": run.start_time,
                "end_time": run.end_time,
            },
            "summary": run.summary(),
            "analysis": {
                "manifold_accuracy": analysis.manifold_accuracy,
                "pattern_summary": analysis.pattern_summary,
                "key_findings": analysis.key_findings,
                "sinq_recommendations": analysis.sinq_recommendations,
            },
            "trace_analyses": [
                {
                    "scenario_name": ta.scenario_name,
                    "expected_manifold": ta.expected_manifold,
                    "actual_classification": {
                        "primary_zone": ta.actual_classification.primary_zone,
                        "confidence": ta.actual_classification.confidence,
                        "zone_scores": ta.actual_classification.zone_scores,
                    },
                    "match_expected": ta.match_expected,
                    "patterns": [
                        {"type": p.pattern_type, "strength": p.strength}
                        for p in ta.patterns_detected
                    ],
                    "summary_stats": ta.summary_stats,
                    "recommendations": ta.recommendations,
                }
                for ta in analysis.trace_analyses
            ],
            "errors": run.errors,
        }

        path = self.output_dir / f"report_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"JSON report: {path}")

    def _generate_markdown_report(
        self,
        run: CollectionRun,
        analysis: RunAnalysis,
        timestamp: str,
    ):
        """Generate human-readable markdown report."""
        lines = []

        # Header
        lines.append("# Attention Exploration E2E Test Report")
        lines.append("")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Run ID:** {run.run_id}")
        lines.append(f"**Model:** {run.model_name}")
        lines.append(f"**Server:** {run.server_url}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        summary = run.summary()
        lines.append(f"- **Total Scenarios:** {summary['total_traces']}")
        lines.append(f"- **Errors:** {summary['total_errors']}")
        lines.append(f"- **Avg Tokens/Second:** {summary['avg_tokens_per_second']:.1f}")
        lines.append("")

        # Key Findings
        lines.append("## Key Findings")
        lines.append("")
        for finding in analysis.key_findings:
            lines.append(f"- {finding}")
        lines.append("")

        # Manifold Classification Accuracy
        lines.append("## Manifold Classification Accuracy")
        lines.append("")
        lines.append("| Zone | Accuracy |")
        lines.append("|------|----------|")
        for zone, acc in sorted(analysis.manifold_accuracy.items()):
            status = "✓" if acc >= 0.7 else "⚠" if acc >= 0.5 else "✗"
            lines.append(f"| {zone} | {acc:.0%} {status} |")
        lines.append("")

        # Pattern Summary
        lines.append("## Detected Patterns")
        lines.append("")
        lines.append("| Pattern | Count |")
        lines.append("|---------|-------|")
        for pattern, count in sorted(
            analysis.pattern_summary.items(), key=lambda x: -x[1]
        ):
            lines.append(f"| {pattern} | {count} |")
        lines.append("")

        # Category Breakdown
        lines.append("## Results by Category")
        lines.append("")

        by_category: Dict[str, List[TraceAnalysis]] = {}
        for ta in analysis.trace_analyses:
            cat = ta.scenario_name.split("_")[0]  # Rough category extraction
            if ta.expected_manifold not in by_category:
                by_category[ta.expected_manifold] = []
            by_category[ta.expected_manifold].append(ta)

        for category, analyses in sorted(by_category.items()):
            matches = sum(1 for a in analyses if a.match_expected)
            lines.append(f"### {category}")
            lines.append(
                f"Accuracy: {matches}/{len(analyses)} ({100*matches/len(analyses):.0f}%)"
            )
            lines.append("")

            # Show mismatches
            mismatches = [a for a in analyses if not a.match_expected]
            if mismatches:
                lines.append("**Misclassified scenarios:**")
                for a in mismatches[:5]:
                    lines.append(
                        f"- {a.scenario_name}: expected {a.expected_manifold}, "
                        f"got {a.actual_classification.primary_zone} "
                        f"(confidence: {a.actual_classification.confidence:.2f})"
                    )
            lines.append("")

        # MoE Insights
        if analysis.moe_global_insights:
            lines.append("## MoE Routing Insights")
            lines.append("")
            for insight in analysis.moe_global_insights:
                lines.append(f"### {insight.insight_type}")
                lines.append(insight.description)
                lines.append("")

        # Sinq Recommendations
        lines.append("## Recommendations for Sinq Implementation")
        lines.append("")
        for i, rec in enumerate(analysis.sinq_recommendations, 1):
            lines.append(f"{i}. {rec}")
            lines.append("")

        # Sample Trace Details
        lines.append("## Sample Trace Details")
        lines.append("")

        # Show one good and one problematic trace per zone
        for zone in set(analysis.manifold_accuracy.keys()):
            zone_traces = [
                ta for ta in analysis.trace_analyses if ta.expected_manifold == zone
            ]
            if not zone_traces:
                continue

            lines.append(f"### {zone}")

            # Good example
            good = next((t for t in zone_traces if t.match_expected), None)
            if good:
                lines.append(f"**Good Example:** {good.scenario_name}")
                lines.append(
                    f"- Classification: {good.actual_classification.primary_zone}"
                )
                lines.append(
                    f"- Confidence: {good.actual_classification.confidence:.2f}"
                )
                lines.append(
                    f"- Interpretation: {good.actual_classification.interpretation}"
                )
                lines.append("")

            # Problem example
            bad = next((t for t in zone_traces if not t.match_expected), None)
            if bad:
                lines.append(f"**Misclassified Example:** {bad.scenario_name}")
                lines.append(f"- Expected: {bad.expected_manifold}")
                lines.append(f"- Got: {bad.actual_classification.primary_zone}")
                lines.append(
                    f"- Confidence: {bad.actual_classification.confidence:.2f}"
                )
                if bad.recommendations:
                    lines.append(f"- Recommendation: {bad.recommendations[0]}")
                lines.append("")

        # Footer
        lines.append("---")
        lines.append(f"*Report generated by attention-explorer E2E test suite*")

        # Write
        path = self.output_dir / f"report_{timestamp}.md"
        with open(path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Markdown report: {path}")

    def _generate_csv_summary(
        self,
        run: CollectionRun,
        analysis: RunAnalysis,
        timestamp: str,
    ):
        """Generate CSV summary for analysis."""
        lines = []

        # Header
        lines.append(
            "scenario_name,category,expected_manifold,actual_zone,confidence,"
            "match,local_mass,mid_mass,long_mass,entropy,n_steps,tokens_per_sec"
        )

        for ta in analysis.trace_analyses:
            stats = ta.summary_stats
            lines.append(
                f"{ta.scenario_name},"
                f"{ta.scenario_name.split('_')[0]},"  # Rough category
                f"{ta.expected_manifold},"
                f"{ta.actual_classification.primary_zone},"
                f"{ta.actual_classification.confidence:.3f},"
                f"{1 if ta.match_expected else 0},"
                f"{stats.get('avg_local_mass', 0):.3f},"
                f"{stats.get('avg_mid_mass', 0):.3f},"
                f"{stats.get('avg_long_mass', 0):.3f},"
                f"{stats.get('avg_entropy', 0):.3f},"
                f"{stats.get('n_steps', 0)},"
                f"{stats.get('tokens_per_second', 0):.1f}"
            )

        path = self.output_dir / f"summary_{timestamp}.csv"
        with open(path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"CSV summary: {path}")

    def _generate_sinq_insights(
        self,
        analysis: RunAnalysis,
        timestamp: str,
    ):
        """Generate sinq implementation insights document."""
        lines = []

        lines.append("# Sinq Implementation Insights")
        lines.append("")
        lines.append("*Extracted from attention exploration E2E testing*")
        lines.append("")

        # Zone Detection Insights
        lines.append("## 1. Manifold Zone Detection")
        lines.append("")
        lines.append("### Current Accuracy")
        lines.append("")

        for zone, acc in sorted(
            analysis.manifold_accuracy.items(), key=lambda x: -x[1]
        ):
            lines.append(f"- **{zone}**: {acc:.0%}")
        lines.append("")

        lines.append("### Detection Strategy Recommendations")
        lines.append("")

        # Generate zone-specific recommendations
        for zone, acc in analysis.manifold_accuracy.items():
            if acc < 0.8:
                lines.append(f"#### {zone} (needs improvement)")
                if zone == "syntax_floor":
                    lines.append("- Consider stricter local_mass threshold (>0.6)")
                    lines.append("- Add n-gram pattern detection")
                    lines.append("- Check for BPE boundary attention")
                elif zone == "semantic_bridge":
                    lines.append("- May need larger context window for mid_mass")
                    lines.append("- Consider sentence boundary detection")
                    lines.append("- Add coreference signal detection")
                elif zone == "long_range":
                    lines.append("- Increase long_mass threshold")
                    lines.append("- Add document structure awareness")
                    lines.append("- Consider positional encoding analysis")
                elif zone == "structure_ripple":
                    lines.append("- Add periodicity detection in attention pattern")
                    lines.append("- Check for code/markdown structure signals")
                    lines.append("- Consider entropy variance over time")
                elif zone == "diffuse":
                    lines.append("- Use entropy as primary signal")
                    lines.append("- Check attention spread metrics")
                    lines.append("- May indicate uncertainty/exploration")
                lines.append("")

        # Pattern Insights
        lines.append("## 2. Pattern-Based Signals")
        lines.append("")

        pattern_insights = {
            "sink_token": (
                "Sink Token Pattern",
                [
                    "Strong BOS attention often indicates uncertainty",
                    "Can signal need for exploration or clarification",
                    "Useful for detecting when model is 'searching'",
                ],
            ),
            "previous_token": (
                "Previous Token Pattern",
                [
                    "Indicates local/sequential processing",
                    "Common in syntax-heavy generation",
                    "May suggest completion rather than reasoning",
                ],
            ),
            "entropy_transition": (
                "Entropy Transition Pattern",
                [
                    "Signals phase change in generation",
                    "Can indicate topic/task shift",
                    "Useful for detecting when to switch strategies",
                ],
            ),
        }

        for pattern_type, count in analysis.pattern_summary.items():
            if pattern_type in pattern_insights:
                name, insights = pattern_insights[pattern_type]
                lines.append(f"### {name}")
                lines.append(f"Detected in {count} traces")
                lines.append("")
                for insight in insights:
                    lines.append(f"- {insight}")
                lines.append("")

        # Fingerprint Usage
        lines.append("## 3. Fingerprint for Fast Classification")
        lines.append("")
        lines.append("The 20D fingerprint vector enables fast zone classification:")
        lines.append("")
        lines.append("```")
        lines.append("fingerprint = [")
        lines.append("    local_mass,    # Attention to offset < 8")
        lines.append("    mid_mass,      # Attention to offset 8-255")
        lines.append("    long_mass,     # Attention to offset >= 256")
        lines.append("    entropy,       # Attention distribution entropy")
        lines.append("    histogram[16]  # Exponential offset bins")
        lines.append("]")
        lines.append("```")
        lines.append("")
        lines.append("### PCA-Based Classification")
        lines.append("")
        lines.append("Project fingerprint to 4 principal components for fast lookup:")
        lines.append("")
        lines.append("- **PC1 (35%)**: Local vs Long-Range")
        lines.append("- **PC2 (22%)**: Focused vs Diffuse")
        lines.append("- **PC3 (15%)**: Semantic Bridge strength")
        lines.append("- **PC4 (10%)**: Structure Ripple periodicity")
        lines.append("")

        # Implementation Suggestions
        lines.append("## 4. Sinq Implementation Suggestions")
        lines.append("")

        for i, rec in enumerate(analysis.sinq_recommendations, 1):
            lines.append(f"### {i}. {rec.split('.')[0]}")
            if "." in rec:
                lines.append(rec.split(".", 1)[1].strip())
            lines.append("")

        # Additional suggestions
        lines.append("### Additional Implementation Notes")
        lines.append("")
        lines.append(
            "1. **Real-time Tracking**: Compute fingerprint per step during generation"
        )
        lines.append(
            "2. **Zone History**: Track manifold trajectory for routing decisions"
        )
        lines.append(
            "3. **Expert Routing**: Use zone + pattern signals for MoE expert selection"
        )
        lines.append(
            "4. **Confidence Thresholds**: Implement fallback for low-confidence zones"
        )
        lines.append("5. **Caching**: Cache fingerprints for similar prompts")
        lines.append("")

        # Data Collection Recommendations
        lines.append("## 5. Future Data Collection")
        lines.append("")
        lines.append("To improve sinq accuracy, collect more data on:")
        lines.append("")

        # Find weak zones
        weak_zones = [z for z, a in analysis.manifold_accuracy.items() if a < 0.7]
        if weak_zones:
            lines.append(f"- **Weak zones**: {', '.join(weak_zones)}")
            lines.append("  - Add more diverse scenarios for these zones")
            lines.append("  - Investigate classification boundary cases")

        lines.append(
            "- **Transition scenarios**: Traces that change zones mid-generation"
        )
        lines.append("- **Edge cases**: Ambiguous prompts that could be multiple zones")
        lines.append("- **MoE correlation**: Which experts activate for which zones")
        lines.append("")

        # Write
        path = self.output_dir / f"sinq_insights_{timestamp}.md"
        with open(path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Sinq insights: {path}")
