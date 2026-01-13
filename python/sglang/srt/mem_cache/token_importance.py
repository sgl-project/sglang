"""
Token Importance Scoring for Smart Context Truncation

Provides importance scores for tokens based on attention patterns, enabling
intelligent context truncation when exceeding max context length.

Key insight: Attention patterns reveal which tokens are semantically critical
(retrieval anchors, reasoning steps) vs. reconstructible (local syntax).

Usage:
    from sglang.srt.mem_cache.token_importance import TokenImportanceScorer

    scorer = TokenImportanceScorer()
    scores = scorer.score_tokens(fingerprints, token_texts)

    # Get indices of most important tokens
    important_idx = scorer.get_top_k_indices(scores, k=1000)

    # For context truncation: keep these tokens, drop the rest
    truncated_tokens = [tokens[i] for i in important_idx]

API Endpoint (if --return-token-importance flag is set):
    POST /v1/token_importance
    {
        "text": "...",
        "max_tokens": 4096
    }
    Returns: {"importance_scores": [...], "truncation_indices": [...]}

Author: SGLang Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Zone importance (same as spectral_eviction.py)
ZONE_IMPORTANCE = {
    "semantic_bridge": 0.95,
    "long_range": 0.85,
    "structure_ripple": 0.70,
    "syntax_floor": 0.30,
    "diffuse": 0.20,
    "unknown": 0.50,
}

# Feature indices in fingerprint vector
FP_LOCAL_MASS = 0
FP_MID_MASS = 1
FP_LONG_MASS = 2
FP_ENTROPY = 3
FP_HISTOGRAM_START = 4


@dataclass
class TokenImportance:
    """Importance score for a single token."""

    index: int
    score: float  # 0.0 (disposable) to 1.0 (critical)
    zone: str  # Attention zone classification
    components: Dict[str, float]  # Score breakdown


@dataclass
class ImportanceResult:
    """Result of token importance scoring."""

    scores: List[TokenImportance]  # Per-token importance
    mean_score: float
    top_k_indices: List[int]  # Indices of most important tokens
    truncation_indices: List[int]  # Optimal truncation order (least important first)


class TokenImportanceScorer:
    """
    Scores tokens by importance for smart context truncation.

    Combines multiple signals from attention fingerprints:
    - Manifold zone: semantic bridges > syntax floor
    - Attention entropy: focused attention = important
    - Long-range mass: high = retrieval anchor
    - Position: start/end tokens often important

    Example:
        scorer = TokenImportanceScorer(
            zone_weight=0.4,
            entropy_weight=0.3,
            long_range_weight=0.2,
            position_weight=0.1
        )
        result = scorer.score_from_fingerprints(fingerprints)

        # Keep top 50% most important tokens
        keep_indices = result.top_k_indices[:len(result.scores) // 2]
    """

    def __init__(
        self,
        zone_weight: float = 0.4,
        entropy_weight: float = 0.3,
        long_range_weight: float = 0.2,
        position_weight: float = 0.1,
    ):
        """
        Args:
            zone_weight: Weight for manifold zone score
            entropy_weight: Weight for attention entropy score
            long_range_weight: Weight for long-range attention mass
            position_weight: Weight for positional importance
        """
        self.zone_weight = zone_weight
        self.entropy_weight = entropy_weight
        self.long_range_weight = long_range_weight
        self.position_weight = position_weight

        # Normalize weights
        total = zone_weight + entropy_weight + long_range_weight + position_weight
        self.zone_weight /= total
        self.entropy_weight /= total
        self.long_range_weight /= total
        self.position_weight /= total

    def score_from_fingerprints(
        self,
        fingerprints: np.ndarray,
        zones: Optional[List[str]] = None,
        token_texts: Optional[List[str]] = None,
        keep_ratio: float = 0.5,
    ) -> ImportanceResult:
        """
        Score tokens by importance using attention fingerprints.

        Args:
            fingerprints: [n_tokens, feature_dim] attention fingerprints
            zones: Optional manifold zone classifications
            token_texts: Optional token texts (for special token detection)
            keep_ratio: Fraction of tokens to include in top_k_indices

        Returns:
            ImportanceResult with per-token scores and truncation indices
        """
        n_tokens = len(fingerprints)
        if n_tokens == 0:
            return ImportanceResult(
                scores=[],
                mean_score=0.0,
                top_k_indices=[],
                truncation_indices=[],
            )

        scores = []

        for i, fp in enumerate(fingerprints):
            # Extract features from fingerprint
            local_mass = fp[FP_LOCAL_MASS] if len(fp) > FP_LOCAL_MASS else 0.5
            mid_mass = fp[FP_MID_MASS] if len(fp) > FP_MID_MASS else 0.5
            long_mass = fp[FP_LONG_MASS] if len(fp) > FP_LONG_MASS else 0.5
            entropy = fp[FP_ENTROPY] if len(fp) > FP_ENTROPY else 0.5

            # Classify zone if not provided
            zone = zones[i] if zones and i < len(zones) else self._classify_zone(fp)

            # Compute component scores
            zone_score = ZONE_IMPORTANCE.get(zone, 0.5)
            entropy_score = 1.0 - min(
                entropy / 4.0, 1.0
            )  # Low entropy = focused = important
            long_range_score = min(
                long_mass * 2, 1.0
            )  # High long-range = retrieval anchor
            position_score = self._position_importance(i, n_tokens)

            # Check for special tokens
            if token_texts and i < len(token_texts):
                text = token_texts[i]
                if self._is_special_token(text):
                    # Boost special tokens (system prompts, markers)
                    zone_score = max(zone_score, 0.8)

            # Combine scores
            final_score = (
                self.zone_weight * zone_score
                + self.entropy_weight * entropy_score
                + self.long_range_weight * long_range_score
                + self.position_weight * position_score
            )

            scores.append(
                TokenImportance(
                    index=i,
                    score=final_score,
                    zone=zone,
                    components={
                        "zone": zone_score,
                        "entropy": entropy_score,
                        "long_range": long_range_score,
                        "position": position_score,
                    },
                )
            )

        # Compute aggregates
        mean_score = np.mean([s.score for s in scores])

        # Sort by importance (descending) for top_k
        sorted_by_importance = sorted(scores, key=lambda s: s.score, reverse=True)
        k = max(1, int(n_tokens * keep_ratio))
        top_k_indices = [s.index for s in sorted_by_importance[:k]]

        # Truncation order: least important first (ascending score)
        sorted_by_truncation = sorted(scores, key=lambda s: s.score)
        truncation_indices = [s.index for s in sorted_by_truncation]

        return ImportanceResult(
            scores=scores,
            mean_score=mean_score,
            top_k_indices=top_k_indices,
            truncation_indices=truncation_indices,
        )

    def _classify_zone(self, fp: np.ndarray) -> str:
        """Classify attention zone from fingerprint features."""
        if len(fp) < 4:
            return "unknown"

        local_mass = fp[FP_LOCAL_MASS]
        mid_mass = fp[FP_MID_MASS]
        long_mass = fp[FP_LONG_MASS]
        entropy = fp[FP_ENTROPY]

        # Simple zone classification based on attention distance distribution
        if local_mass > 0.6 and entropy < 2.5:
            return "syntax_floor"
        elif long_mass > 0.4:
            return "long_range"
        elif mid_mass > 0.4:
            return "semantic_bridge"
        elif local_mass + mid_mass + long_mass < 0.5:
            return "diffuse"
        else:
            return "structure_ripple"

    def _position_importance(self, idx: int, total: int) -> float:
        """Compute positional importance score."""
        if total <= 1:
            return 1.0

        # Start tokens are important (system prompts, instructions)
        if idx < total * 0.1:
            return 0.9 - (idx / (total * 0.1)) * 0.2

        # End tokens are somewhat important (recent context)
        if idx > total * 0.9:
            rel_pos = (idx - total * 0.9) / (total * 0.1)
            return 0.6 + rel_pos * 0.2

        # Middle tokens have baseline importance
        return 0.5

    def _is_special_token(self, text: str) -> bool:
        """Check if token is a special marker."""
        special_markers = [
            "<|im_start|>",
            "<|im_end|>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<s>",
            "</s>",
            "[INST]",
            "[/INST]",
            "<<SYS>>",
            "<</SYS>>",
        ]
        return any(marker in text for marker in special_markers)

    def get_truncation_plan(
        self,
        fingerprints: np.ndarray,
        current_length: int,
        target_length: int,
        zones: Optional[List[str]] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Get indices to keep and drop for context truncation.

        Args:
            fingerprints: Attention fingerprints for all tokens
            current_length: Current context length
            target_length: Desired context length after truncation
            zones: Optional zone classifications

        Returns:
            (keep_indices, drop_indices) - indices to keep and drop
        """
        if current_length <= target_length:
            return list(range(current_length)), []

        keep_ratio = target_length / current_length
        result = self.score_from_fingerprints(
            fingerprints, zones=zones, keep_ratio=keep_ratio
        )

        # Keep top scoring tokens up to target length
        keep_indices = sorted(result.top_k_indices[:target_length])
        drop_indices = [i for i in range(current_length) if i not in keep_indices]

        return keep_indices, drop_indices


class SmartTruncator:
    """
    High-level interface for smart context truncation.

    Designed for use in the serving pipeline when context exceeds max length.

    Example:
        truncator = SmartTruncator(max_context=4096)

        # During request processing:
        if len(tokens) > truncator.max_context:
            tokens, fingerprints = truncator.truncate(tokens, fingerprints)
    """

    def __init__(
        self,
        max_context: int = 4096,
        preserve_start_ratio: float = 0.2,
        preserve_end_ratio: float = 0.1,
    ):
        """
        Args:
            max_context: Maximum context length
            preserve_start_ratio: Always preserve this fraction of start tokens
            preserve_end_ratio: Always preserve this fraction of end tokens
        """
        self.max_context = max_context
        self.preserve_start_ratio = preserve_start_ratio
        self.preserve_end_ratio = preserve_end_ratio
        self.scorer = TokenImportanceScorer()

    def truncate(
        self,
        tokens: List[str],
        fingerprints: Optional[np.ndarray] = None,
        zones: Optional[List[str]] = None,
    ) -> Tuple[List[str], Optional[np.ndarray], List[int]]:
        """
        Truncate tokens to max_context using importance scoring.

        Args:
            tokens: List of token strings
            fingerprints: Optional attention fingerprints
            zones: Optional zone classifications

        Returns:
            (truncated_tokens, truncated_fingerprints, kept_indices)
        """
        n_tokens = len(tokens)

        if n_tokens <= self.max_context:
            return tokens, fingerprints, list(range(n_tokens))

        # Calculate preserved regions
        preserve_start = int(n_tokens * self.preserve_start_ratio)
        preserve_end = int(n_tokens * self.preserve_end_ratio)
        middle_target = self.max_context - preserve_start - preserve_end

        # Start tokens always kept
        keep_indices = set(range(preserve_start))

        # End tokens always kept
        keep_indices.update(range(n_tokens - preserve_end, n_tokens))

        # Score middle tokens
        if fingerprints is not None and len(fingerprints) > preserve_start:
            middle_fp = fingerprints[preserve_start : n_tokens - preserve_end]
            middle_zones = (
                zones[preserve_start : n_tokens - preserve_end] if zones else None
            )

            if len(middle_fp) > 0:
                result = self.scorer.score_from_fingerprints(
                    middle_fp,
                    zones=middle_zones,
                    keep_ratio=middle_target / len(middle_fp),
                )

                # Add most important middle tokens
                for idx in result.top_k_indices[:middle_target]:
                    keep_indices.add(idx + preserve_start)
        else:
            # No fingerprints: uniform sampling of middle
            if middle_target > 0:
                middle_indices = list(range(preserve_start, n_tokens - preserve_end))
                step = max(1, len(middle_indices) // middle_target)
                for i in range(0, len(middle_indices), step):
                    if len(keep_indices) < self.max_context:
                        keep_indices.add(middle_indices[i])

        # Sort and truncate
        keep_indices = sorted(keep_indices)[: self.max_context]

        truncated_tokens = [tokens[i] for i in keep_indices]
        truncated_fp = fingerprints[keep_indices] if fingerprints is not None else None

        return truncated_tokens, truncated_fp, keep_indices


def compute_importance_from_attention(
    attention_scores: np.ndarray,
    token_texts: Optional[List[str]] = None,
) -> ImportanceResult:
    """
    Compute token importance directly from attention scores.

    This is a simpler approach when fingerprints are not available.

    Args:
        attention_scores: [n_tokens, n_tokens] attention matrix or
                         [n_tokens] received attention scores
        token_texts: Optional token texts

    Returns:
        ImportanceResult with importance scores
    """
    if len(attention_scores.shape) == 2:
        # Sum attention received by each token (column sum)
        received_attention = attention_scores.sum(axis=0)
    else:
        received_attention = attention_scores

    # Normalize to [0, 1]
    if received_attention.max() > 0:
        received_attention = received_attention / received_attention.max()

    n_tokens = len(received_attention)
    scores = []

    for i, attn in enumerate(received_attention):
        # Position boost
        pos_score = 0.5
        if i < n_tokens * 0.1:
            pos_score = 0.8
        elif i > n_tokens * 0.9:
            pos_score = 0.6

        # Combine attention received with position
        final_score = 0.7 * attn + 0.3 * pos_score

        scores.append(
            TokenImportance(
                index=i,
                score=final_score,
                zone="unknown",
                components={"attention": attn, "position": pos_score},
            )
        )

    mean_score = np.mean([s.score for s in scores])
    sorted_scores = sorted(scores, key=lambda s: s.score, reverse=True)
    top_k_indices = [s.index for s in sorted_scores[: n_tokens // 2]]
    truncation_indices = [s.index for s in sorted(scores, key=lambda s: s.score)]

    return ImportanceResult(
        scores=scores,
        mean_score=mean_score,
        top_k_indices=top_k_indices,
        truncation_indices=truncation_indices,
    )
