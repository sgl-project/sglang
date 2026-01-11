"""
RoPE De-Rotation: Revealing Pure Semantic Attention

This module provides tools to mathematically "undo" the Rotary Position Embedding
(RoPE) from attention patterns, revealing the underlying semantic relationships
independent of token positions.

=== EDUCATIONAL OVERVIEW ===

What is RoPE?
-------------
RoPE (Rotary Position Embedding) is how modern LLMs know WHERE tokens are in a sequence.
Instead of adding position numbers, RoPE *rotates* the query and key vectors.

Think of it like a compass:
- Each position in the sequence gets a unique "rotation angle"
- When the model computes attention, it compares the *rotated* vectors
- The rotation encodes "how far apart are these tokens?"

The Problem with Raw Attention:
-------------------------------
When you look at raw attention scores, you see a MIX of two things:
1. SEMANTIC similarity - "Do these tokens MEAN related things?"
2. POSITIONAL bias - "Are these tokens NEAR each other?"

These are tangled together. A high attention score might mean:
- The tokens are semantically related (good for understanding)
- OR they're just close together (positional prior)

What De-Rotation Reveals:
-------------------------
By mathematically undoing RoPE, we isolate the SEMANTIC component.
This shows us: "Would these tokens attend to each other if position didn't matter?"

Use Cases:
----------
1. Understanding reasoning: Does the model connect concepts across long distances?
2. Detecting memorization: High de-rotated attention to early tokens = retrieval
3. Finding semantic clusters: Which tokens form "meaning groups"?
4. Debugging: Is attention driven by meaning or just proximity?

=== MATHEMATICAL DETAILS ===

RoPE applies rotation R(pos) to query Q and key K:
    Q_rotated = R(pos_q) @ Q
    K_rotated = R(pos_k) @ K

Attention score = Q_rotated · K_rotated = Q @ R(pos_q)^T @ R(pos_k) @ K
                = Q @ R(pos_k - pos_q) @ K

De-rotation removes R(pos_k - pos_q):
    score_derotated = Q · K  (pure semantic similarity)

"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# =============================================================================
# EDUCATIONAL GLOSSARY
# =============================================================================

GLOSSARY = {
    "rope": {
        "term": "RoPE (Rotary Position Embedding)",
        "simple": "How the model knows WHERE tokens are in the sequence",
        "detailed": "A technique that encodes position by rotating vectors. Each position gets a unique rotation angle, and the model learns to use these rotations to understand relative distances between tokens.",
        "analogy": "Like a clock where each token's position is shown by the angle of the hands. When comparing two tokens, the model looks at how different their angles are.",
        "example": "Token at position 10 is rotated 10 degrees, position 20 is rotated 20 degrees. Their attention score includes information that they're 10 positions apart.",
        "why_it_matters": "RoPE lets the model understand 'this token is 5 positions before that one' without needing absolute position numbers."
    },
    "de_rotation": {
        "term": "De-Rotation",
        "simple": "Mathematically removing position information from attention",
        "detailed": "The inverse operation of RoPE. By 'unrotating' the attention scores, we reveal the pure semantic similarity between tokens - what they would attend to if position didn't matter.",
        "analogy": "Like removing tinted glasses to see true colors. Raw attention is 'tinted' by position; de-rotation removes that tint.",
        "example": "If token A strongly attends to token B in raw attention, de-rotation tells us if that's because they're semantically related or just positionally close.",
        "why_it_matters": "Helps distinguish 'these concepts are related' from 'these words happen to be near each other'."
    },
    "semantic_attention": {
        "term": "Semantic Attention (De-Rotated)",
        "simple": "Attention based purely on meaning, ignoring position",
        "detailed": "After removing positional encoding (de-rotation), what remains is the 'pure meaning' similarity between tokens. High semantic attention means the tokens are conceptually related regardless of where they appear.",
        "analogy": "Like finding friends at a party based on shared interests, not just who's standing nearby.",
        "example": "In 'The cat sat on the mat', semantic attention might connect 'cat' to 'sat' (subject-verb) strongly even if raw attention is dominated by nearby words.",
        "why_it_matters": "Reveals the model's understanding of concepts and relationships, not just surface patterns."
    },
    "positional_attention": {
        "term": "Positional Attention",
        "simple": "Attention driven by how close tokens are",
        "detailed": "The component of attention that comes from RoPE encoding. Nearby tokens naturally have higher positional attention. This is the 'local bias' that helps with grammar and syntax.",
        "analogy": "Like only talking to people sitting next to you at dinner - closeness matters.",
        "example": "Words like 'the', 'a', 'is' often have high positional attention to neighbors because syntax is local.",
        "why_it_matters": "Helps understand when the model is using local patterns (good for grammar) vs. long-range reasoning."
    },
    "attention_sink": {
        "term": "Attention Sink (Index 0-4)",
        "simple": "Special tokens that collect 'leftover' attention",
        "detailed": "The first few tokens (especially <BOS> or system prompt tokens) often receive high attention regardless of content. They act as 'garbage collectors' for attention probability mass that doesn't have a better target.",
        "analogy": "Like a 'miscellaneous' folder on your computer - stuff goes there when it doesn't fit elsewhere.",
        "example": "The <BOS> token might have 40% attention even when it has no semantic meaning - it's just absorbing excess attention.",
        "why_it_matters": "We filter these out to see the 'real' attention patterns. Without filtering, the sink dominates everything."
    },
    "attention_entropy": {
        "term": "Attention Entropy",
        "simple": "How spread out vs focused the attention is",
        "detailed": "A measure of attention distribution. Low entropy = focused on few tokens. High entropy = spread across many tokens. Calculated as -sum(p * log(p)) where p is attention probability.",
        "analogy": "Like a spotlight vs. a floodlight. Spotlight (low entropy) illuminates one spot brightly. Floodlight (high entropy) covers everything dimly.",
        "example": "Entropy=0.5: Very focused, attending to 1-2 tokens. Entropy=4.0: Spread across many tokens, less certain.",
        "why_it_matters": "Low entropy often indicates confident retrieval or syntax. High entropy suggests the model is 'considering many options' - often during reasoning."
    },
    "manifold_zone": {
        "term": "Manifold Zone",
        "simple": "Which 'mode' the attention is operating in",
        "detailed": "Attention patterns cluster into distinct zones based on their characteristics. Each zone represents a different cognitive operation: syntax processing, semantic bridging, or structural reasoning.",
        "zones": {
            "syntax_floor": {
                "name": "Syntax Floor",
                "simple": "Processing grammar and local structure",
                "characteristics": "High local attention, low entropy. Model is handling nearby tokens.",
                "color": "#4CAF50",
                "example": "Attention from 'running' to 'is' in 'The dog is running'"
            },
            "semantic_bridge": {
                "name": "Semantic Bridge",
                "simple": "Connecting meaning across medium distances",
                "characteristics": "Balanced attention, moderate entropy. Model is linking concepts.",
                "color": "#2196F3",
                "example": "Attention from 'conclusion' to 'hypothesis' across a paragraph"
            },
            "structure_ripple": {
                "name": "Structure Ripple",
                "simple": "Processing document structure and long-range patterns",
                "characteristics": "High long-range attention, high entropy. Model is reasoning across the full context.",
                "color": "#FF9800",
                "example": "Attention from the answer to question keywords far earlier"
            }
        },
        "why_it_matters": "Different zones indicate different cognitive operations. This helps understand what the model is 'doing' at each step."
    },
    "rotational_variance": {
        "term": "Rotational Variance",
        "simple": "How much position affects this attention pattern",
        "detailed": "Measures the difference between raw attention and de-rotated attention. High variance = position matters a lot. Low variance = attention is primarily semantic.",
        "analogy": "Like measuring how much your opinion of someone changes based on where you meet them. High variance = very location-dependent. Low variance = you'd feel the same anywhere.",
        "example": "Rotational variance of 0.8: This attention pattern is 80% driven by position. Variance of 0.1: Mostly semantic.",
        "why_it_matters": "Helps identify whether the model is using position as a shortcut or genuine semantic understanding."
    },
    "sinq_anchor": {
        "term": "Sinq Anchor (Coordinate Origin)",
        "simple": "Using the attention sink as a reference point",
        "detailed": "Instead of just filtering out the sink, we use it as a 'compass origin'. All other attention is measured relative to the sink's direction. This reveals the 'heading' of each thought.",
        "analogy": "Like using true north on a compass. The sink is 'north', and we measure which direction the attention is pointing.",
        "example": "If attention to sink is (0.8, 0.2) and to token X is (0.6, 0.7), the relative angle tells us X's semantic direction.",
        "why_it_matters": "Provides a stable reference for comparing attention across different positions and requests."
    }
}


# =============================================================================
# INTERPRETATION TEMPLATES
# =============================================================================

INTERPRETATION_TEMPLATES = {
    "high_semantic_low_positional": {
        "pattern": "High semantic attention, low positional bias",
        "meaning": "The model is connecting these tokens based on MEANING, not just proximity.",
        "interpretation": "This suggests genuine understanding - the model recognizes these concepts are related even though they might be far apart.",
        "examples": [
            "Connecting 'photosynthesis' to 'plants' across a long explanation",
            "Linking a conclusion to evidence presented earlier",
            "Finding the subject of a sentence through complex clauses"
        ],
        "confidence": "high",
        "icon": "brain"
    },
    "high_positional_low_semantic": {
        "pattern": "High positional attention, low semantic similarity",
        "meaning": "The model is using POSITION as a shortcut - these tokens happen to be nearby.",
        "interpretation": "This is normal for grammar/syntax, but suspicious for reasoning. The model might be using positional heuristics rather than understanding.",
        "examples": [
            "Article 'the' attending to the next noun",
            "Punctuation attending to adjacent words",
            "Copy-paste patterns where position matters more than content"
        ],
        "confidence": "medium",
        "icon": "ruler"
    },
    "high_both": {
        "pattern": "High semantic AND positional attention",
        "meaning": "These tokens are both nearby AND semantically related.",
        "interpretation": "Strong signal - the model is confident. Common in well-structured text where related concepts are placed near each other.",
        "examples": [
            "Subject-verb agreement in simple sentences",
            "Modifier-noun pairs like 'red apple'",
            "Question word to immediate answer"
        ],
        "confidence": "very_high",
        "icon": "check-circle"
    },
    "low_both": {
        "pattern": "Low semantic AND positional attention",
        "meaning": "These tokens are neither related nor particularly close.",
        "interpretation": "Weak connection - the model doesn't see a strong relationship. This is the 'background' attention.",
        "examples": [
            "Unrelated sentences in a document",
            "Different topics in a conversation",
            "Filler words to distant content"
        ],
        "confidence": "low",
        "icon": "minus-circle"
    },
    "sink_dominated": {
        "pattern": "Attention dominated by sink tokens (position 0-4)",
        "meaning": "The model is 'dumping' attention to the beginning - it's uncertain.",
        "interpretation": "The sink absorbs attention that doesn't have a clear target. High sink attention often means the model is confused or the query is ambiguous.",
        "examples": [
            "Ambiguous pronoun resolution",
            "Missing context in a question",
            "Early in generation before context is established"
        ],
        "confidence": "uncertain",
        "icon": "question-circle"
    }
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class AttentionMode(Enum):
    """Mode of attention pattern."""
    RAW = "raw"                    # Original attention with RoPE
    DEROTATED = "derotated"        # Semantic attention (RoPE removed)
    POSITIONAL = "positional"      # Positional component only
    SINQ_ANCHORED = "sinq_anchored"  # Relative to sink anchor


@dataclass
class DerotatedAttention:
    """
    Attention analysis with semantic/positional decomposition.

    Educational fields explain what each metric means.
    """
    # Core metrics
    raw_scores: np.ndarray              # Original attention scores
    semantic_scores: np.ndarray         # De-rotated (pure meaning)
    positional_bias: np.ndarray         # Positional component

    # Derived metrics
    rotational_variance: float          # How much position affects attention
    semantic_entropy: float             # Spread of semantic attention
    positional_entropy: float           # Spread of positional attention

    # Interpretation
    dominant_mode: str                  # "semantic", "positional", or "balanced"
    interpretation: Dict[str, Any]      # Human-readable interpretation
    manifold_zone: str                  # syntax_floor, semantic_bridge, etc.

    # Educational annotations
    explanations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "raw_scores": self.raw_scores.tolist() if isinstance(self.raw_scores, np.ndarray) else self.raw_scores,
            "semantic_scores": self.semantic_scores.tolist() if isinstance(self.semantic_scores, np.ndarray) else self.semantic_scores,
            "positional_bias": self.positional_bias.tolist() if isinstance(self.positional_bias, np.ndarray) else self.positional_bias,
            "rotational_variance": self.rotational_variance,
            "semantic_entropy": self.semantic_entropy,
            "positional_entropy": self.positional_entropy,
            "dominant_mode": self.dominant_mode,
            "interpretation": self.interpretation,
            "manifold_zone": self.manifold_zone,
            "explanations": self.explanations,
        }


@dataclass
class RoPEConfig:
    """Configuration for RoPE de-rotation."""
    head_dim: int = 128           # Dimension of each attention head
    base: float = 10000.0         # RoPE base frequency
    is_neox_style: bool = True    # Whether to use NeoX-style rotation
    max_position: int = 131072    # Maximum position for cache


# =============================================================================
# CORE DE-ROTATION IMPLEMENTATION
# =============================================================================

class RoPEDerotator:
    """
    De-rotate attention patterns to reveal semantic similarity.

    Educational Example:
    -------------------
    >>> derotator = RoPEDerotator(head_dim=128)
    >>>
    >>> # Raw attention: position 100 attending to positions [50, 90, 95, 98]
    >>> result = derotator.analyze(
    ...     query_pos=100,
    ...     key_positions=[50, 90, 95, 98],
    ...     attention_scores=[0.1, 0.15, 0.25, 0.5]
    ... )
    >>>
    >>> # Result shows which attention is semantic vs positional
    >>> print(result.interpretation)
    {'pattern': 'high_positional_low_semantic',
     'meaning': 'Nearby tokens (98, 95) dominate due to position, not meaning'}
    """

    def __init__(self, config: Optional[RoPEConfig] = None):
        """
        Initialize the de-rotator.

        Args:
            config: RoPE configuration. Uses model defaults if not provided.
        """
        self.config = config or RoPEConfig()
        self._cos_cache = None
        self._sin_cache = None
        self._build_cache()

    def _build_cache(self):
        """Pre-compute cos/sin cache for efficient de-rotation."""
        inv_freq = 1.0 / (
            self.config.base ** (
                np.arange(0, self.config.head_dim, 2, dtype=np.float32)
                / self.config.head_dim
            )
        )

        positions = np.arange(self.config.max_position, dtype=np.float32)
        freqs = np.outer(positions, inv_freq)  # [max_pos, head_dim/2]

        self._cos_cache = np.cos(freqs)
        self._sin_cache = np.sin(freqs)

    def _get_rotation(self, position: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get cos/sin rotation for a position.

        Educational Note:
        ----------------
        RoPE rotates vectors in 2D subspaces. For position p:
        - cos(p * freq) and sin(p * freq) define the rotation
        - Different frequency components capture different "scales"
        - Low frequencies: long-range patterns
        - High frequencies: local patterns
        """
        return self._cos_cache[position], self._sin_cache[position]

    def compute_relative_rotation(
        self,
        query_pos: int,
        key_pos: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the relative rotation between query and key positions.

        This is the rotation that RoPE "adds" to the attention score.

        Args:
            query_pos: Position of the query token
            key_pos: Position of the key token

        Returns:
            (cos, sin) of the relative rotation

        Educational Note:
        ----------------
        The relative rotation R(q-k) encodes "how far apart are these tokens?"
        - Nearby tokens: small rotation, minimal positional effect
        - Distant tokens: large rotation, significant positional effect
        """
        # Relative position
        delta = query_pos - key_pos

        # Handle negative positions (key after query)
        if delta < 0:
            delta = abs(delta)

        if delta >= self.config.max_position:
            delta = self.config.max_position - 1

        return self._get_rotation(delta)

    def estimate_positional_bias(
        self,
        query_pos: int,
        key_positions: List[int],
    ) -> np.ndarray:
        """
        Estimate the positional bias component of attention.

        This approximates how much of the attention is due to position
        versus semantic content.

        Args:
            query_pos: Position of query token
            key_positions: Positions of key tokens

        Returns:
            Array of positional bias estimates (0 to 1)

        Educational Note:
        ----------------
        Positional bias decays with distance but not monotonically.
        RoPE creates oscillating patterns at different frequencies:
        - Very close tokens: high positional similarity
        - Medium distance: lower but variable
        - Very far: approaches neutral
        """
        biases = []

        for key_pos in key_positions:
            cos, sin = self.compute_relative_rotation(query_pos, key_pos)

            # Positional bias is higher when rotation is smaller
            # Average cos across frequencies as proxy for positional similarity
            # cos(0) = 1 means no rotation (same position)
            # cos(pi) = -1 means maximum rotation (opposite phase)
            avg_cos = np.mean(cos)

            # Map to 0-1 range: (cos + 1) / 2
            # 1 = same position, 0.5 = orthogonal, 0 = opposite
            bias = (avg_cos + 1) / 2
            biases.append(bias)

        return np.array(biases)

    def derotate_scores(
        self,
        query_pos: int,
        key_positions: List[int],
        attention_scores: List[float],
    ) -> np.ndarray:
        """
        Estimate semantic attention by removing positional bias.

        This is an approximation since we don't have the original Q/K vectors.
        We estimate the semantic component by adjusting for positional bias.

        Args:
            query_pos: Position of query token
            key_positions: Positions of attended tokens
            attention_scores: Original attention scores

        Returns:
            Estimated semantic attention scores

        Educational Note:
        ----------------
        This is a simplification. True de-rotation requires:
        1. Access to the original Q and K vectors
        2. Applying inverse rotation R(-delta)
        3. Recomputing dot product

        We approximate by adjusting scores based on expected positional bias.
        """
        scores = np.array(attention_scores)
        positional_bias = self.estimate_positional_bias(query_pos, key_positions)

        # Estimate semantic component
        # If positional bias is high but raw attention is low, semantic is low
        # If positional bias is low but raw attention is high, semantic is high

        # Simple model: semantic = raw / (positional_weight * bias + baseline)
        # This "divides out" the positional boost

        positional_weight = 0.5  # How much position typically contributes
        baseline = 0.5  # Baseline attention without position

        divisor = positional_weight * positional_bias + baseline
        semantic = scores / divisor

        # Normalize to sum to 1 (like attention)
        semantic = semantic / semantic.sum() if semantic.sum() > 0 else semantic

        return semantic

    def analyze(
        self,
        query_pos: int,
        key_positions: List[int],
        attention_scores: List[float],
        sink_threshold: int = 5,
    ) -> DerotatedAttention:
        """
        Full analysis of attention pattern with educational interpretation.

        Args:
            query_pos: Position of query token
            key_positions: Positions of attended tokens
            attention_scores: Original attention scores
            sink_threshold: Positions below this are considered "sink"

        Returns:
            DerotatedAttention with full analysis and explanations

        Educational Example:
        -------------------
        >>> result = derotator.analyze(
        ...     query_pos=100,
        ...     key_positions=[0, 1, 50, 98, 99],
        ...     attention_scores=[0.3, 0.1, 0.1, 0.3, 0.2]
        ... )
        >>>
        >>> # High attention to positions 0,1 (sink) and 98,99 (nearby)
        >>> print(result.interpretation['pattern'])
        'sink_dominated' or 'high_positional_low_semantic'
        >>>
        >>> # The explanation tells you what this means
        >>> print(result.explanations['summary'])
        'Attention is split between sink tokens (0-4) and nearby tokens...'
        """
        scores = np.array(attention_scores)
        positions = np.array(key_positions)

        # Compute positional bias
        positional_bias = self.estimate_positional_bias(query_pos, key_positions)

        # Compute semantic scores
        semantic_scores = self.derotate_scores(query_pos, key_positions, attention_scores)

        # Compute rotational variance
        # High variance = big difference between raw and semantic
        variance = np.mean(np.abs(scores - semantic_scores))

        # Compute entropies
        def entropy(p):
            p = p[p > 0]
            return -np.sum(p * np.log2(p + 1e-10))

        semantic_entropy = entropy(semantic_scores)
        positional_entropy = entropy(positional_bias / positional_bias.sum() if positional_bias.sum() > 0 else positional_bias)

        # Classify the pattern
        # Check for sink dominance
        sink_mask = positions < sink_threshold
        sink_attention = scores[sink_mask].sum() if sink_mask.any() else 0

        if sink_attention > 0.3:
            pattern = "sink_dominated"
        elif variance > 0.3:  # High rotational variance
            if positional_bias.mean() > 0.6:
                pattern = "high_positional_low_semantic"
            else:
                pattern = "high_semantic_low_positional"
        else:
            avg_semantic = semantic_scores.mean()
            avg_positional = positional_bias.mean()
            if avg_semantic > 0.3 and avg_positional > 0.6:
                pattern = "high_both"
            else:
                pattern = "low_both"

        interpretation = INTERPRETATION_TEMPLATES.get(pattern, {})

        # Determine dominant mode
        if variance < 0.1:
            dominant_mode = "balanced"
        elif semantic_scores.max() > positional_bias.max():
            dominant_mode = "semantic"
        else:
            dominant_mode = "positional"

        # Classify manifold zone
        distances = np.abs(query_pos - positions)
        local_mass = scores[distances <= 16].sum() if (distances <= 16).any() else 0
        long_mass = scores[distances > 256].sum() if (distances > 256).any() else 0

        if local_mass > 0.5:
            manifold_zone = "syntax_floor"
        elif long_mass > 0.4:
            manifold_zone = "structure_ripple"
        else:
            manifold_zone = "semantic_bridge"

        # Build educational explanations
        explanations = self._build_explanations(
            scores, semantic_scores, positional_bias,
            variance, pattern, manifold_zone, sink_attention
        )

        return DerotatedAttention(
            raw_scores=scores,
            semantic_scores=semantic_scores,
            positional_bias=positional_bias,
            rotational_variance=float(variance),
            semantic_entropy=float(semantic_entropy),
            positional_entropy=float(positional_entropy),
            dominant_mode=dominant_mode,
            interpretation=interpretation,
            manifold_zone=manifold_zone,
            explanations=explanations,
        )

    def _build_explanations(
        self,
        raw: np.ndarray,
        semantic: np.ndarray,
        positional: np.ndarray,
        variance: float,
        pattern: str,
        zone: str,
        sink_attention: float,
    ) -> Dict[str, str]:
        """Build human-readable explanations for the analysis."""

        explanations = {}

        # Summary
        if pattern == "sink_dominated":
            explanations["summary"] = (
                f"This attention is dominated by 'sink' tokens ({sink_attention:.0%}). "
                "The model is uncertain and directing attention to the beginning of the sequence."
            )
        elif pattern == "high_positional_low_semantic":
            explanations["summary"] = (
                f"Position drives this attention (variance={variance:.2f}). "
                "The model is attending to nearby tokens, likely for grammar/syntax."
            )
        elif pattern == "high_semantic_low_positional":
            explanations["summary"] = (
                f"Semantic similarity drives this attention (variance={variance:.2f}). "
                "The model found meaningful connections regardless of position."
            )
        elif pattern == "high_both":
            explanations["summary"] = (
                "Strong attention from both position AND meaning. "
                "These tokens are nearby and semantically related - high confidence signal."
            )
        else:
            explanations["summary"] = (
                "Weak attention pattern. No strong positional or semantic signal."
            )

        # Zone explanation
        zone_info = GLOSSARY["manifold_zone"]["zones"].get(zone, {})
        explanations["zone"] = zone_info.get("simple", "Unknown zone")

        # Metrics explanation
        explanations["variance"] = (
            f"Rotational variance ({variance:.2f}): "
            + ("Position strongly affects attention" if variance > 0.3 else "Position has minimal effect")
        )

        # Actionable insight
        if pattern == "sink_dominated":
            explanations["insight"] = "Consider: Is the query ambiguous? Is context missing?"
        elif pattern == "high_positional_low_semantic":
            explanations["insight"] = "Normal for syntax processing. Suspicious if expecting reasoning."
        elif pattern == "high_semantic_low_positional":
            explanations["insight"] = "Good sign for reasoning - the model is connecting concepts."
        else:
            explanations["insight"] = "Watch for how this evolves in subsequent tokens."

        return explanations


# =============================================================================
# EDUCATIONAL API
# =============================================================================

def get_glossary() -> Dict[str, Any]:
    """Get the full educational glossary."""
    return GLOSSARY


def get_term_explanation(term: str, detail_level: str = "simple") -> Optional[str]:
    """
    Get explanation for a term at specified detail level.

    Args:
        term: Term to explain (e.g., "rope", "de_rotation")
        detail_level: "simple", "detailed", "analogy", or "example"

    Returns:
        Explanation string or None if not found
    """
    if term not in GLOSSARY:
        return None
    return GLOSSARY[term].get(detail_level)


def get_interpretation_for_pattern(pattern: str) -> Dict[str, Any]:
    """Get interpretation template for a pattern."""
    return INTERPRETATION_TEMPLATES.get(pattern, {})


def explain_attention_step(
    query_token: str,
    key_tokens: List[str],
    attention_scores: List[float],
    analysis: DerotatedAttention,
) -> str:
    """
    Generate a natural language explanation of an attention step.

    Educational tool to help users understand what the model is doing.

    Args:
        query_token: The token that's attending
        key_tokens: Tokens being attended to
        attention_scores: Raw attention scores
        analysis: DerotatedAttention analysis result

    Returns:
        Human-readable explanation

    Example:
    -------
    >>> explanation = explain_attention_step(
    ...     query_token="conclusion",
    ...     key_tokens=["the", "experiment", "shows", "that"],
    ...     attention_scores=[0.1, 0.4, 0.3, 0.2],
    ...     analysis=derotated_result
    ... )
    >>> print(explanation)
    "When generating 'conclusion', the model strongly attends to 'experiment' (40%).
     De-rotation reveals this is primarily SEMANTIC - the model is connecting
     'conclusion' to the key concept 'experiment' regardless of position.
     This is in the 'semantic_bridge' zone, indicating concept linking."
    """
    # Find top attended tokens
    top_indices = np.argsort(attention_scores)[-3:][::-1]

    # Build explanation
    lines = [f"When generating '{query_token}':\n"]

    # Describe top attention targets
    for i, idx in enumerate(top_indices):
        if idx < len(key_tokens):
            token = key_tokens[idx]
            raw_score = attention_scores[idx]
            semantic = analysis.semantic_scores[idx]
            positional = analysis.positional_bias[idx]

            if semantic > positional:
                reason = "SEMANTIC (meaning-based)"
            elif positional > semantic:
                reason = "POSITIONAL (nearby)"
            else:
                reason = "BALANCED (both)"

            lines.append(
                f"  {i+1}. '{token}' receives {raw_score:.0%} attention - "
                f"primarily {reason}"
            )

    # Add zone context
    lines.append(f"\nThis is in the '{analysis.manifold_zone}' zone:")
    lines.append(f"  {analysis.explanations['zone']}")

    # Add insight
    lines.append(f"\nInsight: {analysis.explanations['insight']}")

    return "\n".join(lines)


# =============================================================================
# TESTING
# =============================================================================

# =============================================================================
# FINGERPRINT INTEGRATION
# =============================================================================

def compute_rotational_variance_for_fingerprint(
    query_pos: int,
    key_positions: List[int],
    attention_scores: List[float],
    derotator: Optional[RoPEDerotator] = None,
) -> float:
    """
    Compute rotational variance for use in fingerprint schema v2.

    This is a convenience function that extracts just the rotational variance
    from a full de-rotation analysis, suitable for extending a fingerprint
    from 20 to 21 dimensions.

    Args:
        query_pos: Position of the query token
        key_positions: Positions of attended tokens
        attention_scores: Attention weights

    Returns:
        Rotational variance (0.0 = pure semantic, 1.0 = pure positional)

    Example:
        >>> # During fingerprint computation
        >>> rv = compute_rotational_variance_for_fingerprint(
        ...     query_pos=100,
        ...     key_positions=[50, 90, 95, 98, 99],
        ...     attention_scores=[0.1, 0.1, 0.2, 0.3, 0.3]
        ... )
        >>> # Extend 20-dim fingerprint to 21-dim
        >>> fingerprint_v2 = np.append(fingerprint_v1, rv)
    """
    if derotator is None:
        derotator = RoPEDerotator()

    if not key_positions or not attention_scores:
        return 0.5  # Neutral default

    result = derotator.analyze(query_pos, key_positions, attention_scores)
    return result.rotational_variance


def compute_rotational_variance_batch(
    query_positions: List[int],
    key_positions_batch: List[List[int]],
    attention_scores_batch: List[List[float]],
    derotator: Optional[RoPEDerotator] = None,
) -> np.ndarray:
    """
    Compute rotational variance for a batch of attention patterns.

    Optimized for batch processing during fingerprint computation.

    Args:
        query_positions: List of query positions for each step
        key_positions_batch: List of key position lists
        attention_scores_batch: List of attention score lists

    Returns:
        Array of rotational variance values

    Example:
        >>> # For a batch of 100 decode steps
        >>> rv_batch = compute_rotational_variance_batch(
        ...     query_positions=list(range(100, 200)),
        ...     key_positions_batch=[...],
        ...     attention_scores_batch=[...]
        ... )
        >>> # Append to fingerprint matrix
        >>> fingerprints_v2 = np.column_stack([fingerprints_v1, rv_batch])
    """
    if derotator is None:
        derotator = RoPEDerotator()

    n = len(query_positions)
    variances = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if key_positions_batch[i] and attention_scores_batch[i]:
            result = derotator.analyze(
                query_positions[i],
                key_positions_batch[i],
                attention_scores_batch[i]
            )
            variances[i] = result.rotational_variance
        else:
            variances[i] = 0.5  # Neutral default

    return variances


def extend_fingerprint_with_rotational_variance(
    fingerprint: np.ndarray,
    rotational_variance: float,
) -> np.ndarray:
    """
    Extend a 20-dim fingerprint to 21-dim by adding rotational variance.

    Args:
        fingerprint: 20-dimensional fingerprint (schema v1)
        rotational_variance: Rotational variance value (0-1)

    Returns:
        21-dimensional fingerprint (schema v2)
    """
    if len(fingerprint) >= 21:
        # Already v2 or later
        return fingerprint
    return np.append(fingerprint, rotational_variance)


def extend_fingerprints_batch(
    fingerprints: np.ndarray,
    rotational_variances: np.ndarray,
) -> np.ndarray:
    """
    Extend a batch of 20-dim fingerprints to 21-dim.

    Args:
        fingerprints: Array of shape (N, 20)
        rotational_variances: Array of shape (N,)

    Returns:
        Array of shape (N, 21)
    """
    if fingerprints.shape[1] >= 21:
        return fingerprints
    return np.column_stack([fingerprints, rotational_variances])


def demo():
    """Demonstrate the RoPE de-rotation analysis."""
    print("=" * 60)
    print("RoPE De-Rotation Demo: Understanding Semantic vs Positional Attention")
    print("=" * 60)

    derotator = RoPEDerotator()

    # Example 1: Local syntax (high positional)
    print("\n--- Example 1: Local Syntax ---")
    print("Query: 'running' at position 10")
    print("Keys: 'The', 'dog', 'is' at positions 7, 8, 9")

    result1 = derotator.analyze(
        query_pos=10,
        key_positions=[7, 8, 9],
        attention_scores=[0.1, 0.3, 0.6]
    )

    print(f"\nResult:")
    print(f"  Pattern: {result1.interpretation.get('pattern', 'N/A')}")
    print(f"  Zone: {result1.manifold_zone}")
    print(f"  Dominant mode: {result1.dominant_mode}")
    print(f"  Summary: {result1.explanations['summary']}")

    # Example 2: Long-range reasoning (high semantic)
    print("\n--- Example 2: Long-Range Reasoning ---")
    print("Query: 'therefore' at position 500")
    print("Keys: 'hypothesis' at 50, 'evidence' at 200, 'the' at 498")

    result2 = derotator.analyze(
        query_pos=500,
        key_positions=[50, 200, 498],
        attention_scores=[0.4, 0.35, 0.25]
    )

    print(f"\nResult:")
    print(f"  Pattern: {result2.interpretation.get('pattern', 'N/A')}")
    print(f"  Zone: {result2.manifold_zone}")
    print(f"  Dominant mode: {result2.dominant_mode}")
    print(f"  Summary: {result2.explanations['summary']}")

    # Example 3: Sink dominated
    print("\n--- Example 3: Uncertain (Sink Dominated) ---")
    print("Query: 'it' at position 100")
    print("Keys: BOS at 0, system tokens 1-4, some content at 50")

    result3 = derotator.analyze(
        query_pos=100,
        key_positions=[0, 1, 2, 3, 4, 50],
        attention_scores=[0.4, 0.15, 0.1, 0.1, 0.05, 0.2]
    )

    print(f"\nResult:")
    print(f"  Pattern: {result3.interpretation.get('pattern', 'N/A')}")
    print(f"  Zone: {result3.manifold_zone}")
    print(f"  Dominant mode: {result3.dominant_mode}")
    print(f"  Summary: {result3.explanations['summary']}")
    print(f"  Insight: {result3.explanations['insight']}")


if __name__ == "__main__":
    demo()
