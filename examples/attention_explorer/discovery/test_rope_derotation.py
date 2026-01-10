"""
Tests for RoPE De-Rotation Module

Tests the educational RoPE de-rotation system that reveals semantic vs positional
attention patterns for model interpretability.
"""

import pytest
import numpy as np
from typing import List

from .rope_derotation import (
    RoPEDerotator,
    RoPEConfig,
    DerotatedAttention,
    AttentionMode,
    GLOSSARY,
    INTERPRETATION_TEMPLATES,
    get_glossary,
    get_term_explanation,
    get_interpretation_for_pattern,
    explain_attention_step,
)


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestRoPEConfig:
    """Test RoPEConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RoPEConfig()
        assert config.head_dim == 128
        assert config.base == 10000.0
        assert config.is_neox_style is True
        assert config.max_position == 131072

    def test_custom_values(self):
        """Test custom configuration."""
        config = RoPEConfig(
            head_dim=64,
            base=5000.0,
            is_neox_style=False,
            max_position=8192
        )
        assert config.head_dim == 64
        assert config.base == 5000.0
        assert config.is_neox_style is False
        assert config.max_position == 8192


# =============================================================================
# DEROTATOR INITIALIZATION TESTS
# =============================================================================

class TestRoPEDerotatorInit:
    """Test RoPEDerotator initialization."""

    def test_default_init(self):
        """Test default initialization."""
        derotator = RoPEDerotator()
        assert derotator.config.head_dim == 128
        assert derotator._cos_cache is not None
        assert derotator._sin_cache is not None

    def test_custom_config_init(self):
        """Test initialization with custom config."""
        config = RoPEConfig(head_dim=64, max_position=1024)
        derotator = RoPEDerotator(config=config)
        assert derotator.config.head_dim == 64
        assert derotator._cos_cache.shape[0] == 1024

    def test_cache_shape(self):
        """Test cache dimensions."""
        config = RoPEConfig(head_dim=128, max_position=4096)
        derotator = RoPEDerotator(config=config)

        # Cache should be [max_position, head_dim/2]
        assert derotator._cos_cache.shape == (4096, 64)
        assert derotator._sin_cache.shape == (4096, 64)


# =============================================================================
# ROTATION COMPUTATION TESTS
# =============================================================================

class TestRelativeRotation:
    """Test rotation angle computations."""

    def test_same_position_rotation(self):
        """Same position should have cos=1 rotation (no rotation)."""
        derotator = RoPEDerotator()
        cos, sin = derotator.compute_relative_rotation(query_pos=100, key_pos=100)

        # cos(0) should be 1 for all frequencies
        assert np.allclose(cos, 1.0, atol=1e-5)
        # sin(0) should be 0 for all frequencies
        assert np.allclose(sin, 0.0, atol=1e-5)

    def test_adjacent_position_rotation(self):
        """Adjacent positions should have small rotation."""
        derotator = RoPEDerotator()
        cos_1, sin_1 = derotator.compute_relative_rotation(query_pos=101, key_pos=100)
        cos_10, sin_10 = derotator.compute_relative_rotation(query_pos=110, key_pos=100)

        # Larger distance should have larger rotation (smaller cos average)
        assert np.mean(cos_1) > np.mean(cos_10)

    def test_symmetric_rotation(self):
        """Rotation should be symmetric with respect to distance."""
        derotator = RoPEDerotator()
        cos_fwd, sin_fwd = derotator.compute_relative_rotation(query_pos=110, key_pos=100)
        cos_bwd, sin_bwd = derotator.compute_relative_rotation(query_pos=100, key_pos=110)

        # Both should give same rotation magnitude
        np.testing.assert_allclose(cos_fwd, cos_bwd, atol=1e-5)


# =============================================================================
# POSITIONAL BIAS TESTS
# =============================================================================

class TestPositionalBias:
    """Test positional bias estimation."""

    def test_nearby_tokens_higher_bias(self):
        """Nearby tokens should have higher positional bias."""
        derotator = RoPEDerotator()

        biases = derotator.estimate_positional_bias(
            query_pos=100,
            key_positions=[99, 50, 10]  # Nearby, medium, far
        )

        assert biases[0] > biases[1]  # 99 closer than 50
        assert biases[1] > biases[2]  # 50 closer than 10

    def test_same_position_max_bias(self):
        """Same position should have maximum bias."""
        derotator = RoPEDerotator()

        biases = derotator.estimate_positional_bias(
            query_pos=100,
            key_positions=[100, 99, 50]
        )

        assert biases[0] == pytest.approx(1.0, abs=0.01)  # Same position
        assert biases[0] > biases[1] > biases[2]

    def test_bias_range(self):
        """Positional bias should be in [0, 1] range."""
        derotator = RoPEDerotator()

        biases = derotator.estimate_positional_bias(
            query_pos=1000,
            key_positions=[0, 100, 500, 999, 1000]
        )

        assert all(0 <= b <= 1 for b in biases)


# =============================================================================
# DEROTATION TESTS
# =============================================================================

class TestDerotateScores:
    """Test score de-rotation."""

    def test_output_normalized(self):
        """De-rotated scores should sum to 1."""
        derotator = RoPEDerotator()

        semantic = derotator.derotate_scores(
            query_pos=100,
            key_positions=[50, 90, 95, 98],
            attention_scores=[0.1, 0.15, 0.25, 0.5]
        )

        assert semantic.sum() == pytest.approx(1.0, abs=1e-5)

    def test_output_shape(self):
        """Output shape should match input."""
        derotator = RoPEDerotator()

        positions = [10, 20, 30, 40, 50]
        scores = [0.2, 0.2, 0.2, 0.2, 0.2]

        semantic = derotator.derotate_scores(
            query_pos=100,
            key_positions=positions,
            attention_scores=scores
        )

        assert len(semantic) == len(positions)

    def test_distant_tokens_boosted(self):
        """Distant tokens with high raw attention should get semantic boost."""
        derotator = RoPEDerotator()

        # High attention to distant token (50) vs close token (99)
        semantic = derotator.derotate_scores(
            query_pos=100,
            key_positions=[50, 99],
            attention_scores=[0.5, 0.5]  # Same raw attention
        )

        # Distant token should have higher semantic score after de-rotation
        # (because it's getting same attention despite positional disadvantage)
        assert semantic[0] > semantic[1]


# =============================================================================
# FULL ANALYSIS TESTS
# =============================================================================

class TestAnalyze:
    """Test full analysis pipeline."""

    def test_returns_derotated_attention(self):
        """Analysis should return DerotatedAttention dataclass."""
        derotator = RoPEDerotator()

        result = derotator.analyze(
            query_pos=100,
            key_positions=[50, 90, 95, 98],
            attention_scores=[0.1, 0.15, 0.25, 0.5]
        )

        assert isinstance(result, DerotatedAttention)

    def test_all_fields_populated(self):
        """All fields should be populated."""
        derotator = RoPEDerotator()

        result = derotator.analyze(
            query_pos=100,
            key_positions=[50, 90, 95, 98],
            attention_scores=[0.1, 0.15, 0.25, 0.5]
        )

        assert result.raw_scores is not None
        assert result.semantic_scores is not None
        assert result.positional_bias is not None
        assert result.rotational_variance >= 0
        assert result.semantic_entropy >= 0
        assert result.positional_entropy >= 0
        assert result.dominant_mode in ["semantic", "positional", "balanced"]
        assert result.manifold_zone in ["syntax_floor", "semantic_bridge", "structure_ripple"]
        assert len(result.explanations) > 0

    def test_sink_dominated_detection(self):
        """Should detect sink-dominated patterns."""
        derotator = RoPEDerotator()

        # High attention to sink tokens (0-4)
        result = derotator.analyze(
            query_pos=100,
            key_positions=[0, 1, 2, 50, 99],
            attention_scores=[0.4, 0.2, 0.2, 0.1, 0.1]
        )

        # The pattern field contains human-readable description
        assert "sink" in result.interpretation.get("pattern", "").lower()
        assert "sink" in result.explanations["summary"].lower()

    def test_syntax_floor_zone(self):
        """Should classify local attention as syntax_floor."""
        derotator = RoPEDerotator()

        # Very local attention
        result = derotator.analyze(
            query_pos=100,
            key_positions=[97, 98, 99],  # All within 16 positions
            attention_scores=[0.2, 0.3, 0.5]
        )

        assert result.manifold_zone == "syntax_floor"

    def test_structure_ripple_zone(self):
        """Should classify long-range attention as structure_ripple."""
        derotator = RoPEDerotator()

        # Long-range attention
        result = derotator.analyze(
            query_pos=1000,
            key_positions=[10, 100, 200, 999],  # Most > 256 positions away
            attention_scores=[0.3, 0.3, 0.3, 0.1]
        )

        assert result.manifold_zone == "structure_ripple"

    def test_to_dict_serialization(self):
        """Result should be JSON-serializable via to_dict."""
        derotator = RoPEDerotator()

        result = derotator.analyze(
            query_pos=100,
            key_positions=[50, 90, 95, 98],
            attention_scores=[0.1, 0.15, 0.25, 0.5]
        )

        d = result.to_dict()

        assert isinstance(d, dict)
        assert "raw_scores" in d
        assert "semantic_scores" in d
        assert isinstance(d["raw_scores"], list)  # Should be list, not ndarray


# =============================================================================
# GLOSSARY API TESTS
# =============================================================================

class TestGlossaryAPI:
    """Test educational glossary API."""

    def test_get_glossary_returns_dict(self):
        """get_glossary should return full glossary."""
        glossary = get_glossary()

        assert isinstance(glossary, dict)
        assert len(glossary) > 0
        assert "rope" in glossary
        assert "de_rotation" in glossary

    def test_glossary_entry_structure(self):
        """Each glossary entry should have required fields."""
        glossary = get_glossary()

        for key, entry in glossary.items():
            assert "term" in entry, f"Missing 'term' in {key}"
            assert "simple" in entry, f"Missing 'simple' in {key}"

    def test_get_term_explanation_simple(self):
        """get_term_explanation should return simple explanations."""
        explanation = get_term_explanation("rope", "simple")

        assert explanation is not None
        assert len(explanation) > 0
        assert "position" in explanation.lower() or "where" in explanation.lower()

    def test_get_term_explanation_levels(self):
        """Should support multiple detail levels."""
        simple = get_term_explanation("rope", "simple")
        detailed = get_term_explanation("rope", "detailed")
        analogy = get_term_explanation("rope", "analogy")

        assert simple != detailed  # Different levels should differ
        assert analogy is not None

    def test_get_term_explanation_unknown(self):
        """Unknown terms should return None."""
        result = get_term_explanation("nonexistent_term", "simple")
        assert result is None

    def test_manifold_zone_has_subzones(self):
        """manifold_zone entry should have zone descriptions."""
        glossary = get_glossary()
        zones = glossary.get("manifold_zone", {}).get("zones", {})

        assert "syntax_floor" in zones
        assert "semantic_bridge" in zones
        assert "structure_ripple" in zones


# =============================================================================
# INTERPRETATION TESTS
# =============================================================================

class TestInterpretation:
    """Test interpretation templates."""

    def test_all_patterns_have_templates(self):
        """All expected patterns should have interpretation templates."""
        expected_patterns = [
            "high_semantic_low_positional",
            "high_positional_low_semantic",
            "high_both",
            "low_both",
            "sink_dominated"
        ]

        for pattern in expected_patterns:
            assert pattern in INTERPRETATION_TEMPLATES
            template = INTERPRETATION_TEMPLATES[pattern]
            assert "meaning" in template
            assert "interpretation" in template

    def test_get_interpretation_for_pattern(self):
        """get_interpretation_for_pattern should return template."""
        interp = get_interpretation_for_pattern("high_semantic_low_positional")

        assert interp is not None
        assert "meaning" in interp
        assert "examples" in interp

    def test_template_has_examples(self):
        """Each template should have examples."""
        for pattern, template in INTERPRETATION_TEMPLATES.items():
            assert "examples" in template
            assert len(template["examples"]) > 0


# =============================================================================
# EXPLANATION GENERATION TESTS
# =============================================================================

class TestExplainAttentionStep:
    """Test natural language explanation generation."""

    def test_generates_explanation(self):
        """explain_attention_step should generate explanation."""
        derotator = RoPEDerotator()

        analysis = derotator.analyze(
            query_pos=100,
            key_positions=[50, 90, 95, 98],
            attention_scores=[0.1, 0.15, 0.25, 0.5]
        )

        explanation = explain_attention_step(
            query_token="therefore",
            key_tokens=["hypothesis", "shows", "that", "the"],
            attention_scores=[0.1, 0.15, 0.25, 0.5],
            analysis=analysis
        )

        assert isinstance(explanation, str)
        assert "therefore" in explanation
        assert len(explanation) > 50

    def test_explanation_mentions_zone(self):
        """Explanation should mention the manifold zone."""
        derotator = RoPEDerotator()

        analysis = derotator.analyze(
            query_pos=100,
            key_positions=[97, 98, 99],
            attention_scores=[0.2, 0.3, 0.5]
        )

        explanation = explain_attention_step(
            query_token="running",
            key_tokens=["the", "dog", "is"],
            attention_scores=[0.2, 0.3, 0.5],
            analysis=analysis
        )

        assert analysis.manifold_zone in explanation

    def test_explanation_identifies_attention_type(self):
        """Explanation should identify semantic vs positional attention."""
        derotator = RoPEDerotator()

        # Long-range attention
        analysis = derotator.analyze(
            query_pos=500,
            key_positions=[10, 20, 30],
            attention_scores=[0.5, 0.3, 0.2]
        )

        explanation = explain_attention_step(
            query_token="conclusion",
            key_tokens=["first", "hypothesis", "evidence"],
            attention_scores=[0.5, 0.3, 0.2],
            analysis=analysis
        )

        # Should mention semantic or positional
        assert "SEMANTIC" in explanation or "POSITIONAL" in explanation or "BALANCED" in explanation


# =============================================================================
# ATTENTION MODE ENUM TESTS
# =============================================================================

class TestAttentionMode:
    """Test AttentionMode enum."""

    def test_all_modes_defined(self):
        """All expected modes should be defined."""
        assert AttentionMode.RAW.value == "raw"
        assert AttentionMode.DEROTATED.value == "derotated"
        assert AttentionMode.POSITIONAL.value == "positional"
        assert AttentionMode.SINQ_ANCHORED.value == "sinq_anchored"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token(self):
        """Should handle single token attention."""
        derotator = RoPEDerotator()

        result = derotator.analyze(
            query_pos=100,
            key_positions=[50],
            attention_scores=[1.0]
        )

        assert result.semantic_scores.sum() == pytest.approx(1.0, abs=1e-5)

    def test_zero_scores(self):
        """Should handle zero attention scores gracefully."""
        derotator = RoPEDerotator()

        # This might produce NaN without proper handling
        result = derotator.analyze(
            query_pos=100,
            key_positions=[50, 60, 70],
            attention_scores=[0.0, 0.0, 0.0]
        )

        # Should not crash, should have valid zone
        assert result.manifold_zone is not None

    def test_very_long_range(self):
        """Should handle attention at max position."""
        config = RoPEConfig(max_position=8192)
        derotator = RoPEDerotator(config=config)

        result = derotator.analyze(
            query_pos=8000,
            key_positions=[10, 100, 7999],
            attention_scores=[0.3, 0.3, 0.4]
        )

        assert result.manifold_zone is not None

    def test_all_sink_tokens(self):
        """Should handle attention only to sink tokens."""
        derotator = RoPEDerotator()

        result = derotator.analyze(
            query_pos=100,
            key_positions=[0, 1, 2, 3, 4],
            attention_scores=[0.5, 0.2, 0.15, 0.1, 0.05]
        )

        # The pattern field contains human-readable description
        assert "sink" in result.interpretation.get("pattern", "").lower()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the full analysis pipeline."""

    def test_demo_runs_without_error(self):
        """The demo function should complete without error."""
        from .rope_derotation import demo

        # Should not raise
        demo()

    def test_full_pipeline_local_syntax(self):
        """Full pipeline for local syntax pattern."""
        derotator = RoPEDerotator()

        # Local syntax: "The dog is running"
        result = derotator.analyze(
            query_pos=10,  # "running"
            key_positions=[7, 8, 9],  # "The", "dog", "is"
            attention_scores=[0.1, 0.3, 0.6]  # High to "is"
        )

        # Verify local pattern detection
        assert result.manifold_zone == "syntax_floor"

        # Generate explanation
        explanation = explain_attention_step(
            query_token="running",
            key_tokens=["The", "dog", "is"],
            attention_scores=[0.1, 0.3, 0.6],
            analysis=result
        )

        # Explanation should be coherent
        assert "running" in explanation
        assert len(explanation.split("\n")) >= 3

    def test_full_pipeline_long_range_reasoning(self):
        """Full pipeline for long-range reasoning pattern."""
        derotator = RoPEDerotator()

        # Long-range: conclusion referring to hypothesis
        result = derotator.analyze(
            query_pos=500,  # "therefore"
            key_positions=[10, 50, 200, 498],  # "The", "hypothesis", "evidence", "thus"
            attention_scores=[0.1, 0.4, 0.35, 0.15]
        )

        # Verify long-range detection
        assert result.manifold_zone == "structure_ripple"

        # Serialize and verify
        d = result.to_dict()
        assert d["manifold_zone"] == "structure_ripple"
        assert sum(d["semantic_scores"]) == pytest.approx(1.0, abs=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
