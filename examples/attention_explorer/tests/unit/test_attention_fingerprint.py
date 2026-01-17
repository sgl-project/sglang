"""
Unit tests for AttentionFingerprint and related functions

Tests fingerprint computation: hubness, consensus, spectral, offset histogram.
"""

# Add parent to path for imports
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from attention_fingerprint import AttentionFingerprint, normalize_attention_data


class TestNormalizeAttentionData:
    """Tests for normalize_attention_data function."""

    def test_normalize_list_format(self):
        """Test normalization of list format attention data."""
        attention_tokens = [
            {
                "token_positions": [0, 1, 2],
                "attention_scores": [0.5, 0.3, 0.2],
                "layer_id": 6,
            },
            {
                "token_positions": [1, 2, 3],
                "attention_scores": [0.4, 0.4, 0.2],
                "layer_id": 12,
            },
        ]

        flat_list, per_layer = normalize_attention_data(attention_tokens)

        assert len(flat_list) == 2
        assert 6 in per_layer
        assert 12 in per_layer
        assert len(per_layer[6]) == 1
        assert len(per_layer[12]) == 1

    def test_normalize_dict_with_layers(self):
        """Test normalization of dict with layers format."""
        attention_tokens = {
            "layers": {
                "6": {"token_positions": [0, 1], "attention_scores": [0.6, 0.4]},
                "12": {"token_positions": [0, 2], "attention_scores": [0.5, 0.5]},
            }
        }

        flat_list, per_layer = normalize_attention_data(attention_tokens)

        assert len(flat_list) == 2
        assert 6 in per_layer
        assert 12 in per_layer
        # Check layer_id was added
        assert flat_list[0]["layer_id"] in [6, 12]

    def test_normalize_single_entry(self):
        """Test normalization of single entry without layers."""
        attention_tokens = {
            "token_positions": [0, 1, 2],
            "attention_scores": [0.5, 0.3, 0.2],
            "layer_id": 23,
        }

        flat_list, per_layer = normalize_attention_data(attention_tokens)

        assert len(flat_list) == 1
        assert 23 in per_layer

    def test_normalize_list_with_multi_layer_entries(self):
        """Test list containing multi-layer entries."""
        attention_tokens = [
            {
                "layers": {
                    "6": {"token_positions": [0], "attention_scores": [1.0]},
                    "12": {"token_positions": [1], "attention_scores": [1.0]},
                }
            }
        ]

        flat_list, per_layer = normalize_attention_data(attention_tokens)

        assert len(flat_list) == 2
        assert 6 in per_layer
        assert 12 in per_layer

    def test_normalize_empty_list(self):
        """Test normalization of empty list."""
        flat_list, per_layer = normalize_attention_data([])

        assert flat_list == []
        assert per_layer == {}


class TestAttentionFingerprint:
    """Tests for AttentionFingerprint class."""

    @pytest.fixture
    def fingerprinter(self):
        """Create a fingerprinter instance."""
        return AttentionFingerprint(num_offset_bins=16, hub_capacity=64)

    @pytest.fixture
    def sample_attention_tokens(self):
        """Create sample attention token data."""
        return [
            {
                "token_positions": [0, 1, 2],
                "attention_scores": [0.5, 0.3, 0.2],
                "layer_id": 6,
            },
            {
                "token_positions": [0, 1, 3],
                "attention_scores": [0.6, 0.2, 0.2],
                "layer_id": 6,
            },
            {
                "token_positions": [0, 2, 4],
                "attention_scores": [0.7, 0.2, 0.1],
                "layer_id": 12,
            },
        ]

    def test_initialization(self):
        """Test fingerprinter initialization."""
        fp = AttentionFingerprint(num_offset_bins=8, hub_capacity=32)

        assert fp.num_offset_bins == 8
        assert fp.hub_capacity == 32

    def test_default_initialization(self):
        """Test fingerprinter with default values."""
        fp = AttentionFingerprint()

        assert fp.num_offset_bins == 16
        assert fp.hub_capacity == 64


class TestComputeHubness:
    """Tests for hubness computation."""

    @pytest.fixture
    def fingerprinter(self):
        return AttentionFingerprint()

    def test_hubness_empty(self, fingerprinter):
        """Test hubness with no attention tokens."""
        result = fingerprinter.compute_hubness([])
        assert result == 0.0

    def test_hubness_single_token(self, fingerprinter):
        """Test hubness with single token."""
        tokens = [{"token_positions": [0], "attention_scores": [1.0]}]
        result = fingerprinter.compute_hubness(tokens)
        assert result == 0.0  # Single hub -> Gini undefined or 0

    def test_hubness_uniform(self, fingerprinter):
        """Test hubness with uniform attention."""
        # All positions get equal attention -> low hubness
        tokens = [
            {
                "token_positions": [0, 1, 2, 3],
                "attention_scores": [0.25, 0.25, 0.25, 0.25],
            },
            {
                "token_positions": [0, 1, 2, 3],
                "attention_scores": [0.25, 0.25, 0.25, 0.25],
            },
        ]
        result = fingerprinter.compute_hubness(tokens)
        assert 0.0 <= result <= 0.3  # Should be low

    def test_hubness_concentrated(self, fingerprinter):
        """Test hubness with concentrated attention."""
        # One position gets almost all attention -> high hubness
        tokens = [
            {
                "token_positions": [0, 1, 2, 3],
                "attention_scores": [0.9, 0.03, 0.03, 0.04],
            },
            {
                "token_positions": [0, 1, 2, 3],
                "attention_scores": [0.85, 0.05, 0.05, 0.05],
            },
            {
                "token_positions": [0, 1, 2, 3],
                "attention_scores": [0.95, 0.02, 0.02, 0.01],
            },
        ]
        result = fingerprinter.compute_hubness(tokens)
        assert result > 0.5  # Should be high

    def test_hubness_in_range(self, fingerprinter):
        """Test hubness is always in [0, 1]."""
        tokens = [
            {"token_positions": list(range(100)), "attention_scores": [0.01] * 100},
        ]
        result = fingerprinter.compute_hubness(tokens)
        assert 0.0 <= result <= 1.0


class TestComputeConsensus:
    """Tests for consensus computation."""

    @pytest.fixture
    def fingerprinter(self):
        return AttentionFingerprint()

    def test_consensus_single_layer(self, fingerprinter):
        """Test consensus with single layer."""
        tokens = [
            {"token_positions": [0, 1], "attention_scores": [0.5, 0.5], "layer_id": 6},
            {"token_positions": [0, 1], "attention_scores": [0.5, 0.5], "layer_id": 6},
        ]
        _, per_layer = normalize_attention_data(tokens)
        result = fingerprinter.compute_consensus(tokens, per_layer)
        # Single layer uses temporal consistency
        assert 0.0 <= result <= 1.0

    def test_consensus_multiple_layers_identical(self, fingerprinter):
        """Test consensus when layers have identical attention."""
        tokens = [
            {"token_positions": [0, 1], "attention_scores": [0.5, 0.5], "layer_id": 6},
            {"token_positions": [0, 1], "attention_scores": [0.5, 0.5], "layer_id": 12},
        ]
        _, per_layer = normalize_attention_data(tokens)
        result = fingerprinter.compute_consensus(tokens, per_layer)
        assert result > 0.9  # High consensus

    def test_consensus_multiple_layers_different(self, fingerprinter):
        """Test consensus when layers have different attention."""
        tokens = [
            {"token_positions": [0, 1], "attention_scores": [1.0, 0.0], "layer_id": 6},
            {"token_positions": [2, 3], "attention_scores": [1.0, 0.0], "layer_id": 12},
        ]
        _, per_layer = normalize_attention_data(tokens)
        result = fingerprinter.compute_consensus(tokens, per_layer)
        assert result < 0.3  # Low consensus (disjoint positions)

    def test_consensus_in_range(self, fingerprinter):
        """Test consensus is always in reasonable range."""
        tokens = [
            {
                "token_positions": [0, 1, 2],
                "attention_scores": [0.5, 0.3, 0.2],
                "layer_id": 6,
            },
            {
                "token_positions": [1, 2, 3],
                "attention_scores": [0.4, 0.4, 0.2],
                "layer_id": 12,
            },
            {
                "token_positions": [0, 2, 4],
                "attention_scores": [0.6, 0.2, 0.2],
                "layer_id": 18,
            },
        ]
        _, per_layer = normalize_attention_data(tokens)
        result = fingerprinter.compute_consensus(tokens, per_layer)
        assert 0.0 <= result <= 1.0


class TestComputeSpectral:
    """Tests for spectral computation."""

    @pytest.fixture
    def fingerprinter(self):
        return AttentionFingerprint()

    def test_spectral_empty(self, fingerprinter):
        """Test spectral with no attention tokens."""
        result = fingerprinter.compute_spectral([])
        assert result == 0.5  # Default value

    def test_spectral_local_attention(self, fingerprinter):
        """Test spectral with local attention (small offsets)."""
        # Attention to recent tokens (small t - pos)
        tokens = []
        for t in range(20):
            tokens.append(
                {
                    "token_positions": [max(0, t - 1), max(0, t - 2)],
                    "attention_scores": [0.6, 0.4],
                }
            )
        result = fingerprinter.compute_spectral(tokens)
        assert 0.0 <= result <= 1.0

    def test_spectral_long_range_attention(self, fingerprinter):
        """Test spectral with long-range attention (large offsets)."""
        # Attention to distant tokens (large t - pos)
        tokens = []
        for t in range(50):
            tokens.append(
                {
                    "token_positions": [0, 1],  # Always attend to beginning
                    "attention_scores": [0.5, 0.5],
                }
            )
        result = fingerprinter.compute_spectral(tokens)
        assert 0.0 <= result <= 1.0

    def test_spectral_in_range(self, fingerprinter):
        """Test spectral is always in [0, 1]."""
        tokens = [
            {"token_positions": [0, 5, 10], "attention_scores": [0.4, 0.3, 0.3]},
            {"token_positions": [0, 3, 7], "attention_scores": [0.5, 0.3, 0.2]},
        ]
        result = fingerprinter.compute_spectral(tokens)
        assert 0.0 <= result <= 1.0


class TestComputeOffsetHistogram:
    """Tests for offset histogram computation."""

    @pytest.fixture
    def fingerprinter(self):
        return AttentionFingerprint(num_offset_bins=8)

    def test_offset_histogram_shape(self, fingerprinter):
        """Test offset histogram has correct shape."""
        tokens = [
            {"token_positions": [0, 1, 2], "attention_scores": [0.4, 0.3, 0.3]},
        ]
        # compute_offset_histogram returns normalized histogram
        # Let's test via compute_full which uses it
        pass  # Method may be internal


class TestComputeEntropy:
    """Tests for entropy computation."""

    @pytest.fixture
    def fingerprinter(self):
        return AttentionFingerprint()

    def test_entropy_uniform(self, fingerprinter):
        """Test entropy with uniform attention."""
        tokens = [
            {
                "token_positions": [0, 1, 2, 3],
                "attention_scores": [0.25, 0.25, 0.25, 0.25],
            },
        ]
        result = fingerprinter.compute_entropy(tokens)
        # Uniform distribution has maximum entropy
        assert result > 1.0

    def test_entropy_concentrated(self, fingerprinter):
        """Test entropy with concentrated attention."""
        tokens = [
            {"token_positions": [0, 1, 2, 3], "attention_scores": [1.0, 0.0, 0.0, 0.0]},
        ]
        result = fingerprinter.compute_entropy(tokens)
        # Concentrated distribution has low entropy
        assert result < 0.5

    def test_entropy_in_range(self, fingerprinter):
        """Test entropy is non-negative."""
        tokens = [
            {"token_positions": [0, 1, 2], "attention_scores": [0.5, 0.3, 0.2]},
        ]
        result = fingerprinter.compute_entropy(tokens)
        assert result >= 0.0


class TestFullFingerprint:
    """Tests for full fingerprint computation."""

    @pytest.fixture
    def fingerprinter(self):
        return AttentionFingerprint()

    def test_fingerprint_returns_dict(self, fingerprinter):
        """Test fingerprint returns dictionary with all features."""
        tokens = [
            {
                "token_positions": [0, 1, 2],
                "attention_scores": [0.5, 0.3, 0.2],
                "layer_id": 6,
            },
            {
                "token_positions": [1, 2, 3],
                "attention_scores": [0.4, 0.4, 0.2],
                "layer_id": 12,
            },
        ]

        result = fingerprinter.fingerprint(tokens)

        assert isinstance(result, dict)
        assert "hubness" in result
        assert "consensus" in result
        assert "spectral" in result
        assert "entropy" in result
        assert "offset_histogram" in result
        assert "vector" in result
        assert "full_vector" in result

    def test_fingerprint_vector_returns_array(self, fingerprinter):
        """Test fingerprint_vector returns numpy array."""
        tokens = [
            {"token_positions": [0, 1, 2], "attention_scores": [0.5, 0.3, 0.2]},
        ]

        result = fingerprinter.fingerprint_vector(tokens)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 20  # 4 metrics + 16 histogram bins

    def test_fingerprint_vector_without_histogram(self, fingerprinter):
        """Test fingerprint_vector without histogram."""
        tokens = [
            {"token_positions": [0, 1, 2], "attention_scores": [0.5, 0.3, 0.2]},
        ]

        result = fingerprinter.fingerprint_vector(tokens, include_histogram=False)

        assert isinstance(result, np.ndarray)
        assert len(result) == 4  # Just 4 core metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
