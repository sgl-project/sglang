"""Tests for memory_utils module."""

import pytest
import torch

from sglang.utils.memory_utils import (
    GPUMemoryTracker,
    track_memory,
    get_available_gpu_memory,
    get_gpu_memory_utilization,
    clear_gpu_memory_cache,
    estimate_model_memory_requirements,
)


class TestGPUMemoryTracker:
    """Tests for GPUMemoryTracker class."""
    
    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker for each test."""
        return GPUMemoryTracker()
    
    def test_initial_state(self, tracker):
        """Test that tracker initializes correctly."""
        assert not tracker._tracking
    
    def test_start_stop_tracking(self, tracker):
        """Test basic start/stop tracking cycle."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        tracker.start_tracking()
        assert tracker._tracking
        stats = tracker.stop_tracking()
        assert isinstance(stats, dict)
        assert not tracker._tracking
    
    def test_get_stats_requires_tracking(self, tracker):
        """Test that get_stats returns empty when not tracking."""
        stats = tracker.get_stats()
        assert stats == {}
    
    def test_get_summary_format(self, tracker):
        """Test summary format when not tracking."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        summary = tracker.get_summary()
        assert "CUDA not available" in summary or "Not currently tracking" in summary


class TestTrackMemoryContextManager:
    """Tests for track_memory context manager."""
    
    def test_basic_usage(self):
        """Test basic context manager usage."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        with track_memory(verbose=False) as tracker:
            assert isinstance(tracker, GPUMemoryTracker)
    
    def test_named_section(self):
        """Test tracking with a name."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        with track_memory("test_section", verbose=False) as tracker:
            pass
        
        stats = tracker.get_stats()
        assert isinstance(stats, dict)


class TestMemoryHelperFunctions:
    """Tests for memory helper functions."""
    
    def test_get_available_gpu_memory_unavailable(self):
        """Test when CUDA is not available."""
        if not torch.cuda.is_available():
            result = get_available_gpu_memory()
            assert result == -1.0
    
    def test_get_available_gpu_memory_available(self):
        """Test when CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        result = get_available_gpu_memory(0)
        assert isinstance(result, float)
        assert result > 0
    
    def test_get_gpu_memory_utilization_unavailable(self):
        """Test utilization when CUDA is not available."""
        if not torch.cuda.is_available():
            result = get_gpu_memory_utilization()
            assert result == -1.0
    
    def test_get_gpu_memory_utilization_available(self):
        """Test utilization when CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        result = get_gpu_memory_utilization(0)
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0
    
    def test_clear_gpu_memory_cache(self):
        """Test cache clearing function."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Should not raise any exceptions
        clear_gpu_memory_cache()
        clear_gpu_memory_cache(0)


class TestEstimateModelMemoryRequirements:
    """Tests for memory estimation function."""
    
    def test_basic_estimation(self):
        """Test basic memory estimation."""
        estimates = estimate_model_memory_requirements(
            num_parameters=7_000_000_000,
            precision="bf16",
        )
        
        assert "weights_gb" in estimates
        assert estimates["weights_gb"] > 0
        assert estimates["weights_gb"] < 100  # Should be reasonable
    
    def test_different_precisions(self):
        """Test estimation with different precisions."""
        for precision in ["fp32", "fp16", "bf16", "int8"]:
            estimates = estimate_model_memory_requirements(
                num_parameters=7_000_000_000,
                precision=precision,
            )
            assert estimates["weights_gb"] > 0
    
    def test_with_activation_estimate(self):
        """Test estimation including activations."""
        estimates = estimate_model_memory_requirements(
            num_parameters=7_000_000_000,
            precision="bf16",
            num_layers=32,
        )
        
        assert "activations_gb" in estimates
        assert "total_estimate_gb" in estimates
    
    def test_with_kv_cache_estimate(self):
        """Test estimation including KV cache."""
        estimates = estimate_model_memory_requirements(
            num_parameters=7_000_000_000,
            precision="bf16",
            num_layers=32,
            num_heads=32,
            head_dim=128,
        )
        
        assert "kv_cache_per_token_gb" in estimates
    
    def test_unknown_precision_defaults_to_bf16(self):
        """Test that unknown precision uses bf16 bytes."""
        estimates = estimate_model_memory_requirements(
            num_parameters=1_000_000_000,
            precision="unknown",
        )
        assert estimates["weights_gb"] > 0
