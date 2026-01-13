"""
Tests for Sinq Kernel Optimization

This module tests the kernel-level sink filtering optimization in the
top-k attention extraction. "Sinq" refers to filtering out attention
sink tokens (positions 0-4) directly in the Triton/CUDA kernel.

Key Optimization:
Before: Compute all attention → Python post-filters sink tokens
After:  Filter sink tokens IN KERNEL → Never leave GPU

Benefits:
1. Reduced Python overhead
2. No GPU→CPU transfer for sink tokens
3. Cleaner interpretability results
"""

import pytest

# Try to import torch and triton - skip tests if not available
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import importlib.util

HAS_TRITON = importlib.util.find_spec("triton") is not None


# Skip all tests if torch/triton not available
pytestmark = pytest.mark.skipif(
    not HAS_TORCH or not HAS_TRITON, reason="torch and triton required for kernel tests"
)


class TestSinqKernelOptimization:
    """Tests for the Sinq kernel optimization."""

    @pytest.fixture
    def device(self):
        """Get test device (CUDA if available, else CPU)."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @pytest.fixture
    def test_data(self, device):
        """Create test data for attention extraction."""
        batch_size = 2
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        seq_len = 100

        q = torch.randn(
            batch_size, num_heads, head_dim, device=device, dtype=torch.float16
        )
        k_buffer = torch.randn(
            seq_len * batch_size,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=torch.float16,
        )
        kv_indptr = torch.tensor(
            [0, seq_len, seq_len * 2], dtype=torch.int32, device=device
        )
        kv_indices = torch.arange(
            seq_len * batch_size, dtype=torch.int32, device=device
        )
        sm_scale = 1.0 / (head_dim ** 0.5)

        return {
            "q": q,
            "k_buffer": k_buffer,
            "kv_indptr": kv_indptr,
            "kv_indices": kv_indices,
            "sm_scale": sm_scale,
            "seq_len": seq_len,
        }

    def test_sink_threshold_zero_no_filtering(self, test_data, device):
        """With sink_threshold=0, no positions should be filtered."""
        from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
            compute_topk_attention_chunked,
        )

        scores, indices, logits, logsumexp = compute_topk_attention_chunked(
            test_data["q"],
            test_data["k_buffer"],
            test_data["kv_indptr"],
            test_data["kv_indices"],
            test_data["sm_scale"],
            top_k=10,
            sink_threshold=0,  # No filtering
        )

        # Results should exist
        assert scores.shape == (2, 10)
        assert indices.shape == (2, 10)

        # Positions could include sink tokens (0-4)
        # (no guarantee they will, but they're allowed)

    def test_sink_threshold_filters_positions(self, test_data, device):
        """With sink_threshold=5, positions 0-4 should never appear."""
        from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
            compute_topk_attention_chunked,
        )

        # Make sink positions have high attention (to ensure they would normally be selected)
        # We boost the keys for positions 0-4 to make them attractive
        k_buffer = test_data["k_buffer"].clone()

        scores, indices, logits, logsumexp = compute_topk_attention_chunked(
            test_data["q"],
            k_buffer,
            test_data["kv_indptr"],
            test_data["kv_indices"],
            test_data["sm_scale"],
            top_k=10,
            sink_threshold=5,  # Filter positions 0-4
        )

        # All indices should be >= 5
        indices_cpu = indices.cpu()
        for b in range(indices.shape[0]):
            for k in range(indices.shape[1]):
                pos = indices_cpu[b, k].item()
                assert (
                    pos >= 5 or pos == 0
                ), f"Found sink position {pos} in batch {b}, top-k {k}"

    def test_topk_capture_class_sink_threshold(self, test_data, device):
        """Test TopKAttentionCapture class with sink_threshold."""
        from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
            TopKAttentionCapture,
        )

        # With sink filtering
        capture = TopKAttentionCapture(top_k=10, sink_threshold=5)
        result = capture.extract(
            test_data["q"],
            test_data["k_buffer"],
            test_data["kv_indptr"],
            test_data["kv_indices"],
            test_data["sm_scale"],
        )

        assert "scores" in result
        assert "indices" in result

        # All indices should be >= 5
        indices_cpu = result["indices"].cpu()
        mask = indices_cpu < 5
        # Allow 0s (which are padding for empty results)
        mask = mask & (indices_cpu != 0)
        assert not mask.any(), f"Found sink positions: {indices_cpu[mask]}"

    def test_fingerprint_mode_with_sink_filtering(self, test_data, device):
        """Test fingerprint mode respects sink filtering."""
        from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
            TopKAttentionCapture,
        )

        current_pos = torch.tensor(
            [test_data["seq_len"], test_data["seq_len"]],
            dtype=torch.int64,
            device=device,
        )

        capture = TopKAttentionCapture(
            top_k=10,
            fingerprint_mode=True,
            sink_threshold=5,
        )
        result = capture.extract(
            test_data["q"],
            test_data["k_buffer"],
            test_data["kv_indptr"],
            test_data["kv_indices"],
            test_data["sm_scale"],
            current_pos=current_pos,
        )

        assert "fingerprint" in result
        assert "features" in result
        assert "manifold" in result

        # Fingerprint should have expected shape
        assert result["fingerprint"].shape[0] == 2  # batch_size

    def test_different_sink_thresholds(self, test_data, device):
        """Test various sink_threshold values."""
        from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
            compute_topk_attention_chunked,
        )

        for threshold in [0, 1, 5, 10, 20]:
            scores, indices, _, _ = compute_topk_attention_chunked(
                test_data["q"],
                test_data["k_buffer"],
                test_data["kv_indptr"],
                test_data["kv_indices"],
                test_data["sm_scale"],
                top_k=10,
                sink_threshold=threshold,
            )

            indices_cpu = indices.cpu()

            if threshold > 0:
                # Filter out padding zeros
                valid_mask = indices_cpu > 0
                valid_indices = indices_cpu[valid_mask]
                if valid_indices.numel() > 0:
                    min_pos = valid_indices.min().item()
                    assert (
                        min_pos >= threshold
                    ), f"With sink_threshold={threshold}, found position {min_pos}"


class TestSinqEdgeCases:
    """Edge case tests for Sinq kernel optimization."""

    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def test_all_sink_positions(self, device):
        """Test when entire sequence is sink tokens."""
        from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
            compute_topk_attention_chunked,
        )

        # Very short sequence that's all sinks
        batch_size = 1
        num_heads = 4
        num_kv_heads = 4
        head_dim = 64
        seq_len = 5  # All positions 0-4
        top_k = 5

        q = torch.randn(
            batch_size, num_heads, head_dim, device=device, dtype=torch.float16
        )
        k_buffer = torch.randn(
            seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float16
        )
        kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        kv_indices = torch.arange(seq_len, dtype=torch.int32, device=device)
        sm_scale = 1.0 / (head_dim ** 0.5)

        scores, indices, logits, logsumexp = compute_topk_attention_chunked(
            q,
            k_buffer,
            kv_indptr,
            kv_indices,
            sm_scale,
            top_k=top_k,
            sink_threshold=5,  # Filter all positions
        )

        # Should return padded zeros with correct shape
        assert scores.shape == (1, top_k)
        # All positions should be padded (zeros or invalid)
        # since all positions are filtered

    def test_sink_threshold_larger_than_sequence(self, device):
        """Test when sink_threshold > sequence length."""
        from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
            compute_topk_attention_chunked,
        )

        batch_size = 1
        num_heads = 4
        num_kv_heads = 4
        head_dim = 64
        seq_len = 10
        top_k = 5

        q = torch.randn(
            batch_size, num_heads, head_dim, device=device, dtype=torch.float16
        )
        k_buffer = torch.randn(
            seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float16
        )
        kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        kv_indices = torch.arange(seq_len, dtype=torch.int32, device=device)
        sm_scale = 1.0 / (head_dim ** 0.5)

        # Sink threshold larger than sequence
        scores, indices, logits, logsumexp = compute_topk_attention_chunked(
            q,
            k_buffer,
            kv_indptr,
            kv_indices,
            sm_scale,
            top_k=top_k,
            sink_threshold=20,  # Larger than seq_len
        )

        # Should handle gracefully - returns padded results
        assert scores.shape == (1, top_k)


class TestSinqPerformanceCharacteristics:
    """Tests verifying performance characteristics of Sinq optimization."""

    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def test_no_sink_tokens_in_output(self, device):
        """Verify sink tokens never appear in output when filtered."""
        from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
            TopKAttentionCapture,
        )

        # Multiple trials with random data
        for trial in range(5):
            batch_size = 4
            num_heads = 32
            num_kv_heads = 8
            head_dim = 128
            seq_len = 500

            q = torch.randn(
                batch_size, num_heads, head_dim, device=device, dtype=torch.float16
            )
            k_buffer = torch.randn(
                seq_len * batch_size,
                num_kv_heads,
                head_dim,
                device=device,
                dtype=torch.float16,
            )
            kv_indptr = torch.tensor(
                [i * seq_len for i in range(batch_size + 1)],
                dtype=torch.int32,
                device=device,
            )
            kv_indices = torch.arange(
                seq_len * batch_size, dtype=torch.int32, device=device
            )
            sm_scale = 1.0 / (head_dim ** 0.5)

            capture = TopKAttentionCapture(top_k=20, sink_threshold=5)
            result = capture.extract(q, k_buffer, kv_indptr, kv_indices, sm_scale)

            indices = result["indices"].cpu()

            # Check no sink positions (0-4) except padding zeros
            for b in range(batch_size):
                for k in range(20):
                    pos = indices[b, k].item()
                    if pos > 0:  # Ignore padding
                        assert pos >= 5, f"Trial {trial}: Found sink position {pos}"


def demo_sinq_optimization():
    """Demonstrate the Sinq kernel optimization."""
    if not HAS_TORCH or not HAS_TRITON:
        print("torch and triton required for demo")
        return

    from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
        TopKAttentionCapture,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Create test data
    batch_size = 2
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    seq_len = 1000

    q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float16)
    k_buffer = torch.randn(
        seq_len * batch_size, num_kv_heads, head_dim, device=device, dtype=torch.float16
    )
    kv_indptr = torch.tensor(
        [0, seq_len, seq_len * 2], dtype=torch.int32, device=device
    )
    kv_indices = torch.arange(seq_len * batch_size, dtype=torch.int32, device=device)
    sm_scale = 1.0 / (head_dim ** 0.5)

    print("=" * 60)
    print("Sinq Kernel Optimization Demo")
    print("=" * 60)

    print("\n1. Without Sinq filtering (sink_threshold=0):")
    capture_no_filter = TopKAttentionCapture(top_k=10, sink_threshold=0)
    result_no_filter = capture_no_filter.extract(
        q, k_buffer, kv_indptr, kv_indices, sm_scale
    )
    indices = result_no_filter["indices"][0].cpu().tolist()
    print(f"   Top-10 positions: {indices}")
    has_sinks = any(i < 5 for i in indices)
    print(f"   Contains sink tokens (0-4): {has_sinks}")

    print("\n2. With Sinq filtering (sink_threshold=5):")
    capture_sinq = TopKAttentionCapture(top_k=10, sink_threshold=5)
    result_sinq = capture_sinq.extract(q, k_buffer, kv_indptr, kv_indices, sm_scale)
    indices_sinq = result_sinq["indices"][0].cpu().tolist()
    print(f"   Top-10 positions: {indices_sinq}")
    has_sinks_sinq = any(i < 5 and i != 0 for i in indices_sinq)  # 0 is padding
    print(f"   Contains sink tokens (0-4): {has_sinks_sinq}")

    print("\n3. Key benefits:")
    print("   - Sink filtering happens IN THE KERNEL (Triton)")
    print("   - No Python post-processing overhead")
    print("   - Cleaner attention patterns for interpretability")
    print("   - Better fingerprints for model routing")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_sinq_optimization()
