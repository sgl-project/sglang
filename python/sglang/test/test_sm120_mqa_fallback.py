"""
Unit tests for SM120 MQA fallback kernels.

These tests verify correctness of the PyTorch-native fallback implementations
that replace DeepGEMM's fp8_paged_mqa_logits and fp8_mqa_logits on SM120.

Run: python -m pytest python/sglang/test/test_sm120_mqa_fallback.py -v
"""
import pytest
import torch

from sglang.srt.layers.attention.nsa.sm120_mqa_fallback import (
    _dequant_fp8_with_scale_suffix,
    compute_paged_mqa_schedule_metadata,
    sm120_fp8_mqa_logits,
    sm120_fp8_paged_mqa_logits,
)


def _make_fp8_with_scale(data_f32: torch.Tensor) -> torch.Tensor:
    """Helper: pack float32 data into FP8 + appended scale suffix format.

    For testing, we use a scale of 1.0 so the FP8 values are the raw values.
    The last 4 bytes of each row store the float32 scale.
    """
    device = data_f32.device
    shape = data_f32.shape
    head_dim = shape[-1]

    # Clamp to FP8 E4M3 range
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    data_clamped = data_f32.clamp(-fp8_max, fp8_max)
    data_fp8 = data_clamped.to(torch.float8_e4m3fn)

    # Scale = 1.0 as float32 -> 4 bytes
    scale_val = torch.ones((*shape[:-1], 1), dtype=torch.float32, device=device)
    scale_bytes = scale_val.view(torch.float8_e4m3fn)  # reinterpret as 4 fp8 bytes

    # Concatenate: [data_fp8 | scale_bytes]
    result = torch.cat([data_fp8, scale_bytes], dim=-1)
    return result


class TestDequantFP8:
    def test_roundtrip(self):
        """Dequantized values should approximately match original float32."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = "cuda"
        data = torch.randn(4, 128, device=device)
        packed = _make_fp8_with_scale(data)
        recovered = _dequant_fp8_with_scale_suffix(packed.unsqueeze(-2), 128)
        recovered = recovered.squeeze(-2)
        # FP8 E4M3 has limited precision, allow some tolerance
        torch.testing.assert_close(recovered, data, atol=0.2, rtol=0.1)

    def test_scale_applied(self):
        """Non-unity scale should be applied correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = "cuda"
        head_dim = 128
        data = torch.ones(2, head_dim, device=device) * 0.5
        data_fp8 = data.to(torch.float8_e4m3fn)

        # Scale = 2.0
        scale = torch.full((2, 1), 2.0, dtype=torch.float32, device=device)
        scale_bytes = scale.view(torch.float8_e4m3fn)
        packed = torch.cat([data_fp8, scale_bytes], dim=-1)

        result = _dequant_fp8_with_scale_suffix(packed.unsqueeze(-2), head_dim)
        result = result.squeeze(-2)
        expected = data.float() * 2.0
        torch.testing.assert_close(result, expected, atol=0.1, rtol=0.05)


class TestScheduleMetadata:
    def test_returns_none(self):
        """SM120 schedule metadata is always None (scheduling handled internally)."""
        result = compute_paged_mqa_schedule_metadata(
            torch.tensor([10, 20]), block_size=64, num_sms=84
        )
        assert result is None


class TestPagedMQALogits:
    @pytest.fixture
    def setup(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = "cuda"
        batch = 2
        next_n = 1
        n_heads = 4
        head_dim = 128
        head_dim_with_sf = head_dim + 4
        block_kv = 64
        num_blocks = 8
        max_seq_len = 256

        # Create random FP8 queries
        q_raw = torch.randn(batch, next_n, n_heads, head_dim, device=device) * 0.1
        q_fp8 = _make_fp8_with_scale(q_raw)

        # Create random FP8 KV cache blocks
        kv_raw = torch.randn(num_blocks, block_kv, 1, head_dim, device=device) * 0.1
        kv_fp8 = _make_fp8_with_scale(kv_raw)

        # Head weights
        weights = torch.randn(batch, n_heads, device=device)

        # Sequence lengths
        seqlens = torch.tensor([[100], [64]], dtype=torch.int32, device=device)

        # Block tables: batch 0 uses blocks [0,1], batch 1 uses blocks [2]
        block_tables = torch.zeros(batch, 4, dtype=torch.int32, device=device)
        block_tables[0, :2] = torch.tensor([0, 1])
        block_tables[1, :1] = torch.tensor([2])

        return {
            "q_fp8": q_fp8,
            "kv_fp8": kv_fp8,
            "weights": weights,
            "seqlens": seqlens,
            "block_tables": block_tables,
            "max_seq_len": max_seq_len,
            "batch": batch,
            "next_n": next_n,
        }

    def test_output_shape(self, setup):
        logits = sm120_fp8_paged_mqa_logits(
            setup["q_fp8"],
            setup["kv_fp8"],
            setup["weights"],
            setup["seqlens"],
            setup["block_tables"],
            schedule_metadata=None,
            max_seq_len=setup["max_seq_len"],
        )
        expected_shape = (
            setup["batch"] * setup["next_n"],
            setup["max_seq_len"],
        )
        assert logits.shape == expected_shape

    def test_masked_positions_are_neginf(self, setup):
        logits = sm120_fp8_paged_mqa_logits(
            setup["q_fp8"],
            setup["kv_fp8"],
            setup["weights"],
            setup["seqlens"],
            setup["block_tables"],
            schedule_metadata=None,
            max_seq_len=setup["max_seq_len"],
        )
        # Positions beyond seq_len should be -inf
        seq_len_0 = setup["seqlens"][0, 0].item()
        assert torch.all(logits[0, seq_len_0:] == float("-inf"))

    def test_valid_positions_are_finite(self, setup):
        logits = sm120_fp8_paged_mqa_logits(
            setup["q_fp8"],
            setup["kv_fp8"],
            setup["weights"],
            setup["seqlens"],
            setup["block_tables"],
            schedule_metadata=None,
            max_seq_len=setup["max_seq_len"],
        )
        seq_len_0 = setup["seqlens"][0, 0].item()
        assert torch.all(torch.isfinite(logits[0, :seq_len_0]))

    def test_zero_seqlen(self, setup):
        """Batch element with zero seqlen should produce all -inf."""
        setup["seqlens"][1, 0] = 0
        logits = sm120_fp8_paged_mqa_logits(
            setup["q_fp8"],
            setup["kv_fp8"],
            setup["weights"],
            setup["seqlens"],
            setup["block_tables"],
            schedule_metadata=None,
            max_seq_len=setup["max_seq_len"],
        )
        assert torch.all(logits[1] == float("-inf"))


class TestContiguousMQALogits:
    @pytest.fixture
    def setup(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = "cuda"
        num_q = 4
        n_heads = 4
        head_dim = 128
        head_dim_with_sf = head_dim + 4
        num_k = 200

        # Queries with scale suffix
        q_raw = torch.randn(num_q, n_heads, head_dim, device=device) * 0.1
        q_fp8 = _make_fp8_with_scale(q_raw)

        # KV with scale suffix
        k_raw = torch.randn(num_k, head_dim, device=device) * 0.1
        k_fp8 = _make_fp8_with_scale(k_raw.unsqueeze(-2)).squeeze(-2)
        k_scale = torch.ones(num_k, device=device)

        # Weights
        weights = torch.randn(num_q, n_heads, device=device)

        # Ragged ranges
        ks = torch.tensor([0, 50, 100, 150], dtype=torch.int32, device=device)
        ke = torch.tensor([50, 100, 150, 200], dtype=torch.int32, device=device)

        return {
            "q_fp8": q_fp8,
            "kv_fp8": (k_fp8, k_scale),
            "weights": weights,
            "ks": ks,
            "ke": ke,
            "num_q": num_q,
            "num_k": num_k,
        }

    def test_output_shape(self, setup):
        logits = sm120_fp8_mqa_logits(
            setup["q_fp8"],
            setup["kv_fp8"],
            setup["weights"],
            setup["ks"],
            setup["ke"],
        )
        assert logits.shape[0] == setup["num_q"]
        assert logits.shape[1] >= setup["num_k"]

    def test_masked_outside_range(self, setup):
        logits = sm120_fp8_mqa_logits(
            setup["q_fp8"],
            setup["kv_fp8"],
            setup["weights"],
            setup["ks"],
            setup["ke"],
        )
        # For q=0: valid range [0, 50), positions [50, num_k) should be -inf
        assert torch.all(logits[0, 50:setup["num_k"]] == float("-inf"))

    def test_valid_inside_range(self, setup):
        logits = sm120_fp8_mqa_logits(
            setup["q_fp8"],
            setup["kv_fp8"],
            setup["weights"],
            setup["ks"],
            setup["ke"],
        )
        # For q=0: valid range [0, 50), should be finite
        assert torch.all(torch.isfinite(logits[0, :50]))

    def test_empty_input(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = "cuda"
        q_fp8 = torch.zeros(0, 4, 132, dtype=torch.float8_e4m3fn, device=device)
        k_fp8 = torch.zeros(10, 132, dtype=torch.float8_e4m3fn, device=device)
        k_scale = torch.ones(10, device=device)
        weights = torch.zeros(0, 4, device=device)
        ks = torch.zeros(0, dtype=torch.int32, device=device)
        ke = torch.zeros(0, dtype=torch.int32, device=device)

        logits = sm120_fp8_mqa_logits(q_fp8, (k_fp8, k_scale), weights, ks, ke)
        assert logits.shape[0] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
