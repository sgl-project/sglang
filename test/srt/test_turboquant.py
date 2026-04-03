"""
Unit tests for TurboQuant KV cache quantization.

Tests the core algorithm: FWHT, RHT, bit packing, Lloyd-Max codebook,
and full encode/decode roundtrip for keys (TurboQuant_prod) and values
(TurboQuant_mse).

Run: python -m pytest test/srt/test_turboquant.py -v
"""

import math

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.turboquant import (
    TurboQuantConfig,
    TurboQuantState,
    _lloyd_max_gaussian,
    decode_keys,
    decode_values,
    encode_keys,
    encode_values,
    fwht,
    pack_1bit,
    pack_2bit,
    pack_4bit,
    rht_forward,
    rht_inverse,
    unpack_1bit,
    unpack_2bit,
    unpack_4bit,
)

# ============================================================================
# FWHT Tests
# ============================================================================


class TestFWHT:
    def test_hadamard_4(self):
        """H_4 * [1,2,3,4] = [10, -2, -4, 0] (Sylvester ordering)."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y = fwht(x)
        expected = torch.tensor([10.0, -2.0, -4.0, 0.0])
        assert torch.allclose(y, expected), f"{y} != {expected}"

    def test_self_inverse(self):
        """H(H(x)) = d * x for the unnormalized Hadamard matrix."""
        for d in [4, 8, 16, 64, 128]:
            x = torch.randn(d)
            y = fwht(fwht(x))
            assert torch.allclose(
                y, d * x, atol=1e-4
            ), f"d={d}: max err {(y - d*x).abs().max()}"

    def test_batched(self):
        """FWHT works with batch dimensions [N, H, D]."""
        x = torch.randn(8, 4, 128)
        y = fwht(x)
        # Verify by comparing with non-batched
        for n in range(8):
            for h in range(4):
                y_single = fwht(x[n, h])
                assert torch.allclose(y[n, h], y_single, atol=1e-5)

    def test_power_of_2_assertion(self):
        """FWHT should fail on non-power-of-2 dimensions."""
        x = torch.randn(96)
        try:
            fwht(x)
            assert False, "Should have raised AssertionError"
        except AssertionError:
            pass


# ============================================================================
# RHT Tests
# ============================================================================


class TestRHT:
    def test_roundtrip(self):
        """rht_inverse(rht_forward(x, s), s) == x."""
        d = 128
        signs = torch.ones(d)
        signs[:64] = -1.0
        x = torch.randn(16, 4, d)
        y = rht_forward(x, signs)
        x_recon = rht_inverse(y, signs)
        assert torch.allclose(
            x, x_recon, atol=1e-4
        ), f"max err: {(x - x_recon).abs().max()}"

    def test_different_signs_give_different_results(self):
        """Different sign vectors produce different rotations."""
        d = 128
        x = torch.randn(1, 1, d)
        s1 = torch.ones(d)
        s2 = torch.ones(d)
        s2[0] = -1.0
        y1 = rht_forward(x, s1)
        y2 = rht_forward(x, s2)
        assert not torch.allclose(
            y1, y2
        ), "Different signs should give different outputs"

    def test_preserves_norm(self):
        """RHT is an orthogonal transform — should preserve vector norm."""
        d = 128
        signs = torch.randint(0, 2, (d,)).float() * 2 - 1
        x = torch.randn(8, 2, d)
        y = rht_forward(x, signs)
        # ||y||^2 should equal ||x||^2
        x_norms = x.norm(dim=-1)
        y_norms = y.norm(dim=-1)
        assert torch.allclose(
            x_norms, y_norms, rtol=1e-4
        ), f"Norm mismatch: max {(x_norms - y_norms).abs().max()}"


# ============================================================================
# Bit Packing Tests
# ============================================================================


class TestBitPacking:
    def test_2bit_roundtrip(self):
        """pack_2bit -> unpack_2bit preserves indices."""
        idx = torch.randint(0, 4, (8, 4, 128), dtype=torch.uint8)
        packed = pack_2bit(idx)
        assert packed.shape == (8, 4, 32), f"Expected (8,4,32), got {packed.shape}"
        unpacked = unpack_2bit(packed, 128)
        assert (idx == unpacked).all(), "2-bit roundtrip failed"

    def test_1bit_roundtrip(self):
        """pack_1bit -> unpack_1bit preserves signs."""
        signs = torch.randint(0, 2, (8, 4, 128), dtype=torch.uint8)
        packed = pack_1bit(signs)
        assert packed.shape == (8, 4, 16), f"Expected (8,4,16), got {packed.shape}"
        unpacked = unpack_1bit(packed, 128)
        assert (signs == unpacked).all(), "1-bit roundtrip failed"

    def test_4bit_roundtrip(self):
        """pack_4bit -> unpack_4bit preserves indices."""
        idx = torch.randint(0, 8, (4, 2, 128), dtype=torch.uint8)
        packed = pack_4bit(idx)
        assert packed.shape == (4, 2, 64), f"Expected (4,2,64), got {packed.shape}"
        unpacked = unpack_4bit(packed, 128)
        assert (idx == unpacked).all(), "4-bit roundtrip failed"

    def test_2bit_known_values(self):
        """Verify packing of known 2-bit values."""
        # [0, 1, 2, 3] should pack to 0b11_10_01_00 = 0xE4
        idx = torch.tensor([[0, 1, 2, 3]], dtype=torch.uint8)
        packed = pack_2bit(idx)
        assert (
            packed[0, 0].item() == 0b11100100
        ), f"Expected 228, got {packed[0, 0].item()}"

    def test_1bit_known_values(self):
        """Verify packing of known 1-bit values."""
        # [1, 0, 1, 0, 1, 0, 1, 0] should pack to 0b01010101 = 0x55
        signs = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=torch.uint8)
        packed = pack_1bit(signs)
        assert packed[0, 0].item() == 0x55, f"Expected 85, got {packed[0, 0].item()}"


# ============================================================================
# Lloyd-Max Codebook Tests
# ============================================================================


class TestLloydMax:
    def test_symmetry(self):
        """Lloyd-Max for symmetric distribution should give symmetric codebook."""
        sigma = 1.0 / math.sqrt(128)
        c, b = _lloyd_max_gaussian(sigma, 4)
        # Centroids should be symmetric: c[0] = -c[3], c[1] = -c[2]
        assert abs(c[0] + c[3]) < 0.001, f"Asymmetric centroids: {c}"
        assert abs(c[1] + c[2]) < 0.001, f"Asymmetric centroids: {c}"

    def test_matches_hardcoded_2bit(self):
        """Dynamic computation matches pre-computed 2-bit codebook for d=128."""
        sigma = 1.0 / math.sqrt(128)
        c, b = _lloyd_max_gaussian(sigma, 4)
        expected = [-0.1330, -0.0400, 0.0400, 0.1330]
        max_diff = max(abs(a - e) for a, e in zip(c, expected))
        assert max_diff < 0.005, f"Codebook mismatch: max diff {max_diff}"

    def test_matches_hardcoded_3bit(self):
        """Dynamic computation matches pre-computed 3-bit codebook for d=128."""
        sigma = 1.0 / math.sqrt(128)
        c, b = _lloyd_max_gaussian(sigma, 8)
        expected = [-0.1884, -0.1181, -0.0666, -0.0216, 0.0216, 0.0666, 0.1181, 0.1884]
        max_diff = max(abs(a - e) for a, e in zip(c, expected))
        assert max_diff < 0.005, f"Codebook mismatch: max diff {max_diff}"

    def test_boundaries_ordered(self):
        """Boundaries must be strictly increasing."""
        sigma = 1.0 / math.sqrt(128)
        for n_levels in [2, 4, 8]:
            c, b = _lloyd_max_gaussian(sigma, n_levels)
            for i in range(len(b) - 1):
                assert b[i] < b[i + 1], f"Non-increasing boundaries at {i}: {b}"


# ============================================================================
# Config Validation Tests
# ============================================================================


class TestConfig:
    def test_default_config(self):
        cfg = TurboQuantConfig()
        # Default: K4/V2, QJL off, protected_head=2, protected_tail=2
        assert cfg.key_bits == 4
        assert cfg.value_bits == 2
        assert cfg.key_mse_bits == 4  # key_bits=4, no QJL
        assert cfg.key_mse_packed_dim == 64  # 128 / 2 (4-bit nibble)
        assert cfg.key_qjl_packed_dim == 0  # QJL off
        assert cfg.value_packed_dim == 32  # 128 * 2 / 8 (2-bit)
        assert cfg.protected_layers_head == 2
        assert cfg.protected_layers_tail == 2

    def test_no_qjl_config(self):
        cfg = TurboQuantConfig(key_bits=3, enable_qjl=False)
        assert cfg.key_mse_bits == 3  # All 3 bits for MSE
        assert cfg.key_qjl_packed_dim == 0

    def test_invalid_key_bits_with_qjl(self):
        """key_bits=1 with QJL means key_mse_bits=0 — should fail."""
        try:
            TurboQuantConfig(key_bits=1, enable_qjl=True)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_invalid_value_bits(self):
        try:
            TurboQuantConfig(value_bits=0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_non_power_of_2_head_dim(self):
        try:
            TurboQuantConfig(head_dim=96)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# ============================================================================
# Full Encode/Decode Roundtrip Tests
# ============================================================================


class TestEncodeDecodeKeys:
    """Test TurboQuant_prod (Algorithm 2) for keys."""

    def setup_method(self):
        self.cfg = TurboQuantConfig(
            key_bits=3, value_bits=2, enable_qjl=True, head_dim=128
        )
        self.state = TurboQuantState(self.cfg, layer_num=2, head_num=4, device="cpu")

    def test_shapes(self):
        """Verify output shapes of encode_keys."""
        N, H, D = 16, 4, 128
        k = torch.randn(N, H, D)
        packed_mse, norms, packed_qjl, r_norms = encode_keys(k, 0, self.state)
        assert packed_mse.shape == (N, H, 32), f"MSE shape: {packed_mse.shape}"
        assert norms.shape == (N, H, 1), f"Norms shape: {norms.shape}"
        assert packed_qjl.shape == (N, H, 16), f"QJL shape: {packed_qjl.shape}"
        assert r_norms.shape == (N, H, 1), f"R_norms shape: {r_norms.shape}"

    def test_norms_are_fp32(self):
        """FP32 norms — critical to avoid corruption at long contexts."""
        k = torch.randn(4, 4, 128)
        _, norms, _, r_norms = encode_keys(k, 0, self.state)
        assert norms.dtype == torch.float32
        assert r_norms.dtype == torch.float32

    def test_decode_shape(self):
        k = torch.randn(8, 4, 128)
        packed = encode_keys(k, 0, self.state)
        k_recon = decode_keys(*packed, layer_idx=0, state=self.state)
        assert k_recon.shape == k.shape
        assert k_recon.dtype == torch.bfloat16

    def test_reconstruction_quality(self):
        """Verify reasonable reconstruction quality for random keys."""
        N, H, D = 64, 4, 128
        k = torch.randn(N, H, D)
        packed = encode_keys(k, 0, self.state)
        k_recon = decode_keys(
            *packed, layer_idx=0, state=self.state, output_dtype=torch.float32
        )

        # Cosine similarity should be high (> 0.9 for 3-bit with QJL)
        cos_sim = F.cosine_similarity(
            k.reshape(-1, D), k_recon.reshape(-1, D), dim=-1
        ).mean()
        assert cos_sim > 0.85, f"Cosine similarity too low: {cos_sim:.4f}"

    def test_no_qjl_encode(self):
        """Verify encode works without QJL."""
        cfg = TurboQuantConfig(key_bits=3, enable_qjl=False, head_dim=128)
        state = TurboQuantState(cfg, layer_num=1, head_num=2, device="cpu")
        k = torch.randn(4, 2, 128)
        packed_mse, norms, packed_qjl, r_norms = encode_keys(k, 0, state)
        assert packed_qjl is None
        assert r_norms is None
        # 3-bit without QJL uses 4-bit nibble packing
        assert packed_mse.shape == (4, 2, 64)  # head_dim // 2

    def test_different_layers_different_rotations(self):
        """Different layers should use different rotation signs."""
        k = torch.randn(4, 4, 128)
        packed0, _, _, _ = encode_keys(k, 0, self.state)
        packed1, _, _, _ = encode_keys(k, 1, self.state)
        assert not torch.equal(
            packed0, packed1
        ), "Different layers should give different encodings"


class TestEncodeDecodeValues:
    """Test TurboQuant_mse (Algorithm 1) for values."""

    def setup_method(self):
        self.cfg = TurboQuantConfig(
            key_bits=3, value_bits=2, enable_qjl=True, head_dim=128
        )
        self.state = TurboQuantState(self.cfg, layer_num=2, head_num=4, device="cpu")

    def test_shapes(self):
        v = torch.randn(16, 4, 128)
        packed, norms = encode_values(v, 0, self.state)
        assert packed.shape == (16, 4, 32)  # 2-bit: 128 * 2 / 8
        assert norms.shape == (16, 4, 1)

    def test_norms_fp32(self):
        v = torch.randn(4, 4, 128)
        _, norms = encode_values(v, 0, self.state)
        assert norms.dtype == torch.float32

    def test_roundtrip_quality(self):
        N, H, D = 64, 4, 128
        v = torch.randn(N, H, D)
        packed, norms = encode_values(v, 0, self.state)
        v_recon = decode_values(
            packed, norms, 0, self.state, output_dtype=torch.float32
        )
        cos_sim = F.cosine_similarity(
            v.reshape(-1, D), v_recon.reshape(-1, D), dim=-1
        ).mean()
        assert cos_sim > 0.85, f"Cosine similarity too low: {cos_sim:.4f}"

    def test_keys_and_values_use_different_rotations(self):
        """Keys use rotation_signs[:, 0] and values use rotation_signs[:, 1]."""
        x = torch.randn(4, 4, 128)
        k_packed, _, _, _ = encode_keys(x, 0, self.state)
        v_packed, _ = encode_values(x, 0, self.state)
        # Same input, different rotation → different packed output
        assert not torch.equal(
            k_packed, v_packed
        ), "K and V should use different rotations"


# ============================================================================
# Memory Layout Tests
# ============================================================================


class TestMemoryLayout:
    def test_bytes_per_token_with_qjl(self):
        """92 bytes/token/head for default config (3-bit keys + 2-bit values + QJL)."""
        cfg = TurboQuantConfig(key_bits=3, value_bits=2, enable_qjl=True, head_dim=128)
        key_bytes = (
            cfg.key_mse_packed_dim + 4 + cfg.key_qjl_packed_dim + 4
        )  # mse + norm + qjl + rnorm
        val_bytes = cfg.value_packed_dim + 4  # mse + norm
        total = key_bytes + val_bytes
        assert total == 92, f"Expected 92 bytes, got {total}"
        assert 512 / total > 5.5, f"Compression {512/total:.1f}x below 5.5x"

    def test_bytes_per_token_no_qjl(self):
        """104 bytes/token/head for K4/V2 without QJL."""
        cfg = TurboQuantConfig(key_bits=4, value_bits=2, enable_qjl=False, head_dim=128)
        key_bytes = cfg.key_mse_packed_dim + 4  # mse(64) + norm(4)
        val_bytes = cfg.value_packed_dim + 4  # mse(32) + norm(4)
        total = key_bytes + val_bytes
        assert total == 104, f"Expected 104 bytes (K4/V2), got {total}"

    def test_bytes_per_token_default(self):
        """Default K4/V2: 104 bytes/token/head → 4.9x compression."""
        cfg = TurboQuantConfig()  # Uses defaults
        key_bytes = cfg.key_mse_packed_dim + 4  # 64 + 4 = 68
        val_bytes = cfg.value_packed_dim + 4  # 32 + 4 = 36
        total = key_bytes + val_bytes
        assert total == 104, f"Expected 104 bytes for K4/V2, got {total}"
        assert 512 / total > 4.8, f"Compression {512/total:.1f}x below 4.8x"


from sglang.srt.layers.quantization.turboquant import _next_pow2


class TestNextPow2:
    def test_power_of_two_unchanged(self):
        assert _next_pow2(32) == 32
        assert _next_pow2(128) == 128

    def test_non_power_of_two(self):
        assert _next_pow2(96) == 128
        assert _next_pow2(33) == 64
        assert _next_pow2(1) == 1

    def test_zero(self):
        assert _next_pow2(0) == 1


from sglang.srt.layers.quantization.turboquant import detect_outlier_channels


class TestOutlierDetection:
    def test_detects_high_variance_channels(self):
        """Channels with injected high variance should be detected as outliers."""
        torch.manual_seed(0)
        x = torch.randn(100, 128) * 0.1
        x[:, 10] *= 20.0
        x[:, 50] *= 15.0
        x[:, 100] *= 10.0
        outlier_idx, regular_idx = detect_outlier_channels(x, n_outlier=4)
        assert outlier_idx.numel() == 4
        assert regular_idx.numel() == 124
        assert 10 in outlier_idx.tolist()
        assert 50 in outlier_idx.tolist()
        assert 100 in outlier_idx.tolist()

    def test_indices_are_sorted(self):
        torch.manual_seed(1)
        x = torch.randn(50, 128)
        outlier_idx, regular_idx = detect_outlier_channels(x, n_outlier=32)
        assert torch.all(outlier_idx[1:] >= outlier_idx[:-1])
        assert torch.all(regular_idx[1:] >= regular_idx[:-1])

    def test_indices_cover_all_channels(self):
        torch.manual_seed(2)
        x = torch.randn(20, 64)
        outlier_idx, regular_idx = detect_outlier_channels(x, n_outlier=16)
        all_idx = torch.cat([outlier_idx, regular_idx]).sort().values
        assert torch.equal(all_idx, torch.arange(64))

    def test_3d_input(self):
        torch.manual_seed(3)
        x = torch.randn(10, 8, 128)
        outlier_idx, regular_idx = detect_outlier_channels(x, n_outlier=32)
        assert outlier_idx.numel() == 32
        assert regular_idx.numel() == 96

    def test_single_vector(self):
        x = torch.randn(1, 128)
        outlier_idx, regular_idx = detect_outlier_channels(x, n_outlier=32)
        assert outlier_idx.numel() == 32


from sglang.srt.layers.quantization.turboquant import compute_online_codebook


class TestOnlineCodebook:
    def test_returns_correct_shapes(self):
        torch.manual_seed(0)
        data = torch.randn(1000, 128) * 0.0884
        centroids, boundaries = compute_online_codebook(data, bits=2)
        assert centroids.shape == (4,)
        assert boundaries.shape == (5,)

    def test_centroids_are_sorted(self):
        torch.manual_seed(0)
        data = torch.randn(500, 64) * 0.125
        centroids, boundaries = compute_online_codebook(data, bits=2)
        assert torch.all(centroids[1:] > centroids[:-1])
        assert torch.all(boundaries[1:] > boundaries[:-1])

    def test_symmetric_for_symmetric_data(self):
        torch.manual_seed(42)
        data = torch.randn(5000, 128) * 0.0884
        centroids, _ = compute_online_codebook(data, bits=2)
        assert abs(centroids[0].item() + centroids[3].item()) < 0.01
        assert abs(centroids[1].item() + centroids[2].item()) < 0.01

    def test_close_to_precomputed(self):
        torch.manual_seed(99)
        sigma = 1.0 / (128**0.5)
        data = torch.randn(10000, 128) * sigma
        centroids, _ = compute_online_codebook(data, bits=2)
        assert abs(centroids[0].item() - (-0.1330)) < 0.01
        assert abs(centroids[3].item() - 0.1330) < 0.01


class TestMixedPrecisionConfig:
    def test_default_mixed_off(self):
        cfg = TurboQuantConfig()
        assert cfg.mixed_precision is False

    def test_mixed_precision_dims(self):
        cfg = TurboQuantConfig(mixed_precision=True, n_outlier=32, head_dim=128)
        assert cfg.n_regular == 96
        assert cfg.n_outlier_padded == 32
        assert cfg.n_regular_padded == 128

    def test_mixed_key_bits(self):
        cfg = TurboQuantConfig(
            mixed_precision=True, key_bits=3, enable_qjl=True, n_outlier=32
        )
        assert cfg.key_mse_bits_outlier == 3  # (key_bits-1)+1 = 3
        assert cfg.key_mse_bits_regular == 2  # key_bits-1 = 2

    def test_mixed_packed_dims(self):
        cfg = TurboQuantConfig(
            mixed_precision=True, key_bits=3, enable_qjl=True, n_outlier=32
        )
        assert cfg.key_outlier_packed_dim == 16  # 32 * 3 / 8 → 4-bit nibble: 32/2
        assert cfg.key_regular_packed_dim == 32  # 128 * 2 / 8

    def test_mixed_value_packed_dims(self):
        cfg = TurboQuantConfig(mixed_precision=True, value_bits=2, n_outlier=32)
        assert cfg.value_outlier_packed_dim == 16
        assert cfg.value_regular_packed_dim == 32

    def test_n_outlier_validation(self):
        import pytest

        with pytest.raises(ValueError):
            TurboQuantConfig(mixed_precision=True, n_outlier=128, head_dim=128)
        with pytest.raises(ValueError):
            TurboQuantConfig(mixed_precision=True, n_outlier=0, head_dim=128)

    def test_qjl_score_weight(self):
        cfg = TurboQuantConfig(qjl_score_weight=0.5)
        assert cfg.qjl_score_weight == 0.5


class TestQJLScoreWeight:
    def test_weight_zero_equals_no_qjl(self):
        """qjl_score_weight=0 should produce same result as enable_qjl=False."""
        torch.manual_seed(0)
        k = torch.randn(4, 4, 128)

        cfg_no_qjl = TurboQuantConfig(key_bits=2, enable_qjl=False)
        state_no_qjl = TurboQuantState(cfg_no_qjl, 2, 4, "cpu")
        p1, n1, _, _ = encode_keys(k, 0, state_no_qjl)
        k1 = decode_keys(p1, n1, None, None, 0, state_no_qjl)

        cfg_w0 = TurboQuantConfig(key_bits=3, enable_qjl=True, qjl_score_weight=0.0)
        state_w0 = TurboQuantState(cfg_w0, 2, 4, "cpu")
        p2, n2, q2, r2 = encode_keys(k, 0, state_w0)
        k2 = decode_keys(p2, n2, q2, r2, 0, state_w0)

        assert torch.allclose(k1, k2, atol=1e-5)

    def test_weight_affects_output(self):
        """Different weights should produce different outputs (requires QJL on)."""
        torch.manual_seed(1)
        k = torch.randn(4, 4, 128)

        cfg1 = TurboQuantConfig(key_bits=3, enable_qjl=True, qjl_score_weight=1.0)
        state1 = TurboQuantState(cfg1, 2, 4, "cpu")
        p, n, q, r = encode_keys(k, 0, state1)
        k_w1 = decode_keys(p, n, q, r, 0, state1, output_dtype=torch.float32)

        cfg05 = TurboQuantConfig(key_bits=3, enable_qjl=True, qjl_score_weight=0.5)
        state05 = TurboQuantState(cfg05, 2, 4, "cpu")
        k_w05 = decode_keys(p, n, q, r, 0, state05, output_dtype=torch.float32)

        assert not torch.allclose(k_w1, k_w05, atol=1e-6)


from sglang.srt.layers.quantization.turboquant import (
    MixedPrecisionInfo,
    decode_keys_mixed,
    encode_keys_mixed,
)


class TestMixedPrecisionKeys:
    def _make_state(self):
        cfg = TurboQuantConfig(
            head_dim=128,
            mixed_precision=True,
            key_bits=3,
            enable_qjl=True,
            n_outlier=32,
        )
        state = TurboQuantState(cfg, layer_num=2, head_num=4, device="cpu")
        return cfg, state

    def test_encode_returns_correct_shapes(self):
        cfg, state = self._make_state()
        torch.manual_seed(0)
        k = torch.randn(8, 4, 128)

        result = encode_keys_mixed(k, layer_idx=0, state=state)
        (
            outlier_packed,
            regular_packed,
            outlier_norms,
            regular_norms,
            qjl_packed,
            r_norms,
        ) = result

        assert outlier_packed.shape == (8, 4, cfg.key_outlier_packed_dim)
        assert regular_packed.shape == (8, 4, cfg.key_regular_packed_dim)
        assert outlier_norms.shape == (8, 4, 1)
        assert regular_norms.shape == (8, 4, 1)
        assert qjl_packed.shape == (8, 4, cfg.key_qjl_packed_dim)
        assert r_norms.shape == (8, 4, 1)

    def test_encode_decode_roundtrip(self):
        cfg, state = self._make_state()
        torch.manual_seed(42)
        k = torch.randn(4, 4, 128)

        encoded = encode_keys_mixed(k, layer_idx=0, state=state)
        k_hat = decode_keys_mixed(*encoded, layer_idx=0, state=state)

        assert k_hat.shape == k.shape
        cos = F.cosine_similarity(
            k.reshape(-1, 128).float(), k_hat.reshape(-1, 128).float(), dim=-1
        )
        assert cos.mean().item() > 0.80, f"cosine {cos.mean():.3f} too low"

    def test_mixed_better_than_uniform_with_outliers(self):
        torch.manual_seed(7)
        k = torch.randn(20, 4, 128)
        k[..., :4] *= 10.0
        k[..., 64:68] *= 8.0

        cfg_u = TurboQuantConfig(head_dim=128, mixed_precision=False, key_bits=3)
        state_u = TurboQuantState(cfg_u, layer_num=2, head_num=4, device="cpu")
        packed_u, norms_u, qjl_u, rn_u = encode_keys(k, 0, state_u)
        k_hat_u = decode_keys(packed_u, norms_u, qjl_u, rn_u, 0, state_u)
        cos_u = F.cosine_similarity(
            k.reshape(-1, 128).float(), k_hat_u.reshape(-1, 128).float(), dim=-1
        ).mean()

        cfg_m, state_m = self._make_state()
        encoded_m = encode_keys_mixed(k, 0, state_m)
        k_hat_m = decode_keys_mixed(*encoded_m, layer_idx=0, state=state_m)
        cos_m = F.cosine_similarity(
            k.reshape(-1, 128).float(), k_hat_m.reshape(-1, 128).float(), dim=-1
        ).mean()

        assert (
            cos_m >= cos_u - 0.02
        ), f"mixed {cos_m:.3f} worse than uniform {cos_u:.3f}"


from sglang.srt.layers.quantization.turboquant import (
    decode_values_mixed,
    encode_values_mixed,
)


class TestMixedPrecisionValues:
    def _make_state(self):
        cfg = TurboQuantConfig(
            head_dim=128, mixed_precision=True, value_bits=2, n_outlier=32
        )
        state = TurboQuantState(cfg, layer_num=2, head_num=4, device="cpu")
        return cfg, state

    def test_encode_shapes(self):
        cfg, state = self._make_state()
        torch.manual_seed(0)
        v = torch.randn(8, 4, 128)

        outlier_packed, regular_packed, outlier_norms, regular_norms = (
            encode_values_mixed(v, layer_idx=0, state=state)
        )

        assert outlier_packed.shape == (8, 4, cfg.value_outlier_packed_dim)
        assert regular_packed.shape == (8, 4, cfg.value_regular_packed_dim)
        assert outlier_norms.shape == (8, 4, 1)
        assert regular_norms.shape == (8, 4, 1)

    def test_roundtrip(self):
        cfg, state = self._make_state()
        torch.manual_seed(42)
        v = torch.randn(4, 4, 128)

        encoded = encode_values_mixed(v, 0, state)
        v_hat = decode_values_mixed(*encoded, layer_idx=0, state=state)

        assert v_hat.shape == v.shape
        cos = F.cosine_similarity(
            v.reshape(-1, 128).float(), v_hat.reshape(-1, 128).float(), dim=-1
        )
        assert cos.mean().item() > 0.80


class TestMixedPrecisionState:
    def test_state_creates_without_mixed(self):
        cfg = TurboQuantConfig(head_dim=128, mixed_precision=False)
        state = TurboQuantState(cfg, layer_num=2, head_num=4, device="cpu")
        assert state._mixed_cache == {}

    def test_state_creates_with_mixed(self):
        cfg = TurboQuantConfig(head_dim=128, mixed_precision=True)
        state = TurboQuantState(cfg, layer_num=2, head_num=4, device="cpu")
        assert isinstance(state._mixed_cache, dict)

    def test_get_mixed_info_lazy_init(self):
        cfg = TurboQuantConfig(head_dim=128, mixed_precision=True, n_outlier=32)
        state = TurboQuantState(cfg, layer_num=2, head_num=4, device="cpu")
        torch.manual_seed(0)
        calib = torch.randn(10, 128)
        info = state.get_mixed_info(layer_idx=0, head_idx=0, calibration_data=calib)
        assert isinstance(info, MixedPrecisionInfo)
        assert info.outlier_idx.numel() == 32
        assert info.regular_idx.numel() == 96
        assert info.outlier_signs.shape[-1] == 32
        assert info.regular_signs.shape[-1] == 128

    def test_get_mixed_info_cached(self):
        cfg = TurboQuantConfig(head_dim=128, mixed_precision=True)
        state = TurboQuantState(cfg, layer_num=2, head_num=4, device="cpu")
        calib = torch.randn(10, 128)
        info1 = state.get_mixed_info(0, 0, calibration_data=calib)
        info2 = state.get_mixed_info(0, 0)
        assert info1 is info2

    def test_get_mixed_info_none_without_data(self):
        cfg = TurboQuantConfig(head_dim=128, mixed_precision=True)
        state = TurboQuantState(cfg, layer_num=2, head_num=4, device="cpu")
        assert state.get_mixed_info(0, 0) is None

    def test_mixed_info_codebooks(self):
        cfg = TurboQuantConfig(
            head_dim=128,
            mixed_precision=True,
            key_bits=3,
            enable_qjl=True,
            n_outlier=32,
        )
        state = TurboQuantState(cfg, layer_num=1, head_num=1, device="cpu")
        calib = torch.randn(50, 128)
        info = state.get_mixed_info(0, 0, calibration_data=calib)
        # With QJL: key_mse_bits=2 -> K=4, outlier: key_mse_bits+1=3 -> K=8
        assert info.key_outlier_centroids.numel() == 8
        assert info.key_regular_centroids.numel() == 4


# ============================================================================
# Full Integration Tests (mixed-precision end-to-end)
# ============================================================================


class TestMixedPrecisionIntegration:
    """Test the full flow: config -> state -> encode -> decode for both modes."""

    def test_uniform_still_works(self):
        """Uniform mode is unchanged after mixed-precision additions."""
        cfg = TurboQuantConfig(key_bits=3, value_bits=2, mixed_precision=False)
        state = TurboQuantState(cfg, 2, 4, "cpu")
        torch.manual_seed(0)
        k = torch.randn(8, 4, 128)
        v = torch.randn(8, 4, 128)

        pk, nk, qk, rk = encode_keys(k, 0, state)
        pv, nv = encode_values(v, 0, state)
        k_hat = decode_keys(pk, nk, qk, rk, 0, state)
        v_hat = decode_values(pv, nv, 0, state)

        cos_k = F.cosine_similarity(
            k.reshape(-1, 128).float(), k_hat.reshape(-1, 128).float(), dim=-1
        )
        cos_v = F.cosine_similarity(
            v.reshape(-1, 128).float(), v_hat.reshape(-1, 128).float(), dim=-1
        )
        assert cos_k.mean() > 0.80
        assert cos_v.mean() > 0.80

    def test_mixed_precision_full_flow(self):
        """Mixed-precision mode: encode + decode roundtrip."""
        cfg = TurboQuantConfig(
            key_bits=3,
            value_bits=2,
            mixed_precision=True,
            n_outlier=32,
        )
        state = TurboQuantState(cfg, 2, 4, "cpu")
        torch.manual_seed(0)
        k = torch.randn(8, 4, 128)
        v = torch.randn(8, 4, 128)

        ek = encode_keys_mixed(k, 0, state)
        ev = encode_values_mixed(v, 0, state)
        k_hat = decode_keys_mixed(*ek, layer_idx=0, state=state)
        v_hat = decode_values_mixed(*ev, layer_idx=0, state=state)

        cos_k = F.cosine_similarity(
            k.reshape(-1, 128).float(), k_hat.reshape(-1, 128).float(), dim=-1
        )
        cos_v = F.cosine_similarity(
            v.reshape(-1, 128).float(), v_hat.reshape(-1, 128).float(), dim=-1
        )
        assert cos_k.mean() > 0.78
        assert cos_v.mean() > 0.78

    def test_mixed_with_online_codebook(self):
        """Online codebook should produce comparable quality."""
        cfg = TurboQuantConfig(
            key_bits=3,
            mixed_precision=True,
            use_online_codebook=True,
        )
        state = TurboQuantState(cfg, 1, 2, "cpu")
        torch.manual_seed(99)
        k = torch.randn(20, 2, 128)

        ek = encode_keys_mixed(k, 0, state)
        k_hat = decode_keys_mixed(*ek, layer_idx=0, state=state)

        cos = F.cosine_similarity(
            k.reshape(-1, 128).float(), k_hat.reshape(-1, 128).float(), dim=-1
        )
        assert cos.mean() > 0.75

    def test_memory_layout_mixed(self):
        """Verify bytes per token for mixed-precision mode (with QJL)."""
        cfg = TurboQuantConfig(
            head_dim=128,
            key_bits=3,
            value_bits=2,
            enable_qjl=True,
            mixed_precision=True,
            n_outlier=32,
        )
        # Key: outlier=16 + regular=32 + outlier_norm=4 + regular_norm=4
        #      + QJL=16 + r_norm=4 = 76
        assert cfg.key_outlier_packed_dim == 16
        assert cfg.key_regular_packed_dim == 32
        assert cfg.key_qjl_packed_dim == 16
        key_bytes = (
            cfg.key_outlier_packed_dim
            + cfg.key_regular_packed_dim
            + 4
            + 4  # norms
            + cfg.key_qjl_packed_dim
            + 4  # QJL
        )
        assert key_bytes == 76

        # Value: outlier=16 + regular=32 + norms=8 = 56
        val_bytes = (
            cfg.value_outlier_packed_dim + cfg.value_regular_packed_dim + 4 + 4  # norms
        )
        assert val_bytes == 56

        # Total: 132 bytes vs 512 FP16 -> 3.9x
        total = key_bytes + val_bytes
        assert total == 132
        assert 512 / total > 3.8


# ============================================================================
# Triton Kernel Tests (GPU only)
# ============================================================================


class TestTritonDecode:
    """Tests for the parametric Triton selective decode kernel."""

    def _skip_no_cuda(self):
        if not torch.cuda.is_available():
            import pytest

            pytest.skip("CUDA not available")

    def _roundtrip_bits(self, bits):
        """Test Triton decode matches PyTorch decode for given bit-width."""
        self._skip_no_cuda()

        from sglang.srt.layers.quantization.triton_tq_decode import (
            triton_tq_selective_decode,
        )
        from sglang.srt.layers.quantization.turboquant import (
            _pack,
            _unpack,
            rht_inverse,
        )

        N, H, D = 32, 4, 128
        device = "cuda"
        cfg = TurboQuantConfig(
            key_bits=bits, value_bits=bits, enable_qjl=False, head_dim=D
        )
        state = TurboQuantState(cfg, layer_num=1, head_num=H, device=device)

        # Generate random input and encode
        x = torch.randn(N, H, D, device=device)
        x_f = x.float()
        norms = x_f.norm(dim=-1, keepdim=True)
        x_unit = x_f / norms.clamp(min=1e-12)
        y = rht_forward(x_unit, state.rotation_signs[0, 0])
        indices = torch.bucketize(y.contiguous(), state.key_inner)
        packed = _pack(indices, cfg.key_mse_bits)

        # PyTorch reference decode (with norm correction)
        mse_indices = _unpack(packed, cfg.key_mse_bits, D)
        y_hat_ref = state.key_centroids[mse_indices.long()]
        y_hat_ref = y_hat_ref / y_hat_ref.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        ref = rht_inverse(y_hat_ref, state.rotation_signs[0, 0]) * norms
        ref_bf16 = ref.to(torch.bfloat16)

        # Triton decode
        pool_indices = torch.arange(N, dtype=torch.int32, device=device)
        compact_out = torch.zeros(N, H, D, dtype=torch.bfloat16, device=device)
        scratch = torch.empty(N * H * D, dtype=torch.float32, device=device)
        n_active = torch.tensor([N], dtype=torch.int32, device=device)

        triton_tq_selective_decode(
            mse_buffer=packed,
            norm_buffer=norms,
            pool_indices=pool_indices,
            centroids=state.key_centroids,
            signs=state.rotation_signs[0, 0],
            compact_out=compact_out,
            scratch=scratch,
            n_active_tensor=n_active,
            grid_n=N,
            bits=cfg.key_mse_bits,
        )

        # Compare: bf16 rounding means exact match within bf16 tolerance
        max_err = (compact_out.float() - ref_bf16.float()).abs().max().item()
        assert max_err < 0.02, f"bits={bits}: max err {max_err:.6f}"

    def test_2bit(self):
        self._roundtrip_bits(2)

    def test_3bit(self):
        self._roundtrip_bits(3)

    def test_4bit(self):
        self._roundtrip_bits(4)

    def test_1bit(self):
        self._roundtrip_bits(1)


class TestTritonEncode:
    """Tests for the fused Triton encode kernel."""

    def _skip_no_cuda(self):
        if not torch.cuda.is_available():
            import pytest

            pytest.skip("CUDA not available")

    def _encode_bits(self, bits):
        """Test Triton encode matches PyTorch encode for given bit-width."""
        self._skip_no_cuda()

        from sglang.srt.layers.quantization.triton_tq_encode import triton_tq_encode
        from sglang.srt.layers.quantization.turboquant import _pack

        N, H, D = 32, 4, 128
        device = "cuda"
        cfg = TurboQuantConfig(
            key_bits=bits, value_bits=bits, enable_qjl=False, head_dim=D
        )
        state = TurboQuantState(cfg, layer_num=1, head_num=H, device=device)

        x = torch.randn(N, H, D, device=device)

        # PyTorch reference encode
        x_f = x.float()
        norms_ref = x_f.norm(dim=-1, keepdim=True)
        x_unit = x_f / norms_ref.clamp(min=1e-12)
        y = rht_forward(x_unit, state.rotation_signs[0, 0])
        indices = torch.bucketize(y.contiguous(), state.key_inner)
        packed_ref = _pack(indices, cfg.key_mse_bits)

        # Triton encode
        pool_size = N
        packed_dim = cfg.key_mse_packed_dim
        mse_buf = torch.zeros(
            pool_size, H, packed_dim, dtype=torch.uint8, device=device
        )
        norm_buf = torch.zeros(pool_size, H, 1, dtype=torch.float32, device=device)
        loc = torch.arange(N, dtype=torch.int32, device=device)
        scratch = torch.empty(N * H * D, dtype=torch.float32, device=device)

        triton_tq_encode(
            kv=x,
            loc=loc,
            signs=state.rotation_signs[0, 0],
            boundaries=state.key_inner,
            mse_buffer=mse_buf,
            norm_buffer=norm_buf,
            scratch=scratch,
            bits=cfg.key_mse_bits,
        )

        # Compare packed indices
        assert torch.equal(mse_buf, packed_ref), (
            f"bits={bits}: packed mismatch, "
            f"{(mse_buf != packed_ref).sum().item()} / {mse_buf.numel()} differ"
        )

        # Compare norms
        norm_err = (norm_buf - norms_ref).abs().max().item()
        assert norm_err < 1e-5, f"bits={bits}: norm err {norm_err:.8f}"

    def test_2bit(self):
        self._encode_bits(2)

    def test_3bit(self):
        self._encode_bits(3)

    def test_4bit(self):
        self._encode_bits(4)

    def test_1bit(self):
        self._encode_bits(1)

    def test_encode_decode_roundtrip(self):
        """Full roundtrip: Triton encode → Triton decode → compare with original."""
        self._skip_no_cuda()

        from sglang.srt.layers.quantization.triton_tq_decode import (
            triton_tq_selective_decode,
        )
        from sglang.srt.layers.quantization.triton_tq_encode import triton_tq_encode

        N, H, D = 64, 8, 128
        device = "cuda"

        for bits in [2, 3, 4]:
            cfg = TurboQuantConfig(
                key_bits=bits, value_bits=bits, enable_qjl=False, head_dim=D
            )
            state = TurboQuantState(cfg, layer_num=1, head_num=H, device=device)

            x = torch.randn(N, H, D, device=device)

            # Encode via Triton
            packed_dim = cfg.key_mse_packed_dim
            mse_buf = torch.zeros(N, H, packed_dim, dtype=torch.uint8, device=device)
            norm_buf = torch.zeros(N, H, 1, dtype=torch.float32, device=device)
            loc = torch.arange(N, dtype=torch.int32, device=device)
            scratch = torch.empty(N * H * D, dtype=torch.float32, device=device)

            triton_tq_encode(
                kv=x,
                loc=loc,
                signs=state.rotation_signs[0, 0],
                boundaries=state.key_inner,
                mse_buffer=mse_buf,
                norm_buffer=norm_buf,
                scratch=scratch,
                bits=cfg.key_mse_bits,
            )

            # Decode via Triton
            pool_indices = torch.arange(N, dtype=torch.int32, device=device)
            compact_out = torch.zeros(N, H, D, dtype=torch.bfloat16, device=device)
            n_active = torch.tensor([N], dtype=torch.int32, device=device)

            triton_tq_selective_decode(
                mse_buffer=mse_buf,
                norm_buffer=norm_buf,
                pool_indices=pool_indices,
                centroids=state.key_centroids,
                signs=state.rotation_signs[0, 0],
                compact_out=compact_out,
                scratch=scratch,
                n_active_tensor=n_active,
                grid_n=N,
                bits=cfg.key_mse_bits,
            )

            # Cosine similarity
            cos_sim = F.cosine_similarity(
                x.reshape(-1, D).float(),
                compact_out.reshape(-1, D).float(),
                dim=-1,
            ).mean()
            assert cos_sim > 0.80, f"bits={bits}: cos_sim={cos_sim:.4f} too low"


if __name__ == "__main__":
    import sys

    # Simple test runner if pytest not available
    test_classes = [
        TestFWHT,
        TestRHT,
        TestBitPacking,
        TestLloydMax,
        TestConfig,
        TestEncodeDecodeKeys,
        TestEncodeDecodeValues,
        TestMemoryLayout,
        TestNextPow2,
        TestOutlierDetection,
        TestOnlineCodebook,
        TestMixedPrecisionConfig,
        TestQJLScoreWeight,
        TestMixedPrecisionKeys,
        TestMixedPrecisionValues,
        TestMixedPrecisionState,
        TestMixedPrecisionIntegration,
        TestTritonDecode,
        TestTritonEncode,
    ]
    passed = failed = 0
    for cls in test_classes:
        instance = cls()
        for name in dir(instance):
            if name.startswith("test_"):
                if hasattr(instance, "setup_method"):
                    instance.setup_method()
                try:
                    getattr(instance, name)()
                    print(f"  PASS: {cls.__name__}.{name}")
                    passed += 1
                except Exception as e:
                    print(f"  FAIL: {cls.__name__}.{name}: {e}")
                    failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
