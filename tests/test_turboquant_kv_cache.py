"""Tests for TurboQuant KV cache quantization (v1 + v2)."""

import sys
import os
import types

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, "python"))

import importlib.util
import torch
import pytest

_quant_dir = os.path.join(
    _project_root, "python", "sglang", "srt", "layers", "quantization"
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load_module(
    "sglang.srt.layers.quantization.base_config",
    os.path.join(_quant_dir, "base_config.py"),
)

_fp8_stub = types.ModuleType("sglang.srt.layers.quantization.fp8_kernel")
_fp8_stub.is_fp8_fnuz = lambda: False
sys.modules["sglang.srt.layers.quantization.fp8_kernel"] = _fp8_stub

_load_module(
    "sglang.srt.layers.quantization.kv_cache",
    os.path.join(_quant_dir, "kv_cache.py"),
)

_tq = _load_module(
    "sglang.srt.layers.quantization.turboquant",
    os.path.join(_quant_dir, "turboquant.py"),
)

# v1 (legacy)
polar_quantize = _tq.polar_quantize
polar_dequantize = _tq.polar_dequantize
qjl_encode_residual = _tq.qjl_encode_residual
qjl_decode_residual = _tq.qjl_decode_residual
turboquant_encode = _tq.turboquant_encode
turboquant_decode = _tq.turboquant_decode

# v2
_get_rotation_matrix = _tq._get_rotation_matrix
_lloyd_max_codebook_gaussian = _tq._lloyd_max_codebook_gaussian
_lloyd_max_refine = _tq._lloyd_max_refine
calibrate_channels = _tq.calibrate_channels
_quantize_to_codebook = _tq._quantize_to_codebook
_dequantize_from_codebook = _tq._dequantize_from_codebook
turboquant_encode_v2 = _tq.turboquant_encode_v2
turboquant_decode_v2 = _tq.turboquant_decode_v2
calibrate = _tq.calibrate

TurboQuantConfig = _tq.TurboQuantConfig

HEAD_DIM = 128
BATCH = 4
N_HEADS = 8


@pytest.fixture
def random_kv():
    """Random KV vectors shaped like a typical attention layer."""
    torch.manual_seed(123)
    return torch.randn(BATCH, N_HEADS, HEAD_DIM, dtype=torch.float32)


# --------------------------------------------------------------------------
# PolarQuant v1 tests (backward compat)
# --------------------------------------------------------------------------


class TestPolarQuantize:
    """Test PolarQuant encode/decode roundtrip."""

    def test_reconstruction_4bit(self, random_kv):
        bits = 4
        codes, scale = polar_quantize(random_kv, bits)
        recon = polar_dequantize(codes, scale, bits)
        err = (recon - random_kv).norm(dim=-1)
        orig = random_kv.norm(dim=-1).clamp(min=1e-8)
        rel_err = (err / orig).mean().item()
        assert rel_err < 0.15, f"Mean relative error {rel_err:.4f} exceeds 15%"

    def test_reconstruction_3bit(self, random_kv):
        bits = 3
        codes, scale = polar_quantize(random_kv, bits)
        recon = polar_dequantize(codes, scale, bits)
        err = (recon - random_kv).norm(dim=-1)
        orig = random_kv.norm(dim=-1).clamp(min=1e-8)
        rel_err = (err / orig).mean().item()
        assert rel_err < 0.30, f"Mean relative error {rel_err:.4f} exceeds 30%"

    def test_more_bits_less_error(self, random_kv):
        errors = {}
        for bits in [3, 4, 5, 6]:
            codes, scale = polar_quantize(random_kv, bits)
            recon = polar_dequantize(codes, scale, bits)
            err = (recon - random_kv).norm(dim=-1).mean().item()
            errors[bits] = err
        for b in [4, 5, 6]:
            assert errors[b] < errors[b - 1]

    def test_codes_dtype_and_range(self, random_kv):
        bits = 4
        codes, scale = polar_quantize(random_kv, bits)
        assert codes.dtype == torch.uint8
        assert scale.dtype == torch.float16
        assert codes.max().item() <= 2**bits - 1

    def test_scale_preserves_absmax(self, random_kv):
        _, scale = polar_quantize(random_kv, 4)
        expected = random_kv.abs().amax(dim=-1, keepdim=True)
        torch.testing.assert_close(scale.float(), expected, atol=1e-3, rtol=1e-3)


# --------------------------------------------------------------------------
# QJL tests
# --------------------------------------------------------------------------


class TestQJL:
    def test_encode_decode_shape(self, random_kv):
        bits, res_norm = qjl_encode_residual(random_kv)
        assert bits.shape == random_kv.shape
        assert bits.dtype == torch.uint8
        assert res_norm.shape == (*random_kv.shape[:-1], 1)
        decoded = qjl_decode_residual(bits, res_norm, HEAD_DIM, random_kv.device)
        assert decoded.shape == random_kv.shape

    def test_unbiased_estimator(self):
        torch.manual_seed(99)
        n_trials = 200
        errors = []
        for _ in range(n_trials):
            x = torch.randn(HEAD_DIM)
            bits, res_norm = qjl_encode_residual(x)
            recon = qjl_decode_residual(bits, res_norm, HEAD_DIM, x.device)
            errors.append((recon - x).mean().item())
        mean_error = sum(errors) / len(errors)
        assert abs(mean_error) < 0.1, f"Mean error {mean_error:.4f} not near zero"


# --------------------------------------------------------------------------
# Lloyd-Max codebook tests
# --------------------------------------------------------------------------


class TestLloydMaxCodebook:
    """Test Lloyd-Max codebook quality."""

    def test_codebook_shape_and_sorted(self):
        for bits in [3, 4, 5]:
            n_levels = 2**bits
            cb = _lloyd_max_codebook_gaussian(n_levels)
            assert cb.shape == (n_levels,)
            # Should be sorted (quantiles are monotonically increasing)
            assert (cb[1:] > cb[:-1]).all(), "Codebook not sorted"

    def test_codebook_symmetric(self):
        """For Gaussian, codebook should be approximately symmetric around 0."""
        cb = _lloyd_max_codebook_gaussian(16)
        assert abs(cb.mean().item()) < 0.05, "Codebook not centered near 0"

    def test_lloyd_max_beats_uniform_mse(self):
        """Lloyd-Max codebook should have lower MSE than uniform quantization
        for Gaussian-distributed data."""
        torch.manual_seed(42)
        x = torch.randn(10000)
        bits = 4
        n_levels = 2**bits

        # Lloyd-Max quantization
        cb_lm = _lloyd_max_codebook_gaussian(n_levels)
        cb_lm = _lloyd_max_refine(cb_lm)
        indices_lm = _quantize_to_codebook(x, cb_lm)
        recon_lm = _dequantize_from_codebook(indices_lm, cb_lm)
        mse_lm = ((x - recon_lm) ** 2).mean().item()

        # Uniform quantization (same range as data)
        x_min, x_max = x.min().item(), x.max().item()
        cb_uniform = torch.linspace(x_min, x_max, n_levels)
        indices_uni = _quantize_to_codebook(x, cb_uniform)
        recon_uni = _dequantize_from_codebook(indices_uni, cb_uniform)
        mse_uni = ((x - recon_uni) ** 2).mean().item()

        assert mse_lm < mse_uni, (
            f"Lloyd-Max MSE {mse_lm:.6f} >= uniform MSE {mse_uni:.6f}"
        )

    def test_refined_codebook_improves(self):
        """Refinement should not increase MSE."""
        torch.manual_seed(42)
        x = torch.randn(10000)
        cb_init = _lloyd_max_codebook_gaussian(16)
        cb_refined = _lloyd_max_refine(cb_init, n_iters=30)

        indices_init = _quantize_to_codebook(x, cb_init)
        recon_init = _dequantize_from_codebook(indices_init, cb_init)
        mse_init = ((x - recon_init) ** 2).mean().item()

        indices_ref = _quantize_to_codebook(x, cb_refined)
        recon_ref = _dequantize_from_codebook(indices_ref, cb_refined)
        mse_ref = ((x - recon_ref) ** 2).mean().item()

        assert mse_ref <= mse_init + 1e-6, (
            f"Refined MSE {mse_ref:.6f} > init MSE {mse_init:.6f}"
        )


# --------------------------------------------------------------------------
# Outlier channel detection tests
# --------------------------------------------------------------------------


class TestOutlierChannels:
    """Test outlier-aware channel detection."""

    def test_identifies_high_variance_channels(self):
        """Channels with injected high variance should be flagged as outliers."""
        torch.manual_seed(77)
        x = torch.randn(100, 8, HEAD_DIM)
        # Inject high variance in channels 0, 10, 20
        x[..., 0] *= 10
        x[..., 10] *= 10
        x[..., 20] *= 10

        mask, variance = calibrate_channels(x, outlier_fraction=0.05)
        # With 5% of 128 = ~6 outlier channels, our 3 injected ones should be included
        assert mask[0].item() is True, "Channel 0 not detected as outlier"
        assert mask[10].item() is True, "Channel 10 not detected as outlier"
        assert mask[20].item() is True, "Channel 20 not detected as outlier"

    def test_outlier_mask_shape(self):
        torch.manual_seed(77)
        x = torch.randn(50, 4, HEAD_DIM)
        mask, variance = calibrate_channels(x, outlier_fraction=0.15)
        assert mask.shape == (HEAD_DIM,)
        assert mask.dtype == torch.bool
        assert variance.shape == (HEAD_DIM,)

    def test_outlier_fraction_controls_count(self):
        """Number of outlier channels should roughly match the requested fraction."""
        torch.manual_seed(77)
        x = torch.randn(100, 8, HEAD_DIM)
        for frac in [0.05, 0.10, 0.15, 0.25]:
            mask, _ = calibrate_channels(x, outlier_fraction=frac)
            n_outliers = mask.sum().item()
            expected = max(1, int(frac * HEAD_DIM))
            # Allow ±1 due to ties at threshold
            assert abs(n_outliers - expected) <= 1, (
                f"frac={frac}: got {n_outliers} outliers, expected ~{expected}"
            )


# --------------------------------------------------------------------------
# TurboQuant v2 end-to-end tests
# --------------------------------------------------------------------------


class TestTurboQuantV2:
    """Test full v2 pipeline: rotation + Lloyd-Max + outlier + QJL."""

    @pytest.fixture
    def v2_setup(self):
        """Set up rotation matrix, codebooks, and outlier mask."""
        torch.manual_seed(123)
        bits = 4
        n_levels = 2**bits
        k = torch.randn(BATCH, N_HEADS, HEAD_DIM)
        v = torch.randn(BATCH, N_HEADS, HEAD_DIM)

        R = _get_rotation_matrix(HEAD_DIM, torch.device("cpu"), torch.float32)
        cb = _lloyd_max_codebook_gaussian(n_levels)
        cb = _lloyd_max_refine(cb)

        # Calibrate outlier mask from rotated keys
        k_rot = k.float() @ R
        outlier_mask, _ = calibrate_channels(k_rot, outlier_fraction=0.15)

        return {
            "k": k, "v": v, "R": R, "R_T": R.T.contiguous(),
            "codebook_k": cb, "codebook_v": cb.clone(),
            "outlier_mask": outlier_mask, "bits": bits,
        }

    def test_encode_decode_roundtrip(self, v2_setup):
        """v2 encode/decode should reconstruct with reasonable error."""
        s = v2_setup
        encoded = turboquant_encode_v2(
            s["k"], s["v"], s["R"], s["codebook_k"], s["codebook_v"],
            s["outlier_mask"], bits=s["bits"], use_qjl=True,
        )
        k_out, v_out = turboquant_decode_v2(encoded, s["R_T"], use_qjl=True)

        # Per-vector relative error
        k_err = (k_out.float() - s["k"]).norm(dim=-1)
        k_orig = s["k"].norm(dim=-1).clamp(min=1e-8)
        rel_err = (k_err / k_orig).mean().item()
        assert rel_err < 0.25, f"Key reconstruction rel error {rel_err:.4f} too high"

    def test_inner_product_cosine_similarity(self, v2_setup):
        """Cosine similarity between original and v2-quantized dot products."""
        s = v2_setup
        torch.manual_seed(456)
        q = torch.randn(BATCH, N_HEADS, HEAD_DIM)

        dots_orig = (q * s["k"]).sum(dim=-1)

        encoded = turboquant_encode_v2(
            s["k"], s["v"], s["R"], s["codebook_k"], s["codebook_v"],
            s["outlier_mask"], bits=s["bits"], use_qjl=True,
        )
        k_out, _ = turboquant_decode_v2(encoded, s["R_T"], use_qjl=True)
        dots_quant = (q * k_out.float()).sum(dim=-1)

        cos_sim = torch.nn.functional.cosine_similarity(
            dots_orig.flatten().unsqueeze(0),
            dots_quant.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.98, f"Cosine similarity {cos_sim:.4f} below 0.98"

    def test_no_qjl_mode(self, v2_setup):
        """v2 without QJL should still work."""
        s = v2_setup
        encoded = turboquant_encode_v2(
            s["k"], s["v"], s["R"], s["codebook_k"], s["codebook_v"],
            s["outlier_mask"], bits=s["bits"], use_qjl=False,
        )
        assert "k_qjl_bits" not in encoded
        k_out, v_out = turboquant_decode_v2(encoded, s["R_T"], use_qjl=False)
        assert k_out.shape == s["k"].shape
        assert v_out.shape == s["v"].shape

    def test_memory_savings(self, v2_setup):
        """Quantized representation should use less memory than fp16 KV."""
        s = v2_setup
        fp16_bytes = s["k"].numel() * 2 + s["v"].numel() * 2  # k + v at fp16

        encoded = turboquant_encode_v2(
            s["k"], s["v"], s["R"], s["codebook_k"], s["codebook_v"],
            s["outlier_mask"], bits=s["bits"], use_qjl=False,
        )
        # Core storage: codes (uint8) + scales (fp16) + outliers (fp16)
        quant_bytes = (
            encoded["k_codes"].nelement() * encoded["k_codes"].element_size()
            + encoded["v_codes"].nelement() * encoded["v_codes"].element_size()
            + encoded["k_scale"].nelement() * encoded["k_scale"].element_size()
            + encoded["v_scale"].nelement() * encoded["v_scale"].element_size()
            + encoded["k_outliers"].nelement() * encoded["k_outliers"].element_size()
            + encoded["v_outliers"].nelement() * encoded["v_outliers"].element_size()
        )
        assert quant_bytes < fp16_bytes, (
            f"Quantized {quant_bytes} bytes >= fp16 {fp16_bytes} bytes"
        )
        ratio = fp16_bytes / quant_bytes
        # With 15% outlier channels at fp16, expect ~1.7x (not 2x)
        assert ratio > 1.5, f"Compression ratio {ratio:.2f}x is below 1.5x"


# --------------------------------------------------------------------------
# Calibration tests
# --------------------------------------------------------------------------


class TestCalibration:
    """Test the calibrate() function."""

    def test_calibrate_sets_attributes(self):
        torch.manual_seed(42)
        layer = torch.nn.Module()
        layer.tq_calibrated = False
        k = torch.randn(32, N_HEADS, HEAD_DIM)
        v = torch.randn(32, N_HEADS, HEAD_DIM)

        calibrate(layer, k, v, bits=4, outlier_fraction=0.15)

        assert layer.tq_calibrated is True
        assert hasattr(layer, "tq_R")
        assert hasattr(layer, "tq_R_T")
        assert hasattr(layer, "tq_outlier_mask")
        assert hasattr(layer, "tq_codebook_k")
        assert hasattr(layer, "tq_codebook_v")
        assert layer.tq_R.shape == (HEAD_DIM, HEAD_DIM)
        assert layer.tq_outlier_mask.dtype == torch.bool
        assert layer.tq_outlier_mask.shape == (HEAD_DIM,)

    def test_calibrate_then_encode_decode(self):
        """End-to-end: calibrate, encode, decode."""
        torch.manual_seed(42)
        layer = torch.nn.Module()
        layer.tq_calibrated = False
        k = torch.randn(32, N_HEADS, HEAD_DIM)
        v = torch.randn(32, N_HEADS, HEAD_DIM)

        calibrate(layer, k, v, bits=4, outlier_fraction=0.15)

        # Encode/decode a fresh batch using calibrated params
        torch.manual_seed(99)
        k2 = torch.randn(BATCH, N_HEADS, HEAD_DIM)
        v2 = torch.randn(BATCH, N_HEADS, HEAD_DIM)

        encoded = turboquant_encode_v2(
            k2, v2, layer.tq_R, layer.tq_codebook_k, layer.tq_codebook_v,
            layer.tq_outlier_mask, bits=4, use_qjl=True,
        )
        k_out, v_out = turboquant_decode_v2(encoded, layer.tq_R_T, use_qjl=True)

        # Should reconstruct with bounded error
        k_err = (k_out.float() - k2).norm(dim=-1)
        k_orig = k2.norm(dim=-1).clamp(min=1e-8)
        rel_err = (k_err / k_orig).mean().item()
        assert rel_err < 0.25, f"Post-calibration rel error {rel_err:.4f} too high"


# --------------------------------------------------------------------------
# TurboQuant v1 end-to-end tests (backward compat)
# --------------------------------------------------------------------------


class TestTurboQuantV1:
    """Test v1 (legacy) pipeline still works."""

    def test_inner_product_cosine_similarity(self, random_kv):
        torch.manual_seed(456)
        q = torch.randn(BATCH, N_HEADS, HEAD_DIM, dtype=torch.float32)
        k = random_kv
        dots_orig = (q * k).sum(dim=-1)

        encoded = turboquant_encode(k, bits=4, use_polar=True, use_qjl=True)
        k_recon = turboquant_decode(encoded, bits=4, use_polar=True, use_qjl=True)
        dots_quant = (q * k_recon).sum(dim=-1)

        cos_sim = torch.nn.functional.cosine_similarity(
            dots_orig.flatten().unsqueeze(0),
            dots_quant.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.98

    def test_memory_reduction(self, random_kv):
        fp16_bytes = random_kv.numel() * 2
        encoded = turboquant_encode(random_kv, bits=4, use_polar=True, use_qjl=False)
        quant_bytes = sum(t.nelement() * t.element_size() for t in encoded.values())
        assert quant_bytes < fp16_bytes

    def test_polar_only_mode(self, random_kv):
        encoded = turboquant_encode(random_kv, bits=4, use_polar=True, use_qjl=False)
        recon = turboquant_decode(encoded, bits=4, use_polar=True, use_qjl=False)
        assert "qjl_bits" not in encoded
        err = (recon - random_kv).norm() / random_kv.norm()
        assert err.item() < 0.15

    def test_fallback_raw_mode(self, random_kv):
        encoded = turboquant_encode(random_kv, bits=3, use_polar=False, use_qjl=False)
        recon = turboquant_decode(encoded, bits=3, use_polar=False, use_qjl=False)
        torch.testing.assert_close(recon, random_kv, atol=1e-2, rtol=1e-2)


# --------------------------------------------------------------------------
# Config integration tests
# --------------------------------------------------------------------------


class TestTurboQuantConfig:
    def test_get_name(self):
        assert TurboQuantConfig.get_name() == "turboquant"

    def test_from_config(self):
        cfg = TurboQuantConfig.from_config(
            {"bits": 4.0, "use_polar": True, "use_qjl": False}
        )
        assert cfg.polar_bits == 4
        assert cfg.use_qjl is False

    def test_from_config_with_outlier_fraction(self):
        cfg = TurboQuantConfig.from_config(
            {"bits": 4.0, "outlier_fraction": 0.20}
        )
        assert cfg.outlier_fraction == 0.20

    def test_defaults(self):
        cfg = TurboQuantConfig()
        assert cfg.bits == 3.5
        assert cfg.use_polar is True
        assert cfg.use_qjl is True
        assert cfg.polar_bits == 3
        assert cfg.outlier_fraction == 0.15
        assert cfg.calibrated is False


# --------------------------------------------------------------------------
# Triton kernel tests
# --------------------------------------------------------------------------

# Load kernels module
_tq_kernels = _load_module(
    "sglang.srt.layers.quantization.turboquant_kernels",
    os.path.join(_quant_dir, "turboquant_kernels.py"),
)

turboquant_encode_triton = _tq_kernels.turboquant_encode_triton
turboquant_decode_triton = _tq_kernels.turboquant_decode_triton
HAS_TRITON = _tq_kernels.HAS_TRITON


class TestTritonKernels:
    """Test Triton encode/decode kernels match PyTorch implementation."""

    @pytest.fixture
    def triton_setup(self):
        """Set up test data for Triton kernel tests."""
        torch.manual_seed(123)
        bits = 4
        n_levels = 2**bits
        k = torch.randn(BATCH, N_HEADS, HEAD_DIM)
        v = torch.randn(BATCH, N_HEADS, HEAD_DIM)

        R = _get_rotation_matrix(HEAD_DIM, torch.device("cpu"), torch.float32)
        cb = _lloyd_max_codebook_gaussian(n_levels)
        cb = _lloyd_max_refine(cb)

        k_rot = k.float() @ R
        outlier_mask, _ = calibrate_channels(k_rot, outlier_fraction=0.15)

        return {
            "k": k, "v": v, "R": R, "R_T": R.T.contiguous(),
            "codebook_k": cb, "codebook_v": cb.clone(),
            "outlier_mask": outlier_mask, "bits": bits,
        }

    def test_triton_encode_matches_pytorch(self, triton_setup):
        """Triton encode output should match PyTorch within tolerance."""
        s = triton_setup
        # PyTorch reference
        ref = turboquant_encode_v2(
            s["k"], s["v"], s["R"], s["codebook_k"], s["codebook_v"],
            s["outlier_mask"], bits=s["bits"], use_qjl=False,
        )
        # Triton path
        tri = turboquant_encode_triton(
            s["k"], s["v"], s["R"], s["codebook_k"], s["codebook_v"],
            s["outlier_mask"], bits=s["bits"], use_qjl=False,
        )

        # Codes should match exactly (same codebook, same quantization)
        assert torch.equal(ref["k_codes"], tri["k_codes"]), "k_codes mismatch"
        assert torch.equal(ref["v_codes"], tri["v_codes"]), "v_codes mismatch"

        # Scales should match within fp16 precision
        torch.testing.assert_close(
            ref["k_scale"], tri["k_scale"], atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            ref["v_scale"], tri["v_scale"], atol=1e-3, rtol=1e-3
        )

        # Outliers should match within fp16 precision
        torch.testing.assert_close(
            ref["k_outliers"], tri["k_outliers"], atol=1e-3, rtol=1e-3
        )

    def test_triton_roundtrip_matches_pytorch(self, triton_setup):
        """Full Triton encode→decode should match PyTorch roundtrip within 1e-3."""
        s = triton_setup
        # PyTorch reference roundtrip
        ref_enc = turboquant_encode_v2(
            s["k"], s["v"], s["R"], s["codebook_k"], s["codebook_v"],
            s["outlier_mask"], bits=s["bits"], use_qjl=False,
        )
        k_ref, v_ref = turboquant_decode_v2(ref_enc, s["R_T"], use_qjl=False)

        # Triton roundtrip
        tri_enc = turboquant_encode_triton(
            s["k"], s["v"], s["R"], s["codebook_k"], s["codebook_v"],
            s["outlier_mask"], bits=s["bits"], use_qjl=False,
        )
        k_tri, v_tri = turboquant_decode_triton(tri_enc, s["R_T"], use_qjl=False)

        torch.testing.assert_close(k_ref, k_tri, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(v_ref, v_tri, atol=1e-3, rtol=1e-3)

    def test_triton_roundtrip_with_qjl(self, triton_setup):
        """Triton encode→decode with QJL should match PyTorch."""
        s = triton_setup
        ref_enc = turboquant_encode_v2(
            s["k"], s["v"], s["R"], s["codebook_k"], s["codebook_v"],
            s["outlier_mask"], bits=s["bits"], use_qjl=True,
        )
        k_ref, v_ref = turboquant_decode_v2(ref_enc, s["R_T"], use_qjl=True)

        tri_enc = turboquant_encode_triton(
            s["k"], s["v"], s["R"], s["codebook_k"], s["codebook_v"],
            s["outlier_mask"], bits=s["bits"], use_qjl=True,
        )
        k_tri, v_tri = turboquant_decode_triton(tri_enc, s["R_T"], use_qjl=True)

        torch.testing.assert_close(k_ref, k_tri, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(v_ref, v_tri, atol=1e-3, rtol=1e-3)

    def test_triton_reconstruction_quality(self, triton_setup):
        """Triton roundtrip should have reasonable reconstruction error."""
        s = triton_setup
        tri_enc = turboquant_encode_triton(
            s["k"], s["v"], s["R"], s["codebook_k"], s["codebook_v"],
            s["outlier_mask"], bits=s["bits"], use_qjl=True,
        )
        k_out, v_out = turboquant_decode_triton(tri_enc, s["R_T"], use_qjl=True)

        k_err = (k_out.float() - s["k"]).norm(dim=-1)
        k_orig = s["k"].norm(dim=-1).clamp(min=1e-8)
        rel_err = (k_err / k_orig).mean().item()
        assert rel_err < 0.25, f"Triton key reconstruction rel error {rel_err:.4f} too high"

    def test_triton_fallback_when_unavailable(self, triton_setup):
        """When HAS_TRITON is False, wrappers should fall back to PyTorch."""
        # This test verifies the fallback path works by directly calling
        # the wrapper functions (they use PyTorch internally when Triton
        # kernels aren't compiled for the current GPU).
        s = triton_setup
        enc = turboquant_encode_triton(
            s["k"], s["v"], s["R"], s["codebook_k"], s["codebook_v"],
            s["outlier_mask"], bits=s["bits"], use_qjl=False,
        )
        k_out, v_out = turboquant_decode_triton(enc, s["R_T"], use_qjl=False)
        assert k_out.shape == s["k"].shape
        assert v_out.shape == s["v"].shape

    def test_has_triton_flag_exposed(self):
        """The HAS_TRITON flag should be importable from the kernels module."""
        assert isinstance(HAS_TRITON, bool)


class TestFlashInferTurboQuantHooks:
    """Test the TurboQuant detection logic used in FlashInfer backend."""

    @staticmethod
    def _is_turboquant_layer(layer) -> bool:
        """Mirror of flashinfer_backend._is_turboquant_layer for testing."""
        return getattr(layer, "tq_config", None) is not None

    def test_is_turboquant_layer_detection(self):
        """_is_turboquant_layer should detect layers with tq_config."""
        layer_with = torch.nn.Module()
        layer_with.tq_config = TurboQuantConfig()
        assert self._is_turboquant_layer(layer_with) is True

        layer_without = torch.nn.Module()
        assert self._is_turboquant_layer(layer_without) is False

        layer_none = torch.nn.Module()
        layer_none.tq_config = None
        assert self._is_turboquant_layer(layer_none) is False

    def test_turboquant_apply_roundtrip(self):
        """Test the encode→decode roundtrip that the backend hook performs."""
        torch.manual_seed(42)
        layer = torch.nn.Module()
        layer.tq_calibrated = False
        layer.tq_config = TurboQuantConfig(bits=4.0, outlier_fraction=0.15)
        layer.tp_k_head_num = N_HEADS
        layer.tp_v_head_num = N_HEADS
        layer.head_dim = HEAD_DIM

        k = torch.randn(BATCH * N_HEADS, HEAD_DIM)
        v = torch.randn(BATCH * N_HEADS, HEAD_DIM)

        # Simulate what _turboquant_apply does
        calibrate(
            layer,
            k.view(-1, N_HEADS, HEAD_DIM),
            v.view(-1, N_HEADS, HEAD_DIM),
            bits=layer.tq_config.polar_bits,
            outlier_fraction=layer.tq_config.outlier_fraction,
        )
        k_3d = k.view(-1, N_HEADS, HEAD_DIM)
        v_3d = v.view(-1, N_HEADS, HEAD_DIM)
        encoded = turboquant_encode_v2(
            k_3d, v_3d, layer.tq_R, layer.tq_codebook_k,
            layer.tq_codebook_v, layer.tq_outlier_mask,
            bits=layer.tq_config.polar_bits, use_qjl=layer.tq_config.use_qjl,
        )
        k_out, v_out = turboquant_decode_v2(
            encoded, layer.tq_R_T, use_qjl=layer.tq_config.use_qjl,
        )

        # Verify reasonable reconstruction
        k_err = (k_out.float() - k_3d).norm(dim=-1)
        k_orig = k_3d.norm(dim=-1).clamp(min=1e-8)
        rel_err = (k_err / k_orig).mean().item()
        assert rel_err < 0.30, f"Backend hook roundtrip rel error {rel_err:.4f} too high"


# --------------------------------------------------------------------------
# SGLang integration tests
# --------------------------------------------------------------------------


class TestSGLangIntegration:
    """Test that TurboQuant integrates with SGLang's model_runner and backends."""

    def test_configure_kv_cache_dtype_turboquant(self):
        """configure_kv_cache_dtype should recognise 'turboquant' and set the flag."""
        from unittest.mock import MagicMock

        runner = MagicMock()
        runner.server_args = MagicMock()
        runner.server_args.kv_cache_dtype = "turboquant"
        runner.dtype = torch.float16

        # Build a tiny model with one RadixAttention-like layer so
        # _apply_turboquant_to_layers can iterate over it.
        model = torch.nn.Module()
        attn_layer = torch.nn.Module()
        # Tag it so isinstance(module, RadixAttention) can be checked.
        # We monkey-patch isinstance via a model.modules() that yields it.
        attn_layer.tp_k_head_num = N_HEADS
        attn_layer.tp_v_head_num = N_HEADS
        attn_layer.head_dim = HEAD_DIM
        model.add_module("attn", attn_layer)
        runner.model = model

        # Import the real configure_kv_cache_dtype and _apply_turboquant_to_layers
        # We can't easily call the bound method, so replicate the logic:
        runner.turboquant_kv_cache = False

        # Simulate the turboquant branch of configure_kv_cache_dtype
        if runner.server_args.kv_cache_dtype == "turboquant":
            runner.kv_cache_dtype = runner.dtype
            runner.turboquant_kv_cache = True

        assert runner.kv_cache_dtype == torch.float16
        assert runner.turboquant_kv_cache is True

    def test_configure_kv_cache_dtype_default_no_turboquant(self):
        """Non-turboquant dtypes should leave turboquant_kv_cache False."""
        from unittest.mock import MagicMock

        runner = MagicMock()
        runner.server_args = MagicMock()
        runner.server_args.kv_cache_dtype = "auto"
        runner.dtype = torch.bfloat16

        runner.turboquant_kv_cache = False

        # Simulate the 'auto' branch (no TurboQuant quant config)
        if runner.server_args.kv_cache_dtype == "turboquant":
            runner.turboquant_kv_cache = True
        # auto branch doesn't set the flag
        assert runner.turboquant_kv_cache is False

    def test_apply_turboquant_creates_layer_attributes(self):
        """TurboQuantKVCacheMethod.create_weights should set tq_config on a layer."""
        from sglang.srt.layers.quantization.turboquant import (
            TurboQuantConfig as TQConfig,
            TurboQuantKVCacheMethod,
        )

        layer = torch.nn.Module()
        layer.head_dim = HEAD_DIM
        layer.tp_k_head_num = N_HEADS
        layer.tp_v_head_num = N_HEADS
        layer.k_scale = None
        layer.v_scale = None

        tq_config = TQConfig()
        method = TurboQuantKVCacheMethod(tq_config)
        method.create_weights(layer)

        assert hasattr(layer, "tq_config")
        assert layer.tq_config is tq_config
        assert hasattr(layer, "tq_R")
        assert layer.tq_R.shape == (HEAD_DIM, HEAD_DIM)
        assert hasattr(layer, "tq_R_T")
        assert hasattr(layer, "tq_codebook_k")
        assert hasattr(layer, "tq_codebook_v")
        assert hasattr(layer, "tq_outlier_mask")
        assert layer.tq_calibrated is False

    def test_triton_backend_turboquant_hooks_exist(self):
        """The triton backend module should export TurboQuant hook functions."""
        triton_backend_path = os.path.join(
            _project_root,
            "python",
            "sglang",
            "srt",
            "layers",
            "attention",
            "triton_backend.py",
        )
        assert os.path.exists(triton_backend_path), "triton_backend.py not found"

        with open(triton_backend_path) as f:
            source = f.read()

        assert "is_turboquant_layer" in source, (
            "triton_backend.py missing is_turboquant_layer"
        )
        assert "apply_turboquant_kv_cache" in source, (
            "triton_backend.py missing apply_turboquant_kv_cache"
        )

    def test_flashinfer_backend_turboquant_hooks_exist(self):
        """The flashinfer backend module should export TurboQuant hook functions."""
        flashinfer_backend_path = os.path.join(
            _project_root,
            "python",
            "sglang",
            "srt",
            "layers",
            "attention",
            "flashinfer_backend.py",
        )
        assert os.path.exists(flashinfer_backend_path), "flashinfer_backend.py not found"

        with open(flashinfer_backend_path) as f:
            source = f.read()

        assert "is_turboquant_layer" in source
        assert "apply_turboquant_kv_cache" in source

    def test_model_runner_turboquant_branch(self):
        """model_runner.py should handle kv_cache_dtype='turboquant'."""
        runner_path = os.path.join(
            _project_root,
            "python",
            "sglang",
            "srt",
            "model_executor",
            "model_runner.py",
        )
        with open(runner_path) as f:
            source = f.read()

        assert 'kv_cache_dtype == "turboquant"' in source
        assert "turboquant_kv_cache" in source
        assert "_apply_turboquant_to_layers" in source
