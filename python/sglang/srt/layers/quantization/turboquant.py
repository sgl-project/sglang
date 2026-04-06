"""TurboQuant KV cache quantization — v2: rotation + Lloyd-Max + outlier channels.

Based on the TurboQuant paper (ICLR 2026): https://arxiv.org/abs/2504.19874
Faithful to the vLLM PR #38280 approach:
  1. Random orthogonal rotation to spread energy across dimensions
  2. Lloyd-Max scalar quantization (near-optimal for Gaussian distributions)
  3. Outlier-aware channel allocation (high-variance channels stay at bf16)
  4. QJL 1-bit residual correction on the quantization residual

Pure PyTorch implementation with optional Triton-accelerated kernels.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core math functions
# ---------------------------------------------------------------------------

_SEED = 42


def _get_rotation_matrix(
    head_dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Generate a fixed random orthogonal rotation matrix."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_SEED)
    R, _ = torch.linalg.qr(
        torch.randn(head_dim, head_dim, generator=gen, dtype=torch.float32)
    )
    return R.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# v1 (backward-compat): PolarQuant uniform quantization
# ---------------------------------------------------------------------------


def polar_quantize_v1(
    x: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy PolarQuant: per-vector absmax symmetric uniform quantization."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    n_levels = 2**bits
    codes = ((x / scale + 1) / 2 * (n_levels - 1)).round().clamp(0, n_levels - 1).to(
        torch.uint8
    )
    return codes, scale.to(torch.float16)


def polar_dequantize_v1(
    codes: torch.Tensor, scale: torch.Tensor, bits: int
) -> torch.Tensor:
    """Legacy inverse of polar_quantize_v1."""
    n_levels = 2**bits
    x = codes.float() / (n_levels - 1) * 2 - 1
    return x * scale.float()


# Keep old names as aliases for backward compat
polar_quantize = polar_quantize_v1
polar_dequantize = polar_dequantize_v1


# ---------------------------------------------------------------------------
# QJL 1-bit residual correction (unchanged)
# ---------------------------------------------------------------------------


def _rademacher_matrix(
    d: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Deterministic Rademacher (±1) matrix for QJL projection."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_SEED)
    return (
        torch.randint(0, 2, (d, d), generator=gen, dtype=dtype) * 2 - 1
    ).to(device=device)


def qjl_encode_residual(
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """1-bit QJL: project residual and store sign bits + residual norm."""
    d = residual.shape[-1]
    jl_matrix = _rademacher_matrix(d, residual.device)
    projected = residual.float() @ jl_matrix
    sign_bits = (projected > 0).to(torch.uint8)
    res_norm = residual.float().norm(dim=-1, keepdim=True)
    return sign_bits, res_norm.to(torch.float16)


def qjl_decode_residual(
    sign_bits: torch.Tensor,
    res_norm: torch.Tensor,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Reconstruct residual estimate from QJL sign bits.

    Unbiased estimator: (||r|| · √(π/2) / d) · J^T @ signs
    """
    jl_matrix = _rademacher_matrix(head_dim, device)
    signs = sign_bits.float() * 2 - 1
    direction = signs @ jl_matrix.T
    return direction * (res_norm.float() * (torch.pi / 2) ** 0.5 / head_dim)


# ---------------------------------------------------------------------------
# v2: Lloyd-Max codebook + rotation + outlier-aware quantization
# ---------------------------------------------------------------------------


def _lloyd_max_codebook_gaussian(n_levels: int) -> torch.Tensor:
    """Precompute Lloyd-Max codebook for unit Gaussian distribution.

    Uses quantile initialization (optimal for Gaussian when n_levels is moderate).
    Returns sorted codebook of shape [n_levels].
    """
    # Quantiles of N(0,1) — equivalent to Lloyd-Max centroids for Gaussian
    # when using uniform probability mass bins.
    quantiles = torch.linspace(1 / (2 * n_levels), 1 - 1 / (2 * n_levels), n_levels)
    # Inverse CDF of standard normal (Probit function)
    # Use torch.erfinv: Phi^{-1}(p) = sqrt(2) * erfinv(2p - 1)
    codebook = math.sqrt(2) * torch.erfinv(2 * quantiles - 1)
    return codebook.float()


def _lloyd_max_refine(codebook: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """Refine Lloyd-Max codebook with iterative optimization for N(0,1).

    Runs the Lloyd-Max algorithm: alternate between finding optimal boundaries
    (midpoints of adjacent centroids) and recomputing centroids as conditional
    expectations of N(0,1) within each bin.
    """
    cb = codebook.clone().double()
    n = cb.shape[0]
    for _ in range(n_iters):
        # Boundaries: midpoints between adjacent centroids, plus -inf/+inf
        bounds = torch.zeros(n + 1, dtype=torch.float64)
        bounds[0] = -8.0  # effectively -inf for N(0,1)
        bounds[-1] = 8.0
        for i in range(1, n):
            bounds[i] = (cb[i - 1] + cb[i]) / 2.0

        # Recompute centroids: E[X | bounds[i] < X < bounds[i+1]] for X ~ N(0,1)
        # E[X | a < X < b] = (phi(a) - phi(b)) / (Phi(b) - Phi(a))
        # where phi = pdf, Phi = cdf of standard normal
        for i in range(n):
            a, b = bounds[i], bounds[i + 1]
            phi_a = torch.exp(-0.5 * a**2) / math.sqrt(2 * math.pi)
            phi_b = torch.exp(-0.5 * b**2) / math.sqrt(2 * math.pi)
            # Use erfc for numerical stability at tails
            cdf_a = 0.5 * torch.erfc(-a / math.sqrt(2))
            cdf_b = 0.5 * torch.erfc(-b / math.sqrt(2))
            mass = (cdf_b - cdf_a).clamp(min=1e-30)
            cb[i] = (phi_a - phi_b) / mass

    return cb.float()


def calibrate_channels(
    k_samples: torch.Tensor,
    outlier_fraction: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identify high-variance (outlier) channels that should stay at bf16.

    Args:
        k_samples: [n_tokens, ...., head_dim] tensor of sample key vectors.
        outlier_fraction: fraction of channels to keep at bf16.

    Returns:
        outlier_mask: [head_dim] bool — True for outlier channels.
        variance: [head_dim] per-channel variance.
    """
    flat = k_samples.reshape(-1, k_samples.shape[-1]).float()
    variance = flat.var(dim=0)
    n_outliers = max(1, int(outlier_fraction * k_samples.shape[-1]))
    threshold = variance.topk(n_outliers).values[-1]
    outlier_mask = variance >= threshold
    return outlier_mask, variance


def _quantize_to_codebook(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor codebook lookup. Returns uint8 indices.

    Args:
        x: [..., d] normalized values.
        codebook: [n_levels] sorted codebook centroids.

    Returns:
        indices: [..., d] uint8 codebook indices.
    """
    diff = x.unsqueeze(-1) - codebook.to(device=x.device, dtype=x.dtype)
    indices = diff.abs().argmin(dim=-1)
    return indices.to(torch.uint8)


def _dequantize_from_codebook(
    indices: torch.Tensor, codebook: torch.Tensor
) -> torch.Tensor:
    """Look up codebook values from uint8 indices."""
    return codebook.to(device=indices.device)[indices.long()]


def turboquant_encode_v2(
    k: torch.Tensor,
    v: torch.Tensor,
    R: torch.Tensor,
    codebook_k: torch.Tensor,
    codebook_v: torch.Tensor,
    outlier_mask: torch.Tensor,
    bits: int = 4,
    use_qjl: bool = True,
) -> dict[str, torch.Tensor]:
    """Full TurboQuant v2 encode: rotation + Lloyd-Max + outlier handling + QJL.

    Args:
        k, v: [..., head_dim] key/value tensors.
        R: [head_dim, head_dim] orthogonal rotation matrix.
        codebook_k, codebook_v: [n_levels] Lloyd-Max codebooks.
        outlier_mask: [head_dim] bool — True for channels kept at bf16.
        bits: quantization bits (determines codebook size).
        use_qjl: whether to apply QJL residual correction.
    """
    result: dict[str, torch.Tensor] = {}
    normal_mask = ~outlier_mask

    # 1. Rotate
    k_rot = (k.float() @ R.float()).to(k.dtype)
    v_rot = (v.float() @ R.float()).to(v.dtype)

    # 2. Save outlier channels at fp16
    result["k_outliers"] = k_rot[..., outlier_mask].to(torch.float16)
    result["v_outliers"] = v_rot[..., outlier_mask].to(torch.float16)
    result["outlier_mask"] = outlier_mask

    # 3. Quantize non-outlier channels with Lloyd-Max codebook
    k_normal = k_rot[..., normal_mask]
    v_normal = v_rot[..., normal_mask]

    k_scale = k_normal.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    v_scale = v_normal.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)

    k_norm = k_normal.float() / k_scale
    v_norm = v_normal.float() / v_scale

    k_codes = _quantize_to_codebook(k_norm, codebook_k)
    v_codes = _quantize_to_codebook(v_norm, codebook_v)

    result["k_codes"] = k_codes
    result["k_scale"] = k_scale.to(torch.float16)
    result["v_codes"] = v_codes
    result["v_scale"] = v_scale.to(torch.float16)
    result["codebook_k"] = codebook_k
    result["codebook_v"] = codebook_v

    # 4. QJL residual correction on quantized channels
    if use_qjl:
        k_recon_normal = _dequantize_from_codebook(k_codes, codebook_k) * k_scale
        v_recon_normal = _dequantize_from_codebook(v_codes, codebook_v) * v_scale
        k_residual = k_normal.float() - k_recon_normal
        v_residual = v_normal.float() - v_recon_normal
        k_qjl_bits, k_qjl_norm = qjl_encode_residual(k_residual)
        v_qjl_bits, v_qjl_norm = qjl_encode_residual(v_residual)
        result["k_qjl_bits"] = k_qjl_bits
        result["k_qjl_norm"] = k_qjl_norm
        result["v_qjl_bits"] = v_qjl_bits
        result["v_qjl_norm"] = v_qjl_norm

    return result


def turboquant_decode_v2(
    encoded: dict[str, torch.Tensor],
    R_T: torch.Tensor,
    use_qjl: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode TurboQuant v2: codebook lookup + merge outliers + inverse rotation.

    Args:
        encoded: dict from turboquant_encode_v2.
        R_T: [head_dim, head_dim] transpose of rotation matrix (R^T).
        use_qjl: whether QJL correction was used during encoding.

    Returns:
        k_out, v_out: [..., head_dim] reconstructed key/value tensors.
    """
    codebook_k = encoded["codebook_k"]
    codebook_v = encoded["codebook_v"]
    outlier_mask = encoded["outlier_mask"]
    normal_mask = ~outlier_mask
    head_dim = outlier_mask.shape[0]

    # Decode quantized channels
    k_recon = _dequantize_from_codebook(encoded["k_codes"], codebook_k) * encoded["k_scale"].float()
    v_recon = _dequantize_from_codebook(encoded["v_codes"], codebook_v) * encoded["v_scale"].float()

    # QJL residual correction
    if use_qjl and "k_qjl_bits" in encoded:
        n_normal = normal_mask.sum().item()
        k_res = qjl_decode_residual(
            encoded["k_qjl_bits"], encoded["k_qjl_norm"], n_normal, encoded["k_codes"].device
        )
        v_res = qjl_decode_residual(
            encoded["v_qjl_bits"], encoded["v_qjl_norm"], n_normal, encoded["v_codes"].device
        )
        k_recon = k_recon + k_res
        v_recon = v_recon + v_res

    # Reconstruct full head_dim tensor
    batch_shape = encoded["k_codes"].shape[:-1]
    k_full = torch.zeros(*batch_shape, head_dim, dtype=torch.float32,
                          device=encoded["k_codes"].device)
    v_full = torch.zeros_like(k_full)

    k_full[..., normal_mask] = k_recon.float()
    v_full[..., normal_mask] = v_recon.float()
    k_full[..., outlier_mask] = encoded["k_outliers"].float()
    v_full[..., outlier_mask] = encoded["v_outliers"].float()

    # Inverse rotation
    k_out = (k_full @ R_T.float()).to(torch.float16)
    v_out = (v_full @ R_T.float()).to(torch.float16)

    return k_out, v_out


# ---------------------------------------------------------------------------
# High-level encode / decode (v2 is now the default path)
# ---------------------------------------------------------------------------


def turboquant_encode(
    x: torch.Tensor,
    bits: int = 3,
    use_polar: bool = True,
    use_qjl: bool = True,
) -> dict[str, torch.Tensor]:
    """Encode a KV tensor with TurboQuant (v1 legacy path).

    For the v2 path (rotation + Lloyd-Max + outlier channels), use
    turboquant_encode_v2 directly.
    """
    result: dict[str, torch.Tensor] = {}

    if use_polar:
        codes, radius = polar_quantize(x, bits)
        result["codes"] = codes
        result["radius"] = radius

        if use_qjl:
            recon = polar_dequantize(codes, radius, bits)
            residual = x.float() - recon
            qjl_bits, res_norm = qjl_encode_residual(residual)
            result["qjl_bits"] = qjl_bits
            result["qjl_norm"] = res_norm
    else:
        result["raw"] = x.to(torch.float16)

    return result


def turboquant_decode(
    encoded: dict[str, torch.Tensor],
    bits: int = 3,
    use_polar: bool = True,
    use_qjl: bool = True,
) -> torch.Tensor:
    """Decode a TurboQuant-compressed KV tensor (v1 legacy path)."""
    if not use_polar:
        return encoded["raw"].float()

    recon = polar_dequantize(encoded["codes"], encoded["radius"], bits)

    if use_qjl and "qjl_bits" in encoded:
        head_dim = encoded["codes"].shape[-1]
        device = encoded["codes"].device
        residual_est = qjl_decode_residual(
            encoded["qjl_bits"], encoded["qjl_norm"], head_dim, device
        )
        recon = recon + residual_est

    return recon


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate(
    layer: torch.nn.Module,
    k_samples: torch.Tensor,
    v_samples: torch.Tensor,
    bits: int = 4,
    outlier_fraction: float = 0.15,
) -> None:
    """Calibrate a TurboQuant layer on a sample batch.

    Sets outlier_mask, codebooks, and marks the layer as calibrated.

    Args:
        layer: nn.Module with tq_* attributes (set by create_weights).
        k_samples: [n_tokens, n_heads, head_dim] sample keys.
        v_samples: [n_tokens, n_heads, head_dim] sample values.
        bits: quantization bits.
        outlier_fraction: fraction of channels kept at bf16.
    """
    device = k_samples.device
    head_dim = k_samples.shape[-1]
    R = _get_rotation_matrix(head_dim, device, torch.float32)

    # Rotate samples
    k_rot = k_samples.float() @ R
    v_rot = v_samples.float() @ R

    # Detect outlier channels
    outlier_mask, _ = calibrate_channels(k_rot, outlier_fraction)

    # Compute Lloyd-Max codebook (Gaussian approximation after rotation)
    n_levels = 2**bits
    codebook = _lloyd_max_codebook_gaussian(n_levels)
    codebook = _lloyd_max_refine(codebook, n_iters=20)

    # Store on layer
    layer.tq_R = R.to(device)
    layer.tq_R_T = R.T.to(device)
    layer.tq_outlier_mask = outlier_mask.to(device)
    layer.tq_codebook_k = codebook.to(device)
    layer.tq_codebook_v = codebook.to(device)
    layer.tq_calibrated = True


# ---------------------------------------------------------------------------
# SGLang quantization config & KV cache method
# ---------------------------------------------------------------------------


class TurboQuantConfig(QuantizationConfig):
    """TurboQuant KV cache quantization (rotation + Lloyd-Max + outlier channels + QJL)."""

    def __init__(
        self,
        bits: float = 3.5,
        use_polar: bool = True,
        use_qjl: bool = True,
        outlier_fraction: float = 0.15,
        calibrated: bool = False,
    ):
        super().__init__()
        self.bits = bits
        self.polar_bits = int(bits)
        self.use_polar = use_polar
        self.use_qjl = use_qjl
        self.outlier_fraction = outlier_fraction
        self.calibrated = calibrated

    @classmethod
    def get_name(cls) -> str:
        return "turboquant"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["turboquant_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TurboQuantConfig":
        bits = cls.get_from_keys_or(config, ["bits"], 3.5)
        use_polar = cls.get_from_keys_or(config, ["use_polar"], True)
        use_qjl = cls.get_from_keys_or(config, ["use_qjl"], True)
        outlier_fraction = cls.get_from_keys_or(config, ["outlier_fraction"], 0.15)
        return cls(bits=bits, use_polar=use_polar, use_qjl=use_qjl,
                   outlier_fraction=outlier_fraction)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from sglang.srt.layers.radix_attention import RadixAttention

        if isinstance(layer, RadixAttention):
            return TurboQuantKVCacheMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class TurboQuantKVCacheMethod(BaseKVCacheMethod):
    """KV cache quant method using TurboQuant v2 (rotation + Lloyd-Max + outliers + QJL)."""

    def __init__(self, quant_config: TurboQuantConfig):
        super().__init__(quant_config)
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module):
        """Initialize rotation matrix, placeholder codebooks, and outlier mask."""
        super().create_weights(layer)
        layer.tq_config = self.quant_config
        layer.tq_calibrated = False

        # Rotation matrix (seeded, deterministic)
        # Will be created on the correct device lazily or during calibration.
        # For now store as CPU placeholder.
        head_dim = getattr(layer, "head_dim", 128)
        R = _get_rotation_matrix(head_dim, torch.device("cpu"), torch.float32)
        layer.tq_R = R
        layer.tq_R_T = R.T.contiguous()

        # Placeholder codebooks (Gaussian Lloyd-Max, will be refined on calibration)
        n_levels = 2 ** self.quant_config.polar_bits
        codebook = _lloyd_max_codebook_gaussian(n_levels)
        layer.tq_codebook_k = codebook
        layer.tq_codebook_v = codebook.clone()

        # Placeholder outlier mask (no outliers until calibrated)
        layer.tq_outlier_mask = torch.zeros(head_dim, dtype=torch.bool)

        layer.tq_initialized = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        layer.tq_initialized = True


# ---------------------------------------------------------------------------
# Triton-accelerated encode/decode (optional, falls back to PyTorch)
# ---------------------------------------------------------------------------

try:
    from sglang.srt.layers.quantization.turboquant_kernels import (
        HAS_TRITON as _HAS_TRITON_KERNELS,
        turboquant_decode_triton,
        turboquant_encode_triton,
    )
except ImportError:
    _HAS_TRITON_KERNELS = False
    turboquant_encode_triton = None  # type: ignore[assignment]
    turboquant_decode_triton = None  # type: ignore[assignment]

HAS_TRITON_KERNELS = _HAS_TRITON_KERNELS


# ---------------------------------------------------------------------------
# Shared KV cache hooks (used by attention backends)
# ---------------------------------------------------------------------------

def is_turboquant_layer(layer) -> bool:
    """Check if a layer has TurboQuant KV cache quantization enabled."""
    return getattr(layer, "tq_config", None) is not None


def apply_turboquant_kv_cache(layer, k: torch.Tensor, v: torch.Tensor):
    """Apply TurboQuant encode→decode round-trip to K/V before storing to cache.
    
    On the first call (not yet calibrated), runs calibration on the current batch.
    Returns the round-tripped (lossy-compressed) K and V tensors in fp16,
    ready to be stored into the paged KV cache.

    NOTE: This function uses boolean indexing which is incompatible with CUDA graph
    capture. It skips quantization during CUDA graph capture and only applies during
    normal (non-captured) inference. Calibration is also skipped during capture.
    """
    # Skip during CUDA graph capture — boolean indexing is not capturable
    try:
        from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
        if is_in_piecewise_cuda_graph():
            return k, v
    except ImportError:
        pass

    # Check if we're inside a CUDA graph capture context via torch
    if torch.cuda.is_current_stream_capturing():
        return k, v

    # Lazy calibration on first forward pass
    if not getattr(layer, "tq_calibrated", False):
        # Reshape to (tokens, heads, head_dim) for calibration
        k_cal = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v_cal = v.view(-1, layer.tp_v_head_num, layer.head_dim)
        calibrate(
            layer,
            k_cal,
            v_cal,
            bits=layer.tq_config.polar_bits,
            outlier_fraction=layer.tq_config.outlier_fraction,
        )
    
    # Move TQ params to the correct device if needed
    device = k.device
    if layer.tq_R.device != device:
        layer.tq_R = layer.tq_R.to(device)
        layer.tq_R_T = layer.tq_R_T.to(device)
        layer.tq_outlier_mask = layer.tq_outlier_mask.to(device)
        layer.tq_codebook_k = layer.tq_codebook_k.to(device)
        layer.tq_codebook_v = layer.tq_codebook_v.to(device)
    
    # Reshape to (..., head_dim) for encode/decode
    orig_k_shape = k.shape
    orig_v_shape = v.shape
    k_3d = k.view(-1, layer.tp_k_head_num, layer.head_dim)
    v_3d = v.view(-1, layer.tp_v_head_num, layer.head_dim)
    
    # Encode (quantize) then immediately decode (dequantize)
    encoded = turboquant_encode_v2(
        k_3d,
        v_3d,
        layer.tq_R,
        layer.tq_codebook_k,
        layer.tq_codebook_v,
        layer.tq_outlier_mask,
        bits=layer.tq_config.polar_bits,
        use_qjl=layer.tq_config.use_qjl,
    )
    k_out, v_out = turboquant_decode_v2(
        encoded,
        layer.tq_R_T,
        use_qjl=layer.tq_config.use_qjl,
    )
    
    return k_out.reshape(orig_k_shape), v_out.reshape(orig_v_shape)
