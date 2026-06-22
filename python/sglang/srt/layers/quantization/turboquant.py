"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
(Zandieh, Daliri, Hadian, Mirrokni — Google Research, ICLR 2026, arXiv:2504.19874)

Paper-faithful implementation for SGLang KV cache quantization.

Keys:   TurboQuant_prod (Algorithm 2) — (b-1)-bit MSE + 1-bit QJL
Values: TurboQuant_mse  (Algorithm 1) — b-bit MSE

Default config (key_bits=4, value_bits=2, QJL off):
  Key:   4-bit MSE (64B) + norm (4B)                                  = 68 bytes/head
  Value: 2-bit MSE (32B) + norm (4B)                                  = 36 bytes/head
  Total: 104 bytes vs 512 fp16 = 4.9x compression
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= n."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p *= 2
    return p


# ============================================================================
# Pre-computed Lloyd-Max codebooks for Beta((d-1)/2, (d-1)/2) distribution.
# For d >= 64 this converges to N(0, 1/d).
# Format: {(head_dim, bits): ([centroids], [boundaries])}
# ============================================================================

_CODEBOOKS: Dict[Tuple[int, int], Tuple[List[float], List[float]]] = {
    # d=128 —— the most common head_dim
    (128, 1): (
        [-0.0627, 0.0627],
        [-1.0, 0.0, 1.0],
    ),
    (128, 2): (
        [-0.1330, -0.0400, 0.0400, 0.1330],
        [-1.0, -0.0865, 0.0, 0.0865, 1.0],
    ),
    (128, 3): (
        [-0.1884, -0.1181, -0.0666, -0.0216, 0.0216, 0.0666, 0.1181, 0.1884],
        [-1.0, -0.1533, -0.0924, -0.0441, 0.0, 0.0441, 0.0924, 0.1533, 1.0],
    ),
    (128, 4): (
        [
            -0.2415,
            -0.1828,
            -0.143,
            -0.111,
            -0.0833,
            -0.058,
            -0.0343,
            -0.0113,
            0.0113,
            0.0343,
            0.058,
            0.0833,
            0.111,
            0.143,
            0.1828,
            0.2415,
        ],
        [
            -1.0,
            -0.2122,
            -0.1629,
            -0.127,
            -0.0971,
            -0.0706,
            -0.0462,
            -0.0228,
            -0.0,
            0.0228,
            0.0462,
            0.0706,
            0.0971,
            0.127,
            0.1629,
            0.2122,
            1.0,
        ],
    ),
    # d=64
    (64, 2): (
        [-0.1878, -0.0564, 0.0564, 0.1878],
        [-1.0, -0.1221, 0.0, 0.1221, 1.0],
    ),
    (64, 3): (
        [-0.2661, -0.1667, -0.0940, -0.0305, 0.0305, 0.0940, 0.1667, 0.2661],
        [-1.0, -0.2164, -0.1304, -0.0623, 0.0, 0.0623, 0.1304, 0.2164, 1.0],
    ),
    # d=256
    (256, 2): (
        [-0.0941, -0.0283, 0.0283, 0.0941],
        [-1.0, -0.0612, 0.0, 0.0612, 1.0],
    ),
}


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache quantization.

    Default: K4/V2 (4-bit keys MSE, 2-bit values MSE).
    QJL is off by default — single-sample variance too high for 36+ layer models.
    V2 is nearly "free" by PPL (K/V norm ratio 52x-1274x) — ecosystem consensus.
    Mixed-precision (Section 2.3): outlier channels get +1 bit with
    independent rotations and codebooks.
    """

    key_bits: int = 4
    value_bits: int = 2
    enable_qjl: bool = False
    head_dim: int = 128
    # Mixed-precision outlier handling (Section 2.3)
    mixed_precision: bool = False
    n_outlier: int = 32
    # Online codebook from data (Section 4.1)
    use_online_codebook: bool = False
    # QJL score weight: 1.0 = unbiased (paper), <1.0 = lower variance
    qjl_score_weight: float = 1.0
    # Boundary layer protection: skip TQ for first/last N layers (store as bf16)
    protected_layers_head: int = 2
    protected_layers_tail: int = 2

    def __post_init__(self):
        min_key = 2 if self.enable_qjl else 1
        if self.key_bits < min_key:
            raise ValueError(
                f"key_bits must be >= {min_key} "
                f"({'with' if self.enable_qjl else 'without'} QJL), got {self.key_bits}"
            )
        if self.value_bits < 1:
            raise ValueError(f"value_bits must be >= 1, got {self.value_bits}")
        if self.head_dim & (self.head_dim - 1) != 0:
            raise ValueError(
                f"head_dim must be a power of 2 for FWHT, got {self.head_dim}"
            )
        if self.mixed_precision:
            if self.n_outlier < 1:
                raise ValueError(f"n_outlier must be >= 1, got {self.n_outlier}")
            if self.n_outlier >= self.head_dim:
                raise ValueError(
                    f"n_outlier ({self.n_outlier}) must be < head_dim ({self.head_dim})"
                )

    # --- Uniform-mode properties (existing) ---

    @property
    def key_mse_bits(self) -> int:
        """MSE quantization bits for keys."""
        return self.key_bits - 1 if self.enable_qjl else self.key_bits

    @property
    def key_mse_packed_dim(self) -> int:
        """Bytes per head for packed key MSE indices (d=128: 32 for 2-bit)."""
        b = self.key_mse_bits
        if b <= 2:
            return self.head_dim * b // 8
        return self.head_dim // 2  # 4-bit nibble slots for 3+ bit

    @property
    def key_qjl_packed_dim(self) -> int:
        """Bytes per head for packed QJL signs (d=128: 16)."""
        return self.head_dim // 8 if self.enable_qjl else 0

    @property
    def value_packed_dim(self) -> int:
        """Bytes per head for packed value MSE indices (d=128: 32 for 2-bit)."""
        b = self.value_bits
        if b <= 2:
            return self.head_dim * b // 8
        return self.head_dim // 2

    # --- Mixed-precision properties ---

    @property
    def n_regular(self) -> int:
        return self.head_dim - self.n_outlier

    @property
    def n_outlier_padded(self) -> int:
        return _next_pow2(self.n_outlier)

    @property
    def n_regular_padded(self) -> int:
        return _next_pow2(self.n_regular)

    @property
    def key_mse_bits_outlier(self) -> int:
        """Outlier channels get +1 bit over regular."""
        return self.key_mse_bits + 1

    @property
    def key_mse_bits_regular(self) -> int:
        return self.key_mse_bits

    @staticmethod
    def _packed_dim(n_values: int, bits: int) -> int:
        if bits <= 2:
            return n_values * bits // 8
        return n_values // 2  # 4-bit nibble slots

    @property
    def key_outlier_packed_dim(self) -> int:
        return self._packed_dim(self.n_outlier_padded, self.key_mse_bits_outlier)

    @property
    def key_regular_packed_dim(self) -> int:
        return self._packed_dim(self.n_regular_padded, self.key_mse_bits_regular)

    @property
    def value_bits_outlier(self) -> int:
        return self.value_bits + 1

    @property
    def value_outlier_packed_dim(self) -> int:
        return self._packed_dim(self.n_outlier_padded, self.value_bits_outlier)

    @property
    def value_regular_packed_dim(self) -> int:
        return self._packed_dim(self.n_regular_padded, self.value_bits)


@dataclass
class MixedPrecisionInfo:
    """Per (layer, head) mixed-precision calibration state.

    Created lazily on first encode, cached for all subsequent calls.
    Contains separate rotation signs and codebooks for outlier/regular subsets.
    """

    outlier_idx: torch.Tensor  # [n_outlier] original-space channel indices
    regular_idx: torch.Tensor  # [d - n_outlier] original-space channel indices
    # Key codebooks
    key_outlier_centroids: torch.Tensor  # [K_outlier]
    key_outlier_inner: torch.Tensor  # [K_outlier - 1] inner boundaries
    key_regular_centroids: torch.Tensor  # [K_regular]
    key_regular_inner: torch.Tensor  # [K_regular - 1] inner boundaries
    # Value codebooks
    val_outlier_centroids: torch.Tensor
    val_outlier_inner: torch.Tensor
    val_regular_centroids: torch.Tensor
    val_regular_inner: torch.Tensor
    # Rotation signs (separate per subset)
    outlier_signs: torch.Tensor  # [n_outlier_padded]
    regular_signs: torch.Tensor  # [n_regular_padded]
    n_outlier_padded: int
    n_regular_padded: int


# ============================================================================
# Per-model state
# ============================================================================


class TurboQuantState:
    """Holds codebooks, rotation signs, and QJL projection matrix.

    Created once at pool initialization, shared across all forward passes.
    """

    def __init__(
        self,
        config: TurboQuantConfig,
        layer_num: int,
        head_num: int,
        device: str,
        seed: int = 42,
    ):
        self.config = config
        self.device = device
        d = config.head_dim

        # Codebooks
        self.key_centroids, self.key_boundaries = _get_codebook(
            d, config.key_mse_bits, device
        )
        self.value_centroids, self.value_boundaries = _get_codebook(
            d, config.value_bits, device
        )
        # Inner boundaries for torch.bucketize (without -1.0 / +1.0 endpoints)
        self.key_inner = self.key_boundaries[1:-1].contiguous()
        self.value_inner = self.value_boundaries[1:-1].contiguous()

        # Rotation signs: [layer_num, 2, head_num, d] — dim 1: 0=keys, 1=values
        # Stored as float32 to avoid repeated int→float casts on the hot path.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        raw = torch.randint(0, 2, (layer_num, 2, head_num, d), generator=gen)
        self.rotation_signs = (raw.float() * 2 - 1).to(device)

        # QJL projection matrix S (d × d, i.i.d. N(0,1))
        if config.enable_qjl:
            gen_qjl = torch.Generator(device="cpu")
            gen_qjl.manual_seed(seed + 1_000_000)
            self.qjl_matrix = torch.randn(
                d, d, generator=gen_qjl, dtype=torch.float32
            ).to(device)
        else:
            self.qjl_matrix = None

        logger.info(
            "TurboQuant state: %d layers, %d heads, d=%d, "
            "key=%d-bit MSE%s, value=%d-bit MSE",
            layer_num,
            head_num,
            d,
            config.key_mse_bits,
            " + 1-bit QJL" if config.enable_qjl else "",
            config.value_bits,
        )

        # Mixed-precision lazy cache: (layer_idx, head_idx) -> MixedPrecisionInfo
        self._mixed_cache: Dict[Tuple[int, int], MixedPrecisionInfo] = {}
        self._seed = seed
        self._layer_num = layer_num
        self._head_num = head_num

    def get_mixed_info(
        self,
        layer_idx: int,
        head_idx: int,
        calibration_data: Optional[torch.Tensor] = None,
    ) -> Optional["MixedPrecisionInfo"]:
        """Get or create mixed-precision info for a (layer, head) pair.

        On first call per (layer, head), calibration_data must be provided
        to detect outlier channels. Subsequent calls return the cached result.
        """
        if not self.config.mixed_precision:
            return None

        key = (layer_idx, head_idx)
        if key in self._mixed_cache:
            return self._mixed_cache[key]

        if calibration_data is None:
            return None

        cfg = self.config
        d = cfg.head_dim
        outlier_idx, regular_idx = detect_outlier_channels(
            calibration_data, cfg.n_outlier
        )
        n_out = outlier_idx.numel()
        n_reg = regular_idx.numel()
        n_out_p = _next_pow2(n_out)
        n_reg_p = _next_pow2(n_reg)

        # Deterministic seeds per (layer, head, subset)
        base = self._seed + layer_idx * 100003 + head_idx * 999979
        gen_o = torch.Generator(device="cpu")
        gen_o.manual_seed(base ^ 0x13572468)
        outlier_signs = (
            torch.randint(0, 2, (n_out_p,), generator=gen_o).float() * 2 - 1
        ).to(self.device)

        gen_r = torch.Generator(device="cpu")
        gen_r.manual_seed(base ^ 0x24681357)
        regular_signs = (
            torch.randint(0, 2, (n_reg_p,), generator=gen_r).float() * 2 - 1
        ).to(self.device)

        # Key codebooks for each subset
        ko_c, ko_b = _get_codebook(n_out_p, cfg.key_mse_bits_outlier, self.device)
        kr_c, kr_b = _get_codebook(n_reg_p, cfg.key_mse_bits_regular, self.device)
        # Value codebooks for each subset
        vo_c, vo_b = _get_codebook(n_out_p, cfg.value_bits_outlier, self.device)
        vr_c, vr_b = _get_codebook(n_reg_p, cfg.value_bits, self.device)

        # Online codebook override (Section 4.1)
        if cfg.use_online_codebook:
            cal_flat = calibration_data.reshape(-1, d).float()
            # Key outlier: extract, normalize, pad, rotate, compute codebook
            x_out = cal_flat[:, outlier_idx]
            x_out_norm = x_out.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            x_out_unit = x_out / x_out_norm
            if n_out_p > n_out:
                x_out_unit = torch.nn.functional.pad(x_out_unit, (0, n_out_p - n_out))
            y_out = fwht(x_out_unit * outlier_signs) * (1.0 / math.sqrt(n_out_p))
            ko_c, ko_b = compute_online_codebook(y_out, cfg.key_mse_bits_outlier)
            # Key regular: extract, normalize, pad, rotate, compute codebook
            x_reg = cal_flat[:, regular_idx]
            x_reg_norm = x_reg.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            x_reg_unit = x_reg / x_reg_norm
            if n_reg_p > n_reg:
                x_reg_unit = torch.nn.functional.pad(x_reg_unit, (0, n_reg_p - n_reg))
            y_reg = fwht(x_reg_unit * regular_signs) * (1.0 / math.sqrt(n_reg_p))
            kr_c, kr_b = compute_online_codebook(y_reg, cfg.key_mse_bits_regular)
            # Value codebooks from same calibration
            vo_c, vo_b = compute_online_codebook(y_out, cfg.value_bits_outlier)
            vr_c, vr_b = compute_online_codebook(y_reg, cfg.value_bits)

        info = MixedPrecisionInfo(
            outlier_idx=outlier_idx.to(self.device),
            regular_idx=regular_idx.to(self.device),
            key_outlier_centroids=ko_c,
            key_outlier_inner=ko_b[1:-1].contiguous(),
            key_regular_centroids=kr_c,
            key_regular_inner=kr_b[1:-1].contiguous(),
            val_outlier_centroids=vo_c,
            val_outlier_inner=vo_b[1:-1].contiguous(),
            val_regular_centroids=vr_c,
            val_regular_inner=vr_b[1:-1].contiguous(),
            outlier_signs=outlier_signs,
            regular_signs=regular_signs,
            n_outlier_padded=n_out_p,
            n_regular_padded=n_reg_p,
        )
        self._mixed_cache[key] = info
        return info


# ============================================================================
# Codebook utilities
# ============================================================================


def _get_codebook(
    head_dim: int, bits: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (head_dim, bits)
    if key in _CODEBOOKS:
        c, b = _CODEBOOKS[key]
        return (
            torch.tensor(c, dtype=torch.float32, device=device),
            torch.tensor(b, dtype=torch.float32, device=device),
        )
    # Fallback: compute Lloyd-Max for N(0, 1/d) — not pre-computed for this config
    logger.warning(
        "No pre-computed TurboQuant codebook for (head_dim=%d, bits=%d). "
        "Computing Lloyd-Max from Gaussian approximation. "
        "Consider adding to _CODEBOOKS for production use.",
        head_dim,
        bits,
    )
    sigma = 1.0 / math.sqrt(head_dim)
    c, b = _lloyd_max_gaussian(sigma, 1 << bits)
    _CODEBOOKS[key] = (c, b)
    return (
        torch.tensor(c, dtype=torch.float32, device=device),
        torch.tensor(b, dtype=torch.float32, device=device),
    )


def _lloyd_max_gaussian(
    sigma: float, n_levels: int, max_iter: int = 200
) -> Tuple[List[float], List[float]]:
    """Compute Lloyd-Max optimal quantizer for N(0, sigma^2). No scipy."""
    # Initialize at Gaussian quantile positions
    centroids = [
        sigma * math.sqrt(2) * _erfinv_approx(2 * (i + 0.5) / n_levels - 1)
        for i in range(n_levels)
    ]

    for _ in range(max_iter):
        bounds = [-1.0]
        for i in range(len(centroids) - 1):
            bounds.append((centroids[i] + centroids[i + 1]) / 2)
        bounds.append(1.0)

        # E[X | a < X < b] for X ~ N(0, sigma^2)
        # = sigma * (phi(a/s) - phi(b/s)) / (Phi(b/s) - Phi(a/s))
        new_c = []
        for i in range(n_levels):
            a, b = bounds[i], bounds[i + 1]
            an, bn = a / sigma, b / sigma
            phi_a = math.exp(-0.5 * an * an) / math.sqrt(2 * math.pi)
            phi_b = math.exp(-0.5 * bn * bn) / math.sqrt(2 * math.pi)
            Phi_a = 0.5 * (1 + math.erf(an / math.sqrt(2)))
            Phi_b = 0.5 * (1 + math.erf(bn / math.sqrt(2)))
            den = Phi_b - Phi_a
            new_c.append(sigma * (phi_a - phi_b) / den if den > 1e-15 else (a + b) / 2)
        centroids = new_c

    # Final boundaries
    bounds = [-1.0]
    for i in range(len(centroids) - 1):
        bounds.append((centroids[i] + centroids[i + 1]) / 2)
    bounds.append(1.0)
    return centroids, bounds


def _erfinv_approx(x: float) -> float:
    """Rough inverse error function (for Lloyd-Max initialization only)."""
    if abs(x) >= 1.0:
        x = 0.9999 * (1 if x > 0 else -1)
    a = 0.147
    ln = math.log(1 - x * x)
    t1 = 2 / (math.pi * a) + ln / 2
    t2 = ln / a
    sign = 1 if x >= 0 else -1
    return sign * math.sqrt(math.sqrt(t1 * t1 - t2) - t1)


# ============================================================================
# Outlier channel detection (Section 2.3)
# ============================================================================


def detect_outlier_channels(
    x: torch.Tensor, n_outlier: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Detect outlier channels by variance in original space (Section 2.3).

    The paper: "splitting channels into outlier and non-outlier sets, and
    applying two independent instances of TurboQuant to each, allocating
    higher bit precision to outliers."

    Args:
        x: [..., D] vectors (any batch dims are flattened)
        n_outlier: number of channels to mark as outliers

    Returns:
        outlier_indices: [n_outlier] sorted channel indices (highest variance)
        regular_indices: [D - n_outlier] sorted remaining channel indices
    """
    d = x.shape[-1]
    flat = x.reshape(-1, d).float()
    n_outlier = min(n_outlier, d - 1)

    if flat.shape[0] > 1:
        channel_var = flat.var(dim=0, unbiased=False)
    else:
        channel_var = flat.pow(2).squeeze(0)

    _, sorted_idx = channel_var.sort(descending=True)
    outlier_indices = sorted_idx[:n_outlier].sort().values
    regular_indices = sorted_idx[n_outlier:].sort().values
    return outlier_indices, regular_indices


def compute_online_codebook(
    data: torch.Tensor, bits: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build codebook from actual rotated data (Section 4.1).

    The paper: "online approach requires additional clustering computation
    during every prefill stage, this one-time cost is offset by improved
    performance compared to the offline approach."

    Args:
        data: [..., d] rotated coordinate values (flattened for 1D k-means)
        bits: bit width

    Returns:
        centroids:  [K] float32 tensor
        boundaries: [K+1] float32 tensor (full boundaries including endpoints)
    """
    flat = data.reshape(-1).float()
    sigma = flat.std().item()
    if sigma < 1e-12:
        sigma = 1.0 / math.sqrt(max(data.shape[-1], 1))
    c_list, b_list = _lloyd_max_gaussian(sigma, 1 << bits)
    return (
        torch.tensor(c_list, dtype=torch.float32, device=data.device),
        torch.tensor(b_list, dtype=torch.float32, device=data.device),
    )


# ============================================================================
# Fast Walsh-Hadamard Transform & Randomized Hadamard Transform
# ============================================================================


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform along the last dimension.

    Complexity: O(d log d) where d = x.shape[-1] (must be power of 2).
    """
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, f"FWHT requires power-of-2 dim, got {d}"
    batch_shape = x.shape[:-1]
    h = 1
    while h < d:
        x = x.reshape(*batch_shape, d // (2 * h), 2, h)
        a = x[..., 0, :]
        b = x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2).reshape(*batch_shape, d)
        h *= 2
    return x


def rht_forward(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """RHT forward: y = (1/√d) · H · diag(signs) · x"""
    d = x.shape[-1]
    return fwht(x * signs) * (1.0 / math.sqrt(d))


def rht_inverse(y: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """RHT inverse: x = (1/√d) · diag(signs) · H · y

    Since H is symmetric and H·H = d·I, the inverse of
    (1/√d)·H·D is D·(1/√d)·H (where D = diag(signs), D^-1 = D).
    """
    d = y.shape[-1]
    return fwht(y) * signs * (1.0 / math.sqrt(d))


# ============================================================================
# Bit Packing / Unpacking
# ============================================================================


def pack_2bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 2-bit indices: 4 values per byte.
    [..., D] → [..., D//4] uint8
    """
    D = indices.shape[-1]
    idx = indices.to(torch.uint8).reshape(*indices.shape[:-1], D // 4, 4)
    return idx[..., 0] | (idx[..., 1] << 2) | (idx[..., 2] << 4) | (idx[..., 3] << 6)


def unpack_2bit(packed: torch.Tensor, D: int) -> torch.Tensor:
    """Unpack 2-bit indices: byte → 4 values.
    [..., D//4] uint8 → [..., D] uint8
    """
    out = torch.empty(*packed.shape[:-1], D, dtype=torch.uint8, device=packed.device)
    out[..., 0::4] = packed & 0x03
    out[..., 1::4] = (packed >> 2) & 0x03
    out[..., 2::4] = (packed >> 4) & 0x03
    out[..., 3::4] = (packed >> 6) & 0x03
    return out


def pack_1bit(signs: torch.Tensor) -> torch.Tensor:
    """Pack 1-bit values: 8 per byte.
    [..., D] → [..., D//8] uint8
    """
    D = signs.shape[-1]
    s = signs.to(torch.uint8).reshape(*signs.shape[:-1], D // 8, 8)
    return (
        s[..., 0]
        | (s[..., 1] << 1)
        | (s[..., 2] << 2)
        | (s[..., 3] << 3)
        | (s[..., 4] << 4)
        | (s[..., 5] << 5)
        | (s[..., 6] << 6)
        | (s[..., 7] << 7)
    )


def unpack_1bit(packed: torch.Tensor, D: int) -> torch.Tensor:
    """Unpack 1-bit values: byte → 8 values.
    [..., D//8] uint8 → [..., D] uint8
    """
    out = torch.empty(*packed.shape[:-1], D, dtype=torch.uint8, device=packed.device)
    for i in range(8):
        out[..., i::8] = (packed >> i) & 0x01
    return out


def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack into 4-bit nibbles: 2 per byte. For 3-bit values in 4-bit slots.
    [..., D] → [..., D//2] uint8
    """
    D = indices.shape[-1]
    idx = indices.to(torch.uint8).reshape(*indices.shape[:-1], D // 2, 2)
    return idx[..., 0] | (idx[..., 1] << 4)


def unpack_4bit(packed: torch.Tensor, D: int) -> torch.Tensor:
    """Unpack 4-bit nibbles: byte → 2 values.
    [..., D//2] uint8 → [..., D] uint8
    """
    out = torch.empty(*packed.shape[:-1], D, dtype=torch.uint8, device=packed.device)
    out[..., 0::2] = packed & 0x0F
    out[..., 1::2] = (packed >> 4) & 0x0F
    return out


def _pack(indices: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 1:
        return pack_1bit(indices)
    elif bits == 2:
        return pack_2bit(indices)
    return pack_4bit(indices)


def _unpack(packed: torch.Tensor, bits: int, D: int) -> torch.Tensor:
    if bits == 1:
        return unpack_1bit(packed, D)
    elif bits == 2:
        return unpack_2bit(packed, D)
    return unpack_4bit(packed, D)


# ============================================================================
# Encode / Decode  —  the core TurboQuant algorithms
# ============================================================================


def encode_keys(
    k: torch.Tensor,
    layer_idx: int,
    state: TurboQuantState,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """TurboQuant_prod encoding for keys (Algorithm 2 from paper).

    Args:
        k: [N, H, D] key vectors (bf16/fp16)
        layer_idx: buffer-relative layer index (layer_id - start_layer)
        state: shared TurboQuantState

    Returns:
        packed_mse:  [N, H, packed_dim] uint8 — quantized MSE indices
        norms:       [N, H, 1]          fp32  — key vector norms
        packed_qjl:  [N, H, D//8]       uint8 — QJL sign bits  (None if QJL off)
        r_norms:     [N, H, 1]          fp32  — residual norms  (None if QJL off)
    """
    cfg = state.config
    d = cfg.head_dim
    signs_k = state.rotation_signs[layer_idx, 0]  # [H, D]

    # 1) Norm (FP32 — FP16 corrupts at ~11.4K tokens)
    k_f = k.float()
    norms = k_f.norm(dim=-1, keepdim=True)  # [N, H, 1]

    # 2) Unit vector
    k_unit = k_f / norms.clamp(min=1e-12)

    # 3) Randomized Hadamard Transform
    y = rht_forward(k_unit, signs_k)  # [N, H, D]

    # 4) Lloyd-Max scalar quantization per coordinate
    indices = torch.bucketize(y.contiguous(), state.key_inner)  # [N, H, D]

    # 5) Pack MSE indices
    packed_mse = _pack(indices, cfg.key_mse_bits)

    if not cfg.enable_qjl:
        return packed_mse, norms, None, None

    # ---- QJL correction (1-bit, Algorithm 2 step) ----

    # 6) Compute residual: r = k - dequant_mse(k)
    # Must use norm-corrected reconstruction to match decode_keys()
    y_hat = state.key_centroids[indices.long()]  # centroid lookup
    y_hat = y_hat / y_hat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    k_hat_unit = rht_inverse(y_hat, signs_k)
    residual = k_f - k_hat_unit * norms

    # 7) Residual norm (FP32)
    r_norms = residual.norm(dim=-1, keepdim=True)
    r_unit = residual / r_norms.clamp(min=1e-12)

    # 8) QJL projection: sign(S · r_unit)
    #    S is [D, D]; r_unit is [N, H, D]
    #    matmul broadcasts: [N, H, D] @ [D, D] → [N, H, D]
    qjl_proj = torch.matmul(r_unit, state.qjl_matrix.T)
    qjl_signs = (qjl_proj >= 0).to(torch.uint8)

    # 9) Pack 1-bit signs
    packed_qjl = pack_1bit(qjl_signs)

    return packed_mse, norms, packed_qjl, r_norms


def decode_keys(
    packed_mse: torch.Tensor,
    norms: torch.Tensor,
    packed_qjl: Optional[torch.Tensor],
    r_norms: Optional[torch.Tensor],
    layer_idx: int,
    state: TurboQuantState,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode keys from TurboQuant compressed representation.

    Returns: [*, H, D] tensor in output_dtype.
    """
    cfg = state.config
    d = cfg.head_dim
    signs_k = state.rotation_signs[layer_idx, 0]

    # MSE reconstruction
    indices = _unpack(packed_mse, cfg.key_mse_bits, d)
    y_hat = state.key_centroids[indices.long()]
    # Norm correction: renormalize in rotated domain before inverse RHT
    # (reduces compound quantization error across d dimensions)
    y_hat = y_hat / y_hat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    k_hat_unit = rht_inverse(y_hat, signs_k)
    k_hat = k_hat_unit * norms

    # QJL correction: x̃_qjl = (√(π/2)/d) · γ · S^T · signs
    if packed_qjl is not None and r_norms is not None and state.qjl_matrix is not None:
        qjl_signs = unpack_1bit(packed_qjl, d)
        qjl_float = qjl_signs.float() * 2 - 1  # {0,1} → {-1,+1}
        # S^T · signs: [*, H, D] @ [D, D] → [*, H, D]
        qjl_recon = torch.matmul(qjl_float, state.qjl_matrix)
        scale = math.sqrt(math.pi / 2) / d * state.config.qjl_score_weight
        k_hat = k_hat + scale * r_norms * qjl_recon

    return k_hat.to(output_dtype)


def encode_values(
    v: torch.Tensor,
    layer_idx: int,
    state: TurboQuantState,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """TurboQuant_mse encoding for values (Algorithm 1 from paper).

    Args:
        v: [N, H, D] value vectors
        layer_idx: buffer-relative layer index

    Returns:
        packed_mse: [N, H, packed_dim] uint8
        norms:      [N, H, 1]          fp32
    """
    cfg = state.config
    signs_v = state.rotation_signs[layer_idx, 1]  # [H, D]

    v_f = v.float()
    norms = v_f.norm(dim=-1, keepdim=True)
    v_unit = v_f / norms.clamp(min=1e-12)
    y = rht_forward(v_unit, signs_v)
    indices = torch.bucketize(y.contiguous(), state.value_inner)
    packed = _pack(indices, cfg.value_bits)
    return packed, norms


def decode_values(
    packed_mse: torch.Tensor,
    norms: torch.Tensor,
    layer_idx: int,
    state: TurboQuantState,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode values from TurboQuant_mse representation.

    Returns: [*, H, D] tensor in output_dtype.
    """
    cfg = state.config
    d = cfg.head_dim
    signs_v = state.rotation_signs[layer_idx, 1]

    indices = _unpack(packed_mse, cfg.value_bits, d)
    y_hat = state.value_centroids[indices.long()]
    # Norm correction: renormalize in rotated domain before inverse RHT
    y_hat = y_hat / y_hat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    v_hat_unit = rht_inverse(y_hat, signs_v)
    v_hat = v_hat_unit * norms
    return v_hat.to(output_dtype)


# ============================================================================
# Mixed-precision encode/decode (Section 2.3)
# ============================================================================


def _encode_subset(
    x_subset: torch.Tensor,
    signs: torch.Tensor,
    inner_boundaries: torch.Tensor,
    n_padded: int,
    bits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a channel subset: normalize -> pad -> RHT -> quantize -> pack.

    Args:
        x_subset: [N, H, n_channels] subset of channels
        signs:    [n_padded] rotation signs for this subset
        inner_boundaries: [K-1] inner codebook boundaries
        n_padded: padded dimension for Hadamard
        bits:     MSE bit-width for this subset

    Returns:
        packed:  [N, H, packed_dim] uint8
        norms:   [N, H, 1] fp32
    """
    x_f = x_subset.float()
    norms = x_f.norm(dim=-1, keepdim=True)
    x_unit = x_f / norms.clamp(min=1e-12)

    n_actual = x_subset.shape[-1]
    if n_padded > n_actual:
        x_unit = torch.nn.functional.pad(x_unit, (0, n_padded - n_actual))

    y = rht_forward(x_unit, signs)
    indices = torch.bucketize(y.contiguous(), inner_boundaries)
    packed = _pack(indices, bits)
    return packed, norms


def _decode_subset(
    packed: torch.Tensor,
    norms: torch.Tensor,
    signs: torch.Tensor,
    centroids: torch.Tensor,
    n_padded: int,
    n_actual: int,
    bits: int,
) -> torch.Tensor:
    """Decode a channel subset: unpack -> dequant -> inverse RHT -> trim -> scale.

    Returns: [*, H, n_actual] float32
    """
    indices = _unpack(packed, bits, n_padded)
    y_hat = centroids[indices.long()]
    # Norm correction: renormalize in rotated domain before inverse RHT
    y_hat = y_hat / y_hat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    x_hat = rht_inverse(y_hat, signs)
    if n_padded > n_actual:
        x_hat = x_hat[..., :n_actual]
    return x_hat * norms


def encode_keys_mixed(
    k: torch.Tensor,
    layer_idx: int,
    state: "TurboQuantState",
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Mixed-precision key encoding (Section 2.3 + Algorithm 2).

    Splits channels into outlier/regular subsets, applies independent
    TurboQuant instances with separate rotations and codebooks.

    Args:
        k: [N, H, D] key vectors
        layer_idx: buffer-relative layer index
        state: TurboQuantState with mixed_precision=True

    Returns:
        outlier_packed: [N, H, outlier_packed_dim] uint8
        regular_packed: [N, H, regular_packed_dim] uint8
        outlier_norms:  [N, H, 1] fp32
        regular_norms:  [N, H, 1] fp32
        qjl_packed:     [N, H, D//8] uint8 (None if QJL off)
        r_norms:        [N, H, 1] fp32 (None if QJL off)
    """
    cfg = state.config
    N, H, D = k.shape

    # Calibrate per head using first batch
    for h in range(H):
        state.get_mixed_info(layer_idx, h, calibration_data=k[:, h, :])

    outlier_packed_list = []
    regular_packed_list = []
    outlier_norms_list = []
    regular_norms_list = []

    for h in range(H):
        info = state.get_mixed_info(layer_idx, h)
        k_h = k[:, h, :]  # [N, D]

        k_outlier = k_h[:, info.outlier_idx]  # [N, n_outlier]
        k_regular = k_h[:, info.regular_idx]  # [N, n_regular]

        op, on = _encode_subset(
            k_outlier.unsqueeze(1),
            info.outlier_signs,
            info.key_outlier_inner,
            info.n_outlier_padded,
            cfg.key_mse_bits_outlier,
        )
        rp, rn = _encode_subset(
            k_regular.unsqueeze(1),
            info.regular_signs,
            info.key_regular_inner,
            info.n_regular_padded,
            cfg.key_mse_bits_regular,
        )
        outlier_packed_list.append(op.squeeze(1))
        regular_packed_list.append(rp.squeeze(1))
        outlier_norms_list.append(on.squeeze(1))
        regular_norms_list.append(rn.squeeze(1))

    outlier_packed = torch.stack(outlier_packed_list, dim=1)
    regular_packed = torch.stack(regular_packed_list, dim=1)
    outlier_norms = torch.stack(outlier_norms_list, dim=1)
    regular_norms = torch.stack(regular_norms_list, dim=1)

    if not cfg.enable_qjl:
        return outlier_packed, regular_packed, outlier_norms, regular_norms, None, None

    # QJL on full residual in original space
    k_f = k.float()
    k_hat = _reassemble_mixed_keys(
        outlier_packed,
        regular_packed,
        outlier_norms,
        regular_norms,
        layer_idx,
        state,
    )
    residual = k_f - k_hat.float()
    r_norms = residual.norm(dim=-1, keepdim=True)
    r_unit = residual / r_norms.clamp(min=1e-12)
    qjl_proj = torch.matmul(r_unit, state.qjl_matrix.T)
    qjl_signs = (qjl_proj >= 0).to(torch.uint8)
    qjl_packed = pack_1bit(qjl_signs)

    return (
        outlier_packed,
        regular_packed,
        outlier_norms,
        regular_norms,
        qjl_packed,
        r_norms,
    )


def _reassemble_mixed_keys(
    outlier_packed: torch.Tensor,
    regular_packed: torch.Tensor,
    outlier_norms: torch.Tensor,
    regular_norms: torch.Tensor,
    layer_idx: int,
    state: "TurboQuantState",
) -> torch.Tensor:
    """Decode mixed-precision keys to [N, H, D] float32 (no QJL)."""
    cfg = state.config
    N, H = outlier_packed.shape[:2]
    D = cfg.head_dim
    result = torch.zeros(N, H, D, device=outlier_packed.device, dtype=torch.float32)

    for h in range(H):
        info = state.get_mixed_info(layer_idx, h)

        x_out = _decode_subset(
            outlier_packed[:, h : h + 1, :],
            outlier_norms[:, h : h + 1, :],
            info.outlier_signs,
            info.key_outlier_centroids,
            info.n_outlier_padded,
            info.outlier_idx.numel(),
            cfg.key_mse_bits_outlier,
        ).squeeze(1)

        x_reg = _decode_subset(
            regular_packed[:, h : h + 1, :],
            regular_norms[:, h : h + 1, :],
            info.regular_signs,
            info.key_regular_centroids,
            info.n_regular_padded,
            info.regular_idx.numel(),
            cfg.key_mse_bits_regular,
        ).squeeze(1)

        result[:, h, :][:, info.outlier_idx] = x_out
        result[:, h, :][:, info.regular_idx] = x_reg

    return result


def decode_keys_mixed(
    outlier_packed: torch.Tensor,
    regular_packed: torch.Tensor,
    outlier_norms: torch.Tensor,
    regular_norms: torch.Tensor,
    qjl_packed: Optional[torch.Tensor],
    r_norms: Optional[torch.Tensor],
    layer_idx: int,
    state: "TurboQuantState",
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode mixed-precision keys with QJL correction.

    Returns: [N, H, D] in output_dtype.
    """
    cfg = state.config
    k_hat = _reassemble_mixed_keys(
        outlier_packed,
        regular_packed,
        outlier_norms,
        regular_norms,
        layer_idx,
        state,
    )

    if qjl_packed is not None and r_norms is not None and state.qjl_matrix is not None:
        d = cfg.head_dim
        qjl_signs = unpack_1bit(qjl_packed, d)
        qjl_float = qjl_signs.float() * 2 - 1
        qjl_recon = torch.matmul(qjl_float, state.qjl_matrix)
        scale = math.sqrt(math.pi / 2) / d * cfg.qjl_score_weight
        k_hat = k_hat + scale * r_norms * qjl_recon

    return k_hat.to(output_dtype)


def encode_values_mixed(
    v: torch.Tensor,
    layer_idx: int,
    state: "TurboQuantState",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mixed-precision value encoding (Section 2.3, Algorithm 1).

    Returns:
        outlier_packed: [N, H, outlier_packed_dim]
        regular_packed: [N, H, regular_packed_dim]
        outlier_norms:  [N, H, 1]
        regular_norms:  [N, H, 1]
    """
    cfg = state.config
    N, H, D = v.shape

    for h in range(H):
        state.get_mixed_info(layer_idx, h, calibration_data=v[:, h, :])

    outlier_packed_list = []
    regular_packed_list = []
    outlier_norms_list = []
    regular_norms_list = []

    for h in range(H):
        info = state.get_mixed_info(layer_idx, h)

        v_outlier = v[:, h, :][:, info.outlier_idx].unsqueeze(1)
        v_regular = v[:, h, :][:, info.regular_idx].unsqueeze(1)

        op, on = _encode_subset(
            v_outlier,
            info.outlier_signs,
            info.val_outlier_inner,
            info.n_outlier_padded,
            cfg.value_bits_outlier,
        )
        rp, rn = _encode_subset(
            v_regular,
            info.regular_signs,
            info.val_regular_inner,
            info.n_regular_padded,
            cfg.value_bits,
        )
        outlier_packed_list.append(op.squeeze(1))
        regular_packed_list.append(rp.squeeze(1))
        outlier_norms_list.append(on.squeeze(1))
        regular_norms_list.append(rn.squeeze(1))

    return (
        torch.stack(outlier_packed_list, dim=1),
        torch.stack(regular_packed_list, dim=1),
        torch.stack(outlier_norms_list, dim=1),
        torch.stack(regular_norms_list, dim=1),
    )


def decode_values_mixed(
    outlier_packed: torch.Tensor,
    regular_packed: torch.Tensor,
    outlier_norms: torch.Tensor,
    regular_norms: torch.Tensor,
    layer_idx: int,
    state: "TurboQuantState",
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode mixed-precision values.

    Returns: [N, H, D] in output_dtype.
    """
    cfg = state.config
    N, H = outlier_packed.shape[:2]
    D = cfg.head_dim
    result = torch.zeros(N, H, D, device=outlier_packed.device, dtype=torch.float32)

    for h in range(H):
        info = state.get_mixed_info(layer_idx, h)

        x_out = _decode_subset(
            outlier_packed[:, h : h + 1, :],
            outlier_norms[:, h : h + 1, :],
            info.outlier_signs,
            info.val_outlier_centroids,
            info.n_outlier_padded,
            info.outlier_idx.numel(),
            cfg.value_bits_outlier,
        ).squeeze(1)

        x_reg = _decode_subset(
            regular_packed[:, h : h + 1, :],
            regular_norms[:, h : h + 1, :],
            info.regular_signs,
            info.val_regular_centroids,
            info.n_regular_padded,
            info.regular_idx.numel(),
            cfg.value_bits,
        ).squeeze(1)

        result[:, h, :][:, info.outlier_idx] = x_out
        result[:, h, :][:, info.regular_idx] = x_reg

    return result.to(output_dtype)
