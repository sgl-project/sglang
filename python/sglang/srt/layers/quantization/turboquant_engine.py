"""TurboQuant core engine: codebook, rotation, quantize, pack/unpack.

Implements the TurboQuant algorithm from "TurboQuant: Online Vector Quantization
with Near-optimal Distortion Rate" (ICLR 2026). Data-oblivious quantization via
random rotation + Lloyd-Max scalar quantization for N(0,1).

This module is framework-agnostic (no SGLang/vLLM dependencies).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Lloyd-Max codebook for N(0,1)
# ---------------------------------------------------------------------------

_CODEBOOK_CACHE: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}


def _compute_lloyd_max_gaussian(
    n_levels: int, n_iters: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Lloyd-Max optimal centroids and boundaries for N(0,1)."""
    from scipy.stats import norm

    boundaries = np.linspace(-3.5, 3.5, n_levels + 1)
    boundaries[0] = -1e10
    boundaries[-1] = 1e10
    centroids = np.zeros(n_levels)

    for _ in range(n_iters):
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            p = norm.cdf(hi) - norm.cdf(lo)
            if p > 1e-15:
                centroids[i] = (norm.pdf(lo) - norm.pdf(hi)) / p
            else:
                centroids[i] = (max(lo, -3.5) + min(hi, 3.5)) / 2

        for i in range(1, n_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

    return centroids, boundaries


def get_codebook(bit_width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get (centroids, inner_boundaries) for given bit-width, cached globally.

    Returns:
        centroids: (2^bit_width,) float32
        boundaries: (2^bit_width - 1,) float32  (inner boundaries only)
    """
    if bit_width not in _CODEBOOK_CACHE:
        n_levels = 2 ** bit_width
        centroids, boundaries = _compute_lloyd_max_gaussian(n_levels)
        _CODEBOOK_CACHE[bit_width] = (
            torch.tensor(centroids, dtype=torch.float32),
            torch.tensor(boundaries[1:-1], dtype=torch.float32),
        )
    return _CODEBOOK_CACHE[bit_width]


# ---------------------------------------------------------------------------
# Rotation matrix generation (Haar-distributed random orthogonal)
# ---------------------------------------------------------------------------

_ROTATION_CACHE: Dict[int, torch.Tensor] = {}


def generate_rotation_matrix(d: int, seed: int = 42) -> torch.Tensor:
    """Generate Haar-distributed random orthogonal matrix via QR of Gaussian.

    Cached by (d, seed) key. Result is always float32 on CPU;
    caller should .to(device) as needed.
    """
    key = (d, seed)
    if key not in _ROTATION_CACHE:
        gen = torch.Generator().manual_seed(seed)
        G = torch.randn(d, d, generator=gen)
        Q, R = torch.linalg.qr(G)
        diag_sign = torch.sign(torch.diag(R))
        diag_sign[diag_sign == 0] = 1.0
        Q = Q * diag_sign.unsqueeze(0)
        _ROTATION_CACHE[key] = Q
    return _ROTATION_CACHE[key]


def clear_rotation_cache():
    _ROTATION_CACHE.clear()


# ---------------------------------------------------------------------------
# Bit packing / unpacking (2, 3, 4 bit)
# ---------------------------------------------------------------------------


def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit indices (0-15) into uint8, 2 per byte.
    Layout: byte = lo_nibble | (hi_nibble << 4).
    """
    assert indices.shape[-1] % 2 == 0, "Last dim must be even"
    lo = indices[..., 0::2].to(torch.uint8)
    hi = indices[..., 1::2].to(torch.uint8)
    return lo | (hi << 4)


def unpack_4bit(packed: torch.Tensor, orig_last_dim: int) -> torch.Tensor:
    """Unpack uint8 -> 4-bit indices as int32."""
    lo = (packed & 0x0F).to(torch.int32)
    hi = ((packed >> 4) & 0x0F).to(torch.int32)
    result = torch.stack([lo, hi], dim=-1)
    return result.reshape(*packed.shape[:-1], packed.shape[-1] * 2)[
        ..., :orig_last_dim
    ]


def pack_2bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 2-bit indices (0-3) into uint8, 4 per byte."""
    d = indices.shape[-1]
    assert d % 4 == 0, f"Last dim must be multiple of 4, got {d}"
    idx = indices.to(torch.uint8)
    return (idx[..., 0::4]
            | (idx[..., 1::4] << 2)
            | (idx[..., 2::4] << 4)
            | (idx[..., 3::4] << 6))


def unpack_2bit(packed: torch.Tensor, orig_last_dim: int) -> torch.Tensor:
    """Unpack uint8 -> 2-bit indices as int32."""
    v0 = (packed & 0x03).to(torch.int32)
    v1 = ((packed >> 2) & 0x03).to(torch.int32)
    v2 = ((packed >> 4) & 0x03).to(torch.int32)
    v3 = ((packed >> 6) & 0x03).to(torch.int32)
    result = torch.stack([v0, v1, v2, v3], dim=-1)
    return result.reshape(*packed.shape[:-1], packed.shape[-1] * 4)[
        ..., :orig_last_dim
    ]


def pack_3bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 3-bit indices (0-7) into uint8. Every 8 values → 3 bytes (24 bits).
    Last dim must be a multiple of 8.
    """
    d = indices.shape[-1]
    assert d % 8 == 0, f"Last dim must be multiple of 8, got {d}"
    idx = indices.to(torch.uint8)
    # Reshape to groups of 8
    shape = idx.shape[:-1]
    idx = idx.reshape(*shape, d // 8, 8)
    v = [idx[..., i] for i in range(8)]
    b0 = v[0] | (v[1] << 3) | ((v[2] & 0x03) << 6)
    b1 = (v[2] >> 2) | (v[3] << 1) | (v[4] << 4) | ((v[5] & 0x01) << 7)
    b2 = (v[5] >> 1) | (v[6] << 2) | (v[7] << 5)
    return torch.stack([b0, b1, b2], dim=-1).reshape(*shape, d * 3 // 8)


def unpack_3bit(packed: torch.Tensor, orig_last_dim: int) -> torch.Tensor:
    """Unpack 3-bit packed bytes → int32 indices. 3 bytes → 8 values."""
    shape = packed.shape[:-1]
    n_groups = packed.shape[-1] // 3
    p = packed.reshape(*shape, n_groups, 3)
    b0, b1, b2 = p[..., 0].to(torch.int32), p[..., 1].to(torch.int32), p[..., 2].to(torch.int32)
    v0 = b0 & 0x07
    v1 = (b0 >> 3) & 0x07
    v2 = ((b0 >> 6) | (b1 << 2)) & 0x07
    v3 = (b1 >> 1) & 0x07
    v4 = (b1 >> 4) & 0x07
    v5 = ((b1 >> 7) | (b2 << 1)) & 0x07
    v6 = (b2 >> 2) & 0x07
    v7 = (b2 >> 5) & 0x07
    result = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1)
    return result.reshape(*shape, n_groups * 8)[..., :orig_last_dim]


def pack_indices(indices: torch.Tensor, bit_width: int) -> torch.Tensor:
    """Dispatch to the right packing function based on bit_width."""
    if bit_width == 4:
        return pack_4bit(indices)
    elif bit_width == 3:
        return pack_3bit(indices)
    elif bit_width == 2:
        return pack_2bit(indices)
    raise ValueError(f"Unsupported bit_width: {bit_width}")


def unpack_indices(packed: torch.Tensor, orig_last_dim: int, bit_width: int) -> torch.Tensor:
    """Dispatch to the right unpacking function based on bit_width."""
    if bit_width == 4:
        return unpack_4bit(packed, orig_last_dim)
    elif bit_width == 3:
        return unpack_3bit(packed, orig_last_dim)
    elif bit_width == 2:
        return unpack_2bit(packed, orig_last_dim)
    raise ValueError(f"Unsupported bit_width: {bit_width}")


def packed_bytes_per_dim(n_dims: int, bit_width: int) -> int:
    """Calculate packed byte count for n_dims at given bit_width."""
    if bit_width == 4:
        return n_dims // 2
    elif bit_width == 3:
        return n_dims * 3 // 8
    elif bit_width == 2:
        return n_dims // 4
    raise ValueError(f"Unsupported bit_width: {bit_width}")


def pad_for_packing(n_dims: int, bit_width: int) -> int:
    """Return padded dim that aligns to packing boundary."""
    if bit_width == 4:
        return n_dims + (n_dims % 2)
    elif bit_width == 3:
        return n_dims + (8 - n_dims % 8) % 8
    elif bit_width == 2:
        return n_dims + (4 - n_dims % 4) % 4
    raise ValueError(f"Unsupported bit_width: {bit_width}")


# ---------------------------------------------------------------------------
# Mixed bit-width (outlier treatment) for 2.5-bit and 3.5-bit
# ---------------------------------------------------------------------------


def mixed_bit_config(effective_bits: float, n_groups: int) -> list:
    """Return per-group bit-width list for mixed-precision quantization.

    2.5-bit: 25% groups at 3-bit, 75% at 2-bit
    3.5-bit: 50% groups at 4-bit, 50% at 3-bit

    Since TurboQuant rotation makes all groups i.i.d., the assignment
    is arbitrary (no calibration needed). First groups get higher bits.
    """
    if effective_bits == 2.5:
        n_hi = max(1, round(n_groups * 0.25))
        return [3] * n_hi + [2] * (n_groups - n_hi)
    elif effective_bits == 3.5:
        n_hi = max(1, round(n_groups * 0.5))
        return [4] * n_hi + [3] * (n_groups - n_hi)
    else:
        bw = int(effective_bits)
        return [bw] * n_groups


def mixed_compressed_bytes(kv_lora_rank: int, group_size: int,
                           qk_rope_head_dim: int, group_bits: list,
                           use_qjl: bool = False) -> int:
    """Calculate total compressed bytes for mixed-bit layout."""
    total = 0
    for g, bw in enumerate(group_bits):
        g_start = g * group_size
        g_end = min(g_start + group_size, kv_lora_rank)
        g_dim = g_end - g_start
        total += packed_bytes_per_dim(g_dim, bw)
    total += len(group_bits) * 2  # FP16 norms
    if use_qjl:
        total += kv_lora_rank // 8 + 2  # signs + rnorm
    total += qk_rope_head_dim * 2  # FP16 rope
    return total


def mixed_compress_latent(latent: torch.Tensor, group_bits: list,
                          group_size: int, rotations: dict,
                          device: torch.device) -> tuple:
    """Compress latent with per-group variable bit-width.

    Returns (list_of_packed, norms_tensor, latent_mse_or_None).
    """
    T = latent.shape[0]
    all_packed = []
    all_norms = []
    latent_mse = torch.zeros_like(latent)

    for g, bw in enumerate(group_bits):
        g_start = g * group_size
        g_end = min(g_start + group_size, latent.shape[1])
        g_dim = g_end - g_start
        n_levels = 2 ** bw

        centroids, boundaries = get_codebook(bw)
        centroids = centroids.to(device)
        boundaries = boundaries.to(device)

        L_g = latent[:, g_start:g_end].float()
        norms = L_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        L_norm = L_g / norms
        all_norms.append(norms.squeeze(1))

        Pi = rotations[g_start]
        Y = L_norm @ Pi.T * math.sqrt(g_dim)

        indices = torch.searchsorted(boundaries, Y.reshape(-1))
        indices = indices.clamp(0, n_levels - 1).reshape(T, g_dim)

        Y_hat = centroids[indices] / math.sqrt(g_dim)
        latent_mse[:, g_start:g_end] = (Y_hat @ Pi) * norms

        padded = pad_for_packing(g_dim, bw)
        if padded > g_dim:
            indices = torch.nn.functional.pad(indices, (0, padded - g_dim), value=0)
        packed = pack_indices(indices, bw)
        all_packed.append(packed)

    norms_tensor = torch.stack(all_norms, dim=1).half()
    return all_packed, norms_tensor, latent_mse


def mixed_decompress_latent(all_packed: list, norms: torch.Tensor,
                            group_bits: list, group_size: int,
                            kv_lora_rank: int, rotations: dict,
                            device: torch.device) -> torch.Tensor:
    """Decompress latent from per-group variable bit-width packed data."""
    T = all_packed[0].shape[0]
    norms_f = norms.float()
    latent = torch.zeros(T, kv_lora_rank, dtype=torch.float32, device=device)

    for g, bw in enumerate(group_bits):
        g_start = g * group_size
        g_end = min(g_start + group_size, kv_lora_rank)
        g_dim = g_end - g_start

        centroids, _ = get_codebook(bw)
        centroids = centroids.to(device)

        padded = pad_for_packing(g_dim, bw)
        indices = unpack_indices(all_packed[g], padded, bw)[:, :g_dim]

        Pi = rotations[g_start]
        Y_g = centroids[indices.long()] / math.sqrt(g_dim)
        L_g = Y_g @ Pi
        L_g = L_g * norms_f[:, g].unsqueeze(1)
        latent[:, g_start:g_end] = L_g

    return latent


# ---------------------------------------------------------------------------
# Single-pass quantization
# ---------------------------------------------------------------------------


@torch.no_grad()
def turboquant_quantize_packed(
    W: torch.Tensor,
    bit_width: int = 4,
    group_size: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """Quantize weight matrix and return packed representation.

    Args:
        W: (out_features, in_features) weight matrix
        bit_width: bits per element (4-bit packing only)
        group_size: group size along in_features (None = full row)
        seed: rotation seed

    Returns dict with indices_packed, codebook, norms, seed, group_size, shape, bit_width.
    """
    assert bit_width in (2, 3, 4), f"Supported bit widths: 2, 3, 4. Got {bit_width}"
    M, N = W.shape
    if group_size is None:
        group_size = N

    W = W.float()
    centroids, boundaries = get_codebook(bit_width)
    centroids = centroids.to(W.device)
    boundaries = boundaries.to(W.device)

    all_norms = []
    all_indices = []

    for g_start in range(0, N, group_size):
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start
        W_g = W[:, g_start:g_end]

        norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W_g / norms
        all_norms.append(norms.squeeze(1))

        Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(W.device)
        Y = W_norm @ Pi.T
        scale = math.sqrt(g_dim)
        Y_scaled = Y * scale

        indices = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
        indices = indices.clamp(0, len(centroids) - 1).reshape(M, g_dim)
        all_indices.append(indices)

    full_indices = torch.cat(all_indices, dim=1)
    norms_out = (
        torch.stack(all_norms, dim=1) if len(all_norms) > 1 else all_norms[0]
    )

    padded_N = pad_for_packing(N, bit_width)
    if padded_N > N:
        full_indices = torch.nn.functional.pad(
            full_indices, (0, padded_N - N), value=0
        )

    packed = pack_indices(full_indices, bit_width)

    return {
        "indices_packed": packed,
        "codebook": centroids.cpu(),
        "norms": norms_out.cpu(),
        "seed": seed,
        "group_size": group_size,
        "shape": (M, N),
        "bit_width": bit_width,
    }


@torch.no_grad()
def turboquant_dequantize(packed_data: dict, device: torch.device) -> torch.Tensor:
    """Reconstruct full weight from packed representation."""
    M, N = packed_data["shape"]
    group_size = packed_data["group_size"]
    seed = packed_data["seed"]

    bit_width = packed_data.get("bit_width", 4)
    indices_packed = packed_data["indices_packed"].to(device)
    codebook = packed_data["codebook"].to(device)
    norms = packed_data["norms"].to(device)

    padded_N = pad_for_packing(N, bit_width)
    indices = unpack_indices(indices_packed, padded_N, bit_width)[:, :N]

    n_groups = math.ceil(N / group_size)
    W_approx = torch.zeros(M, N, dtype=torch.float32, device=device)

    for g in range(n_groups):
        g_start = g * group_size
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start

        Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(device)
        scale = math.sqrt(g_dim)

        Y_g = codebook[indices[:, g_start:g_end].long()] / scale
        W_g = Y_g @ Pi

        if norms.dim() == 1:
            W_g = W_g * norms.unsqueeze(1)
        else:
            W_g = W_g * norms[:, g].unsqueeze(1)

        W_approx[:, g_start:g_end] = W_g

    return W_approx


# ---------------------------------------------------------------------------
# Residual (two-pass) quantization
# ---------------------------------------------------------------------------


@torch.no_grad()
def residual_quantize_packed(
    W: torch.Tensor,
    bit_width_1: int = 4,
    bit_width_2: int = 4,
    group_size: Optional[int] = None,
    seed_1: int = 42,
    seed_2: int = 1042,
) -> dict:
    """Two-pass residual TurboQuant: returns packed reps for both passes."""
    pass1 = turboquant_quantize_packed(
        W, bit_width=bit_width_1, group_size=group_size, seed=seed_1
    )

    W_hat1 = turboquant_dequantize(pass1, device=W.device)
    residual = W.float() - W_hat1

    pass2 = turboquant_quantize_packed(
        residual, bit_width=bit_width_2, group_size=group_size, seed=seed_2
    )

    return {
        "pass1": pass1,
        "pass2": pass2,
        "total_bits": bit_width_1 + bit_width_2,
    }


# ---------------------------------------------------------------------------
# On-the-fly forward pass (PyTorch fallback, no Triton)
# ---------------------------------------------------------------------------


@torch.no_grad()
def turboquant_matmul_pytorch(
    x: torch.Tensor,
    indices_packed: torch.Tensor,
    codebook: torch.Tensor,
    weight_norms: torch.Tensor,
    in_features: int,
    group_size: int,
    seed: int,
    bit_width: int = 4,
) -> torch.Tensor:
    """On-the-fly dequant matmul: y = x @ W^T using packed representation.

    Approach C: rotate input instead of dequantizing full weight.
    Supports 2/3/4-bit packed indices.
    """
    B = x.shape[0]
    N = indices_packed.shape[0]
    K = in_features
    device = x.device
    n_groups = math.ceil(K / group_size)
    scale = math.sqrt(group_size)

    padded_K = pad_for_packing(K, bit_width)
    indices = unpack_indices(indices_packed, padded_K, bit_width)[:, :K]

    output = torch.zeros(B, N, dtype=torch.float32, device=device)

    for g in range(n_groups):
        g_start = g * group_size
        g_end = min(g_start + group_size, K)
        g_dim = g_end - g_start

        Pi_g = generate_rotation_matrix(g_dim, seed=seed + g_start).to(device)
        x_rot_g = x[:, g_start:g_end].float() @ Pi_g.T

        idx_g = indices[:, g_start:g_end]
        W_g = codebook[idx_g.long()]

        out_g = x_rot_g @ W_g.T

        if weight_norms.dim() == 1:
            norms_g = weight_norms
        else:
            norms_g = weight_norms[:, g]

        out_g = out_g * (norms_g[None, :] / scale)
        output += out_g

    return output
