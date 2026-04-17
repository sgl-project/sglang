"""TurboQuant KV cache quantization utilities.

Implements PolarQuant (Algorithm 1 from arXiv 2504.19874):
  Walsh-Hadamard rotation → optimal scalar quantization → bit-packing.

Uses deterministic WHT (Walsh-Hadamard Transform) instead of random rotation
matrix. WHT is O(d log d), deterministic, and maximally decorrelates dimensions
— 59.7x better PPL than random rotation at 4-bit (llama.cpp findings).

Supports two dequant modes:
  1. batched_dequantize: Full inverse WHT, returns data in original domain.
  2. batched_dequantize_rotspace: No inverse WHT, returns data in WHT-rotated
     domain. Attention backends rotate Q instead (Query Rotation), which is
     mathematically equivalent but avoids O(d log d) per-token dequant cost.
"""

import math

import numpy as np
import torch




# ---------------------------------------------------------------------------
# Codebook construction (one-time, at init)
# ---------------------------------------------------------------------------


def _lloyds_gaussian(n_centroids: int, sigma: float, n_iter: int = 100):
    """Lloyd's algorithm for optimal scalar quantization of N(0, sigma^2)."""
    from scipy import stats

    centroids = np.zeros(n_centroids)

    def _cond_exp(s, a, b):
        a_s = a / s if np.isfinite(a) else a
        b_s = b / s if np.isfinite(b) else b
        if not np.isfinite(a_s):
            prob = stats.norm.cdf(b_s)
        elif not np.isfinite(b_s):
            prob = stats.norm.sf(a_s)
        else:
            prob = stats.norm.cdf(b_s) - stats.norm.cdf(a_s)
        if prob < 1e-15:
            if np.isfinite(a) and not np.isfinite(b):
                return a + s
            elif not np.isfinite(a) and np.isfinite(b):
                return b - s
            return (a + b) / 2.0 if np.isfinite(a) else 0.0
        return s * (stats.norm.pdf(a_s) - stats.norm.pdf(b_s)) / prob

    for _ in range(n_iter):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        centroids[0] = _cond_exp(sigma, -np.inf, boundaries[0])
        for i in range(1, n_centroids - 1):
            centroids[i] = _cond_exp(sigma, boundaries[i - 1], boundaries[i])
        centroids[-1] = _cond_exp(sigma, boundaries[-1], np.inf)

    return np.sort(centroids)


def build_codebook(bit_width: int, head_dim: int, uniform: bool = False):
    """Return (centroids, boundaries) as numpy arrays for the given config.

    After WHT rotation, each coordinate ~ N(0, 1/d).

    Args:
        uniform: If True, use uniformly-spaced centroids (1 FMA dequant)
                 instead of Lloyd-Max optimal codebook (15 select dequant).
                 Trades ~0.8 dB SNR for 15x fewer ALU ops in decode kernel.
    """
    d = head_dim
    n = 1 << bit_width
    if uniform and bit_width >= 3:
        sigma = 1.0 / math.sqrt(d)
        r = 2.5 * sigma
        centroids = np.linspace(-r, r, n)
    elif bit_width == 1:
        c = math.sqrt(2.0 / (math.pi * d))
        centroids = np.array([-c, c])
    elif bit_width == 2:
        centroids = np.array([-1.51, -0.453, 0.453, 1.51]) / math.sqrt(d)
    else:
        centroids = _lloyds_gaussian(n, sigma=1.0 / math.sqrt(d))
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids.astype(np.float32), boundaries.astype(np.float32)


# ---------------------------------------------------------------------------
# GPU quantize / dequantize
# ---------------------------------------------------------------------------

# Packing helpers:
#   2-bit: 4 values per byte (uint8), fused decode kernel uses 4-way split dot product
#   4-bit: 2 values per byte (uint8), fused decode kernel uses 2-way split dot product


def batched_quantize(
    x: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
    bit_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize KV tensors to packed uint8 + norms + quant_norms.

    Args:
        x: (tokens, heads, dim) in bfloat16/float16.
        signs1: (dim,) float32 — random sign vector for WHT rotation.
        signs2: (dim,) float32 — random sign vector for WHT rotation.
        centroids: (2^b,) float32.
        boundaries: (2^b - 1,) float32.
        bit_width: 2 or 4.

    Returns:
        packed: (tokens, heads, packed_dim) uint8.
        norms: (tokens, heads) bfloat16 — original L2 norms.
        quant_norms: (tokens, heads) bfloat16 — L2 norm of centroid vector
            ||centroids[indices]||, used for norm correction in rotspace dequant.
    """

    from sglang.jit_kernel.hadamard import hadamard_transform

    tokens, heads, dim = x.shape

    # 1. Extract norms
    norms = torch.linalg.norm(x.float(), dim=-1)  # (tokens, heads)

    # 2. Normalize to unit vectors
    safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
    x_unit = x.float() / safe_norms.unsqueeze(-1)

    # 3. WHT rotation: D2 @ H_norm @ D1 (normalized Hadamard, scale=1/√d)
    wht_scale = 1.0 / math.sqrt(dim)
    y = x_unit * signs1
    y = hadamard_transform(y.contiguous(), scale=wht_scale)
    y = y * signs2

    # 4. Quantize: nearest centroid via searchsorted on boundaries
    indices = torch.searchsorted(boundaries, y.reshape(-1)).reshape(tokens, heads, dim)

    # 5. Compute quant_norms: ||centroids[indices]|| for norm correction
    centroid_vals = centroids[indices.long()]  # (tokens, heads, dim) float32
    quant_norms = torch.linalg.norm(centroid_vals, dim=-1)  # (tokens, heads)

    # 6. Pack indices
    indices_u8 = indices.to(torch.uint8)
    if bit_width == 2:
        idx = indices_u8.view(tokens, heads, dim // 4, 4)
        packed = (
            (idx[..., 3] << 6)
            | (idx[..., 2] << 4)
            | (idx[..., 1] << 2)
            | idx[..., 0]
        )
    elif bit_width == 4:
        idx = indices_u8.view(tokens, heads, dim // 2, 2)
        packed = (idx[..., 1] << 4) | idx[..., 0]
    else:
        raise ValueError(f"Unsupported bit_width for packing: {bit_width}")

    return packed, norms.to(x.dtype), quant_norms.to(x.dtype)


def batched_dequantize(
    packed: torch.Tensor,
    norms: torch.Tensor,
    centroids: torch.Tensor,
    bit_width: int,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
) -> torch.Tensor:
    """Dequantize packed uint8 back to bfloat16 tensors.

    Includes inverse WHT rotation — returns data in the original domain.
    No graph-side rotation needed in the attention backend.

    Args:
        packed: (tokens, heads, packed_dim) uint8.
        norms: (tokens, heads) bfloat16.
        centroids: (2^b,) float32.
        bit_width: 2 or 4.
        signs1: (dim,) float32.
        signs2: (dim,) float32.

    Returns:
        x_hat: (tokens, heads, dim) bfloat16 — in original domain.
    """
    from sglang.jit_kernel.hadamard import hadamard_transform

    tokens, heads, packed_dim = packed.shape

    # 1. Unpack indices
    if bit_width == 2:
        dim = packed_dim * 4
        v0 = packed & 0x03
        v1 = (packed >> 2) & 0x03
        v2 = (packed >> 4) & 0x03
        v3 = (packed >> 6) & 0x03
        indices = torch.stack([v0, v1, v2, v3], dim=-1).reshape(tokens, heads, dim)
    elif bit_width == 4:
        dim = packed_dim * 2
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        indices = torch.stack([low, high], dim=-1).reshape(tokens, heads, dim)
    else:
        raise ValueError(f"Unsupported bit_width for unpacking: {bit_width}")

    # 2. Centroid lookup
    y_hat = centroids[indices.long()]  # (tokens, heads, dim) float32

    # 3. Norm correction: normalize y_hat back to unit norm
    y_hat_norm = torch.linalg.norm(y_hat, dim=-1, keepdim=True)
    y_hat_norm = torch.where(y_hat_norm > 1e-10, y_hat_norm, torch.ones_like(y_hat_norm))
    y_hat = y_hat / y_hat_norm

    # 4. Inverse WHT rotation: D1 @ H_norm @ D2 (reverse order, scale=1/√d)
    wht_scale = 1.0 / math.sqrt(signs1.shape[0])
    y_hat = y_hat * signs2
    y_hat = hadamard_transform(y_hat.contiguous(), scale=wht_scale)
    y_hat = y_hat * signs1

    # 5. Scale by original norm
    y_hat = y_hat * norms.float().unsqueeze(-1)

    return y_hat.to(norms.dtype)


def batched_dequantize_rotspace(
    packed: torch.Tensor,
    dequant_scale: torch.Tensor,
    centroids: torch.Tensor,
    bit_width: int,
    head_dim: int = 0,
) -> torch.Tensor:
    """Dequantize packed data to WHT-rotated domain (no inverse WHT).

    Args:
        packed: (tokens, heads, packed_dim) uint8.
        dequant_scale: (tokens, heads) bfloat16 — precomputed norm / max(quant_norm, eps).
        centroids: (2^b,) float32.
        bit_width: 2 or 4.
        head_dim: original head dimension (for 2-bit padding trim). 0 = auto-infer.

    Returns:
        x_rotspace: (tokens, heads, head_dim) bfloat16 — in WHT-rotated domain.
    """
    tokens, heads, packed_dim = packed.shape

    # 1. Unpack indices
    if bit_width == 2:
        dim = packed_dim * 4
        v0 = packed & 0x03
        v1 = (packed >> 2) & 0x03
        v2 = (packed >> 4) & 0x03
        v3 = (packed >> 6) & 0x03
        indices = torch.stack([v0, v1, v2, v3], dim=-1).reshape(tokens, heads, dim)
    elif bit_width == 4:
        dim = packed_dim * 2
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        indices = torch.stack([low, high], dim=-1).reshape(tokens, heads, dim)
    else:
        raise ValueError(f"Unsupported bit_width for unpacking: {bit_width}")

    # 2. Centroid lookup
    y_hat = centroids[indices.long()]  # (tokens, heads, dim) float32

    # 3. Norm-corrected scaling using precomputed dequant_scale
    y_hat = y_hat * dequant_scale.float().unsqueeze(-1)

    return y_hat.to(dequant_scale.dtype)


# ---------------------------------------------------------------------------
# Init helper — called once at pool creation
# ---------------------------------------------------------------------------


class TurboQuantConfig:
    """Holds pre-computed constants for TurboQuant on GPU.

    Supports asymmetric K/V bit widths (e.g., K=4bit, V=2bit).
    """

    def __init__(
        self,
        bit_width: int,
        head_dim: int,
        device: str,
        seed: int = 42,
        k_bit_width: int = 0,
        v_bit_width: int = 0,
        uniform: bool = False,
    ):
        self.head_dim = head_dim
        self.k_bit_width = k_bit_width or bit_width
        self.v_bit_width = v_bit_width or bit_width
        self.bit_width = bit_width  # backwards compat (= k_bit_width)
        self.uniform = uniform

        # K codebook
        k_c_np, k_b_np = build_codebook(self.k_bit_width, head_dim, uniform=uniform)
        self.k_centroids = torch.tensor(k_c_np, dtype=torch.float32, device=device)
        self.k_boundaries = torch.tensor(k_b_np, dtype=torch.float32, device=device)

        # V codebook (may differ from K)
        if self.v_bit_width == self.k_bit_width:
            self.v_centroids = self.k_centroids
            self.v_boundaries = self.k_boundaries
        else:
            v_c_np, v_b_np = build_codebook(self.v_bit_width, head_dim, uniform=uniform)
            self.v_centroids = torch.tensor(v_c_np, dtype=torch.float32, device=device)
            self.v_boundaries = torch.tensor(v_b_np, dtype=torch.float32, device=device)

        # WHT sign vectors (shared for K and V)
        rng = np.random.default_rng(seed)
        self.signs1 = torch.tensor(
            rng.choice([-1.0, 1.0], size=head_dim),
            dtype=torch.float32,
            device=device,
        )
        self.signs2 = torch.tensor(
            rng.choice([-1.0, 1.0], size=head_dim),
            dtype=torch.float32,
            device=device,
        )

        # K packed dim/dtype
        self.k_packed_dim, self.k_packed_dtype = self._packed_params(self.k_bit_width, head_dim)
        # V packed dim/dtype
        self.v_packed_dim, self.v_packed_dtype = self._packed_params(self.v_bit_width, head_dim)

    @staticmethod
    def _packed_params(bits, head_dim):
        if bits == 2:
            assert head_dim % 4 == 0, f"2-bit requires head_dim divisible by 4, got {head_dim}"
            return head_dim // 4, torch.uint8
        elif bits == 4:
            assert head_dim % 2 == 0, f"4-bit requires head_dim divisible by 2, got {head_dim}"
            return head_dim // 2, torch.uint8
        else:
            raise ValueError(f"Unsupported bit_width: {bits}. Use 2 or 4.")

    def rotate_query(self, q: torch.Tensor) -> torch.Tensor:
        """Apply forward WHT rotation to query: Q_rot = D2 @ H @ D1 @ Q.

        Args:
            q: (..., dim) tensor, any shape with last dim = head_dim.

        Returns:
            q_rot: same shape and dtype as q, in WHT-rotated domain.
        """
        from sglang.jit_kernel.hadamard import hadamard_transform_with_signs
        wht_scale = 1.0 / math.sqrt(self.head_dim)
        return hadamard_transform_with_signs(q, self.signs1, self.signs2, scale=wht_scale)

    def inverse_rotate_output(self, o: torch.Tensor) -> torch.Tensor:
        """Apply inverse WHT rotation to attention output: O = D1 @ H_norm @ D2 @ O_rot.

        Args:
            o: (..., dim) tensor, in WHT-rotated domain.

        Returns:
            o_orig: same shape and dtype as o, in original domain.
        """
        from sglang.jit_kernel.hadamard import hadamard_transform_with_signs
        wht_scale = 1.0 / math.sqrt(self.head_dim)
        return hadamard_transform_with_signs(o, self.signs2, self.signs1, scale=wht_scale)

    def fuse_inverse_rotation_into_o_proj(self, o_proj_weight: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Absorb inverse WHT rotation into o_proj weight matrix.

        Pre-computes W_O_rot so that: o_rot @ W_O_rot = inv_WHT(o_rot) @ W_O
        This eliminates inverse_rotate_output at runtime.

        Args:
            o_proj_weight: (num_heads * head_dim, hidden_dim) weight matrix.
            num_heads: number of Q heads.

        Returns:
            Transformed weight with inverse rotation baked in.
        """
        from sglang.jit_kernel.hadamard import hadamard_transform_with_signs
        device = o_proj_weight.device
        dtype = o_proj_weight.dtype
        dim = self.head_dim
        hidden = o_proj_weight.shape[1]

        # inv_WHT per head: W_O[h*d:(h+1)*d, :] = (D1 @ H @ D2) @ W_O[h*d:(h+1)*d, :]
        # Equivalently: transpose each head block, apply forward WHT with swapped signs, transpose back
        w = o_proj_weight.float().view(num_heads, dim, hidden)  # (heads, dim, hidden)

        # Apply inverse rotation to each row of each head block
        # inv_WHT operates on dim dimension: for each column of W, apply D1 @ H @ D2
        # Reshape to (num_heads * hidden, dim), apply WHT, reshape back
        w_t = w.permute(0, 2, 1).contiguous().reshape(-1, dim)  # (heads*hidden, dim)
        wht_scale = 1.0 / math.sqrt(dim)
        w_rot = hadamard_transform_with_signs(w_t, self.signs1, self.signs2, scale=wht_scale)
        w_rot = w_rot.reshape(num_heads, hidden, dim).permute(0, 2, 1).contiguous()
        return w_rot.reshape(num_heads * dim, hidden).to(dtype)
