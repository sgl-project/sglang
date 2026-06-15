# SPDX-License-Identifier: Apache-2.0
"""
GPU DCT-II / IDCT-II via torch.fft — no CPU↔GPU transfers.

Algorithm: Makhoul (1980) "A fast cosine transform in one and two dimensions",
adapted for PyTorch. Operates on the last two spatial dims; input can be any
shape (..., H, W).
"""

import math

import torch

# ---------------------------------------------------------------------------
# 1-D DCT-II / IDCT-II (ortho-normalized, operates on last dim)
# ---------------------------------------------------------------------------


def dct_1d(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """1-D DCT-II via torch.fft. Input: (..., N). Output: same shape."""
    shape = x.shape
    N = shape[-1]
    x = x.reshape(-1, N)

    # Reorder: [x0, x2, x4, ..., xN-1, ..., x3, x1]
    v = torch.cat([x[:, ::2], x[:, 1::2].flip(dims=[1])], dim=1)

    Vc = torch.fft.fft(v, dim=1)

    k = torch.arange(N, dtype=x.dtype, device=x.device) * (-math.pi / (2 * N))
    W = torch.exp(torch.complex(torch.zeros_like(k), k))  # e^{-i*pi*k/(2N)}
    V = (Vc * W).real

    if norm == "ortho":
        V[:, 0] /= math.sqrt(N) * 2
        V[:, 1:] /= math.sqrt(N / 2) * 2

    return (2 * V).reshape(shape)


def idct_1d(X: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """1-D IDCT-II (= scaled DCT-III) via torch.fft. Input: (..., N)."""
    shape = X.shape
    N = shape[-1]
    X_v = X.reshape(-1, N) / 2

    if norm == "ortho":
        X_v = X_v.clone()
        X_v[:, 0] *= math.sqrt(N) * 2
        X_v[:, 1:] *= math.sqrt(N / 2) * 2

    k = torch.arange(N, dtype=X.dtype, device=X.device) * (math.pi / (2 * N))
    W = torch.exp(torch.complex(torch.zeros_like(k), k))  # e^{i*pi*k/(2N)}

    # Build complex input for IFFT
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip(dims=[1])[:, :-1]], dim=1)
    Vc = torch.complex(X_v, V_t_i) * W

    v = torch.fft.ifft(Vc, dim=1).real
    x = torch.zeros_like(v)
    x[:, ::2] = v[:, : N - (N // 2)]
    x[:, 1::2] = v.flip(dims=[1])[:, : N // 2]
    return x.reshape(shape)


# ---------------------------------------------------------------------------
# 2-D DCT-II / IDCT-II (separable: apply 1-D along H then W)
# ---------------------------------------------------------------------------


def dct_2d(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """2-D DCT-II on the last two dims of x (..., H, W)."""
    return dct_1d(dct_1d(x, norm).transpose(-1, -2), norm).transpose(-1, -2)


def idct_2d(X: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """2-D IDCT-II on the last two dims of X (..., H, W)."""
    return idct_1d(idct_1d(X, norm).transpose(-1, -2), norm).transpose(-1, -2)
