# SPDX-License-Identifier: Apache-2.0
"""
3D Rotary Position Embedding (RoPE) for video transformers.

Reference: https://arxiv.org/pdf/2104.09864.pdf
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat


def broadcast(tensors: list[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """
    Broadcast and concatenate tensors along a dimension.
    """
    num_tensors = len(tensors)
    shape_lens = set(len(t.shape) for t in tensors)
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"

    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim

    dims = list(zip(*[list(t.shape) for t in tensors], strict=False))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]

    assert all(
        len(set(t[1])) <= 2 for t in expandable_dims
    ), "invalid dimensions for broadcastable concatenation"

    max_dims = [(t[0], max(t[1])) for t in expandable_dims]
    expanded_dims = [(t[0], (t[1],) * num_tensors) for t in max_dims]
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*[t[1] for t in expanded_dims], strict=False))
    tensors = [
        t[0].expand(*t[1]) for t in zip(tensors, expandable_shapes, strict=False)
    ]

    return torch.cat(tensors, dim=dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.
    """
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class RotaryPositionalEmbedding3D(nn.Module):
    """
    3D Rotary Positional Embedding for video transformers.

    Splits the head dimension across temporal, height, and width dimensions,
    computing separate rotary embeddings for each and concatenating them.
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
    ):
        """
        Args:
            head_dim: Dimension of each attention head
            base: Base value for exponential frequency
        """
        super().__init__()
        self.head_dim = head_dim
        assert self.head_dim % 8 == 0, "head_dim must be a multiple of 8 for 3D RoPE"
        self.base = base

        # Cache for precomputed frequencies
        self.freqs_dict: dict[tuple, torch.Tensor] = {}

    def register_grid_size(self, grid_size: tuple[int, int, int]) -> None:
        """
        Precompute and register frequencies for a given grid size.

        Args:
            grid_size: (T, H, W) tuple of grid dimensions
        """
        if grid_size not in self.freqs_dict:
            self.freqs_dict[grid_size] = self.precompute_freqs_3d(grid_size)

    def precompute_freqs_3d(self, grid_size: tuple[int, int, int]) -> torch.Tensor:
        """
        Precompute 3D rotary frequencies.

        Args:
            grid_size: (num_frames, height, width)

        Returns:
            freqs: [T*H*W, head_dim] tensor of frequencies
        """
        num_frames, height, width = grid_size

        # Split head_dim across 3 dimensions
        # Temporal gets the remainder to ensure exact division
        dim_t = self.head_dim - 4 * (self.head_dim // 6)
        dim_h = 2 * (self.head_dim // 6)
        dim_w = 2 * (self.head_dim // 6)

        # Compute frequency bands for each dimension
        freqs_t = 1.0 / (
            self.base ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t)
        )
        freqs_h = 1.0 / (
            self.base ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h)
        )
        freqs_w = 1.0 / (
            self.base ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w)
        )

        # Create position grids
        grid_t = torch.arange(num_frames, dtype=torch.float32)
        grid_h = torch.arange(height, dtype=torch.float32)
        grid_w = torch.arange(width, dtype=torch.float32)

        # Compute frequencies for each position
        freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
        freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)

        # Duplicate for complex pair representation
        freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

        # Broadcast and concatenate across all 3 dimensions
        freqs = broadcast(
            [
                freqs_t[:, None, None, :],  # [T, 1, 1, dim_t]
                freqs_h[None, :, None, :],  # [1, H, 1, dim_h]
                freqs_w[None, None, :, :],  # [1, 1, W, dim_w]
            ],
            dim=-1,
        )

        # Flatten spatial dimensions: [T, H, W, head_dim] -> [T*H*W, head_dim]
        freqs = rearrange(freqs, "T H W D -> (T H W) D")

        return freqs

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        grid_size: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 3D rotary positional embedding to queries and keys.

        Args:
            q: Query tensor [B, num_heads, seq_len, head_dim]
            k: Key tensor [B, num_heads, seq_len, head_dim]
            grid_size: (T, H, W) tuple of grid dimensions

        Returns:
            (q_rotated, k_rotated): Rotated query and key tensors
        """
        # Register grid size if not cached
        if grid_size not in self.freqs_dict:
            self.register_grid_size(grid_size)

        # Get cached frequencies
        freqs_cis = self.freqs_dict[grid_size].to(q.device)

        # Cast to float32 for precision
        q_, k_ = q.float(), k.float()
        freqs_cis = freqs_cis.float()

        # Compute cos and sin
        cos = freqs_cis.cos()
        sin = freqs_cis.sin()

        # Reshape for broadcasting: [1, 1, seq_len, head_dim]
        cos = rearrange(cos, "n d -> 1 1 n d")
        sin = rearrange(sin, "n d -> 1 1 n d")

        # Apply rotation
        q_ = (q_ * cos) + (rotate_half(q_) * sin)
        k_ = (k_ * cos) + (rotate_half(k_) * sin)

        # Cast back to original dtype
        return q_.type_as(q), k_.type_as(k)


def apply_rotary_emb_3d(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_module: RotaryPositionalEmbedding3D,
    grid_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to apply 3D RoPE.

    Args:
        q: Query tensor [B, num_heads, seq_len, head_dim]
        k: Key tensor [B, num_heads, seq_len, head_dim]
        rope_module: RotaryPositionalEmbedding3D module
        grid_size: (T, H, W) grid dimensions

    Returns:
        (q_rotated, k_rotated): Rotated tensors
    """
    return rope_module(q, k, grid_size)
