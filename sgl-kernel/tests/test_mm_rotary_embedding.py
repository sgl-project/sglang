#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from dataclasses import dataclass
from typing import Tuple, List

import torch

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

try:
    # Cos/sin-based rotary kernel from sgl_diffusion
    from sgl_kernel.rotary_embedding import rotary_embedding_cos_sin as rotary_emb_module
    HAS_ROTARY_EMBEDDING = True
except ImportError:
    rotary_emb_module = None
    HAS_ROTARY_EMBEDDING = False
    print("sgl_kernel.rotary_embedding_cos_sin not available")


@dataclass
class RotaryTestResult:
    name: str
    passed: bool
    q_diff: float = 0.0
    k_diff: float = 0.0
    details: str = ""


def compute_cos_sin_cache(
    max_seq_len: int,
    rotary_dim: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute separate cos and sin caches.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def reference_rotary_neox(
    query: torch.Tensor,  # (num_tokens, num_heads * head_size)
    key: torch.Tensor,  # (num_tokens, num_kv_heads * head_size)
    cos: torch.Tensor,  # (num_tokens, rotary_dim / 2)
    sin: torch.Tensor,  # (num_tokens, rotary_dim / 2)
    head_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation of NeoX-style rotary embedding, 
    using explicit cos/sin caches and Q/K in flattened (num_tokens, heads * dim) layout.
    """
    num_tokens = query.size(0)
    num_heads = query.size(1) // head_size
    num_kv_heads = key.size(1) // head_size
    rotary_dim = cos.size(1) * 2

    q = query.view(num_tokens, num_heads, head_size).float()
    k = key.view(num_tokens, num_kv_heads, head_size).float()

    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    q_rot = q_rot.view(num_tokens, num_heads, rotary_dim // 2, 2)
    k_rot = k_rot.view(num_tokens, num_kv_heads, rotary_dim // 2, 2)

    cos_expanded = cos.float().unsqueeze(1)  # (tokens, 1, rotary_dim/2)
    sin_expanded = sin.float().unsqueeze(1)

    # Apply rotary: x' = x*cos - y*sin, y' = y*cos + x*sin
    q_x = q_rot[..., 0]
    q_y = q_rot[..., 1]
    q_rot_out = torch.stack(
        [
            q_x * cos_expanded - q_y * sin_expanded,
            q_y * cos_expanded + q_x * sin_expanded,
        ],
        dim=-1,
    )

    k_x = k_rot[..., 0]
    k_y = k_rot[..., 1]
    k_rot_out = torch.stack(
        [
            k_x * cos_expanded - k_y * sin_expanded,
            k_y * cos_expanded + k_x * sin_expanded,
        ],
        dim=-1,
    )

    q_rot_out = q_rot_out.view(num_tokens, num_heads, rotary_dim)
    k_rot_out = k_rot_out.view(num_tokens, num_kv_heads, rotary_dim)

    q_out = torch.cat([q_rot_out, q_pass], dim=-1)
    k_out = torch.cat([k_rot_out, k_pass], dim=-1)

    q_out = q_out.view(num_tokens, num_heads * head_size).to(query.dtype)
    k_out = k_out.view(num_tokens, num_kv_heads * head_size).to(key.dtype)

    return q_out, k_out


def _check_rotary_correctness(
    batch_size: int = 2,
    seq_len: int = 128,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    tol: float = 1e-2,
    device: str = "cuda",
) -> RotaryTestResult:
    if not HAS_ROTARY_EMBEDDING:
        return RotaryTestResult("basic (skipped: rotary_embedding not built)", True, details="rotary_embedding extension not found")

    rotary_dim = head_size
    max_seq_len = 8192
    num_tokens = batch_size * seq_len

    query = torch.randn(num_tokens, num_heads * head_size, dtype=dtype, device=device)
    key = torch.randn(num_tokens, num_kv_heads * head_size, dtype=dtype, device=device)
    cos_cache, sin_cache = compute_cos_sin_cache(max_seq_len, rotary_dim, dtype=dtype)
    cos_cache = cos_cache.to(device)
    sin_cache = sin_cache.to(device)
    positions = torch.arange(seq_len, device=device).repeat(batch_size)
    cos = cos_cache[positions]
    sin = sin_cache[positions]

    q_ref, k_ref = reference_rotary_neox(query.clone(), key.clone(), cos, sin, head_size)

    q_out = query.clone()
    k_out = key.clone()
    rotary_emb_module(cos, sin, q_out, k_out, head_size, True)  # is_neox=True

    q_diff = (q_out - q_ref).abs().max().item()
    k_diff = (k_out - k_ref).abs().max().item()
    passed = q_diff < tol and k_diff < tol

    name = f"basic [bs={batch_size}, seq={seq_len}, heads={num_heads}/{num_kv_heads}, dim={head_size}, {dtype}]"
    return RotaryTestResult(name, passed, q_diff, k_diff)


@pytest.mark.parametrize("batch_size", [2, 32, 1])
@pytest.mark.parametrize("seq_len", [1, 128, 512, 2048])
@pytest.mark.parametrize("num_heads, num_kv_heads", [(32, 8), (64, 8), (8, 8), (32, 1)])
@pytest.mark.parametrize("head_size", [64, 128, 256, 80, 320])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rotary_embedding_correctness(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
) -> None:
    if not HAS_ROTARY_EMBEDDING:
        pytest.skip("sgl_kernel.rotary_embedding not available")
    
    result = _check_rotary_correctness(
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
    )
    assert result.passed, f"{result.name} failed: Q={result.q_diff:.2e}, K={result.k_diff:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
