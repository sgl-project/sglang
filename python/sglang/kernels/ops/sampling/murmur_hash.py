"""MurmurHash3 x86_32 for deterministic sampling coins.

Two backends behind one ``murmur_hash32(seed, positions, col_indices)`` entry:

- ``_murmur_hash32_jit``: CUDA JIT kernel (``python/sglang/kernels/jit/csrc/sampling/murmur_hash.cuh``).
- ``_murmur_hash32_triton``: Triton reference, kept as the ROCm fallback.

Both are bit-identical: the JIT kernel reuses the same four-block blend and
finalization. The Triton kernel stays importable so callers and tests can
cross-check the two implementations against each other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_is_hip = torch.version.hip is not None

#: Dtypes accepted for ``positions`` / ``col_indices``; all truncate to uint32
#: in-kernel exactly like the Triton reference's ``.to(tl.uint32)``.
_SUPPORTED_INDEX_DTYPES = (torch.int32, torch.int64, torch.uint32, torch.uint64)


@triton.jit
def rotl32(x, r: tl.constexpr) -> tl.uint32:
    """
    rotate left 32-bit integer x by r bits
    e.g. x = 01110001, r = 2 -> 11000101
    """
    x = x.to(tl.uint64)
    return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF


@triton.jit
def fmix32(h: tl.uint32) -> tl.uint32:
    """
    final mix of 32-bit hash value for MurmurHash
    """
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


@triton.jit
def murmur3_mix(h: tl.uint32, k: tl.uint32) -> tl.uint32:
    """
    Mixes a 32-bit key into the hash state.
    """
    c1: tl.uint32 = 0xCC9E2D51
    c2: tl.uint32 = 0x1B873593
    r1: tl.constexpr = 15
    r2: tl.constexpr = 13
    mm: tl.uint32 = 5
    nn: tl.uint32 = 0xE6546B64

    k = (k * c1) & 0xFFFFFFFF
    k = rotl32(k, r1)
    k = (k * c2) & 0xFFFFFFFF
    h ^= k
    h = rotl32(h, r2)
    h = (h * mm + nn) & 0xFFFFFFFF
    return h


@triton.jit
def murmur_hash32_kernel(
    seed_ptr,
    positions_ptr,
    col_indices_ptr,
    output_ptr,
    num_rows,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    MurmurHash 32-bit implementation for Triton.
    Reference:
    - https://medium.com/@thealonemusk/murmurhash-the-scrappy-algorithm-that-secretly-powers-half-the-internet-2d3f79b4509b
    - https://en.wikipedia.org/wiki/MurmurHash

    We treat 64-bit seed, 32-bit position, and 32-bit col_index as 4 4-byte blocks, and bit-blend them together.
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_idx = pid_row
    col_offsets = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_cols

    # Load inputs
    seed = tl.load(seed_ptr + row_idx).to(tl.uint64)
    pos = tl.load(positions_ptr + row_idx).to(tl.uint32)
    col = tl.load(col_indices_ptr + col_offsets, mask=mask, other=0).to(tl.uint32)

    h: tl.uint32 = 0  # hash accumulator

    # Process seed_low
    k: tl.uint32 = (seed & 0xFFFFFFFF).to(tl.uint32)
    h = murmur3_mix(h, k)

    # Process seed_high
    k = ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32)
    h = murmur3_mix(h, k)

    # Process position block starting from seed32
    h = murmur3_mix(h, pos)

    # Process col block
    h = murmur3_mix(h, col)

    # Finalize (len=16 for seed + pos + col)
    h ^= 16
    h = fmix32(h)

    # Store result as uint32
    tl.store(output_ptr + row_idx * num_cols + col_offsets, h, mask=mask)


@cache_once
def _jit_murmur_hash_module(pos_dtype: torch.dtype, col_dtype: torch.dtype) -> Module:
    """Compile and cache the JIT MurmurHash32 module for the given dtypes."""
    if (
        pos_dtype not in _SUPPORTED_INDEX_DTYPES
        or col_dtype not in _SUPPORTED_INDEX_DTYPES
    ):
        raise RuntimeError(
            f"murmur_hash32: unsupported index dtypes {pos_dtype=} {col_dtype=}; "
            f"expected one of {_SUPPORTED_INDEX_DTYPES}"
        )
    args = make_cpp_args(pos_dtype, col_dtype)
    return load_jit(
        "murmur_hash32",
        *args,
        cuda_files=["sampling/murmur_hash.cuh"],
        cuda_wrappers=[("murmur_hash32", f"MurmurHashKernel<{args}>::launch")],
    )


def _murmur_hash32_jit(
    seed: torch.Tensor, positions: torch.Tensor, col_indices: torch.Tensor
) -> torch.Tensor:
    """CUDA JIT path. Bit-identical to ``_murmur_hash32_triton``."""
    n = seed.shape[0]
    m = col_indices.shape[0]
    out = torch.empty((n, m), dtype=torch.uint32, device=seed.device)
    if n == 0 or m == 0:
        return out
    module = _jit_murmur_hash_module(positions.dtype, col_indices.dtype)
    module.murmur_hash32(seed, positions, col_indices, out.view(-1))
    return out


def _murmur_hash32_triton(
    seed: torch.Tensor, positions: torch.Tensor, col_indices: torch.Tensor
) -> torch.Tensor:
    """Triton reference implementation (kept as ROCm fallback).

    The JIT path validates shapes in its C++ launcher; this path has no such
    guard, so the checks the kernel relies on live here.
    """
    assert seed.ndim == 1 and positions.ndim == 1 and col_indices.ndim == 1, (
        f"inputs must be 1D {seed.shape=} {positions.shape=} {col_indices.shape=}"
    )
    assert seed.shape == positions.shape, (
        f"seed and positions must have the same shape {seed.shape=} {positions.shape=}"
    )
    n = seed.shape[0]
    m = col_indices.shape[0]
    hashed = torch.empty((n, m), dtype=torch.uint32, device=seed.device)
    if n == 0 or m == 0:
        return hashed

    BLOCK_SIZE = 1024
    grid = (n, triton.cdiv(m, BLOCK_SIZE))
    murmur_hash32_kernel[grid](
        seed, positions, col_indices, hashed, n, m, BLOCK_SIZE=BLOCK_SIZE
    )
    return hashed


def murmur_hash32(
    seed: torch.Tensor, positions: torch.Tensor, col_indices: torch.Tensor
) -> torch.Tensor:
    """Bit-identical MurmurHash3 x86_32 of ``seed``, ``positions``, ``col_indices``.

    ``seed`` is ``uint64`` (per row), ``positions`` is a per-row 32-bit index,
    and ``col_indices`` a per-column 32-bit index; the hash of the four 4-byte
    blocks lands in an ``(n, m)`` ``uint32`` tensor. CUDA uses the JIT kernel;
    ROCm falls back to the Triton reference.
    """
    if _is_hip:
        return _murmur_hash32_triton(seed, positions, col_indices)
    return _murmur_hash32_jit(seed, positions, col_indices)
