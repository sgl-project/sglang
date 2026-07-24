# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""CUDA JIT SHA-1 matching ``hashlib.sha1`` for contiguous device bytes.

Used by the presharded weight loader to fingerprint tensors without a full
device-to-host copy of multi-GB parameters. Digests are bit-identical to
``hashlib.sha1(prefix + data_bytes).digest()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_sha1_module() -> Module:
    return load_jit(
        "sha1",
        cuda_files=["memory/sha1.cuh"],
        cuda_wrappers=[
            ("sha1_bytes", "sha1_bytes"),
            ("sha1_prefix_data", "sha1_prefix_data"),
        ],
    )


def _as_u8_1d_cuda(t: torch.Tensor, name: str) -> torch.Tensor:
    if not t.is_cuda:
        raise RuntimeError(f"{name} requires a CUDA tensor")
    if t.dtype != torch.uint8:
        raise RuntimeError(f"{name} requires uint8, got {t.dtype}")
    if t.dim() != 1:
        raise RuntimeError(f"{name} requires a 1-D tensor, got shape {tuple(t.shape)}")
    return t if t.is_contiguous() else t.contiguous()


def sha1_bytes_cuda(data: torch.Tensor) -> bytes:
    """SHA-1 of a 1-D contiguous ``uint8`` CUDA tensor (20 raw bytes)."""
    data = _as_u8_1d_cuda(data, "sha1_bytes_cuda")
    digest = torch.empty(20, dtype=torch.uint8, device=data.device)
    _jit_sha1_module().sha1_bytes(digest, data)
    return bytes(digest.cpu().tolist())


def sha1_prefix_data_cuda(prefix: bytes | torch.Tensor, data: torch.Tensor) -> bytes:
    """SHA-1 of ``prefix || data`` matching ``hashlib.sha1(prefix + data_bytes)``.

    ``prefix`` may be a ``bytes`` object (copied to the data device once) or a
    1-D uint8 CUDA tensor.
    """
    data = _as_u8_1d_cuda(data, "sha1_prefix_data_cuda")
    if isinstance(prefix, (bytes, bytearray)):
        pref = torch.tensor(list(prefix), dtype=torch.uint8, device=data.device)
    else:
        pref = _as_u8_1d_cuda(prefix, "sha1_prefix_data_cuda(prefix)")
        if pref.device != data.device:
            pref = pref.to(device=data.device)

    digest = torch.empty(20, dtype=torch.uint8, device=data.device)
    _jit_sha1_module().sha1_prefix_data(digest, data, pref)
    return bytes(digest.cpu().tolist())


def sha1_hex_cuda(data: torch.Tensor) -> str:
    """Hex digest (40 chars) of a 1-D contiguous ``uint8`` CUDA tensor."""
    return sha1_bytes_cuda(data).hex()
