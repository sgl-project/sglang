"""Worker-only, bitwise verification. Never cast floating-point KV values."""

import hashlib


def page_digests(raw):
    """Synchronously snapshot packed uint8 pages on the caller's CUDA stream."""
    import torch

    if raw.dtype != torch.uint8 or raw.ndim != 2:
        raise ValueError("KV verification expects page-major uint8 bytes")
    host = raw.detach().contiguous().cpu().numpy()
    return [hashlib.sha256(row.tobytes()).digest() for row in host]
