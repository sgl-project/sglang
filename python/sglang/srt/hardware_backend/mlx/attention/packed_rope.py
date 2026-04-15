# SPDX-License-Identifier: Apache-2.0
# Per-request RoPE helper for packed / unified forward passes.

from __future__ import annotations

import mlx.core as mx


def apply_packed_rope(
    attn_module: object,
    queries: mx.array,
    keys: mx.array,
    cu_seqlens: list[int],
    offsets: list[int] | None = None,
) -> tuple[mx.array, mx.array]:
    """Apply per-request RoPE for packed sequences.

    Each segment delimited by ``cu_seqlens`` gets its own RoPE application
    starting at the corresponding offset.  When *offsets* is ``None`` every
    segment starts at position 0 (pure prefill).  For unified prefill+decode
    batches, decode segments carry ``offset=seq_len`` while prefill segments
    keep ``offset=0``.
    """
    q_parts = []
    k_parts = []
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        off = offsets[i] if offsets is not None else 0
        q_parts.append(attn_module.rope(queries[:, :, start:end, :], offset=off))
        k_parts.append(attn_module.rope(keys[:, :, start:end, :], offset=off))
    return mx.concatenate(q_parts, axis=2), mx.concatenate(k_parts, axis=2)