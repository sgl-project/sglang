# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the streaming (chunk-by-chunk) VAE encode/decode caches."""

from __future__ import annotations

import torch


def set_or_copy(state: dict, key, new_value: torch.Tensor) -> None:
    """Write ``new_value`` into ``state[key]`` preserving the storage pointer.

    First write clones into a fresh buffer; subsequent same-shape writes
    ``copy_`` in place. Pointer stability lets the streaming cache reuse its
    buffer slot across chunks without realloc churn.
    """
    cur = state.get(key)
    if cur is not None and cur.shape == new_value.shape:
        cur.copy_(new_value)
    else:
        state[key] = new_value.clone()
