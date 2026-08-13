"""Ownership of KV index tensors taken from a `req_to_token` row.

The radix caches rewrite such a row during rematch, so a view of it that outlives
the current call can change value before it is read. A deferred free consults this
to decide whether it must take a copy of what it queues.
"""

from __future__ import annotations

import torch

# (start, end, device) address ranges of every live req_to_token buffer.
_rows: list[tuple[int, int, torch.device]] = []


def register_req_to_token(req_to_token: torch.Tensor) -> None:
    # Never unregistered: a stale range only over-reports, costing a spare copy.
    start = req_to_token.data_ptr()
    end = start + req_to_token.numel() * req_to_token.element_size()
    _rows.append((start, end, req_to_token.device))


def aliases_req_to_token(value: torch.Tensor) -> bool:
    # Address range, not `_base`: a slice of a tree node value is a view yet owned.
    if not _rows or value.numel() == 0:
        return False
    ptr = value.data_ptr()
    return any(start <= ptr < end and value.device == dev for start, end, dev in _rows)
