"""Ownership tracking for KV index tensors taken from a `req_to_token` row.

The radix caches rewrite such a row during rematch, so a view of it that outlives
the current call -- a request field, a tree node, a deferred free queue -- can
change value before it is read.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

# (start, end, device) address ranges of every live req_to_token buffer.
_rows: list[tuple[int, int, torch.device]] = []
_hooks_installed = False


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


def check(value: Any, what: str) -> None:
    if not isinstance(value, torch.Tensor):
        return
    assert not aliases_req_to_token(value), (
        f"{what} aliases a req_to_token row: the row may be overwritten before this "
        f"tensor is read. Snapshot it with `.to(dtype=torch.int64, copy=True)`."
    )


def _guarded_attr(attr: str, what: str) -> property:
    slot = f"_{attr}_guarded"

    def getter(self) -> Optional[Any]:
        return getattr(self, slot, None)

    def setter(self, value: Any) -> None:
        check(value, what)
        setattr(self, slot, value)

    return property(getter, setter)


def install_debug_hooks() -> None:
    # Must run before any Req/TreeNode exists: a data descriptor added later would
    # shadow values already in `__dict__`.
    global _hooks_installed
    if _hooks_installed:
        return
    _hooks_installed = True

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache import mamba_radix_cache, radix_cache, swa_radix_cache

    Req.prefix_indices = _guarded_attr("prefix_indices", "req.prefix_indices")
    for module in (radix_cache, swa_radix_cache, mamba_radix_cache):
        module.TreeNode.value = _guarded_attr(
            "value", f"{module.__name__.rsplit('.', 1)[-1]}.TreeNode.value"
        )
