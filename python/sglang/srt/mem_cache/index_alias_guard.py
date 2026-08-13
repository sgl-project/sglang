"""Ownership tracking for KV index tensors taken from a `req_to_token` row.

A row of `req_to_token` is mutable: the radix caches overwrite it during rematch.
A tensor viewing that storage which is *stored* somewhere outliving the current
call (a request field, a tree node, a deferred free queue) can therefore change
value before it is read.

`aliases_req_to_token()` answers that question and is always available, so the
allocators can skip taking ownership of inputs that are already owned (tree node
values and their slices). `install_debug_hooks()` additionally turns the rule
into an assertion on the stored-field paths, and is gated by
`SGLANG_DEBUG_MEMORY_POOL`.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

# (start, end, device) address ranges of every live req_to_token buffer.
_rows: list[tuple[int, int, torch.device]] = []
_hooks_installed = False


def register_req_to_token(req_to_token: torch.Tensor) -> None:
    """Ranges are never unregistered: a stale one can only over-report aliasing,
    which costs a redundant copy rather than losing ownership."""
    start = req_to_token.data_ptr()
    end = start + req_to_token.numel() * req_to_token.element_size()
    _rows.append((start, end, req_to_token.device))


def aliases_req_to_token(value: torch.Tensor) -> bool:
    """Whether `value`'s data lives inside a req_to_token buffer.

    An address-range test, not `_base`/storage identity: a slice of a tree node
    value is a view, yet it owns its data as far as this rule is concerned.
    """
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
    """Swap the guarded attributes to asserting properties.

    Safe because this runs from `ReqToTokenPool.__init__`, before any `Req` or
    `TreeNode` exists -- a data descriptor would shadow values already in
    `__dict__`.
    """
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
