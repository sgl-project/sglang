"""Debug-only guard against KV index tensors that alias a `req_to_token` row.

A row of `req_to_token` is mutable: the radix caches overwrite it during rematch.
Any tensor that is a view of that storage and is *stored* somewhere outliving the
current call (a request field, a tree node, a deferred queue) can therefore change
value before it is read. Enabled by `SGLANG_DEBUG_MEMORY_POOL`.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

# (storage data_ptr, device) of every live req_to_token buffer.
_guarded_rows: set[tuple[int, torch.device]] = set()
_hooks_installed = False


def arm(req_to_token: torch.Tensor) -> None:
    """Register a `req_to_token` buffer and install the property hooks."""
    _guarded_rows.add((req_to_token.untyped_storage().data_ptr(), req_to_token.device))
    _install_hooks()


def check(value: Any, what: str) -> None:
    if not _guarded_rows or not isinstance(value, torch.Tensor):
        return
    key = (value.untyped_storage().data_ptr(), value.device)
    assert key not in _guarded_rows, (
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


def _install_hooks() -> None:
    """Swap the guarded attributes to properties.

    Safe because `arm()` runs from `ReqToTokenPool.__init__`, before any `Req` or
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
