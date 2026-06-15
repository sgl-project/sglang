from collections.abc import Sequence
from typing import Any


def apply_torch_npu_patches(torch_npu: Any, patches: Sequence[Sequence[Any]]) -> None:
    """Apply torch_npu patches across old and new torch_npu patch APIs."""
    if hasattr(torch_npu, "_apply_patches"):
        torch_npu._apply_patches(patches)
        return

    if hasattr(torch_npu, "_apply_all_patches"):
        torch_npu._apply_all_patches()
        return

    raise AttributeError(
        "torch_npu must provide either _apply_patches or _apply_all_patches"
    )
