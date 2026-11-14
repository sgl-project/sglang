from collections.abc import Sequence
from typing import Any


def apply_torch_npu_patches(torch_npu: Any, patches: Sequence[Sequence[Any]]) -> None:
    """Apply torch_npu patches across old and new torch_npu patch APIs."""
    if hasattr(torch_npu, "_apply_patches"):
        torch_npu._apply_patches(patches)
        return

    # Newer torch_npu moved the replacement-list API into monkey_patches.
    # _apply_all_patches() only runs registered groups; it cannot apply this list.
    try:
        apply_patches = torch_npu._init.patches.monkey_patches._apply_patches
    except AttributeError as exc:
        raise AttributeError(
            "torch_npu must provide _apply_patches at the top level or in "
            "torch_npu._init.patches.monkey_patches"
        ) from exc
    apply_patches(patches)
