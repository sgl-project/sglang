from __future__ import annotations

from typing import Callable, Container, Iterable


def unloaded_required_params(
    param_names: Iterable[str],
    loaded: Container[str],
    is_optional: Callable[[str], bool],
) -> set[str]:
    """Return required parameters that were not loaded."""
    return {
        name for name in param_names if name not in loaded and not is_optional(name)
    }
