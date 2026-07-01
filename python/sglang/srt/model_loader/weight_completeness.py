from __future__ import annotations

from typing import Callable, Container, Iterable


def required_weight_names(model) -> Iterable[str]:
    yield from (name for name, _ in model.named_parameters())
    yield from _named_persistent_buffers(model)


def _named_persistent_buffers(model) -> Iterable[str]:
    for module_prefix, module in model.named_modules():
        non_persistent = getattr(module, "_non_persistent_buffers_set", set())
        for name, _ in module.named_buffers(recurse=False):
            if name in non_persistent:
                continue
            yield f"{module_prefix}.{name}" if module_prefix else name


def unloaded_required_params(
    param_names: Iterable[str],
    loaded: Container[str],
    is_optional: Callable[[str], bool],
) -> set[str]:
    """Return required parameters that were not loaded."""
    return {
        name for name in param_names if name not in loaded and not is_optional(name)
    }
