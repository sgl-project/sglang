from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeVar

import torch

OOTHookRequirement = Literal["required", "recommended", "optional"]
F = TypeVar("F", bound=Callable)


@dataclass(frozen=True)
class OOTHookPlaceholder:
    """Metadata for an OOT hook placeholder.

    Attributes:
        target: Fully-qualified function path that plugins should replace with
            ``HookType.REPLACE``.
        requirement: Severity for an active OOT platform that does not replace
            this placeholder.
        reason: Human-readable explanation of why the placeholder exists.
    """

    target: str
    requirement: OOTHookRequirement
    reason: str


_OOT_HOOK_PLACEHOLDERS: dict[str, OOTHookPlaceholder] = {}


def oot_hook_placeholder(
    *,
    requirement: OOTHookRequirement,
    reason: str,
) -> Callable[[F], F]:
    """Mark a function as an OOT hook placeholder.

    The plugin loader inspects this registry after general plugins execute. If
    the active platform is out-of-tree and a plugin did not register a
    ``HookType.REPLACE`` hook for the placeholder, the loader emits a log with
    severity based on ``requirement``.

    Args:
        requirement: How important it is for an OOT plugin to replace this
            placeholder. ``required`` emits a warning, ``recommended`` emits an
            info log, and ``optional`` emits a debug log.
        reason: Short explanation included in the validation log.
    """

    def decorator(fn: F) -> F:
        target = f"{fn.__module__}.{fn.__name__}"
        _OOT_HOOK_PLACEHOLDERS[target] = OOTHookPlaceholder(
            target=target,
            requirement=requirement,
            reason=reason,
        )
        return fn

    return decorator


def get_oot_hook_placeholders() -> tuple[OOTHookPlaceholder, ...]:
    """Return registered OOT hook placeholders for plugin validation."""
    return tuple(_OOT_HOOK_PLACEHOLDERS.values())


@oot_hook_placeholder(
    requirement="optional",
    reason="The default follows current_platform.is_out_of_tree().",
)
def is_oot() -> bool:
    """Return whether the active runtime should use OOT fallback behavior.

    This is a hook placeholder for out-of-tree platform plugins. By default it
    follows ``current_platform.is_out_of_tree()`` so module-level checks remain
    correct even when a module is imported before general plugin hooks are
    applied. An OOT platform plugin may still replace this function during
    plugin initialization if it needs a narrower policy than "all OOT
    platforms".
    """
    from sglang.srt.platforms import current_platform

    return current_platform.is_out_of_tree()


@oot_hook_placeholder(
    requirement="required",
    reason="GroupCoordinator needs an OOT-specific torch.device.",
)
def get_oot_group_coordinator_device(local_rank: int) -> torch.device:
    """Return the device used by ``GroupCoordinator`` for an OOT platform.

    ``GroupCoordinator`` needs a real ``torch.device`` for per-rank metadata
    tensors and backend-specific process group setup. Core SGLang cannot infer
    the correct device string for every out-of-tree platform, so OOT platform
    plugins must replace this placeholder when ``is_oot()`` returns ``True``.

    Args:
        local_rank: The process-local device rank passed to the distributed
            group coordinator.

    Raises:
        NotImplementedError: If an OOT platform enables ``is_oot()`` without
            also replacing this function.
    """
    raise NotImplementedError(
        "OOT platform plugins must replace "
        "sglang.srt.utils.oot.get_oot_group_coordinator_device before "
        "GroupCoordinator is initialized."
    )
