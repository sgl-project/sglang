from __future__ import annotations

import torch


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
