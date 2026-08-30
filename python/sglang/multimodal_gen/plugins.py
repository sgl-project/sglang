# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from collections.abc import Callable
from importlib.metadata import entry_points

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.platforms import (
    BUILTIN_PLATFORM_NAMES,
    PLATFORM_PLUGINS_GROUP,
)
from sglang.srt.plugins import execute_plugin_callbacks, load_plugins_by_group
from sglang.srt.plugins.hook_registry import HookRegistry as _HookRegistry
from sglang.srt.plugins.hook_registry import HookType

logger = logging.getLogger(__name__)

GENERAL_PLUGINS_GROUP = "sglang.multimodal_gen.plugins"
_plugins_loaded = False


class HookRegistry(_HookRegistry):
    # Keep diffusion hooks isolated while reusing HookRegistry's cls-relative logic.
    _hooks = defaultdict(list)
    _patched = set()


def plugin_hook(target: str, type: HookType = HookType.AFTER) -> Callable:
    def decorator(hook: Callable) -> Callable:
        HookRegistry.register(target, hook, type)
        return hook

    return decorator


def _get_excluded_dists() -> set[str]:
    selected_platform = envs.SGLANG_DIFFUSION_PLATFORM_OVERRIDE.strip()
    if not selected_platform:
        return set()

    platform_entries = tuple(entry_points(group=PLATFORM_PLUGINS_GROUP))
    if selected_platform.lower() in BUILTIN_PLATFORM_NAMES:
        return {
            entry_point.dist.name
            for entry_point in platform_entries
            if entry_point.dist
        }

    selected_dists = {
        entry_point.dist.name
        for entry_point in platform_entries
        if entry_point.dist and entry_point.name == selected_platform
    }
    return {
        entry_point.dist.name
        for entry_point in platform_entries
        if entry_point.dist and entry_point.dist.name not in selected_dists
    }


def load_plugins() -> None:
    """Load and apply diffusion hooks once in the current process."""
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True

    plugins = load_plugins_by_group(
        GENERAL_PLUGINS_GROUP,
        excluded_dists=_get_excluded_dists(),
        exclusion_reason="diffusion platform selection",
    )
    execute_plugin_callbacks(plugins, HookRegistry, logger, "diffusion")


__all__ = ["HookRegistry", "HookType", "load_plugins", "plugin_hook"]
