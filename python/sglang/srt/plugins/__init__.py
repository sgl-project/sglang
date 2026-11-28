"""
SGLang Unified Plugin Framework.

Supports two types of plugins via setuptools entry_points:
1. Hardware Platform Plugins (sglang.srt.platforms) - register custom hardware platforms
2. General Plugins (sglang.srt.plugins) - inject hooks into functions/methods, replace classes, etc.

Plugins are discovered automatically when installed via pip.
- Platform plugins: use ``SGLANG_PLATFORM`` to select when multiple are installed.
- General plugins: use ``SGLANG_PLUGINS`` (comma-separated) to restrict which are loaded.
"""

import logging
from collections import defaultdict
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from sglang.srt.environ import envs
from sglang.srt.plugins.hook_registry import (
    HookRegistry,
    HookSource,
    _current_plugin_source,
)

logger = logging.getLogger(__name__)

# Entry point group names
PLATFORM_PLUGINS_GROUP = "sglang.srt.platforms"
GENERAL_PLUGINS_GROUP = "sglang.srt.plugins"

# Guard against multiple loads in the same process
_plugins_loaded = False


def load_plugins_by_group(
    group: str,
    excluded_dists: set[str] | None = None,
    required_plugins: frozenset[str] = frozenset(),
) -> dict[str, tuple[Callable[[], Any], str | None]]:
    """
    Discover and load plugins registered under the given entry point group.

    Args:
        group: The setuptools entry_point group name.
        excluded_dists: Distribution names to skip. Plugins from these
            distributions are never ``ep.load()``-ed (avoids importing
            their modules and pulling hardware-specific dependencies).
        required_plugins: Plugin names that raise instead of being skipped
            when missing, filtered out, or failing to load.

    Returns:
        Dictionary mapping plugin name to ``(callable, dist_name)``.
    """
    # SGLANG_PLUGINS whitelist (comma-separated plugin names)
    allowed_set: set[str] | None = None
    allowed_str = envs.SGLANG_PLUGINS.get()
    if allowed_str:
        allowed_set = {x.strip() for x in allowed_str.split(",") if x.strip()}

    discovered = entry_points(group=group)
    missing_required = required_plugins - {ep.name for ep in discovered}
    if missing_required:
        raise RuntimeError(
            f"Cannot discover required plugins {', '.join(sorted(missing_required))}."
        )

    if len(discovered) == 0:
        logger.debug("No plugins found for group %s.", group)
        return {}

    logger.info("Available plugins for group %s:", group)
    for ep in discovered:
        logger.info("  - %s -> %s", ep.name, ep.value)

    plugins: dict[str, tuple[Callable[[], Any], str | None]] = {}
    for ep in discovered:
        if allowed_set is not None and ep.name not in allowed_set:
            if ep.name in required_plugins:
                raise RuntimeError(
                    f"Required plugin {ep.name!r} must be set in SGLANG_PLUGINS."
                )
            logger.info("Skipping plugin %s (not in SGLANG_PLUGINS)", ep.name)
            continue
        dist_name = ep.dist.name if ep.dist else None
        if excluded_dists and dist_name in excluded_dists:
            if ep.name in required_plugins:
                raise RuntimeError(
                    f"Required plugin {ep.name!r} is excluded by SGLANG_PLATFORM."
                )
            logger.info(
                "Skipping plugin %s (dist %s excluded by SGLANG_PLATFORM)",
                ep.name,
                dist_name,
            )
            continue

        token = _current_plugin_source.set(HookSource(ep.name, dist_name))
        try:
            func = ep.load()
            plugins[ep.name] = (func, dist_name)
            logger.info("Loaded plugin %s from group %s", ep.name, group)
        except Exception as e:
            if ep.name in required_plugins:
                raise RuntimeError(f"Required plugin {ep.name!r} failed to load") from e
            logger.exception("Failed to load plugin %s from group %s", ep.name, group)
        finally:
            _current_plugin_source.reset(token)

    return plugins


def _get_excluded_dists() -> set[str]:
    """Compute dist names to skip when ``SGLANG_PLATFORM`` is set.

    Returns dist names that provide a platform plugin but are NOT the one
    selected by ``SGLANG_PLATFORM``.  This prevents unselected platform
    packages from registering hooks that pull their hardware dependencies.
    """
    selected = envs.SGLANG_PLATFORM.get()
    if not selected:
        return set()
    platform_eps = entry_points(group=PLATFORM_PLUGINS_GROUP)
    return {ep.dist.name for ep in platform_eps if ep.dist and ep.name != selected}


def load_plugins():
    """
    Load and execute all general plugins, then apply registered hooks.

    Idempotent - safe to call multiple times. General plugins are functions
    whose side effects (registering hooks, replacing classes, etc.) are the
    desired behavior. Return values are ignored.

    When ``SGLANG_PLATFORM`` is set, general plugins from unselected platform
    packages are automatically skipped (avoids pulling their dependencies).

    After all plugins execute, ``HookRegistry.apply_hooks()`` is called
    automatically so callers only need this single function call.

    This should be called early in every process (main, engine core, workers).
    """
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True

    required_plugins = frozenset(envs.SGLANG_REQUIRED_PLUGINS.get())
    plugins = load_plugins_by_group(
        GENERAL_PLUGINS_GROUP,
        excluded_dists=_get_excluded_dists(),
        required_plugins=required_plugins,
    )

    for name, (func, dist_name) in plugins.items():
        source = HookSource(plugin_name=name, dist_name=dist_name)
        token = _current_plugin_source.set(source)
        try:
            func()
            logger.info("Executed general plugin: %s", name)
        except Exception as e:
            if name in required_plugins:
                raise RuntimeError(
                    f"Required plugin {name!r} failed to execute."
                ) from e
            logger.exception("Failed to execute general plugin: %s", name)
        finally:
            _current_plugin_source.reset(token)

    # Apply all registered hooks (idempotent — already-patched targets are skipped).
    HookRegistry.apply_hooks()

    failed = HookRegistry.failed_targets()
    unapplied = defaultdict(list)
    for target, hooks in HookRegistry._hooks.items():
        start = HookRegistry._applied_count.get(target, 0)
        for index, (_, _, source) in enumerate(hooks[start:], start):
            required = (target, index) in HookRegistry._required
            if required or source and source.plugin_name in required_plugins:
                unapplied[target].append(source.plugin_name if source else "unknown")

    if unapplied:
        errors = []
        for target, plugin_names in sorted(unapplied.items()):
            e = RuntimeError(f"{', '.join(dict.fromkeys(plugin_names))} -> {target}")
            e.__cause__ = failed.get(target)
            errors.append(e)
        raise RuntimeError(
            f"Required plugins could not apply hooks: {'; '.join(str(e) for e in errors)}"
        ) from ExceptionGroup("hook apply failures", errors)
