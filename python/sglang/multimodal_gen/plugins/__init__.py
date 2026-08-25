"""
Plugin framework for sglang-diffusion (``sglang.multimodal_gen``).

Reuses the discovery machinery of :mod:`sglang.srt.plugins` for the diffusion
engine, with diffusion-specific entry point groups and env vars. Two entry
point groups are supported:

1. Platform plugins (``sglang.multimodal_gen.platforms``): a callable
   ``activate() -> str | None`` returning the qualname of a
   :class:`sglang.multimodal_gen.runtime.platforms.interface.Platform`
   subclass, or ``None`` when the hardware is not present.
2. General plugins (``sglang.multimodal_gen.plugins``): a callable
   ``register() -> None`` that registers hooks on the shared
   :class:`sglang.srt.plugins.hook_registry.HookRegistry`.

Selection env vars:
- ``SGLANG_DIFFUSION_PLATFORM``: platform plugin (entry point) name.
- ``SGLANG_DIFFUSION_PLUGINS``: comma-separated plugin name whitelist.

NOTE: this module must not import anything from
``sglang.multimodal_gen.runtime`` at module scope. It is loaded from the
package ``__init__`` before any hook is applied, and pulling in the runtime
here would resolve ``current_platform`` too early.

PLUGIN AUTHORS - partial initialization constraint: ``load_diffusion_plugins()``
runs from ``sglang/multimodal_gen/__init__.py`` *before* ``PipelineConfig``,
``SamplingParams`` and ``DiffGenerator`` are bound on the package. A
``register()`` body must therefore not do
``from sglang.multimodal_gen import <top-level name>``: it raises ImportError,
which this loader swallows into a log line, silently disabling the plugin.
Register hooks by dotted-string target instead; importing *submodules*
(``sglang.multimodal_gen.configs.sample``, ...) is fine.
"""

import logging
from collections.abc import Callable
from typing import Any

from sglang.srt.environ import envs
from sglang.srt.plugins import excluded_dists_for, load_plugins_by_group
from sglang.srt.plugins.hook_registry import (
    HookRegistry,
    HookSource,
    _current_plugin_source,
)

logger = logging.getLogger(__name__)

# Entry point group names
DIFFUSION_PLATFORM_PLUGINS_GROUP = "sglang.multimodal_gen.platforms"
DIFFUSION_GENERAL_PLUGINS_GROUP = "sglang.multimodal_gen.plugins"

# Guard against multiple loads in the same process
_plugins_loaded = False


def discover_diffusion_plugins(
    group: str,
    excluded_dists: set[str] | None = None,
) -> dict[str, tuple[Callable[[], Any], str | None]]:
    """
    Discover and import diffusion plugins registered under ``group``.

    Delegates to :func:`sglang.srt.plugins.load_plugins_by_group` with the
    diffusion whitelist env var, so both engines share one discovery
    implementation.
    """
    return load_plugins_by_group(
        group,
        excluded_dists=excluded_dists,
        whitelist_env=envs.SGLANG_DIFFUSION_PLUGINS,
    )


def _get_excluded_dists() -> set[str]:
    """Dist names to skip when ``SGLANG_DIFFUSION_PLATFORM`` is set.

    Returns dist names that provide a diffusion platform plugin but are NOT
    the one selected by ``SGLANG_DIFFUSION_PLATFORM``. This prevents
    unselected platform packages from registering hooks that pull their
    hardware dependencies.
    """
    return excluded_dists_for(
        DIFFUSION_PLATFORM_PLUGINS_GROUP,
        envs.SGLANG_DIFFUSION_PLATFORM.get(),
    )


def load_diffusion_plugins() -> None:
    """
    Load and execute all diffusion general plugins, then apply their hooks.

    Idempotent - safe to call multiple times. Plugins are functions whose side
    effects (registering hooks, replacing classes, etc.) are the desired
    behavior; return values are ignored.

    This is called from ``sglang/multimodal_gen/__init__.py`` so that it runs
    in every diffusion process, including spawned workers.
    """
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True

    plugins: dict[str, tuple[Callable[[], Any], str | None]] = {}
    try:
        plugins = discover_diffusion_plugins(
            DIFFUSION_GENERAL_PLUGINS_GROUP,
            excluded_dists=_get_excluded_dists(),
        )
    except Exception:
        # This runs from ``sglang/multimodal_gen/__init__.py``, so letting a
        # discovery error escape would make ``import sglang.multimodal_gen``
        # fail outright (e.g. a malformed third-party ``*.dist-info`` breaking
        # ``importlib.metadata.entry_points()``).
        logger.exception(
            "Diffusion plugin discovery failed; continuing without plugins"
        )

    for name, (func, dist_name) in plugins.items():
        source = HookSource(plugin_name=name, dist_name=dist_name)
        token = _current_plugin_source.set(source)
        try:
            func()
            logger.info("Executed diffusion plugin: %s", name)
        except Exception:
            logger.exception("Failed to execute diffusion plugin: %s", name)
        finally:
            _current_plugin_source.reset(token)

    # Apply all registered hooks. HookRegistry is process-global and shared
    # with sglang.srt.plugins: its ``_patched`` set spans both loaders, so a
    # target already patched by whichever loader ran first is skipped here
    # (apply_hooks logs a warning naming the dropped hook sources). Hooks
    # registered on targets nobody patched yet are applied normally.
    HookRegistry.apply_hooks()
