"""
SGLang Platform Discovery and Lazy Initialization.

Provides `current_platform` as a module-level lazy singleton. On first access,
it discovers platform plugins via entry_points and instantiates the appropriate
SRTPlatform subclass.

Usage:
    from sglang.srt.platforms import current_platform
    print(current_platform.device_name)
"""

import logging
import pkgutil
from importlib.metadata import entry_points

from sglang.srt.environ import envs
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.plugins import PLATFORM_PLUGINS_GROUP, load_plugins_by_group

logger = logging.getLogger(__name__)

_current_platform: SRTPlatform | None = None


def _resolve_platform() -> SRTPlatform:
    """
    Discover and instantiate the active platform.

    Discovery flow:
    1. Branch on SGLANG_PLATFORM:

       SGLANG_PLATFORM set (front-loading filter):
         - Enumerate entry_points without importing any plugin modules
         - Only ep.load() + activate() the named plugin
         - Other plugins are never imported (avoids pulling their dependencies)
         - Plugin name not found → RuntimeError
         - activate() returns None → RuntimeError (hardware unavailable)

       SGLANG_PLATFORM unset (auto-discover):
         - Import and activate all discovered plugins
         - 0 activated → fallback base SRTPlatform
         - 1 activated → use it
         - N activated → RuntimeError (must set SGLANG_PLATFORM)

       SGLANG_PLATFORM matches against entry_point names.
    """
    selected = envs.SGLANG_PLATFORM.get()

    if selected:
        # Front-loading filter: only import and activate the specified plugin.
        # Other plugins' modules are never loaded — avoids pulling their deps.
        discovered = entry_points(group=PLATFORM_PLUGINS_GROUP)
        ep_map = {ep.name: ep for ep in discovered}

        if selected not in ep_map:
            available = ", ".join(f"'{n}'" for n in ep_map) if ep_map else "none"
            raise RuntimeError(
                f"SGLANG_PLATFORM={selected!r} not found in discovered platform plugins "
                f"(available: {available}). Install the plugin with 'pip install -e' "
                f"to register its entry_points."
            )

        try:
            plugin_fn = ep_map[selected].load()
            result = plugin_fn()
        except Exception:
            logger.exception("Failed to activate platform plugin: %s", selected)
            raise

        if result is None:
            raise RuntimeError(
                f"Platform plugin {selected!r} is installed but activate() returned None "
                f"(hardware not available on this machine?)."
            )
        logger.info("OOT platform plugin activated: %s -> %s", selected, result)
        return _load_platform_class(result)()

    # Auto-discover: import and activate all plugins, expect exactly one
    all_plugins = load_plugins_by_group(PLATFORM_PLUGINS_GROUP)

    activated: dict[str, str] = {}
    for name, (plugin_fn, _dist) in all_plugins.items():
        try:
            result = plugin_fn()
            if result is not None:
                activated[name] = result
                logger.info("OOT platform plugin activated: %s -> %s", name, result)
        except Exception:
            logger.exception("Failed to activate platform plugin: %s", name)

    if len(activated) == 0:
        logger.warning("No platform detected. Using base SRTPlatform with defaults.")
        return SRTPlatform()

    if len(activated) == 1:
        name, qualname = next(iter(activated.items()))
        return _load_platform_class(qualname)()

    # Multiple activated without SGLANG_PLATFORM
    names_str = ", ".join(f"'{n}'" for n in activated)
    raise RuntimeError(
        f"Multiple platform plugins activated: {names_str}. "
        f"Set SGLANG_PLATFORM to select one."
    )


def _load_platform_class(qualname: str) -> type:
    """Load an SRTPlatform subclass from its fully-qualified class name."""
    cls = pkgutil.resolve_name(qualname)
    if not isinstance(cls, type) or not issubclass(cls, SRTPlatform):
        raise TypeError(
            f"Expected an SRTPlatform subclass, got {type(cls)}: {qualname}"
        )
    return cls


def __getattr__(name: str):
    """Lazy initialization of current_platform on first access."""
    if name == "current_platform":
        global _current_platform
        if _current_platform is None:
            _current_platform = _resolve_platform()
        return _current_platform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
