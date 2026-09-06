# SPDX-License-Identifier: Apache-2.0

import logging
from collections import Counter, defaultdict
from collections.abc import Callable
from enum import Enum
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from sglang.multimodal_gen.runtime.platforms import (
    PLATFORM_PLUGINS_GROUP,
    get_selected_platform_dist,
)
from sglang.srt.environ import envs
from sglang.srt.plugins.hook_registry import HookRegistry as _HookRegistry
from sglang.srt.plugins.hook_registry import (
    HookSource,
    HookType,
    _current_plugin_source,
)

logger = logging.getLogger(__name__)

GENERAL_PLUGINS_GROUP = "sglang.multimodal_gen.plugins"


class _OnceState(Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class _Once:
    """A process-local initialization gate.

    Activation runs during single-threaded process startup, so this is
    deliberately unsynchronized. It exists for idempotency, for the re-entrant
    call a plugin callback can make, and to keep a failure terminal.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.state = _OnceState.NOT_STARTED
        self.error: BaseException | None = None

    def run(self, action: Callable[[], None]) -> bool:
        """Run *action* once; return False only for a re-entrant call."""
        if self.state is _OnceState.COMPLETE:
            return True
        if self.state is _OnceState.FAILED:
            raise RuntimeError(
                f"{self.name} previously failed: {self.error}"
            ) from self.error
        if self.state is _OnceState.RUNNING:
            return False

        self.state = _OnceState.RUNNING
        try:
            action()
        except BaseException as exc:
            self.error = exc
            self.state = _OnceState.FAILED
            raise
        self.state = _OnceState.COMPLETE
        return True

    def reset(self) -> None:
        self.state = _OnceState.NOT_STARTED
        self.error = None


_plugin_registration = _Once("Diffusion plugin registration")
_hook_application = _Once("Diffusion hook application")
_required_dist: str | None = None


class HookRegistry(_HookRegistry):
    # Rebound so diffusion hooks do not land in SRT's registry.
    _hooks = defaultdict(list)
    _patched = set()


def plugin_hook(target: str, type: HookType = HookType.AFTER) -> Callable:
    def decorator(hook: Callable) -> Callable:
        HookRegistry.register(target, hook, type)
        return hook

    return decorator


def _discard_hooks_from_source(source: HookSource) -> None:
    """Discard hooks registered by a plugin that failed to load or execute."""
    for target in tuple(HookRegistry._hooks):
        remaining = [hook for hook in HookRegistry._hooks[target] if hook[2] != source]
        if remaining:
            HookRegistry._hooks[target] = remaining
        else:
            del HookRegistry._hooks[target]


def _get_excluded_dists() -> set[str]:
    selected_dist = get_selected_platform_dist()
    platform_dists = {
        entry_point.dist.name
        for entry_point in entry_points(group=PLATFORM_PLUGINS_GROUP)
        if entry_point.dist
    }
    if selected_dist is None:
        return platform_dists
    return platform_dists - {selected_dist}


def _discover() -> dict[str, tuple[Callable[[], Any], str | None]]:
    allowed: set[str] | None = None
    allowed_str = envs.SGLANG_PLUGINS.get()
    if allowed_str:
        allowed = {name.strip() for name in allowed_str.split(",") if name.strip()}

    discovered = tuple(entry_points(group=GENERAL_PLUGINS_GROUP))
    if not discovered:
        logger.debug("No diffusion plugins found for group %s.", GENERAL_PLUGINS_GROUP)
        return {}

    excluded_dists = _get_excluded_dists()
    required_dist = get_selected_platform_dist()
    candidates: list[EntryPoint] = []
    for entry_point in discovered:
        dist_name = entry_point.dist.name if entry_point.dist else None
        if allowed is not None and entry_point.name not in allowed:
            logger.info(
                "Skipping diffusion plugin %s (not in SGLANG_PLUGINS)",
                entry_point.name,
            )
            continue
        if dist_name in excluded_dists:
            logger.info(
                "Skipping diffusion plugin %s (dist %s is not the selected platform)",
                entry_point.name,
                dist_name,
            )
            continue
        candidates.append(entry_point)

    counts = Counter(entry_point.name for entry_point in candidates)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise RuntimeError(
            "Diffusion plugin entry-point names must be unique: "
            + ", ".join(repr(name) for name in duplicates)
        )

    plugins: dict[str, tuple[Callable[[], Any], str | None]] = {}
    for entry_point in candidates:
        dist_name = entry_point.dist.name if entry_point.dist else None
        source = HookSource(plugin_name=entry_point.name, dist_name=dist_name)
        token = _current_plugin_source.set(source)
        try:
            callback = entry_point.load()
            if not callable(callback):
                raise TypeError(
                    f"Diffusion plugin {entry_point.name!r} must resolve to a callable"
                )
            plugins[entry_point.name] = (callback, dist_name)
            logger.info("Loaded diffusion plugin %s", entry_point.name)
        except Exception:
            _discard_hooks_from_source(source)
            if required_dist and dist_name == required_dist:
                raise
            logger.exception("Failed to load diffusion plugin %s", entry_point.name)
        finally:
            _current_plugin_source.reset(token)

    return plugins


def _require_hooks_applied(required_dist: str) -> None:
    unapplied = sorted(
        target
        for target, hooks in HookRegistry._hooks.items()
        if target not in HookRegistry._patched
        and any(source and source.dist_name == required_dist for _, _, source in hooks)
    )
    if unapplied:
        raise RuntimeError(
            f"Selected platform package {required_dist!r} could not apply hooks on: "
            + ", ".join(unapplied)
        )


def _register_plugins_once() -> str | None:
    plugins = _discover()
    # The selected platform's own plugins carry its hardware contract, so their
    # failures abort startup; third-party ones stay best-effort.
    required_dist = get_selected_platform_dist() if plugins else None

    for name, (func, dist_name) in plugins.items():
        source = HookSource(plugin_name=name, dist_name=dist_name)
        token = _current_plugin_source.set(source)
        try:
            func()
            logger.info("Executed diffusion plugin: %s", name)
        except Exception:
            _discard_hooks_from_source(source)
            if required_dist and dist_name == required_dist:
                raise
            logger.exception("Failed to execute diffusion plugin: %s", name)
        finally:
            _current_plugin_source.reset(token)

    return required_dist


def load_plugins() -> None:
    """Discover and execute diffusion plugin callbacks once per process.

    This phase only registers hooks. It deliberately does not resolve hook
    targets: resolving a dotted target imports its module, which makes a
    seemingly harmless plugin-discovery call capable of importing the entire
    worker runtime.

    Re-entrant calls from a plugin callback return immediately. A failed load is
    terminal for the process because arbitrary callback side effects cannot be
    rolled back safely.
    """
    _ensure_plugins_loaded()


def _ensure_plugins_loaded() -> bool:
    def register() -> None:
        global _required_dist
        _required_dist = _register_plugins_once()

    return _plugin_registration.run(register)


def apply_plugin_hooks() -> None:
    """Apply registered hooks once, at an explicit runtime-safe boundary.

    Hook target resolution is allowed to import target modules. Callers that
    require import ordering, notably spawned accelerator workers, must finish
    platform initialization before entering this phase.
    """
    if not _ensure_plugins_loaded():
        # The outer activation applies the complete registry after registration.
        return

    def apply() -> None:
        HookRegistry.apply_hooks()
        if _required_dist:
            _require_hooks_applied(_required_dist)

    _hook_application.run(apply)


def _reset_lifecycle_for_tests() -> None:
    global _required_dist
    _plugin_registration.reset()
    _hook_application.reset()
    _required_dist = None


__all__ = [
    "HookRegistry",
    "HookType",
    "apply_plugin_hooks",
    "load_plugins",
    "plugin_hook",
]
