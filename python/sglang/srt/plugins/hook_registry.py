"""
Hook registry for SGLang plugins.

Provides before/after/around/replace hooks that can be applied to any
function, method, or class in the sglang codebase. Hooks are registered
during plugin loading and applied before the engine starts.

Usage:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def my_timer(original_fn, *args, **kwargs):
        start = time.perf_counter()
        result = original_fn(*args, **kwargs)
        print(f"Elapsed: {time.perf_counter() - start:.3f}s")
        return result

    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler.schedule",
        my_timer,
        HookType.AROUND,
    )
"""

import contextvars
import functools
import logging
import pkgutil
import sys
import types
from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from typing import NamedTuple

logger = logging.getLogger(__name__)


class HookSource(NamedTuple):
    """Identifies which plugin registered a hook."""

    plugin_name: str  # entry_point name, e.g. "xpu_hooks"
    dist_name: str | None  # distribution name, e.g. "sglang_xpu_platform"


# Set by load_plugins() around each plugin's func() call, read by register().
_current_plugin_source: contextvars.ContextVar[HookSource | None] = (
    contextvars.ContextVar("_current_plugin_source", default=None)
)


def _format_source(source: HookSource | None) -> str:
    """Format source info for log messages."""
    if source is None:
        return "unknown"
    if source.dist_name:
        return f"plugin={source.plugin_name}, dist={source.dist_name}"
    return f"plugin={source.plugin_name}"


class HookType(Enum):
    """Types of hooks that can be applied to functions or classes."""

    BEFORE = "before"  # Execute before original; can modify args
    AFTER = "after"  # Execute after original; can modify return value
    AROUND = "around"  # Wrap original; full control over execution
    REPLACE = "replace"  # Replace the original function or class entirely


class HookRegistry:
    """
    Global registry for function/method/class hooks.

    Thread safety: All registration should happen during load_plugins()
    phase (single-threaded). apply_hooks() should be called once before the
    engine starts serving requests.
    """

    _hooks: dict[str, list[tuple[HookType, Callable, HookSource | None]]] = defaultdict(
        list
    )
    _patched: set[str] = set()

    @classmethod
    def register(
        cls,
        target: str,
        hook: Callable,
        hook_type: HookType = HookType.AFTER,
        *,
        source: HookSource | None = None,
    ):
        """
        Register a hook on a target function, method, or class.

        Args:
            target: Fully-qualified dotted path to the target.
                    e.g. "sglang.srt.managers.scheduler.Scheduler.schedule"
                    or   "sglang.srt.managers.scheduler.Scheduler" (class)
            hook: The hook callable (function or class). Signature depends on hook_type:
                - BEFORE:  fn(*args, **kwargs) -> (args, kwargs) or None
                - AFTER:   fn(result, *args, **kwargs) -> new_result or None
                - AROUND:  fn(original_fn, *args, **kwargs) -> result
                - REPLACE: fn(*args, **kwargs) -> result   (function replacement)
                           MyClass                         (class replacement)
            hook_type: Type of hook (default: AFTER).
            source: Optional source info. If None, auto-read from context var
                set by ``load_plugins()``.

        Raises:
            TypeError: If a class is passed with a hook_type other than REPLACE.
        """
        if isinstance(hook, type) and hook_type != HookType.REPLACE:
            raise TypeError(
                f"Class {hook.__name__} can only be used with HookType.REPLACE, "
                f"got HookType.{hook_type.name}. "
                f"Use a function for BEFORE/AFTER/AROUND hooks."
            )
        resolved_source = source or _current_plugin_source.get()
        # Warn on duplicate REPLACE for the same target
        if hook_type == HookType.REPLACE:
            existing_replace = [
                (h, src) for ht, h, src in cls._hooks[target] if ht == HookType.REPLACE
            ]
            if existing_replace:
                prev, prev_src = existing_replace[-1]
                prev_name = getattr(prev, "__qualname__", None) or repr(prev)
                new_name = getattr(hook, "__qualname__", None) or repr(hook)
                logger.warning(
                    "Multiple REPLACE hooks on '%s': previous (%s [%s]) will be "
                    "overridden by (%s [%s]). The last registered REPLACE takes effect.",
                    target,
                    prev_name,
                    _format_source(prev_src),
                    new_name,
                    _format_source(resolved_source),
                )
        cls._hooks[target].append((hook_type, hook, resolved_source))
        logger.debug(
            "Registered %s hook on %s [%s]",
            hook_type.value,
            target,
            _format_source(resolved_source),
        )

    @classmethod
    def apply_hooks(cls):
        """
        Apply all registered hooks to their target functions/classes.

        This performs the actual monkey-patching. Should be called once after
        all plugins have been loaded and before the engine starts.

        Targets with class REPLACE hooks are applied first, so that
        subsequent method-level hooks (AROUND, BEFORE, AFTER) on child
        attributes resolve against the *replaced* class rather than the
        original.
        """
        sorted_items = sorted(cls._hooks.items(), key=cls._target_sort_key)
        for target, hooks in sorted_items:
            if target in cls._patched:
                continue
            try:
                cls._apply_target(target, hooks)
                cls._patched.add(target)
            except Exception:
                logger.exception("Failed to apply hooks to %s", target)

    @staticmethod
    def _target_sort_key(item):
        """Sort key: class REPLACE targets (tier 0) before all others (tier 1).

        This ensures that when a class is replaced, subsequent method-level
        hooks on ``ClassName.method`` resolve against the replacement class.
        """
        _target, hooks = item
        has_class_replace = any(
            isinstance(h, type) and ht == HookType.REPLACE for ht, h, _ in hooks
        )
        return (0 if has_class_replace else 1, _target)

    @classmethod
    def _apply_target(cls, target: str, hooks: list):
        """Resolve target, build wrapper chain, and replace the original."""
        parts = target.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid target path (need at least module.attr): {target}"
            )

        obj_path, attr_name = parts
        obj = pkgutil.resolve_name(obj_path)

        # Check if the original is a classmethod or staticmethod by
        # inspecting __dict__ before getattr() triggers the descriptor
        # protocol (which would lose the wrapper type for classmethod).
        original = getattr(obj, attr_name)
        is_classmethod = False
        is_staticmethod = False
        if isinstance(obj, type):
            raw_attr = obj.__dict__.get(attr_name)
            if isinstance(raw_attr, classmethod):
                is_classmethod = True
                original = raw_attr.__func__
            elif isinstance(raw_attr, staticmethod):
                is_staticmethod = True
                original = raw_attr.__func__

        # Cross-target conflict detection: if the parent object is a class
        # that was already class-REPLACE'd, and the replacement class defines
        # its own version of this method, a method REPLACE here will silently
        # override the replacement class's implementation.
        if isinstance(obj, type) and obj_path in cls._patched:
            has_method_replace = any(ht == HookType.REPLACE for ht, _, _ in hooks)
            if has_method_replace and attr_name in obj.__dict__:
                replace_sources = [
                    _format_source(src)
                    for ht, _, src in hooks
                    if ht == HookType.REPLACE
                ]
                logger.warning(
                    "Method REPLACE on '%s' will override the class REPLACE's "
                    "own implementation of '%s'. If this is unintended, remove "
                    "the method REPLACE and modify the replacement class "
                    "directly, or use AROUND to wrap it. (from: %s)",
                    target,
                    attr_name,
                    ", ".join(replace_sources),
                )

        # Guard: if the target is a class, only REPLACE is safe. Wrapping a
        # class in a function would break isinstance/issubclass/inheritance.
        if isinstance(original, type):
            bad = [ht for ht, _, _ in hooks if ht != HookType.REPLACE]
            if bad:
                raise TypeError(
                    f"Target '{target}' is a class. Only HookType.REPLACE is "
                    f"allowed for class targets (got {bad[0].value}). "
                    f"To hook a method, use '{target}.<method_name>' instead."
                )

        # Warn about risky hook combinations
        hook_types = [ht for ht, _, _ in hooks]
        around_count = hook_types.count(HookType.AROUND)
        has_replace = HookType.REPLACE in hook_types
        has_others = any(ht != HookType.REPLACE for ht in hook_types)

        if around_count > 1:
            around_sources = [
                _format_source(src) for ht, _, src in hooks if ht == HookType.AROUND
            ]
            logger.warning(
                "Multiple AROUND hooks on '%s' (%d hooks, from: %s). If any AROUND hook "
                "skips calling original_fn, inner hooks will be bypassed.",
                target,
                around_count,
                ", ".join(around_sources),
            )
        if has_replace and has_others:
            logger.info(
                "Target '%s' has both REPLACE and %s hooks. "
                "REPLACE will be applied first, then wrapped by other hooks.",
                target,
                ", ".join(
                    sorted({ht.value for ht in hook_types if ht != HookType.REPLACE})
                ),
            )

        # Build the wrapper chain.
        # Sort: REPLACE hooks first (stable sort preserves registration order
        # within the same type). This ensures AROUND/BEFORE/AFTER always wrap
        # the replaced function, regardless of registration order.
        sorted_hooks = sorted(
            hooks, key=lambda h: (0 if h[0] == HookType.REPLACE else 1)
        )
        wrapped = original
        for hook_type, hook, _src in sorted_hooks:
            if isinstance(hook, type) and hook_type == HookType.REPLACE:
                # Class replacement: direct substitution to preserve type identity.
                # This keeps isinstance(), issubclass(), and inheritance working.
                wrapped = hook
            else:
                wrapped = _wrap_fn(wrapped, hook, hook_type)

        # Restore classmethod/staticmethod decorator if the original had one.
        if is_classmethod:
            wrapped = classmethod(wrapped)
            logger.debug("Preserved @classmethod decorator for %s", target)
        elif is_staticmethod:
            wrapped = staticmethod(wrapped)
            logger.debug("Preserved @staticmethod decorator for %s", target)

        setattr(obj, attr_name, wrapped)

        # Propagate the patch to all other modules that imported the original
        # via ``from source_module import name``.  Python's ``from X import Y``
        # copies the reference at import time; patching X alone leaves
        # importers with a stale binding.
        if wrapped is not original:
            extra = _propagate_patch(original, wrapped, obj)
            if extra:
                logger.debug(
                    "Propagated patch for %s to %d additional module(s)",
                    target,
                    extra,
                )

        sources = sorted({_format_source(src) for _, _, src in hooks})
        logger.info(
            "Applied %d hook(s) to %s (from: %s)",
            len(hooks),
            target,
            ", ".join(sources),
        )

    @classmethod
    def reset(cls):
        """Reset all hooks and patches. Primarily for testing."""
        cls._hooks.clear()
        cls._patched.clear()


def _propagate_patch(original: object, wrapped: object, source_module: object) -> int:
    """Propagate a monkey-patch to all modules holding a stale ``from X import Y`` binding.

    After ``setattr(source_module, name, wrapped)`` updates the defining module,
    other modules that did ``from source_module import name`` still hold a direct
    reference to the old *original* object.  This walks ``sys.modules`` and
    replaces every such stale binding with *wrapped*.

    Returns the number of additional module attributes that were patched.
    """
    patched_count = 0
    for mod in list(sys.modules.values()):
        if mod is source_module or mod is None:
            continue
        if not isinstance(mod, types.ModuleType):
            continue
        try:
            mod_vars = vars(mod)
        except TypeError:
            continue
        for attr_name, attr_value in list(mod_vars.items()):
            if attr_value is original:
                try:
                    setattr(mod, attr_name, wrapped)
                    patched_count += 1
                except (AttributeError, TypeError):
                    pass
    return patched_count


def _wrap_fn(original_fn: Callable, hook: Callable, hook_type: HookType) -> Callable:
    """Create a wrapper function based on the hook type."""
    if hook_type == HookType.REPLACE:

        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):
            return hook(*args, **kwargs)

        wrapper.__wrapped__ = original_fn
        return wrapper

    elif hook_type == HookType.BEFORE:

        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):
            result = hook(*args, **kwargs)
            if result is not None:
                args, kwargs = result
            return original_fn(*args, **kwargs)

        wrapper.__wrapped__ = original_fn
        return wrapper

    elif hook_type == HookType.AFTER:

        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):
            result = original_fn(*args, **kwargs)
            modified = hook(result, *args, **kwargs)
            return modified if modified is not None else result

        wrapper.__wrapped__ = original_fn
        return wrapper

    elif hook_type == HookType.AROUND:

        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):
            return hook(original_fn, *args, **kwargs)

        wrapper.__wrapped__ = original_fn
        return wrapper

    else:
        raise ValueError(f"Unknown hook type: {hook_type}")


def plugin_hook(
    target: str,
    type: HookType = HookType.AFTER,
) -> Callable:
    """Decorator that registers a function or class as a hook on *target*.

    Usage::

        # Function hook (AROUND)
        @plugin_hook("sglang.srt.managers.scheduler.Scheduler.schedule",
                      type=HookType.AROUND)
        def my_timer(original_fn, *args, **kwargs):
            start = time.perf_counter()
            result = original_fn(*args, **kwargs)
            print(f"Elapsed: {time.perf_counter() - start:.3f}s")
            return result

        # Class replacement (REPLACE)
        @plugin_hook("sglang.srt.managers.scheduler.Scheduler",
                      type=HookType.REPLACE)
        class MyScheduler(Scheduler):
            ...

    The decorated function/class is returned unchanged so it can still be
    used directly if needed.
    """

    def decorator(hook: Callable) -> Callable:
        HookRegistry.register(target, hook, type)
        return hook

    return decorator
