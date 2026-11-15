import builtins
import re
from types import FunctionType
from typing import List, Union

from sglang_simulator.hook.base_hook import BaseHook, _register_hooks
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


CLASS_HOOKS: List[BaseHook] = []
_MATCHED_CLASS_HOOKS = set()

_builtins_build_class_ = builtins.__build_class__


def _custom_build_class_(func, name: str, *bases, **kwargs):
    for hook in CLASS_HOOKS:
        if (
            hook.REGEX
            and hook.HOOK_CLASS_NAME
            and re.search(hook.HOOK_CLASS_NAME, name)
        ) or name == hook.HOOK_CLASS_NAME:
            module_name = None
            if isinstance(func, FunctionType):
                module_name = getattr(func, "__globals__", {}).get("__name__", "")
            if (
                hook.REGEX and re.search(hook.HOOK_MODULE_NAME, module_name)
            ) or module_name == hook.HOOK_MODULE_NAME:
                logger.debug(
                    f"Hooking Class: {hook.__name__} into {module_name}|{name}"
                    + (
                        "(Regex is enabled, which might cause unexpected behavior.)"
                        if hook.REGEX
                        else ""
                    )
                )
                target_class = _builtins_build_class_(func, name, *bases, **kwargs)
                hook.hook(target_class)
                _MATCHED_CLASS_HOOKS.add(hook)
                return target_class

    return _builtins_build_class_(func, name, *bases, **kwargs)


def install_class_hooks(hooks: Union[List[BaseHook], BaseHook]):
    _register_hooks(CLASS_HOOKS, hooks)
    builtins.__build_class__ = _custom_build_class_


def is_class_hook_matched(hook: BaseHook) -> bool:
    return hook in _MATCHED_CLASS_HOOKS


def validate_required_class_hooks() -> None:
    unmatched = [
        hook
        for hook in CLASS_HOOKS
        if hook.REQUIRED and hook not in _MATCHED_CLASS_HOOKS
    ]
    if not unmatched:
        return

    hook_names = ", ".join(
        f"{hook.__name__} ({hook.HOOK_MODULE_NAME}.{hook.HOOK_CLASS_NAME})"
        for hook in unmatched
    )
    raise RuntimeError(
        "Required SGLang Simulator hooks did not match imported SGLang classes: "
        f"{hook_names}. The simulator must be adapted to this SGLang revision."
    )


def remove_class_hooks():
    # Clear the registered hooks and reset the build class function.
    # Note: The classes that have been hooked will not be reset.
    CLASS_HOOKS.clear()
    _MATCHED_CLASS_HOOKS.clear()
    builtins.__build_class__ = _builtins_build_class_
