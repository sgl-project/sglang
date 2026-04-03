import builtins
import re
from types import FunctionType
from typing import List, Union

from sglang_simulator.hook.base_hook import BaseHook, _register_hooks
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


CLASS_HOOKS: List[BaseHook] = []

_builtins_build_class_ = builtins.__build_class__


def _custom_build_class_(func, name: str, *bases, **kwargs):
    for hook in CLASS_HOOKS:
        if (
            hook.REGEX and re.search(hook.HOOK_CLASS_NAME, name)
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
                try:
                    hook.hook(target_class)
                except Exception as e:
                    logger.warning(f"Failed to hook class [{name}]. Error: {e}")
                return target_class

    return _builtins_build_class_(func, name, *bases, **kwargs)


def install_class_hooks(hooks: Union[List[BaseHook], BaseHook]):
    _register_hooks(CLASS_HOOKS, hooks)
    builtins.__build_class__ = _custom_build_class_


def remove_class_hooks():
    # Clear the registered hooks and reset the build class function.
    # Note: The classes that have been hooked will not be reset.
    CLASS_HOOKS.clear()
    builtins.__build_class__ = _builtins_build_class_
