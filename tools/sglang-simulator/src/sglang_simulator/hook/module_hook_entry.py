import importlib.abc
import importlib.util
import re
import sys
from typing import List, Union

from sglang_simulator.hook.base_hook import BaseHook, _register_hooks
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")

MODULE_HOOKS: List[BaseHook] = []


class HookMetaPathFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        for hook in MODULE_HOOKS:
            if fullname == hook.HOOK_MODULE_NAME:
                return importlib.util.spec_from_loader(fullname, HookLoader())
        return None


class HookLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        for hook in MODULE_HOOKS:
            if (
                hook.REGEX and re.search(hook.HOOK_MODULE_NAME, module.__name__)
            ) or hook.HOOK_MODULE_NAME == module.__name__:
                parts = hook.HOOK_MODULE_NAME.split(".")
                package = ".".join(parts[:-1])
                resource = parts[-1] + ".py"

                try:
                    module_code = (
                        importlib.resources.files(package)
                        .joinpath(resource)
                        .read_text()
                    )
                    exec(module_code, module.__dict__)
                    logger.debug(
                        f"Hooking Module: {hook.__name__} into {module.__name__}"
                        + (
                            "(Regex is enabled, which might cause unexpected behavior.)"
                            if hook.REGEX
                            else ""
                        )
                    )
                    try:
                        hook.hook(module)
                    except Exception as e:
                        logger.warning(
                            f"Failed to hook module [{module.__name__}]. Error: {e}"
                        )
                    return
                except Exception as e:
                    logger.error(f"Failed to load module code: {e}.")
                    return


def install_module_hooks(hooks: Union[List[BaseHook], BaseHook]):
    _register_hooks(MODULE_HOOKS, hooks)
    sys.meta_path.insert(0, HookMetaPathFinder())


def remove_module_hooks():
    # Clear the registered hooks and reset the module finder.
    # Note: The modules that have been hooked will not be reset.
    MODULE_HOOKS.clear()
    if isinstance(sys.meta_path[0], HookMetaPathFinder):
        sys.meta_path.pop(0)
