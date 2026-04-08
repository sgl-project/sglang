from typing import List, Optional, Union

from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


class BaseHook:
    HOOK_CLASS_NAME: Optional[str] = None
    HOOK_MODULE_NAME: Optional[str] = None
    REGEX: bool = False

    def __init__(self):
        pass

    @classmethod
    def hook(cls, target) -> None:
        """
        Return a new target or simply modify the target reference.
        """
        raise NotImplementedError


def _register_hooks(HOOKS: List[BaseHook], hooks: Union[List[BaseHook], BaseHook]):
    if isinstance(hooks, list):
        for hook in hooks:
            if not issubclass(hook, BaseHook):
                raise TypeError("The hook should inherit from BaseHook.")
            HOOKS.append(hook)
    elif isinstance(hooks, type) and issubclass(hooks, BaseHook):
        HOOKS.append(hooks)
    else:
        raise TypeError(
            "The type of registered hook should be a list of BaseHook or a single BaseHook."
        )
