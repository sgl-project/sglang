"""Lightweight lazy import utilities.

This module is intentionally minimal to avoid importing heavy dependencies
when only the LazyImport class is needed.
"""

import importlib


class LazyImport:
    """Lazy import to make `import sglang` run faster.

    This class delays the actual import of a module until it's first accessed,
    which helps reduce the initial import time of the sglang package.

    Parameters
    ----------
    module_name : str
        The full module path to import (e.g., "sglang.lang.backend.openai")
    class_name : str
        The name of the class/object to import from the module

    Examples
    --------
    >>> OpenAI = LazyImport("sglang.lang.backend.openai", "OpenAI")
    >>> # OpenAI module is not loaded yet
    >>> client = OpenAI()  # Now the module is loaded
    """

    def __init__(self, module_name: str, class_name: str):
        self.module_name = module_name
        self.class_name = class_name
        self._module = None

    def _load(self):
        if self._module is None:
            module = importlib.import_module(self.module_name)
            self._module = getattr(module, self.class_name)
        return self._module

    def __getattr__(self, name: str):
        module = self._load()
        return getattr(module, name)

    def __call__(self, *args, **kwargs):
        module = self._load()
        return module(*args, **kwargs)

    def __repr__(self):
        if self._module is None:
            return f"<LazyImport {self.module_name}.{self.class_name} (not loaded)>"
        return repr(self._module)
