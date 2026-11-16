"""Injected into every process of a serving command through PYTHONPATH.

Chains to any sitecustomize that was already on the path, then arms the
lifetime tracer: torch is wrapped the moment it is imported and the SGLang
graph modules the moment they are imported, so no CUDA work happens here and
unrelated processes only leave an ``installed`` marker.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _chain_previous_sitecustomize() -> None:
    path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", path)
    if spec is None or spec.origin == __file__:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


class _PostImportHook(importlib.abc.MetaPathFinder):
    """Runs the tracer's patch for torch and the SGLang graph modules right
    after each of them is executed; every other import is untouched."""

    def __init__(self, names):
        self._names = set(names)
        self._active = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self._names or fullname in self._active:
            return None
        self._active.add(fullname)
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            self._active.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _WrappedLoader(spec.loader, fullname)
        return spec


class _WrappedLoader(importlib.abc.Loader):
    def __init__(self, loader, fullname):
        self._loader = loader
        self._fullname = fullname

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        from sglang.test.cuda_graph_lifetime import tracer

        if self._fullname == "torch":
            tracer.patch_torch()
        else:
            tracer.patch_sglang_module(self._fullname, module)


_chain_previous_sitecustomize()
try:
    from sglang.test.cuda_graph_lifetime import tracer

    sys.meta_path.insert(0, _PostImportHook(["torch", *tracer._SGLANG_MODULES]))
    tracer.install()
except Exception as exc:  # never break the serving process at import time
    sys.stderr.write(f"[cuda_graph_lifetime] tracer not armed: {exc!r}\n")
