# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/inductor_pass.py

import hashlib
import inspect
import json
import logging
import time
import types
from contextlib import contextmanager
from typing import Any, Callable, Optional, Union

import torch
from torch import fx
from torch._dynamo.utils import lazy_format_graph_code
from torch._inductor.custom_graph_pass import CustomGraphPass

logger = logging.getLogger(__name__)

_pass_context = None


class PassContext:

    def __init__(self, runtime_shape: Optional[int]):
        self.runtime_shape = runtime_shape


def get_pass_context() -> PassContext:
    """Get the current pass context."""
    assert _pass_context is not None
    return _pass_context


@contextmanager
def pass_context(runtime_shape: Optional[int]):
    """A context manager that stores the current pass context,
    usually it is a list of sizes to specialize.
    """
    global _pass_context
    prev_context = _pass_context
    _pass_context = PassContext(runtime_shape)
    try:
        yield
    finally:
        _pass_context = prev_context


class InductorPass(CustomGraphPass):
    """
    A custom graph pass that uses a hash of its source as the UUID.
    This is defined as a convenience and should work in most cases.
    """

    def uuid(self) -> Any:
        """
        Provide a unique identifier for the pass, used in Inductor code cache.
        This should depend on the pass implementation, so that changes to the
        pass result in recompilation.
        By default, the object source is hashed.
        """
        return InductorPass.hash_source(self)

    @staticmethod
    def hash_source(*srcs: Union[str, Any]):
        """
        Utility method to hash the sources of functions or objects.
        :param srcs: strings or objects to add to the hash.
        Objects and functions have their source inspected.
        :return:
        """
        hasher = hashlib.sha256()
        for src in srcs:
            if isinstance(src, str):
                src_str = src
            elif isinstance(src, types.FunctionType):
                src_str = inspect.getsource(src)
            else:
                src_str = inspect.getsource(src.__class__)
            hasher.update(src_str.encode("utf-8"))
        return hasher.hexdigest()

    @staticmethod
    def hash_dict(dict_: dict[Any, Any]):
        """
        Utility method to hash a dictionary, can alternatively be used for uuid.
        :return: A sha256 hash of the json rep of the dictionary.
        """
        encoded = json.dumps(dict_, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def is_applicable_for_shape(self, shape: Optional[int]):
        return True


class CallableInductorPass(InductorPass):
    """
    This class is a wrapper for a callable that automatically provides an
    implementation of the UUID.
    """

    def __init__(
        self, callable: Callable[[fx.Graph], None], uuid: Optional[Any] = None
    ):
        self.callable = callable
        self._uuid = self.hash_source(callable) if uuid is None else uuid

    def __call__(self, graph: torch.fx.Graph):
        self.callable(graph)

    def uuid(self) -> Any:
        return self._uuid


class SGLangInductorPass(InductorPass):

    def __init__(
        self,
    ):
        self.pass_name = self.__class__.__name__

    def dump_graph(self, graph: torch.fx.Graph, stage: str):
        lazy_format_graph_code(stage, graph.owning_module)

    def begin(self):
        self._start_time = time.perf_counter_ns()

    def end_and_log(self):
        self._end_time = time.perf_counter_ns()
        duration_ms = float(self._end_time - self._start_time) / 1.0e6
        logger.debug("%s completed in %.1f ms", self.pass_name, duration_ms)


class PrinterInductorPass(SGLangInductorPass):

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __call__(self, graph: torch.fx.Graph):
        self.dump_graph(graph, self.name)
