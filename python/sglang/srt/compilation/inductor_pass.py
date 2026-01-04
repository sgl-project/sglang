# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/inductor_pass.py

import hashlib
import inspect
import json
import logging
import time
import types
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Callable, Optional, Union

import torch
from torch import fx
from torch._dynamo.utils import lazy_format_graph_code
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch._inductor.pattern_matcher import (
    PatternMatcherPass,
    fwd_only,
    register_replacement,
)

from sglang.srt.compilation.pass_config import PassConfig

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


class SGLangPatternMatcherInductorPass(SGLangInductorPass):
    def __init__(self, pass_config: PassConfig):
        self.pass_config = pass_config
        self.pass_name = self.__class__.__name__
        self.patterns = PatternMatcherPass(self.pass_name)
        self.build_pass()

    def __call__(self, graph: torch.fx.graph):
        if self.pass_config.enable_torch_compile_graph_trace_logs:
            graph_before_str = str(self.dump_graph(graph, f"Before_{self.pass_name}"))

        self.begin()
        count = self.patterns.apply(graph)
        self.end_and_log(count)

        if count > 0 and self.pass_config.enable_torch_compile_graph_trace_logs:
            graph_after_str = str(self.dump_graph(graph, f"After_{self.pass_name}"))
            logger.info("%s", graph_before_str)
            logger.info("%s", graph_after_str)

    @abstractmethod
    def build_pass(self) -> None:
        pass

    def register_replacement_pattern(
        self, pattern: Callable, replacement: Callable, example_inputs: Any, **kwargs
    ) -> None:
        register_replacement(
            search_fn=pattern,
            replace_fn=replacement,
            example_inputs=example_inputs,
            trace_fn=fwd_only,
            pass_dicts=self.patterns,
            **kwargs,
        )

    def end_and_log(self, count: int):
        self._end_time = time.perf_counter_ns()
        duration_ms = float(self._end_time - self._start_time) / 1.0e6
        logger.debug(
            "%s completed in %.1f ms, matched %s times",
            self.pass_name,
            duration_ms,
            count,
        )
