import functools
import logging
import operator
import time
from pathlib import Path
from typing import ClassVar, Optional

import regex as re
import torch
from torch._dynamo.utils import lazy_format_graph_code
from torch._inductor.pattern_matcher import PatternMatcherPass, PatternPrettyPrinter

from sglang.srt.configs.compilation_config import CompilationConfig
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.model_config import ModelConfig

from .inductor_pass import InductorPass

logger = logging.getLogger(__name__)


class SglangInductorPass(InductorPass):
    """
    An inductor pass with access to SGLang PassConfig.
    It provides timing, logging, and dumping utilities.
    """

    dump_prefix: ClassVar[Optional[int]] = None

    def __init__(
        self,
        compilation_config: CompilationConfig,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ):
        self.pass_config = compilation_config.pass_config
        self.model_dtype = model_config.dtype if model_config else None
        self.device = device_config.device if device_config else None
        self.pass_name = self.__class__.__name__

    @staticmethod
    def time_and_log(call_fn):

        @functools.wraps(call_fn)
        def wrapped(self: SglangInductorPass, graph: torch.fx.Graph):
            self.begin()
            self.dump_graph(graph, "before")
            call_fn(self, graph)
            self.dump_graph(graph, "after")
            self.end_and_log()

        return wrapped

    def dump_graph(self, graph: torch.fx.Graph, stage: str):
        i = SglangInductorPass.dump_prefix
        i_str = "" if i is None else f".{i}"
        lazy_format_graph_code(
            f"post_grad{i_str}.{self.pass_name}.{stage}", graph.owning_module
        )

    def begin(self):
        self._start_time = time.perf_counter_ns()

    def end_and_log(self):
        self._end_time = time.perf_counter_ns()
        duration_ms = float(self._end_time - self._start_time) / 1.0e6
        logger.debug("%s completed in %.1f ms", self.pass_name, duration_ms)


class SglangPatternMatcherPass(SglangInductorPass):
    """
    A SglangInductorPass that uses the Inductor pattern matcher.
    Its main use is providing the dump_patterns utility that dumps the
    Inductor pattern matcher patterns into a file, which greatly aids debugging.

    """

    matched_count: int = 0
    """The number of matched patterns in the pass."""

    _OP_OVERLOAD_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"<OpOverload\(op='([^']*)', overload='([^']*)'\)>"
    )

    def _replace_op_overloads(self, string: str) -> str:
        """Replace <OpOverload(..., ...)> with nicer formulations"""
        return self._OP_OVERLOAD_PATTERN.sub(
            lambda m: f"torch.ops.{m.group(1)}.{m.group(2)}",
            string,
        )

    def dump_patterns(
        self,
        compilation_config: CompilationConfig,
        model_config: ModelConfig,
        pm_pass: PatternMatcherPass,
    ):
        """
        TODO(yuan-luo): use pattern object to manually produce pattern graph
        """
