import functools
import logging
import time
from dataclasses import dataclass
from typing import ClassVar

import regex as re
import torch
from torch._dynamo.utils import lazy_format_graph_code

from sglang.srt.compilation.inductor_pass import InductorPass
from sglang.srt.compilation.sglang_config import SGLangConfig

logger = logging.getLogger(__name__)


@dataclass
class InductorCompilationConfig:
    splitting_ops: list[str] | None = None
    use_inductor_graph_partition: bool = False


class SGLangInductorPass(InductorPass):
    """
    An inductor pass with access to SGLang PassConfig.
    It provides timing, logging, and dumping utilities.
    """

    dump_prefix: ClassVar[int | None] = None
    """Keep track of pass index for debug dump ordering."""

    def __init__(self, config: SGLangConfig):
        # Get only the necessary CompilationConfig for the inductor pass, since
        # full `CompilationConfig` contains pointer to model which is unsafe.
        self.compilation_config = InductorCompilationConfig(
            splitting_ops=config.compilation_config.splitting_ops,
            use_inductor_graph_partition=config.compilation_config.use_inductor_graph_partition,
        )
        self.pass_config = config.compilation_config.pass_config
        self.model_dtype = config.model_config.dtype if config.model_config else None
        self.device = config.device_config.device if config.device_config else None
        self.pass_name = self.__class__.__name__

    @staticmethod
    def time_and_log(call_fn):
        @functools.wraps(call_fn)
        def wrapped(self: SGLangInductorPass, graph: torch.fx.Graph):
            self.begin()
            self.dump_graph(graph, "before")
            call_fn(self, graph)
            self.dump_graph(graph, "after")
            self.end_and_log()

        return wrapped

    def dump_graph(self, graph: torch.fx.Graph, stage: str):
        i = SGLangInductorPass.dump_prefix
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


class SGLangPatternMatcherPass(SGLangInductorPass):
    """
    A SGLangInductorPass that uses the Inductor pattern matcher.
    Its main use is providing the dump_patterns utility that dumps the
    Inductor pattern matcher patterns into a file, which greatly aids debugging.

    TODO(yuan-luo): move more utilities to this pass.
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
