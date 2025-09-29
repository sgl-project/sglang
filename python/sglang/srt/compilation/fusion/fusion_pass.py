# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
import time
from abc import abstractmethod
from typing import Any, Callable

import torch
from torch._dynamo.utils import lazy_format_graph_code
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch._inductor.pattern_matcher import (
    PatternMatcherPass,
    fwd_only,
    register_replacement,
)

from sglang.srt.compilation.fusion.fusion_config import FusionConfig
from sglang.srt.compilation.fusion.fusion_context import get_fusion_context
from sglang.srt.compilation.fusion.fusion_utils import hash_source

logger = logging.getLogger(__name__)


class FusionPass(CustomGraphPass):
    def __init__(self, fusion_config: FusionConfig):
        self.fusion_config = fusion_config

        self.pass_name = self.__class__.__name__

        self.patterns = PatternMatcherPass(self.pass_name)

        self.build_pass()

    def __call__(self, graph: torch.fx.graph):
        if self.fusion_config.enable_torch_compile_graph_trace_logs:
            graph_before_str = str(self.dump_graph(graph, f"Before_{self.pass_name}"))

        self.begin()

        count = self.patterns.apply(graph)

        self.end_and_record(count)

        if count > 0 and self.fusion_config.enable_torch_compile_graph_trace_logs:
            graph_after_str = str(self.dump_graph(graph, f"After_{self.pass_name}"))
            logger.debug("%s", graph_before_str)
            logger.debug("%s", graph_after_str)

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

    def dump_graph(self, graph: torch.fx.Graph, stage: str):
        return lazy_format_graph_code(
            stage,
            graph.owning_module,
            include_stride=True,
            include_device=True,
            colored=True,
        )

    def begin(self) -> None:
        self._start_time = time.perf_counter_ns()

    def end_and_record(self, count) -> None:
        self._end_time = time.perf_counter_ns()
        fusion_context = get_fusion_context()
        fusion_context.record_stats(
            self.pass_name, count, self._end_time - self._start_time
        )

    def uuid(self) -> Any:
        return hash_source(self)
