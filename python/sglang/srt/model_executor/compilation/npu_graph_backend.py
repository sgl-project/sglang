# Copyright 2025 SGLang Team
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

from typing import Any

import torch
import torch_npu

from sglang.srt.compilation.npu.compilation_context import CompilationContext


class NPUGraphBackend:
    def __init__(
        self,
        model_runner,
        graph: torch.fx.GraphModule,
        compilation_context: CompilationContext,
    ):
        self.model_runner = model_runner
        self.graph = graph
        self.compilation_context = compilation_context

        self.captured = False
        self.output = None
        self.npu_graph = None

    def __call__(self, *args) -> Any:
        if not self.captured:
            if not self.compilation_context.stream:
                self.compilation_context.stream = torch_npu.npu.Stream()

            torch.cuda.synchronize()

            self.npu_graph = torch_npu.npu.NPUGraph()
            with torch.npu.graph(
                self.npu_graph,
                stream=self.compilation_context.stream,
                pool=self.compilation_context.graph_memory_pool,
            ):

                self.output = self.graph.forward(*args)

            if not self.compilation_context.graph_memory_pool:
                self.compilation_context.graph_memory_pool = self.npu_graph.pool()

            self.npu_graph.replay()
            self.captured = True
        else:
            self.npu_graph.replay()

        return self.output
