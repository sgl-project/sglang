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

import pathlib
import sys

import torch

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.npu.compilation_context import CompilationContext
from sglang.srt.model_executor.compilation.piecewise_npu_graph_compiler_backend import (
    PiecewiseNpuGraphCompilerBackend,
)


class PiecewiseNpuGraphCompiler:
    def __init__(
        self,
        model_runner,
        model: torch.nn.Module,
        compilation_config: CompilationConfig,
        compilation_context: CompilationContext,
        page_size: int,
    ):
        self.backend = PiecewiseNpuGraphCompilerBackend(
            model_runner, compilation_config, compilation_context, page_size
        )
        self.model = model

        torch._dynamo.reset()
        torch.compiler.allow_in_graph(sys.intern)
        torch.compiler.allow_in_graph(pathlib.Path)

        self.compiled_callable = torch.compile(
            self.model, fullgraph=True, dynamic=False, backend=self.backend
        )
