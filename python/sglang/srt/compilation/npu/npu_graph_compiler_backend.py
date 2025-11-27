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

from typing import Callable

import torch

from sglang.srt.compilation.npu.pass_manager import PassManager
from sglang.srt.compilation.npu.passes.w8a8_int8 import (
    DivFuse,
    EraseCopy,
    NpuAddRmsNormDynamicQuantFuse,
    NpuAddRmsNormQuantFuse,
)


class NpuGraphCompilerBackend:
    def __init__(self, model_type: torch.dtype):
        self.model_type = model_type

    def __call__(self, graph: torch.fx.GraphModule, example_inputs) -> Callable:
        if self.model_type == torch.bfloat16:
            NpuGraphCompilerBackend.apply_passes(graph)
        return graph

    def apply_passes(graph_module: torch.fx.GraphModule):
        passManager = PassManager(graph_module)
        passManager.add(NpuAddRmsNormQuantFuse)
        passManager.add(NpuAddRmsNormDynamicQuantFuse)
        passManager.add(DivFuse)
        passManager.add(EraseCopy)
        passManager.apply()
        graph_module.recompile()
