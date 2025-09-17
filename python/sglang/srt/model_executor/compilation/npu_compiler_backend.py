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
from torch._dynamo.eval_frame import DisableContext

from sglang.srt.model_executor.compilation.pass_manager import PassManager
from sglang.srt.model_executor.compilation.passes.w8a8_int8.div_fuse import DivFuse
from sglang.srt.model_executor.compilation.passes.w8a8_int8.erase_copy import EraseCopy
from sglang.srt.model_executor.compilation.passes.w8a8_int8.npu_add_rms_norm_quant_fuse import (
    NpuAddRmsNormQuantFuse,
)


class NpuBackend:
    def __call__(self, graph: torch.fx.GraphModule, example_inputs) -> Callable:
        DisableContext.compiled_function_args[DisableContext.batch_size] = (
            example_inputs
        )
        NpuBackend.apply_passes(graph)
        return graph

    def apply_passes(graph_module: torch.fx.GraphModule):
        passManager = PassManager(graph_module)
        passManager.add(NpuAddRmsNormQuantFuse)
        passManager.add(DivFuse)
        passManager.add(EraseCopy)
        passManager.apply()
        graph_module.recompile()
