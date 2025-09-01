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

import dataclasses
import datetime
import os
import threading
import importlib
from typing import Callable, List, Any

import torch
from sglang.srt.model_executor.compilation.config import CompilationConfig
from sglang.srt.model_executor.compilation.compilation_context import CompilationContext

from torch._dynamo.eval_frame import DisableContext

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

import logging
logger = logging.getLogger(__name__)


class NpuAddRmsNormFuse:
    def pattern(rms_norm_input, residual, rms_norm_weight, scale, offset, v1, v2, v3):
        output = torch.ops.npu.npu_add_rms_norm(
            rms_norm_input,
            residual,
            rms_norm_weight,
            1e-6)
        out0 = output[0]
        out2 = output[2]
        quantized_output = torch.ops.npu.npu_quantize(
            out0,
            scale,
            offset,
            v1,
            v2,
            v3)
        return quantized_output, out2

    def replacement(rms_norm_input, residual, rms_norm_weight, scale, offset, v1, v2, v3):
        output = torch.ops.npu.npu_add_rms_norm_quant(
            rms_norm_input,
            residual,
            rms_norm_weight,
            1. / scale,
            offset,
            epsilon=1e-6)
        quantized_output = output[0]
        out2 = output[2]
        return quantized_output, out2


class NpuBackend:
    graph: torch.fx.GraphModule

    def __init__(
        self,
        model_runner,
        compilation_config: CompilationConfig,
        compilation_context: CompilationContext,
        page_size: int
    ):
        self.model_runner = model_runner
        self.model_config = model_runner.model.config

        self.compilation_config = compilation_config
        self.page_size = page_size
        self.compilation_context = compilation_context

        self.split_gm = None

        self.piecewise_graphs = None
        self.submod_names_to_compile = None

        self.callables = {}
        self.callables_by_branch = {}


    def __call__(self, graph: torch.fx.GraphModule, example_inputs) -> Callable:
        self.graph = graph
        NpuBackend.apply_passes(self.graph)
        return self.graph

    def apply_passes(graph_module: torch.fx.GraphModule):
        torch.fx.replace_pattern(graph_module, NpuAddRmsNormFuse.pattern, NpuAddRmsNormFuse.replacement)
        graph_module.recompile()
