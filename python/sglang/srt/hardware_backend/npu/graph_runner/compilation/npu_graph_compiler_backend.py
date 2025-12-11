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

from sglang.srt.hardware_backend.npu.graph_runner.compilation.pass_manager import (
    PassManager,
)
from sglang.srt.hardware_backend.npu.graph_runner.compilation.passes.fp16 import (
    SplitQkvRmsnormRopeFuse,
)
from sglang.srt.hardware_backend.npu.graph_runner.compilation.passes.w8a8_int8 import (
    DivFuse,
    EraseCopy,
    NpuAddRmsNormDynamicQuantFuse,
    NpuAddRmsNormQuantFuse,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size


class NpuGraphCompilerBackend:
    def __init__(self, model_runner):
        self.model_type = model_runner.model_config.dtype

    def __call__(self, graph: torch.fx.GraphModule, example_inputs) -> Callable:
        if self.model_type == torch.bfloat16:
            self.apply_passes(graph)
        return graph

    def init(self, config):
        config = config.hf_config

        hidden_size = config.hidden_size

        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads

        head_dim = getattr(config, "head_dim", None)
        self.rms_norm_eps = config.rms_norm_eps

        total_num_heads = num_heads
        attn_tp_size = get_attention_tp_size()

        assert total_num_heads % attn_tp_size == 0
        num_heads = total_num_heads // attn_tp_size
        total_num_kv_heads = num_kv_heads
        num_kv_heads = max(1, total_num_kv_heads // attn_tp_size)

        self.head_dim = head_dim or hidden_size // total_num_heads
        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim

        self.q_shape = (self.head_dim, self.q_size)
        self.k_shape = (self.head_dim, self.kv_size)

    def apply_passes(self, graph_module: torch.fx.GraphModule):
        passManager = PassManager(graph_module)
        passManager.add(
            SplitQkvRmsnormRopeFuse,
            q_size=self.q_size,
            kv_size=self.kv_size,
            head_dim=self.head_dim,
            q_shape=self.q_shape,
            k_shape=self.k_shape,
            variance_epsilon=self.rms_norm_eps,
        )
        passManager.add(NpuAddRmsNormQuantFuse)
        passManager.add(NpuAddRmsNormDynamicQuantFuse)
        passManager.add(DivFuse)
        passManager.add(EraseCopy)
        passManager.apply()
        graph_module.recompile()
