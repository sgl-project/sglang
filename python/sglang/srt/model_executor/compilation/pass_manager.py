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

import torch
from sglang.srt.model_executor.compilation.passes.npu_add_rms_norm_fuse import NpuAddRmsNormFuse


class PassManager:
    def __init__(self, graph_module: torch.fx.GraphModule):
        self.graph_module = graph_module
        self.passes = []

    def add(self, pass_):
        self.passes.append(pass_)

    def apply(self):
        for pass_ in self.passes:
            torch.fx.replace_pattern(self.graph_module, NpuAddRmsNormFuse.pattern, NpuAddRmsNormFuse.replacement)
        self.graph_module.recompile()
