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

from sglang.srt.model_executor.compilation.npu_compiler_backend import NpuBackend


class NpuGraphCompiler:
    def __init__(self, model: torch.nn.Module):
        torch._dynamo.reset()

        self.backend = NpuBackend()
        self.compiled_callable = torch.compile(
            model, fullgraph=True, dynamic=False, backend=self.backend
        )
