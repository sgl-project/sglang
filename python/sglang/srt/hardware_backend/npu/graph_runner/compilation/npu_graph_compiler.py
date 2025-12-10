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

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.utils.common import get_compiler_backend


class NpuGraphCompiler:
    def __init__(
        self,
        model_runner,
        model: torch.nn.Module,
        compilation_config: CompilationConfig,
    ):
        torch._dynamo.reset()

        if compilation_config is None:
            compilation_config = CompilationConfig(compiler="npugraph")

        backend = get_compiler_backend(
            compilation_config=compilation_config, model_runner=model_runner
        )
        backend.init(model_runner.model_config)

        self.compiled_callable = torch.compile(
            model, fullgraph=True, dynamic=False, backend=backend
        )
