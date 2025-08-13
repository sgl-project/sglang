# Copyright 2023-2024 SGLang Team
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
"""Run the model with cuda graph and torch.compile."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.model_executor.graph_runner import GraphRunner

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class CudaGraphRunner(GraphRunner):
    """A CudaGraphRunner runs the forward pass of a model with cuda graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        # Parse args
        super().__init__(model_runner)
