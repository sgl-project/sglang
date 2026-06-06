# Copyright 2023-2026 SGLang Team
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
"""Model-agnostic breakable CUDA graph (BCG) primitives.

Shared by the LLM runtime (``sglang.srt.model_executor``) and the diffusion
runtime (``sglang.multimodal_gen``). Capture a forward region as a sequence of
``torch.cuda.CUDAGraph`` segments separated by eager break points inserted via
:func:`eager_on_graph`-decorated callables.
"""

from sglang.srt.breakable_cuda_graph.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    break_graph,
    eager_on_graph,
)
from sglang.srt.breakable_cuda_graph.context import (
    enable_breakable_cuda_graph,
    is_in_breakable_cuda_graph,
)

__all__ = [
    "BreakableCUDAGraph",
    "BreakableCUDAGraphCapture",
    "break_graph",
    "eager_on_graph",
    "enable_breakable_cuda_graph",
    "is_in_breakable_cuda_graph",
]
