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
"""Backward-compatible re-export shim.

The breakable CUDA graph primitives moved to the model-agnostic package
:mod:`sglang.srt.breakable_cuda_graph` so the diffusion runtime
(``sglang.multimodal_gen``) can share them with the LLM runtime. This module
preserves the historical import path.
"""

from sglang.srt.breakable_cuda_graph.breakable_cuda_graph import (  # noqa: F401
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    break_graph,
    eager_on_graph,
    get_current_replay_token,
    get_current_stream,
)

__all__ = [
    "eager_on_graph",
    "BreakableCUDAGraph",
    "BreakableCUDAGraphCapture",
    "break_graph",
    "get_current_stream",
    "get_current_replay_token",
]
