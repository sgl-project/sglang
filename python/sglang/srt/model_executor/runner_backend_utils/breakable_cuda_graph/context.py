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
"""Runtime state for the breakable CUDA graph runner."""

from __future__ import annotations

from contextlib import contextmanager

_in_breakable_cuda_graph = False


def is_in_breakable_cuda_graph() -> bool:
    return _in_breakable_cuda_graph


@contextmanager
def enable_breakable_cuda_graph():
    global _in_breakable_cuda_graph
    _in_breakable_cuda_graph = True
    try:
        yield
    finally:
        _in_breakable_cuda_graph = False


BREAKABLE_CUDA_GRAPH_CAPTURE_FAILED_MSG = (
    "Breakable CUDA graph failed.\n"
    "Possible solutions:\n"
    "1. switch to the tc_piecewise prefill CUDA graph backend by "
    "--cuda-graph-backend-prefill=tc_piecewise\n"
    "2. disable the prefill CUDA graph by --cuda-graph-backend-prefill=disabled.\n"
    "3. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
    "4. set --cuda-graph-max-bs-prefill to a smaller value (e.g., 2048)\n"
    "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
)
