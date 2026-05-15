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
"""Runtime state for the breakable CUDA graph (BCG) runner.

Kept intentionally separate from ``compilation/piecewise_context_manager.py``:
BCG no longer inherits from the torch.compile-based PCG path, so its
capture/replay lifecycle is managed on its own.
"""

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
