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
"""Shared graph memory pool used by the speculative-draft cuda graph
runners. The new DecodeCudaGraphRunner and PrefillCudaGraphRunner
backends each own their pool internally; this global is retained for the
EAGLE / multi-step draft runners that haven't been folded into the new
backend interface.
"""

from __future__ import annotations

from typing import Any, Optional

_global_graph_memory_pool: Optional[Any] = None


def get_global_graph_memory_pool() -> Optional[Any]:
    return _global_graph_memory_pool


def set_global_graph_memory_pool(val: Any) -> None:
    global _global_graph_memory_pool
    _global_graph_memory_pool = val
