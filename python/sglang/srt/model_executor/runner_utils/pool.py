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
"""Process-wide CUDA graph memory pool shared across the prefill and
decode graph backends. The two phases never replay concurrently, so
sharing one pool reserves only the larger phase's capture footprint.
"""

from __future__ import annotations

from typing import Any, Optional

from sglang.srt.runtime_context import get_resources


def get_global_graph_memory_pool() -> Optional[Any]:
    return get_resources().graph_memory_pool


def set_global_graph_memory_pool(val: Any) -> None:
    get_resources().graph_memory_pool = val


def get_or_create_global_graph_memory_pool(device_module: Any) -> Any:
    """Return the shared graph memory pool, creating it on first use so
    later backends reuse the same handle."""
    resources = get_resources()
    if resources.graph_memory_pool is None:
        resources.graph_memory_pool = device_module.graph_pool_handle()
    return resources.graph_memory_pool
