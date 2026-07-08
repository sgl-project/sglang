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
"""Backward-compatible re-export shim for the moved BCG context helpers.

See :mod:`sglang.srt.breakable_cuda_graph.context`.
"""

from sglang.srt.breakable_cuda_graph.context import (  # noqa: F401
    BCG_FAILURE_HINT,
    enable_breakable_cuda_graph,
    is_in_breakable_cuda_graph,
)

__all__ = [
    "BCG_FAILURE_HINT",
    "enable_breakable_cuda_graph",
    "is_in_breakable_cuda_graph",
]
