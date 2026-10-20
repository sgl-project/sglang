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
"""Liveness tracking for captured CUDA graphs.

``torch.cuda.empty_cache()`` releases the caching allocator's inactive
segments wholesale. While CUDA graphs are captured, some pooled segments
are memory the captured graphs still reference; releasing them makes the
next graph replay fail with ``cudaErrorIllegalAddress`` (cuGraphLaunch
700). Serving paths that empty the allocator cache must check
:func:`have_live_cuda_graphs` first.

Graph backends report their lifecycle through
:func:`note_cuda_graphs_captured` (first captured graph) and
:func:`note_cuda_graphs_released` (cleanup). The counter is conservative:
a backend that drops its graphs without reporting a release keeps the
liveness signal True.
"""

from threading import Lock

_live_graph_backends = 0
_lock = Lock()


def note_cuda_graphs_captured() -> None:
    """Record that a graph backend now holds at least one captured graph."""
    global _live_graph_backends
    with _lock:
        _live_graph_backends += 1


def note_cuda_graphs_released() -> None:
    """Record that a graph backend released every captured graph."""
    global _live_graph_backends
    with _lock:
        if _live_graph_backends > 0:
            _live_graph_backends -= 1


def have_live_cuda_graphs() -> bool:
    """True while any graph backend in this process holds captured graphs."""
    with _lock:
        return _live_graph_backends > 0
