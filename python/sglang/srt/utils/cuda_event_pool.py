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
"""Reusable CUDA events for hot-path cross-stream waits.

``dst.wait_stream(src)`` creates a CUDA event on every call, and event creation
can stall on the CUDA context lock. :func:`wait_stream` records a prewarmed
no-timing event instead, like TensorRT-LLM KVCM2's ``CachedCudaEvent``. The
event goes back to the pool at once: ``cudaStreamWaitEvent`` captures the event
state when it is enqueued, so a later record does not affect an earlier wait.
"""

from collections import deque

import torch

# An event is held only while one call enqueues its record and wait.
_POOL_SIZE = 64
_pools: dict[int, deque[torch.cuda.Event]] = {}


def prewarm_cuda_event_pool(device: int) -> None:
    """Create the event pool that enables :func:`wait_stream` on ``device``."""
    if device in _pools:
        return
    with torch.cuda.device(device):
        stream = torch.cuda.Stream()
        events = [torch.cuda.Event() for _ in range(_POOL_SIZE)]
        # torch.cuda.Event creates its CUDA event lazily, on the first record.
        for event in events:
            event.record(stream)
        stream.synchronize()
    _pools[device] = deque(events)


def wait_stream(dst: torch.cuda.Stream, src: torch.cuda.Stream) -> None:
    """Same as ``dst.wait_stream(src)``, reusing a prewarmed event if possible.

    Uses PyTorch's path when ``src``'s device has no pool, the pool is empty,
    or the current stream is capturing a CUDA graph.
    """
    # With no pool at all, streams of any backend pass straight through.
    pool = _pools.get(src.device_index) if _pools else None
    if pool is None or torch.cuda.is_current_stream_capturing():
        dst.wait_stream(src)
        return
    # popleft() and append() are atomic under the GIL, so no lock is needed.
    try:
        event = pool.popleft()
    except IndexError:
        dst.wait_stream(src)
        return
    event.record(src)
    event.wait(dst)
    pool.append(event)
