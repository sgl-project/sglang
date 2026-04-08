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
"""Breakable CUDA graph: piecewise CUDA graph capture without torch.compile.

Each segment between graph breaks gets its own capture_begin(pool)/capture_end()
cycle so that PyTorch's caching allocator can reclaim intermediates between
segments. All segments share the same memory pool for cross-segment reuse.
"""

import logging
from contextvars import ContextVar
from typing import Any, Callable, List, Optional

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "non_graph",
    "BreakableCUDAGraph",
    "BreakableCUDAGraphContext",
]

# Context vars for communication between BreakableCUDAGraphContext and non_graph.
# Set during capture, None otherwise.
_breakable_graph_var: ContextVar[Optional["BreakableCUDAGraph"]] = ContextVar(
    "breakable_graph", default=None
)
_current_segment_var: ContextVar[Optional[torch.cuda.CUDAGraph]] = ContextVar(
    "current_segment", default=None
)
_capture_pool_var: ContextVar[Any] = ContextVar("capture_pool", default=None)


def non_graph(enable: bool):
    """Decorator to mark a function as a graph break point.

    During breakable CUDA graph capture, calls to the decorated function will:
    1. End the current segment's capture (capture_end)
    2. Run the function eagerly
    3. Start a new segment's capture (capture_begin with same pool)

    Outside of capture, the function runs normally with no overhead.
    """

    def decorator(inner: Callable):
        if not enable:
            return inner

        def wrapper(*args, **kwargs):
            bg = _breakable_graph_var.get()
            if bg is None:
                # Not in breakable capture — just run normally
                return inner(*args, **kwargs)

            # End current segment capture — releases intermediates back to pool
            current = _current_segment_var.get()
            current.capture_end()
            bg._segments.append(current)

            logger.debug("Graph break at: %s", inner.__name__)

            # Run the function eagerly (attention, etc.)
            # Allocations here go to normal memory, not the graph pool,
            # because capture_end() exited pool allocation mode.
            output = inner(*args, **kwargs)

            # Create replay closure that re-runs the function and writes
            # into the same output tensor so downstream graph segments
            # (which reference this address) see fresh data.
            def replay_fn():
                new_out = inner(*args, **kwargs)
                if torch.is_tensor(output) and torch.is_tensor(new_out):
                    output.copy_(new_out)
                    return output
                return new_out

            bg._break_fns.append(replay_fn)

            # Start new segment capture — can reuse pool memory freed by
            # the previous segment's capture_end().
            pool = _capture_pool_var.get()
            new_segment = torch.cuda.CUDAGraph()
            new_segment.capture_begin(pool=pool, capture_error_mode="global")
            _current_segment_var.set(new_segment)

            return output

        return wrapper

    return decorator


class BreakableCUDAGraph:
    """A collection of CUDA graph segments with eager breaks between them.

    Each segment is a standard torch.cuda.CUDAGraph captured with the same
    pool, so intermediates are shared across segments via the pool.
    """

    def __init__(self):
        self._segments: List[torch.cuda.CUDAGraph] = []
        self._break_fns: List[Callable] = []

    def replay(self):
        for i, graph in enumerate(self._segments):
            graph.replay()
            if i < len(self._break_fns):
                self._break_fns[i]()


class BreakableCUDAGraphContext:
    """Context manager for capturing a breakable CUDA graph.

    Usage:
        graph = BreakableCUDAGraph()
        with BreakableCUDAGraphContext(graph, pool=pool, stream=stream):
            output = model.forward(...)
        # graph now contains segments, call graph.replay() to re-run
    """

    def __init__(
        self,
        cuda_graph: BreakableCUDAGraph,
        pool=None,
        stream: Optional[torch.cuda.Stream] = None,
        capture_error_mode: str = "global",
    ):
        assert isinstance(cuda_graph, BreakableCUDAGraph)
        self.graph = cuda_graph
        self.pool = pool
        self.stream = stream
        self.capture_error_mode = capture_error_mode

    def __enter__(self):
        # Save and switch to capture stream
        self._old_stream = torch.cuda.current_stream()
        if self.stream is not None:
            torch.cuda.set_stream(self.stream)

        # Start first segment
        first_segment = torch.cuda.CUDAGraph()
        first_segment.capture_begin(
            pool=self.pool, capture_error_mode=self.capture_error_mode
        )

        # Set context vars so non_graph can find us
        self._bg_token = _breakable_graph_var.set(self.graph)
        self._seg_token = _current_segment_var.set(first_segment)
        self._pool_token = _capture_pool_var.set(self.pool)

        return self.graph

    def __exit__(self, exc_type, exc_val, exc_tb):
        # End last segment
        last_segment = _current_segment_var.get()
        if last_segment is not None:
            last_segment.capture_end()
            self.graph._segments.append(last_segment)

        # Reset context vars
        _breakable_graph_var.reset(self._bg_token)
        _current_segment_var.reset(self._seg_token)
        _capture_pool_var.reset(self._pool_token)

        # Restore stream
        torch.cuda.set_stream(self._old_stream)

        return False
