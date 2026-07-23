"""FullXPUGraphBackend — Intel XPU full-graph capture (torch.xpu.XPUGraph).

Mirrors FullCudaGraphBackend with XPU-specific differences:
  - Captures via torch.xpu.graph(xpu_graph=...) into torch.xpu.XPUGraph.
  - Shares the global graph memory pool with the prefill backend so that
    decode + prefill graphs occupy max(decode, prefill) rather than their sum.
  - No set_graph_pool_id: SymmetricMemoryContext is never triggered on XPU
    (oneCCL has no ncclMemAlloc equivalent; enable_symm_mem defaults False).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_memory_pool,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )


class FullXPUGraphBackend(BaseCudaGraphBackend):
    """One torch.xpu.XPUGraph per shape for Intel XPU devices."""

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
    ) -> None:
        self._graphs: Dict[Any, torch.xpu.XPUGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream: Optional[torch.xpu.Stream] = None

    @contextmanager
    def capture_session(self, stream: torch.xpu.Stream):
        if self._pool is None:
            self._pool = get_or_create_global_graph_memory_pool(self._device_module)
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        dummies: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        graph = torch.xpu.XPUGraph()

        with self._device_module.graph(
            xpu_graph=graph, pool=self._pool, stream=self._capture_stream
        ):
            out = forward_fn()

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None
