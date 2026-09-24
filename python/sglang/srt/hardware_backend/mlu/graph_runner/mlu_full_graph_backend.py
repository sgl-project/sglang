"""FullMLUGraphBackend — Cambricon MLU full-graph capture (torch.mlu.MLUGraph).

Mirrors FullCudaGraphBackend with MLU-specific differences:
  - Captures via torch.mlu.graph(...) into torch.mlu.MLUGraph.
  - Shares the global graph memory pool with the prefill backend so that
    decode + prefill graphs occupy max(decode, prefill) rather than their sum.
  - No set_graph_pool_id: SymmetricMemoryContext is never triggered on MLU
    (symm-mem all-gather falls back to CNCL).
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


class FullMLUGraphBackend(BaseCudaGraphBackend):
    """One torch.mlu.MLUGraph per shape for Cambricon MLU devices."""

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
    ) -> None:
        self._graphs: Dict[Any, torch.mlu.MLUGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream: Optional[torch.mlu.Stream] = None

    @contextmanager
    def capture_session(self, stream: torch.mlu.Stream):
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
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        graph = torch.mlu.MLUGraph()

        with self._device_module.graph(
            graph, pool=self._pool, stream=self._capture_stream
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
