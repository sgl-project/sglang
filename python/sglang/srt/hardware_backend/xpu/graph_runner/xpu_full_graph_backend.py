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

from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_memory_pool,
    graph_pool_capture_scope,
    graph_pool_replay_scope,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey


def _allocate_output_buffer(output: Any) -> Optional[torch.Tensor]:
    if not torch.is_tensor(output) or output.ndim == 0:
        return None
    return torch.empty_like(output)


def _output_fits_buffer(output: Any, output_buffer: torch.Tensor) -> bool:
    return (
        torch.is_tensor(output)
        and output.ndim == output_buffer.ndim
        and output.shape[1:] == output_buffer.shape[1:]
        and output.shape[0] <= output_buffer.shape[0]
        and output.dtype == output_buffer.dtype
        and output.device == output_buffer.device
    )


def _copy_output_to_buffer(
    output: Any, output_buffer: torch.Tensor
) -> Optional[torch.Tensor]:
    if not _output_fits_buffer(output, output_buffer):
        return None
    shared_output = output_buffer[: output.shape[0]]
    shared_output.copy_(output)
    return shared_output


class FullXPUGraphBackend(BaseCudaGraphBackend):
    """One torch.xpu.XPUGraph per shape for Intel XPU devices."""

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
        reuse_output_buffer: bool = False,
    ) -> None:
        self._graphs: Dict[Any, torch.xpu.XPUGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._cuda_graph_runner = cuda_graph_runner
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream: Optional[torch.xpu.Stream] = None
        self._reuse_output_buffer = reuse_output_buffer
        self._output_buffer: Optional[torch.Tensor] = None

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
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        runner = self._cuda_graph_runner
        profiler = (
            getattr(runner, "_profiler", None)
            if getattr(runner, "enable_profile_cuda_graph", False)
            else None
        )

        warmup_output = None
        for warmup_step in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            output = forward_fn()
            if self._reuse_output_buffer and warmup_step == 1:
                warmup_output = output
            del output
            if profiler is not None:
                profiler.step()
            if post_warmup_hook is not None:
                post_warmup_hook()

        if self._reuse_output_buffer and self._output_buffer is None:
            # Prefill captures the largest shape first and replays one shape at
            # a time, so all graphs can share this eager-tail input buffer.
            self._output_buffer = _allocate_output_buffer(warmup_output)
            self._reuse_output_buffer = self._output_buffer is not None
        del warmup_output

        graph = torch.xpu.XPUGraph()

        with (
            graph_pool_capture_scope(),
            self._device_module.graph(
                xpu_graph=graph, pool=self._pool, stream=self._capture_stream
            ),
        ):
            out = forward_fn()
            if self._reuse_output_buffer:
                output_buffer = self._output_buffer
                assert output_buffer is not None
                shared_output = _copy_output_to_buffer(out, output_buffer)
                self._reuse_output_buffer = shared_output is not None
                if shared_output is not None:
                    out = shared_output

        if profiler is not None:
            profiler.step()

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
        with graph_pool_replay_scope():
            self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._output_buffer = None
        self._pool = None
