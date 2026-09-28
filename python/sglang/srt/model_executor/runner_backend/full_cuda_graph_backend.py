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
"""FullCudaGraphBackend — captures the entire model forward as one
torch.cuda.CUDAGraph per shape.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_utils.pool import (
    GraphPoolPrecarve,
    get_or_create_global_graph_memory_pool,
    graph_pool_capture_scope,
    graph_pool_replay_scope,
)
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

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


class FullCudaGraphBackend(BaseCudaGraphBackend):
    """One torch.cuda.CUDAGraph per shape; attention metadata is
    captured inside the graph. Memory-saver-aware.
    """

    def __init__(
        self,
        cuda_graph_runner: Optional[BaseCudaGraphRunner] = None,
        *,
        enable_memory_saver: bool = False,
        reuse_output_buffer: bool = False,
        device_module: Optional[Any] = None,
        pool: Optional[Any] = None,
        warmup_steps: int = 2,
        warmup_barrier: Optional[Callable[[], None]] = None,
    ) -> None:
        # Standalone executors supply device/pool directly; native runners keep
        # their existing shared pool, two warmups, and TP barrier by default.
        if type(warmup_steps) is not int or warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")
        if cuda_graph_runner is not None:
            if device_module is not None or warmup_barrier is not None:
                raise ValueError("runner and explicit execution context are exclusive")
            device_module = cuda_graph_runner.device_module
            warmup_barrier = cuda_graph_runner.model_runner.tp_group.barrier
        if device_module is None:
            raise ValueError("a runner or device_module is required")
        if reuse_output_buffer and warmup_steps == 0:
            raise ValueError("output buffer reuse requires a warmup output")
        self._graphs: Dict[Any, torch.cuda.CUDAGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._capture_inputs: Dict[Any, Any] = {}
        self._pool = pool
        self._cuda_graph_runner = cuda_graph_runner
        self._device_module = device_module
        self._warmup_steps = warmup_steps
        self._warmup_barrier = warmup_barrier
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._precarve = GraphPoolPrecarve()
        self._reuse_output_buffer = reuse_output_buffer
        self._output_buffer: Optional[torch.Tensor] = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream, *, pool: Optional[Any] = None):
        previous_pool = self._pool
        if pool is not None:
            self._pool = pool
        elif self._pool is None:
            self._pool = get_or_create_global_graph_memory_pool(self._device_module)
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None
            if pool is not None:
                self._pool = previous_pool

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        # When per-bs capture traces are enabled (--enable-profile-cuda-graph +
        # SGLANG_GRAPH_BATCH_CAPTURE), the runner created a scheduled
        # torch profiler (wait=2, active=1) and exposed it as _profiler. We step()
        # past the two warmup runs so only the capture run is recorded, and each
        # batch size produces its own trace via the profiler's on_trace_ready.
        # With --enable-profile-cuda-graph alone the runner leaves _profiler None
        # (its unscheduled profiler records the whole capture in one pass), so no
        # stepping happens here.
        runner = self._cuda_graph_runner
        profiler = (
            getattr(runner, "_profiler", None)
            if getattr(runner, "enable_profile_cuda_graph", False)
            else None
        )

        # Warmups load kernels and pay one-time setup before capture. A caller that
        # coordinates its own warmup may capture without executing extra forwards.
        # post_warmup_hook lets the attention backend reset state that warmup mutated.
        warmup_output = None
        for warmup_step in range(self._warmup_steps):
            self._device_module.synchronize()
            if self._warmup_barrier is not None:
                self._warmup_barrier()
            with self._precarve.measure():
                output = forward_fn()
            if self._reuse_output_buffer and warmup_step == self._warmup_steps - 1:
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

        graph = torch.cuda.CUDAGraph()

        graph_ctx: Callable[..., AbstractContextManager]
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            graph_ctx = partial(
                self._memory_saver_adapter.cuda_graph,
                tag=GPU_MEMORY_TYPE_CUDA_GRAPH,
                # replays read capture-time state from the pool, so a pause must back it up
                enable_cpu_backup=True,
            )
        else:
            graph_ctx = self._device_module.graph

        with (
            graph_pool_capture_scope(),
            graph_ctx(cuda_graph=graph, pool=self._pool, stream=self._capture_stream),
        ):
            self._precarve.mint()
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
        self._capture_inputs[shape_key] = capture_inputs

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

    def release_shape(self, shape_key: ShapeKey) -> None:
        """Drop one graph before releasing its captured inputs and outputs."""
        graph = self._graphs.pop(shape_key, None)
        if graph is not None:
            graph.reset()
        self._outputs.pop(shape_key, None)
        self._capture_inputs.pop(shape_key, None)

    def cleanup(self) -> None:
        for graph in self._graphs.values():
            graph.reset()
        self._graphs.clear()
        self._outputs.clear()
        self._capture_inputs.clear()
        self._output_buffer = None
        self._pool = None
