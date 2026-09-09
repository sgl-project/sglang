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

from contextlib import AbstractContextManager, contextmanager, nullcontext
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
    get_or_create_global_graph_memory_pool,
)
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey


class FullCudaGraphBackend(BaseCudaGraphBackend):
    """One torch.cuda.CUDAGraph per shape; attention metadata is
    captured inside the graph. Memory-saver-aware.
    """

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
        nccl_ep_capacity: Optional[int] = None,
    ) -> None:
        self._nccl_ep_resources = None
        if nccl_ep_capacity is not None:
            if enable_memory_saver:
                raise ValueError("NCCL EP Graph does not support memory saver")
            from sglang.srt.layers.moe.token_dispatcher.nccl_ep_graph import (
                NcclEpGraphResources,
            )

            self._nccl_ep_resources = NcclEpGraphResources(
                nccl_ep_capacity, shutdown=self.cleanup
            )
        self._graphs: Dict[Any, torch.cuda.CUDAGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        context = (
            self._nccl_ep_resources.capture_session()
            if self._nccl_ep_resources is not None
            else nullcontext()
        )
        with context:
            try:
                if self._pool is None:
                    self._pool = (
                        self._device_module.graph_pool_handle()
                        if self._nccl_ep_resources is not None
                        else get_or_create_global_graph_memory_pool(self._device_module)
                    )
                set_graph_pool_id(self._pool)
                self._capture_stream = stream
                yield
            except BaseException:
                if self._nccl_ep_resources is not None:
                    self.cleanup()
                raise
            finally:
                self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        if self._nccl_ep_resources is not None:
            self._nccl_ep_resources.require_session("capture")
            if shape_key in self._graphs:
                raise ValueError(
                    "NCCL EP bucket already captured; start a new generation"
                )
        # Two warmups so kernels are loaded and one-time setup is paid before capture.
        # post_warmup_hook lets the attention backend reset state that warmup mutated.
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        graph = torch.cuda.CUDAGraph()

        graph_ctx: Callable[..., AbstractContextManager]
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            graph_ctx = partial(
                self._memory_saver_adapter.cuda_graph,
                tag=GPU_MEMORY_TYPE_CUDA_GRAPH,
            )
        else:
            graph_ctx = self._device_module.graph

        try:
            with graph_ctx(
                cuda_graph=graph, pool=self._pool, stream=self._capture_stream
            ):
                out = forward_fn()
        except BaseException:
            if self._nccl_ep_resources is not None:
                # This executable has not entered _graphs yet. A traceback can
                # retain it after failure, so reset it before the session closes
                # the persistent native resources it may have captured.
                self._device_module.synchronize()
                graph.reset()
            raise

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        context = (
            self._nccl_ep_resources.submission_session("replay")
            if self._nccl_ep_resources is not None
            else nullcontext()
        )
        with context:
            yield

    def require_replay_session(self):
        """Protect static-input writes as well as the graph launch itself."""
        if self._nccl_ep_resources is not None:
            self._nccl_ep_resources.require_session("replay")

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        self.require_replay_session()
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        context = (
            self._nccl_ep_resources.submission_session("cleanup")
            if self._nccl_ep_resources is not None
            else nullcontext()
        )
        with context:
            if self._nccl_ep_resources is not None:
                self._device_module.synchronize()
                for graph in self._graphs.values():
                    graph.reset()
            self._graphs.clear()
            self._outputs.clear()
            self._pool = None
            if self._nccl_ep_resources is not None:
                self._nccl_ep_resources.close()
