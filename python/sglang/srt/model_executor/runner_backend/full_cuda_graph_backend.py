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


# Optional post-replay hook for the pinned-window paged-experts fallback (replay-twice). After a captured
# replay, a registered hook may stage data the graph could not read in-graph — for paged experts, the
# window-missing (cold) experts that decide_bounded deferred — directly into the GPU slots out-of-graph, then
# return True to request another replay of the SAME graph (the residency maps it reads are fixed-address, so
# the next replay sees the staged experts resident). Bounded by _POST_REPLAY_MAX_TRIES. A no-op with zero
# overhead unless a hook is registered, so the default monolithic captured path is byte-for-byte unchanged.
_post_replay_hook: Optional[Callable[[], bool]] = None
_POST_REPLAY_MAX_TRIES = 8


def set_post_replay_hook(fn: Optional[Callable[[], bool]]) -> None:
    """Register (or clear, with ``None``) the post-replay hook. Idempotent; the paged-experts pager calls
    this once when the first windowed layer sets up its captured residency state."""
    global _post_replay_hook
    _post_replay_hook = fn


class FullCudaGraphBackend(BaseCudaGraphBackend):
    """One torch.cuda.CUDAGraph per shape; attention metadata is
    captured inside the graph. Memory-saver-aware.
    """

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
    ) -> None:
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
        if self._pool is None:
            self._pool = get_or_create_global_graph_memory_pool(self._device_module)
        set_graph_pool_id(self._pool)
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

        with graph_ctx(cuda_graph=graph, pool=self._pool, stream=self._capture_stream):
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
        if _post_replay_hook is not None:
            tries = 0
            while _post_replay_hook() and tries < _POST_REPLAY_MAX_TRIES:
                self._graphs[shape_key].replay()
                tries += 1
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None
