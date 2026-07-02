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
"""BreakableCudaGraphBackend — segment-captured graphs with eager break
markers (eager_on_graph decorators on attention / mamba layers).
No torch.compile.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin import (
    DedupedCudaGraphMixin,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
    enable_breakable_cuda_graph,
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


# Per-step post-replay hook (the breakable analog of the full backend's). Fires once after replay() returns
# — i.e. after the full forward + all eager breaks complete and before the next step — so it is a safe
# boundary to mutate state the captured gathers read (e.g. the paged-experts freq-window re-pin). Unlike the
# full backend's hook this does NOT loop/re-replay: BCG pages cold experts inline at the break, so there is
# nothing to converge; the hook is purely a per-step boundary callback. None = no hook installed.
_post_replay_hook: Optional[Callable[[], None]] = None


def set_post_replay_hook(fn: Optional[Callable[[], None]]) -> None:
    """Register a once-per-step callback fired after each decode replay (between steps)."""
    global _post_replay_hook
    _post_replay_hook = fn


class BreakableCudaGraphBackend(DedupedCudaGraphMixin, BaseCudaGraphBackend):
    """Segmented capture: graphs break at attention / mamba boundaries;
    attention metadata is recomputed at replay outside captured segments.
    """

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
        debug_eager: bool = False,
    ) -> None:
        self._model_runner = cuda_graph_runner.model_runner
        self._graphs: Dict[Any, BreakableCUDAGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._debug_eager = debug_eager
        self._shared_output_buffer: Optional[Any] = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            raise NotImplementedError(
                "Breakable CUDA graph is not compatible with memory saver mode"
            )

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        if self._pool is None:
            self._pool = get_or_create_global_graph_memory_pool(self._device_module)
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        self._shared_output_buffer = None
        self.begin_cuda_graph_capture()
        try:
            with self.replay_session():
                yield
        finally:
            try:
                self.end_cuda_graph_capture()
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

        graph = BreakableCUDAGraph(self.deduped_cuda_graph)
        captured_fn = (
            eager_on_graph(True)(forward_fn) if self._debug_eager else forward_fn
        )
        size = shape_key.size
        with BreakableCUDAGraphCapture(
            cuda_graph=graph,
            pool=self._pool,
            stream=self._capture_stream,
        ):
            out = captured_fn()
            if self._shared_output_buffer is not None:
                self._copy_output_to_buffer(out, self._shared_output_buffer, size)

        if self._shared_output_buffer is None:
            self._shared_output_buffer = out
            stored = self._slice_output(out, size)
        else:
            stored = self._slice_output(self._shared_output_buffer, size)

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = stored

    def _slice_output(self, output: Any, num_tokens: int) -> Any:
        if output is None:
            return None
        if torch.is_tensor(output):
            return output[:num_tokens]
        if isinstance(output, PPProxyTensors):
            return output[:num_tokens]
        if isinstance(output, tuple):
            return tuple(self._slice_output(item, num_tokens) for item in output)
        if isinstance(output, list):
            return [self._slice_output(item, num_tokens) for item in output]
        if dataclasses.is_dataclass(output) and not isinstance(output, type):
            # e.g. LogitsProcessorOutput: slice the per-token tensor fields (next_token_logits /
            # hidden_states), leave None / scalar / list fields (logprobs, filled later by the sampler) as-is.
            return dataclasses.replace(
                output,
                **{
                    f.name: getattr(output, f.name)[:num_tokens]
                    for f in dataclasses.fields(output)
                    if torch.is_tensor(getattr(output, f.name))
                },
            )
        raise TypeError(f"Unsupported BCG output type: {type(output)}")

    def _copy_output_to_buffer(
        self, output: Any, output_buffer: Any, num_tokens: int
    ) -> None:
        if output is None or output_buffer is None:
            if output is None and output_buffer is None:
                return
            raise ValueError(
                "BCG output structure changed between capture sizes: "
                f"{type(output)} vs {type(output_buffer)}"
            )
        if torch.is_tensor(output) and torch.is_tensor(output_buffer):
            output_buffer[:num_tokens].copy_(output[:num_tokens])
            return
        if isinstance(output, PPProxyTensors) and isinstance(
            output_buffer, PPProxyTensors
        ):
            if output.tensors.keys() != output_buffer.tensors.keys():
                raise ValueError(
                    "BCG output proxy structure changed between capture sizes: "
                    f"{output.tensors.keys()} != {output_buffer.tensors.keys()}"
                )
            for key, tensor in output.tensors.items():
                self._copy_output_to_buffer(
                    tensor, output_buffer.tensors[key], num_tokens
                )
            return
        if isinstance(output, (list, tuple)) and isinstance(
            output_buffer, type(output)
        ):
            if len(output) != len(output_buffer):
                raise ValueError(
                    "BCG output sequence structure changed between capture sizes: "
                    f"{len(output)} != {len(output_buffer)}"
                )
            for item, buffer in zip(output, output_buffer):
                self._copy_output_to_buffer(item, buffer, num_tokens)
            return
        if dataclasses.is_dataclass(output) and dataclasses.is_dataclass(output_buffer):
            for f in dataclasses.fields(output):
                v = getattr(output, f.name)
                if torch.is_tensor(
                    v
                ):  # copy the per-token tensor fields into the captured buffer
                    self._copy_output_to_buffer(
                        v, getattr(output_buffer, f.name), num_tokens
                    )
            return
        raise TypeError(
            "Unsupported BCG output buffer pair: "
            f"{type(output)} vs {type(output_buffer)}"
        )

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        with enable_breakable_cuda_graph():
            yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        self._graphs[shape_key].replay()
        if _post_replay_hook is not None:
            _post_replay_hook()  # once-per-step boundary (no re-replay; cold already staged at the break)
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self.close()
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None
        self._shared_output_buffer = None
