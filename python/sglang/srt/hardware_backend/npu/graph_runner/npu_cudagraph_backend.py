"""NPUCudaGraphBackend — Full-style backend for the Ascend NPU.

Mirror of ``FullCudaGraphBackend`` with two differences:
  - Captures into ``torch.npu.NPUGraph`` via ``torch.npu.graph(...)``.
  - Exposes ``replay_with_input_update(shape_key, seq_lens, attr_name)``
    so ``NPUGraphRunner`` can swap the recorded graph's input bindings
    on-the-fly for variable seq_lens at replay time (NPU's
    ``NPUGraph.update(...)`` API).

Imports of ``torch.npu`` are deferred so the module loads on non-NPU
hosts without error (the methods are not callable on those hosts).
"""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager, contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import numpy as np
import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.cuda_graph_backend.base_cudagraph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.utils import empty_context, get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class NPUCudaGraphBackend(BaseCudaGraphBackend):
    """Single-graph capture for Ascend NPU.

    Same lifecycle as ``FullCudaGraphBackend`` (one graph per shape;
    attention metadata captured inside the graph). Replay path may use
    ``NPUGraph.update(...)`` to substitute fresh seq_lens without
    re-recording.
    """

    def __init__(self, *, enable_memory_saver: bool = False) -> None:
        self._graphs: Dict[Any, Any] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = None
        self._tp_group = None
        self._memory_saver_adapter: Optional[Any] = None
        self._capture_stream = None
        self._enable_memory_saver = enable_memory_saver
        self._enable_torch_compile = False

    def prepare(self, runner) -> None:
        self._device_module = runner.device_module
        self._tp_group = runner.model_runner.tp_group
        self._memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self._enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        self._enable_torch_compile = runner.enable_torch_compile

    def can_run(self, forward_batch: "ForwardBatch") -> bool:
        return True

    @contextmanager
    def capture_session(self, stream):
        if self._pool is None:
            self._pool = self._device_module.graph_pool_handle()
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: Any,
        forward_fn: Callable[[], Any],
        dummies: Optional[Any] = None,
    ) -> None:
        import torch_npu  # noqa: F401  (verifies NPU availability)

        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()

        graph = torch.npu.NPUGraph()

        if self._enable_torch_compile:
            skip_guard_context = torch.compiler.set_stance(skip_guard_eval_unsafe=True)
        else:
            skip_guard_context = empty_context()

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
            graph_ctx = torch.npu.graph

        with skip_guard_context, graph_ctx(
            graph,
            pool=self._pool,
            stream=self._capture_stream,
            auto_dispatch_capture=True,
        ):
            out = forward_fn()

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out

    def has_shape(self, shape_key: Any) -> bool:
        return shape_key in self._graphs

    def replay(
        self,
        shape_key: Any,
        static_forward_batch: "ForwardBatch",
        **kwargs,
    ) -> Any:
        # Default replay path used when no async input update is needed
        # (e.g. deepseek-nsa). NPUGraphRunner uses
        # ``replay_with_input_update`` for the common case.
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def replay_with_input_update(
        self,
        shape_key: Any,
        seq_lens: list,
        attr_name: str,
        attr_type: Any,
    ) -> Any:
        """Spawn an update thread that rebinds seq_lens on the recorded
        NPU graph, then replay. Used by ``NPUGraphRunner.replay`` when
        the model is not deepseek-nsa.
        """
        if isinstance(attr_type, torch.Tensor):
            seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))

        graph = self._graphs[shape_key]

        def _update():
            graph.update(cpu_update_input=[{attr_name: seq_lens}])

        thread = threading.Thread(target=_update)
        thread.start()
        graph.replay()
        thread.join()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None
