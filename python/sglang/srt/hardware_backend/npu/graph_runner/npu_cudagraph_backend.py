"""NPUCudaGraphBackend — Ascend NPU full-graph capture (torch.npu.NPUGraph).

Mirrors FullCudaGraphBackend with two differences:
  - Captures via torch.npu.graph(...) into torch.npu.NPUGraph.
  - replay_with_input_update(shape_key, seq_lens, attr_name) rebinds
    the recorded graph's input bindings for variable seq_lens at replay
    time (NPU's NPUGraph.update(...) API).

torch.npu is imported lazily inside methods so the module loads on
non-NPU hosts.
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
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.utils import empty_context, get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )


class NPUCudaGraphBackend(BaseCudaGraphBackend):
    """One torch.npu.NPUGraph per shape; attention metadata captured
    inside the graph. replay_with_input_update substitutes fresh
    seq_lens without re-recording."""

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
    ) -> None:
        self._graphs: Dict[Any, Any] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        self._enable_torch_compile = getattr(
            cuda_graph_runner, "enable_torch_compile", False
        )

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
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        import torch_npu  # noqa: F401  (verifies NPU availability)

        # Two warmups so kernels are loaded and one-time setup is paid before capture.
        # post_warmup_hook lets the attention backend reset state that warmup mutated.
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

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

    def can_run(self, forward_batch: ForwardBatch, shape_key: Any) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        yield

    def replay(
        self,
        shape_key: Any,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def replay_with_input_update(
        self,
        shape_key: Any,
        seq_lens: Any,
        attr_name: str = None,
        attr_type: Any = None,
        cpu_update_input: list = None,
    ) -> Any:
        """Rebind seq_lens on the recorded NPU graph in a background
        thread, then replay. Used when the model is not deepseek-nsa.

        Two calling conventions:
        1. (legacy) seq_lens + attr_name + attr_type:
           Constructs cpu_update_input=[{attr_name: seq_lens}] internally.
        2. cpu_update_input: A list of {attr_name: seq_lens} dicts,
           one per speculative step.  Used by EAGLE draft runners.
        """
        if cpu_update_input is None:
            if isinstance(attr_type, torch.Tensor):
                seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))
            cpu_update_input = [{attr_name: seq_lens}]

        graph = self._graphs[shape_key]

        def _update():
            graph.update(cpu_update_input=cpu_update_input)

        thread = threading.Thread(target=_update)
        thread.start()
        graph.replay()
        thread.join()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None
