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
from sglang.srt.model_executor.runner.shape_key import ShapeKey
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
        self._device_id = self._device_module.current_device()
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
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
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

        with (
            skip_guard_context,
            graph_ctx(
                graph,
                pool=self._pool,
                stream=self._capture_stream,
                auto_dispatch_capture=True,
            ),
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

    def replay_with_input_update(
        self,
        shape_key: ShapeKey,
        seq_lens: Any,
        attr_name: str = None,
        attr_type: Any = None,
        cpu_update_input: list = None,
    ) -> Any:
        """Replay a graph, updating recorded operator inputs when needed.

        Requires auto_dispatch_capture=True. With no update records, replay
        directly without preparing or applying updates. Otherwise, run updates
        in a background thread concurrently with replay, join the thread, and
        re-raise any update exception in the caller.

        Two calling conventions:
        1. Provide seq_lens and attr_name, with optional attr_type; leave
           cpu_update_input=None. The method builds a one-element list for
           broadcast to all records. For example:
               seq_lens=[100, 200], attr_name="actual_seq_lengths_kv"
           becomes:
               [{"actual_seq_lengths_kv": [100, 200]}]
           A Tensor instance as attr_type selects conversion of seq_lens to a
           CPU int32 tensor. Its dtype and device are not used. Otherwise,
           seq_lens is used unchanged.

        2. Provide cpu_update_input as a list of update dictionaries; callers
           can pass seq_lens=None. The list is forwarded unchanged to
           graph.update(), and seq_lens, attr_name, and attr_type are ignored.
           A one-element list broadcasts the same updates to all records:
               [{"actual_seq_lengths_kv": [100, 200]}]
           A longer list must contain one dictionary per record, in capture
           order. This example requires exactly two update records:
               [{"actual_seq_lengths_kv": [101, 201]},
                {"actual_seq_lengths_kv": [102, 202]}]
           The inner length lists describe requests, not update records.
           Dictionaries may contain multiple input updates; supported keys
           and values depend on the recorded operator and its update handler.
        """
        graph = self._graphs[shape_key]
        # Read the update record count and require automatic capture to be enabled.
        # Fail if the capture state cannot be inspected.
        try:
            auto_dispatch_capture = graph.auto_dispatch_capture
            update_count = len(graph.graph_dispatch_mode.graph_dispatch_records)
        except (AttributeError, TypeError) as e:
            raise RuntimeError(
                "Cannot inspect NPU graph update records; "
                "check torch_npu graph API compatibility"
            ) from e
        if not auto_dispatch_capture:
            raise RuntimeError("NPU graph updates require auto_dispatch_capture=True")
        # With no update records, skip graph.update() and replay directly.
        if update_count == 0:
            graph.replay()
            return self._outputs[shape_key]

        if cpu_update_input is None:
            if isinstance(attr_type, torch.Tensor):
                seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))
            cpu_update_input = [{attr_name: seq_lens}]

        update_errors: list[Exception] = []

        def _update():
            try:
                self._device_module.set_device(self._device_id)
                graph.update(cpu_update_input=cpu_update_input)
            except Exception as e:
                update_errors.append(e)

        thread = threading.Thread(target=_update)
        thread.start()
        graph.replay()
        thread.join()
        if update_errors:
            raise update_errors[0]
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None
