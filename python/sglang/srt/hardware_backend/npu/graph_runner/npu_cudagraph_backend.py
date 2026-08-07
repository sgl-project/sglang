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

import queue
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
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.utils import empty_context, get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )


class NPUCudaGraphBackend(FullCudaGraphBackend):
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
        self._update_queue: queue.Queue = queue.Queue()
        self._update_thread = threading.Thread(
            target=self._update_worker,
            name="npu-graph-update",
            daemon=True,
        )
        self._update_thread.start()
        self._bound_update_signatures: Dict[Any, Any] = {}

    def _update_worker(self) -> None:
        """Reuse one device-bound worker for NPUGraph input updates."""
        self._device_module.set_device(self._device_id)
        while True:
            task = self._update_queue.get()
            if task is None:
                return
            graph, cpu_update_input, done, errors, reuse_dispatch = task
            try:
                if reuse_dispatch:
                    dispatch_mode = graph.graph_dispatch_mode
                    with torch.npu.stream(dispatch_mode.update_stream):
                        for record in dispatch_mode.graph_dispatch_records:
                            torch.npu.graph_task_update_begin(
                                dispatch_mode.update_stream, record.handle
                            )
                            record.op_cache_entry(*record.args, **record.kwargs)
                            torch.npu.graph_task_update_end(dispatch_mode.update_stream)
                            record.event.record(dispatch_mode.update_stream)
                else:
                    graph.update(cpu_update_input=cpu_update_input)
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()

    @staticmethod
    def _update_signature(cpu_update_input: Any) -> Any:
        """Build a stable value signature for small host graph inputs."""

        def freeze(value: Any) -> Any:
            if isinstance(value, dict):
                return tuple(sorted((key, freeze(item)) for key, item in value.items()))
            if isinstance(value, (list, tuple)):
                return tuple(freeze(item) for item in value)
            if isinstance(value, np.ndarray):
                return (
                    str(value.dtype),
                    tuple(value.shape),
                    tuple(value.reshape(-1).tolist()),
                )
            if isinstance(value, torch.Tensor):
                host = value.detach().cpu()
                return (
                    str(host.dtype),
                    tuple(host.shape),
                    tuple(host.reshape(-1).tolist()),
                )
            return value

        return freeze(cpu_update_input)

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
        signature = self._update_signature(cpu_update_input)

        if self._bound_update_signatures.get(shape_key) == signature:
            # Denoising iterations within one output block retain the same
            # host-side attention attributes. The external attention tasks and
            # their events must still be submitted for every replay, but their
            # arguments no longer need handler lookup and rebinding.
            done = threading.Event()
            errors = []
            self._update_queue.put((graph, None, done, errors, True))
            graph.replay()
            done.wait()
            if errors:
                raise errors[0]
            return self._outputs[shape_key]

        done = threading.Event()
        errors = []
        self._update_queue.put((graph, cpu_update_input, done, errors, False))
        graph.replay()
        done.wait()
        if errors:
            raise errors[0]
        self._bound_update_signatures[shape_key] = signature
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._bound_update_signatures.clear()
        self._pool = None
