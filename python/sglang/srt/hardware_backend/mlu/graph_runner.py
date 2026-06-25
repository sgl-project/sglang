from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, Union

import torch
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from torch.profiler import ProfilerActivity, profile

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class MLUCudaGraphBackend(BaseCudaGraphBackend):
    """Full-graph capture backend for torch.mlu.MLUGraph."""

    def __init__(self, cuda_graph_runner: DecodeCudaGraphRunner, **_: object) -> None:
        self._runner = cuda_graph_runner
        self._graphs = {}
        self._outputs = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream = None

    @contextmanager
    def capture_session(self, stream):
        if self._pool is None:
            self._pool = self._device_module.graph_pool_handle()
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None

    def capture_one(self, shape_key, forward_fn, dummies=None, post_warmup_hook=None):
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        graph = self._runner._create_device_graph()
        out = self._runner._capture_graph(
            graph, self._pool, self._capture_stream, forward_fn
        )
        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out

    def can_run(self, forward_batch, shape_key) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        yield

    def replay(self, shape_key, static_forward_batch, **kwargs):
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None


class MLUGraphRunner(DecodeCudaGraphRunner):
    """A MLUGraphRunner runs the forward pass of a model with mlu graph and torch.compile."""

    def __init__(self, model_runner: "ModelRunner"):
        super().__init__(model_runner)
        self.update_attr_name = None
        self.update_attr_type = None
        self.model_runner = model_runner
        self._init_arch_map()

    def _init_arch_map(self):
        self.attr_name: Dict[str, str] = {}
        self.attr_type: Dict[str, Union[list, torch.Tensor]] = {}

    def _create_device_graph(self):
        return torch.mlu.MLUGraph()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        with torch.mlu.graph(
            graph,
            pool=pool,
            stream=stream,
            # auto_dispatch_capture=True,
        ):
            out = run_once_fn()
        return out

    def _cache_loc_dtype(self):
        return torch.int32

    def _position_dtype(self):
        return torch.int32

    def _init_profile_context_and_memory_record(self):
        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.MLU],
            record_shapes=True,
        )
        torch.mlu.memory._record_memory_history()
        return profile_context

    def _post_process_after_profile(self, prof_context):
        snapshot_file = (
            f"mlu_graph_runner_memory_usage_{os.getpid()}_{time.time_ns()}.pickle"
        )
        torch.mlu.memory._dump_snapshot(snapshot_file)
        torch.mlu.memory._record_memory_history(enabled=None)
        log_message = (
            "Sorted by MLU Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="mlu_time_total", row_limit=10
            )
            + "\n\nSorted by CPU Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="cpu_time_total", row_limit=10
            )
            + f"\n\nMemory Usage is saved to {snapshot_file}\n"
        )
        logger.info(log_message)
