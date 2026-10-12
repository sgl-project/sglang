"""CUDA graphs of the DeepSeek-V4.1 late layers under decoder SWA bounded replay,
keyed by tail rows. A prefill CUDA graph, keyed by token count, replays one from a
graph break: the tail's row count does not follow the token count."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import msgspec
import torch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_memory_pool,
    graph_pool_capture_scope,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_device_module

Tensors = Tuple[Optional[torch.Tensor], ...]

# Eager runs before a capture, as for the prefill graph: lazy kernels compile here.
_WARMUP_RUNS = 2


class _CapturedLateLayers(msgspec.Struct):
    graph: BreakableCUDAGraph
    # The graph reads its inputs and writes its outputs at these addresses.
    inputs: Tensors
    outputs: Tensors


class LateLayerGraphs:
    def __init__(self) -> None:
        self._captured_of_rows: Dict[int, _CapturedLateLayers] = {}

    @property
    def is_empty(self) -> bool:
        return not self._captured_of_rows

    def run(self, inputs: Tensors, forward: Callable[..., Tensors]) -> Tensors:
        """Run ``forward`` over ``inputs`` through the graph of their row count,
        capturing it on first use. The outputs are valid until the next run."""
        rows = inputs[0].shape[0]
        captured = self._captured_of_rows.get(rows)
        if captured is None:
            captured = self._captured_of_rows[rows] = _capture(inputs, forward)
            return captured.outputs
        for static, live in zip(captured.inputs, inputs):
            if static is not None:
                static.copy_(live)
        captured.graph.replay()
        return captured.outputs


def _capture(inputs: Tensors, forward: Callable[..., Tensors]) -> _CapturedLateLayers:
    static_inputs = tuple(None if t is None else t.clone() for t in inputs)
    for _ in range(_WARMUP_RUNS):
        forward(*static_inputs)
    graph = BreakableCUDAGraph()
    device_module = get_device_module()
    with (
        graph_pool_capture_scope(),
        BreakableCUDAGraphCapture(
            cuda_graph=graph,
            pool=get_or_create_global_graph_memory_pool(device_module),
            stream=device_module.current_stream(),
            barrier_fn=get_parallel().tp_group.barrier,
        ),
    ):
        outputs = forward(*static_inputs)
    return _CapturedLateLayers(graph=graph, inputs=static_inputs, outputs=outputs)
