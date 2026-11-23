# SPDX-License-Identifier: Apache-2.0
"""CUDA graph replay of a text encoder's eager forward."""

from collections import OrderedDict

import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_tp_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class TextEncoderGraphs:
    """CUDA graphs of the text-only encoder forward, one per prompt length.

    The eager forward launches ~2600 small kernels and is host-bound; a replay
    runs the same kernels, so the output is bit-identical. A length is captured
    on its second request; moved or replaced weights drop every graph.
    """

    max_graphs = 8

    def __init__(self):
        self.graphs = OrderedDict()
        self.seen = set()
        self.weights = None
        self.pool = None
        self.disabled = False

    def __call__(self, encoder, inputs, run):
        ids = inputs["input_ids"]
        if (
            self.disabled
            or ids.shape[0] != 1
            or "pixel_values" in inputs
            or not all(torch.is_tensor(v) and v.is_cuda for v in inputs.values())
            or torch.compiler.is_compiling()
            or torch.cuda.is_current_stream_capturing()
            or (model_parallel_is_initialized() and get_tp_group().world_size > 1)
        ):
            return run(inputs)
        tensors = [*encoder.parameters(), *encoder.buffers()]
        if not all(t.is_cuda for t in tensors):
            return run(inputs)
        # graphs read weights by address; offload, reload or replacement moves them
        weights = tuple(t.data_ptr() for t in tensors)
        if weights != self.weights:
            self.graphs.clear()
            self.seen.clear()
            self.weights = weights
        key = ids.shape[1]
        entry = self.graphs.get(key)
        if entry is None:
            if key not in self.seen:
                self.seen.add(key)
                return run(inputs)
            try:
                entry = self._capture(inputs, run)
            except Exception as e:
                logger.warning("Text encoder CUDA graph disabled: %s", e)
                self.disabled = True
                return run(inputs)
            self.graphs[key] = entry
            if len(self.graphs) > self.max_graphs:
                self.graphs.popitem(last=False)
        self.graphs.move_to_end(key)
        graph, static_inputs, output = entry
        for name, tensor in static_inputs.items():
            tensor.copy_(inputs[name])
        graph.replay()
        return output

    def _capture(self, inputs, run):
        static_inputs = {name: tensor.clone() for name, tensor in inputs.items()}
        # the side-stream warmup settles lazy allocations before capture
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(static_inputs)
        torch.cuda.current_stream().wait_stream(stream)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        # thread_local: an async output-save thread may use CUDA during capture
        with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
            output = run(static_inputs)
        return graph, static_inputs, output
