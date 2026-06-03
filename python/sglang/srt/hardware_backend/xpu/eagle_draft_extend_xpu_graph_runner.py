"""Run the model with xpu graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker


class EAGLEDraftExtendXpuGraphRunner(EAGLEDraftExtendCudaGraphRunner):
    def __init__(self, eagle_worker: EagleDraftWorker):
        super().__init__(eagle_worker)

    def _create_graph(self):
        return torch.xpu.XPUGraph()

    def _capture_init(self, run_once_fn):
        for _ in range(2):
            torch.xpu.synchronize()
            self.model_runner.tp_group.barrier()
            run_once_fn()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        with torch.xpu.graph(graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out
