# Copyright 2023-2024 SGLang Team
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
"""Run the model with npu graph and torch.compile."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.cuda_graph_runner import (
    CudaGraphRunner,
    get_global_graph_memory_pool,
    global_graph_memory_pool,
    set_global_graph_memory_pool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_npu

logger = logging.getLogger(__name__)

_is_npu = is_npu()

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

if _is_npu:
    torch.cuda.CUDAGraph = torch.npu.NPUGraph
    torch.cuda.synchronize = torch.npu.synchronize
    torch.cuda.graph = torch.npu.graph
    torch.cuda.stream = torch.npu.stream
    torch.cuda.Stream = torch.npu.Stream
    torch.cuda.current_stream = torch.npu.current_stream


class NPUGraphRunner(CudaGraphRunner):
    """A NPUGraphRunner runs the forward pass of a model with npu graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)

    def _create_graph(self):
        return torch.npu.NPUGraph()

    def _capture_init(self, run_once_fn):
        for _ in range(2):
            torch.npu.synchronize()
            self.model_runner.tp_group.barrier()
            run_once_fn()

    def _capture_graph(self, graph, stream, run_once_fn):
        if get_global_graph_memory_pool() is None:
            set_global_graph_memory_pool(torch.npu.graph_pool_handle())
        # Set graph pool id globally to be able to use symmetric memory
        set_graph_pool_id(get_global_graph_memory_pool())
        with torch.npu.graph(
            graph,
            pool=global_graph_memory_pool,
            stream=stream,
            auto_dispatch_capture=True,
        ):
            out = run_once_fn()
        return out

    def replay_update(self, seq_lens):
        self.graphs[self.bs].update(
            cpu_update_input=[{"actual_seq_lengths_kv": seq_lens}]
        )

    def _update_and_replay(self, forward_batch: ForwardBatch):
        seq_lens = forward_batch.seq_lens.cpu().tolist() + [0] * (self.bs - self.raw_bs)

        thread = threading.Thread(target=self.replay_update, args=(seq_lens,))
        thread.start()
        self.graphs[self.bs].replay()
        thread.join()
