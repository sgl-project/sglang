# Copyright 2025 SGLang Team
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
"""Run the model with npu graph and torch.compile"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Dict, Union

import numpy as np
import torch

from sglang.srt.configs.model_config import AttentionArch, is_deepseek_dsa
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

from sglang.srt.utils import is_npu

logger = logging.getLogger(__name__)

if is_npu():
    torch.cuda.CUDAGraph = torch.npu.NPUGraph
    torch.cuda.synchronize = torch.npu.synchronize
    torch.cuda.graph = torch.npu.graph
    torch.cuda.stream = torch.npu.stream
    torch.cuda.Stream = torch.npu.Stream
    torch.cuda.current_stream = torch.npu.current_stream


class EAGLEDraftNpuGraphRunner(EAGLEDraftCudaGraphRunner):
    def __init__(self, eagle_worker: EagleDraftWorker):
        super().__init__(eagle_worker)
        self.update_attr_name = None
        self.update_attr_type = None
        self._init_arch_map()

    def _init_arch_map(self):
        self.attr_name: Dict[str, str] = {
            AttentionArch.MLA: "actual_seq_lengths_kv",
            AttentionArch.MHA: "context_lens",
        }
        self.attr_type: Dict[str, Union[list, torch.Tensor]] = {
            AttentionArch.MLA: [],
            AttentionArch.MHA: torch.Tensor(),
        }

    def _create_graph(self):
        return torch.npu.NPUGraph()

    def _capture_init(self, run_once_fn):
        for _ in range(2):
            torch.npu.synchronize()
            self.model_runner.tp_group.barrier()
            run_once_fn()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        with torch.npu.graph(
            graph, pool=pool, stream=stream, auto_dispatch_capture=True
        ):
            out = run_once_fn()
        return out

    def _is_dense_mha(self):
        # Only dense MHA models have a per-layer context_lens updatable op in the
        # captured draft graph (so the replay update must be expanded per layer).
        # MLA and hybrid (mamba/linear-attention) models keep the original single
        # actual_seq_lengths_kv update that covers all layers.
        mc = self.model_runner.model_config
        return (
            mc.attention_arch == AttentionArch.MHA
            and getattr(self.model_runner, "mambaish_config", None) is None
        )

    def _get_update_attr_name(self):
        arch = AttentionArch.MHA if self._is_dense_mha() else AttentionArch.MLA
        return self.attr_name[arch]

    def _get_update_attr_type(self):
        arch = AttentionArch.MHA if self._is_dense_mha() else AttentionArch.MLA
        return self.attr_type[arch]

    def _replay_update(self, seq_lens_list):
        # The captured draft graph records one seq-len updatable op per executed
        # full-attention layer per draft step. That count isn't reliably derivable
        # from the (base) model config (a full-size duplicate draft has one per
        # layer; a 1-layer MTP module has one; MLA layers may build two attn
        # submodules but execute one), so read the exact number straight from the
        # graph's recorded ops. Records are in step-outer / layer-inner order, so
        # repeat each step's seq_lens `per_step` times.
        #
        # graph_dispatch_mode / graph_dispatch_records are torch_npu internals;
        # access them defensively and fall back to one update per step (the
        # single-shared-op case) if a future torch_npu version changes them.
        steps = len(seq_lens_list)
        per_step = 1
        if steps:
            dispatch_mode = getattr(self.graphs[self.bs], "graph_dispatch_mode", None)
            records = getattr(dispatch_mode, "graph_dispatch_records", None)
            if records is not None:
                per_step = max(1, len(records) // steps)

        cpu_update_input = []
        for seq_lens in seq_lens_list:
            if isinstance(self.update_attr_type, torch.Tensor):
                seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))
            for _ in range(per_step):
                cpu_update_input.append({self.update_attr_name: seq_lens})

        self.graphs[self.bs].update(cpu_update_input=cpu_update_input)

    def _replay(self, forward_batch: ForwardBatch):
        self.update_attr_name = self._get_update_attr_name()
        self.update_attr_type = self._get_update_attr_type()
        if not is_deepseek_dsa(self.model_runner.model_config.hf_config):
            seq_lens_for_each_draft_step = []
            for speculative_step_id in range(self.speculative_num_steps - 1):
                seq_lens_cpu = forward_batch.seq_lens_cpu + speculative_step_id + 1
                seq_lens = seq_lens_cpu.tolist() + [0] * (self.bs - self.raw_bs)
                seq_lens_for_each_draft_step.append(seq_lens)
            thread = threading.Thread(
                target=self._replay_update, args=(seq_lens_for_each_draft_step,)
            )
            thread.start()
            self.graphs[self.bs].replay()
            thread.join()
        else:
            self.graphs[self.bs].replay()

    def _cache_loc_dtype(self):
        return torch.int32
