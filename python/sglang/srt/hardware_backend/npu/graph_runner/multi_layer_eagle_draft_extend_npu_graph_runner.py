# Copyright 2024-2025 SGLang Team
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

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleDraftExtendCudaGraphRunner,
    MultiLayerEagleMultiStepDraftExtendCudaGraphRunner,
)
from sglang.srt.utils import get_available_gpu_memory

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.speculative.multi_layer_eagle_worker import (
        MultiLayerEagleDraftWorker,
    )


class MultiLayerEagleDraftExtendNpuGraphRunner(
    MultiLayerEagleDraftExtendCudaGraphRunner
):
    def __init__(self, eagle_worker: MultiLayerEagleDraftWorker, step: int):
        super().__init__(eagle_worker, step)

    def _replay_graph(self, shape_key, forward_batch):
        seq_lens = self.buffers.seq_lens_cpu[: self.raw_bs].tolist() + [0] * (
            self.bs - self.raw_bs
        )
        return self.backend.replay_with_input_update(
            shape_key,
            seq_lens=seq_lens,
            attr_name="actual_seq_kvlen",
            attr_type=[],
        )


class MultiLayerEagleMultiStepDraftExtendNpuGraphRunner(
    MultiLayerEagleMultiStepDraftExtendCudaGraphRunner
):
    def __init__(self, eagle_worker: MultiLayerEagleDraftWorker):
        super().__init__(eagle_worker)

    def _init_and_capture(self):
        if cuda_graph_fully_disabled():
            self.runners = [None] * self.speculative_num_steps
            return

        self.runners: List[Optional[MultiLayerEagleDraftExtendNpuGraphRunner]] = []
        buffer_len_list: List[int] = []

        for step in range(self.speculative_num_steps):
            if self.draft_extend_attn_backend_list[step]:
                runner = MultiLayerEagleDraftExtendNpuGraphRunner(
                    self.eagle_worker, step
                )
                self.runners.append(runner)

                self.seq_len_fill_value = runner.seq_len_fill_value
                self.max_bs = runner.max_bs
                buffer_len_list.append(runner.max_num_token)
                self.offsets.append(self.offsets[-1] + runner.max_num_token)
            else:
                self.runners.append(None)

        self.cuda_graph_buffers["seq_lens_cpu"] = torch.full(
            (self.max_bs,),
            self.seq_len_fill_value,
            dtype=torch.int32,
        )

        with torch.device(self.device):
            self.cuda_graph_buffers["input_ids"] = torch.zeros(
                (self.offsets[-1],), dtype=torch.int64
            )
            self.cuda_graph_buffers["out_cache_loc"] = torch.ones(
                (self.offsets[-1],), dtype=torch.int64
            )
            self.cuda_graph_buffers["positions"] = torch.zeros(
                (self.offsets[-1],), dtype=torch.int64
            )

            self.cuda_graph_buffers["seq_lens"] = torch.full(
                (self.max_bs,),
                self.seq_len_fill_value,
                dtype=torch.int32,
            )
            self.cuda_graph_buffers["req_pool_indices"] = torch.zeros(
                (self.max_bs,), dtype=torch.int64
            )
            self.cuda_graph_buffers["num_correct_drafts"] = torch.full(
                (self.max_bs,), 1, dtype=torch.int32
            )
            self.cuda_graph_buffers["num_accept_tokens"] = torch.full(
                (self.max_bs,), 1, dtype=torch.int32
            )

        for step in range(self.speculative_num_steps - 1, -1, -1):
            if self.runners[step] is not None:
                tic = time.perf_counter()
                before_mem = get_available_gpu_memory(self.device, self.gpu_id)
                logger.info(
                    f"Capture draft extend CUDA graph begin. step={step}, "
                    f"avail mem={before_mem:.2f} GB"
                )

                self.runners[step].init_buffers_and_capture(
                    self.cuda_graph_buffers,
                    self.offsets[step],
                    (
                        self.runners[step + 1]
                        if step + 1 < self.speculative_num_steps
                        else None
                    ),
                )

                after_mem = get_available_gpu_memory(self.device, self.gpu_id)
                logger.info(
                    "Capture draft extend CUDA graph end. "
                    f"step={step}, elapsed={time.perf_counter() - tic:.2f} s, "
                    f"mem usage={(before_mem - after_mem):.2f} GB, "
                    f"avail mem={after_mem:.2f} GB."
                )
