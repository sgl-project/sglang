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
from typing import TYPE_CHECKING

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleDraftExtendCudaGraphRunner,
    MultiLayerEagleMultiStepDraftExtendCudaGraphRunner,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


class MultiLayerEagleDraftExtendNpuGraphRunner(
    MultiLayerEagleDraftExtendCudaGraphRunner
):
    def _replay_graph(self, shape_key, forward_batch):
        seq_lens = self.buffers.seq_lens_cpu[: self.raw_bs].tolist() + [0] * (
            self.bs - self.raw_bs
        )
        use_fia_v2 = (
            self.model_runner.model_config.attention_arch == AttentionArch.MLA
            and get_global_server_args().kv_cache_dtype == "fp8_e4m3"
        )
        attr_name = (
            "actual_seq_kvlen"
            if use_fia_v2
            else "actual_seq_lengths_kv"
        )
        return self.backend.replay_with_input_update(
            shape_key,
            seq_lens=seq_lens,
            attr_name=attr_name,
            attr_type=[],
        )


class MultiLayerEagleMultiStepDraftExtendNpuGraphRunner(
    MultiLayerEagleMultiStepDraftExtendCudaGraphRunner
):
    def _create_runner(self, step: int) -> MultiLayerEagleDraftExtendNpuGraphRunner:
        return MultiLayerEagleDraftExtendNpuGraphRunner(self.eagle_worker, step)

    def _cuda_graph_disabled(self) -> bool:
        return cuda_graph_fully_disabled()
