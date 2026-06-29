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

from typing import TYPE_CHECKING

import torch

from sglang.srt.configs.model_config import AttentionArch, is_deepseek_dsa
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker


class EAGLEDraftExtendNpuGraphRunner(EAGLEDraftExtendCudaGraphRunner):
    def __init__(self, eagle_worker: EagleDraftWorker):
        self.use_fia_v2 = (
            eagle_worker.draft_runner.model_config.attention_arch == AttentionArch.MLA
            and get_global_server_args().kv_cache_dtype == "fp8_e4m3"
        )
        super().__init__(eagle_worker)

    def _cache_loc_dtype(self):
        return torch.int32

    def _replay_graph(self, shape_key, forward_batch):
        if not is_deepseek_dsa(self.model_runner.model_config.hf_config):
            seq_lens = forward_batch.seq_lens_cpu.tolist() + [0] * (
                self.bs - self.raw_bs
            )
            return self.backend.replay_with_input_update(
                shape_key,
                seq_lens=seq_lens,
                attr_name=(
                    "actual_seq_kvlen"
                    if self.use_fia_v2
                    else "actual_seq_lengths_kv"
                ),
                attr_type=[],
            )
        else:
            return self.backend.replay(shape_key, forward_batch)
