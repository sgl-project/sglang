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

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Union

import torch

from sglang.srt.configs.model_config import AttentionArch, is_deepseek_dsa
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker


class EAGLEDraftNpuGraphRunner(EAGLEDraftCudaGraphRunner):
    def __init__(self, eagle_worker: EagleDraftWorker):
        self._init_arch_map()
        super().__init__(eagle_worker)

    def _init_arch_map(self):
        self.attr_name: Dict[str, str] = {
            AttentionArch.MLA: "actual_seq_lengths_kv",
            AttentionArch.MHA: "context_lens",
        }
        self.attr_type: Dict[str, Union[list, torch.Tensor]] = {
            AttentionArch.MLA: [],
            AttentionArch.MHA: torch.Tensor(),
        }

    def _cache_loc_dtype(self):
        return torch.int32

    def _get_update_attr_name(self):
        return self.attr_name[AttentionArch.MLA]

    def _get_update_attr_type(self):
        return self.attr_type[AttentionArch.MLA]

    def _replay_graph(self, shape_key, forward_batch):
        if not is_deepseek_dsa(self.model_runner.model_config.hf_config):
            seq_lens_for_each_draft_step = []
            for speculative_step_id in range(self.speculative_num_steps - 1):
                seq_lens_cpu = (
                    forward_batch.seq_lens_cpu[: self.raw_bs] + speculative_step_id + 1
                )
                seq_lens = seq_lens_cpu.tolist() + [0] * (self.bs - self.raw_bs)
                seq_lens_for_each_draft_step.append(seq_lens)
            attr_name = self._get_update_attr_name()
            cpu_update_input = [{attr_name: sl} for sl in seq_lens_for_each_draft_step]
            return self.backend.replay_with_input_update(
                shape_key, seq_lens=None, cpu_update_input=cpu_update_input
            )
        else:
            return self.backend.replay(shape_key, forward_batch)
