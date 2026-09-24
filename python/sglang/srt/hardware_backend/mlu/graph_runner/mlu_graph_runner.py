# Copyright 2023-2026 SGLang Team
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
"""Run the model with MLU graph and torch.compile.

MLUGraphRunner is a thin subclass of DecodeCudaGraphRunner: the
factory returns FullMLUGraphBackend for MLU devices, so all
capture/replay mechanics live in the backend. This class adds:
  - Smaller cache_loc dtype (int32 instead of int64).
  - Profile context override that dumps an MLU memory snapshot to disk.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.profiler import ProfilerActivity, profile

from sglang.srt.model_executor.runner import DecodeCudaGraphRunner

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class MLUGraphRunner(DecodeCudaGraphRunner):
    """A MLUGraphRunner runs the forward pass of a model with mlu graph and torch.compile."""

    def __init__(self, model_runner: "ModelRunner"):
        super().__init__(model_runner)

    def _cache_loc_dtype(self):
        return torch.int32

    def _init_profile_context_and_memory_record(self):
        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.MLU],
            record_shapes=True,
        )
        torch.mlu.memory._record_memory_history()
        return profile_context

    def _post_process_after_profile(self, prof_context):
        torch.mlu.memory._dump_snapshot("mlu_graph_runner_memory_usage.pickle")
        torch.mlu.memory._record_memory_history(enabled=None)
        log_message = (
            "Sorted by MLU Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="self_mlu_time_total"
            )
            + "\n\nSorted by CPU Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="self_cpu_time_total"
            )
            + "\n\nMemory Usage is saved to mlu_graph_runner_memory_usage.pickle\n"
        )
        logger.info(log_message)
