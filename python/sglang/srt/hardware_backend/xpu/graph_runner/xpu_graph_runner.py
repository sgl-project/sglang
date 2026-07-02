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
"""Run the model with XPU graph.

XPUGraphRunner is a thin subclass of DecodeCudaGraphRunner. All
capture/replay mechanics live in XPUCudaGraphBackend (resolved via
resolve_decode_backend for device == 'xpu'). This class adds:
  - XPU-specific asserts (no TP/DP/PP, no spec, no piecewise prefill).
  - _create_device_graph / _capture_graph overrides (dead code in the
    current architecture — capture lives in XPUCudaGraphBackend.capture_one —
    kept for structural symmetry with NPUGraphRunner).
  - Profile context override using XPU profiler activities.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Optional

import torch
from torch.profiler import ProfilerActivity, profile

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class XPUGraphRunner(DecodeCudaGraphRunner):
    """Runs the forward pass of a model with XPU graph."""

    def __init__(
        self,
        model_runner: ModelRunner,
        *,
        attn_backend=None,
        speculative_num_steps: Optional[int] = None,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        assert model_runner.tp_size == 1, "XPUGraphRunner does not support TP > 1 yet."
        assert model_runner.dp_size == 1, "XPUGraphRunner does not support DP yet."
        assert model_runner.pp_size == 1, "XPUGraphRunner does not support PP yet."
        assert (
            not model_runner.server_args.enable_memory_saver
        ), "XPUGraphRunner does not support Torch Memory Saver yet."
        assert (
            not model_runner.server_args.enable_lora
        ), "XPUGraphRunner does not support LoRA yet."
        assert (
            model_runner.spec_algorithm == SpeculativeAlgorithm.NONE
        ), "XPUGraphRunner does not support speculative inference yet."

        super().__init__(
            model_runner,
            attn_backend=attn_backend,
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

        assert (
            not self.enable_two_batch_overlap
        ), "XPUGraphRunner does not support two batch overlap yet."
        assert (
            not self.require_mlp_tp_gather
        ), "XPUGraphRunner does not support MLP TP gather yet."
        assert (
            not self.require_mlp_sync
        ), "XPUGraphRunner does not support MLP sync yet."
        assert (
            not self.require_gathered_buffer
        ), "XPUGraphRunner does not support gathered buffer yet."
        assert (
            not self.is_encoder_decoder
        ), "XPUGraphRunner does not support encoder-decoder models yet."

    # Dead code in the current architecture (capture lives in
    # XPUCudaGraphBackend.capture_one). Kept for symmetry with NPUGraphRunner.
    def _create_device_graph(self):
        return torch.xpu.XPUGraph()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.model_runner.server_args.enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        graph_fn = (
            partial(memory_saver_adapter.cuda_graph, tag=GPU_MEMORY_TYPE_CUDA_GRAPH)
            if memory_saver_adapter.enabled
            else self.device_module.graph
        )
        with graph_fn(xpu_graph=graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    def _init_profile_context_and_memory_record(self):
        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            record_shapes=True,
        )
        torch.xpu.memory._record_memory_history()
        return profile_context

    def _post_process_after_profile(self, prof_context):
        torch.xpu.memory._dump_snapshot("xpu_graph_runner_memory_usage.pickle")
        torch.xpu.memory._record_memory_history(enabled=None)
        log_message = (
            "Sorted by XPU Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="self_xpu_time_total"
            )
            + "\n\nSorted by CPU Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="self_cpu_time_total"
            )
            + "\n\nMemory Usage is saved to xpu_graph_runner_memory_usage.pickle\n"
        )
        logger.info(log_message)
