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
"""Run the model with xpu graph and torch.compile."""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch.profiler import ProfilerActivity, profile

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class XPUGraphRunner(CudaGraphRunner):
    """A XPUGraphRunner runs the forward pass of a model with xpu graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)

        # model_runner.server_args.disable_cuda_graph_padding
        # require_attn_tp_gather(model_runner.server_args)
        # model_runner.server_args.enable_pdmux

        assert (
            not self.model_runner.server_args.enable_piecewise_cuda_graph
        ), "XPUGraphRunner does not support Piecewise Graph yet."

        assert (
            not self.model_runner.server_args.enable_memory_saver
        ), "XPUGraphRunner does not support Torch Memory Saver yet."

        assert (
            not self.model_runner.server_args.enable_lora
        ), "XPUGraphRunner does not support LoRA yet."
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
            model_runner.spec_algorithm == SpeculativeAlgorithm.NONE
        ), "XPUGraphRunner does not support speculative inference yet."
        # TODO add compile support for encoder-decoder models
        assert (
            not self.is_encoder_decoder
        ), "XPUGraphRunner does not support encoder-decoder models yet."
        assert self.dp_size == 1, "XPUGraphRunner does not support DP yet."
        assert self.pp_size == 1, "XPUGraphRunner does not support PP yet."

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
        # torch.xpu.memory._record_memory_history()
        return profile_context

    def _post_process_after_profile(self, prof_context):
        # torch.xpu.memory._dump_snapshot(f"xpu_graph_runner_memory_usage.pickle")
        # torch.xpu.memory._record_memory_history(enabled=None)
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
