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
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
import sglang.srt.model_executor.cuda_graph_runner

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

from torch._dynamo.eval_frame import DisableContext
from sglang.srt.model_executor.compilation.dynamo import (
    patch_dynamo_context,
    patch_dynamo_context_call,
    restore_dynamo_context_call
)
from sglang.srt.model_executor.compilation.npu_graph_compiler import NpuGraphCompiler

@contextmanager
def patch_model_npu(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
):
    yield model

class NPUGraphRunner(CudaGraphRunner):
    """A NPUGraphRunner runs the forward pass of a model with npu graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        if model_runner.server_args.enable_torch_compile:
            patch_dynamo_context()
        sglang.srt.model_executor.cuda_graph_runner.patch_model = patch_model_npu

        super().__init__(model_runner)

    def _create_device_graph(self):
        return torch.npu.NPUGraph()

    def _capture_graph(self, graph, pool, stream, run_once_fn, bs: int):
        if self.enable_torch_compile:
            compiler = NpuGraphCompiler(run_once_fn)

            patch_dynamo_context_call()
            DisableContext.batch_size = bs
            try:
                # compilation
                out = compiler.compiled_callable()

                # capture function and args
                out = compiler.compiled_callable()
            finally:
                DisableContext.batch_size = None
                restore_dynamo_context_call()

            assert bs in DisableContext.compiled_function
            assert DisableContext.compiled_function[bs]
            assert bs in DisableContext.compiled_function_args
            assert DisableContext.compiled_function_args[bs]

            with torch.npu.graph(
                graph,
                pool=pool,
                stream=stream,
                auto_dispatch_capture=True,
            ):
                compiled_function = DisableContext.compiled_function[bs]
                args = DisableContext.compiled_function_args[bs]
                compiled_function(*args)

        else:
            with torch.npu.graph(
                graph,
                pool=pool,
                stream=stream,
                auto_dispatch_capture=True,
            ):
                out = run_once_fn()
        return out

    def _update_inputs(self, seq_lens):
        self.graphs[self.bs].update(
            cpu_update_input=[{"actual_seq_lengths_kv": seq_lens}]
        )

    def _cache_loc_dtype(self):
        return torch.int32

    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        if not skip_attn_backend_init:
            self.replay_prepare(forward_batch, pp_proxy_tensors)
        else:
            # In speculative decoding, these two fields are still needed.
            self.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.positions[: self.raw_num_token].copy_(forward_batch.positions)

        # Replay
        seq_lens = forward_batch.seq_lens.cpu().tolist() + [0] * (self.bs - self.raw_bs)
        thread = threading.Thread(target=self._update_inputs, args=(seq_lens,))
        thread.start()
        self.graphs[self.bs].replay()
        thread.join()

        output = self.output_buffers[self.bs]
        if isinstance(output, LogitsProcessorOutput):
            return LogitsProcessorOutput(
                next_token_logits=output.next_token_logits[: self.raw_num_token],
                hidden_states=(
                    output.hidden_states[: self.raw_num_token]
                    if output.hidden_states is not None
                    else None
                ),
            )
        else:
            assert isinstance(output, PPProxyTensors)
            return PPProxyTensors({k: v[: self.bs] for k, v in output.tensors.items()})
