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
from typing import TYPE_CHECKING, Optional, Union

import torch

from sglang.srt.configs.model_config import is_deepseek_nsa
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors


class NPUGraphRunner(CudaGraphRunner):
    """A NPUGraphRunner runs the forward pass of a model with npu graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)

    def _create_device_graph(self):
        return torch.npu.NPUGraph()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
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
        if not is_deepseek_nsa(self.model_runner.model_config.hf_config):
            seq_lens = forward_batch.seq_lens.cpu().tolist() + [0] * (
                self.bs - self.raw_bs
            )
            thread = threading.Thread(target=self._update_inputs, args=(seq_lens,))
            thread.start()
            self.graphs[self.bs].replay()
            thread.join()
        else:
            self.graphs[self.bs].replay()

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
