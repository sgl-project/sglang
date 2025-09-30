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
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import torch
import torch_npu
import tqdm
from torch_npu.profiler import ProfilerActivity, profile

from sglang.srt.configs.model_config import is_deepseek_nsa
from sglang.srt.distributed import get_tensor_model_parallel_rank, graph_capture
from sglang.srt.layers.torchao_utils import save_gemlite_cache
from sglang.srt.model_executor.cuda_graph_runner import (
    CudaGraphRunner,
    freeze_gc,
    patch_model,
)
from sglang.srt.utils import empty_context, get_available_gpu_memory

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

    def capture(self) -> None:
        profile_context = empty_context()
        if self.enable_profile_cuda_graph:
            output_dir = os.path.join(
                os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp"), "graph_capture_profile"
            )
            if not Path(output_dir).exists():
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Profiling starts for graph capture for NPU. Traces will be saved to: {output_dir}"
            )
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=[torch_npu.profiler.ExportType.Text],
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
                l2_cache=False,
                op_attr=False,
                data_simplification=False,
            )
            profile_context = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                record_shapes=True,
                profile_memory=True,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    output_dir, analyse_flag=False
                ),
                experimental_config=experimental_config,
            )

        # Trigger NPU graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with freeze_gc(
            self.model_runner.server_args.enable_cudagraph_gc
        ), graph_capture() as graph_capture_context:
            with profile_context as prof:
                self.stream = graph_capture_context.stream
                avail_mem = get_available_gpu_memory(
                    self.model_runner.device,
                    self.model_runner.gpu_id,
                    empty_cache=False,
                )
                # Reverse the order to enable better memory sharing across NPU graphs.
                capture_range = (
                    tqdm.tqdm(list(reversed(self.capture_bs)))
                    if get_tensor_model_parallel_rank() == 0
                    else reversed(self.capture_bs)
                )
                for i, bs in enumerate(capture_range):
                    if get_tensor_model_parallel_rank() == 0:
                        avail_mem = get_available_gpu_memory(
                            self.model_runner.device,
                            self.model_runner.gpu_id,
                            empty_cache=False,
                        )
                        capture_range.set_description(
                            f"Capturing batches ({bs=} {avail_mem=:.2f} GB)"
                        )

                    with patch_model(
                        self.model_runner.model,
                        bs in self.compile_bs,
                        num_tokens=bs * self.num_tokens_per_bs,
                        tp_group=self.model_runner.tp_group,
                    ) as forward:
                        (
                            graph,
                            output_buffers,
                        ) = self.capture_one_batch_size(bs, forward)
                        self.graphs[bs] = graph
                        self.output_buffers[bs] = output_buffers

                    # Save gemlite cache after each capture
                    save_gemlite_cache()

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
