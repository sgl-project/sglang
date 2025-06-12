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
"""Run the model with npu graph engine and torch.compile."""

from __future__ import annotations

import bisect
import inspect
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import tqdm

from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.utils import (
    get_available_gpu_memory,
    get_device_memory_capacity,
    rank0_log,
    get_compiler_backend,
    get_device,
    set_npu_compiler_config
)
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.model_executor.cuda_graph_runner import _to_torch, patch_model, get_batch_sizes_to_capture, \
    set_torch_compile_config, DeviceRunnerBase
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class NpuGraphRunner(DeviceRunnerBase):
    """A NpuGraphRunner runs the forward pass of a model with npu graph engine and torch.compile."""
    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        try:
            self.warm_up()
        except RuntimeError as e:
            raise Exception(
                f"compile npu graph failed: {e}\n"
                "Possible solutions:\n"
                "1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
                "2. set --torch-compile-max-bs to a smaller value (e.g., 16)\n"
                "3. disable torch compile by not using --enable-torch-compile\n"
                "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
            )

    def warm_up(self):
        if not self.enable_torch_compile:
            rank0_log("enable_torch_compile is False, model will run eagerly on npu, this may cause performance loss."
                      "please set --enable-torch-compile")
            return

        compile_mode = os.environ.get(
            "SGLANG_TORCH_COMPILE_MODE", "max-autotune"
        )
        # support max-autotune currently, will support reduce-overhead(similar to cuda graph) in the future
        assert compile_mode in ["max-autotune"], "Only max-autotune is supported for now while use npu"
        npu_compile_config = {
            "experimental_config": {
                "frozen_parameter": True,
                "tiling_schedule_optimize": True,
                "topology_sorting_strategy": "StableRDFS",
            },
            "inference_config": {
                "dynamic_gears_merge_policy": "zip"
            },
            "mode": compile_mode
            # add compile_config here
        }
        set_npu_compiler_config(npu_compile_config)
        rank0_log("Warming up npu graph")

        self.model_runner.model.compile_forward = torch.compile(
            torch.no_grad()(self.model_runner.model.forward),
            fullgraph=True,
            backend=get_compiler_backend(),
            mode = compile_mode
        )
        compile_range = (
            tqdm.tqdm(list(reversed(self.compile_bs)))
            if get_tensor_model_parallel_rank() == 0
            else reversed(self.compile_bs)
        )
        for bs in compile_range:
            if get_tensor_model_parallel_rank() == 0:
                avail_mem = get_available_gpu_memory(
                    self.model_runner.device,
                    self.model_runner.gpu_id,
                    empty_cache=False,
                )
                compile_range.set_description(
                    f"Capturing batches ({avail_mem=:.2f} GB)"
                )
            num_tokens = bs * self.num_tokens_per_bs
            forward_batch = self.prepare_forward_batch(bs, num_tokens)

            if get_device().startswith("npu"):
                import torchair
                torchair.inference.set_dim_gears(forward_batch.input_ids, dim_gears={0: self.compile_bs})

            if forward_batch.lora_paths is not None:
                self.model_runner.lora_manager.prepare_lora_batch(forward_batch)

            # Attention backend
            self.model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.encoder_lens,
                forward_batch.forward_mode,
                forward_batch.spec_info,
            )

            # Run and capture
            def run_once():
                # Clean intermediate result cache for DP attention
                forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None

                kwargs = {}
                if (
                        self.pp_size > 1
                        and "pp_proxy_tensors" in inspect.signature(self.model_runner.model.compile_forward).parameters
                ):
                    kwargs["pp_proxy_tensors"] = forward_batch.pp_proxy_tensors

                logits_output_or_pp_proxy_tensors = self.model_runner.model.compile_forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                )
                return logits_output_or_pp_proxy_tensors

            for _ in range(2):
                torch.npu.synchronize()
                self.model_runner.tp_group.barrier()
                run_once()

        return

    @contextmanager
    def compile_context(self, forward_batch: ForwardBatch):
        if self.enable_torch_compile and forward_batch.batch_size in self.compile_bs:
            yield self.model_runner.model.compile_forward
        else:
            yield self.model_runner.model.forward

    def can_run(self, forward_batch: ForwardBatch):
        raise NotImplementedError("Did not support for npu currently.")

    def replay(
            self,
            forward_batch: ForwardBatch,
            skip_attn_backend_init: bool = False,
            pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        raise NotImplementedError("Did not support for npu currently.")