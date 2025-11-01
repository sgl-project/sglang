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

import inspect
import logging
import threading
from typing import TYPE_CHECKING, Optional, Union

import torch
import tqdm

from sglang.srt.configs.model_config import is_deepseek_nsa
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.utils import get_available_gpu_memory, get_compiler_backend

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors


class NPUGraphRunner(CudaGraphRunner):
    """A NPUGraphRunner runs the forward pass of a model with npu graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)

    def initialize(self):
        if self.enable_torch_compile:
            self.warm_up()
        else:
            super().initialize()

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
        if self.enable_torch_compile:
            if not skip_attn_backend_init:
                forward_batch.attn_backend.init_forward_metadata(forward_batch)

            kwargs = {}
            if pp_proxy_tensors is not None:
                kwargs["pp_proxy_tensors"] = pp_proxy_tensors

            with torch.no_grad():
                return self.model_runner.model.compile_forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                )

        if not skip_attn_backend_init:
            self.replay_prepare(forward_batch, pp_proxy_tensors)
        else:
            # In speculative decoding, these two fields are still needed.
            self.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.positions[: self.raw_num_token].copy_(forward_batch.positions)

        # Replay
        if not is_deepseek_nsa(self.model_runner.model_config.hf_config):
            if forward_batch.forward_mode.is_target_verify():
                seq_lens_cpu = forward_batch.seq_lens.cpu() + self.num_tokens_per_bs
                seq_lens = seq_lens_cpu.tolist() + [0] * (self.bs - self.raw_bs)
            else:
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

    def prepare_forward_batch(self, bs: int, num_tokens: int) -> ForwardBatch:
        # Graph inputs
        with torch.device(self.model_runner.device):
            input_ids = torch.zeros((num_tokens,), dtype=torch.int64)
            req_pool_indices = torch.zeros((bs,), dtype=torch.int64)
            seq_lens = torch.full((bs,), self.seq_len_fill_value, dtype=torch.int64)
            out_cache_loc = torch.zeros((num_tokens,), dtype=torch.int32)
            positions = torch.zeros((num_tokens,), dtype=torch.int64)
            num_token_non_padded = torch.tensor(num_tokens, dtype=torch.int32)

        if self.is_encoder_decoder:
            encoder_lens = self.encoder_lens[:bs]
        else:
            encoder_lens = None
        mrope_positions = None

        # pipeline parallelism
        if self.pp_size > 1:
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:num_tokens] for k, v in self.pp_proxy_tensors.items()}
            )

        if self.require_mlp_tp_gather:
            global_num_tokens = torch.tensor(
                [
                    num_tokens // self.dp_size + (i < (num_tokens % self.dp_size))
                    for i in range(self.dp_size)
                ],
                dtype=torch.int64,
                device=input_ids.device,
            )
        elif self.require_attn_tp_gather:
            global_num_tokens = torch.tensor(
                [num_tokens], dtype=torch.int64, device=input_ids.device
            )
        else:
            global_num_tokens = None
            gathered_buffer = None

        spec_info = self.get_spec_info(num_tokens)
        if self.capture_hidden_mode != CaptureHiddenMode.FULL:
            self.capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        forward_batch = ForwardBatch(
            forward_mode=self.capture_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            encoder_lens=encoder_lens,
            return_logprob=False,
            positions=positions,
            global_num_tokens_gpu=global_num_tokens,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=self.capture_hidden_mode,
            num_token_non_padded=num_token_non_padded,
            global_forward_mode=None,
            mm_inputs=[None] * bs,
            lora_ids=[None] * bs,
            global_num_tokens_cpu=[num_tokens],
        )
        return forward_batch

    def warm_up(self):
        @torch.compile(dynamic=True, backend=get_compiler_backend())
        def run_for_init(input):
            return input + 1

        input = torch.zeros([1]).to(self.model_runner.device)
        run_for_init(input)

        backend = get_compiler_backend("reduce-overhead")
        self.model_runner.model.compile_forward = torch.compile(
            torch.no_grad()(self.model_runner.model.forward),
            fullgraph=True,
            dynamic=True,
            backend=backend,
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
                    f"Capturing batches ({bs=} {avail_mem=:.2f} GB)"
                )
            num_tokens = bs * self.num_tokens_per_bs
            forward_batch = self.prepare_forward_batch(bs, num_tokens)
            forward_batch.attn_backend.init_forward_metadata(forward_batch)

            # Run and compile
            def run_once():
                # Clean intermediate result cache for DP attention
                forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = (
                    None
                )

                kwargs = {}
                if (
                    self.pp_size > 1
                    and "pp_proxy_tensors"
                    in inspect.signature(
                        self.model_runner.model.compile_forward
                    ).parameters
                ):
                    kwargs["pp_proxy_tensors"] = forward_batch.pp_proxy_tensors
                self.mark_static(forward_batch, kwargs.get("pp_proxy_tensors"))

                with torch.no_grad():
                    logits_output_or_pp_proxy_tensors = (
                        self.model_runner.model.compile_forward(
                            forward_batch.input_ids,
                            forward_batch.positions,
                            forward_batch,
                            **kwargs,
                        )
                    )
                    return logits_output_or_pp_proxy_tensors

            torch.npu.synchronize()
            self.model_runner.tp_group.barrier()
            run_once()

    def mark_static(
        self, forward_batch: ForwardBatch, pp_proxy_tensors: PPProxyTensors = None
    ):
        def mark_tensor_static(model_input, is_cache=False):
            if model_input is not None:
                if isinstance(model_input, torch.Tensor):
                    torch._dynamo.mark_static(model_input)
                elif is_cache:
                    for buffer_per_layer in model_input:
                        torch._dynamo.mark_static(buffer_per_layer)
                elif isinstance(model_input, PPProxyTensors):
                    for pp_out in model_input.tensors.items():
                        torch._dynamo.mark_static(pp_out)
                elif isinstance(model_input, tuple):
                    for value in model_input:
                        torch._dynamo.mark_static(value)
                else:
                    raise ValueError(
                        f"Unsupported type with mark static: {type(model_input)}"
                    )

        mark_tensor_static(pp_proxy_tensors)
        mark_tensor_static(forward_batch.input_ids)
        mark_tensor_static(forward_batch.positions)
        mark_tensor_static(forward_batch.input_embeds)
        mark_tensor_static(forward_batch.out_cache_loc)
        mark_tensor_static(forward_batch.attn_backend.forward_metadata.block_tables)
        try:
            mark_tensor_static(forward_batch.token_to_kv_pool.k_buffer, is_cache=True)
            mark_tensor_static(forward_batch.token_to_kv_pool.v_buffer, is_cache=True)
        except AttributeError as e:
            mark_tensor_static(forward_batch.token_to_kv_pool.kv_buffer, is_cache=True)

    def can_run(self, forward_batch: ForwardBatch):
        if not self.enable_torch_compile:
            return super().can_run(forward_batch)

        return (
            forward_batch.forward_mode.is_decode()
            and forward_batch.batch_size in self.compile_bs
        )
