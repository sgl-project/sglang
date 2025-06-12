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
    get_device
)
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.model_executor.cuda_graph_runner import _to_torch, patch_model, get_batch_sizes_to_capture, \
    set_torch_compile_config

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class NpuGraphRunner:
    """A NpuGraphRunner runs the forward pass of a model with npu graph engine and torch.compile."""
    def __init__(self, model_runner: ModelRunner):
        self.model_runner = model_runner
        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder
        self.enable_dp_attention = model_runner.server_args.enable_dp_attention
        self.enable_sp_layernorm = model_runner.server_args.enable_sp_layernorm
        self.speculative_algorithm = model_runner.server_args.speculative_algorithm
        self.tp_size = model_runner.server_args.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.pp_size = model_runner.server_args.pp_size

        # Batch sizes to capture
        _, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        rank0_log(f"Compile npu graph bs {self.compile_bs}")
        self.capture_forward_mode = ForwardMode.DECODE
        self.capture_hidden_mode = CaptureHiddenMode.NULL
        self.num_tokens_per_bs = 1
        if model_runner.spec_algorithm.is_eagle():
            if self.model_runner.is_draft_worker:
                raise RuntimeError("This should not happen")
            else:
                self.capture_forward_mode = ForwardMode.TARGET_VERIFY
                self.num_tokens_per_bs = (
                    self.model_runner.server_args.speculative_num_draft_tokens
                )

        # Attention backend
        self.max_bs = max(self.compile_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
        # todo: use npu attention backend
        if global_server_args_dict["attention_backend"] == "flashmla":
            self.model_runner.attn_backend.init_cuda_graph_state(self.max_bs)
        else:
            self.model_runner.attn_backend.init_cuda_graph_state(self.max_num_token)
        self.seq_len_fill_value = (
            self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
        )
        # FIXME(lsyin): leave it here for now, I don't know whether it is necessary
        self.encoder_len_fill_value = 0
        self.seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )

        if self.enable_torch_compile:
            set_torch_compile_config()

        if self.model_runner.server_args.lora_paths is not None:
            self.model_runner.lora_manager.init_cuda_graph_batch_info(self.max_bs)

        # Graph inputs
        with torch.device(get_device()):
            self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )
            self.out_cache_loc = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.mrope_positions = torch.zeros((3, self.max_bs), dtype=torch.int64)
            self.num_token_non_padded = torch.zeros((1,), dtype=torch.int32)

            # pipeline parallelism
            if self.pp_size > 1:
                self.pp_proxy_tensors = {
                    "hidden_states": torch.zeros(
                        (self.max_bs, self.model_runner.model_config.hidden_size),
                        dtype=torch.bfloat16,
                    ),
                    "residual": torch.zeros(
                        (self.max_bs, self.model_runner.model_config.hidden_size),
                        dtype=torch.bfloat16,
                    ),
                }

            # Speculative_inference
            if (
                    model_runner.spec_algorithm.is_eagle3()
                    and not model_runner.is_draft_worker
            ):
                self.hidden_states = torch.zeros(
                    (
                        self.max_num_token,
                        3 * self.model_runner.model_config.hidden_size,
                    ),
                    dtype=self.model_runner.dtype,
                )
                self.model_runner.model.set_eagle3_layers_to_capture()
            elif model_runner.spec_algorithm.is_eagle():
                self.hidden_states = torch.zeros(
                    (self.max_num_token, self.model_runner.model_config.hidden_size),
                    dtype=self.model_runner.dtype,
                )

            if self.is_encoder_decoder:
                # NOTE: encoder_lens can influence the full_text_row_masked_out_mask tensor when doing mixed batch
                self.encoder_lens = torch.full(
                    (self.max_bs,), self.encoder_len_fill_value, dtype=torch.int32
                )
            else:
                self.encoder_lens = None
            if self.enable_dp_attention or self.enable_sp_layernorm:
                # TODO(ch-wan): SP layernorm should use a different logic to manage gathered_buffer
                self.gathered_buffer = torch.zeros(
                    (
                        self.max_bs * self.dp_size * self.num_tokens_per_bs,
                        self.model_runner.model_config.hidden_size,
                    ),
                    dtype=self.model_runner.dtype,
                )
                self.global_num_tokens_gpu = torch.zeros(
                    (self.dp_size,), dtype=torch.int32
                )

        # warm_up
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
            rank0_log("enable_torch_compile is False, model will run eagerly, this may cause performance loss."
                      "please set --enable-torch-compile")
            return
        rank0_log("Warming up npu graph")
        compile_mode = os.environ.get(
            "SGLANG_TORCH_COMPILE_MODE", "max-autotune"
        )
        assert compile_mode in ["max-autotune", "reduce-overhead"], \
            "Only max-autotune and reduce-overhead are supported for now while use npu"

        self.model_runner.model.compile_forward = torch.compile(
            torch.no_grad()(self.model_runner.model.forward),
            fullgraph=True,
            backend=get_compiler_backend(),
            mode = os.environ.get(
                "SGLANG_TORCH_COMPILE_MODE", "max-autotune"
            )
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

            # Graph inputs
            input_ids = self.input_ids[:num_tokens]
            req_pool_indices = self.req_pool_indices[:bs]
            seq_lens = self.seq_lens[:bs]
            out_cache_loc = self.out_cache_loc[:num_tokens]
            positions = self.positions[:num_tokens]
            if self.is_encoder_decoder:
                encoder_lens = self.encoder_lens[:bs]
            else:
                encoder_lens = None
            mrope_positions = self.mrope_positions[:, :bs]
            self.num_token_non_padded[...] = num_tokens

            # pipeline parallelism
            if self.pp_size > 1:
                pp_proxy_tensors = PPProxyTensors(
                    {k: v[:num_tokens] for k, v in self.pp_proxy_tensors.items()}
                )

            if self.enable_dp_attention or self.enable_sp_layernorm:
                self.global_num_tokens_gpu.copy_(
                    torch.tensor(
                        [
                            num_tokens // self.dp_size + (i < bs % self.dp_size)
                            for i in range(self.dp_size)
                        ],
                        dtype=torch.int32,
                        device=input_ids.device,
                    )
                )
                global_num_tokens = self.global_num_tokens_gpu
                gathered_buffer = self.gathered_buffer[:num_tokens]
            else:
                global_num_tokens = None
                gathered_buffer = None

            spec_info = self.get_spec_info(num_tokens)
            if self.capture_hidden_mode != CaptureHiddenMode.FULL:
                self.capture_hidden_mode = (
                    spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
                )
            if self.model_runner.server_args.lora_paths is not None:
                # Currently, if the lora_path in `lora_paths` is None, the lora backend will use a
                # different logic to handle lora, so we need to set `lora_paths` to a list of non-None
                # values if lora is enabled.
                lora_paths = [next(iter(self.model_runner.server_args.lora_paths))] * bs
            else:
                lora_paths = None
            print("self.capture_forward_mode", self.capture_forward_mode)
            forward_batch = ForwardBatch(
                forward_mode=self.capture_forward_mode,
                batch_size=bs,
                input_ids=input_ids,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool=self.model_runner.token_to_kv_pool,
                attn_backend=self.model_runner.attn_backend,
                out_cache_loc=out_cache_loc,
                seq_lens_sum=seq_lens.sum(),
                encoder_lens=encoder_lens,
                return_logprob=False,
                positions=positions,
                global_num_tokens_gpu=global_num_tokens,
                gathered_buffer=gathered_buffer,
                mrope_positions=mrope_positions,
                spec_algorithm=self.model_runner.spec_algorithm,
                spec_info=spec_info,
                capture_hidden_mode=self.capture_hidden_mode,
                lora_paths=lora_paths,
                num_token_non_padded=self.num_token_non_padded,
                global_forward_mode=self.capture_forward_mode,
            )

            if get_device().startswith("npu"):
                import torchair
                torchair.inference.set_dim_gears(forward_batch.input_ids, dim_gears={0: self.compile_bs})

            if lora_paths is not None:
                self.model_runner.lora_manager.prepare_lora_batch(forward_batch)

            # Attention backend
            self.model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
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
                    kwargs["pp_proxy_tensors"] = pp_proxy_tensors

                logits_output_or_pp_proxy_tensors = self.model_runner.model.compile_forward(
                    input_ids,
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

    def can_run_with_reduce_overhead(self, forward_batch: ForwardBatch):
        """Check if the batch size is supported."""
        return False

    def replay_with_reduce_overhead(self):
        pass

    def get_spec_info(self, num_tokens: int):
        spec_info = None
        if self.model_runner.spec_algorithm.is_eagle():
            from sglang.srt.speculative.eagle_utils import EagleVerifyInput

            if self.model_runner.is_draft_worker:
                raise RuntimeError("This should not happen.")
            else:
                spec_info = EagleVerifyInput(
                    draft_token=None,
                    custom_mask=torch.zeros(
                        (num_tokens * self.model_runner.model_config.context_len),
                        dtype=torch.bool,
                        device=get_device(),
                    ),
                    positions=None,
                    retrive_index=None,
                    retrive_next_token=None,
                    retrive_next_sibling=None,
                    retrive_cum_len=None,
                    draft_token_num=self.model_runner.server_args.speculative_num_draft_tokens,
                    spec_steps=self.model_runner.server_args.speculative_num_steps,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                )

        return spec_info

    @contextmanager
    def compile_context(self, forward_batch: ForwardBatch):
        if self.enable_torch_compile and forward_batch.batch_size in self.compile_bs:
            yield self.model_runner.model.compile_forward
        else:
            yield self.model_runner.model.forward