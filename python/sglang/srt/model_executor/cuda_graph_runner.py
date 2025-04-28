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
"""Run the model with cuda graph and torch.compile."""

from __future__ import annotations

import bisect
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable

import torch
import tqdm

from sglang.srt.custom_op import CustomOp
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import GroupCoordinator, graph_capture
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.fused_moe_native import fused_moe_forward_native
from sglang.srt.layers.torchao_utils import save_gemlite_cache
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.patch_torch import monkey_patch_torch_compile
from sglang.srt.utils import (
    get_available_gpu_memory,
    get_device_memory_capacity,
    is_hip,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

_is_hip = is_hip()


def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int):
    for sub in model._modules.values():
        if isinstance(sub, CustomOp):
            if reverse:
                sub._forward_method = sub.forward_cuda
                setattr(sub, "is_torch_compile", False)
            else:
                # NOTE: Temporarily workaround MoE
                if "FusedMoE" in sub.__class__.__name__:
                    if num_tokens == 1:
                        # The performance of torch.compile on this layer is not always good when bs > 1,
                        # so we decide to only use torch.compile when bs =1
                        sub._forward_method = fused_moe_forward_native
                else:
                    sub._forward_method = sub.forward_native
                setattr(sub, "is_torch_compile", True)
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse, num_tokens)


@contextmanager
def patch_model(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
):
    """Patch the model to make it compatible with with torch.compile"""
    backup_ca_comm = None

    try:
        if enable_compile:
            _to_torch(model, reverse=False, num_tokens=num_tokens)
            backup_ca_comm = tp_group.ca_comm
            # Use custom-allreduce here.
            # We found the custom allreduce is much faster than the built-in allreduce in torch,
            # even with ENABLE_INTRA_NODE_COMM=1.
            # tp_group.ca_comm = None
            yield torch.compile(
                torch.no_grad()(model.forward),
                mode=os.environ.get(
                    "SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
                ),
                dynamic=False,
            )
        else:
            yield model.forward
    finally:
        if enable_compile:
            _to_torch(model, reverse=True, num_tokens=num_tokens)
            tp_group.ca_comm = backup_ca_comm


def set_torch_compile_config():
    import torch._dynamo.config
    import torch._inductor.config

    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

    # FIXME: tmp workaround
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024

    monkey_patch_torch_compile()


def get_batch_sizes_to_capture(model_runner: ModelRunner):
    server_args = model_runner.server_args
    capture_bs = server_args.cuda_graph_bs

    if capture_bs is None:
        if server_args.speculative_algorithm is None:
            if server_args.disable_cuda_graph_padding:
                capture_bs = list(range(1, 33)) + list(range(40, 161, 16))
            else:
                capture_bs = [1, 2, 4, 8] + list(range(16, 161, 8))
        else:
            # Since speculative decoding requires more cuda graph memory, we
            # capture less.
            capture_bs = (
                list(range(1, 9)) + list(range(10, 33, 2)) + list(range(40, 161, 16))
            )

        gpu_mem = get_device_memory_capacity()
        # Batch size of each rank will not become so large when DP is on
        if gpu_mem is not None and gpu_mem > 81920 and server_args.dp_size == 1:
            capture_bs += list(range(160, 257, 8))

    if max(capture_bs) > model_runner.req_to_token_pool.size:
        # In some case (e.g., with a small GPU or --max-running-requests), the #max-running-requests
        # is very small. We add more values here to make sure we capture the maximum bs.
        capture_bs += [model_runner.req_to_token_pool.size - 1] + [
            model_runner.req_to_token_pool.size
        ]

    capture_bs = list(sorted(set(capture_bs)))

    assert len(capture_bs) > 0 and capture_bs[0] > 0
    capture_bs = [bs for bs in capture_bs if bs <= model_runner.req_to_token_pool.size]
    if server_args.cuda_graph_max_bs:
        capture_bs = [bs for bs in capture_bs if bs <= server_args.cuda_graph_max_bs]
    compile_bs = (
        [bs for bs in capture_bs if bs <= server_args.torch_compile_max_bs]
        if server_args.enable_torch_compile
        else []
    )
    return capture_bs, compile_bs


# Reuse this memory pool across all cuda graph runners.
global_graph_memory_pool = None


def get_global_graph_memory_pool():
    return global_graph_memory_pool


def set_global_graph_memory_pool(val):
    global global_graph_memory_pool
    global_graph_memory_pool = val


class CudaGraphRunner:
    """A CudaGraphRunner runs the forward pass of a model with cuda graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        # Parse args
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

        # Batch sizes to capture
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)

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
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
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

        # Graph inputs
        with torch.device("cuda"):
            self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )
            self.out_cache_loc = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.mrope_positions = torch.zeros((3, self.max_bs), dtype=torch.int64)

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

        # Capture
        try:
            with self.model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n"
                "Possible solutions:\n"
                "1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
                "2. set --cuda-graph-max-bs to a smaller value (e.g., 16)\n"
                "3. disable torch compile by not using --enable-torch-compile\n"
                "4. disable cuda graph by --disable-cuda-graph. (Not recommonded. Huge perf loss)\n"
                "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
            )

    @contextmanager
    def model_capture_mode(self):
        if hasattr(self.model_runner.model, "capture_mode"):
            self.model_runner.model.capture_mode = True
        if hasattr(self.model_runner.token_to_kv_pool, "capture_mode"):
            self.model_runner.token_to_kv_pool.capture_mode = True

        yield

        if hasattr(self.model_runner.model, "capture_mode"):
            self.model_runner.model.capture_mode = False
        if hasattr(self.model_runner.token_to_kv_pool, "capture_mode"):
            self.model_runner.token_to_kv_pool.capture_mode = False

    def can_run(self, forward_batch: ForwardBatch):
        if self.enable_dp_attention or self.enable_sp_layernorm:
            total_global_tokens = sum(forward_batch.global_num_tokens_cpu)

            is_bs_supported = forward_batch.can_run_dp_cuda_graph and (
                total_global_tokens in self.graphs
                if self.disable_padding
                else total_global_tokens <= self.max_bs
            )
        else:
            is_bs_supported = (
                forward_batch.batch_size in self.graphs
                if self.disable_padding
                else forward_batch.batch_size <= self.max_bs
            )

        # NOTE: cuda graph cannot handle mixed batch (encoder_len = 0)
        # If mixed batch cannot be supported, then encoder_lens can be removed in cuda graph
        # because the full_text_row_masked_out_mask tensor will always be ones
        is_encoder_lens_supported = (
            torch.all(forward_batch.encoder_lens > 0)
            if self.is_encoder_decoder
            else True
        )
        return is_bs_supported and is_encoder_lens_supported

    def capture(self):
        with graph_capture() as graph_capture_context:
            self.stream = graph_capture_context.stream
            avail_mem = get_available_gpu_memory(
                self.model_runner.device, self.model_runner.gpu_id, empty_cache=False
            )
            # Reverse the order to enable better memory sharing across cuda graphs.
            capture_range = (
                tqdm.tqdm(list(reversed(self.capture_bs)))
                if get_tensor_model_parallel_rank() == 0
                else reversed(self.capture_bs)
            )
            for bs in capture_range:
                if get_tensor_model_parallel_rank() == 0:
                    avail_mem = get_available_gpu_memory(
                        self.model_runner.device,
                        self.model_runner.gpu_id,
                        empty_cache=False,
                    )
                    capture_range.set_description(
                        f"Capturing batches ({avail_mem=:.2f} GB)"
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

    def capture_one_batch_size(self, bs: int, forward: Callable):
        graph = torch.cuda.CUDAGraph()
        stream = self.stream
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
        )

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

            logits_output = forward(input_ids, forward_batch.positions, forward_batch)
            return logits_output.next_token_logits, logits_output.hidden_states

        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()

            run_once()

        global global_graph_memory_pool
        with torch.cuda.graph(graph, pool=global_graph_memory_pool, stream=stream):
            out = run_once()

        global_graph_memory_pool = graph.pool()
        return graph, out

    def recapture_if_needed(self, forward_batch: ForwardBatch):
        # If the capture_hidden_mode changes, we need to recapture the graph
        hidden_mode_from_spec_info = getattr(
            forward_batch.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL
        )
        if (
            forward_batch.capture_hidden_mode == CaptureHiddenMode.FULL
            and self.capture_hidden_mode != CaptureHiddenMode.FULL
        ):
            self.capture_hidden_mode = CaptureHiddenMode.FULL
            self.capture()
        elif (
            forward_batch.capture_hidden_mode != CaptureHiddenMode.FULL
            and self.capture_hidden_mode != hidden_mode_from_spec_info
        ):
            self.capture_hidden_mode = hidden_mode_from_spec_info
            self.capture()

    def replay_prepare(self, forward_batch: ForwardBatch):
        self.recapture_if_needed(forward_batch)

        raw_bs = forward_batch.batch_size
        raw_num_token = raw_bs * self.num_tokens_per_bs

        # Pad
        if self.enable_dp_attention or self.enable_sp_layernorm:
            index = bisect.bisect_left(
                self.capture_bs, sum(forward_batch.global_num_tokens_cpu)
            )
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        if bs != raw_bs:
            self.seq_lens.fill_(1)
            self.out_cache_loc.zero_()

        # Common inputs
        self.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        self.out_cache_loc[:raw_num_token].copy_(forward_batch.out_cache_loc)
        self.positions[:raw_num_token].copy_(forward_batch.positions)
        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                self.seq_lens_cpu.fill_(1)
            self.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)

        if self.is_encoder_decoder:
            self.encoder_lens[:raw_bs].copy_(forward_batch.encoder_lens)
        if forward_batch.mrope_positions is not None:
            self.mrope_positions[:, :raw_bs].copy_(forward_batch.mrope_positions)
        if self.enable_dp_attention or self.enable_sp_layernorm:
            self.global_num_tokens_gpu.copy_(forward_batch.global_num_tokens_gpu)

        if hasattr(forward_batch.spec_info, "hidden_states"):
            self.hidden_states[:raw_num_token] = forward_batch.spec_info.hidden_states

        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            self.req_pool_indices,
            self.seq_lens,
            forward_batch.seq_lens_sum + (bs - raw_bs),
            self.encoder_lens,
            forward_batch.forward_mode,
            forward_batch.spec_info,
            seq_lens_cpu=self.seq_lens_cpu,
        )

        # Store fields
        self.raw_bs = raw_bs
        self.raw_num_token = raw_num_token
        self.bs = bs

    def replay(
        self, forward_batch: ForwardBatch, skip_attn_backend_init: bool = False
    ) -> LogitsProcessorOutput:
        if not skip_attn_backend_init:
            self.replay_prepare(forward_batch)
        else:
            # In speculative decoding, these two fields are still needed.
            self.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.positions[: self.raw_num_token].copy_(forward_batch.positions)

        # Replay
        self.graphs[self.bs].replay()
        next_token_logits, hidden_states = self.output_buffers[self.bs]

        logits_output = LogitsProcessorOutput(
            next_token_logits=next_token_logits[: self.raw_num_token],
            hidden_states=(
                hidden_states[: self.raw_num_token]
                if hidden_states is not None
                else None
            ),
        )
        return logits_output

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
                        device="cuda",
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
