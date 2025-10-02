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
import gc
import inspect
import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import tqdm
from torch.profiler import ProfilerActivity, profile

from sglang.srt.context_manager import set_forward_context
from sglang.srt.custom_op import CustomOp
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator, graph_capture
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_tp_rank,
    get_attention_tp_size,
    set_dp_buffer_len,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.torchao_utils import save_gemlite_cache
from sglang.srt.model_executor.compilation.backend import SGLangBackend
from sglang.srt.model_executor.compilation.decorators import (
    install_torch_compiled,
    set_compiled,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
    enable_num_token_non_padded,
)
from sglang.srt.patch_torch import monkey_patch_torch_compile
from sglang.srt.two_batch_overlap import TboCudaGraphRunnerPlugin
from sglang.srt.utils import (
    empty_context,
    get_available_gpu_memory,
    get_device_memory_capacity,
    log_info_on_rank0,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

# Detect whether the current forward pass is in capture mode
is_capture_mode = False


def get_is_capture_mode():
    return is_capture_mode


@contextmanager
def model_capture_mode():
    global is_capture_mode
    is_capture_mode = True

    yield

    is_capture_mode = False


@contextmanager
def freeze_gc(enable_cudagraph_gc: bool):
    """
    Optimize garbage collection during CUDA graph capture.
    Clean up, then freeze all remaining objects from being included
    in future collections if GC is disabled during capture.
    """
    gc.collect()
    should_freeze = not enable_cudagraph_gc
    if should_freeze:
        gc.freeze()
    try:
        yield
    finally:
        if should_freeze:
            gc.unfreeze()


def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int):
    for sub in model._modules.values():
        if isinstance(sub, CustomOp):
            if reverse:
                sub.leave_torch_compile()
            else:
                sub.enter_torch_compile(num_tokens=num_tokens)
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
    capture_bs = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    compile_bs = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    return capture_bs, compile_bs


# Reuse this memory pool across all cuda graph runners.
global_graph_memory_pool = None


def get_global_graph_memory_pool():
    return global_graph_memory_pool


def set_global_graph_memory_pool(val):
    global global_graph_memory_pool
    global_graph_memory_pool = val


class PiecewiseCudaGraphRunner:
    """A PiecewiseCudaGraphRunner runs the forward pass of a model with cuda graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        # Parse args
        self.model_runner = model_runner
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.graphs = {}
        self.output_buffers = {}
        self.tp_size = model_runner.server_args.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.pp_size = model_runner.server_args.pp_size

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        # Batch sizes to capture
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        log_info_on_rank0(logger, f"Capture cuda graph bs {self.capture_bs}")
        self.capture_forward_mode = ForwardMode.EXTEND
        self.capture_hidden_mode = CaptureHiddenMode.NULL
        self.num_tokens_per_bs = 1

        # If returning hidden states is enabled, set initial capture hidden mode to full to avoid double-capture on startup
        if model_runner.server_args.enable_return_hidden_states:
            self.capture_hidden_mode = CaptureHiddenMode.FULL

        # Attention backend
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
        self.model_runner.attn_backend.init_cuda_graph_state(
            self.max_bs, self.max_num_token
        )
        self.seq_len_fill_value = (
            self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
        )
        self.seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )
        self.extend_prefix_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )
        self.extend_seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )
        self.extend_logprob_start_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )

        # Graph inputs
        with torch.device(self.device):
            self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )
            self.out_cache_loc = torch.zeros(
                (self.max_num_token,), dtype=self._cache_loc_dtype()
            )
            self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.mrope_positions = torch.zeros(
                (3, self.max_num_token), dtype=torch.int64
            )
            self.orig_seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )
            self.num_token_non_padded = torch.zeros((1,), dtype=torch.int32)
            self.tbo_plugin = TboCudaGraphRunnerPlugin()

            # pipeline parallelism
            if self.pp_size > 1:
                raise NotImplementedError(
                    "Pipeline parallelism is not supported for piecewise cuda graph yet"
                )

            self.encoder_lens = None

            self.global_num_tokens_gpu = None
            self.global_num_tokens_for_logprob_gpu = None

            self.custom_mask = torch.ones(
                (
                    (self.seq_lens.sum().item() + self.max_num_token)
                    * self.num_tokens_per_bs
                ),
                dtype=torch.bool,
                device=self.device,
            )
            self.next_token_logits_buffer = torch.zeros(
                (self.max_num_token, self.model_runner.model_config.vocab_size),
                dtype=torch.float,
                device=self.device,
            )
            self.extend_seq_lens = torch.tensor(
                [0], dtype=torch.int32, device=self.device
            )
            self.extend_prefix_lens = torch.tensor(
                [0], dtype=torch.int32, device=self.device
            )
            self.extend_start_loc = torch.tensor(
                [0], dtype=torch.int32, device=self.device
            )

        self.attention_layers = self.model_runner.attention_layers

        # Capture
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def _cache_loc_dtype(self):
        return torch.int64

    def can_run(self, forward_batch: ForwardBatch):
        if self.require_mlp_tp_gather:
            cuda_graph_bs = (
                max(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                else max(forward_batch.global_num_tokens_cpu)
            )
        else:
            cuda_graph_bs = forward_batch.batch_size

        is_bs_supported = (
            cuda_graph_bs in self.graphs
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )

        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

        # NOTE: cuda graph cannot handle mixed batch (encoder_len = 0)
        # If mixed batch cannot be supported, then encoder_lens can be removed in cuda graph
        # because the full_text_row_masked_out_mask tensor will always be ones
        is_encoder_lens_supported = (
            torch.all(forward_batch.encoder_lens > 0)
            if self.is_encoder_decoder
            else True
        )

        requested_capture_hidden_mode = max(
            forward_batch.capture_hidden_mode,
            (
                forward_batch.spec_info.capture_hidden_mode
                if getattr(forward_batch.spec_info, "capture_hidden_mode", None)
                is not None
                else CaptureHiddenMode.NULL
            ),
        )
        capture_hidden_mode_matches = (
            requested_capture_hidden_mode == CaptureHiddenMode.NULL
            or requested_capture_hidden_mode == self.capture_hidden_mode
        )
        is_tbo_supported = (
            forward_batch.can_run_tbo if self.enable_two_batch_overlap else True
        )

        return (
            is_bs_supported
            and is_encoder_lens_supported
            and is_tbo_supported
            and capture_hidden_mode_matches
        )

    def capture(self) -> None:
        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        # with freeze_gc(
        #     self.model_runner.server_args.enable_cudagraph_gc
        # ), graph_capture() as graph_capture_context:
        self.stream = torch.cuda.current_stream()
        avail_mem = get_available_gpu_memory(
            self.model_runner.device,
            self.model_runner.gpu_id,
            empty_cache=False,
        )
        # Reverse the order to enable better memory sharing across cuda graphs.
        capture_range = (
            tqdm.tqdm(list((self.capture_bs)))
            if get_tensor_model_parallel_rank() == 0
            else (self.capture_bs)
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

            with set_compiled(True):
                self.capture_one_batch_size(bs)
                print(f"--- Capture one batch size for bsz {bs} ---")
                self.capture_one_batch_size(bs)

            # Save gemlite cache after each capture
            save_gemlite_cache()

    def capture_one_batch_size(self, num_tokens: int):
        stream = self.stream
        bs = 1

        # Graph inputs
        input_ids = self.input_ids[:num_tokens]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        # create a random one
        self.out_cache_loc[:num_tokens].copy_(torch.randint(0, 1000, (num_tokens,)))
        out_cache_loc = self.out_cache_loc[:num_tokens]
        orig_seq_lens = self.orig_seq_lens[:bs]
        positions = self.positions[:num_tokens]
        seq_lens_cpu = self.seq_lens_cpu[:bs]

        encoder_lens = None
        mrope_positions = self.mrope_positions[:, :num_tokens]
        next_token_logits_buffer = None
        self.num_token_non_padded[...] = num_tokens

        extend_seq_lens = self.extend_seq_lens[:bs]
        extend_prefix_lens = self.extend_prefix_lens[:bs]
        extend_start_loc = self.extend_start_loc[:bs]
        extend_prefix_lens_cpu = self.extend_prefix_lens_cpu[:bs]
        extend_seq_lens_cpu = self.extend_seq_lens_cpu[:bs]
        extend_logprob_start_lens_cpu = self.extend_logprob_start_lens_cpu[:bs]

        # pipeline parallelism
        if self.pp_size > 1:
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:num_tokens] for k, v in self.pp_proxy_tensors.items()}
            )

        global_dp_buffer_len = None

        spec_info = self.get_spec_info(num_tokens)
        if self.capture_hidden_mode != CaptureHiddenMode.FULL:
            self.capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        if self.model_runner.server_args.enable_lora:
            # It is safe to capture CUDA graph using empty LoRA id, as the LoRA kernels will always be launched whenever
            # `--enable-lora` is set to True (and return immediately if the LoRA id is empty for perf optimization).
            lora_ids = [None] * bs
        else:
            lora_ids = None

        forward_batch = ForwardBatch(
            forward_mode=self.capture_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            next_token_logits_buffer=next_token_logits_buffer,
            orig_seq_lens=orig_seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            encoder_lens=encoder_lens,
            return_logprob=False,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_logprob_start_lens_cpu=extend_logprob_start_lens_cpu,
            positions=positions,
            global_num_tokens_gpu=self.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=self.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            mrope_positions=None,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=self.capture_hidden_mode,
            num_token_non_padded=self.num_token_non_padded,
            global_forward_mode=self.capture_forward_mode,
            lora_ids=lora_ids,
        )
        self.tbo_plugin.capture_one_batch_size(forward_batch, num_tokens=num_tokens)

        if lora_ids is not None:
            self.model_runner.lora_manager.prepare_lora_batch(forward_batch)

        # # Attention backend
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)

        # Run and capture
        def run_once():
            # Clean intermediate result cache for DP attention
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(global_dp_buffer_len, num_tokens)

            kwargs = {}
            if (
                self.pp_size > 1
                and "pp_proxy_tensors" in inspect.signature(forward).parameters
            ):
                kwargs["pp_proxy_tensors"] = PPProxyTensors(
                    {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
                )
            with set_forward_context(forward_batch, self.attention_layers):
                self.model_runner.model.forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                )
            return

        for _ in range(2):
            self.device_module.synchronize()
            self.model_runner.tp_group.barrier()
            run_once()

        if get_global_graph_memory_pool() is None:
            set_global_graph_memory_pool(self.device_module.graph_pool_handle())
        # Set graph pool id globally to be able to use symmetric memory
        set_graph_pool_id(get_global_graph_memory_pool())

        return

    def replay_prepare(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ):

        num_tokens = len(forward_batch.input_ids)
        bs = forward_batch.batch_size

        self.input_ids[:num_tokens].copy_(forward_batch.input_ids)
        self.positions[:num_tokens].copy_(forward_batch.positions)
        self.req_pool_indices[:bs].copy_(forward_batch.req_pool_indices)
        self.out_cache_loc[:num_tokens].copy_(forward_batch.out_cache_loc)
        self.seq_lens[:bs].copy_(forward_batch.seq_lens)
        # self.mrope_positions[:, :num_tokens].copy_(forward_batch.mrope_positions)
        # if forward_batch.next_token_logits_buffer is not None:
        #     self.next_token_logits_buffer[:num_tokens].copy_(forward_batch.next_token_logits_buffer)

        input_ids = self.input_ids[:num_tokens]
        positions = self.positions[:num_tokens]
        req_pool_indices = self.req_pool_indices[:bs]
        out_cache_loc = self.out_cache_loc[:num_tokens]
        seq_lens = self.seq_lens[:bs]
        if forward_batch.next_token_logits_buffer is not None:
            next_token_logits_buffer = self.next_token_logits_buffer[:num_tokens]
        else:
            next_token_logits_buffer = None
        mrope_positions = None

        static_forward_batch = ForwardBatch(
            forward_mode=forward_batch.forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=seq_lens,
            next_token_logits_buffer=next_token_logits_buffer,
            orig_seq_lens=forward_batch.orig_seq_lens,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=forward_batch.seq_lens_sum,
            encoder_lens=forward_batch.encoder_lens,
            return_logprob=forward_batch.return_logprob,
            extend_seq_lens=forward_batch.extend_seq_lens,
            extend_prefix_lens=forward_batch.extend_prefix_lens,
            extend_start_loc=forward_batch.extend_start_loc,
            extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            extend_logprob_start_lens_cpu=forward_batch.extend_logprob_start_lens_cpu,
            positions=positions,
            global_num_tokens_gpu=forward_batch.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=forward_batch.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=forward_batch.global_dp_buffer_len,
            mrope_positions=mrope_positions,
            spec_algorithm=forward_batch.spec_algorithm,
            spec_info=forward_batch.spec_info,
            capture_hidden_mode=forward_batch.capture_hidden_mode,
            num_token_non_padded=forward_batch.num_token_non_padded,
            global_forward_mode=forward_batch.global_forward_mode,
            lora_ids=forward_batch.lora_ids,
        )

        return static_forward_batch

    def replay(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        static_forward_batch = self.replay_prepare(forward_batch, **kwargs)

        # Replay
        with set_forward_context(static_forward_batch, self.attention_layers):
            with set_compiled(True):
                return self.model_runner.model.forward(
                    static_forward_batch.input_ids,
                    static_forward_batch.positions,
                    static_forward_batch,
                    **kwargs,
                )

    def get_spec_info(self, num_tokens: int):
        spec_info = None
        if (
            self.model_runner.spec_algorithm.is_eagle()
            or self.model_runner.spec_algorithm.is_standalone()
        ):
            from sglang.srt.speculative.eagle_utils import EagleVerifyInput

            if self.model_runner.is_draft_worker:
                raise RuntimeError("This should not happen.")
            else:
                spec_info = EagleVerifyInput(
                    draft_token=None,
                    custom_mask=self.custom_mask,
                    positions=None,
                    retrive_index=None,
                    retrive_next_token=None,
                    retrive_next_sibling=None,
                    retrive_cum_len=None,
                    spec_steps=self.model_runner.server_args.speculative_num_steps,
                    topk=self.model_runner.server_args.speculative_eagle_topk,
                    draft_token_num=self.model_runner.server_args.speculative_num_draft_tokens,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    seq_lens_sum=None,
                    seq_lens_cpu=None,
                )

        return spec_info


CUDA_GRAPH_CAPTURE_FAILED_MSG = (
    "Possible solutions:\n"
    "1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
    "2. set --cuda-graph-max-bs to a smaller value (e.g., 16)\n"
    "3. disable torch compile by not using --enable-torch-compile\n"
    "4. disable CUDA graph by --disable-cuda-graph. (Not recommended. Huge performance loss)\n"
    "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
)
