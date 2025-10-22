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
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Union

import torch
import tqdm

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.compile import install_torch_compiled, set_compiled
from sglang.srt.compilation.piecewise_context_manager import set_forward_context
from sglang.srt.custom_op import CustomOp
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_tp_rank,
    get_attention_tp_size,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.torchao_utils import save_gemlite_cache
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.two_batch_overlap import TboCudaGraphRunnerPlugin
from sglang.srt.utils import get_available_gpu_memory, log_info_on_rank0

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
def patch_model(model: torch.nn.Module, compiler: str):
    try:
        if compiler != "eager":
            _to_torch(model, reverse=False, num_tokens=16)
        yield model
    finally:
        _to_torch(model, reverse=True, num_tokens=16)


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

        assert (
            self.model_runner.server_args.piecewise_cuda_graph_tokens is not None
        ), "piecewise_cuda_graph_tokens is not set"
        assert self.model_runner.server_args.piecewise_cuda_graph_compiler in [
            "eager",
            "inductor",
        ], "By now, only eager and inductor are supported for piecewise cuda graph compiler."
        self.compile_config = CompilationConfig(
            self.model_runner.server_args.piecewise_cuda_graph_tokens,
            self.model_runner.server_args.piecewise_cuda_graph_compiler,
        )

        # Batch sizes to capture
        self.capture_num_tokens = self.compile_config.get_capture_sizes()
        log_info_on_rank0(
            logger, f"Capture cuda graph num tokens {self.capture_num_tokens}"
        )
        self.capture_forward_mode = ForwardMode.EXTEND
        self.capture_hidden_mode = CaptureHiddenMode.NULL

        # If returning hidden states is enabled, set initial capture hidden mode to full to avoid double-capture on startup
        if model_runner.server_args.enable_return_hidden_states:
            self.capture_hidden_mode = CaptureHiddenMode.FULL

        # Attention backend
        self.max_num_tokens = max(self.capture_num_tokens)

        # Graph inputs
        with torch.device(self.device):
            self.input_ids = torch.zeros((self.max_num_tokens,), dtype=torch.int64)
            self.out_cache_loc = torch.zeros(
                (self.max_num_tokens,), dtype=self._cache_loc_dtype()
            )
            self.positions = torch.zeros((self.max_num_tokens,), dtype=torch.int64)
            self.tbo_plugin = TboCudaGraphRunnerPlugin()

        self.attention_layers = self.model_runner.attention_layers

        if get_global_graph_memory_pool() is None:
            set_global_graph_memory_pool(self.device_module.graph_pool_handle())
        # Set graph pool id globally to be able to use symmetric memory
        set_graph_pool_id(get_global_graph_memory_pool())

        with patch_model(
            self.model_runner.model.model, self.compile_config.compiler
        ) as patched_model:
            install_torch_compiled(
                patched_model,
                fullgraph=True,
                dynamic_arg_dims=None,
                compile_config=self.compile_config,
                graph_pool=get_global_graph_memory_pool(),
            )

            with set_compiled(True):
                self.warmup_and_capture()

            # Capture
            try:
                with model_capture_mode():
                    self.capture()
            except RuntimeError as e:
                raise Exception(
                    f"Capture cuda graph failed: {e}\n{PIECEWISE_CUDA_GRAPH_CAPTURE_FAILED_MSG}"
                )

        self.raw_num_tokens = 0

    def warmup_and_capture(self):
        num_tokens = 2
        with torch.device(self.device):
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=1,
                input_ids=torch.randint(0, 100, (num_tokens,), device=self.device),
                req_pool_indices=torch.arange(1, device=self.device),
                seq_lens=torch.tensor([num_tokens], device=self.device),
                next_token_logits_buffer=None,
                orig_seq_lens=torch.tensor([num_tokens], device=self.device),
                seq_lens_cpu=torch.tensor([num_tokens]),
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool=self.model_runner.token_to_kv_pool,
                attn_backend=self.model_runner.attn_backend,
                out_cache_loc=torch.randint(0, 100, (num_tokens,), device=self.device),
                seq_lens_sum=num_tokens,
                encoder_lens=None,
                return_logprob=False,
                extend_seq_lens=torch.tensor([num_tokens], device=self.device),
                extend_prefix_lens=torch.tensor([num_tokens], device=self.device),
                extend_start_loc=torch.tensor([0], device=self.device),
                extend_prefix_lens_cpu=torch.tensor([num_tokens]),
                extend_seq_lens_cpu=torch.tensor([num_tokens]),
                extend_logprob_start_lens_cpu=torch.tensor([num_tokens]),
                positions=torch.arange(num_tokens, device=self.device),
                global_num_tokens_gpu=None,
                global_num_tokens_for_logprob_gpu=None,
                dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
                global_dp_buffer_len=None,
                mrope_positions=None,
                spec_algorithm=None,
                spec_info=None,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                num_token_non_padded=None,
                global_forward_mode=ForwardMode.EXTEND,
                lora_ids=None,
            )

        with set_forward_context(forward_batch, self.attention_layers):
            _ = self.model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )

    def _cache_loc_dtype(self):
        return torch.int64

    def can_run(self, forward_batch: ForwardBatch):
        num_tokens = len(forward_batch.input_ids)
        # TODO(yuwei): support return logprob
        if forward_batch.return_logprob:
            return False
        if num_tokens <= self.max_num_tokens:
            return True
        return False

    def capture(self) -> None:
        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with freeze_gc(
            self.model_runner.server_args.enable_cudagraph_gc
        ), graph_capture() as graph_capture_context:
            self.stream = graph_capture_context.stream
            avail_mem = get_available_gpu_memory(
                self.model_runner.device,
                self.model_runner.gpu_id,
                empty_cache=False,
            )
            # Reverse the order to enable better memory sharing across cuda graphs.
            capture_range = (
                tqdm.tqdm(list(reversed(self.capture_num_tokens)))
                if get_tensor_model_parallel_rank() == 0
                else reversed(self.capture_num_tokens)
            )
            for i, num_tokens in enumerate(capture_range):
                if get_tensor_model_parallel_rank() == 0:
                    avail_mem = get_available_gpu_memory(
                        self.model_runner.device,
                        self.model_runner.gpu_id,
                        empty_cache=False,
                    )
                    capture_range.set_description(
                        f"Capturing num tokens ({num_tokens=} {avail_mem=:.2f} GB)"
                    )

                with set_compiled(True):
                    self.capture_one_batch_size(num_tokens)

                # Save gemlite cache after each capture
                save_gemlite_cache()

    def capture_one_batch_size(self, num_tokens: int):
        stream = self.stream
        bs = 1

        # Graph inputs
        input_ids = self.input_ids[:num_tokens]
        out_cache_loc = self.out_cache_loc[:num_tokens]
        positions = self.positions[:num_tokens]

        # pipeline parallelism
        if self.pp_size > 1:
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:num_tokens] for k, v in self.pp_proxy_tensors.items()}
            )

        global_dp_buffer_len = None

        if self.model_runner.server_args.enable_lora:
            # It is safe to capture CUDA graph using empty LoRA id, as the LoRA kernels will always be launched whenever
            # `--enable-lora` is set to True (and return immediately if the LoRA id is empty for perf optimization).
            lora_ids = [None] * bs
        else:
            lora_ids = None

        with torch.device(self.device):
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=bs,
                input_ids=input_ids,
                req_pool_indices=torch.arange(bs, device=self.device),
                seq_lens=torch.tensor([num_tokens], device=self.device),
                next_token_logits_buffer=None,
                orig_seq_lens=torch.tensor([num_tokens], device=self.device),
                seq_lens_cpu=torch.tensor([num_tokens]),
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool=self.model_runner.token_to_kv_pool,
                attn_backend=self.model_runner.attn_backend,
                out_cache_loc=out_cache_loc,
                seq_lens_sum=num_tokens,
                encoder_lens=None,
                return_logprob=False,
                extend_seq_lens=torch.tensor([num_tokens], device=self.device),
                extend_prefix_lens=torch.tensor([num_tokens], device=self.device),
                extend_start_loc=torch.tensor([0], device=self.device),
                extend_prefix_lens_cpu=torch.tensor([num_tokens]),
                extend_seq_lens_cpu=torch.tensor([num_tokens]),
                extend_logprob_start_lens_cpu=torch.tensor([num_tokens]),
                positions=positions,
                global_num_tokens_gpu=None,
                global_num_tokens_for_logprob_gpu=None,
                dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
                global_dp_buffer_len=None,
                mrope_positions=None,
                spec_algorithm=None,
                spec_info=None,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                num_token_non_padded=None,
                global_forward_mode=ForwardMode.EXTEND,
                lora_ids=None,
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
            # FIXME: the implementation is hacky. `is_extend_in_batch`` is for determining the deepep mode.
            # It is True in this context but we need to set it to use low latency deepep mode.
            set_is_extend_in_batch(False)

            kwargs = {}
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

        return

    def replay_prepare(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        num_tokens = len(forward_batch.input_ids)
        index = bisect.bisect_left(self.capture_num_tokens, num_tokens)
        static_num_tokens = self.capture_num_tokens[index]
        self.raw_num_tokens = num_tokens
        if static_num_tokens != num_tokens:
            self.out_cache_loc.zero_()
        bs = forward_batch.batch_size

        self.input_ids[:num_tokens].copy_(forward_batch.input_ids)
        self.positions[:num_tokens].copy_(forward_batch.positions)
        self.out_cache_loc[:num_tokens].copy_(forward_batch.out_cache_loc)

        input_ids = self.input_ids[:static_num_tokens]
        positions = self.positions[:static_num_tokens]
        out_cache_loc = self.out_cache_loc[:static_num_tokens]

        next_token_logits_buffer = None
        mrope_positions = None

        static_forward_batch = ForwardBatch(
            forward_mode=forward_batch.forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
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
            extend_num_tokens=forward_batch.extend_num_tokens,
            extend_input_logprob_token_ids_gpu=forward_batch.extend_input_logprob_token_ids_gpu,
            positions=positions,
            global_num_tokens_gpu=forward_batch.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=forward_batch.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=forward_batch.dp_padding_mode,
            global_dp_buffer_len=forward_batch.global_dp_buffer_len,
            mrope_positions=mrope_positions,
            spec_algorithm=forward_batch.spec_algorithm,
            spec_info=forward_batch.spec_info,
            capture_hidden_mode=forward_batch.capture_hidden_mode,
            num_token_non_padded=forward_batch.num_token_non_padded,
            global_forward_mode=forward_batch.global_forward_mode,
            lora_ids=forward_batch.lora_ids,
            sampling_info=forward_batch.sampling_info,
            mm_inputs=forward_batch.mm_inputs,
            temp_scaled_logprobs=forward_batch.temp_scaled_logprobs,
            temperature=forward_batch.temperature,
            top_p_normalized_logprobs=forward_batch.top_p_normalized_logprobs,
            top_p=forward_batch.top_p,
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
                output = self.model_runner.model.forward(
                    static_forward_batch.input_ids,
                    static_forward_batch.positions,
                    static_forward_batch,
                    **kwargs,
                )
            if isinstance(output, LogitsProcessorOutput):
                return LogitsProcessorOutput(
                    next_token_logits=output.next_token_logits[: self.raw_num_tokens],
                    hidden_states=(
                        output.hidden_states[: self.raw_num_tokens]
                        if output.hidden_states is not None
                        else None
                    ),
                )
            else:
                assert isinstance(output, PPProxyTensors)
                # TODO(Yuwei): support PP Support
                raise NotImplementedError(
                    "PPProxyTensors is not supported in PiecewiseCudaGraphRunner yet."
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


PIECEWISE_CUDA_GRAPH_CAPTURE_FAILED_MSG = (
    "Possible solutions:\n"
    "1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
    "2. set --piecewise-cuda-graph-max-tokens to a smaller value (e.g., 512)\n"
    "3. disable Piecewise CUDA graph by unset --enable-piecewise-cuda-graph\n"
    "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
)
