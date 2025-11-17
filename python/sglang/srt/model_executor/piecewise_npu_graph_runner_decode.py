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
"""Run the model with npu graph and torch.compile."""

from __future__ import annotations

import bisect
import gc
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import torch._dynamo.config
import tqdm

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.npu.compilation_context import CompilationContext
from sglang.srt.compilation.npu.patch_dynamo import (
    patch_dynamo_context,
    patch_dynamo_context_call,
    restore_dynamo_context_call,
)
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.layers.attention.ascend_backend import AscendAttnBackend
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.compilation.piecewise_npu_graph_compiler import (
    PiecewiseNpuGraphCompiler,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
    enable_num_token_non_padded,
)
from sglang.srt.two_batch_overlap import TboCudaGraphRunnerPlugin
from sglang.srt.utils import (
    get_available_gpu_memory,
    get_device_memory_capacity,
    rank0_log,
)

torch._dynamo.config.skip_nnmodule_hook_guards = True
torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.guard_nn_modules = False

import logging

from torch._dynamo.eval_frame import DisableContext

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.model_executor.cuda_graph_runner import model_capture_mode

torch.cuda.CUDAGraph = torch.npu.NPUGraph
torch.cuda.synchronize = torch.npu.synchronize
torch.cuda.graph = torch.npu.graph
torch.cuda.stream = torch.npu.stream
torch.cuda.Stream = torch.npu.Stream
torch.cuda.current_stream = torch.npu.current_stream
torch.cuda.graph_pool_handle = torch.npu.graph_pool_handle


class CompiledGraph:
    def __init__(
        self,
        bs: int,
        forward_batch: ForwardBatch,
        attn_backend: AscendAttnBackend,
        callable,
    ):
        self.bs = bs
        self.forward_batch = forward_batch
        self.attn_backend = attn_backend
        self.callable = callable


def get_batch_sizes_to_capture(model_runner: ModelRunner):
    server_args = model_runner.server_args
    capture_bs = server_args.cuda_graph_bs

    if capture_bs is None:
        if server_args.speculative_algorithm is None:
            if server_args.disable_cuda_graph_padding:
                capture_bs = list(range(1, 33)) + list(range(48, 161, 16))
            else:
                capture_bs = [1, 2, 4, 8] + list(range(16, 161, 8))
        else:
            # Since speculative decoding requires more npu graph memory, we
            # capture less.
            capture_bs = (
                list(range(1, 9))
                + list(range(10, 33, 2))
                + list(range(40, 64, 8))
                + list(range(80, 161, 16))
            )

        gpu_mem = get_device_memory_capacity()
        if gpu_mem is not None and gpu_mem > 96 * 1024:
            capture_bs += list(range(160, 257, 8))
        if gpu_mem is not None and gpu_mem > 180 * 1000:
            capture_bs += list(range(256, 513, 16))

    if max(capture_bs) > model_runner.req_to_token_pool.size:
        # In some cases (e.g., with a small GPU or --max-running-requests), the #max-running-requests
        # is very small. We add more values here to make sure we capture the maximum bs.
        capture_bs += [model_runner.req_to_token_pool.size]

    if server_args.enable_two_batch_overlap:
        capture_bs = [bs for bs in capture_bs if bs >= 2]

    if server_args.cuda_graph_max_bs:
        capture_bs = [bs for bs in capture_bs if bs <= server_args.cuda_graph_max_bs]
        if max(capture_bs) < server_args.cuda_graph_max_bs:
            capture_bs += list(
                range(max(capture_bs), server_args.cuda_graph_max_bs + 1, 16)
            )
    capture_bs = [bs for bs in capture_bs if bs <= model_runner.req_to_token_pool.size]
    capture_bs = list(sorted(set(capture_bs)))
    assert len(capture_bs) > 0 and capture_bs[0] > 0, f"{capture_bs=}"
    compile_bs = (
        [bs for bs in capture_bs if bs <= server_args.torch_compile_max_bs]
        if server_args.enable_torch_compile
        else []
    )

    return capture_bs, compile_bs


class PiecewiseNPUGraphRunnerDecode:
    """A PiecewiseNPUGraphRunnerDecode runs the forward pass of a model with npu graph and torch.compile."""

    def __init__(
        self, model_runner: ModelRunner, compilation_config: CompilationConfig
    ):
        model_runner.attn_backend.enable_piecewise_npu_graph_decode = True

        patch_dynamo_context()

        self.inference_counter = 1
        self.init_forward_metadata_was_done = True

        # Parse args
        self.model_runner = model_runner
        if compilation_config is None:
            compilation_config = CompilationConfig()
            compilation_config.splitting_ops = ["atb._npu_paged_attention"]
        self.compilation_config = compilation_config
        self.compilation_context = CompilationContext()

        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder
        self.enable_dp_attention = model_runner.server_args.enable_dp_attention
        # self.enable_sp_layernorm = model_runner.server_args.enable_sp_layernorm
        self.enable_two_batch_overlap = (
            model_runner.server_args.enable_two_batch_overlap
        )
        self.speculative_algorithm = model_runner.server_args.speculative_algorithm
        self.tp_size = model_runner.server_args.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.pp_size = model_runner.server_args.pp_size

        # Batch sizes to capture
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        rank0_log(f"Capture npu graph bs {self.capture_bs}")
        self.capture_forward_mode: int = ForwardMode.DECODE
        self.capture_hidden_mode: int = CaptureHiddenMode.NULL
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
        self.model_runner.attn_backend.init_cuda_graph_state(
            self.max_bs, self.max_num_token
        )
        self.seq_len_fill_value = (
            self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
        )
        # FIXME(lsyin): leave it here for now, I don't know whether it is necessary
        self.encoder_len_fill_value = 0
        self.seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )

        # Graph inputs
        with torch.device(self.model_runner.device):
            self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )
            self.out_cache_loc = torch.zeros((self.max_num_token,), dtype=torch.int32)
            self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.mrope_positions = torch.zeros((3, self.max_bs), dtype=torch.int64)
            self.num_token_non_padded = torch.zeros((1,), dtype=torch.int32)
            self.tbo_plugin = TboCudaGraphRunnerPlugin()

            self.block_tables = torch.full((160, 160), 0, dtype=torch.int32)

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
            if model_runner.spec_algorithm.is_eagle3():
                self.model_runner.model.set_eagle3_layers_to_capture()

            if self.is_encoder_decoder:
                # NOTE: encoder_lens can influence the full_text_row_masked_out_mask tensor when doing mixed batch
                self.encoder_lens = torch.full(
                    (self.max_bs,), self.encoder_len_fill_value, dtype=torch.int32
                )
            else:
                self.encoder_lens = None

            if self.enable_dp_attention:  # or self.enable_sp_layernorm:
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

        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Graph compilation failed: {e}\n{NPU_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def can_run(self, forward_batch: ForwardBatch):
        if self.enable_dp_attention:  # or self.enable_sp_layernorm:
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

        # NOTE: npu graph cannot handle mixed batch (encoder_len = 0)
        # If mixed batch cannot be supported, then encoder_lens can be removed in npu graph
        # because the full_text_row_masked_out_mask tensor will always be ones
        is_encoder_lens_supported = (
            torch.all(forward_batch.encoder_lens > 0)
            if self.is_encoder_decoder
            else True
        )

        is_tbo_supported = (
            forward_batch.can_run_tbo if self.enable_two_batch_overlap else True
        )

        can_run_value = (
            is_bs_supported and is_encoder_lens_supported and is_tbo_supported
        )
        return can_run_value

    def capture(self, forward_batch_: ForwardBatch = None, bs_: int = None):
        with graph_capture() as graph_capture_context:
            self.stream = graph_capture_context.stream

            self.model_runner.tp_group.barrier()

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

                (compiled_graph, output_buffers) = self.capture_one_batch_size(
                    bs, self.model_runner.model.forward, forward_batch_=forward_batch_
                )
                self.graphs[bs] = compiled_graph
                self.output_buffers[bs] = output_buffers

    def init_forward_metadata_attn_backend(
        self, bs: int, attn_backend: AscendAttnBackend, forward_batch: ForwardBatch
    ):
        attn_backend.forward_metadata.block_tables = self.block_tables

        seq_lens_cpu_int = forward_batch.seq_lens_cpu_int
        seq_lens_cpu_int[
            : attn_backend.forward_metadata.seq_lens_cpu_int.shape[0]
        ].copy_(attn_backend.forward_metadata.seq_lens_cpu_int)
        attn_backend.forward_metadata.seq_lens_cpu_int = seq_lens_cpu_int

    def init_forward_batch(
        self, bs: int, attn_backend: AscendAttnBackend, forward_batch_: ForwardBatch
    ) -> ForwardBatch:
        if forward_batch_:
            return forward_batch_

        num_tokens = bs * self.num_tokens_per_bs

        with torch.device(self.model_runner.device):
            req_pool_indices = torch.zeros((bs,), dtype=torch.int32)
            seq_lens = torch.full((bs,), self.seq_len_fill_value, dtype=torch.int32)
            out_cache_loc = torch.zeros((bs,), dtype=torch.int32)
            positions = torch.zeros((bs,), dtype=torch.int64)
            input_ids = torch.zeros((bs,), dtype=torch.int64)
            mrope_positions = torch.zeros((3, self.max_num_token), dtype=torch.int64)

        assert self.is_encoder_decoder == False
        encoder_lens = None
        num_token_non_padded = None

        assert self.pp_size <= 1
        assert self.enable_dp_attention == False
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
            attn_backend=attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum(),
            encoder_lens=encoder_lens,
            return_logprob=False,
            positions=positions,
            global_num_tokens_gpu=global_num_tokens,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=self.capture_hidden_mode,
            num_token_non_padded=self.num_token_non_padded,
            global_forward_mode=self.capture_forward_mode,
        )

        seq_lens_cpu_int = torch.zeros((bs,), dtype=torch.int32, device="cpu")
        forward_batch.seq_lens_cpu_int = seq_lens_cpu_int

        seq_lens_cpu = torch.full((bs,), 1, dtype=torch.int32, device="cpu")
        forward_batch.seq_lens_cpu = seq_lens_cpu

        for i in range(bs):
            forward_batch.global_forward_mode = None
            forward_batch.input_ids[i] = 323
            forward_batch.num_token_non_padded = None
            forward_batch.out_cache_loc[i] = 134
            forward_batch.positions[i] = 6
            forward_batch.seq_lens[i] = 7
            forward_batch.seq_lens_cpu[i] = 7
            forward_batch.seq_lens_cpu_int[i] = 7
            forward_batch.req_pool_indices[i] = 1
        forward_batch.seq_lens_sum = sum(forward_batch.seq_lens)

        if self.enable_dp_attention:  # or self.enable_sp_layernorm:
            assert False
        assert self.pp_size <= 1
        assert self.enable_dp_attention == False
        assert enable_num_token_non_padded(self.model_runner.server_args) == False
        assert self.enable_two_batch_overlap == False

        attn_backend.init_forward_metadata(forward_batch)

        self.init_forward_metadata_attn_backend(bs, attn_backend, forward_batch)

        # Clean intermediate result cache for DP attention
        forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
        return forward_batch

    def capture_one_batch_size(
        self,
        bs: int,
        forward: Callable,
        forward_batch_: ForwardBatch = None,
        compile: bool = True,
    ):
        attn_backend = self.model_runner.attn_backend
        attn_backend.init_cuda_graph_state(bs, self.max_num_token)

        self.model_runner.attn_backend = attn_backend

        for _ in range(2):
            forward_batch = self.init_forward_batch(bs, attn_backend, forward_batch_)

            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()

            self.model_runner.attn_backend.graph_mode = True
            self.model_runner.model(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )

        forward_batch = self.init_forward_batch(bs, attn_backend, forward_batch_)

        self.compilation_context.stream = self.stream
        self.model_runner.attn_backend.graph_mode = True

        compiler = PiecewiseNpuGraphCompiler(
            self.model_runner,
            self.model_runner.model,
            self.compilation_config,
            self.compilation_context,
            self.model_runner.page_size,
        )

        patch_dynamo_context_call()
        DisableContext.batch_size = bs

        logits_output_or_pp_proxy_tensors = compiler.compiled_callable(
            forward_batch.input_ids, forward_batch.positions, forward_batch
        )

        try:
            logits_output_or_pp_proxy_tensors = compiler.compiled_callable(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )
        finally:
            DisableContext.batch_size = None
            restore_dynamo_context_call()

        assert DisableContext.compiled_function
        assert DisableContext.compiled_function_args

        compiled_graph = CompiledGraph(
            bs, forward_batch, None, compiler.compiled_callable
        )

        torch._dynamo.reset()
        gc.collect()

        return (compiled_graph, logits_output_or_pp_proxy_tensors)

    def replay_prepare(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        raw_bs = forward_batch.batch_size
        raw_num_token = raw_bs * self.num_tokens_per_bs

        # Pad
        if self.enable_dp_attention:
            index = bisect.bisect_left(
                self.capture_bs, sum(forward_batch.global_num_tokens_cpu)
            )
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)

        bs = self.capture_bs[index]
        compiled_graph = self.graphs[bs]

        compiled_graph.forward_batch.input_ids[
            : forward_batch.input_ids.shape[0]
        ].copy_(forward_batch.input_ids)
        forward_batch.input_ids = compiled_graph.forward_batch.input_ids

        compiled_graph.forward_batch.seq_lens[: forward_batch.seq_lens.shape[0]].copy_(
            forward_batch.seq_lens
        )
        forward_batch.seq_lens = compiled_graph.forward_batch.seq_lens

        compiled_graph.forward_batch.req_pool_indices[
            : forward_batch.req_pool_indices.shape[0]
        ].copy_(forward_batch.req_pool_indices)
        forward_batch.req_pool_indices = compiled_graph.forward_batch.req_pool_indices

        compiled_graph.forward_batch.out_cache_loc[
            : forward_batch.out_cache_loc.shape[0]
        ].copy_(forward_batch.out_cache_loc)
        forward_batch.out_cache_loc = compiled_graph.forward_batch.out_cache_loc

        compiled_graph.forward_batch.positions[
            : forward_batch.positions.shape[0]
        ].copy_(forward_batch.positions)
        forward_batch.positions = compiled_graph.forward_batch.positions

        if forward_batch.seq_lens_cpu is not None:
            compiled_graph.forward_batch.seq_lens_cpu[
                : forward_batch.seq_lens_cpu.shape[0]
            ].copy_(forward_batch.seq_lens_cpu)
            forward_batch.seq_lens_cpu = compiled_graph.forward_batch.seq_lens_cpu

        if pp_proxy_tensors:
            for key in self.pp_proxy_tensors.keys():
                dim = pp_proxy_tensors[key].shape[0]
                self.pp_proxy_tensors[key][:dim].copy_(pp_proxy_tensors[key])

        if self.is_encoder_decoder:
            assert False

        if forward_batch.mrope_positions is not None:
            compiled_graph.forward_batch.mrope_positions[:, :raw_num_token].copy_(
                forward_batch.mrope_positions
            )

        if self.enable_dp_attention:
            assert False

        if enable_num_token_non_padded(self.model_runner.server_args):
            assert False

        if self.enable_two_batch_overlap:
            assert False

        # Store fields
        self.raw_bs = raw_bs
        self.raw_num_token = raw_num_token
        self.bs = bs

    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        self.replay_prepare(forward_batch, pp_proxy_tensors)
        compiled_graph = self.graphs[self.bs]

        def init():
            attn_backend = self.model_runner.attn_backend
            forward_batch.attn_backend = attn_backend

            compiled_graph: CompiledGraph = self.graphs[self.bs]

            attn_backend = self.model_runner.attn_backend
            if not self.init_forward_metadata_was_done:
                attn_backend.init_forward_metadata(forward_batch)
                self.init_forward_metadata_was_done = True
            else:
                if forward_batch.extend_seq_lens is not None:
                    attn_backend.forward_metadata.extend_seq_lens_cpu_int = (
                        forward_batch.extend_seq_lens.cpu().int()
                    )
                attn_backend.forward_metadata.seq_lens_cpu_int = (
                    forward_batch.seq_lens_cpu.int()
                )

            self.init_forward_metadata_attn_backend(
                self.bs, attn_backend, compiled_graph.forward_batch
            )

        init()

        self.model_runner.attn_backend.graph_mode = True

        DisableContext.compiled_function[self.bs](
            *DisableContext.compiled_function_args[self.bs]
        )

        output = self.output_buffers[self.bs]

        if isinstance(output, LogitsProcessorOutput):
            result = LogitsProcessorOutput(
                next_token_logits=output.next_token_logits[: self.raw_num_token],
                hidden_states=(
                    output.hidden_states[: self.raw_num_token]
                    if output.hidden_states is not None
                    else None
                ),
            )
        else:
            assert isinstance(output, PPProxyTensors)
            result = PPProxyTensors(
                {k: v[: self.bs] for k, v in output.tensors.items()}
            )

        return result

    def get_spec_info(self, num_tokens: int):
        spec_info = None
        if self.model_runner.spec_algorithm.is_eagle():
            from sglang.srt.speculative.eagle_utils import EagleVerifyInput

            if self.model_runner.is_draft_worker:
                raise RuntimeError("This should not happen.")
            else:
                spec_info = EagleVerifyInput(
                    draft_token=None,
                    custom_mask=torch.ones(
                        (num_tokens * self.model_runner.model_config.context_len),
                        dtype=torch.bool,
                        device=self.model_runner.device,
                    ),
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


NPU_GRAPH_CAPTURE_FAILED_MSG = (
    "Possible solutions:\n"
    "1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
    "2. set --cuda-graph-max-bs to a smaller value (e.g., 16)\n"
    "3. disable torch compile by not using --enable-torch-compile\n"
    "4. disable CUDA graph by --disable-cuda-graph. (Not recommended. Huge performance loss)\n"
    "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
)
