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
from torch._dynamo.eval_frame import DisableContext

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
from sglang.srt.hardware_backend.npu.graph_runner.compilation.compilation_context import (
    CompilationContext,
)
from sglang.srt.hardware_backend.npu.graph_runner.compilation.patch_dynamo import (
    patch_dynamo_context,
    patch_dynamo_context_call,
    restore_dynamo_context_call,
)
from sglang.srt.hardware_backend.npu.graph_runner.compilation.piecewise_npu_graph_compiler import (
    PiecewiseNpuGraphCompiler,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    PPProxyTensors,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import get_available_gpu_memory

torch._dynamo.config.skip_nnmodule_hook_guards = True
torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.guard_nn_modules = False

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


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


class PiecewiseNPUGraphRunnerDecode(CudaGraphRunner):
    """A PiecewiseNPUGraphRunnerDecode runs the forward pass of a model with npu graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        model_runner.attn_backend.enable_piecewise_npu_graph_decode = True
        patch_dynamo_context()
        self.init_forward_metadata_was_done = True

        # Parse args
        self.model_runner = model_runner
        compilation_config = get_global_server_args().compilation_config
        if compilation_config is None:
            compilation_config = CompilationConfig(
                compiler="piecewise", splitting_ops=["atb._npu_paged_attention"]
            )
        self.compilation_config = compilation_config
        self.compilation_context = CompilationContext()

        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile

        # Graph inputs
        with torch.device(self.model_runner.device):
            self.num_token_non_padded = torch.zeros((1,), dtype=torch.int32)
            self.block_tables = torch.full((160, 160), 0, dtype=torch.int32)

        super().__init__(model_runner)

    def can_run(self, forward_batch: ForwardBatch):
        return (
            (self.pp_size <= 1)
            and (not self.is_encoder_decoder)
            and (not self.enable_two_batch_overlap)
            and super().can_run(forward_batch)
        )

    def capture(self, forward_batch_: ForwardBatch = None, bs_: int = None) -> None:
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
            encoder_lens=None,
            return_logprob=False,
            positions=positions,
            global_num_tokens_gpu=None,
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
            model_runner=self.model_runner,
            model=self.model_runner.model,
            compilation_config=self.compilation_config,
            compilation_context=self.compilation_context,
            batch_size=bs,
        )

        patch_dynamo_context_call()
        DisableContext.batch_size = bs

        logits_output_or_pp_proxy_tensors = compiler.compiled_callable(
            forward_batch.input_ids, forward_batch.positions, forward_batch
        )

        compiled_graph = CompiledGraph(
            bs, forward_batch, None, compiler.compiled_callable
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
        if self.require_mlp_tp_gather:
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            max_batch_size = (
                max_num_tokens / self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                else max_num_tokens
            )
            index = bisect.bisect_left(self.capture_bs, max_batch_size)
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

        if forward_batch.mrope_positions is not None:
            compiled_graph.forward_batch.mrope_positions[:, :raw_num_token].copy_(
                forward_batch.mrope_positions
            )

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
