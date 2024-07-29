"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Run the model with cuda graph."""

import bisect
from contextlib import contextmanager

import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.decode import _grouped_size_compiled_for_decode_kernels
from vllm.distributed.parallel_state import graph_capture
from vllm.model_executor.custom_op import CustomOp

from sglang.srt.layers.logits_processor import (
    LogitProcessorOutput,
    LogitsMetadata,
    LogitsProcessor,
)
from sglang.srt.managers.schedule_batch import (
    Batch,
    ForwardMode,
    InputMetadata,
    init_flashinfer_args,
)
from sglang.srt.utils import monkey_patch_vllm_all_gather


def _to_torch(model: torch.nn.Module, reverse: bool = False):
    for sub in model._modules.values():
        if isinstance(sub, CustomOp):
            if reverse:
                sub._forward_method = sub.forward_cuda
            else:
                sub._forward_method = sub.forward_native
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse)


@contextmanager
def patch_model(
    model: torch.nn.Module, use_compile: bool, tp_group: "GroupCoordinator"
):
    backup_ca_comm = None

    try:
        if use_compile:
            _to_torch(model)
            monkey_patch_vllm_all_gather()
            backup_ca_comm = tp_group.ca_comm
            tp_group.ca_comm = None
            yield torch.compile(model.forward, mode="max-autotune-no-cudagraphs")
        else:
            yield model.forward
    finally:
        if use_compile:
            _to_torch(model, reverse=True)
            monkey_patch_vllm_all_gather(reverse=True)
            tp_group.ca_comm = backup_ca_comm


class CudaGraphRunner:
    def __init__(self, model_runner, max_batch_size_to_capture, use_torch_compile):
        self.model_runner = model_runner
        self.graphs = {}
        self.input_buffers = {}
        self.output_buffers = {}
        self.flashinfer_handlers = {}
        self.graph_memory_pool = None

        # Common inputs
        self.max_bs = max_batch_size_to_capture
        self.input_ids = torch.zeros((self.max_bs,), dtype=torch.int32, device="cuda")
        self.req_pool_indices = torch.zeros(
            (self.max_bs,), dtype=torch.int32, device="cuda"
        )
        self.seq_lens = torch.ones((self.max_bs,), dtype=torch.int32, device="cuda")
        self.position_ids_offsets = torch.zeros(
            (self.max_bs,), dtype=torch.int32, device="cuda"
        )
        self.out_cache_loc = torch.zeros(
            (self.max_bs,), dtype=torch.int32, device="cuda"
        )

        # FlashInfer inputs
        self.flashinfer_workspace_buffer = (
            self.model_runner.flashinfer_workspace_buffers[0]
        )
        self.flashinfer_kv_indptr = torch.zeros(
            (self.max_bs + 1,), dtype=torch.int32, device="cuda"
        )
        self.flashinfer_kv_indices = torch.zeros(
            (self.max_bs * model_runner.model_config.context_len,),
            dtype=torch.int32,
            device="cuda",
        )
        self.flashinfer_kv_last_page_len = torch.ones(
            (self.max_bs,), dtype=torch.int32, device="cuda"
        )

        self.compile_bs = [1, 2, 4, 8, 16, 24, 32] if use_torch_compile else []

    def can_run(self, batch_size):
        return batch_size < self.max_bs

    def capture(self, batch_size_list):
        self.batch_size_list = batch_size_list
        with graph_capture() as graph_capture_context:
            self.stream = graph_capture_context.stream
            for bs in batch_size_list:
                with patch_model(
                    self.model_runner.model,
                    bs in self.compile_bs,
                    self.model_runner.tp_group,
                ) as forward:
                    (
                        graph,
                        input_buffers,
                        output_buffers,
                        flashinfer_handler,
                    ) = self.capture_one_batch_size(bs, forward)
                    self.graphs[bs] = graph
                    self.input_buffers[bs] = input_buffers
                    self.output_buffers[bs] = output_buffers
                    self.flashinfer_handlers[bs] = flashinfer_handler

    def capture_one_batch_size(self, bs, forward):
        graph = torch.cuda.CUDAGraph()
        stream = self.stream

        # Common inputs
        input_ids = self.input_ids[:bs]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        position_ids_offsets = self.position_ids_offsets[:bs]
        out_cache_loc = self.out_cache_loc[:bs]

        # FlashInfer inputs
        if not _grouped_size_compiled_for_decode_kernels(
            self.model_runner.model_config.num_attention_heads
            // self.model_runner.tp_size,
            self.model_runner.model_config.get_num_kv_heads(self.model_runner.tp_size),
        ):
            use_tensor_cores = True
        else:
            use_tensor_cores = False
        flashinfer_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.flashinfer_workspace_buffer,
            "NHD",
            use_cuda_graph=True,
            use_tensor_cores=use_tensor_cores,
            paged_kv_indptr_buffer=self.flashinfer_kv_indptr[: bs + 1],
            paged_kv_indices_buffer=self.flashinfer_kv_indices,
            paged_kv_last_page_len_buffer=self.flashinfer_kv_last_page_len[:bs],
        )
        init_flashinfer_args(
            ForwardMode.DECODE,
            self.model_runner,
            req_pool_indices,
            seq_lens,
            None,
            flashinfer_decode_wrapper,
        )

        # Run and capture
        def run_once():
            input_metadata = InputMetadata.create(
                self.model_runner,
                forward_mode=ForwardMode.DECODE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                prefix_lens=None,
                position_ids_offsets=position_ids_offsets,
                out_cache_loc=out_cache_loc,
                return_logprob=False,
                top_logprobs_nums=0,
                skip_flashinfer_init=True,
            )
            input_metadata.flashinfer_decode_wrapper = flashinfer_decode_wrapper

            return forward(input_ids, input_metadata.positions, input_metadata)

        for _ in range(2):
            run_once()

        torch.cuda.synchronize()
        with torch.cuda.graph(graph, pool=self.graph_memory_pool, stream=stream):
            out = run_once()
        torch.cuda.synchronize()
        self.graph_memory_pool = graph.pool()
        return graph, None, out, flashinfer_decode_wrapper

    def replay(self, batch: Batch):
        assert batch.out_cache_loc is not None
        raw_bs = len(batch.reqs)

        # Pad
        index = bisect.bisect_left(self.batch_size_list, raw_bs)
        bs = self.batch_size_list[index]
        if bs != raw_bs:
            self.seq_lens.fill_(1)
            self.position_ids_offsets.zero_()
            self.out_cache_loc.zero_()

        # Common inputs
        self.input_ids[:raw_bs] = batch.input_ids
        self.req_pool_indices[:raw_bs] = batch.req_pool_indices
        self.seq_lens[:raw_bs] = batch.seq_lens
        self.position_ids_offsets[:raw_bs] = batch.position_ids_offsets
        self.out_cache_loc[:raw_bs] = batch.out_cache_loc

        # FlashInfer inputs
        init_flashinfer_args(
            ForwardMode.DECODE,
            self.model_runner,
            self.req_pool_indices[:bs],
            self.seq_lens[:bs],
            None,
            self.flashinfer_handlers[bs],
        )

        # Replay
        self.graphs[bs].replay()
        output = self.output_buffers[bs]

        # Unpad
        if bs != raw_bs:
            output = LogitProcessorOutput(
                next_token_logits=output.next_token_logits[:raw_bs],
                next_token_logprobs=None,
                normalized_prompt_logprobs=None,
                input_token_logprobs=None,
                input_top_logprobs=None,
                output_top_logprobs=None,
            )

        # Extract logprobs
        if batch.return_logprob:
            output.next_token_logprobs = torch.nn.functional.log_softmax(
                output.next_token_logits, dim=-1
            )
            return_top_logprob = any(x > 0 for x in batch.top_logprobs_nums)
            if return_top_logprob:
                logits_metadata = LogitsMetadata(
                    forward_mode=ForwardMode.DECODE,
                    top_logprobs_nums=batch.top_logprobs_nums,
                )
                output.output_top_logprobs = LogitsProcessor.get_top_logprobs(
                    output.next_token_logprobs, logits_metadata
                )[1]

        return output
